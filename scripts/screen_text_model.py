"""Run ONE pinned candidate per process on a frozen development suite, CUDA NF4."""
import argparse
import csv
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.training.a1_runtime import ROOT, file_hash, write_json
from src.training.model_candidates import DEFAULT_CATALOG, read_candidates, verify_candidate
from src.training.text_screening import grade, messages_for, summarize


def lora_targets(model_type):
    if model_type in ("qwen2", "qwen3", "llama"):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type == "phi3":
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    if model_type == "lfm2":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    raise ValueError("No reviewed LoRA target layout for model type: " + model_type)


def lora_gradient_probe(model, tokenizer, dtype):
    """Backward-only toy check, no optimizer/update/save; not task fine-tuning."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model.get_input_embeddings().to(dtype=dtype)
    model.get_output_embeddings().to(dtype=dtype)
    targets = lora_targets(model.config.model_type)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
        target_modules=targets,
        task_type="CAUSAL_LM", bias="none"))
    model.train()
    model.config.use_cache = False
    # Separate toy text: no benchmark answer or validation target enters the backward probe.
    encoded = tokenizer("Đây là phép kiểm tra kỹ thuật gradient LoRA, không phải dữ liệu nha khoa.", return_tensors="pt").to("cuda")
    labels = encoded["input_ids"].clone()
    with torch.autocast("cuda", dtype=dtype):
        loss = model(**encoded, labels=labels).loss
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    result = {"performed": True, "optimizer_steps": 0, "adapter_saved": False,
              "model_type": model.config.model_type, "target_modules": targets,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "loss": float(loss.detach()),
              "finite_gradients": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
              "nonzero_gradient": any(bool(torch.count_nonzero(g)) for g in gradients)}
    model.zero_grad(set_to_none=True)
    if not math.isfinite(result["loss"]) or not result["finite_gradients"] or not result["nonzero_gradient"]:
        raise RuntimeError("LoRA gradient feasibility check failed")
    return {**result, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-lora", action="store_true")
    args = parser.parse_args()
    candidates = {spec["id"]: spec for spec in read_candidates(args.catalog)}
    if args.model not in candidates:
        parser.error("Unknown candidate")
    if args.output_dir.exists():
        parser.error("Output already exists; choose a new directory")
    manifest = json.loads((args.suite / "manifest.json").read_text(encoding="utf-8"))
    for filename, key in (("cases.jsonl", "cases_sha256"), ("prompts.json", "prompts_sha256")):
        if file_hash(args.suite / filename) != manifest[key]:
            raise ValueError("Frozen suite checksum mismatch")
    cases = [json.loads(line) for line in (args.suite / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = {row["id"]: row["messages"] for row in json.loads((args.suite / "prompts.json").read_text(encoding="utf-8"))}
    if len(cases) != manifest["case_count"] or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Frozen suite has wrong count/duplicate IDs")
    if any(prompts.get(case["id"]) != messages_for(case) for case in cases):
        raise ValueError("Prompt/schema code changed since suite was frozen")
    spec = candidates[args.model]
    directory, model_manifest = verify_candidate(spec)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"candidate": spec, "suite_manifest": manifest, "catalog_sha256": file_hash(args.catalog),
        "weights_sha256": model_manifest["sha256"], "adapter": None, "max_new_tokens": 384,
        "max_input_tokens": 4096, "seed": 20261105, "thinking": False,
        "generation": "greedy, one attempt, no output repair, no truncation",
        "quantization": "NF4 double quantization, BF16 compute; all candidates on same GPU",
        "source_code_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/screen_text_model.py", "src/training/text_screening.py", "src/agents/contracts.py",
            "src/data/a1_dataset.py", "src/training/a1_evaluation.py", "src/training/model_candidates.py")},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}
    write_json(args.output_dir / "runtime.json", metadata)
    rows = []
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        set_seed(metadata["seed"])
        torch.set_num_threads(6)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA/BF16 required for this matched screen")
        torch.cuda.reset_peak_memory_stats()
        metadata["gpu"] = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
            quantization_config=quantization, device_map={"": 0}, dtype=dtype,
            attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
        if not model.is_loaded_in_4bit or any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Model offloaded or not NF4")
        torch.cuda.synchronize()
        metadata["load_seconds"] = round(time.perf_counter() - started, 4)
        metadata["model_type"] = model.config.model_type
        metadata["parameter_count"] = model.num_parameters()
        write_json(args.output_dir / "runtime.json", metadata)
        print(f"Loaded {args.model} in {metadata['load_seconds']:.1f}s", flush=True)
        with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                rendered = tokenizer.apply_chat_template(prompts[case["id"]], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
                input_tokens = int(inputs["input_ids"].shape[1])
                if input_tokens > metadata["max_input_tokens"]:
                    raise ValueError("Prompt too long; refusing silent truncation")
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    output = model.generate(**inputs, max_new_tokens=metadata["max_new_tokens"],
                        do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                generated = output[0, input_tokens:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                row = {"id": case["id"], "agent": case["agent"], "input": case["input"],
                    "expected": case["expected"], "raw_output": raw,
                    "input_tokens": input_tokens, "generated_tokens": int(generated.shape[0]),
                    "hit_token_limit": int(generated.shape[0]) == metadata["max_new_tokens"],
                    "generation_seconds": round(elapsed, 4),
                    "rendered_prompt_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(),
                    **grade(case, raw)}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{args.model} {number}/{len(cases)} {case['id']}: control={row['task_control_pass']} {elapsed:.1f}s", flush=True)
        metadata["inference_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated()/2**20, 1)
        metadata["inference_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved()/2**20, 1)
        if args.check_lora:
            try:
                metadata["lora_feasibility"] = lora_gradient_probe(model, tokenizer, dtype)
            except (ValueError, RuntimeError) as exc:
                # Preserve completed inference even if training feasibility fails.
                metadata["lora_feasibility"] = {"performed": True, "passed": False,
                    "optimizer_steps": 0, "adapter_saved": False,
                    "error": type(exc).__name__ + ": " + str(exc)}
            torch.cuda.synchronize()
            metadata["including_probe_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated()/2**20, 1)
            metadata["including_probe_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved()/2**20, 1)
        write_json(args.output_dir / "runtime.json", metadata)
        summary = {"status": "completed", "model": args.model,
            "lora_feasible_short_context": metadata.get("lora_feasibility", {}).get("passed"), **summarize(rows)}
        write_json(args.output_dir / "summary.json", summary)
        with (args.output_dir / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "agent", "raw_output", "content_faithful", "complete", "vietnamese_fluency", "invented_facts", "reviewer", "notes"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in ("id", "agent", "raw_output")})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(args.output_dir / "summary.json", {"status": "failed", "model": args.model,
            "error": type(exc).__name__ + ": " + str(exc), **summarize(rows)})
        raise


if __name__ == "__main__":
    main()
