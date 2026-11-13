"""Persistent offline inference only. One Qwen3 base and two separate LoRA adapters."""
import argparse
from contextlib import redirect_stdout, nullcontext
import json
from pathlib import Path
import sys
import time

from src.training.a1_runtime import ROOT, file_hash


def verified_artifacts(spec):
    directory = ROOT / spec["model_dir"]
    identity = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if (identity["repo"] != spec["base_model"] or identity["revision"] != spec["revision"]
            or directory.resolve() != (ROOT / spec["model_dir"]).resolve()):
        raise ValueError("Model nền không khớp checkpoint đã chọn")
    for name, checksum in identity["sha256"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError("Model nền checksum không khớp: " + name)
    a1 = ROOT / spec["a1_run_dir"]
    a2 = ROOT / spec["a2_run_dir"]
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    summary = read(a1 / "summary.json")
    if summary["status"] != "completed" or read(a2 / "run_status.json")["status"] != "completed":
        raise ValueError("Chỉ dùng các lần train hoàn tất")
    adapter1 = Path(summary["best_adapter_dir"])
    manifest = read(adapter1.parent / "adapter_manifest.json")
    if manifest["base_model"] != spec["base_model"] or manifest["revision"] != spec["revision"]:
        raise ValueError("Adapter A1 khác checkpoint nền")
    for name, checksum in manifest["files_sha256"].items():
        if file_hash(adapter1 / name) != checksum:
            raise ValueError("Adapter A1 checksum không khớp: " + name)
    config2 = read(a2 / "config.json")
    if config2["model"] != spec["base_model"] or config2["revision"] != spec["revision"]:
        raise ValueError("Adapter A2 khác checkpoint nền")
    for name, checksum in read(a2 / "download_manifest.json")["sha256"].items():
        if file_hash(a2 / name) != checksum:
            raise ValueError("Adapter A2 checksum không khớp: " + name)
    adapter2 = a2 / "adapter"
    for adapter in (adapter1, adapter2):
        cfg = read(adapter / "adapter_config.json")
        if cfg["peft_type"] != "LORA" or cfg.get("modules_to_save"):
            raise ValueError("Runtime chia sẻ chỉ hỗ trợ adapter LoRA không thay embedding/head")
    return directory, adapter1, adapter2, identity


def load(spec):
    import torch
    from peft import PeftModel
    from src.training.a1_runtime import load_base
    directory, adapter1, adapter2, identity = verified_artifacts(spec)
    cfg = {"base_model": spec["base_model"], "model_dir": spec["model_dir"]}
    model, tokenizer, dtype = load_base(cfg)
    model.gradient_checkpointing_disable()
    model = PeftModel.from_pretrained(model, adapter1, adapter_name="a1", is_trainable=False,
                                      local_files_only=True)
    model.load_adapter(adapter2, adapter_name="a2", is_trainable=False, local_files_only=True)
    adapter3 = None
    if spec.get("a3_run_dir"):
        run3 = ROOT / spec["a3_run_dir"]
        config3 = json.loads((run3 / "config.json").read_text(encoding="utf-8"))
        if config3["model"] != spec["base_model"] or config3["revision"] != spec["revision"]:
            raise ValueError("Adapter A3 khác model nền")
        if json.loads((run3 / "run_status.json").read_text())["status"] != "completed":
            raise ValueError("A3 chưa train hoàn tất")
        for name, checksum in json.loads((run3 / 'download_manifest.json').read_text())['sha256'].items():
            if file_hash(run3 / name) != checksum:
                raise ValueError('Adapter A3 checksum không khớp: ' + name)
        adapter3 = run3 / "adapter"
        model.load_adapter(adapter3, adapter_name="a3", is_trainable=False, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    identity = {"repo": identity["repo"], "revision": identity["revision"],
                "gpu": torch.cuda.get_device_name(0), "quantization": "NF4",
                "agents": {"A1": {"variant": "finetuned", "adapter": str(adapter1),
                           "sha256": file_hash(adapter1 / "adapter_model.safetensors")},
                           "A2": {"variant": "finetuned", "adapter": str(adapter2),
                           "sha256": file_hash(adapter2 / "adapter_model.safetensors")},
                           "A3": {"variant": "base"}, "A4": {"variant": "base"}}, "image_agent": False}
    if adapter3 is not None:
        identity["agents"]["A3"] = {"variant": "finetuned", "adapter": str(adapter3),
            "sha256": file_hash(adapter3 / "adapter_model.safetensors")}
    return model, tokenizer, dtype, identity


def generate(model, tokenizer, dtype, spec, request, identity):
    import torch
    role = request["role"]
    if role not in ("A1", "A2", "A3", "A4"):
        raise ValueError("Vai trò không hợp lệ")
    messages = request["messages"]
    if not isinstance(messages, list) or not messages or any(
            m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str)
            for m in messages):
        raise ValueError("Messages không hợp lệ")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt")
    limit = spec.get("max_input_tokens", 4096)
    if prompt.shape[1] > limit:
        raise ValueError(f'{role}_INPUT_TOO_LONG: {prompt.shape[1]} > {limit}; cần rút gọn hội thoại hoặc ngữ cảnh')
    adapted = identity["agents"][role]["variant"] == "finetuned"
    if adapted:
        model.set_adapter(role.lower())
    adapter_context = nullcontext() if adapted else model.disable_adapter()
    model.eval()
    prompt = prompt.to("cuda")
    started = time.perf_counter()
    with adapter_context, torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=spec.get("generation_tokens", {}).get(role, 512), do_sample=False,
            temperature=None, top_p=None, top_k=None, use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated = output[0, prompt.shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not raw:
        raise RuntimeError("Qwen3 trả nội dung rỗng")
    return {"id": request["id"], "role": role, "status": "ok", "raw_output": raw,
            "identity": {"repo": identity["repo"], "revision": identity["revision"], **identity["agents"][role]},
            "input_tokens": prompt.shape[1], "output_tokens": len(generated),
            "generation_seconds": time.perf_counter() - started,
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))["text_models"]
    try:
        with redirect_stdout(sys.stderr):
            model, tokenizer, dtype, identity = load(spec)
        print(json.dumps({"event": "ready", "status": "ok", "identity": identity}, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__ + ": " + str(exc)}, ensure_ascii=False), flush=True)
        return 1
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 100000:
                raise ValueError("Request vượt giới hạn")
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                response = generate(model, tokenizer, dtype, spec, request, identity)
                print(f'{request["role"]}: {response["input_tokens"]} input / {response["output_tokens"]} output tokens, '
                      f'{response["generation_seconds"]:.1f}s',file=sys.stderr,flush=True)
        except Exception as exc:
            response = {"id": request.get("id"), "role": request.get("role"), "status": "error",
                        "error": type(exc).__name__ + ": " + str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
