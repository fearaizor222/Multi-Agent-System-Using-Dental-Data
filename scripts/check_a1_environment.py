"""Kiểm tra CUDA/QLoRA trên mẫu train; không lưu adapter và không đánh giá test."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets", "pydantic")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def encode_assistant_sample(tokenizer, messages):
    if len(messages) != 3 or [item["role"] for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("Mẫu cần system/user/assistant")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                         enable_thinking=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template không khớp prefix; không được mask theo độ dài ước đoán")
    if len(prompt) >= len(full):
        raise ValueError("Không có token đáp án để huấn luyện")
    return full, [-100] * len(prompt) + full[len(prompt):], len(prompt)


def completion_only_loss(model, batch, prompt_length, token_weights=None):
    """Cùng masked causal loss, chỉ dựng logits ở vị trí dự đoán đáp án cuối."""
    import torch
    length = batch["input_ids"].shape[1]
    if batch["input_ids"].shape[0] != 1 or not 0 < prompt_length < length:
        raise ValueError("Hàm kiểm tra chỉ hỗ trợ một mẫu với đáp án cuối không rỗng")
    if not bool((batch["labels"][:, :prompt_length] == -100).all()) or bool((batch["labels"][:, prompt_length:] == -100).any()):
        raise ValueError("Mask không đúng dạng prompt + đáp án liên tục")
    positions = torch.arange(prompt_length-1, length-1, device=batch["input_ids"].device)
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], logits_to_keep=positions)
    targets = batch["labels"][:, prompt_length:]
    if output.logits.shape[:2] != targets.shape:
        raise ValueError("Model không trả logits đúng vị trí đáp án")
    losses=torch.nn.functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), targets.reshape(-1),reduction="none")
    if token_weights is None:
        return losses.mean()
    if token_weights.shape!=targets.shape or not bool(torch.isfinite(token_weights).all()) or bool((token_weights<=0).any()):
        raise ValueError("Trọng số token phải dương, hữu hạn và khớp đáp án")
    weights=token_weights.float().reshape(-1)
    return (losses*weights).sum()/weights.sum()


def model_smoke(config, report):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from scripts.check_a1_dataset import check_dataset

    model_dir = ROOT / config["model_dir"]
    model_manifest = json.loads((model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    if model_manifest["repo"] != config["base_model"] or model_manifest["revision"] != config["revision"]:
        raise ValueError("Model khác checkpoint đã chốt")
    for name, checksum in model_manifest["sha256"].items():
        if digest(model_dir / name) != checksum:
            raise ValueError("Model checksum sai: " + name)
    report["model_integrity"] = "passed"
    dataset_dir = ROOT / config["dataset_dir"]
    report["dataset_integrity"] = check_dataset(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [encode_assistant_sample(tokenizer, row["messages"]) for row in rows]
    index = max(range(len(samples)), key=lambda i: len(samples[i][0]))
    ids, labels, prompt_length = samples[index]
    if len(ids) > config["max_length"]:
        raise ValueError("Mẫu vượt max_length; không tự cắt đáp án")
    train_metadata = [json.loads(line) for line in (dataset_dir / "train.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    report["sample"] = {"id": train_metadata[index]["id"], "split": "train", "tokens": len(ids),
                         "prompt_tokens_masked": prompt_length, "supervised_tokens": len(ids)-prompt_length,
                         "assistant_only_loss": True, "truncated": False, "all_train_prefixes_checked": len(samples)}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    report["compute_dtype"] = str(dtype)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    print("Loading local Qwen3 in NF4 on cuda:0...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                quantization_config=quantization, device_map={"": 0}, torch_dtype=dtype,
                attn_implementation="sdpa", low_cpu_mem_usage=True)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model không được nạp ở 4-bit")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("Có tham số bị offload ngoài GPU; smoke này cần xác nhận GPU thật")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    # PEFT đổi trọng số không lượng tử hóa sang FP32. Embedding/head đóng băng
    # được giữ ở compute dtype để giảm VRAM; normalization vẫn FP32.
    model.get_input_embeddings().to(dtype=dtype)
    model.get_output_embeddings().to(dtype=dtype)
    report["frozen_embedding_dtype"] = str(model.get_input_embeddings().weight.dtype)
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                      lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, parameter in trainable):
        raise RuntimeError("Chỉ adapter LoRA được phép học")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
    report["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
    report["memory_after_load_mib"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    batch = {"input_ids": torch.tensor([ids], device="cuda"),
             "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
             "labels": torch.tensor([labels], device="cuda")}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=config["learning_rate"])
    model.train()
    print("Running one forward/backward/optimizer step on longest train sample...", flush=True)
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = completion_only_loss(model, batch, prompt_length)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss không hữu hạn")
    loss.backward()
    gradients = [parameter.grad for _, parameter in trainable if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("Gradient thiếu hoặc không hữu hạn")
    gradient_l1 = sum(float(gradient.detach().float().abs().sum()) for gradient in gradients)
    if not math.isfinite(gradient_l1) or gradient_l1 <= 0:
        raise RuntimeError("Gradient bằng 0 hoặc không hợp lệ")
    torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    changed = [name for name, parameter in trainable if not torch.equal(parameter.detach().cpu(), before[name])]
    if not changed:
        raise RuntimeError("Không có adapter được cập nhật")
    if not all(bool(torch.isfinite(parameter).all()) for _, parameter in trainable):
        raise RuntimeError("Adapter không hữu hạn sau cập nhật")
    peak = torch.cuda.max_memory_allocated()/2**20
    report["training_smoke"] = {"optimizer_steps": 1, "loss": float(loss.detach()),
        "gradient_l1": gradient_l1, "changed_adapter_tensors": len(changed),
        "elapsed_seconds": round(time.perf_counter()-start, 2),
        "peak_allocated_mib": round(peak, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20, 1),
        "allocated_below_dedicated_vram": peak < torch.cuda.get_device_properties(0).total_memory/2**20,
        "saved_adapter": False, "evaluated_test_split": False, "quality_evaluated": False}
    if not report["training_smoke"]["allocated_below_dedicated_vram"]:
        raise RuntimeError("Bước thử vượt bộ nhớ VRAM riêng")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/a1_training_smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("File kết quả đã có; chọn --output khác")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
              "python_version": sys.version, "platform": platform.platform(), "status": "running",
              "purpose": "environment_smoke_only", "full_training_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["packages"] = {name: importlib.metadata.version(name) for name in PACKAGES}
        for name in ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets"):
            __import__(name)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch chưa dùng được CUDA")
        gpu = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": gpu.name, "total_mib": round(gpu.total_memory/2**20, 1),
            "compute_capability": list(torch.cuda.get_device_capability(0)), "torch_cuda": torch.version.cuda,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],text=True).strip()}
        matrix = torch.randn(128, 128, device="cuda")
        if not torch.isfinite(matrix @ matrix).all():
            raise RuntimeError("Phép toán CUDA không hợp lệ")
        del matrix
        torch.cuda.empty_cache()
        report["cuda_matrix_check"] = "passed"
        if args.model_smoke:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            report["config"] = config
            model_smoke(config, report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
