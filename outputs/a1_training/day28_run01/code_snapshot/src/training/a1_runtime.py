"""Nạp cùng một cấu hình NF4 cho baseline và adapter A1 trên GPU 4 GB."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def verify_inputs(config):
    from scripts.check_a1_dataset import check_dataset
    model_dir = ROOT / config["model_dir"]
    manifest = json.loads((model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["repo"] != config["base_model"] or manifest["revision"] != config["revision"]:
        raise ValueError("Sai checkpoint Qwen3 đã chốt")
    for name, checksum in manifest["sha256"].items():
        if file_hash(model_dir / name) != checksum:
            raise ValueError("Model checksum không khớp: " + name)
    dataset_dir = ROOT / config["dataset_dir"]
    dataset_check = check_dataset(dataset_dir)
    return {"model_sha256": manifest["sha256"], "dataset_check": dataset_check,
            "dataset_manifest_sha256": file_hash(dataset_dir / "manifest.json"),
            "model_dir": str(model_dir.resolve()), "dataset_dir": str(dataset_dir.resolve())}


def load_base(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cấu hình đã kiểm tra yêu cầu CUDA với BF16")
    dtype = torch.bfloat16
    model_dir = ROOT / config["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
        quantization_config=quantization, device_map={"": 0}, dtype=dtype,
        attn_implementation="sdpa", low_cpu_mem_usage=True)
    if not model.is_loaded_in_4bit or any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("Model phải dùng NF4 và nằm trên GPU")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.get_input_embeddings().to(dtype=dtype)
    model.get_output_embeddings().to(dtype=dtype)
    return model, tokenizer, dtype


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets")}
