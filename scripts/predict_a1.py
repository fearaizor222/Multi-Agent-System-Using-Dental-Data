"""Chạy thử bộ phân luồng A1 đã lưu; chỉ in kế hoạch, không gọi tác tử/công cụ."""
import argparse
import json
from pathlib import Path

from src.data.a1_dataset import A1Input, SYSTEM_PROMPT, validate_pair
from src.training.a1_evaluation import strict_json
from src.training.a1_runtime import ROOT, load_base, file_hash, write_json


def load_candidate(run_dir):
    """Kiểm tra checksum, nạp một lần để dùng cho CLI hoặc đánh giá phát triển."""
    from peft import PeftModel
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Lượt huấn luyện chưa hoàn tất")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    integrity = json.loads((run_dir / "input_integrity.json").read_text(encoding="utf-8"))
    for filename, checksum in integrity["model_sha256"].items():
        if file_hash(ROOT / config["model_dir"] / filename) != checksum:
            raise ValueError("Model nền checksum không khớp: " + filename)
    adapter_dir = Path(summary["best_adapter_dir"])
    manifest = json.loads((adapter_dir.parent / "adapter_manifest.json").read_text(encoding="utf-8"))
    if manifest["base_model"] != config["base_model"] or manifest["revision"] != config["revision"]:
        raise ValueError("Adapter khác model nền/revision")
    for filename, checksum in manifest["files_sha256"].items():
        if file_hash(adapter_dir / filename) != checksum:
            raise ValueError("Adapter checksum không khớp: " + filename)
    model, tokenizer, dtype = load_base(config)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, dtype, config, adapter_dir


def predict(run_dir, context, max_input_tokens=None):
    import torch
    model, tokenizer, dtype, config, adapter_dir = load_candidate(run_dir)
    context = A1Input.model_validate(context).model_dump()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt")
    if max_input_tokens is not None and prompt.shape[1]>max_input_tokens:
        raise ValueError(f"A1_INPUT_TOO_LONG: {prompt.shape[1]} > {max_input_tokens}")
    prompt=prompt.to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=config["generation"]["max_new_tokens"], do_sample=False,
            temperature=None, top_p=None, top_k=None, use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, prompt.shape[1]:], skip_special_tokens=True)
    result = {"input": context, "adapter_dir": str(adapter_dir), "raw_output": raw,
              "valid": False, "decision": None, "error": None, "tools_executed": False}
    try:
        _, decision = validate_pair(context, strict_json(raw))
        result.update(valid=True, decision=decision)
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question")
    source.add_argument("--input-json", type=Path, help="Đầu vào theo A1Input, gồm capability do ứng dụng xác nhận")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("File đầu ra đã có; chọn file khác")
    if args.question is not None:
        # Phần dịch vụ/ảnh chưa được tích hợp vào runtime: không mặc định chúng khả dụng.
        context = {"message": args.question, "history": [], "attachments": [],
                   "capabilities": {"A2": True, "A5": False, "lookup_clinic_service": False}}
    else:
        context = json.loads(args.input_json.read_text(encoding="utf-8"))
    A1Input.model_validate(context)
    result = predict(args.run_dir, context)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
