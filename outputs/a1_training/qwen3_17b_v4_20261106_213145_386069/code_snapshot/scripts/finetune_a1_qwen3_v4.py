"""Prepare A1 v4, then explicitly --train Qwen3-1.7B and export report figures."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.check_a1_environment import encode_assistant_sample
from scripts.prepare_a1_dataset import export
from scripts.train_a1 import validate_config
from src.training.a1_runtime import ROOT, file_hash, verify_inputs, write_json
from src.training.a1_strategy import assistant_weights, plan_epoch

DEFAULT_CONFIG = ROOT / "config/a1_qwen3_17b_v4.json"


def training_input_audit(config, rows, tokenizer):
    """CPU tokenization/sampling only; never loads weights or computes gradients."""
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Preflight chỉ nhận các mẫu train")
    plans = [plan_epoch(rows, config["seed"] + epoch, config.get("sampling"))
             for epoch in range(1, config["epochs"] + 1)]
    lengths = []
    for row in rows:
        ids, labels, prefix = encode_assistant_sample(tokenizer, row["messages"])
        if len(ids) > config["max_length"]:
            raise ValueError(f"Mẫu {row['id']} dài {len(ids)} token, vượt max_length={config['max_length']}; không tự cắt dữ liệu")
        assistant_weights(tokenizer, row["messages"], ids, prefix, config.get("token_weighting"))
        lengths.append(len(ids))
    return {"unique_train_samples": len(rows), "train_decisions": dict(Counter(row["target"]["decision"] for row in rows)),
            "train_tokens": {"min": min(lengths), "max": max(lengths), "truncated_samples": 0},
            "sampling_epochs": [detail for _, detail in plans]}


def preflight(config):
    validate_config(config)
    if config["base_model"] != "Qwen/Qwen3-1.7B":
        raise ValueError("Entry point này dành riêng cho Qwen3-1.7B")
    source = ROOT / config["source_dataset"]
    if not source.is_file():
        raise FileNotFoundError("Không có scenarios.json đã chỉ định: " + str(source))
    dataset_dir = ROOT / config["dataset_dir"]
    if dataset_dir.exists():
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != file_hash(source):
            raise ValueError("scenarios.json khác bản đã xuất; đổi dataset_dir sang thư mục mới, không ghi đè dữ liệu cũ")
    else:
        export(source, dataset_dir, config["seed"], ROOT / config["model_dir"])
    integrity = verify_inputs(config)
    rows = [json.loads(line) for line in (dataset_dir / "train.metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model_dir"], local_files_only=True, trust_remote_code=False)
    audit = training_input_audit(config, rows, tokenizer)
    try:
        plotting_version = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Thiếu matplotlib; cài requirements-a1-plots.txt bằng môi trường .venv-a1") from exc
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Cần môi trường .venv-a1 với CUDA/BF16; preflight chưa nạp trọng số hoặc huấn luyện")
    return {"status": "ready", "source": str(source.resolve()), "source_sha256": file_hash(source),
            "dataset_dir": str(dataset_dir.resolve()), "integrity": integrity, **audit,
            "environment": {"python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0), "matplotlib": plotting_version},
            "model_weights_loaded": False, "inference_executed": False, "gradient_executed": False,
            "scope": "File/data checks and tokenizer only. Test labels checked for split integrity, not evaluated by a model."}


def execute_training(config_path, output_dir, prepare, runner=subprocess.run):
    """Only invoked by --train; child failure prevents a misleading success report."""
    if output_dir.exists():
        raise FileExistsError("Thư mục run đã tồn tại; chọn tên mới để giữ log cũ: " + str(output_dir))
    if file_hash(config_path) != prepare["config_sha256"]:
        raise ValueError("Cấu hình thay đổi sau preflight; kiểm tra lại trước khi huấn luyện")
    command = [sys.executable, "-X", "utf8", "-m", "scripts.train_a1", "--config", str(config_path), "--output-dir", str(output_dir)]
    try:
        runner(command, cwd=ROOT, check=True)
    finally:
        if output_dir.is_dir():
            write_json(output_dir / "preflight.json", prepare)
            for relative in ("scripts/finetune_a1_qwen3_v4.py", "scripts/plot_a1_training.py", "src/training/a1_plotting.py", "requirements-a1-plots.txt"):
                destination = output_dir / "code_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
    runner([sys.executable, "-X", "utf8", "-m", "scripts.report_a1_training", str(output_dir)], cwd=ROOT, check=True)
    runner([sys.executable, "-X", "utf8", "-m", "scripts.plot_a1_training", "--run-dir", str(output_dir)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="Check/export data, tokenizer lengths and quotas; no weights or gradients")
    mode.add_argument("--train", action="store_true", help="Explicitly run baseline, training, validation and saved-log plotting")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir.resolve() if args.output_dir else ROOT / f"outputs/a1_training/qwen3_17b_v4_{timestamp}"
    if args.train and output.exists():
        parser.error("Output đã tồn tại; chọn tên run mới")
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    prepare = preflight(config)
    if file_hash(config_path) != config_hash:
        raise ValueError("Cấu hình thay đổi trong lúc preflight; chưa chạy huấn luyện")
    prepare["config_sha256"] = config_hash
    audit_path = args.preflight_report or ROOT / f"outputs/a1_setup/qwen3_v4_{timestamp}/preflight.json"
    if audit_path.exists():
        parser.error("Preflight report đã tồn tại; chọn tên mới")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(audit_path, prepare)
    print(json.dumps({"status": prepare["status"], "split_counts": prepare["integrity"]["dataset_check"]["by_split"],
        "train_tokens": prepare["train_tokens"], "samples_per_epoch": prepare["sampling_epochs"][0]["samples"],
        "epochs": config["epochs"], "preflight_report": str(audit_path), "training_requested": args.train}, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print("Preflight complete. No weights loaded, inference or training executed.", flush=True)
        return
    execute_training(config_path, output, prepare)
    print("Completed. Report:", output / "report.md", "Figures:", output / "figures", flush=True)


if __name__ == "__main__":
    main()
