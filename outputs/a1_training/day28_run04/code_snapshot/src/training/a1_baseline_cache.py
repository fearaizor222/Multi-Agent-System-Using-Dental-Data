"""Chỉ tái sử dụng baseline khi model, prompt, dữ liệu và bộ chấm trùng khớp."""
import json
from pathlib import Path

from src.training.a1_evaluation import score_response, summarize
from src.training.a1_runtime import ROOT, file_hash


def protected_hashes(config, integrity):
    directory=ROOT/config["dataset_dir"]
    manifest_path=directory/"manifest.json"
    if file_hash(manifest_path)!=integrity["dataset_manifest_sha256"]:
        raise ValueError("Manifest nguồn baseline/dataset đã thay đổi")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes={}
    for split in ("validation","test"):
        for suffix in (".jsonl",".metadata.jsonl"):
            name=split+suffix
            checksum=file_hash(directory/name)
            if checksum!=manifest["files_sha256"][name]:
                raise ValueError("Holdout checksum không khớp: "+name)
            hashes[name]=checksum
    return hashes


def checked_baseline(source, config, integrity, validation, versions, allow_train_only_change=False):
    source = Path(source)
    saved_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    saved_integrity = json.loads((source / "input_integrity.json").read_text(encoding="utf-8"))
    runtime = json.loads((source / "runtime.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["test_evaluated"]:
        raise ValueError("Nguồn baseline phải là lượt validation đã hoàn tất")
    for key in ("base_model", "revision", "model_dir", "generation"):
        if saved_config[key] != config[key]:
            raise ValueError("Không được tái sử dụng baseline khi thay đổi " + key)
    if saved_integrity["model_sha256"]!=integrity["model_sha256"]:
        raise ValueError("Baseline khác model_sha256")
    data_changed=(saved_config["dataset_dir"]!=config["dataset_dir"] or saved_integrity["dataset_manifest_sha256"]!=integrity["dataset_manifest_sha256"])
    protected=None
    if data_changed:
        if not allow_train_only_change:
            raise ValueError("Baseline khác dataset_dir hoặc dataset_manifest_sha256")
        protected=protected_hashes(config,integrity)
        if protected!=protected_hashes(saved_config,saved_integrity):
            raise ValueError("Không được tái sử dụng baseline khi thay holdout")
    if runtime["packages"] != versions:
        raise ValueError("Phiên bản thư viện baseline không trùng")
    for relative in ("src/training/a1_runtime.py", "src/training/a1_evaluation.py", "src/data/a1_dataset.py"):
        path = ROOT / relative
        key = str(path.relative_to(ROOT))
        if runtime["code_sha256"][key] != file_hash(path):
            raise ValueError("Mã nạp model/bộ chấm/policy đã thay đổi: " + relative)
    prediction_path = source / "baseline_validation/predictions.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(predictions) != len(validation) or len({row["id"] for row in predictions}) != len(predictions):
        raise ValueError("Số mẫu baseline không khớp")
    for row, expected in zip(predictions, validation):
        if row["id"] != expected["id"] or row["input"] != expected["input"] or row["expected"] != expected["target"]:
            raise ValueError("Mẫu/nhãn baseline không trùng validation hiện tại")
        recalculated = score_response(row["input"], row["expected"], row["raw_output"])
        if any(row[key] != recalculated[key] for key in ("raw_json_valid", "schema_valid", "context_valid", "decision_correct", "targets_correct", "service_topics_correct", "image_ids_correct", "control_exact")):
            raise ValueError("Điểm baseline không khớp khi chấm lại raw output")
    metrics = json.loads((source / "baseline_validation/metrics.json").read_text(encoding="utf-8"))
    if not metrics["completed"] or metrics["count"] != len(predictions):
        raise ValueError("Baseline chưa đủ mẫu")
    recalculated_metrics = summarize(predictions)
    if any(metrics[key] != recalculated_metrics[key] for key in ("correct", "rates", "decision_macro_f1")):
        raise ValueError("Tổng hợp baseline không khớp raw output")
    return metrics, predictions, {"source_run": str(source.resolve()),
        "raw_predictions_sha256": file_hash(prediction_path),
        "metrics_sha256": file_hash(source / "baseline_validation/metrics.json"),
        "validation_examples_verified": len(predictions), "score_recalculation": "passed",
        "reuse_scope":"identical_protected_splits_train_changed" if data_changed else "identical_dataset",
        "protected_file_sha256":protected,
        "source_dataset_manifest_sha256":saved_integrity["dataset_manifest_sha256"],
        "current_dataset_manifest_sha256":integrity["dataset_manifest_sha256"]}
