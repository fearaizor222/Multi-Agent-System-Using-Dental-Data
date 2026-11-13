"""Không tái sử dụng điểm cũ khi điều kiện so sánh đã đổi."""
import json
from copy import deepcopy

import pytest

from src.training.a1_baseline_cache import checked_baseline
from src.training.a1_evaluation import score_response, summarize
from src.training.a1_runtime import ROOT, file_hash


@pytest.fixture
def cache(tmp_path):
    source = tmp_path / "source"
    (source / "baseline_validation").mkdir(parents=True)
    config = {"base_model": "model", "revision": "revision", "model_dir": "local/model", "dataset_dir": "local/data",
              "generation": {"max_new_tokens": 256}}
    integrity = {"model_sha256": {"model": "123"}, "dataset_manifest_sha256": "456"}
    versions = {"torch": "version"}
    context = {"message": "Xin chào", "history": [], "attachments": [],
               "capabilities": {"A2": True, "A5": True, "lookup_clinic_service": True}}
    target = {"decision": "respond", "targets": [], "service_topics": [], "image_ids": [],
              "resolved_request": None, "message": "Xin chào!"}
    validation = [{"id": "one", "input": context, "target": target}]
    raw = json.dumps(target)
    row = {"id": "one", "category": "social", "input": context, "expected": target, "raw_output": raw,
           "elapsed_seconds": 1, **score_response(context, target, raw)}
    runtime = {"packages": versions, "code_sha256": {str((ROOT / name).relative_to(ROOT)): file_hash(ROOT / name)
        for name in ("src/training/a1_runtime.py", "src/training/a1_evaluation.py", "src/data/a1_dataset.py")}}
    for name, content in {"config.json": config, "summary.json": {"status": "completed", "test_evaluated": False},
                          "input_integrity.json": integrity, "runtime.json": runtime,
                          "baseline_validation/metrics.json": summarize([row])}.items():
        (source / name).write_text(json.dumps(content), encoding="utf-8")
    (source / "baseline_validation/predictions.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
    return source, config, integrity, validation, versions


def test_cached_raw_scores_accepted_with_changed_training_hyperparameters(cache):
    source, config, integrity, validation, versions = cache
    config = {**config, "epochs": 1, "learning_rate": .0002}
    metrics, rows, provenance = checked_baseline(source, config, integrity, validation, versions)
    assert metrics["correct"]["control_exact"] == 1
    assert provenance["validation_examples_verified"] == 1


def test_different_generation_or_dataset_cannot_use_baseline(cache):
    source, config, integrity, validation, versions = cache
    changed = deepcopy(config)
    changed["generation"]["max_new_tokens"] = 128
    with pytest.raises(ValueError, match="generation"):
        checked_baseline(source, changed, integrity, validation, versions)
    changed_integrity = {**integrity, "dataset_manifest_sha256": "new"}
    with pytest.raises(ValueError, match="dataset_manifest"):
        checked_baseline(source, config, changed_integrity, validation, versions)


def test_changed_aggregate_score_is_rejected(cache):
    source, config, integrity, validation, versions = cache
    path = source / "baseline_validation/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["correct"]["control_exact"] = 99
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="Tổng hợp"):
        checked_baseline(source, config, integrity, validation, versions)


def test_train_change_requires_identical_untampered_protected_splits(cache,tmp_path):
    source,config,integrity,validation,versions=cache
    configs=[];integrities=[]
    for label in ("old","new"):
        folder=tmp_path/label;folder.mkdir()
        hashes={}
        for split in ("validation","test"):
            for suffix in (".jsonl",".metadata.jsonl"):
                p=folder/(split+suffix);p.write_text("protected",encoding="utf-8")
                hashes[p.name]=file_hash(p)
        manifest=folder/"manifest.json"
        manifest.write_text(json.dumps({"files_sha256":hashes,"training_version":label}),encoding="utf-8")
        configs.append({**config,"dataset_dir":str(folder)})
        integrities.append({**integrity,"dataset_manifest_sha256":file_hash(manifest)})
    (source/"config.json").write_text(json.dumps(configs[0]),encoding="utf-8")
    (source/"input_integrity.json").write_text(json.dumps(integrities[0]),encoding="utf-8")
    _,_,provenance=checked_baseline(source,configs[1],integrities[1],validation,versions,allow_train_only_change=True)
    assert provenance["reuse_scope"]=="identical_protected_splits_train_changed"
    # Cả khi người sửa cập nhật checksum của bản mới, nội dung holdout khác vẫn bị từ chối.
    folder=tmp_path/"new"
    (folder/"test.jsonl").write_text("changed",encoding="utf-8")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    manifest["files_sha256"]["test.jsonl"]=file_hash(folder/"test.jsonl")
    (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    integrities[1]["dataset_manifest_sha256"]=file_hash(folder/"manifest.json")
    with pytest.raises(ValueError,match="thay holdout"):
        checked_baseline(source,configs[1],integrities[1],validation,versions,allow_train_only_change=True)
