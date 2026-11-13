"""Protect prepare-only mode, v4 quotas and failure propagation without GPU training."""
import json
import subprocess
from copy import deepcopy

import pytest

from scripts import finetune_a1_qwen3_v4 as workflow
from scripts.prepare_a1_dataset import build_records
from src.training.a1_runtime import ROOT, file_hash
from src.training.a1_strategy import plan_epoch


def config():
    return json.loads((ROOT / "config/a1_qwen3_17b_v4.json").read_text(encoding="utf-8"))


def train_rows():
    source = json.loads((ROOT / config()["source_dataset"]).read_text(encoding="utf-8"))
    return [row for row in build_records(source, config()["seed"]) if row["split"] == "train"]


def test_v4_sampling_covers_all_802_train_examples_and_456_route_examples():
    rows = train_rows()
    c = config()
    for epoch in range(1, c["epochs"] + 1):
        order, detail = plan_epoch(rows, c["seed"] + epoch, c["sampling"])
        assert set(order) == set(range(len(rows)))
        assert len(rows) == 802 and detail["samples"] == 1200
        assert detail["decision_counts"]["route"] == 480
        assert sum(row["target"]["decision"] == "route" for row in rows) == 456
    old = json.loads((ROOT / "config/a1_training_v5.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Quota"):
        plan_epoch(rows, c["seed"], old["sampling"])


def test_preflight_refuses_holdout_rows():
    rows = deepcopy(train_rows()[:1])
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="train"):
        workflow.training_input_audit(config(), rows, None)


def test_preflight_never_truncates_long_training_samples():
    c = config()
    c["sampling"] = None
    c["max_length"] = 2

    class TooLong:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3, 4] if messages[-1]["role"] == "assistant" else [1]

    with pytest.raises(ValueError, match="không tự cắt"):
        workflow.training_input_audit(c, train_rows()[:1], TooLong())


def test_prepare_only_never_calls_training(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"epochs":2}', encoding="utf-8")
    record = {"status": "ready", "integrity": {"dataset_check": {"by_split": {"train": 802, "validation": 59, "test": 59}}},
              "train_tokens": {"max": 882}, "sampling_epochs": [{"samples": 1200}]}
    monkeypatch.setattr(workflow, "preflight", lambda _: record)
    monkeypatch.setattr(workflow, "execute_training", lambda *args, **kwargs: pytest.fail("prepare-only tried to train"))
    monkeypatch.setattr("sys.argv", ["workflow", "--config", str(cfg), "--prepare-only", "--preflight-report", str(tmp_path / "audit.json")])
    workflow.main()
    assert json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))["status"] == "ready"


def test_training_failure_does_not_run_reporting_or_plotting(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        workflow.execute_training(cfg, tmp_path / "new_run", {"config_sha256": file_hash(cfg)}, runner=fail)
    assert len(calls) == 1 and "scripts.train_a1" in calls[0]


def test_changed_config_cannot_start_training(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="thay đổi"):
        workflow.execute_training(cfg, tmp_path / "new_run", {"config_sha256": "old"}, runner=lambda *args, **kwargs: pytest.fail("spawned training"))


def test_existing_dataset_is_not_overwritten_when_source_changes(monkeypatch, tmp_path):
    c = config()
    c.update(source_dataset="source.json", dataset_dir="prepared")
    (tmp_path / "source.json").write_text("{}", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text('{"source_sha256":"stale"}', encoding="utf-8")
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="không ghi đè"):
        workflow.preflight(c)
    assert json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))["source_sha256"] == "stale"
