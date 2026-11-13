"""Freeze the existing A1 data/config for Modal without loading model weights."""
import argparse
import copy
import json
from pathlib import Path

from scripts.check_a1_dataset import check_dataset
from scripts.train_a1 import validate_config
from src.training.a1_runtime import ROOT, file_hash

CONFIG = ROOT / "config/a1_mediphi_38b_modal.json"


def load_payload():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    source = ROOT / config["source_dataset"]
    dataset = ROOT / config["dataset_dir"]
    checked = check_dataset(dataset, source)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    files = {name: (dataset / name).read_bytes() for name in dataset_manifest["files_sha256"]}
    manifest = json.loads((ROOT / config["model_dir"] / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["repo"] != config["base_model"] or manifest["revision"] != config["revision"]:
        raise ValueError("MediPhi checkpoint identity mismatch")
    for name in manifest["sha256"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe model filename")
    splits = {f"{split}{suffix}": file_hash(dataset / f"{split}{suffix}")
              for split in ("train", "validation", "test") for suffix in (".jsonl", ".metadata.jsonl")}
    return {"config": config, "source_bytes": source.read_bytes(), "dataset_files": files,
            "dataset_manifest": dataset_manifest,
            "source_sha256": file_hash(source), "split_sha256": splits,
            "model_manifest": manifest, "dataset_check": checked}


def materialize(payload, root):
    """Rebuild the same records at Linux paths, retaining exact split bytes."""
    root = Path(root)
    config = copy.deepcopy(payload["config"])
    source = root / config["source_dataset"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload["source_bytes"])
    if file_hash(source) != payload["source_sha256"]:
        raise ValueError("Source bytes changed in transfer")
    dataset = root / config["dataset_dir"]
    dataset.mkdir(parents=True, exist_ok=False)
    for name, content in payload["dataset_files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe dataset filename")
        target = dataset / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = copy.deepcopy(payload["dataset_manifest"])
    manifest["source"] = str(source.resolve())
    (dataset / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, expected in payload["split_sha256"].items():
        if file_hash(dataset / name) != expected:
            raise ValueError("Split changed in transfer: " + name)
    check_dataset(dataset, source)
    return config


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    p = load_payload()
    print(json.dumps({"status": "ready", "model": p["config"]["base_model"],
        "split_counts": p["dataset_check"]["by_split"], "epochs": p["config"]["epochs"],
        "training_started": False}, ensure_ascii=False, indent=2))
