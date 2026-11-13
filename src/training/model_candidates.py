"""Candidate catalog/integrity checks; independent of production model settings."""
import json
import re
from pathlib import Path

from src.training.a1_runtime import ROOT, file_hash

DEFAULT_CATALOG = ROOT / "config/text_model_candidates.json"


def read_candidates(path=DEFAULT_CATALOG):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    models = catalog["models"]
    ids = [item["id"] for item in models]
    if not models or len(ids) != len(set(ids)):
        raise ValueError("Empty/duplicate candidate IDs")
    for item in models:
        if (not re.fullmatch(r"[a-z0-9_]+", item["id"])
                or not re.fullmatch(r"[\w.-]+/[\w.-]+", item["repo"])
                or not re.fullmatch(r"[a-f0-9]{40}", item["revision"])):
            raise ValueError("Unsafe candidate identity")
    return models


def candidate_directory(spec):
    return ROOT / "data/models" / spec["repo"].replace("/", "--") / spec["revision"]


def verify_candidate(spec, directory=None):
    directory = Path(directory) if directory is not None else candidate_directory(spec)
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != spec[key] for key in ("repo", "revision")):
        raise ValueError("Candidate manifest identity mismatch")
    hashes = manifest.get("sha256", {})
    if not hashes or not any(name.endswith(".safetensors") for name in hashes):
        raise ValueError("Candidate manifest has no weights")
    for name, checksum in hashes.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file():
            raise ValueError("Invalid/missing model file: " + name)
        if file_hash(target) != checksum:
            raise ValueError("Candidate checksum mismatch: " + name)
    return directory, manifest
