"""Read-only cleanup plan, restricted to project checkpoints and HF model caches."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/model_research_catalog_20261106.json")
    parser.add_argument("--download-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    downloaded = json.loads((args.download_dir / "download_inventory.json").read_text(encoding="utf-8"))
    if downloaded["catalog_sha256"] != sha256(args.catalog):
        raise ValueError("Catalog changed since download")
    if {item["repo"] for item in downloaded["models"] if item["verified"]} != {item["repo"] for item in catalog["models"]}:
        raise ValueError("Download list incomplete")
    keep_repos = {item["repo"] for item in [*catalog["models"], *catalog["support_models"]]}
    model_root = (ROOT / "data/models").resolve()
    cache_root = (Path.home() / ".cache/huggingface/hub").resolve()
    keep_local = {repo.replace("/", "--") for repo in keep_repos}
    keep_cache = {"models--" + name for name in keep_local}
    targets = []
    for root, names, kind in ((model_root, keep_local, "project_checkpoint"),
                              (cache_root, keep_cache, "huggingface_model_cache")):
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name in names:
                continue
            if kind == "huggingface_model_cache" and not path.name.startswith("models--"):
                continue
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("Refusing linked/outside cleanup target: " + str(path))
            targets.append({"path": str(path.resolve()), "allowed_root": str(root), "kind": kind,
                            "bytes": folder_size(path), "name": path.name})
    lock_root = cache_root / ".locks"
    if lock_root.exists():
        for path in sorted(lock_root.iterdir()):
            if path.is_dir() and path.name.startswith("models--") and path.name not in keep_cache:
                if path.is_symlink() or not path.resolve().is_relative_to(lock_root):
                    raise ValueError("Refusing linked/outside cache lock target")
                targets.append({"path": str(path.resolve()), "allowed_root": str(lock_root),
                                "kind": "excluded_model_cache_locks", "bytes": folder_size(path), "name": path.name})
    artifacts = {}
    for path in (ROOT / "outputs/a1_training").rglob("adapter_model.safetensors"):
        artifacts[str(path.resolve())] = sha256(path)
    plan = {"authorization": "User explicitly requested deleting models outside the selected catalog after filtering.",
            "catalog": str(args.catalog.resolve()), "catalog_sha256": sha256(args.catalog),
            "download_inventory": str((args.download_dir / "download_inventory.json").resolve()),
            "allowed_roots": [str(model_root), str(cache_root), str(lock_root)],
            "keep_repos": sorted(keep_repos), "kept_local_names": sorted(keep_local),
            "kept_cache_names": sorted(keep_cache), "targets": targets,
            "expected_removed_bytes": sum(item["bytes"] for item in targets),
            "retained_a1_adapter_hashes": artifacts,
            "note": "Plan only. No files deleted by this script. Datasets, vector indices, reports, virtual environments and selected-model A1 adapters are outside cleanup targets."}
    output = args.download_dir / "cleanup_plan.json"
    if output.exists():
        raise FileExistsError("Cleanup plan already exists")
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(output), "targets": targets,
                      "expected_removed_gb": round(plan["expected_removed_bytes"] / 1e9, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
