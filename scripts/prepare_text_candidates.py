"""Download pinned text candidates; never change production model assignments."""
import argparse
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_models import huggingface_dns
from huggingface_hub import HfApi, snapshot_download
from src.training.a1_runtime import file_hash, write_json
from src.training.model_candidates import DEFAULT_CATALOG, candidate_directory, read_candidates, verify_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--dns-fallback", action="store_true")
    parser.add_argument("--refresh-assets", action="store_true", help="Fetch missing official tokenizer/template assets without replacing verified weights")
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    candidates = read_candidates(args.catalog)
    wanted = set(args.models or [item["id"] for item in candidates])
    if wanted - {item["id"] for item in candidates}:
        parser.error("Unknown candidate")
    if args.inventory.exists():
        parser.error("Inventory already exists; choose a new file")
    rows = []
    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md", "LICENSE", "LICENSE.*"]
    with huggingface_dns(args.dns_fallback):
        for spec in candidates:
            if spec["id"] not in wanted:
                continue
            directory = candidate_directory(spec)
            if (directory / "download_manifest.json").exists():
                _, manifest = verify_candidate(spec)
                action = "verified_existing"
                if args.refresh_assets:
                    info = HfApi(token=False).model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
                    if info.sha != spec["revision"] or info.gated:
                        raise ValueError("Unexpected revision or gated candidate")
                    missing = [item for item in info.siblings
                               if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
                               and item.rfilename not in manifest["sha256"]]
                    if missing:
                        snapshot_download(spec["repo"], revision=spec["revision"], local_dir=directory,
                                          allow_patterns=[item.rfilename for item in missing], token=False, max_workers=2)
                        for item in missing:
                            checksum = file_hash(directory / item.rfilename)
                            if item.lfs is not None and checksum != item.lfs.sha256:
                                raise ValueError("Downloaded asset hash mismatch: " + item.rfilename)
                            manifest["sha256"][item.rfilename] = checksum
                        manifest["assets_refreshed_at"] = datetime.now(timezone.utc).isoformat()
                        write_json(directory / "download_manifest.json", manifest)
                        verify_candidate(spec)
                        action = "verified_existing_added_assets"
            else:
                info = HfApi(token=False).model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
                if info.sha != spec["revision"] or info.gated:
                    raise ValueError("Unexpected revision or gated candidate")
                print(f"Downloading {spec['id']} {spec['repo']} @ {info.sha}", flush=True)
                snapshot_download(spec["repo"], revision=spec["revision"], local_dir=directory,
                                  allow_patterns=patterns, token=False, max_workers=2)
                hashes = {}
                for item in info.siblings:
                    if not any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns):
                        continue
                    checksum = file_hash(directory / item.rfilename)
                    if item.lfs is not None and checksum != item.lfs.sha256:
                        raise ValueError("Downloaded file hash mismatch: " + item.rfilename)
                    hashes[item.rfilename] = checksum
                manifest = {"repo": spec["repo"], "revision": info.sha,
                            "downloaded_at": datetime.now(timezone.utc).isoformat(), "sha256": hashes}
                write_json(directory / "download_manifest.json", manifest)
                verify_candidate(spec)
                action = "downloaded_verified"
            row = {**spec, "directory": str(directory), "action": action,
                   "bytes": sum((directory / name).stat().st_size for name in manifest["sha256"]),
                   "manifest_sha256": file_hash(directory / "download_manifest.json")}
            rows.append(row)
            print(f"Ready: {spec['id']} ({row['bytes']/1e9:.2f} GB) -> {directory}", flush=True)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.inventory, {"catalog_sha256": file_hash(args.catalog), "models": rows,
                               "production_config_changed": False, "image_agent": "deferred"})


if __name__ == "__main__":
    main()
