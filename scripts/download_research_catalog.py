"""Download/catalog integrity ONLY. No model loading, inference, gradient or training."""
import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import requests
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def dns_fallback(enabled):
    original, cache = socket.getaddrinfo, {}

    def resolve(host, port, *args, **kwargs):
        try:
            return original(host, port, *args, **kwargs)
        except socket.gaierror:
            if not isinstance(host, str) or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))):
                raise
            if host not in cache:
                request = urllib.request.Request("https://1.1.1.1/dns-query?name=" + host + "&type=A",
                                                 headers={"Accept": "application/dns-json"})
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = json.load(response)
                cache[host] = next(item["data"] for item in payload.get("Answer", []) if item["type"] == 1)
            return original(cache[host], port, *args, **kwargs)

    if enabled:
        socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original


def choose_files(siblings):
    root_files = [item for item in siblings if "/" not in item.rfilename and "\\" not in item.rfilename]
    safe = [item for item in root_files if re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", item.rfilename)]
    weights = safe or [item for item in root_files if re.fullmatch(r"pytorch_model(?:-\d+-of-\d+)?\.bin", item.rfilename)]
    if not weights:
        raise ValueError("No complete native checkpoint found; refusing adapter-only/GGUF substitution")
    assets = [item for item in root_files if item.rfilename.endswith((".json", ".model", ".txt", ".jinja", ".md", ".py"))
              or item.rfilename.startswith(("LICENSE", "NOTICE"))]
    selected = {item.rfilename: item for item in [*weights, *assets]}
    return list(selected.values()), "safetensors" if safe else "pytorch_bin"


def download_ranges(spec, item, directory):
    """Stream a large immutable LFS file in verified HTTP ranges; hash checked later."""
    target = directory / item.rfilename
    if target.exists():
        return
    partial = directory / ("." + item.rfilename + ".download.part")
    directory.mkdir(parents=True, exist_ok=True)
    total = item.size
    if total is None or (partial.exists() and partial.stat().st_size > total):
        raise ValueError("Invalid partial download length")
    chunk_size = 64 * 1024 * 1024
    previous_percent = -1
    with requests.Session() as session:
        while (partial.stat().st_size if partial.exists() else 0) < total:
            for attempt in range(5):
                start = partial.stat().st_size if partial.exists() else 0
                end = min(start + chunk_size, total) - 1
                try:
                    metadata = get_hf_file_metadata(hf_hub_url(spec["repo"], item.rfilename,
                        revision=spec["revision"]), token=False, timeout=15)
                    if metadata.size != total:
                        raise ValueError("Range source size differs from pinned file metadata")
                    with session.get(metadata.location, headers={"Range": f"bytes={start}-{end}",
                                     "Accept-Encoding": "identity"}, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                            raise ValueError("Server did not return the exact requested range")
                        received = 0
                        with partial.open("ab") as stream:
                            for block in response.iter_content(4 * 1024 * 1024):
                                if block:
                                    if received + len(block) > end - start + 1:
                                        raise ValueError("Range response exceeds expected length")
                                    stream.write(block)
                                    received += len(block)
                        if received != end - start + 1:
                            raise OSError("Incomplete range response")
                    break
                except (requests.RequestException, OSError) as exc:
                    if attempt == 4:
                        raise
                    print(f"Range retry {attempt+1}: {type(exc).__name__}", flush=True)
                    time.sleep(min(2 ** attempt, 8))
            completed = partial.stat().st_size
            percent = int(completed * 100 / total)
            if percent // 5 != previous_percent // 5:
                print(f"{spec['id']} {item.rfilename}: {completed/1e9:.2f}/{total/1e9:.2f} GB ({percent}%)", flush=True)
                previous_percent = percent
    if partial.stat().st_size != total:
        raise ValueError("Incomplete ranged checkpoint")
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/model_research_catalog_20261106.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dns-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this same catalog's partial file downloads")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    scope = catalog["scope"]
    if any(scope[key] for key in ("inference_allowed", "benchmark_allowed", "gradient_probe_allowed", "finetuning_allowed")):
        raise ValueError("This downloader only accepts a download-only session")
    if args.output_dir.exists():
        if not args.resume or checksum(args.output_dir / "catalog_snapshot.json") != checksum(args.catalog):
            raise ValueError("Existing output requires --resume with an identical catalog")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.catalog, args.output_dir / "catalog_snapshot.json")
    model_root = (ROOT / "data/models").resolve()
    api, plans, rows = HfApi(token=False), [], []
    with dns_fallback(args.dns_fallback):
        for spec in catalog["models"]:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", spec["repo"]) or not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
                raise ValueError("Unsafe model identity")
            directory = (model_root / spec["repo"].replace("/", "--") / spec["revision"]).resolve()
            if not directory.is_relative_to(model_root):
                raise ValueError("Model path escapes model root")
            info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
            if info.sha != spec["revision"] or info.gated:
                raise ValueError("Unexpected revision or inaccessible candidate: " + spec["repo"])
            files, weight_format = choose_files(info.siblings)
            missing_bytes = sum(item.size or 0 for item in files if not (directory / item.rfilename).exists())
            plans.append({"id": spec["id"], "repo": spec["repo"], "revision": spec["revision"],
                          "directory": str(directory), "format": weight_format, "new_bytes_estimate": missing_bytes,
                          "files": [item.rfilename for item in files]})
        write_json(args.output_dir / "download_plan.json", plans)
        required = sum(item["new_bytes_estimate"] for item in plans)
        available = shutil.disk_usage(model_root).free
        if available < required + 3 * 1024**3:
            raise RuntimeError(f"Insufficient free disk: need {required} bytes plus 3 GiB reserve")
        print(f"Download-only catalog: {len(plans)} models; new files approximately {required/1e9:.2f} GB", flush=True)
        for spec, plan in zip(catalog["models"], plans):
            directory = Path(plan["directory"])
            manifest_path = directory / "download_manifest.json"
            existing = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if any(manifest.get(key) != spec[key] for key in ("repo", "revision")):
                    raise ValueError("Existing manifest identity mismatch")
                for name, expected in manifest["sha256"].items():
                    target = (directory / name).resolve()
                    if not target.is_relative_to(directory) or checksum(target) != expected:
                        raise ValueError("Existing model file mismatch: " + name)
                existing = manifest["sha256"]
            info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
            files, weight_format = choose_files(info.siblings)
            missing = [item.rfilename for item in files if item.rfilename not in existing]
            print(f"Preparing {spec['id']} ({weight_format}); {len(missing)} files to fetch", flush=True)
            large = [item for item in files if item.rfilename in missing
                     and item.rfilename.endswith((".bin", ".safetensors")) and (item.size or 0) > 1_000_000_000]
            for item in large:
                download_ranges(spec, item, directory)
            missing = [name for name in missing if name not in {item.rfilename for item in large}]
            if missing:
                snapshot_download(spec["repo"], revision=spec["revision"], local_dir=directory,
                                  allow_patterns=missing, token=False, max_workers=2)
            hashes = dict(existing)
            for item in files:
                name = item.rfilename
                actual = existing.get(name) or checksum(directory / name)
                if item.lfs is not None and actual != item.lfs.sha256:
                    raise ValueError("Published LFS SHA-256 mismatch: " + name)
                if item.size is not None and (directory / name).stat().st_size != item.size:
                    raise ValueError("File size mismatch: " + name)
                hashes[name] = actual
            write_json(manifest_path, {"repo": spec["repo"], "revision": spec["revision"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "sha256": hashes,
                "weight_format": weight_format, "verification": "file size and SHA-256; LFS hashes matched where supplied",
                "model_loaded": False, "inference_run": False, "training_run": False})
            row = {**spec, "directory": str(directory), "weight_format": weight_format,
                   "bytes": sum((directory / name).stat().st_size for name in hashes),
                   "manifest_sha256": checksum(manifest_path), "file_count": len(hashes), "verified": True}
            rows.append(row)
            write_json(args.output_dir / "download_progress.json", {"completed": len(rows), "total": len(plans), "models": rows})
            print(f"Verified {spec['id']}: {row['bytes']/1e9:.2f} GB", flush=True)
        write_json(args.output_dir / "download_inventory.json", {"catalog_sha256": checksum(args.catalog),
            "downloader_sha256": checksum(Path(__file__)),
            "models": rows, "new_files_bytes_estimate": required, "model_loads": 0,
            "inference_runs": 0, "benchmarks": 0, "gradient_probes": 0, "training_runs": 0,
            "image_agent": "deferred", "cleanup": "separate allowlist-checked step"})
        print("Downloads and file verification complete. No models loaded or tested.", flush=True)


if __name__ == "__main__":
    main()
