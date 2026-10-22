"""Tải các model đã chốt revision, chỉ lấy cấu hình/tokenizer/safetensors."""
import argparse
import contextlib
import fnmatch
import json
import os
import socket
import sys
import urllib.request
from datetime import datetime, timezone

# Dùng downloader Python để tùy chọn DNS chỉ tác động trong tiến trình này.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, snapshot_download
from src.core.model_registry import MODEL_SPECS, model_directory
from src.rag.local_embeddings import file_sha256


@contextlib.contextmanager
def huggingface_dns(enabled):
    original = socket.getaddrinfo
    cache = {}

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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "both", *MODEL_SPECS], default="all")
    parser.add_argument("--dns-fallback", action="store_true", help="Dùng DNS qua HTTPS cho host Hugging Face khi DNS máy lỗi; vẫn kiểm tra TLS")
    args = parser.parse_args()
    with huggingface_dns(args.dns_fallback):
        selected = MODEL_SPECS if args.model == "all" else (["reranker", "draft_llm"] if args.model == "both" else [args.model])
        for name in selected:
            spec = MODEL_SPECS[name]
            directory = model_directory(name)
            patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]
            info = HfApi(token=False).model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
            print(f"Đang tải {spec['repo']} revision {info.sha}", flush=True)
            snapshot_download(spec["repo"], revision=spec["revision"], local_dir=directory,
                              allow_patterns=patterns, token=False, max_workers=2)
            checksums = {}
            for item in info.siblings:
                if not any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns):
                    continue
                checksum = file_sha256(directory / item.rfilename)
                if item.lfs is not None and checksum != item.lfs.sha256:
                    raise ValueError("Sai checksum model tải về: " + item.rfilename)
                checksums[item.rfilename] = checksum
            manifest = {**spec, "downloaded_at": datetime.now(timezone.utc).isoformat(), "sha256": checksums}
            (directory / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Đã tải và kiểm tra {name}: {directory}", flush=True)


if __name__ == "__main__":
    main()
