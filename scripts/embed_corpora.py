"""Tạo embedding local cho kho nội bộ và kho công khai, có checkpoint."""
import argparse
import json
import os
import sys
from pathlib import Path

from src.rag.embedding_store import INDEX_ROOT, build_index, load_corpus
from src.rag.local_embeddings import LocalE5Embeddings


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", choices=["internal", "public", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--reuse-from", type=Path, help="Thư mục gốc index cũ; dùng lại vector có cùng model và văn bản")
    args = parser.parse_args()
    if args.reuse_from and not args.reuse_from.is_dir():
        parser.error("Thư mục --reuse-from không tồn tại")
    encoder = LocalE5Embeddings(batch_size=args.batch_size, threads=args.threads)
    print(f"Model: {encoder.identity['model']}, {encoder.dimension} chiều, CPU", flush=True)
    result = {}
    for warehouse in (["internal", "public"] if args.warehouse == "both" else [args.warehouse]):
        documents, sources = load_corpus(warehouse)
        print(f"Kho {warehouse}: kiểm tra {len(documents)} chunk trước khi nhúng", flush=True)
        state = build_index(documents, sources, args.output_dir / warehouse, encoder,
                            reuse_from=args.reuse_from / warehouse if args.reuse_from else None)
        result[warehouse] = {key: state[key] for key in ["status", "count", "elapsed_seconds", "counts_by_language"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
