"""Gom các file chunk đã xử lý và kiểm tra với PDF gốc trước khi tạo index mới."""
import argparse
import json
import sys
from pathlib import Path

from src.data.internal_corpus import prepare_internal_corpus
from src.rag.embedding_store import ROOT


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_internal_corpus(args.chunks, args.pdf_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ["count", "pdf_count", "counts_by_source", "duplicate_ids_skipped", "scope"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
