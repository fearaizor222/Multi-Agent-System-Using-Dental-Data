"""Tạo embedding local cho kho nội bộ và kho công khai, có checkpoint."""
import argparse
import json
import os
import sys
from pathlib import Path

from src.data.internal_corpus import load_internal_corpus
from src.data.imported_internal_chunks import load_imported_internal_chunks
from src.rag.embedding_store import INDEX_ROOT, build_index, load_corpus
from src.rag.local_embeddings import LocalE5Embeddings, passage_body


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
    parser.add_argument("--internal-corpus", type=Path, help="Kho chunks.jsonl + manifest.json do prepare_internal_corpus tạo")
    parser.add_argument("--internal-chunks-import", type=Path, help="File public-chunk-v1 được chỉ định làm kho nội bộ")
    args = parser.parse_args()
    if args.internal_chunks_import and (args.internal_corpus or args.warehouse != "internal"):
        parser.error("--internal-chunks-import cần --warehouse internal và không đi cùng --internal-corpus")
    if args.internal_corpus and args.warehouse == "public":
        parser.error("--internal-corpus chỉ dùng khi có kho internal")
    if args.reuse_from and not args.reuse_from.is_dir():
        parser.error("Thư mục --reuse-from không tồn tại")
    encoder = LocalE5Embeddings(batch_size=args.batch_size, threads=args.threads)
    print(f"Model: {encoder.identity['model']}, {encoder.dimension} chiều, CPU", flush=True)
    result = {}
    for warehouse in (["internal", "public"] if args.warehouse == "both" else [args.warehouse]):
        documents, sources = (load_imported_internal_chunks(args.internal_chunks_import)
                              if warehouse == "internal" and args.internal_chunks_import else
                              load_internal_corpus(args.internal_corpus)
                              if warehouse == "internal" and args.internal_corpus else load_corpus(warehouse))
        print(f"Kho {warehouse}: kiểm tra {len(documents)} chunk trước khi nhúng", flush=True)
        limit = 380
        if args.internal_chunks_import:
            # Imported counts are advisory; keep them and count the exact embedding input again.
            # Preserve all content/IDs; the E5 encoder supports 512 tokens without truncation.
            limit = 512
            lengths = encoder.token_lengths(["passage: " + passage_body(doc) for doc in documents])
            for doc, actual in zip(documents, lengths):
                doc.metadata["imported_token_count"] = doc.metadata["token_count"]
                doc.metadata["token_count"] = actual
                doc.metadata["token_count_recalculated"] = True
        state = build_index(documents, sources, args.output_dir / warehouse, encoder,
                            reuse_from=args.reuse_from / warehouse if args.reuse_from else None,
                            max_passage_tokens=limit)
        result[warehouse] = {key: state[key] for key in ["status", "count", "elapsed_seconds", "counts_by_language"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
