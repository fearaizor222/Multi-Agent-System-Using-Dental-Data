"""Tìm đoạn tài liệu trong index embedding đã lưu."""
import argparse
import json
import sys
from pathlib import Path

from src.rag.embedding_store import INDEX_ROOT, DenseIndex
from src.rag.local_embeddings import LocalE5Embeddings
from src.rag.bm25_retriever import BM25Retriever
from src.rag.hybrid_retriever import HybridRetriever


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--warehouse", choices=["internal", "public", "both"], default="internal")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--source", choices=["byt", "msd", "pmc"])
    parser.add_argument("--language", choices=["vi", "en"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "hybrid-rerank"], default="hybrid")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--bm25-weight", type=float, default=0.4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source and args.warehouse == "internal":
        parser.error("--source dùng cho kho công khai")
    if args.top_k < 1 or args.candidate_k < 1 or not args.query.strip():
        parser.error("Cần câu hỏi không rỗng, top-k và candidate-k dương")
    encoder = LocalE5Embeddings() if args.method != "bm25" else None
    reranker = None
    if args.method == "hybrid-rerank":
        from src.rag.reranker import LocalReranker
        reranker = LocalReranker()
    result = {"query": args.query, "method": args.method, "warehouses": {}}
    for warehouse in (["internal", "public"] if args.warehouse == "both" else [args.warehouse]):
        directory = args.index_dir / warehouse
        options = {"top_k": args.top_k, "source": args.source if warehouse == "public" else None,
                   "language": args.language}
        if args.method == "dense":
            hits = DenseIndex(directory, encoder=encoder).search(args.query, **options)
        elif args.method == "bm25":
            hits = BM25Retriever.from_disk(directory).retrieve(args.query, **options)
        else:
            retriever = HybridRetriever.from_disk(directory, encoder=encoder, candidate_k=args.candidate_k,
                dense_weight=args.dense_weight, sparse_weight=args.bm25_weight, rrf_k=args.rrf_k)
            if reranker is not None:
                from src.rag.reranked_retriever import RerankedRetriever
                retriever = RerankedRetriever(retriever, reranker, candidate_k=max(args.candidate_k, args.top_k))
            hits = retriever.retrieve(args.query, **options)
        result["warehouses"][warehouse] = [doc.model_dump() for doc in hits]
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
