"""So sánh vector, BM25 và RRF trên các câu hỏi có đoạn tham chiếu đã chọn."""
import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.rag.embedding_store import INDEX_ROOT, ROOT
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256


def run(cases_path, index_dir, output_dir, top_k=5):
    fixture = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if top_k < 1 or not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cần top_k dương và danh sách câu hỏi không rỗng, không trùng ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = LocalE5Embeddings()
    retrievers, setup = {}, {}
    for warehouse in sorted({case["warehouse"] for case in cases}):
        start = time.perf_counter()
        retrievers[warehouse] = HybridRetriever.from_disk(index_dir / warehouse, encoder=encoder,
            dense_weight=0.6, sparse_weight=0.4, rrf_k=60, candidate_k=max(30, top_k))
        dense_index = retrievers[warehouse].dense_retriever._disk_index
        setup[warehouse] = {"chunks": len(dense_index.documents), "load_and_bm25_seconds": time.perf_counter() - start,
            "records_sha256": dense_index.manifest["records_sha256"],
            "vectors_sha256": dense_index.manifest["vectors_sha256"]}
    # Xác minh đoạn tham chiếu tồn tại và thuộc đúng phạm vi tìm trước khi chạy.
    for case in cases:
        catalog = {doc.id: doc for doc in retrievers[case["warehouse"]].documents}
        if not case["relevant_ids"]:
            raise ValueError("Thiếu đoạn tham chiếu: " + case["id"])
        for doc_id in case["relevant_ids"]:
            doc = catalog[doc_id]
            for field in ["source", "language"]:
                if case.get(field) and doc.metadata.get(field) != case[field]:
                    raise ValueError("Đoạn tham chiếu không thuộc bộ lọc: " + case["id"])
    results, rows = [], []
    for case in cases:
        retriever = retrievers[case["warehouse"]]
        options = {"top_k": max(30, top_k), "source": case.get("source"), "language": case.get("language")}
        start = time.perf_counter()
        dense = retriever.dense_retriever.retrieve(case["query"], **options)
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        sparse = retriever.sparse_retriever.retrieve(case["query"], **options)
        sparse_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        hybrid = retriever.fuse(dense, sparse, top_k=top_k)
        fusion_ms = (time.perf_counter() - start) * 1000
        hits = {"dense": dense[:top_k], "bm25": sparse[:top_k], "hybrid": hybrid}
        elapsed = {"dense": dense_ms, "bm25": sparse_ms, "hybrid": dense_ms + sparse_ms + fusion_ms}
        metrics, serialized = {}, {}
        for method, documents in hits.items():
            if len({doc.id for doc in documents}) != len(documents):
                raise ValueError("Kết quả trùng ID")
            for doc in documents:
                if any(case.get(f) and doc.metadata.get(f) != case[f] for f in ["source", "language"]):
                    raise ValueError("Kết quả sai bộ lọc")
                if not (doc.metadata.get("source_spans") or doc.metadata.get("locators")):
                    raise ValueError("Mất thông tin vị trí nguồn: " + doc.id)
            rank = next((i for i, doc in enumerate(documents, 1) if doc.id in case["relevant_ids"]), None)
            metrics[method] = {"first_known_reference_rank": rank, "known_reference_hit_at_1": rank == 1,
                "known_reference_hit_at_k": rank is not None, "reciprocal_rank_at_k": 1 / rank if rank else 0,
                "query_ms": elapsed[method]}
            serialized[method] = [doc.model_dump() for doc in documents]
            rows.append({"case_id": case["id"], "warehouse": case["warehouse"], "source": case.get("source", "all"),
                "query": case["query"], "method": method, **metrics[method]})
        results.append({"case": case, "metrics": metrics, "results": serialized})
        print(case["id"] + ": " + ", ".join(f"{method}={data['first_known_reference_rank']}" for method, data in metrics.items()), flush=True)
    unchanged = all(file_sha256(index_dir / name / filename) == state[key]
        for name, state in setup.items()
        for filename, key in [("records.jsonl", "records_sha256"), ("vectors.npy", "vectors_sha256")])
    if not unchanged:
        raise ValueError("Index thay đổi trong quá trình chạy")
    methods = {}
    for method in ["dense", "bm25", "hybrid"]:
        values = [result["metrics"][method] for result in results]
        methods[method] = {"known_reference_hits_at_1": sum(x["known_reference_hit_at_1"] for x in values),
            "known_reference_hits_at_k": sum(x["known_reference_hit_at_k"] for x in values),
            "mrr_at_k_known_references": statistics.mean(x["reciprocal_rank_at_k"] for x in values),
            "median_query_ms": statistics.median(x["query_ms"] for x in values)}
    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "case_version": fixture["version"],
        "cases_sha256": file_sha256(cases_path), "scope": fixture["scope"], "case_count": len(cases), "top_k": top_k,
        "config": {"candidate_k": max(30, top_k), "rrf_k": 60, "dense_weight": 0.6, "bm25_weight": 0.4,
                   "bm25_k1": 1.5, "bm25_b": 0.75, "bm25_idf": "log1p((N-df+0.5)/(df+0.5))"},
        "corpora": setup, "index_unchanged": unchanged, "filter_and_locator_checks_passed": True,
        "methods": methods, "latency_note": "Đo tuần tự trên máy local; hybrid = dense + BM25 + fusion. Chưa tính tải model/index, không phải benchmark tốc độ triển khai."}
    for filename, data in [("summary.json", summary), ("results.json", results)]:
        (output_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# So sánh truy xuất ngày 20/10/2026", "", fixture["scope"], "",
        f"{len(cases)} câu; lấy {top_k} đoạn/câu. Cùng model và bộ lọc cho ba cách tìm.", "",
        "| Cách tìm | Có đoạn tham chiếu ở top 1 | Có ở top " + str(top_k) + " | MRR theo đoạn đã chọn |",
        "|---|---:|---:|---:|"]
    for method, value in methods.items():
        lines.append(f"| {method} | {value['known_reference_hits_at_1']}/{len(cases)} | {value['known_reference_hits_at_k']}/{len(cases)} | {value['mrr_at_k_known_references']:.3f} |")
    lines.extend(["", "## Thứ hạng đoạn tham chiếu đầu tiên", "", "Dấu —: không thấy đoạn đã chọn trong top kết quả; có thể vẫn có đoạn liên quan khác chưa gán nhãn.", "",
        "| Câu hỏi | Vector | BM25 | Kết hợp |", "|---|---:|---:|---:|"])
    for result in results:
        ranks = [str(result["metrics"][method]["first_known_reference_rank"] or "—") for method in methods]
        lines.append("| " + result["case"]["id"] + " | " + " | ".join(ranks) + " |")
    lines.extend(["", "Kết quả đầy đủ, nội dung và metadata nguồn nằm trong results.json. Số liệu này dùng để rà phần truy xuất, chưa đánh giá câu trả lời LLM hoặc tính đúng về chuyên môn.", "",
        "Không chỉnh trọng số theo các câu thử. Các lượt chạy lại được lưu vào thư mục khác để giữ lần so sánh ban đầu."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "config/retrieval_smoke_cases.json")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/retrieval_checks/20-10-2026/comparison_v1")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        parser.error("Thư mục đã có kết quả; chọn --output-dir khác để giữ lần chạy trước")
    run(args.cases, args.index_dir, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
