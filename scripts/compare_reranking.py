"""Đo RRF trước/sau cross-encoder trên bộ cũ và 20 câu bổ sung."""
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
from src.rag.reranker import LocalReranker


def first_rank(documents, relevant):
    return next((i for i, doc in enumerate(documents, 1) if doc.id in relevant), None)


def write_markdown(summary, rows, directory):
    count, top_k = summary["case_count"], summary["top_k"]
    lines = ["# Kết quả thử reranker", "", summary["scope"], "",
        "| Cách tìm | Đoạn tham chiếu ở top 1 | Ở top " + str(top_k) + " | MRR theo các đoạn đã chọn |",
        "|---|---:|---:|---:|"]
    for method, value in summary["results"].items():
        lines.append(f"| {method} | {value['known_reference_hits_at_1']}/{count} | {value['known_reference_hits_at_k']}/{count} | {value['mrr_at_k_known_references']:.3f} |")
    lines.extend(["", f"Rerank trung vị: {summary['median_rerank_ms'] / 1000:.1f} giây/câu, {summary['candidate_k']} ứng viên, CPU.",
        f"Số cặp bị rút ngắn do vượt giới hạn model: {summary['truncated_pairs']}.", "",
        "## Từng câu", "", "Dấu —: không có đúng đoạn đã chọn trong top kết quả; có thể có đoạn liên quan khác chưa được gán nhãn.", "",
        "| ID câu | RRF | Sau rerank |", "|---|---:|---:|"])
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['rrf_rank'] or '—'} | {row['reranked_rank'] or '—'} |")
    lines.extend(["", "Không điều chỉnh nhãn hoặc trọng số theo kết quả lượt chạy này. Cần đọc cả đoạn trả về trước khi kết luận câu nào thật sự sai."])
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", default=[ROOT / "config/retrieval_smoke_cases.json", ROOT / "config/retrieval_edge_cases.json"])
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reranking_checks/day21_v1")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.candidate_k:
        parser.error("Cần 1 <= top-k <= candidate-k")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Chọn thư mục đầu ra mới để giữ kết quả cũ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [case for path in args.cases for case in json.loads(path.read_text(encoding="utf-8"))["cases"]]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Trùng ID câu hỏi")
    encoder, reranker = LocalE5Embeddings(), LocalReranker()
    retrievers = {warehouse: HybridRetriever.from_disk(args.index_dir / warehouse, encoder=encoder, candidate_k=args.candidate_k)
                  for warehouse in {case["warehouse"] for case in cases}}
    snapshots = {name: {filename: file_sha256(args.index_dir / name / filename) for filename in ["records.jsonl", "vectors.npy"]} for name in retrievers}
    for case in cases:
        catalog = {doc.id: doc for doc in retrievers[case["warehouse"]].documents}
        if not case["relevant_ids"]:
            raise ValueError("Thiếu tham chiếu")
        for doc_id in case["relevant_ids"]:
            doc = catalog[doc_id]
            if any(case.get(key) and doc.metadata.get(key) != case[key] for key in ["source", "language"]):
                raise ValueError("Tham chiếu nằm ngoài bộ lọc: " + case["id"])
    records, rows = [], []
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            retriever = retrievers[case["warehouse"]]
            start = time.perf_counter()
            candidates = retriever.retrieve(case["query"], top_k=args.candidate_k, source=case.get("source"), language=case.get("language"))
            retrieval_ms = (time.perf_counter() - start) * 1000
            reranked = reranker.rerank(case["query"], candidates, top_k=args.top_k)
            for doc in reranked:
                if not (doc.metadata.get("source_spans") or doc.metadata.get("locators")):
                    raise ValueError("Mất locator")
                if any(case.get(key) and doc.metadata.get(key) != case[key] for key in ["source", "language"]):
                    raise ValueError("Sai bộ lọc")
            before, after = first_rank(candidates[:args.top_k], case["relevant_ids"]), first_rank(reranked, case["relevant_ids"])
            row = {"case_id": case["id"], "query": case["query"], "warehouse": case["warehouse"],
                "source": case.get("source", "all"), "rrf_rank": before, "reranked_rank": after,
                "retrieval_ms": retrieval_ms, "rerank_ms": reranker.last_run["milliseconds"],
                "truncated_pairs": reranker.last_run["truncated_pairs"]}
            rows.append(row)
            record = {"case": case, "metrics": row, "candidate_ids": [doc.id for doc in candidates],
                "rrf": [doc.model_dump() for doc in candidates[:args.top_k]],
                "reranked": [doc.model_dump() for doc in reranked]}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(rows)}/{len(cases)} {case['id']}: RRF={before}, rerank={after}, {row['rerank_ms'] / 1000:.1f}s", flush=True)
    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases),
        "case_files": {str(path.relative_to(ROOT)): file_sha256(path) for path in args.cases},
        "candidate_k": args.candidate_k, "top_k": args.top_k, "reranker": reranker.identity,
        "model_pair_max_length": reranker.max_length, "batch_size": reranker.batch_size,
        "corpora": snapshots, "results": {}, "scope": "Các đoạn tham chiếu được chọn trước, chưa gán nhãn đầy đủ hoặc thẩm định chuyên môn; bộ mới có câu diễn đạt lại chủ đề cũ, không phải đánh giá độc lập.",
        "median_rerank_ms": statistics.median(row["rerank_ms"] for row in rows),
        "truncated_pairs": sum(row["truncated_pairs"] for row in rows)}
    for method, key in [("rrf", "rrf_rank"), ("reranked", "reranked_rank")]:
        summary["results"][method] = {"known_reference_hits_at_1": sum(row[key] == 1 for row in rows),
            "known_reference_hits_at_k": sum(row[key] is not None for row in rows),
            "mrr_at_k_known_references": statistics.mean(1 / row[key] if row[key] else 0 for row in rows)}
    summary["improved"] = [row["case_id"] for row in rows if (row["reranked_rank"] or 999) < (row["rrf_rank"] or 999)]
    summary["regressed"] = [row["case_id"] for row in rows if (row["reranked_rank"] or 999) > (row["rrf_rank"] or 999)]
    summary["index_unchanged"] = all(file_sha256(args.index_dir / warehouse / filename) == checksum
        for warehouse, files in snapshots.items() for filename, checksum in files.items())
    if not summary["index_unchanged"]:
        raise ValueError("Index thay đổi")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(summary, rows, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
