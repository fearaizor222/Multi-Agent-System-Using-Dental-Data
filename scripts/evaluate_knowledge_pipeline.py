"""Chạy liên tiếp các câu hỏi qua A2 → A3, lưu lỗi và nguồn của từng lượt."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.graph.pipeline_review import review_run
from src.graph.runtime import index_paths, load_runtime
from src.rag.embedding_store import INDEX_ROOT, ROOT, write_json
from src.rag.local_embeddings import file_sha256


def write_tables(output, rows, completed, runtime, startup_ms):
    fields = ["id", "question", "status", "elapsed_ms", "reference_in_retrieved", "draft_claims", "checked_claims",
              "not_checked_claims", "verification_completed", "citation_integrity_ok", "audited_quote_count", "model_attempts", "invalid_model_attempts", "error"]
    with (output / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row["metrics"], **row["review"]})
    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "run_completed": completed, "case_count": len(rows),
        "startup_ms": startup_ms, "statuses": dict(Counter(row["metrics"]["status"] for row in rows)),
        "verification_completed_cases": sum(row["review"]["verification_completed"] for row in rows),
        "citation_integrity_issue_cases": [row["metrics"]["id"] for row in rows if not row["review"]["citation_integrity_ok"]],
        "total_draft_claims": sum(row["review"]["draft_claims"] for row in rows),
        "total_checked_claims": sum(row["review"]["checked_claims"] for row in rows),
        "not_checked_claims": sum(row["review"]["not_checked_claims"] for row in rows),
        "claim_statuses": dict(sum((Counter(row["review"]["claim_statuses"]) for row in rows), Counter())),
        "total_audited_quotes": sum(row["review"]["audited_quote_count"] for row in rows),
        "invalid_model_attempts": sum(row["review"]["invalid_model_attempts"] for row in rows),
        "reference_in_retrieved_cases": sum(row["metrics"]["reference_in_retrieved"] is True for row in rows),
        "cases_with_retrieval_reference": sum(row["metrics"]["reference_in_retrieved"] is not None for row in rows),
        "median_case_ms": statistics.median(row["metrics"]["elapsed_ms"] for row in rows) if rows else None,
        "models": runtime.models, "indices": runtime.indices,
        "scope": "Chạy phần truy xuất, bản nháp và kiểm chứng; chưa có A1, VLM hay câu trả lời tổng hợp cuối. Không tính độ chính xác y khoa."}
    write_json(output / "summary.json", summary)
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "config/retrieval_smoke_cases.json")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--internal-index", type=Path)
    parser.add_argument("--public-index", type=Path)
    parser.add_argument("--candidate-k", type=int, default=6)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Thư mục kết quả đã có; chọn thư mục mới")
    if args.candidate_k < 3 or args.max_cases is not None and args.max_cases < 1:
        parser.error("candidate-k >= 3 và max-cases phải dương")
    paths = index_paths(args.index_dir, args.internal_index, args.public_index)
    raw = args.cases.read_bytes()
    cases = [case for case in json.loads(raw)["cases"] if case.get("warehouse", "internal") == "internal"]
    if args.max_cases:
        cases = cases[:args.max_cases]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("Bộ thử rỗng hoặc trùng ID")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) or not case["query"].strip() for case in cases):
        parser.error("ID chỉ dùng chữ, số, _ và -; câu hỏi không được rỗng")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cases_used.json").write_bytes(raw)
    code_paths = [Path(__file__), *list((ROOT / "src/agents").glob("*.py")), *list((ROOT / "src/graph").glob("*.py")),
                  ROOT / "src/rag/entailment.py", ROOT / "src/rag/reranker.py"]
    write_json(args.output_dir / "code_files.json", {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths})
    write_json(args.output_dir / "run_status.json", {"status": "initializing", "case_count": len(cases),
        "cases_sha256": hashlib.sha256(raw).hexdigest(), "candidate_k": args.candidate_k,
        "selected_case_ids": [case["id"] for case in cases]})
    print(f"Nạp model một lần cho {len(cases)} câu hỏi", flush=True)
    start = time.perf_counter()
    runtime = load_runtime(paths, args.candidate_k)
    startup_ms = (time.perf_counter() - start) * 1000
    internal_records = {doc.id: doc for doc in runtime.internal.retriever.documents}
    public_records = {doc.id: doc for doc in runtime.public.retriever.documents}
    rows = []
    case_folder = args.output_dir / "cases"
    case_folder.mkdir()
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for number, case in enumerate(cases, 1):
            start = time.perf_counter()
            state = {"question": case["query"], "trace": [], "model_calls": []}
            seen = 0
            print(f"[{number}/{len(cases)}] {case['id']}: {case['query']}", flush=True)
            write_json(args.output_dir / "run_status.json", {"status": "running", "case": case["id"],
                "completed_cases": len(rows), "selected_case_ids": [item["id"] for item in cases], "candidate_k": args.candidate_k})
            try:
                for update in runtime.pipeline.stream(state, stream_mode="values"):
                    state = dict(update)
                    write_json(case_folder / (case["id"] + ".json"), state)
                    if len(state.get("trace", [])) > seen:
                        seen = len(state["trace"])
                        step = state["trace"][-1]
                        print(f"  {step['step']}: {step['status']} ({step['milliseconds'] / 1000:.1f}s)", flush=True)
            except (ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
                state.update(status="error", error=type(exc).__name__ + ": " + str(exc))
            write_json(case_folder / (case["id"] + ".json"), state)
            review = review_run(state, internal_records, public_records)
            references = case.get("relevant_ids", [])
            retrieved = {item["chunk_id"] for item in state.get("internal_evidence", [])}
            metrics = {"id": case["id"], "question": case["query"], "status": state.get("status", "error"),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "reference_in_retrieved": bool(retrieved & set(references)) if references else None,
                "error": state.get("error", "")}
            row = {"case": case, "metrics": metrics, "review": review}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            write_tables(args.output_dir, rows, False, runtime, startup_ms)
    unchanged = all(file_sha256(Path(source["path"]) / name) == checksum
                    for source in runtime.indices.values() for name, checksum in source["sha256"].items())
    if not unchanged:
        raise ValueError("Index đã thay đổi trong lúc chạy")
    summary = write_tables(args.output_dir, rows, True, runtime, startup_ms)
    summary["source_indices_unchanged"] = True
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "run_status.json", {"status": "completed", "completed_cases": len(rows),
        "candidate_k": args.candidate_k, "selected_case_ids": [case["id"] for case in cases]})
    print(json.dumps({key: value for key, value in summary.items() if key not in {"models", "indices"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
