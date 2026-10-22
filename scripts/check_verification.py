"""Rà A3 trên bộ đoạn cố định để tách lỗi kiểm chứng khỏi lỗi truy xuất."""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.agents.contracts import ClaimAssessment, Draft, DraftClaim, citation_of, evidence_bundle
from src.agents.evidence_review import reconcile_public
from src.agents.verification import VerificationAgent
from src.core.llm import get_llm
from src.rag.embedding_store import DenseIndex, INDEX_ROOT, ROOT
from src.rag.entailment import LocalEntailment
from src.rag.local_embeddings import file_sha256


class FixedPublicEvidence:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k):
        return [doc.model_copy(deep=True) for doc in self.documents[:top_k]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "config/verification_cases.json")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/verification_checks/day22_v1")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Chọn thư mục đầu ra mới để giữ các lượt chạy cũ")
    fixture_bytes = args.cases.read_bytes()
    fixture = json.loads(fixture_bytes.decode("utf-8"))
    if not fixture["cases"] or len({case["id"] for case in fixture["cases"]}) != len(fixture["cases"]):
        raise ValueError("Bộ ca rỗng hoặc trùng ID")
    indices = {name: DenseIndex(args.index_dir / name) for name in ["internal", "public"]}
    catalog = {name: {doc.id: doc for doc in index.documents} for name, index in indices.items()}
    # Kiểm tra ID trước khi tải model, tránh chạy một bộ ca không khớp corpus.
    for case in fixture["cases"]:
        catalog["internal"][case["internal_id"]]
        for doc_id in case["public_ids"]:
            catalog["public"][doc_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cases_used.json").write_bytes(fixture_bytes)
    llm, nli = get_llm(streaming=False, override_provider="huggingface"), LocalEntailment()
    print("Đã tải model local, bắt đầu đối chiếu các ca.", flush=True)
    records, rows = [], []
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case in fixture["cases"]:
            print("Đang chạy " + case["id"], flush=True)
            internal = evidence_bundle([catalog["internal"][case["internal_id"]]], "I", "internal")
            draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text=case["claim"], citations=[citation_of(internal[0])])])
            agent = VerificationAgent(llm, FixedPublicEvidence([catalog["public"][doc_id] for doc_id in case["public_ids"]]), entailment=nli)
            start = time.perf_counter()
            record = {"case": case}
            try:
                result = agent.run(case["question"], draft, internal)
                check = result["checks"][0]
                record["result"] = result
                actual = check["public_support"]
                row = {"case_id": case["id"], "expected_public": case["expected_public"], "actual_public": actual,
                       "claim_status": check["status"], "matched": actual == case["expected_public"], "model_error": False}
            except (ValueError, RuntimeError, IndexError, TypeError) as exc:
                record["error"] = type(exc).__name__ + ": " + str(exc)
                row = {"case_id": case["id"], "expected_public": case["expected_public"], "actual_public": "model_error",
                       "claim_status": "not_evaluated", "matched": False, "model_error": True}
            row["elapsed_ms"] = (time.perf_counter() - start) * 1000
            record.update(metrics=row, model_calls=agent.last_trace)
            records.append(record)
            rows.append(row)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(rows)}/{len(fixture['cases'])} {case['id']}: {row['actual_public']} (expected {case['expected_public']})", flush=True)
    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "cases_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "case_count": len(rows), "scope": fixture["scope"], "matched_public_labels": sum(row["matched"] for row in rows),
        "model_output_errors": sum(row["model_error"] for row in rows),
        "false_support_cases": [row["case_id"] for row in rows if row["actual_public"] == "supported" and row["expected_public"] != "supported"],
        "missed_support_cases": [row["case_id"] for row in rows if row["expected_public"] == "supported" and row["actual_public"] != "supported"],
        "median_case_ms": statistics.median(row["elapsed_ms"] for row in rows),
        "models": {"llm": llm.identity, "nli": nli.identity, "nli_threshold": nli.threshold},
        "assessment_required_fields": ClaimAssessment.model_json_schema()["required"],
        "corpora": {name: {key: index.manifest[key] for key in ["records_sha256", "vectors_sha256"]} for name, index in indices.items()}}
    summary["index_unchanged"] = all(file_sha256(args.index_dir / name / filename) == index.manifest[key]
        for name, index in indices.items() for filename, key in [("records.jsonl", "records_sha256"), ("vectors.npy", "vectors_sha256")])
    old_path = ROOT / "outputs/agent_checks/day21/pipeline_v5.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))["verification"]["checks"][0]
        assessment = SimpleNamespace(**old["llm_assessment"])
        corrected, ids = reconcile_public(assessment, old["nli_assessment"]["public"])
        summary["old_evidence_selection_replay"] = {"old_status": old["public_support"], "new_status": corrected,
            "llm_ids": assessment.public_evidence_ids, "nli_support_ids": old["nli_assessment"]["public"]["support_ids"],
            "accepted_ids": ids, "scope": "Phát lại lựa chọn nguồn đã lưu để kiểm tra lỗi ghép mã; không phải lượt model mới và không cộng vào 8 ca."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Kiểm tra A3 ngày 22/10", "", fixture["scope"], "",
        f"Khớp nhãn công khai đã chọn: {summary['matched_public_labels']}/{len(rows)}. Lỗi đầu ra model: {summary['model_output_errors']}.",
        "Lỗi model được ghi riêng, không tính là trả lời đúng hoặc kiểm chứng đúng.", "",
        "| Ca | Nhãn tham chiếu | Kết quả công khai | Kết quả nhận định |", "|---|---|---|---|"]
    lines.extend(f"| {row['case_id']} | {row['expected_public']} | {row['actual_public']} | {row['claim_status']} |" for row in rows)
    lines.extend(["", "Các ca dùng đoạn cố định, chưa đo chất lượng truy xuất hoặc chất lượng câu trả lời cuối. Xem results.jsonl để kiểm tra câu trích và phản hồi gốc."])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["matched_public_labels", "model_output_errors", "false_support_cases", "index_unchanged"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
