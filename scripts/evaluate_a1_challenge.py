"""Đánh giá adapter đã lưu trên ca phát triển mới; không đụng tập test A1."""
import argparse
import json
from pathlib import Path
import re

from scripts.predict_a1 import load_candidate
from src.data.a1_dataset import chat_messages, normalized, validate_pair
from src.training.a1_evaluation import evaluate
from src.training.a1_runtime import ROOT, file_hash, write_json


def literal_terms_preserved(row, required):
    if not required:
        return None
    if not row["context_valid"] or row["predicted"]["decision"] != "route":
        return False
    text = normalized(row["predicted"]["resolved_request"])
    return all(re.search(r"(?<!\w)" + re.escape(normalized(term)) + r"(?!\w)", text) is not None
               for term in required)


def run(run_dir, source, output):
    if output.exists():
        raise ValueError("Không ghi đè đánh giá đã có")
    manifest = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if file_hash(source) != manifest["sha256"]:
        raise ValueError("Ca phát triển thay đổi sau khi đóng băng")
    blob = json.loads(source.read_text(encoding="utf-8"))
    if blob["kind"] != "synthetic_development_challenge_not_independent_test" or blob["used_for_training"]:
        raise ValueError("Chỉ dùng bộ ca phát triển giữ riêng")
    records = blob["records"]
    if len(records) != manifest["count"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("Sai số lượng/trùng ID")
    for r in records:
        validate_pair(r["input"],r["target"])
        if r["split"] != "development_challenge" or r["messages"] != chat_messages(r["input"],r["target"]):
            raise ValueError("Sai split hoặc hội thoại")
    model, tokenizer, dtype, config, adapter_dir = load_candidate(run_dir)
    metrics, rows = evaluate(model, tokenizer, records, output, config["generation"], dtype,
                            lambda i,n,m: print(f"Challenge {i}/{n}: control={m['correct']['control_exact']}",flush=True))
    checks = [{"id":row["id"],"required":record["required_literal_terms"],
               "passed":literal_terms_preserved(row,record["required_literal_terms"])}
              for row,record in zip(rows,records) if record["required_literal_terms"]]
    summary = {"kind":blob["kind"],"source_sha256":manifest["sha256"],"adapter_dir":str(adapter_dir),
               "test_evaluated":False,"metrics":metrics,"literal_term_checks":checks,
               "literal_terms_passed":sum(c["passed"] for c in checks),"literal_terms_count":len(checks),
               "scope":"Authored development probes; literal matching is only a narrow term-preservation check, not semantic or clinical grading."}
    write_json(output/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("adapter_dir","literal_terms_passed","literal_terms_count")},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=ROOT/"data/curated/a1_supervisor_v2/development_challenge.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.run_dir,args.source,args.output_dir)


if __name__ == "__main__":
    main()
