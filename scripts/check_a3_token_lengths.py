"""Tokenizer-only A3 v2 length check. No weights, inference, training or API calls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from scripts.train_a3 import MODELS, ROOT, MAX_LENGTH, configure_tokenizer, encode_sample


def check():
    from transformers import AutoTokenizer
    data = ROOT / "data/processed/a3_verification_v2"
    work = ROOT / "outputs/a3_data_repair_v2"
    files=[data/(split+suffix) for split in ('train','val','test') for suffix in ('.jsonl','.metadata.jsonl')]
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    rows = []
    for split in ("train", "val", "test"):
        records = [json.loads(line) for line in (data / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]
        metas = [json.loads(line) for line in (data / (split + ".metadata.jsonl")).read_text(encoding="utf-8").splitlines()]
        assert len(records) == len(metas)
        rows += [(split, record, meta) for record, meta in zip(records, metas)]
    report, failures = {}, defaultdict(list)
    for model_id, spec in MODELS.items():
        path = ROOT / "data/models" / spec["repo"].replace("/", "--") / spec["revision"]
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False), model_id)
        sizes = defaultdict(list)
        for split, record, meta in rows:
            try:
                encoded = encode_sample(tokenizer, record, max_length=10**6)
                size = len(encoded["input_ids"])
                sizes[split].append(size)
                if size > MAX_LENGTH:
                    failures[meta["repair_id"]].append({"model": model_id, "tokens": size})
            except ValueError as exc:
                failures[meta["repair_id"]].append({"model": model_id, "error": str(exc)})
        report[model_id] = {split: {"count": len(values), "min": min(values), "max": max(values),
            "median": statistics.median(values), "over_limit": sum(v > MAX_LENGTH for v in values)}
            for split, values in sizes.items()}
        print(model_id, json.dumps(report[model_id]), flush=True)
    excluded_path = work / "length_exclusions.jsonl"
    old = [json.loads(line) for line in excluded_path.read_text(encoding="utf-8").splitlines()] if excluded_path.exists() else []
    by_id = {row["id"]: row for row in old}
    by_id.update({key: {"id": key, "failures": issues} for key, issues in failures.items()})
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in by_id.values()), encoding="utf-8")
    assert hashes=={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}, 'Dataset changed during token check'
    (data / "token_lengths.json").write_text(json.dumps({"models": report, "max_length": MAX_LENGTH, "source_sha256":hashes,
        "current_rows_over_limit_or_prefix_error": len(failures), "weights_loaded": False}, indent=2), encoding="utf-8")
    print("Current rows needing exclusion:", len(failures), flush=True)
    return len(failures)


if __name__ == "__main__":
    check()
