"""Kiểm tra, chia nhóm và xuất dữ liệu hội thoại A1, không gọi LLM/mạng."""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.data.a1_dataset import (A1Input, A1Decision, POLICY_VERSION, SYSTEM_PROMPT,
                                chat_messages, normalized, validate_pair, validate_records)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/curated/a1_supervisor_v1/scenarios.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_groups(groups, seed):
    """Nối cả các nhóm có cùng message trước khi stratify theo category."""
    parent = {group["id"]: group["id"] for group in groups}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    messages = []
    for group in groups:
        for case in group["cases"]:
            key = normalized(case["input"]["message"])
            if key in owners:
                parent[find(group["id"])] = find(owners[key])
            tokens = set(re.findall(r"\w+", key))
            for previous_id, previous_tokens in messages:
                if min(len(tokens), len(previous_tokens)) >= 5 and len(tokens & previous_tokens) / len(tokens | previous_tokens) >= .85:
                    parent[find(group["id"])] = find(previous_id)
            messages.append((group["id"], tokens))
            owners[key] = group["id"]
    components = defaultdict(list)
    for group in groups:
        components[find(group["id"])].append(group)
    strata = defaultdict(list)
    priorities = ["capability", "context", "image", "adversarial", "mixed", "handoff",
                  "clarification", "out_of_scope", "social", "service", "knowledge"]
    for component in components.values():
        categories = {group["category"] for group in component}
        category = next((category for category in priorities if category in categories), sorted(categories)[0])
        strata[category].append(component)
    assignments = {}
    rng = random.Random(seed)
    for category, components in sorted(strata.items()):
        components.sort(key=lambda component: min(group["id"] for group in component))
        rng.shuffle(components)
        holdout = max(1, round(len(components) * .15)) if len(components) >= 5 else 0
        for index, component in enumerate(components):
            split = "test" if index < holdout else "validation" if index < 2 * holdout else "train"
            for group in component:
                assignments[group["id"]] = split
    return assignments


def build_records(source, seed=20261027):
    if source["policy_version"] != POLICY_VERSION:
        raise ValueError("Sai policy_version")
    groups = source["groups"]
    if len({group["id"] for group in groups}) != len(groups):
        raise ValueError("Trùng group ID")
    assignments = source.get("split_assignments")
    if assignments is None:
        assignments = split_groups(groups, seed)
    elif (set(assignments) != {group["id"] for group in groups}
          or any(split not in {"train", "validation", "test"} for split in assignments.values())):
        raise ValueError("split_assignments phải bao phủ đúng mọi nhóm với split hợp lệ")
    rows = []
    for group in groups:
        for index, case in enumerate(group["cases"], 1):
            context, target = validate_pair(case["input"], case["target"])
            rows.append({"id": f"{group['id']}_{index:02}", "group_id": group["id"],
                         "category": group["category"], "split": assignments[group["id"]],
                         "provenance": "synthetic_authored", "review_status": "unreviewed",
                         "policy_version": POLICY_VERSION, "tags": case.get("tags", []),
                         "input": context, "target": target, "messages": chat_messages(context, target)})
    validate_records(rows)
    return rows


def export(source_path, output, seed=20261027, tokenizer_path=None):
    if output.exists():
        raise ValueError("Thư mục đầu ra đã tồn tại; chọn phiên bản mới để giữ split cũ")
    source = read_json(source_path)
    records = build_records(source, seed)
    summary = validate_records(records)
    token_lengths = {}
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        for row in records:
            token_lengths[row["id"]] = len(tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                                        enable_thinking=False))
        lengths = sorted(token_lengths.values())
        summary["tokens"] = {"min": min(lengths), "max": max(lengths), "median": lengths[len(lengths)//2],
                              "over_2048": sum(length > 2048 for length in lengths),
                              "tokenizer": str(tokenizer_path), "enable_thinking": False}
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        selected = [row for row in records if row["split"] == split]
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
        with (output / f"{split}.metadata.jsonl").open("w", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["id", "group_id", "split", "category", "review_status", "message", "history", "attachments",
              "capabilities", "expected_decision", "expected_targets", "expected_service_topics", "expected_json",
              "tags", "tokens", "reviewer", "review_notes"]
    with (output / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"id": row["id"], "group_id": row["group_id"], "split": row["split"],
                "category": row["category"], "review_status": row["review_status"], "message": row["input"]["message"],
                "history": json.dumps(row["input"]["history"], ensure_ascii=False),
                "attachments": json.dumps(row["input"]["attachments"], ensure_ascii=False),
                "capabilities": json.dumps(row["input"]["capabilities"], ensure_ascii=False),
                "expected_decision": row["target"]["decision"], "expected_targets": ",".join(row["target"]["targets"]),
                "expected_service_topics": ",".join(row["target"]["service_topics"]),
                "expected_json": json.dumps(row["target"], ensure_ascii=False), "tags": ",".join(row["tags"]),
                "tokens": token_lengths.get(row["id"], ""), "reviewer": "", "review_notes": ""})
    write_json(output / "input.schema.json", A1Input.model_json_schema())
    write_json(output / "output.schema.json", A1Decision.model_json_schema())
    (output / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    summary.update({"policy_version": POLICY_VERSION, "split_seed": seed,
        "source": str(source_path.resolve()), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "built_at": datetime.now(timezone.utc).isoformat(), "provenance": "synthetic_authored",
        "human_reviewed_records": 0, "training_executed": False,
        "scope": "A1 ingress routing only; not a clinical benchmark; not external generalization evidence",
        "near_duplicate_audit": "See near_duplicates.json; lexical comparison is not a semantic leakage guarantee."})
    # Báo các câu gần nhau giữa split để rà; history/capability có thể khác một cách có chủ ý.
    cross_near = []
    fingerprints = [(row, set(re.findall(r"\w+", normalized(row["input"]["message"])))) for row in records]
    for i, (left, lt) in enumerate(fingerprints):
        for right, rt in fingerprints[i+1:]:
            if left["split"] == right["split"] or min(len(lt), len(rt)) < 5:
                continue
            similarity = len(lt & rt) / len(lt | rt)
            if similarity >= .85:
                cross_near.append({"left": left["id"], "right": right["id"], "jaccard": round(similarity, 4)})
    write_json(output / "near_duplicates.json", cross_near)
    summary["cross_split_near_message_pairs"] = len(cross_near)
    summary["files_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(output.iterdir()) if path.is_file()}
    write_json(output / "manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output_dir, args.seed, args.tokenizer_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
