"""Read-only checks for schema, provenance and train/val/test separation in A3 v2."""
from collections import Counter, defaultdict
import hashlib
import json

from scripts.repair_a3_dataset_v2 import (OUT, WORK, SOURCE, FILES, MODELS, build_options,
    load_catalog, norm, parse_sample, read, dump, validate_audit, source_context)


def check():
    report = {"splits": {}, "overlap": {}, "errors": []}
    catalog = load_catalog()
    manifest = json.loads((WORK / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in FILES}
    assert all(hashlib.sha256((WORK / "source_backup" / name).read_bytes()).hexdigest() == digest
               for name, digest in manifest["sha256"].items())
    approved = {row["id"]: row for row in read(WORK / "accepted.jsonl")}
    all_planned = {row["id"]: row for row in read(WORK / "plan.jsonl") + read(WORK / "variant_plan.jsonl")}
    buckets, inputs = {}, set()
    for split in ("train", "val", "test"):
        records, metas = read(OUT / (split + ".jsonl")), read(OUT / (split + ".metadata.jsonl"))
        assert records and len(records) == len(metas)
        bucket = {key: set() for key in ("chunks", "groups", "families", "claims", "internal")}
        counts = Counter()
        for record, meta in zip(records, metas):
            payload, gold = parse_sample(record)
            family, origin = meta["family_id"], meta["origin"]
            identifier = meta["repair_id"]
            approved_row = approved[identifier]
            assert gold == approved_row["gold"]
            assert payload == approved_row["payload"]
            for model, key in zip(MODELS, ("review_a", "review_b")):
                review = approved_row[key]
                validate_audit(approved_row, review)
                assert review["quality"] == "clear"
                assert all(review[field] == gold[field] for field in ("draft_support", "public_support"))
            for key in ("draft_support", "public_support", "quote_ids", "reason"):
                assert meta[key] == gold[key]
            assert meta["question"] == payload["question"] and meta["claim"] == payload["claim"]
            assert meta["split"] == approved_row["split"] == split
            assert meta["evidence_chunks"] == approved_row["evidence_chunks"]
            assert build_options(catalog, meta["evidence_chunks"]) == payload["public_quote_options"]
            assert source_context(catalog, meta["evidence_chunks"]) == payload["public_sources"]
            if meta['internal_evidence_origin']=='assembled_reviewed_contexts_from_same_split':
                donors=meta['internal_context_donors']
                assert len(donors)==len(payload['internal_evidence'])
                for donor,item in zip(donors,payload['internal_evidence']):
                    assert all_planned[donor]['split']==split
                    assert all_planned[donor]['payload']['internal_evidence'][0]['content']==item['content']
            if meta['internal_evidence_origin']=='copied_public_source_context_for_identity_control':
                assert gold['draft_support']==gold['public_support']
            user_input = json.dumps(record["messages"][:2], sort_keys=True, ensure_ascii=False)
            assert user_input not in inputs, "Duplicate full input"
            inputs.add(user_input)
            bucket["chunks"].update(meta["evidence_chunks"])
            bucket["groups"].update(meta["evidence_groups"])
            bucket["families"].add(family)
            bucket["claims"].add(norm(payload["claim"]))
            bucket["internal"].update(norm(e["content"]) for e in payload["internal_evidence"])
            counts[gold["draft_support"] + "/" + gold["public_support"]] += 1
        buckets[split] = bucket
        assert len(counts)==9, f"Missing joint-label case in {split}"
        assert min(counts.values()) >= (20 if split=='train' else 5), f"Rare joint-label case too sparse in {split}"
        report["splits"][split] = {"count": len(records), "joint": dict(counts), "groups": len(bucket["groups"])}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps = {key: len(buckets[a][key] & buckets[b][key]) for key in buckets[a]}
        report["overlap"][a + "/" + b] = overlaps
        assert not any(overlaps.values()), f"Cross-split exposure: {a}/{b}: {overlaps}"
    report["status"] = "passed_structural_provenance_and_split_checks"
    report["scope"] = "Includes AI-review agreement and source-substring checks; not proof every label is clinically correct."
    dump(OUT / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    check()
