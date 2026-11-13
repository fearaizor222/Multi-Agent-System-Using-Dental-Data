from copy import deepcopy
import hashlib
import json
import threading

import pytest

from scripts import repair_a3_dataset_v2 as repair


def row(ident="one"):
    return {"id": ident, "family_id": ident, "group_id": ident, "origin": "test:" + ident, "payload": {
        "question": "Thông số của đối tượng là gì?", "claim": "Thông số là 3 mm.",
        "internal_evidence": [{"evidence_id": "I1", "content": "Thông số là 3 mm."}],
        "public_quote_options": [{"quote_id": "P1.Q1", "evidence_id": "P1", "text": "Thông số là 3 mm."}]}}


def answer(ident="one"):
    return {"id": ident, "draft_support": "supported", "public_support": "supported", "quote_ids": ["P1.Q1"],
        "internal_quotes": [{"evidence_id": "I1", "text": "Thông số là 3 mm."}],
        "draft_rationale": "Nội bộ xác nhận đủ thông số.", "public_rationale": "Nguồn công khai xác nhận thông số.",
        "reason": "Nội bộ và công khai đều xác nhận thông số 3 mm.", "quality": "clear", "issue": ""}


def test_disagreement_is_quarantined_not_forced_to_a_target(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "WORK", tmp_path)
    first, second = answer(), answer()
    second["draft_support"] = "insufficient"
    accepted, quarantine = repair.consensus([row()], {"one": first}, {"one": second})
    assert not accepted and quarantine[0]["quarantine_reason"] == "label_disagreement"
    second = answer()
    second["reason"] = "Câu tiếng Việt của lượt duyệt thứ hai."
    accepted, _ = repair.consensus([row()], {"one": first}, {"one": second})
    assert accepted[0]["gold"]["reason"] == second["reason"]


def test_invented_quote_and_insufficient_with_citations_rejected():
    result = answer()
    result["internal_quotes"][0]["text"] = "Thông số là 4 mm."
    with pytest.raises(ValueError, match="invented"):
        repair.validate_audit(row(), result)
    result = answer()
    result["public_support"] = "insufficient"
    with pytest.raises(ValueError, match="invalid public"):
        repair.validate_audit(row(), result)


def test_multilevel_numeric_tables_quarantined_even_if_both_llms_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "WORK", tmp_path)
    sample = row()
    sample["payload"]["public_quote_options"][0]["text"] = "Absent | Present | Absent | Present"
    accepted, quarantine = repair.consensus([sample], {"one": answer()}, {"one": answer()})
    assert not accepted
    assert quarantine[0]["quarantine_reason"] == "flattened_multilevel_table_needs_human_review"


def test_supported_numeric_details_require_explicit_source_values(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "WORK", tmp_path)
    sample = row()
    sample['payload']['claim'] = 'Thông số là 3 mm, kiểm tra mỗi 6 tháng.'
    accepted, quarantine = repair.consensus([sample], {'one': answer()}, {'one': answer()})
    assert not accepted
    assert quarantine[0]['quarantine_reason'] == 'numeric_detail_not_explicit_in_internal_support'


def test_changed_source_context_cannot_reuse_old_review_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, 'WORK', tmp_path)
    sample = row()
    model = repair.MODELS[0]
    repair.write(tmp_path / ('reviews_' + model + '.jsonl'), [
        {'key': repair.review_key(sample,model), 'id': sample['id'], 'result': answer()}])
    assert repair.cached_reviews([sample],model)
    sample['payload']['public_sources'] = [{'evidence_id':'P1','title':'New context','heading_path':[]}]
    assert not repair.cached_reviews([sample],model)


def test_english_reason_normalization_uses_only_approved_relations_and_ids():
    sample = row()
    sample['gold']={key:answer()[key] for key in ('draft_support','public_support','quote_ids','reason')}
    sample['gold']['reason']='Internal evidence supports the claim; public source P9.Q9 supports it.'
    fixed,rejected=repair.normalize_reason_language([sample])
    assert not rejected
    assert 'P1.Q1' in fixed[0]['gold']['reason'] and 'P9.Q9' not in fixed[0]['gold']['reason']
    assert fixed[0]['gold']['draft_support']=='supported'
    assert '3 mm' not in fixed[0]['gold']['reason']  # No new domain assertion is invented.


def test_logical_labels_are_consistency_checks_not_forced_answers(tmp_path,monkeypatch):
    monkeypatch.setattr(repair,'WORK',tmp_path)
    sample=row()
    sample['logical_negation']=True
    sample['derived_expected']={'draft_support':'contradicted','public_support':'contradicted'}
    accepted,q=repair.consensus([sample],{'one':answer()},{'one':answer()})
    assert not accepted and q[0]['quarantine_reason']=='logical_negation_consistency_check_failed'


def test_internal_stock_templates_crossing_groups_are_not_exported(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "WORK", tmp_path)
    rows = [row("one"), row("two")]
    reviews = {ident: answer(ident) for ident in ("one", "two")}
    accepted, quarantine = repair.consensus(rows, reviews, reviews)
    assert not accepted and len(quarantine) == 2


def test_bad_auditor_result_does_not_discard_other_good_results(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, "WORK", tmp_path)
    reviewer = object.__new__(repair.Reviewer)
    reviewer.clients = [None]
    reviewer.cache_lock = threading.Lock()
    bad = answer("bad")
    bad["internal_quotes"][0]["text"] = "invented"
    reviewer.request = lambda *args: {"results": [answer("one"), bad]}
    results = reviewer.audit([row(), row("bad")], repair.MODELS[0])
    assert set(results) == {"one"}
    cached = repair.read(tmp_path / ("reviews_" + repair.MODELS[0] + ".jsonl"))
    assert len(cached) == 1


def test_prepare_groups_before_distractors_and_preserves_source(tmp_path, monkeypatch):
    source, work = tmp_path / "v1", tmp_path / "work"
    source.mkdir()
    monkeypatch.setattr(repair, "SOURCE", source)
    monkeypatch.setattr(repair, "WORK", work)
    catalog, records, metadata = {}, [], []
    for provider in ("BYT", "MSD", "PMC"):
        for index in range(12):
            doc = "BYT-book" if provider == "BYT" else provider + str(index)
            headings = [f"Disease {index}", "Section"]
            title = "Document " + doc
            for suffix in ("main", "noise"):
                cid = provider + str(index) + suffix
                content = f"{title}\n{' > '.join(headings)}\n" + f"Thông tin về bệnh {index} tại nguồn {provider}. " * 5
                catalog[cid] = {"id": cid, "title": title, "content": content,
                    "metadata": {"document_id": doc, "heading_path": headings}}
            cid = provider + str(index) + "main"
            payload = row()["payload"]
            payload["question"] = "Question " + cid
            payload["claim"] = "Claim " + cid
            payload["public_quote_options"] = repair.build_options(catalog, [cid])
            gold = {k: answer()[k] for k in ("draft_support", "public_support", "quote_ids", "reason")}
            records.append({"messages": [{"role": "system", "content": repair.SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
                {"role": "assistant", "content": json.dumps(gold)}]})
            metadata.append({"public_chunk_id": cid, "public_document_id": doc, **gold})
    for split, start, end in (("train", 0, 24), ("val", 24, 36)):
        repair.write(source / (split + ".jsonl"), records[start:end])
        repair.write(source / (split + ".metadata.jsonl"), metadata[start:end])
    monkeypatch.setattr(repair, "load_catalog", lambda: catalog)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    plan = repair.prepare()
    assert len(plan) == 36
    groups, chunks = {}, {}
    for split in ("train", "val", "test"):
        rows = [r for r in plan if r["split"] == split]
        groups[split] = {g for r in rows for g in r["evidence_groups"]}
        chunks[split] = {c for r in rows for c in r["evidence_chunks"]}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not groups[a] & groups[b] and not chunks[a] & chunks[b]
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert repair.prepare() == plan
