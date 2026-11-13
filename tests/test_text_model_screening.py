"""Screening must not mistake formatting/control success for factual accuracy."""
import json

import pytest

from scripts.build_text_screening import build_cases
from scripts.report_text_screening import remove_outer_fence
from src.training.a1_runtime import file_hash
from src.training.model_candidates import read_candidates, verify_candidate
from src.training.text_screening import grade, messages_for, summarize


def case(key):
    return next(item for item in build_cases([]) if item["id"] == key)


def test_suite_prompts_do_not_contain_targets_and_all_non_a1_require_manual_review():
    selected = case("a3_missing")
    messages = messages_for(selected)
    assert len(messages) == 2
    assert '"expected"' not in messages[1]["content"]
    assert grade(selected, "no json")["requires_human_content_review"]
    assert summarize([])["winner"] is None


def test_a2_valid_json_with_nonexistent_source_fails():
    selected = case("a2_definition")
    raw = json.dumps({"status": "draft", "claims": [{"text": "Một nhận định.", "evidence_ids": ["I99"]}]})
    result = grade(selected, raw)
    assert result["schema_valid"] and not result["task_control_pass"]


def test_a2_control_success_does_not_claim_content_quality():
    selected = case("a2_definition")
    raw = json.dumps({"status": "draft", "claims": [{"text": "Nội dung chưa được người đánh giá kiểm tra.", "evidence_ids": ["I1"]}]})
    result = grade(selected, raw)
    assert result["task_control_pass"] and result["requires_human_content_review"]


@pytest.mark.parametrize("quote,key", [("Câu tự bịa.", "P1"), ("Máng nhai là một khí cụ bằng nhựa.", "P99")])
def test_a3_rejects_invented_quote_or_source(quote, key):
    raw = json.dumps({"draft_support": "supported", "public_support": "supported",
        "public_evidence_ids": [key], "public_quotes": [{"evidence_id": key, "text": quote}], "reason": "Đối chiếu."})
    assert not grade(case("a3_support"), raw)["task_control_pass"]


def test_a3_literal_quote_with_wrong_relation_still_fails():
    selected = case("a3_contradiction")
    raw = json.dumps({"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["P1"],
        "public_quotes": [{"evidence_id": "P1", "text": selected["input"]["public_evidence"][0]["content"]}], "reason": "Sai nhãn."})
    result = grade(selected, raw)
    assert result["literal_quotes_valid"] and not result["task_control_pass"]


def test_a4_cannot_select_unsupported_claim():
    raw = json.dumps({"status": "partial", "message": "Câu trả lời.", "used_claim_ids": ["c1", "c2"]})
    assert not grade(case("a4_partial"), raw)["task_control_pass"]


def test_duplicate_json_keys_are_not_repaired():
    assert not grade(case("a4_missing"), '{"status":"answered","status":"insufficient_evidence","message":"Thiếu.","used_claim_ids":[]}')["json_valid"]


def test_fence_diagnostic_does_not_repair_wrong_json_or_unwrap_objects():
    duplicate = '{"status":"answered","status":"insufficient_evidence","message":"Thiếu.","used_claim_ids":[]}'
    assert not grade(case("a4_missing"), remove_outer_fence("```json\n" + duplicate + "\n```"))["json_valid"]
    wrapped = '{"answer":{"status":"insufficient_evidence","message":"Thiếu.","used_claim_ids":[]}}'
    assert not grade(case("a4_missing"), remove_outer_fence("```json\n" + wrapped + "\n```"))["schema_valid"]


@pytest.mark.parametrize("raw", ['Text before\n```json\n{}\n```', '```json\n{}\n```\nText after'])
def test_fence_diagnostic_rejects_surrounding_commentary(raw):
    assert remove_outer_fence(raw) == raw


def test_fence_diagnostic_only_changes_secondary_score():
    selected = case("a4_missing")
    raw = '```json\n{"status":"insufficient_evidence","message":"Chưa đủ bằng chứng.","used_claim_ids":[]}\n```'
    assert not grade(selected, raw)["schema_valid"]
    assert grade(selected, remove_outer_fence(raw))["task_control_pass"]


def test_empty_weight_manifest_and_path_traversal_are_rejected(tmp_path):
    spec = {"repo": "a/b", "revision": "a"*40}
    manifest_path = tmp_path / "download_manifest.json"
    manifest_path.write_text(json.dumps({**spec, "sha256": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no weights"):
        verify_candidate(spec, tmp_path)
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"weights")
    manifest_path.write_text(json.dumps({**spec, "sha256": {"../outside.safetensors": file_hash(outside)}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid/missing"):
        verify_candidate(spec, tmp_path)


def test_candidates_are_pinned_and_independent_of_production():
    specs = read_candidates()
    assert len(specs) == 4
    assert all(len(item["revision"]) == 40 for item in specs)
    assert {item["id"] for item in specs} == {"qwen25_05b", "qwen3_06b", "qwen25_15b", "qwen3_17b"}


def test_multifamily_catalog_is_not_just_qwen_sizes():
    from src.training.a1_runtime import ROOT
    path = ROOT / "config/text_model_candidates_multifamily.json"
    specs = read_candidates(path)
    assert len({item["family"] for item in specs}) == len(specs) == 4
    assert len({item["repo"].split("/")[0] for item in specs}) == 4
    assert len([item for item in specs if item["family"] == "Qwen"]) == 1
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert all(item["weight_access"] == "unavailable_with_current_credentials" for item in metadata["access_checks"])
    assert not {item["repo"] for item in metadata["access_checks"]} & {item["repo"] for item in specs}
