"""Rà nguồn và thống kê lô câu hỏi, không dùng model thật trong unit test."""
import json
from copy import deepcopy

from scripts.evaluate_knowledge_pipeline import write_tables
from scripts.check_verification import public_result_row
from src.graph.pipeline_review import review_run
from src.graph.runtime import index_paths
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from tests.test_knowledge_pipeline import FixedRetriever, PUBLIC_QUOTE, ScriptedLLM, doc


def saved_state():
    model = ScriptedLLM({"status": "draft", "claims": [{"text": "Khí cụ bằng nhựa.", "evidence_ids": ["I1"]}]},
        {"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE]})
    graph = create_knowledge_pipeline(FixedRetriever([doc()]), FixedRetriever([doc("public-1", "public")]), model)
    return graph.invoke({"question": "Trình bày về máng nhai"})


def review(state):
    return review_run(state, {"internal-1": doc()}, {"public-1": doc("public-1", "public")})


def test_review_checks_actual_source_records_and_quotes():
    result = review(saved_state())
    assert result["verification_completed"]
    assert result["internal_citation_count"] == 1
    assert result["audited_quote_count"] == 1
    assert result["citation_integrity_ok"]


def test_review_detects_valid_id_with_changed_source_text():
    state = saved_state()
    state["internal_evidence"][0]["content"] += " Thêm nội dung không có trong nguồn."
    result = review(state)
    assert not result["citation_integrity_ok"]
    assert any(issue.startswith("internal_source_mismatch") for issue in result["citation_integrity_issues"])


def test_review_detects_changed_page_metadata_even_with_same_text():
    state = saved_state()
    state["internal_evidence"][0]["metadata"]["source_spans"] = [{"pdf_page": 999}]
    assert not review(state)["citation_integrity_ok"]


def test_review_detects_wrong_quote_offset_and_forged_public_citation():
    state = saved_state()
    state["verification"]["checks"][0]["public_quote_checks"][0]["char_start"] += 1
    state["verification"]["checks"][0]["public_citations"][0]["chunk_id"] = "not-in-index"
    result = review(state)
    assert any(issue.startswith("invalid_public_quote") for issue in result["citation_integrity_issues"])
    assert any(issue.startswith("invalid_public_citation") for issue in result["citation_integrity_issues"])


def test_failure_before_verification_is_not_counted_as_completed():
    model = ScriptedLLM("bad json", "bad again")
    graph = create_knowledge_pipeline(FixedRetriever([doc()]), FixedRetriever([]), model)
    state = graph.invoke({"question": "Câu hỏi"})
    result = review(state)
    assert result["status"] == "error"
    assert not result["verification_completed"]
    assert result["checked_claims"] == 0
    assert result["invalid_model_attempts"] == 2


def test_new_question_does_not_inherit_previous_draft_or_verification():
    model = ScriptedLLM({"status": "draft", "claims": [{"text": "Khí cụ bằng nhựa.", "evidence_ids": ["I1"]}]},
        {"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE]},
        {"status": "insufficient_evidence", "claims": []},
        {"status": "insufficient_evidence", "claims": []})
    graph = create_knowledge_pipeline(FixedRetriever([doc()]), FixedRetriever([doc("public-1", "public")]), model)
    first = graph.invoke({"question": "Câu một", "trace": [], "model_calls": []})
    second = graph.invoke({"question": "Câu hai", "trace": [], "model_calls": []})
    assert first["verification"]["checks"]
    assert second["draft"]["claims"] == []
    assert not second.get("verification")
    assert len(second["trace"]) == 2
    assert second["repair_attempted"] is False
    assert second["revision_history"] == []


def test_index_selection_can_mix_new_internal_with_current_public(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new-internal"
    for folder in (old / "internal", old / "public", new):
        folder.mkdir(parents=True)
        for name in ("manifest.json", "records.jsonl", "vectors.npy"):
            (folder / name).touch()
    paths = index_paths(old, internal_index=new)
    assert paths["internal"] == new.resolve()
    assert paths["public"] == (old / "public").resolve()


def test_batch_summary_separates_runtime_failure_from_verdict(tmp_path):
    good = review(saved_state())
    bad = deepcopy(good)
    bad.update(status="error", verification_completed=False, checked_claims=0, claim_statuses={}, invalid_model_attempts=2)
    class Runtime:
        models = {"scope": "scripted test"}
        indices = {}
    rows = [{"metrics": {"id": name, "question": name, "status": assessment["status"], "elapsed_ms": 10,
                          "reference_in_retrieved": True, "error": ""}, "review": assessment}
            for name, assessment in [("good", good), ("bad", bad)]]
    summary = write_tables(tmp_path, rows, True, Runtime(), 5)
    assert summary["statuses"] == {"supported": 1, "error": 1}
    assert summary["verification_completed_cases"] == 1
    assert summary["total_checked_claims"] == 1
    assert summary["invalid_model_attempts"] == 2
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["run_completed"]


def test_unchecked_claim_is_never_counted_as_matching_medical_label():
    row = public_result_row({"id": "c1", "expected_public": "insufficient"},
        {"status": "not_checked", "public_support": None, "error_type": "ClaimTooLong"})
    assert not row["matched"]
    assert row["model_error"]
    assert row["actual_public"] == "not_checked"
