"""Kiểm tra A4 không phát tán bản nháp thiếu bằng chứng; dữ liệu đều dựng sẵn."""
from copy import deepcopy

import pytest

from src.agents.contracts import Evidence,Draft,DraftClaim,citation_of
from src.agents.synthesis import SynthesisAgent,FinalAnswer
from src.rag.entailment import aggregate_predictions
from scripts.run_dental_assistant import render_result,run_turn


def nli(evidence_id,support=True):
    scores={"entailment":.95 if support else .02,"neutral":.04 if support else .97,"contradiction":.01}
    return aggregate_predictions([{"evidence_id":evidence_id,"truncated":False,"scores":scores,
                                  "premise_token_start":0,"premise_token_end":10}])


def supported_state(text="Nội dung A có trong tài liệu.",suffix="a",internal_id="I1"):
    internal=Evidence(evidence_id=internal_id,warehouse="internal",source_id="book-"+suffix,chunk_id="internal-"+suffix,
        title="Tài liệu nội bộ "+suffix,content=text,metadata={"source_spans":[{"pdf_page":10}],"warehouse":"internal"})
    public=Evidence(evidence_id="P1",warehouse="public",source_id="public-"+suffix,chunk_id="chunk-"+suffix,
        title="Tài liệu công khai "+suffix,content="Tài liệu công khai "+suffix+"\nMục\n"+text,
        metadata={"heading_path":["Mục"],"source_url":"https://example.org/source-"+suffix,"warehouse":"public"})
    draft=Draft(status="draft",claims=[DraftClaim(claim_id="c1",text=text,citations=[citation_of(internal)])])
    check={"claim_id":"c1","status":"supported","draft_support":"supported","public_support":"supported",
           "public_evidence_ids":["P1"],"public_quotes":[{"evidence_id":"P1","text":text}],
           "public_evidence":[public.model_dump()],"public_citations":[citation_of(public).model_dump()],
           "nli_assessment":{"internal":nli(internal_id),"public":nli("P1")},"reason":"TECHNICAL_REASON"}
    return {"status":"supported","draft":draft.model_dump(),"internal_evidence":[internal.model_dump()],
            "verification":{"checks":[check]},"review":{"citation_integrity_ok":True,"citation_integrity_issues":[],"verification_completed":True}}


def test_supported_claim_is_preserved_verbatim_with_checked_sources():
    state=supported_state()
    result=SynthesisAgent().run(state)
    FinalAnswer.model_validate(result)
    assert result["status"]=="answered" and result["model_used"] is False
    assert result["claims"][0]["text"]==state["draft"]["claims"][0]["text"]
    assert result["claims"][0]["reference_ids"]==["1","2"]
    assert result["sources"][0]["pages"]==[10]
    assert result["sources"][1]["url"]=="https://example.org/source-a"
    assert "TECHNICAL_REASON" not in result["message"] and "nli_assessment" not in result["message"]


@pytest.mark.parametrize("status",["insufficient","not_checked","invalid_citation"])
def test_unapproved_claim_is_not_printed_or_put_in_history(status):
    state=supported_state("SECRET_UNAPPROVED_TEXT")
    state["verification"]["checks"][0]["status"]=status
    result=SynthesisAgent().run(state)
    assert result["status"]=="insufficient_evidence"
    assert result["claims"]==result["sources"]==[]
    assert "SECRET_UNAPPROVED_TEXT" not in result["message"]


def test_partial_answer_only_contains_supported_claims():
    state=supported_state()
    claim=deepcopy(state["draft"]["claims"][0]);claim.update(claim_id="c2",text="UNAPPROVED_SECOND_CLAIM")
    state["draft"]["claims"].append(claim)
    state["verification"]["checks"].append({"claim_id":"c2","status":"insufficient"})
    result=SynthesisAgent().run(state)
    assert result["status"]=="partial" and len(result["claims"])==1
    assert "UNAPPROVED_SECOND_CLAIM" not in result["message"]
    assert result["omitted_claims"]==[{"claim_id":"c2","reason":"insufficient"}]


def test_contradiction_withholds_even_other_supported_claims():
    state=supported_state()
    other=deepcopy(state["draft"]["claims"][0]);other.update(claim_id="c2",text="CONTRADICTED_TEXT")
    state["draft"]["claims"].append(other)
    state["verification"]["checks"].append({"claim_id":"c2","status":"contradicted"})
    result=SynthesisAgent().run(state)
    assert result["status"]=="conflicting_evidence" and result["claims"]==[]
    assert all(c["text"] not in result["message"] for c in state["draft"]["claims"])


def test_conflicting_nli_evidence_is_not_overridden_by_supported_flag():
    state=supported_state()
    state["verification"]["checks"][0]["nli_assessment"]["public"]["conflicting_evidence"]=True
    assert SynthesisAgent().run(state)["status"]=="conflicting_evidence"


@pytest.mark.parametrize("change",["review","missing_review","citation","quote","nli_flag","nli_missing","nli_nan","duplicate_check"])
def test_corrupted_support_metadata_fails_closed(change):
    state=supported_state();check=state["verification"]["checks"][0]
    if change=="review":state["review"]["citation_integrity_ok"]=False
    if change=="missing_review":state.pop("review")
    if change=="citation":check["public_citations"][0]["chunk_id"]="forged"
    if change=="quote":check["public_quotes"][0]["text"]="A quote not in the source"
    if change=="nli_flag":check["nli_assessment"]["public"]["predictions"][0]["scores"]={"entailment":.02,"neutral":.97,"contradiction":.01}
    if change=="nli_missing":check.pop("nli_assessment")
    if change=="nli_nan":check["nli_assessment"]["public"]["predictions"][0]["scores"]["entailment"]=float("nan")
    if change=="duplicate_check":state["verification"]["checks"].append(deepcopy(check))
    result=SynthesisAgent().run(state)
    assert result["status"]=="needs_review" and result["claims"]==[] and result["audit_issues"]
    assert state["draft"]["claims"][0]["text"] not in result["message"]


def test_neutral_source_removed_by_a3_can_remain_in_nli_predictions():
    state=supported_state();check=state["verification"]["checks"][0]
    extra=deepcopy(check["public_evidence"][0]);extra.update(evidence_id="P2",chunk_id="other")
    check["public_evidence"].append(extra)
    predictions=check["nli_assessment"]["public"]["predictions"]+nli("P2",False)["predictions"]
    check["nli_assessment"]["public"]=aggregate_predictions(predictions)
    result=SynthesisAgent().run(state)
    assert result["status"]=="answered" and len(result["sources"])==2


def test_support_from_a_different_public_id_is_not_accepted():
    state=supported_state();check=state["verification"]["checks"][0]
    extra=deepcopy(check["public_evidence"][0]);extra.update(evidence_id="P2",chunk_id="other")
    check["public_evidence"].append(extra)
    check["nli_assessment"]["public"]=aggregate_predictions(nli("P1",False)["predictions"]+nli("P2")["predictions"])
    assert SynthesisAgent().run(state)["status"]=="needs_review"


def test_reused_public_short_ids_from_different_claims_get_distinct_global_references():
    state=supported_state()
    other=supported_state("Nội dung B có trong tài liệu.","b","I2")
    other["draft"]["claims"][0]["claim_id"]="c2"
    other["verification"]["checks"][0]["claim_id"]="c2"
    state["draft"]["claims"]+=other["draft"]["claims"]
    state["internal_evidence"]+=other["internal_evidence"]
    state["verification"]["checks"]+=other["verification"]["checks"]
    result=SynthesisAgent().run(state)
    assert result["status"]=="answered"
    assert result["claims"][1]["reference_ids"]==["3","4"]
    assert [s["chunk_id"] for s in result["sources"] if s["warehouse"]=="public"]==["chunk-a","chunk-b"]


def test_no_draft_reports_insufficient_instead_of_exposing_model_logs():
    result=SynthesisAgent().run({"status":"insufficient_evidence","model_calls":[{"raw":"PRIVATE_MODEL_LOG"}]})
    assert result["status"]=="insufficient_evidence" and "PRIVATE_MODEL_LOG" not in result["message"]


def test_cli_does_not_fall_back_to_raw_draft_when_final_answer_is_missing():
    result={"knowledge":{"draft":{"claims":[{"text":"UNVERIFIED_TEXT","citations":[]}]}}}
    assert "UNVERIFIED_TEXT" not in render_result(result)


def test_saved_turn_keeps_raw_review_but_displays_only_a4_message(tmp_path):
    knowledge=supported_state("WITHHELD_TEXT")
    knowledge["verification"]["checks"][0]["status"]="insufficient"
    final=SynthesisAgent().run(knowledge)
    class Graph:
        def invoke(self,state):
            return {"status":"knowledge_result","knowledge":knowledge,"final_answer":final,"message":final["message"]}
    result=run_turn(Graph(),{"message":"Câu hỏi"},tmp_path/"turn.json")
    assert "WITHHELD_TEXT" not in result["display_message"]
    assert result["knowledge"]["draft"]["claims"][0]["text"]=="WITHHELD_TEXT"
