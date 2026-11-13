"""Hồi quy sâu răng/định nghĩa dài và không dùng một nguồn tự kiểm chứng."""
from copy import deepcopy

import pytest

from src.agents.contracts import evidence_bundle, independent_public_sources, Draft, DraftClaim, citation_of
from src.agents.verification import VerificationAgent
from src.agents.drafting import DraftingAgent
from src.agents.synthesis import SynthesisAgent
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from src.graph.pipeline_review import review_run
from src.rag.base import DentalDocument
from src.rag.entailment import aggregate_predictions
from tests.test_knowledge_pipeline import ScriptedLLM, FixedRetriever


DEFINITION = "Sâu răng là tình trạng tổn thương mất mô cứng của răng."


def source(key, body, warehouse="public", provider="msd", document_id=None):
    return DentalDocument(id=key, title="Tài liệu", content=body,
                          metadata={"warehouse": warehouse, "document_id": document_id or key,
                                    "source": provider, "source_url": "https://example.org/" + key})


def evidence(body, warehouse="internal"):
    return evidence_bundle([source("one", body, warehouse)], "I" if warehouse == "internal" else "R", warehouse)


def test_definition_does_not_guess_from_incidental_topic_mention():
    model = ScriptedLLM()
    result = DraftingAgent(model).run("sâu răng là gì", evidence("Nhạy cảm răng có thể do nghiến răng. Cần phân biệt với sâu răng."))
    assert result.status == "insufficient_evidence" and not result.claims
    assert model.calls == 0


def test_definition_is_extracted_and_appended_purpose_is_separated():
    result = DraftingAgent(ScriptedLLM()).run("Máng nhai là gì?", evidence(
        "1.1.3. Định nghĩa\nMáng nhai là một khí cụ bằng nhựa đặt giữa hai cung răng được thực hiện\nvới mục đích chẩn đoán và điều trị rối loạn cắn khớp."))
    assert result.claims[0].text == "Máng nhai là một khí cụ bằng nhựa."
    assert result.claims[0].citations[0].warehouse == "internal"


@pytest.mark.parametrize("text", [
    "Không đúng khi nói " + DEFINITION,
    "Nếu chưa xác định nguyên nhân:\n" + DEFINITION,
    "Sâu răng sữa là tình trạng tổn thương mất mô cứng của răng.",
])
def test_definition_does_not_strip_negation_conditions_or_broaden_subtype(text):
    assert DraftingAgent(ScriptedLLM()).run("sâu răng là gì", evidence(text)).claims == []


def test_definition_keeps_conditions_in_its_sentence():
    body = "Bệnh X là tổn thương chỉ khi có điều kiện Y."
    assert DraftingAgent(ScriptedLLM()).run("Bệnh X là gì", evidence(body)).claims[0].text == body


def test_definition_separates_alias_from_the_main_fact():
    body = "Bệnh X là một tình trạng tổn thương của mô, thường được gọi là tên Y."
    agent = DraftingAgent(ScriptedLLM())
    result = agent.run("Bệnh X là gì", evidence(body))
    assert result.claims[0].text == "Bệnh X là một tình trạng tổn thương của mô."
    assert agent.last_grounding[0]["original_sentence"] == body


def test_a2_rejects_public_evidence_even_with_explicit_override():
    data = evidence(DEFINITION, "public")
    for options in ({}, {"warehouse": "public"}):
        with pytest.raises(ValueError, match="nội bộ"):
            DraftingAgent(ScriptedLLM()).run("sâu răng là gì", data, **options)
    with pytest.raises(ValueError, match="nội bộ"):
        DraftingAgent(ScriptedLLM()).run("sâu răng là gì", [], warehouse="public")


def test_generated_unsupported_claim_is_removed_before_a3():
    class Neutral:
        def assess(self, claim, evidence):
            return {"status": "insufficient", "conflicting_evidence": False}
    model = ScriptedLLM({"status": "draft", "claims": [{"text": "Một nhận định sai.", "evidence_ids": ["I1"]}]})
    result = DraftingAgent(model, Neutral()).run("Câu hỏi về nguyên nhân", evidence("Thông tin khác."))
    assert result.status == "insufficient_evidence" and not result.claims


class SupportingNLI:
    def assess(self, claim, evidence):
        return aggregate_predictions([{"evidence_id": e.evidence_id, "truncated": False,
            "premise_token_start": 0, "premise_token_end": 10,
            "scores": {"entailment": .99, "neutral": .005, "contradiction": .005}} for e in evidence])


class PublicRetriever:
    def __init__(self, primary, other):
        self.primary, self.other = primary, other
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append(("draft", query))
        return [self.primary]

    def retrieve_excluding(self, query, top_k, **kwargs):
        self.calls.append(("corroboration", kwargs))
        # Deliberately ignore exclusions here: production agent must recheck them.
        return [self.other]


def public_fallback_state(other_provider="byt", other_document="byt-doc"):
    internal = source("narrow-book", "Nhạy cảm răng cần phân biệt với sâu răng.", "internal", "textbook")
    primary = source("msd-chunk", DEFINITION, document_id="msd-doc")
    other = source("byt-chunk", DEFINITION, provider=other_provider, document_id=other_document)
    public = PublicRetriever(primary, other)
    model = ScriptedLLM({"draft_support": "supported", "public_support": "supported",
                         "public_evidence_ids": ["P1"], "public_quotes": [{"evidence_id": "P1", "text": DEFINITION}]})
    # Historical proof-audit fixture only. Production A2 no longer creates public drafts.
    items = evidence_bundle([primary], "R", "public")
    draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text=DEFINITION, citations=[citation_of(items[0])])])
    checks = VerificationAgent(model, public, SupportingNLI()).run("sâu răng là gì", draft, items)
    state = {"question": "sâu răng là gì", "draft": draft.model_dump(), "draft_origin": "public",
             "draft_evidence": [e.model_dump() for e in items], "internal_evidence": [],
             "repair_attempted": True, "verification": checks,
             "trace": [{"step": "a3_verify", "status": checks['status'], "milliseconds": 1}], "model_calls": [],
             "status": checks["status"], "revision_history": [{"draft": {"claims": []}}]}
    catalogs = ({internal.id: internal}, {primary.id: primary, other.id: other})
    state["review"] = review_run(state, *catalogs)
    return state, public, catalogs


def test_archived_public_proof_preserves_sources_without_a2_generation():
    state, retriever, _ = public_fallback_state()
    assert state["repair_attempted"] and state["draft_origin"] == "public"
    assert state["revision_history"][0]["draft"]["claims"] == []
    assert state["review"]["citation_integrity_ok"]
    result = SynthesisAgent().run(state)
    assert result["status"] == "answered" and result["evidence_basis"] == "public_cross_source"
    assert {s["warehouse"] for s in result["sources"]} == {"public"}
    assert len({s["source_id"] for s in result["sources"]}) == 2
    excluded = next(c[1] for c in retriever.calls if c[0] == "corroboration")
    assert excluded["document_ids"] == {"msd-doc"} and excluded["sources"] == {"msd"}


@pytest.mark.parametrize("provider,document", [("msd", "another-msd-doc"), ("byt", "msd-doc")])
def test_same_provider_or_document_cannot_self_verify_and_repair_is_bounded(provider, document):
    state, retriever, _ = public_fallback_state(provider, document)
    assert state["verification"]["checks"][0]["status"] == "insufficient"
    assert SynthesisAgent().run(state)["claims"] == []
    assert not any(c[0] == "draft" for c in retriever.calls)


@pytest.mark.parametrize("change", ["warehouse", "source_content", "policy", "nli", "provider"])
def test_public_branch_audit_and_a4_reject_forged_proof(change):
    state, _, catalogs = public_fallback_state()
    if change == "warehouse": state["draft_evidence"][0]["warehouse"] = "internal"
    if change == "source_content": state["draft_evidence"][0]["content"] += " Thêm nội dung."
    check = state["verification"]["checks"][0]
    if change == "policy": check["verification_policy"] = "internal_public_v1"
    if change == "nli": check["nli_assessment"].pop("draft")
    if change == "provider": check["public_evidence"][0]["metadata"]["source"] = "msd"
    state["review"] = review_run(state, *catalogs)
    assert SynthesisAgent().run(state)["status"] == "needs_review"


def test_insufficient_public_sources_stay_insufficient():
    internal = source("internal", "Nội dung khác", "internal")
    model = ScriptedLLM()
    state = create_knowledge_pipeline(FixedRetriever([internal]), FixedRetriever([]), model).invoke({"question": "Bệnh X là gì"})
    assert state["status"] == "insufficient_evidence" and not state["draft"]["claims"]
    assert model.calls == 0
