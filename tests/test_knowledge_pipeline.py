"""Kiểm tra hợp đồng dữ liệu và các nhánh lỗi; LLM trong file này là dữ liệu test."""
import json
import unittest

from langchain_core.messages import AIMessage

from src.agents.contracts import Draft, DraftClaim, Evidence, audit_citations, citation_of, evidence_bundle
from src.agents.drafting import DraftingAgent
from src.agents.structured_output import ModelOutputError, parse_json
from src.agents.verification import VerificationAgent
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from src.rag.base import DentalDocument
from src.rag.reranked_retriever import RerankedRetriever

PUBLIC_QUOTE = {"evidence_id": "P1", "text": "Máng nhai là khí cụ bằng nhựa đặt giữa hai cung răng."}


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        value = self.responses.pop(0)
        return AIMessage(content=json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)


def doc(doc_id="internal-1", warehouse="internal"):
    return DentalDocument(id=doc_id, title="Máng nhai", content="Máng nhai là khí cụ bằng nhựa đặt giữa hai cung răng.",
        metadata={"warehouse": warehouse, "parent_id": "chapter-12", "document_id": "textbook" if warehouse == "internal" else "MSD-example",
                  "source_spans": [{"pdf_page": 260}], "source": "msd" if warehouse == "public" else "textbook", "language": "vi"})


class FixedRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def retrieve(self, query, **options):
        self.calls.append((query, options))
        return self.documents[:options["top_k"]]


def valid_draft(evidence):
    return Draft(status="draft", claims=[DraftClaim(claim_id="c1", text="Máng nhai là khí cụ bằng nhựa.", citations=[citation_of(evidence[0])])])


class KnowledgePipelineTests(unittest.TestCase):
    def setUp(self):
        self.internal = evidence_bundle([doc()], "I", "internal")

    def test_draft_resolves_short_ids_to_original_chunk_and_source(self):
        model = ScriptedLLM({"status": "draft", "claims": [{"text": "Khí cụ bằng nhựa.", "evidence_ids": ["I1"]}]})
        result = DraftingAgent(model).run("Trình bày về máng nhai", self.internal)
        citation = result.claims[0].citations[0]
        self.assertEqual(citation.chunk_id, "internal-1")
        self.assertEqual(citation.source_id, "textbook")
        self.assertEqual(audit_citations(result.claims[0], self.internal), [])

    def test_draft_rejects_invented_citation(self):
        model = ScriptedLLM(*[{"status": "draft", "claims": [{"text": "Nội dung.", "evidence_ids": ["I999"]}]}] * 2)
        with self.assertRaisesRegex(ModelOutputError, "trích dẫn"):
            DraftingAgent(model).run("Câu hỏi", self.internal)

    def test_draft_rejects_public_context(self):
        with self.assertRaisesRegex(ValueError, "nội bộ"):
            DraftingAgent(ScriptedLLM()).run("Câu hỏi", evidence_bundle([doc(warehouse="public")], "P", "public"))

    def test_empty_evidence_does_not_call_llm(self):
        model = ScriptedLLM()
        result = DraftingAgent(model).run("Câu hỏi", [])
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(model.calls, 0)

    def test_invalid_json_is_retried_then_stopped(self):
        model = ScriptedLLM("not json", '{"status":"draft","claims":[]}')
        agent = DraftingAgent(model)
        with self.assertRaises(ModelOutputError):
            agent.run("Câu hỏi", self.internal)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(agent.last_trace), 2)

    def test_json_can_be_repaired_once(self):
        model = ScriptedLLM("not json", {"status": "insufficient_evidence", "claims": []})
        result = DraftingAgent(model).run("Câu hỏi", self.internal)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(model.calls, 2)

    def test_forged_chunk_is_flagged_before_public_retrieval(self):
        draft = valid_draft(self.internal)
        draft.claims[0].citations[0].chunk_id = "unseen-chunk"
        public, model = FixedRetriever([]), ScriptedLLM()
        result = VerificationAgent(model, public).run("Câu hỏi", draft, self.internal)
        self.assertEqual(result["checks"][0]["status"], "invalid_citation")
        self.assertEqual(model.calls, 0)
        self.assertEqual(public.calls, [])

    def test_verifier_uses_public_corpus_and_preserves_provenance(self):
        public = FixedRetriever([doc("public-1", "public")])
        model = ScriptedLLM({"draft_support": "supported", "public_support": "supported",
                            "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE], "reason": "Hai đoạn cùng hỗ trợ nhận định."})
        result = VerificationAgent(model, public).run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["status"], "supported")
        citation = result["checks"][0]["public_citations"][0]
        self.assertEqual(citation["chunk_id"], "public-1")
        self.assertEqual(citation["warehouse"], "public")
        self.assertEqual(public.calls[0][1]["top_k"], 2)

    def test_related_but_insufficient_evidence_does_not_pass(self):
        model = ScriptedLLM({"draft_support": "supported", "public_support": "insufficient",
                            "public_evidence_ids": [], "reason": "Nguồn công khai chưa nói rõ điều này."})
        result = VerificationAgent(model, FixedRetriever([])).run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["checks"][0]["status"], "insufficient")

    def test_verifier_rejects_invented_public_evidence(self):
        model = ScriptedLLM(*[{"draft_support": "supported", "public_support": "supported",
                            "public_evidence_ids": ["P999"], "public_quotes": [{"evidence_id": "P999", "text": PUBLIC_QUOTE["text"]}], "reason": "Có hỗ trợ."}] * 2)
        agent = VerificationAgent(model, FixedRetriever([doc("public-1", "public")]))
        result = agent.run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["checks"][0]["status"], "not_checked")
        self.assertEqual(result["checks"][0]["error_type"], "ModelOutputError")
        self.assertEqual(result["checks"][0]["public_citations"], [])
        self.assertEqual(len(agent.last_trace), 2)

    def test_one_bad_claim_does_not_discard_the_next_claim(self):
        model = ScriptedLLM("bad JSON", "bad JSON again", {"draft_support": "supported", "public_support": "supported",
            "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE]})
        draft = valid_draft(self.internal)
        draft.claims.append(draft.claims[0].model_copy(deep=True, update={"claim_id": "c2", "text": "Nhận định thứ hai."}))
        agent = VerificationAgent(model, FixedRetriever([doc("public-1", "public")]))
        result = agent.run("Câu hỏi", draft, self.internal)
        self.assertEqual([check["status"] for check in result["checks"]], ["not_checked", "supported"])
        self.assertEqual(result["not_checked_claims"], 1)
        self.assertEqual([entry["claim_id"] for entry in agent.last_trace], ["c1", "c1", "c2"])

    def test_long_claim_is_not_silently_truncated_or_sent_to_llm(self):
        from src.rag.entailment import ClaimTooLong
        class RejectingNLI:
            def validate_claim(self, text):
                raise ClaimTooLong("quá dài")
        model, public = ScriptedLLM(), FixedRetriever([doc("public-1", "public")])
        result = VerificationAgent(model, public, RejectingNLI()).run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["checks"][0]["status"], "not_checked")
        self.assertIsNone(result["checks"][0]["public_support"])
        self.assertEqual(result["checks"][0]["error_type"], "ClaimTooLong")
        self.assertEqual(model.calls, 0)
        self.assertEqual(public.calls, [])

    def test_verifier_can_retry_internal_id_in_public_field(self):
        model = ScriptedLLM(
            {"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["I1"], "public_quotes": [{"evidence_id": "I1", "text": PUBLIC_QUOTE["text"]}], "reason": "Có căn cứ."},
            {"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE], "reason": "Có căn cứ."})
        agent = VerificationAgent(model, FixedRetriever([doc("public-1", "public")]))
        result = agent.run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(model.calls, 2)
        self.assertIn("error", agent.last_trace[0])
        self.assertTrue(agent.last_trace[1]["valid"])

    def test_contradiction_is_not_confused_with_absence(self):
        model = ScriptedLLM({"draft_support": "contradicted", "public_support": "insufficient",
                            "public_evidence_ids": [], "reason": "Nhận định trái với đoạn nội bộ."})
        result = VerificationAgent(model, FixedRetriever([doc("public-1", "public")])).run("Câu hỏi", valid_draft(self.internal), self.internal)
        self.assertEqual(result["checks"][0]["status"], "contradicted")

    def test_graph_runs_retrieve_draft_verify(self):
        model = ScriptedLLM({"status": "draft", "claims": [{"text": "Khí cụ bằng nhựa.", "evidence_ids": ["I1"]}]},
                            {"draft_support": "supported", "public_support": "supported", "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE], "reason": "Có căn cứ."})
        graph = create_knowledge_pipeline(FixedRetriever([doc()]), FixedRetriever([doc("public-1", "public")]), model)
        state = graph.invoke({"question": "Trình bày về máng nhai"})
        self.assertEqual(state["status"], "supported")
        self.assertEqual([item["step"] for item in state["trace"]], ["a2_retrieve", "a2_draft", "a3_verify"])

    def test_graph_stops_before_verification_on_invalid_draft(self):
        model, public = ScriptedLLM("not json", "still not json"), FixedRetriever([])
        graph = create_knowledge_pipeline(FixedRetriever([doc()]), public, model)
        state = graph.invoke({"question": "Trình bày về máng nhai"})
        self.assertEqual(state["status"], "error")
        self.assertEqual(public.calls, [])
        self.assertEqual(len(state["model_calls"][0]["calls"]), 2)

    def test_graph_stops_on_empty_internal_evidence(self):
        model, public = ScriptedLLM(), FixedRetriever([])
        state = create_knowledge_pipeline(FixedRetriever([]), public, model).invoke({"question": "Câu hỏi"})
        self.assertEqual(state["status"], "insufficient_evidence")
        self.assertEqual(model.calls, 0)

    def test_rerank_wrapper_filters_before_candidate_selection(self):
        retriever = FixedRetriever([doc()])
        class Passthrough:
            def rerank(self, query, documents, top_k):
                return documents[:top_k]
        wrapped = RerankedRetriever(retriever, Passthrough(), candidate_k=10)
        wrapped.retrieve("Câu hỏi", top_k=2, source="msd", language="vi")
        self.assertEqual(retriever.calls[0][1], {"top_k": 10, "source": "msd", "language": "vi"})
        with self.assertRaises(ValueError):
            wrapped.retrieve("Câu hỏi", top_k=11)

    def test_parser_rejects_extra_text(self):
        self.assertEqual(parse_json('```json\n{"ok":true}\n```'), {"ok": True})
        with self.assertRaises(ValueError):
            parse_json('{"ok":true} thêm nội dung')


if __name__ == "__main__":
    unittest.main()
