"""Rà câu trích và lỗi hai model đồng ý nhưng dùng hai nguồn khác nhau."""
import json
import unittest

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agents.contracts import ClaimAssessment, Draft, DraftClaim, EvidenceQuote, citation_of, evidence_bundle
from src.agents.evidence_review import audit_quote, body_start, paragraph_contexts, quote_contexts, reconcile_public
from src.agents.verification import VerificationAgent
from src.rag.base import DentalDocument


def public_evidence(body="Không phải mọi trường hợp đều có cùng nguyên nhân."):
    return evidence_bundle([DentalDocument(id="public-chunk", title="Tiêu đề",
        content="Tiêu đề\nĐề mục\n" + body, metadata={"warehouse": "public", "document_id": "public-doc", "heading_path": ["Đề mục"]})], "P", "public")[0]


def assessment(evidence_id="P1", text="Nội dung có trong nguồn."):
    return ClaimAssessment(draft_support="supported", public_support="supported", public_evidence_ids=[evidence_id],
        public_quotes=[EvidenceQuote(evidence_id=evidence_id, text=text)], reason="Đã đối chiếu.")


class EvidenceReviewTests(unittest.TestCase):
    def test_candidate_prescreen_strips_headings_but_keeps_full_paragraph_conditions(self):
        evidence = public_evidence("Nếu thiếu dữ liệu, không được kết luận.\n\nĐoạn thứ hai.")
        original = evidence.model_dump_json()
        contexts = paragraph_contexts([evidence])
        self.assertEqual([c.content for c in contexts],
                         ["Nếu thiếu dữ liệu, không được kết luận.", "Đoạn thứ hai."])
        self.assertEqual(evidence.model_dump_json(), original)

    def test_neutral_candidates_do_not_trigger_llm_quote_invention(self):
        from tests.test_knowledge_pipeline import FixedRetriever, ScriptedLLM, doc, valid_draft
        internal = evidence_bundle([doc()], "I", "internal")
        class NLI:
            def assess(self, claim, evidence):
                supported = bool(evidence and evidence[0].warehouse == "internal")
                return {"status": "supported" if supported else "insufficient", "support_ids": ["I1"] if supported else [],
                        "contradiction_ids": [], "conflicting_evidence": False}
        llm = ScriptedLLM()
        result = VerificationAgent(llm, FixedRetriever([doc("public-1", "public")]), NLI()).run(
            "question", valid_draft(internal), internal)
        check = result["checks"][0]
        self.assertEqual(llm.calls, 0)
        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["decision_origin"], "no_public_semantic_evidence")
        self.assertEqual(len(check["public_evidence"]), 1)
        self.assertEqual(check["public_citations"], [])

    def test_candidate_support_cannot_replace_nli_for_the_actual_quote(self):
        from tests.test_knowledge_pipeline import FixedRetriever, PUBLIC_QUOTE, ScriptedLLM, doc, valid_draft
        internal = evidence_bundle([doc()], "I", "internal")
        class NLI:
            def __init__(self): self.calls = 0
            def assess(self, claim, evidence):
                self.calls += 1
                supported = self.calls < 3
                return {"status": "supported" if supported else "insufficient",
                        "support_ids": [evidence[0].evidence_id] if supported else [],
                        "contradiction_ids": [], "conflicting_evidence": False}
        llm = ScriptedLLM({"draft_support": "supported", "public_support": "supported",
                          "public_evidence_ids": ["P1"], "public_quotes": [PUBLIC_QUOTE]})
        check = VerificationAgent(llm, FixedRetriever([doc("public-1", "public")]), NLI()).run(
            "question", valid_draft(internal), internal)["checks"][0]
        self.assertEqual(check["public_candidate_nli"]["status"], "supported")
        self.assertEqual(check["nli_assessment"]["public"]["status"], "insufficient")
        self.assertEqual(check["status"], "insufficient")
        self.assertEqual(check["public_citations"], [])

    def test_quote_offsets_are_original_chunk_character_offsets(self):
        evidence = public_evidence("Dòng đầu.\nMột câu có bằng chứng.\nDòng sau.")
        result = audit_quote(evidence, EvidenceQuote(evidence_id="P1", text="Một câu có bằng chứng."))
        self.assertEqual(evidence.content[result["char_start"]:result["char_end"]], result["quote"])
        self.assertEqual(result["chunk_id"], "public-chunk")
        self.assertEqual(result["offset_basis"], "chunk.content")
        self.assertEqual(result["context"], "Một câu có bằng chứng.")

    def test_only_whitespace_differences_are_allowed(self):
        evidence = public_evidence("Nội dung có\n  xuống dòng.")
        result = audit_quote(evidence, EvidenceQuote(evidence_id="P1", text="Nội dung có xuống dòng."))
        self.assertEqual(result["quote"], "Nội dung có\n  xuống dòng.")
        with self.assertRaises(ValueError):
            audit_quote(evidence, EvidenceQuote(evidence_id="P1", text="Nội dung không xuống dòng."))

    def test_heading_cannot_be_used_as_evidence(self):
        evidence = public_evidence()
        for quote in ["Tiêu đề", "Đề mục"]:
            with self.assertRaisesRegex(ValueError, "nội dung gốc"):
                audit_quote(evidence, EvidenceQuote(evidence_id="P1", text=quote))

    def test_short_quote_keeps_negation_and_condition_from_context(self):
        evidence = public_evidence("Nếu chưa đủ dữ liệu, không được kết luận có hỗ trợ.")
        checked = audit_quote(evidence, EvidenceQuote(evidence_id="P1", text="có hỗ trợ"))
        self.assertIn("không được", checked["context"])
        self.assertIn("Nếu chưa đủ dữ liệu", checked["context"])

    def test_quote_must_belong_to_same_evidence_id(self):
        with self.assertRaisesRegex(ValueError, "không khớp"):
            audit_quote(public_evidence(), EvidenceQuote(evidence_id="P2", text="Không phải"))

    def test_definitive_assessment_requires_quotes_for_all_selected_ids(self):
        with self.assertRaises(ValidationError):
            ClaimAssessment(draft_support="supported", public_support="supported", public_evidence_ids=["P1"], reason="Có liên quan.")
        result = ClaimAssessment(draft_support="supported", public_support="insufficient", public_evidence_ids=[], reason="Chưa rõ.")
        self.assertEqual(result.public_quotes, [])

    def test_explanation_is_optional_but_evidence_is_not(self):
        result = ClaimAssessment(draft_support="supported", public_support="supported", public_evidence_ids=["P1"],
            public_quotes=[EvidenceQuote(evidence_id="P1", text="Nội dung gốc.")])
        self.assertEqual(result.reason, "")
        with self.assertRaises(ValidationError):
            ClaimAssessment(draft_support="supported", public_support="supported", public_evidence_ids=["P1"])

    def test_different_evidence_ids_are_not_model_agreement(self):
        result = assessment()
        status, ids = reconcile_public(result, {"support_ids": ["P2"], "contradiction_ids": [], "conflicting_evidence": False})
        self.assertEqual((status, ids), ("insufficient", []))
        self.assertEqual(reconcile_public(result, {"support_ids": ["P1"], "contradiction_ids": [], "conflicting_evidence": False}), ("supported", ["P1"]))

    def test_conflicting_evidence_is_not_accepted(self):
        result = assessment()
        self.assertEqual(reconcile_public(result, {"support_ids": ["P1"], "contradiction_ids": ["P1"], "conflicting_evidence": True}), ("insufficient", []))

    def test_nli_only_receives_context_for_selected_quote(self):
        internal = evidence_bundle([DentalDocument(id="internal", title="", content="Thông tin nội bộ.",
            metadata={"warehouse": "internal", "parent_id": "chapter"})], "I", "internal")
        public = public_evidence("Nội dung có trong nguồn.")
        class Retriever:
            def retrieve(self, query, top_k):
                return [DentalDocument(id=public.chunk_id, title=public.title, content=public.content, metadata=public.metadata)]
        class LLM:
            def invoke(self, messages):
                return AIMessage(content=assessment().model_dump_json())
        class NLI:
            def __init__(self): self.received = []
            def assess(self, claim, evidence):
                self.received.append(evidence)
                ids = [item.evidence_id for item in evidence]
                return {"status": "supported", "support_ids": ids, "contradiction_ids": [], "conflicting_evidence": False}
        nli = NLI()
        draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text="Nhận định.", citations=[citation_of(internal[0])])])
        result = VerificationAgent(LLM(), Retriever(), nli).run("Câu hỏi", draft, internal)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(nli.received[1][0].content, "Nội dung có trong nguồn.")
        self.assertEqual(result["checks"][0]["public_quote_checks"][0]["chunk_id"], public.chunk_id)
        self.assertIn("llm_ms", result["checks"][0]["timings"])

    def test_schema_change_does_not_modify_original_evidence(self):
        evidence = public_evidence("Nội dung có trong nguồn.")
        original = evidence.model_dump_json()
        audits, contexts = quote_contexts(assessment(), [evidence])
        self.assertEqual(evidence.model_dump_json(), original)
        self.assertEqual(contexts[0].chunk_id, evidence.chunk_id)
        self.assertEqual(audits[0]["context_start"], body_start(evidence))

    def test_empty_public_evidence_does_not_call_llm(self):
        internal = evidence_bundle([DentalDocument(id="internal", title="", content="Thông tin nội bộ.",
            metadata={"warehouse": "internal", "parent_id": "chapter"})], "I", "internal")
        draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text="Nhận định.", citations=[citation_of(internal[0])])])
        class EmptyRetriever:
            def retrieve(self, query, top_k): return []
        class UnusedLLM:
            def invoke(self, messages): raise AssertionError("Không cần gọi LLM khi thiếu nguồn công khai")
        result = VerificationAgent(UnusedLLM(), EmptyRetriever()).run("Câu hỏi", draft, internal)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["checks"][0]["public_support"], "insufficient")
        self.assertTrue(result["checks"][0]["llm_not_called"])

    def test_paragraph_contexts_preserves_pipe_table_headers_and_colon_stems(self):
        table_body = (
            "Table 5:\nComplications observed\n | Trismus | Dry socket\n"
            " | Absent | Present\n3 days | 6 (5.5%) | 6 (5.2%)"
        )
        table_contexts = paragraph_contexts([public_evidence(table_body)])
        self.assertEqual([c.content for c in table_contexts], [table_body])
        self.assertEqual(audit_quote(public_evidence(table_body), EvidenceQuote(evidence_id="P1", text=table_body))["context"], table_body)

        stem_body = "X quang:\n- Đưa gutta-percha qua lỗ rò.\n- Tiêu xương ranh giới rõ.\n\nĐoạn tách rời."
        stem_contexts = paragraph_contexts([public_evidence(stem_body)])
        self.assertEqual([c.content for c in stem_contexts],
                         ["X quang:\n- Đưa gutta-percha qua lỗ rò.\n- Tiêu xương ranh giới rõ.", "Đoạn tách rời."])
        self.assertIn("X quang:", audit_quote(public_evidence(stem_body), EvidenceQuote(evidence_id="P1", text=stem_contexts[0].content))["context"])

    def test_prescreen_uses_heading_and_cross_lingual_pmc_fallback(self):
        from tests.test_knowledge_pipeline import FixedRetriever, ScriptedLLM, doc, valid_draft
        internal = evidence_bundle([doc()], "I", "internal")
        byt_doc = DentalDocument(id="byt-1", title="Bệnh A", content="Bệnh A\nMục 1\n- Có thể kết hợp khí cụ.",
            metadata={"warehouse": "public", "document_id": "BYT-1", "heading_path": ["Mục 1"], "source": "byt", "language": "vi"})
        class HeadingNLI:
            def __init__(self): self.calls = 0; self.received = []
            def assess(self, claim, evidence):
                self.calls += 1; self.received.append(evidence)
                if self.calls == 1:
                    return {"status": "supported", "support_ids": ["I1"], "contradiction_ids": [], "conflicting_evidence": False}
                if self.calls == 2:
                    preds = [{"evidence_id": e.evidence_id, "truncated": False,
                              "scores": {"entailment": 0.9 if "#F" in e.evidence_id else 0.1, "neutral": 0.8 if "#F" not in e.evidence_id else 0.05, "contradiction": 0.05}}
                             for e in evidence]
                    return {"status": "supported", "support_ids": [e.evidence_id for e in evidence if "#F" in e.evidence_id],
                            "contradiction_ids": [], "conflicting_evidence": False, "predictions": preds}
                return {"status": "supported", "support_ids": ["P1"], "contradiction_ids": [], "conflicting_evidence": False}
        llm = ScriptedLLM({"draft_support": "supported", "public_support": "supported",
                           "public_evidence_ids": ["P1"], "public_quotes": [{"evidence_id": "P1", "text": "- Có thể kết hợp khí cụ."}]})
        nli = HeadingNLI()
        res = VerificationAgent(llm, FixedRetriever([byt_doc]), nli).run("Câu hỏi", valid_draft(internal), internal)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(res["status"], "supported")
        self.assertIn("Mục 1", nli.received[2][0].content)

        pmc_doc = DentalDocument(id="pmc-1", title="Guideline", content="Guideline\nAbstract\nThe 2023 ESE S3 guideline covers pulpitis.",
            metadata={"warehouse": "public", "document_id": "PMC1", "heading_path": ["Abstract"], "source": "pmc", "language": "en"})
        class LowCrossLingualNLI:
            def assess(self, claim, evidence):
                if evidence and evidence[0].warehouse == "internal":
                    return {"status": "supported", "support_ids": ["I1"], "contradiction_ids": [], "conflicting_evidence": False}
                return {"status": "insufficient", "support_ids": [], "contradiction_ids": [], "conflicting_evidence": False,
                        "predictions": [{"evidence_id": e.evidence_id, "truncated": False,
                                         "scores": {"entailment": 0.05, "neutral": 0.9, "contradiction": 0.05}} for e in evidence]}
        llm2 = ScriptedLLM({"draft_support": "supported", "public_support": "supported",
                            "public_evidence_ids": ["P1"], "public_quotes": [{"evidence_id": "P1", "text": "The 2023 ESE S3 guideline covers pulpitis."}]})
        res2 = VerificationAgent(llm2, FixedRetriever([pmc_doc]), LowCrossLingualNLI()).run("Câu hỏi", valid_draft(internal), internal)
        self.assertEqual(llm2.calls, 1)
        self.assertEqual(res2["checks"][0]["llm_assessment"]["public_support"], "supported")


if __name__ == "__main__":
    unittest.main()
