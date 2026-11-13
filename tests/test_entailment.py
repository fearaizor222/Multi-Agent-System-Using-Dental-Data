"""Quy tắc ghép NLI; các phản hồi dựng sẵn không dùng để đo chất lượng model."""
import unittest
from langchain_core.messages import AIMessage
from src.agents.contracts import Draft, DraftClaim, citation_of, evidence_bundle
from src.agents.drafting import DraftingAgent
from src.agents.verification import VerificationAgent
from src.rag.base import DentalDocument
from src.rag.entailment import aggregate_predictions
from src.rag.entailment import LocalEntailment


def prediction(evidence_id="P1", entailment=0.95, contradiction=0.01, truncated=False):
    return {"evidence_id": evidence_id, "truncated": truncated,
            "scores": {"entailment": entailment, "contradiction": contradiction, "neutral": 1 - entailment - contradiction}}


class StubLLM:
    def invoke(self, messages):
        return AIMessage(content='{"draft_support":"supported","public_support":"supported","public_evidence_ids":["P1"],"public_quotes":[{"evidence_id":"P1","text":"Bằng chứng công khai."}],"reason":"Có hỗ trợ."}')


class StubRetriever:
    def retrieve(self, query, top_k):
        return [DentalDocument(id="pub-1", title="", content="Bằng chứng công khai.",
            metadata={"warehouse": "public", "document_id": "pub", "source": "msd"})]


class EntailmentTests(unittest.TestCase):
    def internal_and_draft(self):
        internal = evidence_bundle([DentalDocument(id="int-1", title="", content="Nội dung nội bộ.",
            metadata={"warehouse": "internal", "parent_id": "chapter"})], "I", "internal")
        draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text="Nhận định.", citations=[citation_of(internal[0])])])
        return internal, draft

    def test_positive_and_missing_evidence(self):
        self.assertEqual(aggregate_predictions([prediction()])["status"], "supported")
        self.assertEqual(aggregate_predictions([])["status"], "insufficient")

    def test_contradiction(self):
        result = aggregate_predictions([prediction(entailment=0.01, contradiction=0.97)])
        self.assertEqual(result["status"], "contradicted")
        self.assertEqual(result["contradiction_ids"], ["P1"])

    def test_conflicting_sources_are_not_accepted(self):
        result = aggregate_predictions([prediction(), prediction("P2", 0.01, 0.97)])
        self.assertEqual(result["status"], "insufficient")
        self.assertTrue(result["conflicting_evidence"])

    def test_truncated_pair_and_weak_score_are_not_accepted(self):
        self.assertEqual(aggregate_predictions([prediction(truncated=True)])["status"], "insufficient")
        self.assertEqual(aggregate_predictions([prediction(entailment=0.79)])["status"], "insufficient")

    def test_nli_can_block_llm_supported_verdict(self):
        internal, draft = self.internal_and_draft()
        class ContradictingNLI:
            def assess(self, claim, evidence):
                return aggregate_predictions([prediction(evidence[0].evidence_id, 0.01, 0.97)])
        result = VerificationAgent(StubLLM(), StubRetriever(), ContradictingNLI()).run("Câu hỏi", draft, internal)
        check = result["checks"][0]
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(check["status"], "contradicted")
        self.assertEqual(check["llm_assessment"]["draft_support"], "supported")
        self.assertTrue(check["model_disagreement"])

    def test_nli_neutral_does_not_pass_llm_support(self):
        internal, draft = self.internal_and_draft()
        class NeutralNLI:
            def assess(self, claim, evidence):
                return aggregate_predictions([prediction(evidence[0].evidence_id, 0.05, 0.05)])
        result = VerificationAgent(StubLLM(), StubRetriever(), NeutralNLI()).run("Câu hỏi", draft, internal)
        self.assertEqual(result["checks"][0]["status"], "insufficient")

    def test_duplicate_draft_claims_do_not_inflate_claim_count(self):
        internal, _ = self.internal_and_draft()
        class DuplicateLLM:
            def invoke(self, messages):
                return AIMessage(content='{"status":"draft","claims":[{"text":"Nội dung.","evidence_ids":["I1"]},{"text":"Nội dung.","evidence_ids":["I1"]}]}')
        result = DraftingAgent(DuplicateLLM()).run("Câu hỏi", internal)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.discarded_duplicate_claims, 1)

    def test_long_evidence_windows_have_batch_axis_and_cover_all_tokens(self):
        import torch
        from types import SimpleNamespace
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [7] * len(text.split())}
            def num_special_tokens_to_add(self, pair):
                return 3
            def prepare_for_model(self, first, pair_ids, **kwargs):
                values = torch.tensor([1, *first, 2, *pair_ids, 3])
                return {"input_ids": values, "attention_mask": torch.ones_like(values)}
        shapes = []
        def model(**inputs):
            shapes.append(tuple(inputs["input_ids"].shape))
            assert inputs["input_ids"].ndim == 2
            return SimpleNamespace(logits=torch.tensor([[8.0, 0.0, 0.0]]))
        nli = LocalEntailment.__new__(LocalEntailment)
        nli.tokenizer, nli.model, nli.labels, nli.threshold = Tokenizer(), model, ["entailment", "neutral", "contradiction"], 0.8
        internal, _ = self.internal_and_draft()
        internal[0].content = "word " * 700
        result = nli.assess("claim " * 10, internal)
        self.assertGreater(len(shapes), 1)
        self.assertTrue(all(shape[0] == 1 and shape[1] <= 512 for shape in shapes))
        self.assertEqual(result["predictions"][0]["premise_token_start"], 0)
        self.assertEqual(result["predictions"][-1]["premise_token_end"], 700)
        self.assertEqual(result["support_ids"], ["I1"])


if __name__ == "__main__":
    unittest.main()
