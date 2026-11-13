"""Hồi quy lỗi nhận định dài lấn át câu hỏi và loại mất ứng viên."""
import unittest

from src.agents.public_retrieval import retrieve_public_evidence
from src.rag.base import DentalDocument
from src.rag.reranker import RetrievalQueryTooLong
from src.rag.reranked_retriever import RerankedRetriever


def doc(key, score=0):
    return DentalDocument(id=key, title=key, content=key, score=score,
                          metadata={"warehouse": "public"})


class PublicRetrievalTests(unittest.TestCase):
    def test_protects_question_hit_even_if_claim_branch_scores_are_larger(self):
        calls = []
        class Retriever:
            def retrieve(self, query, top_k):
                calls.append((query, top_k))
                return [doc("definition", -2), doc("question-second")] if query == "question" else [doc("braces", 10), doc("claim-second", 9)]
            def rank_candidates(self, query, documents, top_k):
                self.assert_query = query
                # A query about definition rejects the high-scoring claim-only braces hit.
                return [d for d in documents if d.id in ("definition", "question-second")][:top_k]
        hits, trace = retrieve_public_evidence(Retriever(), "question", "long claim")
        self.assertEqual(calls, [("question", 2), ("long claim", 2)])
        self.assertEqual([d.id for d in hits], ["definition", "question-second"])
        self.assertEqual(trace["selected_chunk_ids"], [d.id for d in hits])
        self.assertEqual(trace["rerank_query"], "question")
        self.assertIn("braces", trace["candidate_chunk_ids"])

    def test_deduplicates_by_chunk_id_and_does_not_modify_sources(self):
        shared = doc("same")
        original = shared.model_dump_json()
        class Retriever:
            def retrieve(self, query, top_k):
                return [shared, doc(query)]
        hits, _ = retrieve_public_evidence(Retriever(), "question", "claim")
        self.assertEqual([d.id for d in hits], ["same", "question"])
        self.assertEqual(shared.model_dump_json(), original)

    def test_identical_queries_are_not_run_twice(self):
        class Retriever:
            calls = 0
            def retrieve(self, query, top_k):
                self.calls += 1
                return [doc("same")]
        retriever = Retriever()
        hits, trace = retrieve_public_evidence(retriever, " same ", "same")
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(len(trace["branches"]), 1)
        self.assertEqual(len(hits), 1)

    def test_empty_question_branch_preserves_claim_results(self):
        class Retriever:
            def retrieve(self, query, top_k):
                return [] if query == "question" else [doc("claim-1"), doc("claim-2")]
        hits, _ = retrieve_public_evidence(Retriever(), "question", "claim")
        self.assertEqual([d.id for d in hits], ["claim-1", "claim-2"])

    def test_query_length_error_is_not_hidden_or_truncated(self):
        class Retriever:
            def retrieve(self, query, top_k):
                raise RetrievalQueryTooLong("too long")
        with self.assertRaises(RetrievalQueryTooLong):
            retrieve_public_evidence(Retriever(), "question", "claim")

    def test_second_rerank_preserves_rrf_provenance_and_original_document(self):
        original = doc("source", 4)
        original.metadata["retrieval"] = {"rrf_score": .01, "rrf_rank": 3}
        saved = original.model_dump_json()
        class Ranker:
            def rerank(self, query, documents, top_k):
                result = documents[0].model_copy(deep=True, update={"score": 7})
                result.metadata["retrieval"] = {"rrf_score": 4, "rrf_rank": 1, "rerank_logit": 7}
                return [result]
        result = RerankedRetriever(None, Ranker()).rank_candidates("question", [original])[0]
        self.assertEqual(result.metadata["retrieval"]["rrf_score"], .01)
        self.assertEqual(result.metadata["retrieval"]["rrf_rank"], 3)
        self.assertEqual(result.score, 7)
        self.assertEqual(original.model_dump_json(), saved)


if __name__ == "__main__":
    unittest.main()
