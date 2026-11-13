"""Kiểm tra liên kết điểm với chunk và giữ metadata; model ở đây chỉ dùng cho test."""
import unittest
from types import SimpleNamespace

import torch

from src.rag.base import DentalDocument
from src.rag.reranker import LocalReranker


class TestTokenizer:
    __test__ = False
    def __call__(self, query, texts=None, **kwargs):
        if texts is None:
            return {"input_ids": [1, 2]}
        if kwargs.get("return_length"):
            return {"length": [600 if text == "long" else 30 for text in texts]}
        return {"input_ids": torch.tensor([[0] if text == "first" else [1] for text in texts])}


class RerankerTests(unittest.TestCase):
    def setUp(self):
        self.reranker = LocalReranker.__new__(LocalReranker)
        self.reranker.tokenizer = TestTokenizer()
        self.reranker.model = lambda input_ids: SimpleNamespace(logits=input_ids.float())
        self.reranker.batch_size, self.reranker.max_length = 2, 512
        self.reranker.identity = {"repo": "unit-test-only", "revision": "test"}
        self.reranker.last_run = {}
        self.documents = [DentalDocument(id="a", title="", content="first", score=0.01,
            metadata={"warehouse": "public", "retrieval": {"method": "hybrid", "dense_rank": 2},
                      "locators": [{"url": "https://example.org/a"}]}),
            DentalDocument(id="b", title="", content="long", score=0.009,
            metadata={"warehouse": "public", "retrieval": {"method": "hybrid", "dense_rank": 1},
                      "locators": [{"url": "https://example.org/b"}]})]

    def test_reordering_keeps_chunk_source_and_old_ranks(self):
        hits = self.reranker.rerank("query", self.documents, top_k=2)
        self.assertEqual([doc.id for doc in hits], ["b", "a"])
        self.assertEqual(hits[0].metadata["retrieval"]["rrf_rank"], 2)
        self.assertEqual(hits[0].metadata["retrieval"]["dense_rank"], 1)
        self.assertEqual(hits[0].metadata["locators"][0]["url"], "https://example.org/b")
        hits[0].metadata["locators"][0]["url"] = "changed"
        self.assertEqual(self.documents[1].metadata["locators"][0]["url"], "https://example.org/b")
        self.assertEqual(self.documents[1].metadata["retrieval"]["method"], "hybrid")
        self.assertEqual(self.reranker.last_run["truncated_pairs"], 1)

    def test_duplicate_ids_and_invalid_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "trùng"):
            self.reranker.rerank("query", [self.documents[0], self.documents[0]])
        with self.assertRaises(ValueError):
            self.reranker.rerank(" ", self.documents)
        with self.assertRaises(ValueError):
            self.reranker.rerank("query", self.documents, top_k=0)
        self.assertEqual(self.reranker.rerank("query", []), [])

    def test_nan_scores_are_rejected(self):
        self.reranker.model = lambda input_ids: SimpleNamespace(logits=torch.full((len(input_ids), 1), float("nan")))
        with self.assertRaisesRegex(ValueError, "điểm không hợp lệ"):
            self.reranker.rerank("query", self.documents)


if __name__ == "__main__":
    unittest.main()
