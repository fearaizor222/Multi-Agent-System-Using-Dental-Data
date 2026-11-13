"""Kiểm tra lọc nguồn, thứ hạng ghép và bảo toàn dữ liệu nguồn."""
import unicodedata
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.rag.base import DentalDocument
from src.rag.bm25_retriever import BM25Retriever, tokenize
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.embedding_store import build_index


class SmallEncoder:
    dimension = 2
    identity = {"model": "test-only", "dimension": 2}
    runtime = {"device": "test-only"}

    def token_lengths(self, inputs):
        return [len(text.split()) for text in inputs]

    def encode_inputs(self, inputs):
        return np.asarray([[1.0, 0.0] for _ in inputs], dtype=np.float32)

    def embed_query(self, text):
        return [1.0, 0.0]


def document(doc_id, content, source="msd", language="vi"):
    return DentalDocument(id=doc_id, title="", content=content,
        metadata={"source": source, "language": language, "warehouse": "public",
                  "locators": [{"url": "https://example.org/" + doc_id}]})


class FixedDense:
    def __init__(self, documents):
        self.documents = documents
        self.last_options = None

    def retrieve(self, query, top_k, source=None, language=None):
        self.last_options = (top_k, source, language)
        return [d.model_copy(deep=True, update={"score": 0.9}) for d in self.documents
                if (source is None or d.metadata["source"] == source)
                and (language is None or d.metadata["language"] == language)][:top_k]


class RetrievalTests(unittest.TestCase):
    def test_unicode_and_negation_are_kept(self):
        text = "Ý nghĩa: KHÔNG do mảng bám, ARCON/NON-ARCON 0.2%"
        self.assertEqual(tokenize(text), tokenize(unicodedata.normalize("NFD", text)))
        self.assertIn("không", tokenize(text))
        self.assertIn("ý", tokenize(text))
        self.assertIn("arcon", tokenize(text))

    def test_one_document_has_positive_score_and_unknown_query_no_hits(self):
        retriever = BM25Retriever([document("a", "máng nhai")])
        hits = retriever.retrieve("máng nhai")
        self.assertEqual([d.id for d in hits], ["a"])
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(retriever.retrieve("xyznonexistent"), [])

    def test_filters_are_applied_before_top_k(self):
        docs = [document("en", "fluoride fluoride fluoride", "pmc", "en"),
                document("vi", "fluoride", "msd", "vi")]
        sparse = BM25Retriever(docs)
        dense = FixedDense(docs)
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        self.assertEqual([d.id for d in sparse.retrieve("fluoride", top_k=1, source="msd", language="vi")], ["vi"])
        self.assertEqual([d.id for d in hybrid.retrieve("fluoride", top_k=1, source="msd", language="vi")], ["vi"])
        self.assertEqual(dense.last_options, (30, "msd", "vi"))
        self.assertEqual(hybrid.retrieve("fluoride", source="byt"), [])

    def test_fusion_keeps_both_rankings_and_does_not_mutate_source(self):
        a, b = document("a", "máng"), document("b", "máng nhai")
        hybrid = HybridRetriever(dense_retriever=FixedDense([a, b]), sparse_retriever=BM25Retriever([a, b]),
                                 dense_weight=0.6, sparse_weight=0.4, rrf_k=60)
        hits = hybrid.fuse([a, b], [b, a], top_k=2)
        self.assertEqual(hits[0].id, "a")
        self.assertAlmostEqual(hits[0].score, 0.6 / 61 + 0.4 / 62)
        self.assertEqual(hits[0].metadata["retrieval"]["dense_rank"], 1)
        self.assertEqual(hits[0].metadata["retrieval"]["bm25_rank"], 2)
        hits[0].metadata["locators"][0]["url"] = "changed"
        self.assertEqual(a.metadata["locators"][0]["url"], "https://example.org/a")
        self.assertNotIn("retrieval", a.metadata)

    def test_duplicate_results_are_not_double_counted(self):
        a = document("a", "máng")
        hybrid = HybridRetriever(dense_retriever=FixedDense([a]), sparse_retriever=BM25Retriever([a]))
        hits = hybrid.fuse([a, a], [a], top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].score, 1 / 61)

    def test_bm25_copies_metadata_and_rejects_duplicate_add_without_changing_index(self):
        a = document("a", "máng nhai")
        retriever = BM25Retriever([a])
        hits = retriever.retrieve("máng")
        hits[0].metadata["locators"][0]["url"] = "changed"
        self.assertNotIn("retrieval", a.metadata)
        self.assertEqual(a.metadata["locators"][0]["url"], "https://example.org/a")
        with self.assertRaisesRegex(ValueError, "Trùng ID"):
            retriever.add_documents([a])
        self.assertEqual(len(retriever.documents), 1)

    def test_invalid_requests_and_weights(self):
        sparse = BM25Retriever()
        dense = FixedDense([])
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        for retriever in [sparse, hybrid]:
            with self.assertRaises(ValueError):
                retriever.retrieve(" ")
            with self.assertRaises(ValueError):
                retriever.retrieve("máng", top_k=0)
        for weights in [(0, 0), (-1, 1), (float("nan"), 1)]:
            with self.assertRaises(ValueError):
                HybridRetriever(dense_retriever=dense, sparse_retriever=sparse,
                                dense_weight=weights[0], sparse_weight=weights[1])

    def test_disk_index_reuses_existing_vectors_and_is_read_only(self):
        a = document("a", "máng nhai")
        a.metadata["token_count"] = 3
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            build_index([a], {"input": "test-only"}, directory, SmallEncoder())
            # BM25 trên đĩa không cần khởi tạo model embedding.
            with patch("src.rag.embedding_store.LocalE5Embeddings", side_effect=AssertionError("model was loaded")):
                sparse = BM25Retriever.from_disk(directory)
                self.assertEqual(sparse.retrieve("máng")[0].id, "a")
            hybrid = HybridRetriever.from_disk(directory, encoder=SmallEncoder())
            self.assertEqual(hybrid.retrieve("máng", source="msd")[0].id, "a")
            for retriever in [sparse, hybrid]:
                with self.assertRaisesRegex(ValueError, "chỉ đọc"):
                    retriever.add_documents([document("b", "máng nhai")])
                self.assertEqual(len(retriever.documents), 1)
            # Windows không xóa được file memmap khi retriever còn giữ nó.
            del retriever, hybrid


if __name__ == "__main__":
    unittest.main()
