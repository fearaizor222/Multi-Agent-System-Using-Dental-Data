"""Kiểm tra checkpoint, quan hệ ID/vector và bảo vệ index khi dữ liệu đổi."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rag.base import DentalDocument
from src.rag.embedding_store import DenseIndex, build_index
from src.rag.local_embeddings import passage_body


class TestEncoder:
    __test__ = False
    dimension = 3
    identity = {"model": "unit-test-only", "dimension": 3}
    runtime = {"device": "unit-test-only"}

    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def token_lengths(self, inputs):
        return [len(text.split()) for text in inputs]

    def encode_inputs(self, inputs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("interrupted")
        values = np.array([[len(text), int(hashlib.sha256(text.encode()).hexdigest()[:5], 16) % 997, 1]
                           for text in inputs], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class EmbeddingStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.documents = [DentalDocument(id=str(i), title="Tiêu đề", content="từ " * n,
                            metadata={"token_count": n + 3, "language": "vi"}) for i, n in enumerate([3, 15, 1, 8])]

    def test_resume_keeps_original_id_order_after_length_sort(self):
        encoder = TestEncoder(fail_on_call=2)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            build_index(self.documents, {"input": "hash"}, self.path, encoder, checkpoint_size=2)
        with self.assertRaisesRegex(ValueError, "Index chưa chạy xong"):
            DenseIndex(self.path)
        self.assertEqual(json.loads((self.path / "manifest.json").read_text())["completed_rows"], 2)
        encoder = TestEncoder()
        build_index(self.documents, {"input": "hash"}, self.path, encoder, checkpoint_size=2)
        self.assertEqual(encoder.calls, 1)
        index = DenseIndex(self.path)
        expected = TestEncoder().encode_inputs(["passage: " + passage_body(doc) for doc in self.documents])
        np.testing.assert_allclose(index.vectors, expected)
        self.assertEqual([doc.id for doc in index.documents], [doc.id for doc in self.documents])

    def test_changed_source_or_model_requires_new_directory(self):
        build_index(self.documents, {"input": "hash"}, self.path, TestEncoder())
        with self.assertRaisesRegex(ValueError, "model hoặc dữ liệu đã đổi"):
            build_index(self.documents, {"input": "changed"}, self.path, TestEncoder())
        encoder = TestEncoder()
        encoder.identity = {"model": "different-model", "dimension": 3}
        with self.assertRaisesRegex(ValueError, "model hoặc dữ liệu đã đổi"):
            build_index(self.documents, {"input": "hash"}, self.path, encoder)

    def test_metadata_tampering_is_detected(self):
        build_index(self.documents, {"input": "hash"}, self.path, TestEncoder())
        with (self.path / "records.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "records.jsonl"):
            DenseIndex(self.path)

    def test_oversize_input_is_rejected_before_embedding(self):
        self.documents[0].content = "từ " * 600
        encoder = TestEncoder()
        with self.assertRaisesRegex(ValueError, "Token không khớp hoặc vượt giới hạn"):
            build_index(self.documents, {"input": "hash"}, self.path, encoder)
        self.assertEqual(encoder.calls, 0)

    def test_import_can_explicitly_use_512_token_limit_without_truncation(self):
        self.documents[0].content = "từ " * 399
        self.documents[0].metadata["token_count"] = 402
        encoder = TestEncoder()
        with self.assertRaisesRegex(ValueError, "Token không khớp hoặc vượt giới hạn"):
            build_index(self.documents, {}, self.path, encoder)
        state = build_index(self.documents, {}, self.path, encoder, max_passage_tokens=512)
        self.assertEqual(state["passage_token_limit"], 512)
        self.assertEqual(state["max_passage_tokens"], 402)
        self.assertEqual(DenseIndex(self.path).documents[0].content, self.documents[0].content)

    def test_token_limit_cannot_exceed_encoder_capacity(self):
        with self.assertRaisesRegex(ValueError, "max_passage_tokens"):
            build_index(self.documents, {}, self.path, TestEncoder(), max_passage_tokens=513)

    def test_new_version_only_embeds_changed_text_and_keeps_new_ids(self):
        previous = self.path / "old"
        current = self.path / "new"
        build_index(self.documents, {"input": "old-hash"}, previous, TestEncoder())
        changed = [doc.model_copy(deep=True) for doc in self.documents]
        for doc in changed:
            doc.id = "new-" + doc.id
        changed[1].content = "nội dung mới cần nhúng"
        changed[1].metadata["token_count"] = 8
        encoder = TestEncoder()
        state = build_index(changed, {"input": "new-hash"}, current, encoder, reuse_from=previous)
        self.assertEqual(state["reused_rows"], 3)
        self.assertEqual(state["new_embedding_rows"], 1)
        self.assertEqual(encoder.calls, 1)
        index = DenseIndex(current)
        expected = TestEncoder().encode_inputs(["passage: " + passage_body(doc) for doc in changed])
        np.testing.assert_allclose(index.vectors, expected)
        self.assertTrue(all(doc.id.startswith("new-") for doc in index.documents))

    def test_reuse_rejects_other_model_before_creating_vector_files(self):
        previous = self.path / "old"
        current = self.path / "new"
        build_index(self.documents, {"input": "hash"}, previous, TestEncoder())
        encoder = TestEncoder()
        encoder.identity = {"model": "another-model", "dimension": 3}
        with self.assertRaisesRegex(ValueError, "Không tái sử dụng vector từ model khác"):
            build_index(self.documents, {"input": "hash"}, current, encoder, reuse_from=previous)
        self.assertFalse((current / "vectors.npy").exists())


if __name__ == "__main__":
    unittest.main()
