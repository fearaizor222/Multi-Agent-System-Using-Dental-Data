"""Embedding E5 local, dùng cùng một cấu hình cho tài liệu và câu hỏi."""
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
INPUT_VERSION = "e5-passages-v1"


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def passage_body(document):
    # Chunk công khai đã chứa tên tài liệu và đường dẫn đề mục.
    if document.metadata.get("warehouse") == "public":
        return document.content
    return document.title + "\n" + document.content


class LocalE5Embeddings(Embeddings):
    def __init__(self, batch_size=32, threads=6):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if batch_size < 1 or threads < 1:
            raise ValueError("batch_size và threads phải dương")
        torch.set_num_threads(threads)
        self.batch_size = batch_size
        snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
        self.model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True, trust_remote_code=False)
        self.model.max_seq_length = 512
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.identity = {
            "model": MODEL, "revision": REVISION, "dimension": self.dimension,
            "normalize_embeddings": True, "dtype": "float32", "input_version": INPUT_VERSION,
            "passage_prefix": "passage: ", "query_prefix": "query: ", "max_sequence_length": 512,
            "model_files_sha256": {name: file_sha256(snapshot / name) for name in [
                "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                "sentence_bert_config.json", "modules.json", "1_Pooling/config.json"]},
        }
        self.runtime = {"device": "cpu", "threads": threads, "batch_size": batch_size,
                        **{name: importlib.metadata.version(name) for name in ["torch", "sentence-transformers", "transformers", "numpy"]}}

    def token_lengths(self, inputs):
        result = []
        for start in range(0, len(inputs), 256):
            batch = self.model.tokenizer(inputs[start:start + 256], padding=False, truncation=False,
                                         add_special_tokens=True, return_length=True, verbose=False)
            result.extend(batch["length"])
        return result

    def encode_inputs(self, inputs):
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        if max(self.token_lengths(inputs)) > 512:
            raise ValueError("Đầu vào vượt 512 token; cần chia lại trước khi embedding")
        vectors = self.model.encode(inputs, batch_size=self.batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Model trả vector không hợp lệ")
        return vectors

    def embed_documents(self, texts):
        return self.encode_inputs(["passage: " + text for text in texts]).tolist()

    def embed_query(self, text):
        if not text.strip():
            raise ValueError("Câu hỏi rỗng")
        return self.encode_inputs(["query: " + text.strip()])[0].tolist()
