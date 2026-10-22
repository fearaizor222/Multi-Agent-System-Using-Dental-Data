"""Cross-encoder xếp lại các ứng viên từ RRF; không dùng điểm làm độ tin cậy y khoa."""
import time

import numpy as np

from src.core.model_registry import checked_model
from src.rag.local_embeddings import passage_body


class LocalReranker:
    def __init__(self, batch_size=4, threads=6, max_length=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1 or threads < 1 or not 32 <= max_length <= 512:
            raise ValueError("Cấu hình reranker không hợp lệ")
        torch.set_num_threads(threads)
        directory, self.identity = checked_model("reranker")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float32).eval()
        self.batch_size, self.max_length = batch_size, max_length
        self.last_run = {}

    def rerank(self, query, documents, top_k=5):
        import torch
        if not query.strip() or top_k < 1:
            raise ValueError("Cần câu hỏi không rỗng và top_k dương")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Ứng viên rerank trùng ID")
        if not documents:
            self.last_run = {"candidates": 0, "truncated_pairs": 0, "milliseconds": 0}
            return []
        query_ids = self.tokenizer(query, add_special_tokens=False)["input_ids"]
        if len(query_ids) > 128:
            raise ValueError("Câu hỏi rerank vượt 128 token; cần rút gọn truy vấn")
        start = time.perf_counter()
        scores, truncated = [], 0
        for offset in range(0, len(documents), self.batch_size):
            texts = [passage_body(doc) for doc in documents[offset:offset + self.batch_size]]
            lengths = self.tokenizer([query] * len(texts), texts, truncation=False, padding=False,
                                      return_length=True, verbose=False)["length"]
            truncated += sum(length > self.max_length for length in lengths)
            inputs = self.tokenizer([query] * len(texts), texts, padding=True, truncation="only_second",
                max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                scores.extend(self.model(**inputs).logits.view(-1).float().tolist())
        if len(scores) != len(documents) or not np.isfinite(scores).all():
            raise ValueError("Reranker trả điểm không hợp lệ")
        ordered = sorted(range(len(documents)), key=lambda i: (-scores[i], i))[:top_k]
        results = []
        for row in ordered:
            doc = documents[row].model_copy(deep=True, update={"score": scores[row]})
            previous = doc.metadata.get("retrieval", {}).copy()
            doc.metadata["retrieval"] = {**previous, "method": "hybrid_rerank", "rrf_rank": row + 1,
                "rrf_score": documents[row].score, "rerank_logit": scores[row],
                "reranker": self.identity["repo"], "reranker_revision": self.identity["revision"]}
            results.append(doc)
        self.last_run = {"candidates": len(documents), "truncated_pairs": truncated,
                         "milliseconds": (time.perf_counter() - start) * 1000}
        return results
