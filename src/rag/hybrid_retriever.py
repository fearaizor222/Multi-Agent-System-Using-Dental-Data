"""Ghép thứ hạng vector và BM25 bằng weighted Reciprocal Rank Fusion."""
import math

from src.core.config import settings
from src.rag.base import BaseRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.vector_store import DenseVectorRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, documents=None, dense_retriever=None, sparse_retriever=None,
                 dense_weight=None, sparse_weight=None, rrf_k=None, candidate_k=30):
        self.documents = list(documents) if documents is not None else []
        self.dense_weight = settings.DENSE_WEIGHT if dense_weight is None else dense_weight
        self.sparse_weight = settings.SPARSE_WEIGHT if sparse_weight is None else sparse_weight
        self.rrf_k = settings.RRF_K if rrf_k is None else rrf_k
        self.candidate_k = candidate_k
        weights = [self.dense_weight, self.sparse_weight]
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Trọng số phải hữu hạn, không âm và có tổng > 0")
        if not isinstance(self.rrf_k, int) or self.rrf_k < 1 or candidate_k < 1:
            raise ValueError("rrf_k và candidate_k phải dương")
        self.dense_retriever = dense_retriever if dense_retriever is not None else DenseVectorRetriever(self.documents)
        self.sparse_retriever = sparse_retriever if sparse_retriever is not None else BM25Retriever(self.documents)
        self._read_only = False

    @classmethod
    def from_disk(cls, directory, encoder=None, **kwargs):
        dense = DenseVectorRetriever.from_disk(directory, encoder=encoder)
        result = cls(documents=dense.documents, dense_retriever=dense,
                     sparse_retriever=BM25Retriever(dense.documents), **kwargs)
        result._read_only = True
        return result

    def add_documents(self, documents):
        if self._read_only:
            raise ValueError("Index lưu trên đĩa chỉ đọc; tạo index mới khi dữ liệu đổi")
        additions = list(documents)
        ids = [doc.id for doc in self.documents + additions]
        if len(ids) != len(set(ids)):
            raise ValueError("Trùng ID chunk")
        self.dense_retriever.add_documents(additions)
        self.sparse_retriever.add_documents(additions)
        self.documents.extend(additions)

    def retrieve(self, query, top_k=None, source=None, language=None):
        top_k = settings.RAG_TOP_K if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k phải dương")
        if not query.strip():
            raise ValueError("Câu hỏi rỗng")
        candidate_k = max(top_k, self.candidate_k)
        options = {"top_k": candidate_k, "source": source, "language": language}
        dense = self.dense_retriever.retrieve(query, **options) if self.dense_weight else []
        sparse = self.sparse_retriever.retrieve(query, **options) if self.sparse_weight else []
        return self.fuse(dense, sparse, top_k=top_k)

    def fuse(self, dense, sparse, top_k):
        """Lưu thứ hạng/điểm từng nhánh; score đầu ra là RRF, không phải xác suất."""
        documents, scores, details = {}, {}, {}
        for method, hits, weight in [("dense", dense, self.dense_weight), ("bm25", sparse, self.sparse_weight)]:
            if not weight:
                continue
            seen = set()
            for rank, doc in enumerate(hits, 1):
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                documents.setdefault(doc.id, doc)
                scores[doc.id] = scores.get(doc.id, 0) + weight / (self.rrf_k + rank)
                details.setdefault(doc.id, {}).update({method + "_rank": rank, method + "_score": doc.score})
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
        results = []
        for doc_id in ranked:
            doc = documents[doc_id].model_copy(deep=True, update={"score": scores[doc_id]})
            doc.metadata["retrieval"] = {"method": "hybrid", "rrf_k": self.rrf_k,
                "dense_weight": self.dense_weight, "bm25_weight": self.sparse_weight,
                "candidate_k": max(top_k, self.candidate_k), **details[doc_id]}
            results.append(doc)
        return results
