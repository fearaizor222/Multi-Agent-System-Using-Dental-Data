"""Dùng chung một reranker cho hai kho; giữ nguyên bộ lọc của HybridRetriever."""
from src.rag.base import BaseRetriever
from src.rag.hybrid_retriever import HybridRetriever


class RerankedRetriever(BaseRetriever):
    def __init__(self, retriever, reranker, candidate_k=30):
        if candidate_k < 1:
            raise ValueError("candidate_k phải dương")
        self.retriever, self.reranker, self.candidate_k = retriever, reranker, candidate_k

    @classmethod
    def from_disk(cls, directory, reranker, encoder=None, candidate_k=30):
        return cls(HybridRetriever.from_disk(directory, encoder=encoder, candidate_k=candidate_k), reranker, candidate_k)

    def add_documents(self, documents):
        raise ValueError("Index lưu trên đĩa chỉ đọc")

    def retrieve(self, query, top_k=5, source=None, language=None):
        if top_k < 1 or top_k > self.candidate_k:
            raise ValueError("Cần 1 <= top_k <= candidate_k")
        documents = self.retriever.retrieve(query, top_k=self.candidate_k, source=source, language=language)
        return self.reranker.rerank(query, documents, top_k=top_k)
