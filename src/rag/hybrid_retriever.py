"""Hybrid Retriever implementing Reciprocal Rank Fusion (RRF) between Dense and Sparse retrieval."""

from typing import List, Dict, Optional
from collections import defaultdict
from src.rag.base import BaseRetriever, DentalDocument
from src.rag.bm25_retriever import BM25Retriever
from src.rag.vector_store import DenseVectorRetriever
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("rag.hybrid")


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever combining Dense Vector and Sparse BM25 search via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        documents: Optional[List[DentalDocument]] = None,
        dense_retriever: Optional[DenseVectorRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        rrf_k: int = None
    ):
        self.documents = documents or []
        self.dense_retriever = dense_retriever or DenseVectorRetriever(self.documents)
        self.sparse_retriever = sparse_retriever or BM25Retriever(self.documents)

        self.dense_weight = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.SPARSE_WEIGHT
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K

    def add_documents(self, documents: List[DentalDocument]) -> None:
        """Add documents to both dense and sparse indexes."""
        self.documents.extend(documents)
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)

    def retrieve(self, query: str, top_k: int = None) -> List[DentalDocument]:
        """Perform Hybrid Search using Reciprocal Rank Fusion (RRF)."""
        top_k = top_k or settings.RAG_TOP_K

        # Retrieve candidates from both sources with wider candidate window
        candidate_k = top_k * 3
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_k)

        logger.info(
            f"Hybrid Search cho query: '{query[:40]}...' "
            f"(Dense: {len(dense_results)}, Sparse: {len(sparse_results)})"
        )

        doc_by_id: Dict[str, DentalDocument] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        # 1. Process Dense Ranks
        for rank, doc in enumerate(dense_results, start=1):
            doc_by_id[doc.id] = doc
            rrf_scores[doc.id] += self.dense_weight * (1.0 / (self.rrf_k + rank))

        # 2. Process Sparse Ranks
        for rank, doc in enumerate(sparse_results, start=1):
            doc_by_id[doc.id] = doc
            rrf_scores[doc.id] += self.sparse_weight * (1.0 / (self.rrf_k + rank))

        # 3. Sort by aggregated RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        hybrid_results: List[DentalDocument] = []
        for did in sorted_doc_ids[:top_k]:
            doc_copy = doc_by_id[did].model_copy()
            doc_copy.score = round(rrf_scores[did], 6)
            hybrid_results.append(doc_copy)

        return hybrid_results
