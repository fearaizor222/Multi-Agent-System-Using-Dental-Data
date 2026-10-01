"""RAG module providing Dense, Sparse, and Hybrid Retrieval capabilities."""

from src.rag.base import DentalDocument, BaseRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.vector_store import DenseVectorRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.sample_data import load_sample_dental_documents

__all__ = [
    "DentalDocument",
    "BaseRetriever",
    "BM25Retriever",
    "DenseVectorRetriever",
    "HybridRetriever",
    "load_sample_dental_documents",
]
