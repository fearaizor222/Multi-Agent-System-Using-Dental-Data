"""RAG module providing Dense, Sparse, and Hybrid Retrieval capabilities."""

from src.rag.base import DentalDocument, BaseRetriever

try:
    from src.rag.bm25_retriever import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    from src.rag.vector_store import DenseVectorRetriever
except ImportError:
    DenseVectorRetriever = None

try:
    from src.rag.hybrid_retriever import HybridRetriever
except ImportError:
    HybridRetriever = None

try:
    from src.rag.sample_data import load_sample_dental_documents
except ImportError:
    load_sample_dental_documents = None

__all__ = [
    "DentalDocument",
    "BaseRetriever",
    "BM25Retriever",
    "DenseVectorRetriever",
    "HybridRetriever",
    "load_sample_dental_documents",
]
