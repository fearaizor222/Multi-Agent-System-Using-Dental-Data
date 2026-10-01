"""Dense Vector Retriever using Embeddings and Vector Store."""

import math
from typing import List, Optional
from src.rag.base import BaseRetriever, DentalDocument
from src.core.llm import get_embeddings
from src.core.logger import get_logger

logger = get_logger("rag.vector_store")


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseVectorRetriever(BaseRetriever):
    """Dense retriever leveraging embedding models and vector similarity."""

    def __init__(self, documents: Optional[List[DentalDocument]] = None):
        self.embeddings = get_embeddings()
        self.documents: List[DentalDocument] = []
        self.vectors: List[List[float]] = []

        if documents:
            self.add_documents(documents)

    def add_documents(self, documents: List[DentalDocument]) -> None:
        """Embed and index dental documents."""
        if not documents:
            return

        texts_to_embed = [
            f"Tiêu đề: {d.title}\nNội dung: {d.content}\nTừ khóa: {', '.join(d.keywords)}"
            for d in documents
        ]

        try:
            new_vectors = self.embeddings.embed_documents(texts_to_embed)
            self.documents.extend(documents)
            self.vectors.extend(new_vectors)
            logger.info(f"Đã vector hoá và thêm {len(documents)} tài liệu vào Dense Index.")
        except Exception as e:
            logger.error(f"Lỗi khi embed documents: {e}")

    def retrieve(self, query: str, top_k: int = 4) -> List[DentalDocument]:
        """Perform dense semantic search via cosine similarity."""
        if not self.documents or not self.vectors:
            return []

        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Lỗi khi embed query: {e}")
            return []

        scored_docs = []
        for doc, vec in zip(self.documents, self.vectors):
            score = cosine_similarity(query_vector, vec)
            doc_copy = doc.model_copy()
            doc_copy.score = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored_docs[:top_k]
