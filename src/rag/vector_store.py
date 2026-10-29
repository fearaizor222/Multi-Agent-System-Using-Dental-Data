"""Dense Vector Retriever using Embeddings and Vector Store."""

import math
from typing import List, Optional
from src.rag.base import BaseRetriever, DentalDocument
from src.core.llm import get_embeddings
from src.core.logger import get_logger
from src.rag.local_embeddings import passage_body

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

    def __init__(self, documents: Optional[List[DentalDocument]] = None, embeddings=None):
        self._disk_index = None
        self.embeddings = embeddings if embeddings is not None else get_embeddings()
        self.documents: List[DentalDocument] = []
        self.vectors: List[List[float]] = []

        if documents:
            self.add_documents(documents)

    @classmethod
    def from_disk(cls, directory, encoder=None):
        from src.rag.embedding_store import DenseIndex
        retriever = cls.__new__(cls)
        retriever._disk_index = DenseIndex(directory, encoder=encoder)
        retriever.documents = retriever._disk_index.documents
        retriever.vectors = retriever._disk_index.vectors
        retriever.embeddings = None
        return retriever

    def add_documents(self, documents: List[DentalDocument]) -> None:
        """Embed and index dental documents."""
        if not documents:
            return
        if self._disk_index is not None:
            raise ValueError("Index lưu trên đĩa chỉ đọc; chạy embed_corpora để tạo phiên bản mới")

        texts_to_embed = [passage_body(d) for d in documents]

        try:
            new_vectors = self.embeddings.embed_documents(texts_to_embed)
            self.documents.extend(documents)
            self.vectors.extend(new_vectors)
            logger.info(f"Đã vector hoá và thêm {len(documents)} tài liệu vào Dense Index.")
        except Exception as e:
            raise RuntimeError("Không tạo được embedding cho tài liệu") from e

    def retrieve(self, query: str, top_k: int = 4, source=None, language=None) -> List[DentalDocument]:
        """Perform dense semantic search via cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k phải dương")
        if not query.strip():
            raise ValueError("Câu hỏi rỗng")
        if self._disk_index is not None:
            return self._disk_index.search(query, top_k=top_k, source=source, language=language)
        if not self.documents or not self.vectors:
            return []

        try:
            query_vector = self.embeddings.embed_query(query)
        except Exception as e:
            raise RuntimeError("Không tạo được embedding cho câu hỏi") from e

        scored_docs = []
        for doc, vec in zip(self.documents, self.vectors):
            if (source is not None and doc.metadata.get("source") != source) or (language is not None and doc.metadata.get("language") != language):
                continue
            score = cosine_similarity(query_vector, vec)
            doc_copy = doc.model_copy(deep=True)
            doc_copy.score = float(score)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored_docs[:top_k]
