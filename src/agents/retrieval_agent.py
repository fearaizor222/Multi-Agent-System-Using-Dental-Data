"""Agent 2: Dental Retrieval Agent — Quản lý Hybrid RAG (Dense Vector + Sparse BM25) cho Kho Nội bộ và Công khai."""
import json
from pathlib import Path
from typing import List, Optional

from src.agents.contracts import Evidence, evidence_bundle
from src.core.logger import get_logger
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import DenseVectorRetriever

logger = get_logger("agents.retrieval")

ROOT_DIR = Path(__file__).resolve().parents[2]


class ChromaLocalEmbeddings:
    """Embedding local ONNX all-MiniLM-L6-v2 qua Chroma, không tốn quota API và chạy trực tiếp trên CPU."""

    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self.fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            results.extend(self.fn(texts[i:i + batch_size]))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.fn([text])[0]


class DentalRetrievalAgent:
    """Tác tử Truy xuất dữ liệu: Thực hiện tìm kiếm Hybrid RAG trên cả hai kho tri thức."""

    def __init__(self, internal_retriever=None, public_retriever=None, local_embedding=True):
        self.embeddings = ChromaLocalEmbeddings() if local_embedding else None
        self.internal_retriever = internal_retriever or self._init_internal_retriever()
        self.public_retriever = public_retriever or self._init_public_retriever()

    def _init_internal_retriever(self) -> HybridRetriever:
        """Khởi tạo Hybrid Retriever cho kho nội bộ (Giáo trình Cắn khớp học RHM)."""
        file_path = ROOT_DIR / "data/processed/khop_can_hoc_sections.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file kho nội bộ: {file_path}")

        logger.info(f"Đang tải kho tài liệu nội bộ: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        documents = []
        for item in raw_docs:
            doc = DentalDocument.model_validate(item)
            doc.metadata["warehouse"] = "internal"
            if "parent_id" not in doc.metadata and "document_id" not in doc.metadata:
                doc.metadata["parent_id"] = doc.id
            documents.append(doc)

        logger.info(f"Kho nội bộ sẵn sàng: {len(documents)} tài liệu.")
        dense = DenseVectorRetriever(documents=documents, embeddings=self.embeddings)
        return HybridRetriever(documents=documents, dense_retriever=dense)

    def _init_public_retriever(self) -> HybridRetriever:
        """Khởi tạo Hybrid Retriever cho kho công khai (Bộ Y Tế - Hướng dẫn điều trị RHM)."""
        file_path = ROOT_DIR / "data/processed/byt_diseases_sections.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file kho công khai: {file_path}")

        logger.info(f"Đang tải kho tài liệu công khai: {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        documents = []
        for item in raw_docs:
            doc = DentalDocument.model_validate(item)
            doc.metadata["warehouse"] = "public"
            if "parent_id" not in doc.metadata and "document_id" not in doc.metadata:
                doc.metadata["parent_id"] = doc.id
            documents.append(doc)

        logger.info(f"Kho công khai sẵn sàng: {len(documents)} tài liệu.")
        dense = DenseVectorRetriever(documents=documents, embeddings=self.embeddings)
        return HybridRetriever(documents=documents, dense_retriever=dense)

    def retrieve_internal(self, query: str, top_k: int = 3) -> List[Evidence]:
        """Truy xuất tài liệu từ kho nội bộ và đóng gói thành Evidence bundle với prefix 'I'."""
        hits = self.internal_retriever.retrieve(query, top_k=top_k)
        if not hits:
            logger.warning("Không tìm thấy tài liệu phù hợp trong kho nội bộ.")
            return []
        return evidence_bundle(hits, prefix="I", warehouse="internal")

    def retrieve_public(self, query: str, top_k: int = 2) -> List[Evidence]:
        """Truy xuất tài liệu từ kho công khai và đóng gói thành Evidence bundle với prefix 'P'."""
        hits = self.public_retriever.retrieve(query, top_k=top_k)
        if not hits:
            logger.warning("Không tìm thấy tài liệu phù hợp trong kho công khai.")
            return []
        return evidence_bundle(hits, prefix="P", warehouse="public")
