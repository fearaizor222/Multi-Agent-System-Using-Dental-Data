"""Agent 2: Closed-data RAG Agent — Chuyên trách truy xuất dữ liệu Đóng / Nội bộ (Giáo trình RHM)."""
import json
from pathlib import Path
from typing import List

from src.agents.contracts import Evidence, evidence_bundle
from src.core.logger import get_logger
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import DenseVectorRetriever

logger = get_logger("agents.internal_rag")
ROOT_DIR = Path(__file__).resolve().parents[2]


class ChromaLocalEmbeddings:
    """Embedding local ONNX qua ChromaDB, chạy trực tiếp trên CPU."""
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


class ClosedDataRAGAgent:
    """Agent 2: Tác tử RAG Dữ liệu đóng (Kho nội bộ: Giáo trình Cắn khớp học RHM)."""

    def __init__(self, retriever: HybridRetriever = None):
        self.embeddings = ChromaLocalEmbeddings()
        self.retriever = retriever or self._build_internal_retriever()

    def _build_internal_retriever(self) -> HybridRetriever:
        file_path = ROOT_DIR / "data/processed/khop_can_hoc_sections.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy kho tài liệu nội bộ: {file_path}")

        logger.info(f"[Agent 2] Nạp kho dữ liệu đóng (Nội bộ): {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        documents = []
        for item in raw_docs:
            doc = DentalDocument.model_validate(item)
            doc.metadata["warehouse"] = "internal"
            if "parent_id" not in doc.metadata and "document_id" not in doc.metadata:
                doc.metadata["parent_id"] = doc.id
            documents.append(doc)

        logger.info(f"[Agent 2] Đã sẵn sàng {len(documents)} tài liệu nội bộ.")
        dense = DenseVectorRetriever(documents=documents, embeddings=self.embeddings)
        return HybridRetriever(documents=documents, dense_retriever=dense)

    def retrieve(self, query: str, top_k: int = 3) -> List[Evidence]:
        """Tìm kiếm Hybrid RAG trên kho đóng và trả về bằng chứng y văn nội bộ [I1, I2...]."""
        hits = self.retriever.retrieve(query, top_k=top_k)
        if not hits:
            logger.warning("[Agent 2] Không tìm thấy dữ liệu đóng phù hợp.")
            return []

        # Lọc nhiễu chuyên khoa: Nếu câu hỏi hỏi về các bệnh ngoài phạm vi cắn khớp (như sâu răng, tẩy trắng, nướu...)
        # và tài liệu cắn khớp học không thực sự chứa thực thể đó thì không trả về tài liệu lạc đề (máng nhai/đắp lẹm)
        q_lower = query.lower()
        filtered = []
        for h in hits:
            text = (h.title + " " + h.content).lower()
            if any(term in q_lower for term in ["sâu răng", "răng sâu"]) and "sâu răng" not in text:
                continue
            if any(term in q_lower for term in ["chảy máu chân răng", "viêm nướu", "viêm lợi"]) and not any(k in text for k in ["nướu", "lợi", "nha chu"]):
                continue
            filtered.append(h)

        if not filtered:
            logger.info("[Agent 2] Không có tài liệu cắn khớp học chuyên biệt cho câu hỏi này (chuyển giao cho dữ liệu mở Agent 3).")
            return []

        return evidence_bundle(filtered, prefix="I", warehouse="internal")
