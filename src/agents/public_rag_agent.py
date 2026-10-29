"""Agent 3: Open-data RAG Agent — Chuyên trách truy xuất dữ liệu Mở / Công khai (Bộ Y Tế, MSD Manual)."""
import json
from pathlib import Path
from typing import List

from src.agents.contracts import Evidence, evidence_bundle
from src.core.logger import get_logger
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import DenseVectorRetriever
from src.agents.internal_rag_agent import ChromaLocalEmbeddings

logger = get_logger("agents.public_rag")
ROOT_DIR = Path(__file__).resolve().parents[2]


class OpenDataRAGAgent:
    """Agent 3: Tác tử RAG Dữ liệu mở (Kho công khai: Hướng dẫn chẩn đoán & điều trị RHM - Bộ Y Tế)."""

    def __init__(self, retriever: HybridRetriever = None):
        self.embeddings = ChromaLocalEmbeddings()
        self.retriever = retriever or self._build_public_retriever()

    def _build_public_retriever(self) -> HybridRetriever:
        file_path = ROOT_DIR / "data/processed/byt_diseases_sections.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy kho tài liệu công khai: {file_path}")

        logger.info(f"[Agent 3] Nạp kho dữ liệu mở (Công khai): {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        documents = []
        for item in raw_docs:
            doc = DentalDocument.model_validate(item)
            doc.metadata["warehouse"] = "public"
            if "parent_id" not in doc.metadata and "document_id" not in doc.metadata:
                doc.metadata["parent_id"] = doc.id
            documents.append(doc)

        logger.info(f"[Agent 3] Đã sẵn sàng {len(documents)} tài liệu công khai.")
        dense = DenseVectorRetriever(documents=documents, embeddings=self.embeddings)
        return HybridRetriever(documents=documents, dense_retriever=dense)

    def retrieve(self, query: str, top_k: int = 3) -> List[Evidence]:
        """Tìm kiếm Hybrid RAG trên kho mở và trả về bằng chứng công khai [P1, P2...]."""
        hits = self.retriever.retrieve(query, top_k=top_k)
        if not hits:
            logger.warning("[Agent 3] Không tìm thấy dữ liệu mở phù hợp.")
            return []
        return evidence_bundle(hits, prefix="P", warehouse="public")
