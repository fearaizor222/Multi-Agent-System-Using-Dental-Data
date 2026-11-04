"""Agent 2: Dental Retrieval Agent — Quản lý Hybrid RAG (Qdrant Dense Vector + Sparse BM25) cho Kho Nội bộ và Công khai."""
import json
from pathlib import Path
from typing import List, Optional

from src.agents.contracts import Evidence, evidence_bundle
from src.core.logger import get_logger
from src.rag.base import DentalDocument
from src.rag.embedding_store import load_corpus
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.qdrant_store import (
    COLLECTION_INTERNAL,
    COLLECTION_PUBLIC,
    QdrantDentalStore,
    QdrantDenseRetriever,
)

logger = get_logger("agents.retrieval")
ROOT_DIR = Path(__file__).resolve().parents[2]


class DentalRetrievalAgent:
    """Tác tử Truy xuất dữ liệu: Thực hiện tìm kiếm Hybrid RAG trên cả hai kho tri thức bằng Qdrant."""

    def __init__(
        self,
        internal_retriever: Optional[HybridRetriever] = None,
        public_retriever: Optional[HybridRetriever] = None,
        store: Optional[QdrantDentalStore] = None,
    ):
        self.store = store or QdrantDentalStore()
        self.internal_retriever = internal_retriever or self._init_internal_retriever()
        self.public_retriever = public_retriever or self._init_public_retriever()

    def _init_internal_retriever(self) -> HybridRetriever:
        """Khởi tạo Hybrid Retriever cho kho nội bộ (Giáo trình Cắn khớp học RHM + Aging)."""
        documents = []
        try:
            docs, _ = load_corpus("internal")
            documents = docs
            logger.info(f"[RetrievalAgent] Đã tải {len(documents)} chunks từ internal_v1.")
        except Exception as e:
            logger.warning(f"[RetrievalAgent] Fallback sang file json đơn giản: {e}")
            file_path = ROOT_DIR / "data/processed/khop_can_hoc_sections.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_docs = json.load(f)
                for item in raw_docs:
                    doc = DentalDocument.model_validate(item)
                    doc.metadata["warehouse"] = "internal"
                    documents.append(doc)

        logger.info(f"[RetrievalAgent] Kho nội bộ Qdrant sẵn sàng: {len(documents)} tài liệu.")
        dense = QdrantDenseRetriever(collection_name=COLLECTION_INTERNAL, store=self.store)
        return HybridRetriever(documents=documents, dense_retriever=dense)

    def _init_public_retriever(self) -> HybridRetriever:
        """Khởi tạo Hybrid Retriever cho kho công khai (Bộ Y Tế + MSD)."""
        documents = []
        try:
            docs, _ = load_corpus("public")
            clinical_docs = [d for d in docs if d.metadata.get("source") in ("byt", "msd")]
            documents = clinical_docs if clinical_docs else docs
            logger.info(f"[RetrievalAgent] Đã tải {len(documents)} chunks từ public_v1.")
        except Exception as e:
            logger.warning(f"[RetrievalAgent] Fallback sang file json đơn giản: {e}")
            file_path = ROOT_DIR / "data/processed/byt_diseases_sections.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_docs = json.load(f)
                for item in raw_docs:
                    doc = DentalDocument.model_validate(item)
                    doc.metadata["warehouse"] = "public"
                    documents.append(doc)

        logger.info(f"[RetrievalAgent] Kho công khai Qdrant sẵn sàng: {len(documents)} tài liệu.")
        dense = QdrantDenseRetriever(collection_name=COLLECTION_PUBLIC, store=self.store)
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

    def retrieve_all(self, query: str, internal_k: int = 3, public_k: int = 2) -> List[Evidence]:
        """Truy xuất song song từ cả hai kho và hợp nhất danh sách bằng chứng lâm sàng."""
        internal_evidence = self.retrieve_internal(query, top_k=internal_k)
        public_evidence = self.retrieve_public(query, top_k=public_k)
        return internal_evidence + public_evidence
