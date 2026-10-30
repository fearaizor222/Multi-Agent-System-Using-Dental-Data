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

from pydantic import BaseModel, Field

logger = get_logger("agents.public_rag")
ROOT_DIR = Path(__file__).resolve().parents[2]


class PublicSpecialistAssessment(BaseModel):
    """Báo cáo hội chẩn chuyên khoa của Agent 3 (Hướng dẫn chẩn đoán & điều trị Bộ Y Tế)."""
    specialist_role: str = "Bác sĩ Chuyên khoa Bệnh học Răng Hàm Mặt — Theo Hướng dẫn Bộ Y Tế (Kho mở)"
    query_perspective: str = Field(description="Lăng kính phác đồ quốc gia & chuẩn hóa quy trình điều trị")
    is_applicable: bool = Field(description="Đã khớp được phác đồ chẩn đoán của Bộ Y Tế hay chưa")
    evidence: List[Evidence] = Field(default_factory=list, description="Danh sách chứng cứ trích dẫn [P]")
    specialist_opinion: str = Field(description="Ý kiến hội chẩn chuyên môn theo Hướng dẫn Bộ Y Tế")


class OpenDataRAGAgent:
    """Agent 3: Tác tử RAG Dữ liệu mở (Kho công khai: Hướng dẫn chẩn đoán & điều trị RHM - Bộ Y Tế)."""

    def __init__(self, retriever: HybridRetriever = None, llm=None):
        self.llm = llm
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
        """Truy xuất danh sách Evidence (tương thích ngược)."""
        assessment = self.consult(query, top_k=top_k)
        return assessment.evidence

    def consult(self, query: str, top_k: int = 3) -> PublicSpecialistAssessment:
        """Tác tử Bác sĩ Chuyên khoa Phác đồ Bộ Y Tế: Đối chiếu hướng dẫn điều trị chuẩn quốc gia và hội chẩn."""
        q_lower = query.lower()
        hits = self.retriever.retrieve(query, top_k=top_k)
        if not hits:
            logger.warning("[Agent 3] Không tìm thấy dữ liệu mở phù hợp.")
            return PublicSpecialistAssessment(
                query_perspective="Đối chiếu 221 Hướng dẫn chẩn đoán và điều trị RHM — Bộ Y Tế",
                is_applicable=False,
                evidence=[],
                specialist_opinion="Chưa tìm thấy hướng dẫn điều trị đặc thù của Bộ Y Tế khớp chính xác với triệu chứng này."
            )

        evidence = evidence_bundle(hits, prefix="P", warehouse="public")

        # Phân tích phác đồ y khoa theo nhóm bệnh lý lâm sàng
        if any(term in q_lower for term in ["sâu răng", "răng sâu", "lỗ sâu"]):
            opinion = (
                "Hội chẩn theo Hướng dẫn Bộ Y Tế (Quyết định 3108/QĐ-BYT - BYT-SEC-009): "
                "Phác đồ điều trị sâu răng chia thành 2 mức độ tổn thương rõ ràng: "
                "(1) Tổn thương sâu răng sớm: Nguyên tắc là tăng cường tái khoáng, ngăn chặn hủy khoáng; "
                "chỉ định liệu pháp Fluor nồng độ cao (gel, véc ni áp tại chỗ) kết hợp chải răng kem Fluor tại nhà. "
                "(2) Tổn thương đã hình thành lỗ sâu: Bắt buộc can thiệp phục hồi mô cứng: "
                "lấy sạch ngà mủn nhiễm khuẩn, sửa soạn đáy và thành lỗ sâu, hàn lót che phủ bảo vệ tủy, "
                "sau đó hàn kín phục hồi mô cứng bằng vật liệu thích hợp (Composite/GIC) và kiểm soát mảng bám định kỳ."
            )
        elif any(term in q_lower for term in ["viêm nướu", "chảy máu nướu", "viêm lợi", "chảy máu chân răng", "nha chu"]):
            opinion = (
                "Hội chẩn theo Hướng dẫn Bộ Y Tế: Phác đồ chuẩn cho bệnh viêm lợi / viêm quanh răng: "
                "Nguyên tắc cốt lõi là loại bỏ toàn bộ nguyên nhân gây viêm cục bộ. Bác sĩ cần lấy sạch cao răng trên và dưới nướu, "
                "bơm rửa dung dịch sát khuẩn Chlorhexidine 0.12%, kiểm soát ổ viêm và hướng dẫn kỹ thuật chải răng đúng cách, dùng chỉ nha khoa."
            )
        elif any(term in q_lower for term in ["thái dương hàm", "khớp thái dương hàm", "kêu lục cục", "há miệng"]):
            opinion = (
                "Hội chẩn theo Hướng dẫn Bộ Y Tế: Đối chiếu hướng dẫn bệnh lý khớp thái dương hàm: "
                "Khuyến cáo thực hiện chẩn đoán hình ảnh chuyên khoa (chụp X-quang Panorama OPG hoặc MRI khớp) "
                "để đánh giá tương quan lồi cầu và diện khớp. Ưu tiên tuyệt đối phác đồ điều trị nội khoa bảo tồn (thuốc giảm đau kháng viêm, "
                "vật lý trị liệu cơ hàm, giải tỏa stress) phối hợp máng nhai chuyên khoa."
            )
        else:
            primary_title = evidence[0].title if evidence else "Hướng dẫn RHM"
            opinion = (
                f"Hội chẩn theo Hướng dẫn Bộ Y Tế (Căn cứ văn bản: {primary_title}): "
                "Cần tuân thủ nghiêm ngặt nguyên tắc vô khuẩn và bảo tồn tối đa mô lành; "
                "thăm khám lâm sàng kết hợp cận lâm sàng trước khi quyết định can thiệp thủ thuật."
            )

        return PublicSpecialistAssessment(
            query_perspective="Đối chiếu Hướng dẫn Chẩn đoán & Điều trị Răng Hàm Mặt — Quyết định Bộ Y Tế",
            is_applicable=True,
            evidence=evidence,
            specialist_opinion=opinion
        )
