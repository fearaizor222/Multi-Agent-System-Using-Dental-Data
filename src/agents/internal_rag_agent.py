"""Agent 2: Closed-data RAG Agent — Chuyên trách truy xuất dữ liệu Đóng / Nội bộ (Giáo trình RHM)."""
import json
from pathlib import Path
from typing import List

from src.agents.contracts import Evidence, evidence_bundle
from src.core.logger import get_logger
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import DenseVectorRetriever

from pydantic import BaseModel, Field

logger = get_logger("agents.internal_rag")
ROOT_DIR = Path(__file__).resolve().parents[2]


class InternalSpecialistAssessment(BaseModel):
    """Báo cáo hội chẩn chuyên khoa của Agent 2 (Cắn khớp học & Khớp Thái Dương Hàm)."""
    specialist_role: str = "Bác sĩ Chuyên khoa Cắn khớp học & Khớp Thái Dương Hàm (Kho nội bộ)"
    query_perspective: str = Field(description="Lăng kính chuyên khoa cắn khớp khi phân tích triệu chứng")
    is_applicable: bool = Field(description="Ca bệnh có chỉ định can thiệp cắn khớp/TMJ hay không")
    evidence: List[Evidence] = Field(default_factory=list, description="Danh sách chứng cứ trích dẫn [I]")
    specialist_opinion: str = Field(description="Ý kiến hội chẩn chuyên môn của Bác sĩ Cắn khớp")


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
        """Truy xuất danh sách Evidence (tương thích ngược)."""
        assessment = self.consult(query, top_k=top_k)
        return assessment.evidence

    def consult(self, query: str, top_k: int = 3) -> InternalSpecialistAssessment:
        """Tác tử Bác sĩ Chuyên khoa Cắn khớp & TMJ: Phân tích ca bệnh, truy xuất và hội chẩn chuyên sâu."""
        q_lower = query.lower()
        is_occlusal_related = any(term in q_lower for term in [
            "thái dương hàm", "khớp cắn", "máng nhai", "kêu lục cục", "tiếng click",
            "há miệng", "mỏi hàm", "nghiến răng", "cắn sâu", "cắn chéo", "chấn thương khớp cắn",
            "mài chỉnh", "lồi cầu", "đĩa khớp", "cơ cắn", "cơ thái dương"
        ])

        # 1. Nếu triệu chứng hoàn toàn ngoài phạm vi cắn khớp (sâu răng, tủy, nướu thông thường)
        if not is_occlusal_related and any(term in q_lower for term in ["sâu răng", "răng sâu", "tẩy trắng", "trám răng", "chảy máu chân răng"]):
            logger.info("[Agent 2] Hội chẩn: Ca bệnh không thuộc phạm vi cắn khớp/TMJ, chuyển giao toàn quyền cho Agent 3.")
            return InternalSpecialistAssessment(
                query_perspective="Đánh giá cơ sinh học nhai & rối loạn khớp thái dương hàm (TMJ)",
                is_applicable=False,
                evidence=[],
                specialist_opinion=(
                    "Dưới góc độ Chuyên khoa Cắn khớp: Triệu chứng người bệnh mô tả thuộc về bệnh lý mô cứng hoặc mô nha chu thông thường, "
                    "không xuất phát từ nguyên nhân sai lệch khớp cắn hay chấn thương cơ khớp thái dương hàm. "
                    "Giáo trình Cắn khớp học RHM1 không có chỉ định can thiệp máng nhai hay mài chỉnh cắn cho ca bệnh này."
                )
            )

        # 2. Truy xuất tài liệu cắn khớp học
        hits = self.retriever.retrieve(query, top_k=top_k)
        filtered = []
        for h in hits:
            text = (h.title + " " + h.content).lower()
            if any(term in q_lower for term in ["sâu răng", "răng sâu"]) and "sâu răng" not in text:
                continue
            filtered.append(h)

        if not filtered:
            return InternalSpecialistAssessment(
                query_perspective="Đánh giá cơ sinh học nhai & rối loạn khớp thái dương hàm (TMJ)",
                is_applicable=False,
                evidence=[],
                specialist_opinion="Giáo trình Cắn khớp học không tìm thấy tài liệu phù hợp cho trường hợp này."
            )

        evidence = evidence_bundle(filtered, prefix="I", warehouse="internal")

        # 3. Hội chẩn chuyên khoa cắn khớp dựa trên chứng cứ trích xuất
        if any(k in q_lower for k in ["kêu lục cục", "tiếng click", "thái dương hàm", "há miệng"]):
            opinion = (
                "Hội chẩn Chuyên khoa Cắn khớp & TMJ: Triệu chứng đau và tiếng kêu lục cục khi há miệng là biểu hiện điển hình của "
                "Rối loạn nội khớp Thái dương hàm (tiếng click do lệch đĩa khớp ra trước có hồi phục). "
                "Theo Chương 13 Giáo trình RHM1: Chỉ định can thiệp bảo tồn hàng đầu là làm máng nhai ổn định (stabilization splint) "
                "để giải tỏa lực ép lên lồi cầu và tái lập hướng dẫn cắn khớp thuận lợi; kết hợp bài tập há ngậm miệng có hướng dẫn trước gương (Chương 9) "
                "và tránh ăn nhai thức ăn quá dai hoặc cứng. Tuyệt đối tránh mài chỉnh răng ồ ạt hoặc phẫu thuật xâm lấn ở giai đoạn sớm."
            )
        elif any(k in q_lower for k in ["nghiến răng", "cận chức năng", "mòn răng"]):
            opinion = (
                "Hội chẩn Chuyên khoa Cắn khớp & TMJ: Nghiến răng là hoạt động cận chức năng gây quá tải lực cắn và mòn vỡ men ngà. "
                "Theo Chương 12 Giáo trình RHM1: Cần chỉ định làm máng nhai thư giãn ban đêm để bảo vệ mô cứng răng và giảm co thắt nhóm cơ nhai."
            )
        else:
            titles = ", ".join([e.title for e in evidence[:2]])
            opinion = (
                f"Hội chẩn Chuyên khoa Cắn khớp: Căn cứ tài liệu chuyên sâu ({titles}), "
                "khuyến nghị kiểm tra kỹ các điểm vướng cộm cắn khớp ở tư thế lồng múi tối đa và các vận động trượt hàm ra trước/sang bên để loại trừ chấn thương khớp cắn."
            )

        return InternalSpecialistAssessment(
            query_perspective="Phân tích tương quan cắn khớp, hoạt động cơ nhai và chức năng khớp thái dương hàm (TMJ)",
            is_applicable=True,
            evidence=evidence,
            specialist_opinion=opinion
        )
