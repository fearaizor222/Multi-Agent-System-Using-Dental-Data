"""Agent 5: Drafting & Response Synthesis Agent — Soạn thảo bản nháp nhận định và tổng hợp câu trả lời y khoa."""
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("agents.drafting_response")

DRAFTING_SYSTEM_PROMPT = """Bạn là Trợ lý Thông tin Nha khoa C20 AI (Agent 5: Drafting & Consultation Synthesis Agent).
Nhiệm vụ: Căn cứ vào các ý kiến hội chẩn chuyên khoa và tài liệu y khoa được cung cấp, hãy trình bày bài hướng dẫn và tư vấn thông tin y tế chi tiết, chuẩn xác gửi tới người hỏi.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI CHỈ DÙNG TIẾNG VIỆT CHUẨN XÁC, không chêm tiếng nước ngoài.
2. BÁM SÁT TÀI LIỆU VÀ Ý KIẾN HỘI CHẨN ĐƯỢC CẤP: Tuyệt đối không tự bịa đặt hoặc suy diễn các tác dụng phụ, biến chứng hay thuốc lạ không có trong tài liệu (như đau đầu, sưng mắt, chóng mặt, tiêu chảy, kem chống nắng...).
3. Trình bày rõ ràng theo từng phần:
   - Các phương án xử lý/điều trị theo phác đồ (ví dụ: tái khoáng hóa bằng Fluoride ở giai đoạn sớm vs hàn trám phục hồi khi đã có lỗ sâu).
   - Biện pháp vệ sinh, chăm sóc và dự phòng mảng bám tại nhà.
   - Lời khuyên đến phòng khám Răng Hàm Mặt để được bác sĩ thăm khám và thực hiện kỹ thuật."""



class ClaimItem(BaseModel):
    claim_id: str = Field(description="Mã nhận định, ví dụ c1, c2")
    text: str = Field(description="Nội dung nhận định súc tích")
    citations: List[str] = Field(description="Danh sách mã trích dẫn, ví dụ ['I1', 'P1']")


class DraftResponse(BaseModel):
    key_claims: List[ClaimItem] = Field(description="Các nhận định cốt lõi kèm trích dẫn")
    draft_response: str = Field(description="Bài tư vấn hoàn chỉnh cho người bệnh")


class DraftingResponseAgent:
    """Agent 5: Soạn thảo bản nháp nhận định và tổng hợp câu trả lời hoàn chỉnh."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.2, streaming=False)

    def run(
        self,
        question: str,
        internal_evidence: Optional[List[Dict]] = None,
        public_evidence: Optional[List[Dict]] = None,
        vision_findings: Optional[Dict] = None,
        internal_opinion: Optional[str] = None,
        public_opinion: Optional[str] = None
    ) -> DraftResponse:
        """Soạn thảo câu trả lời dựa trên ý kiến hội chẩn chuyên khoa (A2, A3), chứng cứ y văn và thị giác (A4)."""
        evidence_lines = []

        # Ý kiến hội chẩn chuyên khoa của Agent 2 & Agent 3
        opinions = []
        if internal_opinion:
            opinions.append(f"👨‍⚕️ [Ý kiến Chuyên khoa Cắn khớp & TMJ (Agent 2)]:\n{internal_opinion}")
        if public_opinion:
            opinions.append(f"👨‍⚕️ [Ý kiến Chuyên khoa Phác đồ Bộ Y Tế (Agent 3)]:\n{public_opinion}")
        opinions_block = "\n\n".join(opinions) if opinions else ""

        # 1. Bằng chứng kho đóng (Agent 2)
        if internal_evidence:
            for e in internal_evidence:
                title = e.get("title", "").strip()
                content = e.get("content", "").strip()[:900].replace("\n", " ")
                evidence_lines.append(f"- [{e.get('evidence_id')}] (Nội bộ: {title}): {content}")

        # 2. Bằng chứng kho mở (Agent 3)
        if public_evidence:
            for e in public_evidence:
                title = e.get("title", "").strip()
                content = e.get("content", "").strip()[:900].replace("\n", " ")
                evidence_lines.append(f"- [{e.get('evidence_id')}] (Bộ Y Tế: {title}): {content}")

        # 3. Kết quả thị giác (Agent 4)
        if vision_findings:
            vf = vision_findings
            evidence_lines.append(f"- [Thị giác]: {vf.get('preliminary_impression')}")

        clean_context = "\n".join(evidence_lines) if evidence_lines else "(Không có tài liệu chuyên biệt)"

        is_local_model = hasattr(self.llm, "threads") or getattr(self.llm, "_llm_type", "") == "local-qwen2.5-cpu"

        if is_local_model:
            user_prompt = f"""Ý kiến hội chẩn của các Bác sĩ chuyên khoa:
{opinions_block}

Tài liệu chứng cứ trích dẫn:
{clean_context}

Câu hỏi của người bệnh: "{question}"

Dựa vào ý kiến hội chẩn và tài liệu chuyên môn trên, hãy trình bày nội dung tư vấn đầy đủ và chi tiết bằng tiếng Việt:"""
        else:
            user_prompt = f"""Tài liệu y khoa tham khảo:
{clean_context}

Câu hỏi của người bệnh: "{question}"

Bác sĩ hãy tổng hợp các tài liệu trên để viết bài tư vấn nha khoa hoàn chỉnh gửi người bệnh:
1. Nêu 1-3 nhận định cốt lõi (key claims) kèm mã trích dẫn ([I...], [P...]).
2. Trình bày bài tư vấn hoàn chỉnh gửi tới người bệnh (Nhận định sơ bộ, Hướng dẫn chăm sóc, Cảnh báo khi nào cần đi khám).

Chỉ trả về DUY NHẤT một JSON hợp lệ tuân thủ schema:
{{
  "key_claims": [
    {{"claim_id": "c1", "text": "...", "citations": ["I1", "P1"]}}
  ],
  "draft_response": "Nội dung bài tư vấn hoàn chỉnh..."
}}"""


        logger.info("[Agent 5] Đang soạn thảo bản nháp và phản hồi tư vấn...")
        response = self.llm.invoke([
            SystemMessage(content=DRAFTING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        text = response.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            return DraftResponse.model_validate(data)
        except Exception as e:
            logger.info(f"[Agent 5] Định dạng văn bản tự do (Local Model/Fallback): {e}")
            import re
            citations = re.findall(r"\[([IP]\d+)\]", text)
            key_claims = []
            if citations:
                key_claims.append(ClaimItem(
                    claim_id="c1",
                    text="Nhận định lâm sàng dựa trên tài liệu nha khoa trích dẫn",
                    citations=list(dict.fromkeys(citations))
                ))
            elif internal_evidence or public_evidence:
                all_cits = [e.get("evidence_id") for e in (internal_evidence or []) + (public_evidence or []) if e.get("evidence_id")]
                key_claims.append(ClaimItem(
                    claim_id="c1",
                    text="Tổng hợp từ tri thức Giáo trình RHM và Hướng dẫn Bộ Y Tế",
                    citations=all_cits[:3]
                ))

            # Loại bỏ ký tự tiếng Trung nếu mô hình sub-1B lỡ sinh ra
            cleaned_draft = re.sub(r"[\u4e00-\u9fff]+", "", text)

            # Làm sạch nếu còn sót các header markdown
            for marker in ["### DỮ LIỆU ĐÓNG", "### DỮ LIỆU MỞ", "#### [I1]", "#### [P1]"]:
                if marker in cleaned_draft:
                    cleaned_draft = cleaned_draft.replace(marker, "")

            # Chuẩn hóa khoảng trắng
            cleaned_draft = re.sub(r"[ \t]+", " ", cleaned_draft).strip()

            return DraftResponse(
                key_claims=key_claims,
                draft_response=cleaned_draft if cleaned_draft else "Chưa thể tạo bản nháp tư vấn."
            )


