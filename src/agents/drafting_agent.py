"""Agent 5: Drafting & Response Synthesis Agent — Soạn thảo bản nháp nhận định và tổng hợp câu trả lời y khoa."""
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("agents.drafting_response")

DRAFTING_SYSTEM_PROMPT = """Bạn là Bác sĩ Nha khoa C20 AI (Agent 5: Drafting & Consultation Synthesis Agent).
Nhiệm vụ: Căn cứ vào các tài liệu y khoa tham khảo được cung cấp, hãy viết bài tư vấn chuyên môn rõ ràng, chuẩn xác y khoa gửi người bệnh.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI CHỈ DÙNG TIẾNG VIỆT CHUẨN XÁC, không chêm tiếng Trung hay tiếng nước ngoài, không bịa đặt những chi tiết ngoài tài liệu (như kem chống nắng, tiêu chảy, thuốc lạ).
2. Trả lời đúng trọng tâm câu hỏi của người bệnh (xưng 'Bác sĩ' và 'bạn'):
   - Nêu rõ các phương án chẩn đoán / điều trị phù hợp theo đúng phác đồ y khoa trong tài liệu.
   - Hướng dẫn biện pháp chăm sóc, vệ sinh răng miệng và kiểm soát mảng bám tại nhà.
   - Khuyến cáo khi nào cần đến cơ sở y tế / phòng khám Răng Hàm Mặt để được bác sĩ can thiệp chuyên môn."""



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
        vision_findings: Optional[Dict] = None
    ) -> DraftResponse:
        """Soạn thảo câu trả lời dựa trên kho đóng (A2), kho mở (A3) và thị giác (A4)."""
        evidence_lines = []

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
            user_prompt = f"""Tài liệu y khoa tham khảo:
{clean_context}

Câu hỏi của người bệnh: "{question}"

Bác sĩ nha khoa hãy tư vấn rõ các phương án chẩn đoán, điều trị cụ thể và lời khuyên chăm sóc cho người bệnh:"""
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


