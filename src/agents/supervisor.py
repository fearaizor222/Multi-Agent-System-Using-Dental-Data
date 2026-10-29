"""Agent 1: Supervisor Agent — Tiếp nhận, phân loại ý định, Triage cấp cứu nha khoa và điều phối 5 Agent."""
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("agents.supervisor")

SUPERVISOR_SYSTEM_PROMPT = """Bạn là Agent 1: Tác tử Điều phối (Supervisor & Clinical Triage Agent) thuộc Hệ thống Nha khoa 5 Agent C20.
Cấu trúc hệ thống gồm 5 tác tử:
- Agent 1: Supervisor (Điều phối & Triage)
- Agent 2: Closed-data RAG (Dữ liệu đóng / Giáo trình nội bộ)
- Agent 3: Open-data RAG (Dữ liệu mở / Hướng dẫn Bộ Y Tế & MSD)
- Agent 4: Dental Vision (Phân tích ảnh lâm sàng & X-quang)
- Agent 5: Drafting Response (Tạo bản nháp & Tổng hợp tư vấn)

Nhiệm vụ của bạn:
1. Đánh giá dấu hiệu cấp cứu nha khoa (Triage Evaluation):
   - Cờ đỏ cấp cứu: Sưng tấy vùng mặt/hàm lan rộng, sốt cao kèm cứng khít hàm (không há được miệng), khó nuốt/khó thở do phù nề, chảy máu chân răng/miệng ồ ạt không cầm, chấn thương hàm mặt nặng (gãy vỡ răng lộ tủy cấp, trật khớp thái dương hàm).
   - Nếu có cờ đỏ: Đặt is_emergency = True, intent = "emergency_triage".
2. Nếu có đường dẫn hình ảnh (image_path):
   - Đặt intent = "vision_analysis" để kích hoạt Agent 4 (Thị giác).
3. Nếu chỉ hỏi thông tin hành chính, giờ khám, đặt hẹn, giá:
   - Đặt intent = "service_info".
4. Nếu là câu hỏi chuyên môn y khoa thông thường:
   - Đặt intent = "clinical_medical".

Chỉ trả về DUY NHẤT một JSON hợp lệ theo schema."""


class TriageDecision(BaseModel):
    is_emergency: bool = Field(description="True nếu có dấu hiệu cấp cứu nha khoa")
    emergency_level: Literal["critical", "urgent", "routine"] = Field(description="Mức độ khẩn cấp")
    intent: Literal["emergency_triage", "vision_analysis", "clinical_medical", "service_info"] = Field(description="Ý định xử lý")
    triage_reason: str = Field(description="Lý do phân loại ngắn gọn")
    recommended_action: str = Field(description="Hành động tức thời cho người bệnh")


class SupervisorAgent:
    """Agent 1: Phân tích câu hỏi, đánh giá cờ đỏ cấp cứu và điều phối luồng."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0, streaming=False)

    def evaluate(self, question: str, image_path: Optional[str] = None) -> TriageDecision:
        """Đánh giá intent và phân loại cấp cứu."""
        # Fast path kiểm tra cấp cứu
        critical_keywords = [
            "khó thở", "khó nuốt", "sốt cao", "sưng to cả mặt", "chảy máu không cầm",
            "gãy xương hàm", "chấn thương hàm mặt", "rách mô hở tủy"
        ]
        has_critical_sign = any(kw in question.lower() for kw in critical_keywords)

        if has_critical_sign and ("sưng" in question.lower() or "sốt" in question.lower() or "chảy máu" in question.lower()):
            logger.warning(f"Phát hiện dấu hiệu cấp cứu nha khoa trực tiếp: {question}")
            return TriageDecision(
                is_emergency=True,
                emergency_level="critical",
                intent="emergency_triage",
                triage_reason="Phát hiện triệu chứng cấp cứu sưng viêm lan tỏa hoặc chấn thương có nguy cơ biến chứng đường thở/nhiễm trùng huyết.",
                recommended_action="Đến ngay Khoa Cấp cứu hoặc Bệnh viện Răng Hàm Mặt gần nhất để được can thiệp y tế tức thời."
            )

        if image_path:
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="vision_analysis",
                triage_reason="Người bệnh có gửi kèm hình ảnh nha khoa, cần kích hoạt Agent 4 (Thị giác).",
                recommended_action="Chuyển hình ảnh cho Agent 4 (Dental Vision) phân tích."
            )

        # Fast path kiểm tra câu hỏi dịch vụ / hành chính
        service_keywords = [
            "giá", "chi phí", "bao nhiêu tiền", "giờ làm việc", "mấy giờ", "mở cửa",
            "địa chỉ", "ở đâu", "đặt lịch", "hẹn khám", "chủ nhật", "thứ 7", "bảo hiểm"
        ]
        if any(kw in question.lower() for kw in service_keywords) and not any(kw in question.lower() for kw in ["đau", "nhức", "sưng", "chảy máu", "tủy", "mòn", "ê buốt"]):
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="service_info",
                triage_reason="Câu hỏi liên quan đến chi phí, lịch làm việc hoặc thủ tục hành chính.",
                recommended_action="Cung cấp thông tin dịch vụ, bảng giá và hướng dẫn đặt lịch."
            )

        # Với mô hình local CPU, ưu tiên phân luồng nhanh để tiết kiệm 8-10s suy luận JSON
        is_local_model = hasattr(self.llm, "threads") or getattr(self.llm, "_llm_type", "") == "local-qwen2.5-cpu"
        if is_local_model:
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="clinical_medical",
                triage_reason="Câu hỏi bệnh lý nha khoa lâm sàng được phân loại tự động bởi Supervisor.",
                recommended_action="Chuyển tiếp đến Agent 2 (Kho nội bộ) và Agent 3 (Kho công khai) để truy xuất tài liệu."
            )

        user_prompt = f"Câu hỏi của bệnh nhân: \"{question}\"\nHãy phân loại ý định và đánh giá cấp cứu."
        try:
            response = self.llm.invoke([
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text) if text else {}
            if not isinstance(parsed, dict):
                parsed = {}

            # Chuẩn hóa an toàn các trường để tránh lỗi Pydantic
            valid_intents = ("emergency_triage", "vision_analysis", "clinical_medical", "service_info")
            intent = parsed.get("intent") if parsed.get("intent") in valid_intents else "clinical_medical"
            valid_levels = ("critical", "urgent", "routine")
            emergency_level = parsed.get("emergency_level") if parsed.get("emergency_level") in valid_levels else "routine"

            return TriageDecision(
                is_emergency=bool(parsed.get("is_emergency", False)),
                emergency_level=emergency_level,
                intent=intent,
                triage_reason=parsed.get("triage_reason") or "Câu hỏi bệnh lý nha khoa thông thường.",
                recommended_action=parsed.get("recommended_action") or "Chuyển tiếp đến Agent 2 và Agent 3 truy xuất tri thức."
            )
        except Exception:
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="clinical_medical",
                triage_reason="Câu hỏi bệnh lý nha khoa thông thường.",
                recommended_action="Chuyển tiếp đến Agent 2 và Agent 3 truy xuất tri thức."
            )

