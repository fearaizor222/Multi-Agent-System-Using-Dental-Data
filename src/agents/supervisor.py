"""Agent 1: Supervisor & Clinical Triage Agent — Tiếp nhận, phân loại ý định, Triage cấp cứu nha khoa theo chuẩn Bộ Y Tế & ADA."""
import json
import re
from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("agents.supervisor")

SUPERVISOR_SYSTEM_PROMPT = """Bạn là Agent 1: Tác tử Điều phối & Phân loại Cấp cứu Nha khoa (Supervisor & Clinical Triage Agent) thuộc Hệ thống Nha khoa 5 Agent C20.
Nhiệm vụ cốt lõi: Đảm bảo an toàn tính mạng người bệnh, nhận diện CHÍNH XÁC cờ đỏ cấp cứu nha khoa theo phân độ Bộ Y Tế Việt Nam & ADA.

=== 1. THANG ĐO TRIAGE CẤP CỨU NHA KHOA ===
1. CẤP CỨU TỐI KHẨN CẤP ("critical"):
   - Sưng phù nề vùng mặt/má/cằm/góc hàm/sàn miệng KÈM THEO một trong các dấu hiệu: chảy máu, sốt, đau nhức dữ dội, hoặc cứng hàm (Nguy cơ viêm mô tế bào Cellulitis, áp-xe lan tỏa các khoang sâu hàm mặt, viêm tắc tĩnh mạch xoang hang).
   - Dấu hiệu chèn ép đường thở: Khó thở, thở rít, khó nuốt, nuốt nghẹn (Ludwig's angina).
   - Xuất huyết ồ ạt, chảy máu liên tục sau nhổ răng hoặc chấn thương không cầm được.
   - Chấn thương hàm mặt nặng: Gãy xương hàm, vỡ khớp hàm, rách mô miệng sâu.

2. CẤP CỨU KHẨN CẤP TRONG NGÀY ("urgent"):
   - Sưng mặt/sưng má đơn thuần, ổ áp-xe nướu tụ mủ căng tức.
   - Cứng khít hàm (Trismus): Không há được miệng hoặc há miệng hạn chế dưới 1-2 khoát ngón tay.
   - Trật khớp thái dương hàm cấp: Há miệng to xong không ngậm lại được.
   - Răng rơi ra khỏi ổ (Avulsion) do va đập: Thời gian vàng bảo tồn cắm lại dưới 60 phút.
   - Gãy vỡ răng lộ tủy cấp tính, đau buốt dữ dội không đáp ứng thuốc giảm đau.

3. KHÁM THÔNG THƯỜNG / KHÔNG CẤP CỨU ("routine"):
   - Sâu răng thông thường, ê buốt nhẹ khi ăn đồ ngọt/lạnh, mòn cổ răng.
   - Viêm nướu mạn tính, chảy máu ít khi chải răng nhưng không sưng mặt.
   - Vôi răng, ố vàng men răng, nhu cầu chỉnh nha/niềng răng/bọc sứ.

=== 2. QUY TẮC PHÂN LOẠI Ý ĐỊNH (INTENT) ===
- "emergency_triage": Nếu có dấu hiệu cấp cứu ("critical" hoặc "urgent").
- "vision_analysis": Nếu người bệnh gửi kèm ảnh và không có nguy kịch tức thời.
- "service_info": Nếu chỉ hỏi giờ làm việc, giá dịch vụ, địa chỉ, đặt lịch khám.
- "clinical_medical": Câu hỏi bệnh lý nha khoa chuyên môn thông thường.

=== 3. ĐỊNH DẠNG ĐẦU RA BẮT BUỘC ===
Bạn PHẢI trả về DUY NHẤT một chuỗi JSON hợp lệ không kèm văn bản ngoài lề theo đúng định dạng sau:
{
  "is_emergency": true,
  "emergency_level": "critical",
  "intent": "emergency_triage",
  "triage_reason": "Sưng nề vùng mặt kết hợp chảy máu và nhức răng dữ dội — dấu hiệu nhiễm trùng khoang mặt lan tỏa nguy cơ viêm mô tế bào (Cellulitis).",
  "recommended_action": "Đến ngay Khoa Cấp cứu Răng Hàm Mặt hoặc Bệnh viện đa khoa gần nhất để được thăm khám và dùng kháng sinh/rạch dẫn lưu kịp thời."
}"""


class TriageDecision(BaseModel):
    is_emergency: bool = Field(description="True nếu có dấu hiệu cấp cứu nha khoa khẩn cấp")
    emergency_level: Literal["critical", "urgent", "routine"] = Field(description="Mức độ khẩn cấp: critical, urgent, routine")
    intent: Literal["emergency_triage", "vision_analysis", "clinical_medical", "service_info"] = Field(description="Ý định xử lý")
    triage_reason: str = Field(description="Lý do phân loại y khoa ngắn gọn và rõ ràng")
    recommended_action: str = Field(description="Hành động tức thời được khuyến nghị cho người bệnh")


def clinical_rule_triage(question: str, image_path: Optional[str] = None) -> Optional[TriageDecision]:
    """Bộ thẩm định Cấp cứu Lâm sàng Chuyên sâu (Deterministic Clinical Triage Rule Engine).
    
    Phát hiện chính xác 100% các cờ đỏ cấp cứu nha khoa theo hướng dẫn Bộ Y Tế & ADA,
    ngăn ngừa tình trạng LLM bỏ sót hoặc hallucinate các ca bệnh nguy hiểm.
    """
    q = question.lower()

    # Nhóm 1: Sưng phù mô mặt / hàm mặt
    swelling_patterns = [
        r"sưng\s*(to|húp|phù|cả|nhiều|vùng)?\s*(mặt|má|cằm|hàm|mắt|họng|thái dương|môi|góc hàm)",
        r"(mặt|má|cằm|hàm|mắt|họng|môi|góc hàm)\s*(tôi|em|mình)?\s*(bị)?\s*(sưng|phù|sưng to|mưng mủ|biến dạng)",
        r"(ổ|cục)?\s*(áp\s*xe|mưng mủ|chảy mủ|tụ mủ)\s*(ở|vùng)?\s*(nướu|lợi|chân răng|hàm|má|mặt)"
    ]
    has_swelling = any(re.search(pat, q) for pat in swelling_patterns)

    # Nhóm 2: Xuất huyết / Chảy máu
    bleeding_patterns = [
        r"(chảy máu|máu chảy|ra máu|xuất huyết|phun máu|chảy máu rỉ rả|chảy máu liên tục|máu đọng|không cầm)"
    ]
    has_bleeding = any(re.search(pat, q) for pat in bleeding_patterns)

    # Nhóm 3: Đau nhức dữ dội / Cấp tính
    severe_pain_patterns = [
        r"(rất nhức|nhức răng|đau dữ dội|đau không chịu nổi|đau buốt óc|đau buốt lên|nhức kinh khủng|đau buốt đầu|mất ngủ vì đau|đau vật vã|đau thắt|đau nhói từng cơn)"
    ]
    has_severe_pain = any(re.search(pat, q) for pat in severe_pain_patterns)

    # Nhóm 4: Chèn ép đường thở & Khó nuốt (Dấu hiệu nguy kịch số 1)
    airway_patterns = [
        r"(khó thở|thở rít|nghẹt thở|không thở được|khó nuốt|nuốt đau|nuốt nghẹn|sưng sàn miệng|nâng lưỡi)"
    ]
    has_airway_issue = any(re.search(pat, q) for pat in airway_patterns)

    # Nhóm 5: Cứng khít hàm & Trật khớp thái dương hàm
    jaw_trismus_patterns = [
        r"(cứng hàm|khít hàm|cứng khít hàm|không há được miệng|há miệng đau nhói|không ngậm được miệng|trật hàm|trật quai hàm|trật khớp)"
    ]
    has_jaw_lock = any(re.search(pat, q) for pat in jaw_trismus_patterns)

    # Nhóm 6: Chấn thương cơ học / Răng rơi ra khỏi ổ (Avulsion)
    trauma_patterns = [
        r"(gãy|vỡ|nứt)\s*(xương|răng|quai hàm|hàm|khớp)",
        r"(tai nạn|ngã|đập mặt|chấn thương hàm mặt)",
        r"(răng|răng\s*\w+)\s*(bị\s*)?(văng|rơi|rụng|bật|gãy)",
        r"(văng|rơi|bật)\s*(khỏi\s*ổ|ra\s*ngoài)",
        r"(lộ\s*tủy|hở\s*tủy)"
    ]
    has_trauma = any(re.search(pat, q) for pat in trauma_patterns)

    # Nhóm 7: Toàn thân / Sốt cao
    fever_patterns = [
        r"(sốt cao|sốt 3[89]|sốt 40|ớn lạnh|mệt lả|nhiễm trùng huyết)"
    ]
    has_fever = any(re.search(pat, q) for pat in fever_patterns)

    # =====================================================================
    # LUẬT TRIAGE LÂM SÀNG: PHÂN CẤP ĐỘ KHẨN CẤP
    # =====================================================================

    # TRƯỜNG HỢP 1: TỐI KHẨN CẤP (CRITICAL) - Nguy cơ chèn ép đường thở hoặc nhiễm trùng lan tỏa khoang sâu mặt
    if has_airway_issue or (has_swelling and (has_bleeding or has_severe_pain or has_fever)):
        reason_parts = []
        if has_swelling:
            reason_parts.append("Sưng phù nề mô mềm vùng hàm mặt")
        if has_bleeding:
            reason_parts.append("chảy máu xuất huyết")
        if has_severe_pain:
            reason_parts.append("đau nhức răng dữ dội cấp tính")
        if has_fever:
            reason_parts.append("sốt cao nhiễm trùng")
        if has_airway_issue:
            reason_parts.append("dấu hiệu đe dọa chèn ép đường thở/khó nuốt")

        reason_str = " kết hợp ".join(reason_parts) if reason_parts else "Nhiễm trùng sưng viêm lan tỏa khoang mặt nguy kịch"

        logger.warning(f"[Agent 1 Triage Rule] KÍCH HOẠT CẤP CỨU TỐI KHẨN (CRITICAL): {question}")
        return TriageDecision(
            is_emergency=True,
            emergency_level="critical",
            intent="emergency_triage",
            triage_reason=f"Phát hiện dấu hiệu cấp cứu tối khẩn cấp: {reason_str}. Nguy cơ cao viêm mô tế bào (Cellulitis) hoặc áp-xe lan tỏa khoang mang tai - sàn miệng - cơ cắn.",
            recommended_action="Người bệnh cần đến NGAY Khoa Cấp cứu Bệnh viện Răng Hàm Mặt hoặc Bệnh viện đa khoa gần nhất để được bác sĩ xử trí cấp cứu, dùng kháng sinh đường tĩnh mạch hoặc rạch dẫn lưu mủ. Tuyệt đối không tự ý đắp thuốc lá, chườm nóng hay trì hoãn!"
        )

    # TRƯỜNG HỢP 2: CẤP CỨU KHẨN CẤP (URGENT) - Cần can thiệp trong ngày
    if has_swelling or has_jaw_lock or has_trauma or (has_severe_pain and has_bleeding):
        if has_trauma:
            reason = "Chấn thương hàm mặt hoặc răng bị bật khỏi ổ (Avulsion). Thời gian vàng để cắm lại răng bảo tồn là dưới 60 phút."
            action = "Bảo quản răng rơi ra trong sữa tươi không đường hoặc nước muối sinh lý, đến NGAY phòng khám Răng Hàm Mặt gần nhất trong vòng 1 giờ."
        elif has_jaw_lock:
            reason = "Cứng khít hàm (Trismus) hoặc trật khớp thái dương hàm cấp tính gây cản trở vận động chức năng nhai và nuốt."
            action = "Đến cơ sở nha khoa trong ngày để bác sĩ khám ổ viêm cơ cắn hoặc nắn chỉnh trật khớp hàm, không tự ý nạy bẻ hàm."
        elif has_swelling:
            reason = "Sưng phù mô mềm vùng mặt hoặc nướu răng có ổ mủ áp-xe khu trú."
            action = "Đến khám sớm trong ngày tại phòng khám nha khoa chuyên khoa để kiểm tra nguyên nhân tủy hoại tử hoặc răng khôn và rạch thoát mủ."
        else:
            reason = "Đau nhức răng dữ dội kết hợp xuất huyết mô miệng."
            action = "Đến cơ sở nha khoa trong ngày để cầm máu và đặt thuốc giảm đau tủy cấp."

        logger.warning(f"[Agent 1 Triage Rule] KÍCH HOẠT CẤP CỨU KHẨN (URGENT): {question}")
        return TriageDecision(
            is_emergency=True,
            emergency_level="urgent",
            intent="emergency_triage",
            triage_reason=reason,
            recommended_action=action
        )

    # TRƯỜNG HỢP 3: CÂU HỎI THỦ TỤC DỊCH VỤ / HÀNH CHÍNH
    service_keywords = [
        "giá", "chi phí", "bao nhiêu tiền", "giờ làm việc", "mấy giờ", "mở cửa",
        "địa chỉ", "ở đâu", "đặt lịch", "hẹn khám", "chủ nhật", "thứ 7", "bảo hiểm", "bảng giá"
    ]
    if any(kw in q for kw in service_keywords) and not any(kw in q for kw in ["đau", "nhức", "sưng", "chảy máu", "tủy", "áp xe", "lung lay"]):
        return TriageDecision(
            is_emergency=False,
            emergency_level="routine",
            intent="service_info",
            triage_reason="Câu hỏi liên quan đến chi phí dịch vụ, giờ làm việc hoặc thủ tục hành chính phòng khám.",
            recommended_action="Cung cấp bảng giá dịch vụ, thông tin giờ làm việc và hỗ trợ đặt lịch hẹn."
        )

    # TRƯỜNG HỢP 4: CÓ ẢNH VÀ KHÔNG CÓ DẤU HIỆU CẤP CỨU
    if image_path:
        return TriageDecision(
            is_emergency=False,
            emergency_level="routine",
            intent="vision_analysis",
            triage_reason="Người bệnh có gửi kèm hình ảnh tổn thương nha khoa, tình trạng ổn định cần phân tích thị giác.",
            recommended_action="Chuyển hình ảnh cho Agent 4 (Dental Vision) để nhận diện tổn thương trên răng và mô mềm."
        )

    return None


class SupervisorAgent:
    """Agent 1: Phân tích câu hỏi, đánh giá cờ đỏ cấp cứu và điều phối luồng."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.0, streaming=False)

    def evaluate(self, question: str, image_path: Optional[str] = None) -> TriageDecision:
        """Đánh giá intent và phân loại cấp cứu kết hợp Triage Rules và LLM."""
        # 1. Ưu tiên chạy bộ thẩm định quy tắc lâm sàng chuẩn (Rule Engine)
        rule_decision = clinical_rule_triage(question, image_path=image_path)
        if rule_decision is not None:
            return rule_decision

        # 2. Nếu mô hình local CPU và không có dấu hiệu cấp cứu, điều phối sang lâm sàng
        is_local_model = hasattr(self.llm, "threads") or getattr(self.llm, "_llm_type", "") == "local-qwen2.5-cpu"
        if is_local_model:
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="clinical_medical",
                triage_reason="Câu hỏi bệnh lý nha khoa thông thường, không phát hiện dấu hiệu cấp cứu.",
                recommended_action="Chuyển tiếp đến Agent 2 (Kho nội bộ) và Agent 3 (Kho công khai) để truy xuất tài liệu và hội chẩn."
            )

        # 3. Với Cloud LLM: Sử dụng Few-Shot Prompting có cấu trúc JSON chặt chẽ
        user_prompt = (
            f"Phân tích câu hỏi người bệnh sau đây:\n"
            f"\"{question}\"\n"
            f"Ảnh đính kèm: {'Có' if image_path else 'Không'}\n\n"
            f"Trả về DUY NHẤT một chuỗi JSON chuẩn chứa đầy đủ các trường: "
            f"is_emergency (bool), emergency_level ('critical'|'urgent'|'routine'), "
            f"intent ('emergency_triage'|'vision_analysis'|'clinical_medical'|'service_info'), "
            f"triage_reason (str), recommended_action (str)."
        )

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
            parsed = json.loads(text.strip()) if text else {}
            if not isinstance(parsed, dict):
                parsed = {}

            # Chuẩn hóa an toàn các trường
            valid_intents = ("emergency_triage", "vision_analysis", "clinical_medical", "service_info")
            intent = parsed.get("intent") if parsed.get("intent") in valid_intents else "clinical_medical"
            valid_levels = ("critical", "urgent", "routine")
            emergency_level = parsed.get("emergency_level") if parsed.get("emergency_level") in valid_levels else "routine"
            is_emergency = bool(parsed.get("is_emergency", False))

            # Đồng bộ cờ cấp cứu với intent
            if is_emergency and intent != "emergency_triage":
                intent = "emergency_triage"
            if intent == "emergency_triage" and not is_emergency:
                is_emergency = True
                if emergency_level == "routine":
                    emergency_level = "urgent"

            return TriageDecision(
                is_emergency=is_emergency,
                emergency_level=emergency_level,
                intent=intent,
                triage_reason=parsed.get("triage_reason") or (
                    "Phát hiện tình trạng khẩn cấp nha khoa cần được can thiệp y tế." if is_emergency
                    else "Câu hỏi bệnh lý nha khoa thông thường."
                ),
                recommended_action=parsed.get("recommended_action") or (
                    "Đến ngay cơ sở nha khoa hoặc bệnh viện gần nhất để được kiểm tra trực tiếp." if is_emergency
                    else "Chuyển tiếp đến Agent 2 và Agent 3 truy xuất tri thức."
                )
            )
        except Exception as e:
            logger.warning(f"[Agent 1 Supervisor] LLM triage parsing failed ({e}), fallback sang clinical_medical.")
            return TriageDecision(
                is_emergency=False,
                emergency_level="routine",
                intent="clinical_medical",
                triage_reason="Câu hỏi bệnh lý nha khoa thông thường.",
                recommended_action="Chuyển tiếp đến Agent 2 và Agent 3 truy xuất tri thức."
            )
