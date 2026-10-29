"""Agent 4 / Synthesis Node: Dental Consultant Agent — Tổng hợp tư vấn lâm sàng dựa trên kết quả kiểm chứng y khoa."""
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("agents.consultant")

CONSULTANT_SYSTEM_PROMPT = """Bạn là Bác sĩ Tư vấn Nha khoa Răng Hàm Mặt giàu kinh nghiệm thuộc Hệ thống Đa Tác tử C20.
Nhiệm vụ của bạn là tổng hợp các thông tin đã được kiểm chứng y khoa để đưa ra câu trả lời tư vấn hoàn chỉnh, chuyên nghiệp và an tâm cho người bệnh.

Nguyên tắc bắt buộc khi trả lời:
1. Tính xác thực y khoa: Chỉ khẳng định các nội dung đã được Agent 5 (Kiểm chứng) xác nhận là "supported". Nếu có nhận định "insufficient" hoặc "contradicted", phải giải thích rõ ràng là y văn chưa đủ bằng chứng hoặc có mâu thuẫn để người bệnh thận trọng.
2. Tích hợp kết quả thị giác (nếu có): Nếu có thông tin từ Agent Thị giác, hãy tóm tắt các dấu hiệu trực quan (loại ảnh, vùng răng, tổn thương nhận thấy).
3. Văn phong y khoa nhân ái: Chu đáo, dễ hiểu, tránh thuật ngữ quá hàn lâm nhưng vẫn đảm bảo tính chuẩn xác.
4. Trích dẫn minh bạch: Khi đưa ra thông tin then chốt, chú thích mã nguồn ở cuối câu (ví dụ: [I1], [P1]).
5. Khuyến cáo thực tế: Hướng dẫn chăm sóc giảm đau/vệ sinh răng miệng an toàn và luôn khuyến khích thăm khám trực tiếp tại cơ sở chuyên khoa RHM có uy tín."""


class DentalConsultantAgent:
    """Tác tử Bác sĩ Tư vấn: Tổng hợp bản nháp, kết quả thị giác và kiểm chứng y khoa thành phản hồi trọn vẹn."""

    def __init__(self, llm=None):
        self.llm = llm or get_llm(temperature=0.2, streaming=False)

    def synthesize(
        self,
        question: str,
        vision_findings: Optional[Dict] = None,
        draft: Optional[Dict] = None,
        verification: Optional[Dict] = None,
        internal_evidence: Optional[List[Dict]] = None
    ) -> str:
        """Tổng hợp câu trả lời tư vấn hoàn chỉnh."""
        context_parts = []

        # 1. Thông tin thị giác
        if vision_findings:
            v_text = (
                f"- Loại hình ảnh: {vision_findings.get('image_type', 'Ảnh lâm sàng')}\n"
                f"- Cấu trúc liên quan: {', '.join(vision_findings.get('teeth_involved', [])) or 'Cung răng'}\n"
                f"- Nhận định hình ảnh: {vision_findings.get('preliminary_impression', '')}\n"
            )
            pathologies = vision_findings.get("detected_pathologies", [])
            if pathologies:
                v_text += "- Bệnh lý quan sát thấy: " + "; ".join(f"{p.get('name')} tại {p.get('location')} ({p.get('severity')})" for p in pathologies)
            context_parts.append(f"### KẾT QUẢ PHÂN TÍCH HÌNH ẢNH (AGENT 3 - VISION):\n{v_text}")

        # 2. Bản nháp nhận định
        claims = draft.get("claims", []) if draft else []
        if claims:
            c_text = "\n".join(f"- [{c.get('claim_id')}] {c.get('text')} (Trích dẫn: {', '.join(cit.get('evidence_id') for cit in c.get('citations', []))})" for c in claims)
            context_parts.append(f"### BẢN NHÁP NHẬN ĐỊNH LÂM SÀNG (AGENT 4 - DRAFTING):\n{c_text}")

        # 3. Kết quả kiểm chứng y khoa
        checks = verification.get("checks", []) if verification else []
        if checks:
            v_text = "\n".join(f"- [{ch.get('claim_id')}] Trạng thái: {ch.get('status')} | Đánh giá: {ch.get('reason', '')}" for ch in checks)
            context_parts.append(f"### KẾT QUẢ KIỂM CHỨNG BẰNG CHỨNG (AGENT 5 - VERIFICATION):\n{v_text}")

        # 4. Đoạn trích bằng chứng
        if internal_evidence:
            e_text = "\n".join(f"- [{e.get('evidence_id')}] {e.get('title')}: {e.get('content')[:180]}..." for e in internal_evidence)
            context_parts.append(f"### TÀI LIỆU Y VĂN THAM KHẢO:\n{e_text}")

        full_context = "\n\n".join(context_parts)
        user_prompt = (
            f"Câu hỏi của người bệnh: \"{question}\"\n\n"
            f"DỮ LIỆU TỔNG HỢP TỪ CÁC TÁC TỬ TRONG PIPELINE:\n{full_context}\n\n"
            "Hãy soạn thảo câu trả lời tư vấn nha khoa chi tiết, cấu trúc rõ ràng với các đề mục:"
            "\n1. Tình trạng ghi nhận & Đánh giá sơ bộ"
            "\n2. Phân tích bệnh lý chuyên sâu (có trích dẫn nguồn y văn)"
            "\n3. Hướng xử trí & Lời khuyên chăm sóc"
            "\n4. Cảnh báo & Khuyến cáo thăm khám chuyên khoa"
        )

        logger.info("Đang tạo bản tư vấn tổng hợp từ Dental Consultant...")
        response = self.llm.invoke([
            SystemMessage(content=CONSULTANT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        return response.content
