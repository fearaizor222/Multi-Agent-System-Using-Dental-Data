"""Dental Consultant Agent acting as an experienced dental clinician."""

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.core.llm import get_llm
from src.graph.state import AgentState
from src.core.logger import get_logger

logger = get_logger("agents.dental_consultant")

CONSULTANT_SYSTEM_PROMPT = """Bạn là Bác Sĩ Tư Vấn Răng Hàm Mặt AI (Dental Consultant Agent) trong hệ thống Multi-Agent Y tế.
Nhiệm vụ của bạn là lắng nghe triệu chứng của bệnh nhân, kết hợp cùng các tài liệu y văn lâm sàng được cung cấp từ bộ phận tra cứu (RAG), để đưa ra lời tư vấn chuyên môn rõ ràng, khoa học và thấu cảm.

CẤU TRÚC PHẢN HỒI BẮT BUỘC:
1. 🩺 **Nhận định sơ bộ**: Tóm tắt lại triệu chứng và phân tích nguyên nhân tiềm ẩn (ví dụ: viêm tủy, mòn cổ răng, sâu răng, răng khôn,...).
2. 📚 **Cơ sở y khoa (dẫn chứng từ tài liệu)**: Giải thích ngắn gọn cơ chế bệnh sinh dựa trên tài liệu được tra cứu.
3. 💡 **Hướng dẫn xử lý an toàn tại nhà**: Các biện pháp giảm nhẹ triệu chứng tạm thời (súc miệng nước muối, tránh kích thích nhiệt,...). TUYỆT ĐỐI không tự ý kê đơn thuốc kháng sinh hay thuốc đặc trị.
4. ⚠️ **Dấu hiệu cảnh báo & Khuyến nghị thăm khám**: Nêu rõ khi nào cần đi khám gấp (sưng mặt, sốt, đau dữ dội về đêm) và các thủ thuật nha khoa nha sĩ có thể sẽ thực hiện (chụp phim X-quang, cạo vôi, trám răng hoặc chữa tủy).

Phong cách giao tiếp: Chuyên nghiệp, ân cần, dễ hiểu đối với người bệnh.
"""


def dental_consultant_node(state: AgentState) -> dict:
    """Agent node that generates comprehensive dental consultation for the user."""
    messages = state.get("messages", [])
    rag_context = state.get("rag_context", [])

    # Format context for the LLM
    context_str = ""
    if rag_context:
        context_items = []
        for doc in rag_context:
            context_items.append(
                f"- Tiêu đề: {doc.get('title')}\n"
                f"  Nội dung: {doc.get('content')}"
            )
        context_str = "\n\n".join(context_items)
    else:
        context_str = "Không có tài liệu tham khảo cụ thể từ kho dữ liệu."

    user_query = messages[0].content if messages else "Tư vấn răng miệng"

    prompt = (
        f"{CONSULTANT_SYSTEM_PROMPT}\n\n"
        f"--- TÀI LIỆU Y VĂN THAM KHẢO TỪ HYBRID RAG ---\n"
        f"{context_str}\n\n"
        f"--- CÂU HỎI / TRIỆU CHỨNG CỦA BỆNH NHÂN ---\n"
        f"{user_query}\n\n"
        f"Hãy soạn lời tư vấn chi tiết theo cấu trúc 4 phần đã quy định:"
    )

    llm = get_llm(temperature=0.3)
    response = llm.invoke([HumanMessage(content=prompt)])

    agent_message = AIMessage(
        content=response.content,
        name="dental_consultant"
    )

    return {
        "messages": [agent_message],
        "next": "FINISH",
        "step_count": state.get("step_count", 0) + 1,
    }
