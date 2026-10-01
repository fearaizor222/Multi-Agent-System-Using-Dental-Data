"""LangChain tools for Dental Multi-Agent system."""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.sample_data import load_sample_dental_documents
from src.core.logger import get_logger

logger = get_logger("tools.dental")

# Singleton hybrid retriever instance for tools
_sample_docs = load_sample_dental_documents()
_hybrid_retriever = HybridRetriever(documents=_sample_docs)


@tool
def search_dental_knowledge(query: str) -> str:
    """
    Tìm kiếm thông tin bệnh lý, triệu chứng, phác đồ điều trị nha khoa từ cơ sở dữ liệu chuyên ngành
    bằng cơ chế Hybrid RAG (kết hợp Dense Vector Search và Sparse BM25 Search).

    Args:
        query: Câu hỏi hoặc từ khóa nha khoa cần tra cứu (ví dụ: 'đau buốt nước lạnh', 'viêm tủy răng', 'răng khôn').
    """
    logger.info(f"Tool search_dental_knowledge được gọi với query: '{query}'")
    results = _hybrid_retriever.retrieve(query, top_k=3)

    if not results:
        return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức nha khoa hiện tại."

    formatted = []
    for idx, doc in enumerate(results, 1):
        formatted.append(
            f"--- Tài liệu {idx} (Mã: {doc.id}, Điểm RRF: {doc.score}) ---\n"
            f"Tiêu đề: {doc.title} [Chuyên mục: {doc.category}]\n"
            f"Nội dung: {doc.content}\n"
            f"Từ khóa liên quan: {', '.join(doc.keywords)}"
        )

    return "\n\n".join(formatted)


@tool
def assess_dental_urgency(symptoms: str, pain_level: int, has_fever: bool = False, has_swelling: bool = False) -> str:
    """
    Đánh giá mức độ khẩn cấp (Triage Assessment) của tình trạng răng miệng dựa trên triệu chứng.

    Args:
        symptoms: Mô tả triệu chứng của bệnh nhân.
        pain_level: Thang điểm đau từ 1 đến 10 (1: rất nhẹ, 10: dữ dội không chịu nổi).
        has_fever: Có bị sốt hay không.
        has_swelling: Có xuất hiện sưng phù nướu, má hoặc vùng mặt hay không.
    """
    urgency = "THẤP"
    recommendation = "Bệnh nhân có thể theo dõi thêm và đặt lịch hẹn khám định kỳ trong tuần."

    if has_fever or has_swelling or pain_level >= 8:
        urgency = "KHẨN CẤP / CẤP CỨU"
        recommendation = "Cần đến ngay phòng khám nha khoa hoặc bệnh viện chuyên khoa Răng Hàm Mặt để xử lý dẫn lưu mủ, giảm áp lực tủy hoặc kiểm soát nhiễm trùng lan tỏa."
    elif pain_level >= 5 or "nhức về đêm" in symptoms.lower():
        urgency = "TRUNG BÌNH - NGUY CƠ VIÊM TỦY"
        recommendation = "Cần khám nha sĩ trong vòng 24 - 48 giờ để được chụp phim X-quang và can thiệp chữa tủy kịp thời tránh biến chứng áp-xe."

    return (
        f"KẾT QUẢ ĐÁNH GIÁ MỨC ĐỘ KHẨN CẤP:\n"
        f"- Phân loại: {urgency}\n"
        f"- Điểm đau ghi nhận: {pain_level}/10\n"
        f"- Triệu chứng cảnh báo: Sốt ({'Có' if has_fever else 'Không'}), Sưng mặt/nướu ({'Có' if has_swelling else 'Không'})\n"
        f"- Khuyến nghị lâm sàng: {recommendation}"
    )


@tool
def lookup_tooth_info(tooth_number: int) -> str:
    """
    Tra cứu giải phẫu và vị trí của răng theo hệ thống đánh số Quốc tế FDI (ví dụ: 11, 16, 26, 36, 48, ...).

    Args:
        tooth_number: Số hiệu răng theo chuẩn FDI (11-48) hoặc ký hiệu thông thường (1-8).
    """
    # Mapping cơ bản
    tooth_map = {
        18: "Răng khôn hàm trên bên phải (Răng cối lớn thứ 3)",
        28: "Răng khôn hàm trên bên trái (Răng cối lớn thứ 3)",
        38: "Răng khôn hàm dưới bên trái (Răng cối lớn thứ 3)",
        48: "Răng khôn hàm dưới bên phải (Răng cối lớn thứ 3)",
        16: "Răng cối lớn thứ nhất hàm trên bên phải (Răng số 6 - giữ vai trò chịu lực nhai chính)",
        26: "Răng cối lớn thứ nhất hàm trên bên trái (Răng số 6 - chịu lực nhai chính)",
        36: "Răng cối lớn thứ nhất hàm dưới bên trái (Răng số 6 - chịu lực nhai chính)",
        46: "Răng cối lớn thứ nhất hàm dưới bên phải (Răng số 6 - chịu lực nhai chính)",
    }

    info = tooth_map.get(tooth_number)
    if info:
        return f"Thông tin giải phẫu răng FDI #{tooth_number}: {info}."

    return f"Răng #{tooth_number}: Thuộc cung răng người lớn. Cần bảo tồn cấu trúc men ngà và tủy răng."
