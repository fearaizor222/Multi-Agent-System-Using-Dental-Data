"""Supervisor Agent coordinating the multi-agent dental workflow."""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.core.llm import get_llm
from src.graph.state import AgentState
from src.core.logger import get_logger

logger = get_logger("agents.supervisor")

MEMBERS = ["rag_retriever", "dental_consultant"]
OPTIONS = MEMBERS + ["FINISH"]


class SupervisorRouterSchema(BaseModel):
    """Routing decision schema for the supervisor."""

    next: Literal["rag_retriever", "dental_consultant", "FINISH"] = Field(
        description="The next worker agent to route to, or 'FINISH' if the query has been completely answered."
    )
    reasoning: str = Field(
        description="Short rationale explaining why this agent or FINISH was chosen."
    )


SUPERVISOR_SYSTEM_PROMPT = """Bạn là Supervisor (Trưởng nhóm điều phối) của hệ thống Multi-Agent Chuyên khoa Răng Hàm Mặt.
Nhiệm vụ của bạn là phân tích yêu cầu của người dùng và lịch sử hội thoại để quyết định chuyển giao công việc cho Agent chuyên trách phù hợp nhất.

Các thành viên trong nhóm của bạn gồm:
1. 'rag_retriever': Chuyên gia tìm kiếm tài liệu, bệnh lý, triệu chứng và phác đồ điều trị nha khoa (sử dụng Hybrid RAG). Hãy gọi agent này ĐẦU TIÊN khi người dùng hỏi về bệnh lý, răng miệng, triệu chứng cần tra cứu tài liệu chuyên môn.
2. 'dental_consultant': Bác sĩ tư vấn nha khoa lâm sàng. Agent này phân tích thông tin từ RAG, giải thích cho bệnh nhân bằng giọng văn ân cần, đưa ra lời khuyên chăm sóc, cảnh báo mức độ khẩn cấp và hướng dẫn thăm khám.
3. 'FINISH': Chỉ chọn khi câu hỏi của người dùng ĐÃ ĐƯỢC bác sĩ tư vấn trả lời đầy đủ và chu đáo.

QUY TẮC ĐIỀU PHỐI:
- Nếu người dùng mới đặt câu hỏi về bệnh lý / triệu chứng nha khoa và chưa có tài liệu RAG tra cứu -> Chọn 'rag_retriever'.
- Nếu tài liệu RAG đã được tra cứu xong và đang cần bác sĩ tư vấn giải thích -> Chọn 'dental_consultant'.
- Nếu bác sĩ tư vấn ('dental_consultant') vừa hoàn thành câu trả lời cho bệnh nhân -> Chọn 'FINISH'.
"""


def supervisor_agent_node(state: AgentState) -> dict:
    """Supervisor node deciding the next agent in the execution graph."""
    step_count = state.get("step_count", 0)

    # Prevent potential runaway loops
    if step_count >= 6:
        logger.warning("Đã chạm giới hạn số bước tối đa (6 bước). Kết thúc luồng hội thoại.")
        return {"next": "FINISH", "step_count": step_count + 1}

    # Heuristic fast-path: if dental_consultant already responded, we can FINISH
    messages = state.get("messages", [])
    if messages and len(messages) >= 2:
        last_msg = messages[-1]
        if getattr(last_msg, "name", None) == "dental_consultant":
            logger.info("dental_consultant đã hoàn thành câu trả lời -> FINISH.")
            return {"next": "FINISH", "step_count": step_count + 1}

    # If rag_context is empty and user just asked a question, route to rag_retriever
    rag_context = state.get("rag_context", [])
    if not rag_context:
        logger.info("Chưa có rag_context -> Điều phối sang rag_retriever.")
        return {"next": "rag_retriever", "step_count": step_count + 1}

    # If rag_context exists and dental_consultant hasn't spoken yet, route to dental_consultant
    logger.info("Đã có rag_context -> Điều phối sang dental_consultant.")
    return {"next": "dental_consultant", "step_count": step_count + 1}
