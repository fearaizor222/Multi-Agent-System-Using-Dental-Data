"""Định nghĩa và biên dịch StateGraph cho Hệ thống 5 Agent Nha khoa C20:
1. Agent 1: Supervisor (Điều phối & Triage)
2. Agent 2: Closed-data RAG (Kho nội bộ: Giáo trình Cắn khớp học RHM)
3. Agent 3: Open-data RAG (Kho công khai: Hướng dẫn điều trị Bộ Y Tế)
4. Agent 4: Dental Vision (Thị giác & Xử lý ảnh lâm sàng/X-quang)
5. Agent 5: Drafting Response (Tạo bản nháp nhận định & Trả lời tư vấn)
"""
import time
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.supervisor import SupervisorAgent
from src.agents.internal_rag_agent import ClosedDataRAGAgent
from src.agents.public_rag_agent import OpenDataRAGAgent
from src.agents.vision_agent import DentalVisionAgent
from src.agents.drafting_agent import DraftingResponseAgent
from src.core.llm import get_llm
from src.core.logger import get_logger

logger = get_logger("graph.full_dental")


class Dental5AgentState(TypedDict, total=False):
    # Đầu vào
    question: str
    image_path: Optional[str]

    # Agent 1: Supervisor & Triage
    is_emergency: bool
    emergency_level: str
    intent: str
    triage_reason: str
    recommended_action: str

    # Agent 4: Vision Agent
    vision_findings: Optional[dict]

    # Agent 2: Closed-data RAG (Kho nội bộ)
    internal_evidence: List[dict]

    # Agent 3: Open-data RAG (Kho công khai)
    public_evidence: List[dict]

    # Agent 5: Drafting Response (Tạo bản nháp trả lời)
    key_claims: List[dict]
    final_response: str

    status: str
    trace: List[dict]


def create_full_5agent_pipeline(
    llm=None,
    supervisor_agent=None,
    internal_rag_agent=None,
    public_rag_agent=None,
    vision_agent=None,
    drafting_agent=None,
    offline_mode: bool = False
):
    """Xây dựng và biên dịch StateGraph cho 5 Agent chuẩn (hỗ trợ cả Online Cloud và Offline Local CPU)."""
    if offline_mode:
        from src.core.local_chat import LocalChatModel
        llm = llm or LocalChatModel()
        logger.info("[FullDentalGraph] Khởi tạo hệ sinh thái 5 Agent ở CHẾ ĐỘ OFFLINE LOCAL (Qwen2.5-0.5B trên CPU)")
    else:
        try:
            llm = llm or get_llm(streaming=False)
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Cloud LLM ({e}), tự động chuyển sang Offline Local CPU...")
            from src.core.local_chat import LocalChatModel
            llm = LocalChatModel()

    supervisor = supervisor_agent or SupervisorAgent(llm)
    internal_rag = internal_rag_agent or ClosedDataRAGAgent()
    public_rag = public_rag_agent or OpenDataRAGAgent()
    vision = vision_agent or DentalVisionAgent()
    drafting = drafting_agent or DraftingResponseAgent(llm)


    def record_step(state: Dental5AgentState, name: str, start_time: float, status: str = "success") -> List[dict]:
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace = list(state.get("trace", []))
        trace.append({
            "step": name,
            "status": status,
            "duration_ms": round(duration_ms, 2)
        })
        return trace

    # --- NODE 1: AGENT 1 - SUPERVISOR & TRIAGE ---
    def node_agent1_supervisor(state: Dental5AgentState) -> Dental5AgentState:
        start = time.perf_counter()
        question = state.get("question", "")
        image_path = state.get("image_path")
        decision = supervisor.evaluate(question, image_path=image_path)
        logger.info(f"[Agent 1 Supervisor] Intent: {decision.intent} | Emergency: {decision.is_emergency}")
        trace = record_step(state, "agent1_supervisor", start)
        return {
            "is_emergency": decision.is_emergency,
            "emergency_level": decision.emergency_level,
            "intent": decision.intent,
            "triage_reason": decision.triage_reason,
            "recommended_action": decision.recommended_action,
            "trace": trace,
            "status": "triaged"
        }

    # --- NODE CẤP CỨU THOÁT NHANH ---
    def node_emergency_exit(state: Dental5AgentState) -> Dental5AgentState:
        triage_msg = (
            "🚨 **CẢNH BÁO CẤP CỨU Y KHOA NHA KHOA KHẨN CẤP** 🚨\n\n"
            f"**Lý do phân loại:** {state.get('triage_reason')}\n\n"
            f"**Hướng dẫn xử trí tức thời:**\n{state.get('recommended_action')}\n\n"
            "⚠️ **Lưu ý:** Vui lòng không tự ý uống thêm thuốc giảm đau liều cao hoặc đắp thuốc lá dân gian. "
            "Hãy gọi người nhà hỗ trợ đưa đến bệnh viện hoặc phòng khám chuyên khoa Răng Hàm Mặt có khoa cấp cứu gần nhất!"
        )
        return {
            "final_response": triage_msg,
            "status": "emergency_stopped"
        }

    # --- NODE THÔNG TIN HÀNH CHÍNH/DỊCH VỤ ---
    def node_service_exit(state: Dental5AgentState) -> Dental5AgentState:
        service_msg = (
            "📋 **THÔNG TIN DỊCH VỤ PHÒNG KHÁM RĂNG HÀM MẶT**\n\n"
            "- **Giờ làm việc:** 08:00 – 20:00 (Thứ 2 đến Thứ 7), 08:00 – 17:00 (Chủ nhật).\n"
            "- **Dịch vụ chuyên sâu:** Khám tổng quát & Tẩy trắng răng, Điều trị tủy nội nha, Nhổ răng khôn vi phẫu, Máng nhai thư giãn khớp TMJ, Chỉnh nha & Niềng răng trong suốt.\n"
            "- **Đặt lịch hẹn:** Quý khách có thể để lại số điện thoại hoặc liên hệ hotline để lễ tân hỗ trợ sắp xếp bác sĩ phù hợp."
        )
        return {
            "final_response": service_msg,
            "status": "service_answered"
        }

    # --- NODE 4: AGENT 4 - DENTAL VISION AGENT ---
    def node_agent4_vision(state: Dental5AgentState) -> Dental5AgentState:
        start = time.perf_counter()
        image_path = state.get("image_path")
        findings = vision.analyze(image_path, user_query=state.get("question", ""))
        logger.info(f"[Agent 4 Vision] Đã phân tích ảnh: {findings.image_type} | Bệnh lý: {len(findings.detected_pathologies)}")
        trace = record_step(state, "agent4_vision", start)
        return {
            "vision_findings": findings.model_dump(),
            "trace": trace,
            "status": "vision_analyzed"
        }

    # --- NODE 2: AGENT 2 - CLOSED-DATA RAG (KHO NỘI BỘ) ---
    def node_agent2_internal_rag(state: Dental5AgentState) -> Dental5AgentState:
        start = time.perf_counter()
        query = state.get("question", "")

        # Nếu có kết quả ảnh từ Agent 4, bổ sung triệu chứng vào query
        if state.get("vision_findings"):
            vf = state["vision_findings"]
            addons = " ".join([p.get("name", "") for p in vf.get("detected_pathologies", [])])
            if addons:
                query = f"{query} {addons}"

        evidence = internal_rag.retrieve(query, top_k=3)
        logger.info(f"[Agent 2 Closed RAG] Truy xuất được {len(evidence)} đoạn y văn nội bộ [I].")
        trace = record_step(state, "agent2_closed_rag", start)
        return {
            "internal_evidence": [e.model_dump() for e in evidence],
            "trace": trace,
            "status": "internal_retrieved"
        }

    # --- NODE 3: AGENT 3 - OPEN-DATA RAG (KHO CÔNG KHAI) ---
    def node_agent3_public_rag(state: Dental5AgentState) -> Dental5AgentState:
        start = time.perf_counter()
        query = state.get("question", "")

        if state.get("vision_findings"):
            vf = state["vision_findings"]
            addons = " ".join([p.get("name", "") for p in vf.get("detected_pathologies", [])])
            if addons:
                query = f"{query} {addons}"

        evidence = public_rag.retrieve(query, top_k=3)
        logger.info(f"[Agent 3 Open RAG] Truy xuất được {len(evidence)} đoạn y văn công khai [P].")
        trace = record_step(state, "agent3_open_rag", start)
        return {
            "public_evidence": [e.model_dump() for e in evidence],
            "trace": trace,
            "status": "public_retrieved"
        }

    # --- NODE 5: AGENT 5 - DRAFTING & RESPONSE SYNTHESIS ---
    def node_agent5_drafting(state: Dental5AgentState) -> Dental5AgentState:
        start = time.perf_counter()
        draft_result = drafting.run(
            question=state.get("question", ""),
            internal_evidence=state.get("internal_evidence", []),
            public_evidence=state.get("public_evidence", []),
            vision_findings=state.get("vision_findings")
        )
        logger.info(f"[Agent 5 Drafting] Đã tạo {len(draft_result.key_claims)} nhận định và hoàn thành bản nháp tư vấn.")
        trace = record_step(state, "agent5_drafting", start)
        return {
            "key_claims": [c.model_dump() for c in draft_result.key_claims],
            "final_response": draft_result.draft_response,
            "trace": trace,
            "status": "completed"
        }

    # --- ĐỊNH NGHĨA CONDITIONAL ROUTING TỪ SUPERVISOR ---
    def route_supervisor(state: Dental5AgentState) -> str:
        if state.get("is_emergency"):
            return "emergency_exit"
        if state.get("intent") == "service_info":
            return "service_exit"
        if state.get("image_path"):
            return "agent4_vision"
        return "agent2_internal_rag"

    # --- KHỞI TẠO STATE GRAPH ---
    workflow = StateGraph(Dental5AgentState)

    workflow.add_node("agent1_supervisor", node_agent1_supervisor)
    workflow.add_node("emergency_exit", node_emergency_exit)
    workflow.add_node("service_exit", node_service_exit)
    workflow.add_node("agent4_vision", node_agent4_vision)
    workflow.add_node("agent2_internal_rag", node_agent2_internal_rag)
    workflow.add_node("agent3_public_rag", node_agent3_public_rag)
    workflow.add_node("agent5_drafting", node_agent5_drafting)

    # Cạnh bắt đầu
    workflow.add_edge(START, "agent1_supervisor")

    # Phân luồng từ Agent 1 (Supervisor)
    workflow.add_conditional_edges("agent1_supervisor", route_supervisor, {
        "emergency_exit": "emergency_exit",
        "service_exit": "service_exit",
        "agent4_vision": "agent4_vision",
        "agent2_internal_rag": "agent2_internal_rag"
    })

    # Nếu có ảnh: Agent 4 (Vision) -> Agent 2 (Closed RAG)
    workflow.add_edge("agent4_vision", "agent2_internal_rag")

    # Sau kho đóng: Agent 2 (Closed RAG) -> Agent 3 (Open RAG)
    workflow.add_edge("agent2_internal_rag", "agent3_public_rag")

    # Sau kho mở: Agent 3 (Open RAG) -> Agent 5 (Drafting Response)
    workflow.add_edge("agent3_public_rag", "agent5_drafting")

    # Điểm kết thúc
    workflow.add_edge("emergency_exit", END)
    workflow.add_edge("service_exit", END)
    workflow.add_edge("agent5_drafting", END)

    return workflow.compile()
