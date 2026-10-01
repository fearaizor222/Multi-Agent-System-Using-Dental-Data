"""LangGraph workflow definition for Dental Multi-Agent system."""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.agents.supervisor import supervisor_agent_node
from src.agents.rag_retriever_agent import rag_retriever_node
from src.agents.dental_consultant import dental_consultant_node
from src.core.logger import get_logger

logger = get_logger("graph.workflow")


def router_condition(state: AgentState) -> Literal["rag_retriever", "dental_consultant", "__end__"]:
    """Conditional routing function reading supervisor's decision."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node


def build_dental_agent_graph():
    """Build and compile the Multi-Agent StateGraph."""
    logger.info("Đang khởi tạo LangGraph StateGraph cho Dental Multi-Agent...")

    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("dental_consultant", dental_consultant_node)

    # 2. Add Edges
    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        router_condition,
        {
            "rag_retriever": "rag_retriever",
            "dental_consultant": "dental_consultant",
            END: END,
        }
    )

    # After worker agents finish, return control to supervisor
    workflow.add_edge("rag_retriever", "supervisor")
    workflow.add_edge("dental_consultant", "supervisor")

    # 3. Compile Graph
    app = workflow.compile()
    logger.info("LangGraph workflow đã được biên dịch thành công.")
    return app
