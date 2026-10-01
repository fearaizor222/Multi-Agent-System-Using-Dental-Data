"""Base agent definitions and helper utilities."""

from typing import Callable, Any
from langchain_core.messages import AIMessage
from src.graph.state import AgentState
from src.core.logger import get_logger

logger = get_logger("agents.base")


def create_agent_node(agent_name: str, agent_callable: Callable[[AgentState], Any]):
    """Wrapper to standardize agent node execution and logging."""

    def node(state: AgentState) -> dict:
        logger.info(f"===> Bắt đầu Agent Node: [{agent_name}]")
        try:
            result = agent_callable(state)
            logger.info(f"<=== Hoàn thành Agent Node: [{agent_name}]")
            return result
        except Exception as e:
            logger.error(f"Lỗi xảy ra trong Agent Node [{agent_name}]: {e}", exc_info=True)
            return {
                "messages": [AIMessage(content=f"Đã xảy ra lỗi tại agent {agent_name}: {str(e)}")],
                "next": "FINISH"
            }

    return node
