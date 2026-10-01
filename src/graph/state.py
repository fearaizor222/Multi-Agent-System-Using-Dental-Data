"""Definition of the shared AgentState for LangGraph."""

import operator
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any, List
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """The central state dictionary passed between agents in the graph."""

    # Chat history with automatic list concatenation
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # The next agent or node to route execution to ('dental_consultant', 'rag_agent', 'FINISH', etc.)
    next: str

    # Retrieved dental context from Hybrid RAG (dense + sparse)
    rag_context: List[Dict[str, Any]]

    # Extracted structured patient / clinical data (tooth number, chief complaint, etc.)
    patient_data: Dict[str, Any]

    # Current active step / iteration count to prevent infinite routing loops
    step_count: int
