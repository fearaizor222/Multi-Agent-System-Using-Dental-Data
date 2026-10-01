"""Tools module exposing dental and RAG tools."""

from src.tools.dental_tools import (
    search_dental_knowledge,
    assess_dental_urgency,
    lookup_tooth_info,
)

__all__ = [
    "search_dental_knowledge",
    "assess_dental_urgency",
    "lookup_tooth_info",
]
