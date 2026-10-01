"""Base data models and interfaces for the RAG subsystem."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DentalDocument(BaseModel):
    """Document representation for dental clinical knowledge."""

    id: str
    title: str
    content: str
    category: Optional[str] = "Chung"
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = 0.0


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[DentalDocument]:
        """Retrieve relevant documents for a given query."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[DentalDocument]) -> None:
        """Add documents to the retriever index."""
        pass
