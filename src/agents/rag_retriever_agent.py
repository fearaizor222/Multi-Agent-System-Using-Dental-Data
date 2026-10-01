"""RAG Retriever Agent specializing in searching Dental Knowledge Base."""

from langchain_core.messages import AIMessage
from src.graph.state import AgentState
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.sample_data import load_sample_dental_documents
from src.core.logger import get_logger

logger = get_logger("agents.rag_retriever")

# Initialize default retriever with sample data
_sample_docs = load_sample_dental_documents()
_hybrid_retriever = HybridRetriever(documents=_sample_docs)


def rag_retriever_node(state: AgentState) -> dict:
    """Agent node that performs Hybrid RAG retrieval based on patient's inquiry."""
    messages = state.get("messages", [])
    if not messages:
        return {"rag_context": [], "next": "dental_consultant"}

    # Extract user's latest query
    query = messages[-1].content if messages else ""
    logger.info(f"RAG Retriever đang tìm kiếm dữ liệu nha khoa cho câu hỏi: '{query[:60]}...'")

    # Perform Hybrid Retrieval (Vector + BM25 + RRF)
    retrieved_docs = _hybrid_retriever.retrieve(query, top_k=3)

    context_dicts = [
        {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "category": doc.category,
            "score": doc.score,
            "keywords": doc.keywords,
        }
        for doc in retrieved_docs
    ]

    # Format clinical summary message for the next agent
    if retrieved_docs:
        summary_lines = [f"[RAG Retriever] Đã tra cứu {len(retrieved_docs)} tài liệu lâm sàng phù hợp:"]
        for idx, doc in enumerate(retrieved_docs, 1):
            summary_lines.append(f"  {idx}. [{doc.id}] {doc.title} (Điểm tin cậy RRF: {doc.score:.4f})")
        summary_text = "\n".join(summary_lines)
    else:
        summary_text = "[RAG Retriever] Không tìm thấy tài liệu phù hợp trực tiếp trong kho tri thức."

    agent_message = AIMessage(content=summary_text, name="rag_retriever")

    return {
        "messages": [agent_message],
        "rag_context": context_dicts,
        "next": "dental_consultant",
        "step_count": state.get("step_count", 0) + 1,
    }
