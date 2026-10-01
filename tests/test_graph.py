import os
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pytest
except ImportError:
    pytest = None
from langchain_core.messages import HumanMessage
from src.graph.workflow import build_dental_agent_graph
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.bm25_retriever import BM25Retriever
from src.rag.vector_store import DenseVectorRetriever


def test_hybrid_retriever():
    """Test Hybrid Retrieval combining Dense and Sparse search with RRF."""
    sample_docs = [
        DentalDocument(
            id="DOC-1",
            title="Bệnh sâu răng",
            content="Sâu răng do vi khuẩn sinh acid phá hủy mô cứng men và ngà răng.",
            keywords=["sâu răng", "acid", "vi khuẩn"]
        ),
        DentalDocument(
            id="DOC-2",
            title="Viêm nướu chân răng",
            content="Viêm nướu chảy máu chân răng khi đánh răng, nhiều mảng bám vôi răng.",
            keywords=["viêm nướu", "chảy máu", "vôi răng"]
        ),
    ]

    retriever = HybridRetriever(documents=sample_docs)
    results = retriever.retrieve("chảy máu chân răng vôi răng", top_k=2)

    assert len(results) > 0
    assert results[0].id == "DOC-2"
    assert results[0].score > 0.0


def test_langgraph_workflow_execution():
    """Test that LangGraph successfully compiles and runs a query end-to-end."""
    app = build_dental_agent_graph()
    assert app is not None

    initial_state = {
        "messages": [HumanMessage(content="Răng của tôi bị buốt khi ăn đồ lạnh")],
        "next": "",
        "rag_context": [],
        "patient_data": {},
        "step_count": 0,
    }

    output = app.invoke(initial_state)

    assert "messages" in output
    assert len(output["messages"]) >= 2
    # Verify RAG context was populated
    assert "rag_context" in output
    assert len(output["rag_context"]) > 0


if __name__ == "__main__":
    print("1. Kiểm tra Hybrid Retriever...")
    test_hybrid_retriever()
    print("   -> Hybrid Retriever: THÀNH CÔNG!")

    print("2. Kiểm tra LangGraph Workflow...")
    test_langgraph_workflow_execution()
    print("   -> LangGraph Workflow: THÀNH CÔNG!")

    print("\n✅ TẤT CẢ KIỂM THỬ ĐÃ VƯỢT QUA!")
