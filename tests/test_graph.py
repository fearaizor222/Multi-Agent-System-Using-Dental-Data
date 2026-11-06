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

from src.graph.full_dental_graph import create_full_5agent_pipeline
from src.rag.base import DentalDocument
from src.rag.hybrid_retriever import HybridRetriever


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


def test_langgraph_workflow_compilation():
    """Test that Full 5-Agent LangGraph compiles successfully."""
    app = create_full_5agent_pipeline(offline_mode=True)
    assert app is not None


if __name__ == "__main__":
    print("1. Kiểm tra Hybrid Retriever...")
    test_hybrid_retriever()
    print("   -> Hybrid Retriever: THÀNH CÔNG!")

    print("2. Kiểm tra LangGraph Workflow...")
    test_langgraph_workflow_compilation()
    print("   -> LangGraph Workflow: THÀNH CÔNG!")

    print("\n✅ TẤT CẢ KIỂM THỬ ĐÃ VƯỢT QUA!")
