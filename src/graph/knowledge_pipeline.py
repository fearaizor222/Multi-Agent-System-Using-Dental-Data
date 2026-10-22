"""Luồng A2 → A3; dùng để thử trước khi ghép Supervisor và A4."""
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.contracts import Draft, Evidence, evidence_bundle
from src.agents.drafting import DraftingAgent
from src.agents.verification import VerificationAgent


class KnowledgeState(TypedDict, total=False):
    question: str
    status: str
    internal_evidence: list[dict]
    draft: dict
    verification: dict
    trace: list[dict]
    model_calls: list[dict]
    error: str


def create_knowledge_pipeline(internal_retriever, public_retriever, llm, entailment=None):
    drafting, verification = DraftingAgent(llm), VerificationAgent(llm, public_retriever, entailment=entailment)

    def step(state, name, operation):
        start = time.perf_counter()
        try:
            result = operation()
        except (ValueError, RuntimeError, IndexError, TypeError) as exc:
            result = {"status": "error", "error": type(exc).__name__ + ": " + str(exc)}
        result["trace"] = [*state.get("trace", []), {"step": name, "milliseconds": (time.perf_counter() - start) * 1000,
                                                  "status": result.get("status")}]
        return result

    def retrieve(state):
        def operation():
            hits = internal_retriever.retrieve(state["question"], top_k=3)
            evidence = evidence_bundle(hits, "I", "internal")
            return {"status": "retrieved" if evidence else "insufficient_evidence", "internal_evidence": [item.model_dump() for item in evidence]}
        return step(state, "a2_retrieve", operation)

    def draft(state):
        def operation():
            result = drafting.run(state["question"], [Evidence.model_validate(item) for item in state["internal_evidence"]])
            return {"draft": result.model_dump(), "status": result.status}
        result = step(state, "a2_draft", operation)
        result["model_calls"] = [*state.get("model_calls", []), {"stage": "draft", "calls": drafting.last_trace}]
        return result

    def verify(state):
        def operation():
            result = verification.run(state["question"], Draft.model_validate(state["draft"]),
                [Evidence.model_validate(item) for item in state["internal_evidence"]])
            return {"verification": result, "status": result["status"]}
        result = step(state, "a3_verify", operation)
        result["model_calls"] = [*state.get("model_calls", []), {"stage": "verify", "calls": verification.last_trace}]
        return result

    graph = StateGraph(KnowledgeState)
    graph.add_node("a2_retrieve", retrieve)
    graph.add_node("a2_draft", draft)
    graph.add_node("a3_verify", verify)
    graph.add_edge(START, "a2_retrieve")
    graph.add_conditional_edges("a2_retrieve", lambda state: "a2_draft" if state["status"] == "retrieved" else END)
    graph.add_conditional_edges("a2_draft", lambda state: "a3_verify" if state["status"] == "draft" else END)
    graph.add_edge("a3_verify", END)
    return graph.compile()
