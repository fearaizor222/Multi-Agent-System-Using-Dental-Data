"""Chạy thử A2 tạo nháp từ kho nội bộ, A3 đối chiếu kho công khai."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from src.core.llm import get_llm
from src.agents.contracts import Draft, DraftClaim, Evidence
from src.agents.verification import VerificationAgent
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from src.rag.embedding_store import INDEX_ROOT, ROOT
from src.rag.local_embeddings import LocalE5Embeddings
from src.rag.reranker import LocalReranker
from src.rag.reranked_retriever import RerankedRetriever
from src.rag.entailment import LocalEntailment


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Máng nhai là gì?")
    parser.add_argument("--provider", choices=["huggingface", "google", "ollama"], default="huggingface")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--check-faults", action="store_true", help="Thử mã chunk sai và nhận định cố ý sửa sai; lưu tách với kết quả tự sinh")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "outputs/agent_checks/day21" / (datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S") + ".json")
    if not args.question.strip() or args.candidate_k < 3:
        parser.error("Cần câu hỏi không rỗng và candidate-k >= 3")
    if args.output.exists():
        parser.error("File đầu ra đã có; chọn --output khác để giữ lượt chạy trước")
    encoder, reranker = LocalE5Embeddings(), LocalReranker()
    internal = RerankedRetriever.from_disk(args.index_dir / "internal", reranker, encoder, args.candidate_k)
    public = RerankedRetriever.from_disk(args.index_dir / "public", reranker, encoder, args.candidate_k)
    llm = get_llm(streaming=False, override_provider=args.provider)
    entailment = LocalEntailment()
    pipeline = create_knowledge_pipeline(internal, public, llm, entailment=entailment)
    result = {}
    seen_steps = 0
    for state in pipeline.stream({"question": args.question, "trace": [], "model_calls": []}, stream_mode="values"):
        result = dict(state)
        if len(result.get("trace", [])) > seen_steps:
            seen_steps = len(result["trace"])
            entry = result["trace"][-1]
            print(f"{entry['step']}: {entry['status']} ({entry['milliseconds'] / 1000:.1f}s)", flush=True)
    result["run_at"] = datetime.now(timezone.utc).isoformat()
    if args.check_faults and result.get("draft", {}).get("claims"):
        evidence = [Evidence.model_validate(item) for item in result["internal_evidence"]]
        original = Draft.model_validate(result["draft"])
        verifier = VerificationAgent(llm, public, entailment=entailment)
        forged = original.model_copy(deep=True)
        forged.claims = [forged.claims[0]]
        forged.claims[0].citations[0].chunk_id = "not-in-retrieved-context"
        fault_cases = [{"name": "forged_chunk_id", "controlled_input": True,
            "result": verifier.run(args.question, forged, evidence), "model_calls": verifier.last_trace.copy()}]
        # Ca cố ý sai chỉ áp dụng cho ví dụ máng nhai, không thay câu trả lời tự sinh.
        if args.question == "Máng nhai là gì?":
            false_claim = Draft(status="draft", claims=[DraftClaim(claim_id="fault-c1",
                text="Máng nhai là một khí cụ bằng gỗ dùng để nhổ răng.", citations=original.claims[0].citations)])
            try:
                check = verifier.run(args.question, false_claim, evidence)
                fault_cases.append({"name": "changed_claim_text", "controlled_input": True,
                    "injected_claim": false_claim.model_dump(), "result": check, "model_calls": verifier.last_trace.copy()})
            except (ValueError, RuntimeError) as exc:
                fault_cases.append({"name": "changed_claim_text", "controlled_input": True,
                    "error": str(exc), "model_calls": verifier.last_trace.copy()})
        result["controlled_fault_checks"] = fault_cases
    result["models"] = {"reranker": reranker.identity, "llm": getattr(llm, "identity", {"provider": args.provider}),
                        "entailment": entailment.identity, "nli_threshold": entailment.threshold}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Luồng:", " → ".join(item["step"] for item in result["trace"]))
    print("Trạng thái:", result["status"])
    for claim in result.get("draft", {}).get("claims", []):
        print(claim["claim_id"] + ": " + claim["text"] + " " + " ".join("[" + citation["evidence_id"] + "]" for citation in claim["citations"]))
    for check in result.get("verification", {}).get("checks", []):
        print(check["claim_id"] + ": " + check["status"] + " — " + check["reason"])
    if result.get("error"):
        print("Lỗi:", result["error"])
    print("Kết quả:", args.output)
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
