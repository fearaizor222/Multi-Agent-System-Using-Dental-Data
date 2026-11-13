"""Rà A3 với truy xuất thật và nhận định cố định; không tính là chạy A1/A2."""
import argparse
import json
from pathlib import Path
import time

from src.agents.contracts import Draft, DraftClaim, citation_of, evidence_bundle
from src.agents.synthesis import SynthesisAgent
from src.agents.verification import VerificationAgent
from src.graph.pipeline_review import review_run
from src.graph.runtime import index_paths, load_runtime
from src.rag.local_embeddings import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/assistant_local.json"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))["knowledge"]
    paths = index_paths(config["index_dir"], config["internal_index"], config["public_index"])
    runtime = load_runtime(paths, config["candidate_k"])
    catalogs = ({d.id: d for d in runtime.internal.retriever.documents},
                {d.id: d for d in runtime.public.retriever.documents})
    fixture = json.loads(Path("config/verification_cases.json").read_text(encoding="utf-8"))
    selected = {"related-topic-regression", "plastic-supported", "wood-contradicted", "arcon-unrelated-appliance"}
    cases = [c for c in fixture["cases"] if c["id"] in selected]
    results = []
    for case in cases:
        started = time.perf_counter()
        internal = evidence_bundle([catalogs[0][case["internal_id"]]], "I", "internal")
        draft = Draft(status="draft", claims=[DraftClaim(claim_id="c1", text=case["claim"],
                      citations=[citation_of(internal[0])])])
        agent = VerificationAgent(runtime.llm, runtime.public, runtime.entailment)
        verification = agent.run(case["question"], draft, internal)
        state = {"question": case["question"], "draft": draft.model_dump(), "status": verification["status"],
                 "internal_evidence": [e.model_dump() for e in internal], "verification": verification,
                 "trace": [{"step": "a3_verify", "status": verification["status"],
                            "milliseconds": (time.perf_counter() - started) * 1000}],
                 "model_calls": [{"stage": "verify", "calls": agent.last_trace}]}
        state["review"] = review_run(state, *catalogs)
        final = SynthesisAgent().run(state)
        check = verification["checks"][0]
        row = {"case_id": case["id"], "question": case["question"], "claim": case["claim"],
               "seconds": time.perf_counter() - started, "knowledge": state, "final_answer": final}
        # Fixed-fixture public labels do not apply to newly retrieved documents.
        # Only assert retrieval coverage for the known missing passage and guard obvious false support.
        row["retrieved_known_splint_passage"] = any(
            e["chunk_id"] == "chunk-d6624140838f5a99cc46cbcbc0bc802642ca2a6d7049f06d8e5caff52760fec6"
            for e in check.get("public_evidence", []))
        results.append(row)
        (args.output_dir / (case["id"] + ".json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["id"], check["status"], check.get("public_support"),
              "known_passage=", row["retrieved_known_splint_passage"], "final=", final["status"], flush=True)
    unchanged = all(file_sha256(Path(info["path"]) / name) == value
                    for info in runtime.indices.values() for name, value in info["sha256"].items())
    bad = [r["case_id"] for r in results if
           r["knowledge"]["verification"]["checks"][0]["status"] == "not_checked"
           or r["final_answer"]["status"] == "needs_review"
           or not r["knowledge"]["review"]["citation_integrity_ok"]
           or (r["case_id"] in ("wood-contradicted", "arcon-unrelated-appliance")
               and r["final_answer"]["claims"])
           or (r["case_id"] == "related-topic-regression" and not r["retrieved_known_splint_passage"])]
    summary = {"scope": "A3 retrieval/verification + A4 component check with fixed claims; A1/A2 not executed; not a clinical benchmark",
               "status": "complete", "case_count": len(results), "models": runtime.models,
               "technical_or_guard_failures": bad,
               "indices": runtime.indices, "index_unchanged": unchanged,
               "code_sha256": {name: file_sha256(name) for name in (
                   "src/agents/verification.py", "src/agents/public_retrieval.py", "scripts/check_a3_retrieval.py")},
               "results": [{"case_id": r["case_id"], "known_passage": r["retrieved_known_splint_passage"],
                            "check_status": r["knowledge"]["verification"]["checks"][0]["status"],
                            "final_status": r["final_answer"]["status"],
                            "citation_integrity_ok": r["knowledge"]["review"]["citation_integrity_ok"]}
                           for r in results]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(bool(bad) or not unchanged)


if __name__ == "__main__":
    raise SystemExit(main())
