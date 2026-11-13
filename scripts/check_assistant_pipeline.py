"""Chạy A1–A3 model thật và A4 theo quy tắc; tách kiểm tra luồng khỏi chất lượng."""
import argparse
import json
from pathlib import Path
import sys

from src.core.a1_process import ROOT
from src.graph.assistant_runtime import load_assistant
from src.training.a1_runtime import file_hash,write_json
from scripts.run_dental_assistant import run_turn
from src.agents.synthesis import FinalAnswer


def synthesis_guard(result):
    if result["status"]!="knowledge_result":
        return not result.get("final_answer")
    try:
        final=FinalAnswer.model_validate(result.get("final_answer",{}))
        knowledge=result["knowledge"]
        draft={c["claim_id"]:c["text"] for c in knowledge.get("draft",{}).get("claims",[])}
        supported={c["claim_id"] for c in knowledge.get("verification",{}).get("checks",[]) if c.get("status")=="supported"}
        return (result["display_message"]==final.message and all(c.claim_id in supported and draft.get(c.claim_id)==c.text for c in final.claims))
    except (ValueError,TypeError,KeyError):
        return False


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/assistant_local.json")
    parser.add_argument("--cases",type=Path,default=ROOT/"config/assistant_smoke_cases.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error("Chọn thư mục kết quả mới")
    source=json.loads(args.cases.read_text(encoding="utf-8"))
    graph,config=load_assistant(args.config)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.output_dir/"config.json",config)
    write_json(args.output_dir/"cases_used.json",source)
    code=[*list((ROOT/"src/agents").glob("*.py")),*list((ROOT/"src/graph").glob("*.py")),
          ROOT/"src/core/a1_process.py",ROOT/"src/training/a1_protocol.py",ROOT/"scripts/a1_worker.py",ROOT/"scripts/predict_a1.py"]
    write_json(args.output_dir/"code_files.json",{str(p.relative_to(ROOT)):file_hash(p) for p in code})
    histories={};rows=[]
    for index,case in enumerate(source["cases"],1):
        context=case["input"].copy()
        if case.get("continue_from"):
            context["history"]=histories[case["continue_from"]]
        write_json(args.output_dir/"run_status.json",{"status":"running","case":case["id"],"completed":len(rows),"total":len(source["cases"])})
        print(f"[{index}/{len(source['cases'])}] {case['id']}",flush=True)
        result=run_turn(graph,context,args.output_dir/(case["id"]+".json"))
        histories[case["id"]]=[*context.get("history",[]),{"role":"user","content":context["message"]},
                               {"role":"assistant","content":result["display_message"]}]
        a3=any(t["step"]=="a3_verify" for t in result["trace"])
        a4=any(t["step"]=="a4_synthesize" for t in result["trace"])
        guard=synthesis_guard(result)
        passed=(result["status"] in case["expected_statuses"] and result["executed_targets"]==case["expected_targets"]
                and (not case.get("expect_a3") or a3) and (not case.get("expect_a4") or a4) and guard
                and (not case.get("expected_final_statuses") or result.get("final_answer",{}).get("status") in case["expected_final_statuses"])
                and (not case.get("expected_evidence_basis") or result.get("final_answer",{}).get("evidence_basis")==case["expected_evidence_basis"])
                and len(result.get("final_answer",{}).get("claims",[]))>=case.get("min_answer_claims",0)
                and (not case.get("expected_resolved_question") or result.get("question_resolution",{}).get("resolved_question")==case["expected_resolved_question"]))
        rows.append({"id":case["id"],"status":result["status"],"decision":result.get("decision",{}).get("decision"),
                     "executed_targets":result["executed_targets"],"a3_reached":a3,"a4_reached":a4,
                     "synthesis_guard_passed":guard,"final_status":result.get("final_answer",{}).get("status"),
                     "technical_expectation_passed":passed,
                     "knowledge_status":result.get("knowledge",{}).get("status"),
                     "citation_integrity_ok":result.get("knowledge",{}).get("review",{}).get("citation_integrity_ok"),
                     "error":result.get("error","")})
        write_json(args.output_dir/"results.json",rows)
    summary={"status":"completed","cases":len(rows),"technical_expectations_passed":sum(r["technical_expectation_passed"] for r in rows),
             "a3_reached":sum(r["a3_reached"] for r in rows),"a4_reached":sum(r["a4_reached"] for r in rows),
             "synthesis_guard_passed":sum(r["synthesis_guard_passed"] for r in rows),"results":rows,"cases_sha256":file_hash(args.cases),
             "scope":source["scope"],"a1_test_evaluated":False}
    write_json(args.output_dir/"summary.json",summary)
    write_json(args.output_dir/"run_status.json",{"status":"completed","completed":len(rows),"total":len(rows)})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary["technical_expectations_passed"]==len(rows) else 1


if __name__=="__main__":
    raise SystemExit(main())
