"""Chạy A1 → A2 → A3 → A4; hỗ trợ một câu hoặc chat nhiều lượt."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

from src.core.a1_process import ROOT
from src.graph.assistant_runtime import load_assistant


def render_result(result):
    if result.get("message"):
        return result["message"]
    final=result.get("final_answer",{})
    if final.get("message"):
        return final["message"]
    return "Chưa hoàn tất bước tổng hợp câu trả lời. Bạn vui lòng thử lại sau."


def run_turn(graph,context,output):
    if output.exists():
        raise ValueError("File lượt chạy đã có")
    result=graph.invoke({"input":context})
    result["run_at"]=datetime.now(timezone.utc).isoformat()
    result["display_message"]=render_result(result)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(result["display_message"],flush=True)
    print("Kết quả:",output,flush=True)
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question",nargs="?")
    parser.add_argument("--input-json",type=Path)
    parser.add_argument("--chat",action="store_true")
    parser.add_argument("--config",type=Path,default=ROOT/"config/assistant_local.json")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    if sum((args.question is not None,args.input_json is not None,args.chat))!=1:
        parser.error("Chọn câu hỏi, --input-json hoặc --chat")
    output=args.output_dir or ROOT/"outputs/assistant_sessions"/datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    if output.exists():
        parser.error("Thư mục đầu ra đã có; chọn thư mục mới")
    graph,config=load_assistant(args.config)
    output.mkdir(parents=True,exist_ok=False)
    (output/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.chat:
        context=json.loads(args.input_json.read_text(encoding="utf-8")) if args.input_json else {"message":args.question}
        result=run_turn(graph,context,output/"turn_001.json")
        return 1 if result["status"] in ("error","blocked") else 0
    history=[]
    turn=0
    print("Nhập câu hỏi. /reset bắt đầu hội thoại mới; /exit kết thúc.",flush=True)
    while True:
        try:
            question=input("Bạn: ").strip()
        except (EOFError,KeyboardInterrupt):
            break
        if question=="/exit":
            break
        if question=="/reset":
            history=[]
            print("Đã bắt đầu hội thoại mới.",flush=True)
            continue
        if not question:
            continue
        turn+=1
        result=run_turn(graph,{"message":question,"history":history.copy()},output/f"turn_{turn:03d}.json")
        # Chỉ đưa nội dung đã hiển thị vào history, không đưa raw kế hoạch/model log.
        if result["status"] not in ("error","blocked"):
            history.extend([{"role":"user","content":question},
                            {"role":"assistant","content":result["display_message"]}])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
