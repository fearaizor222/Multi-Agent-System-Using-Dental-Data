"""So sánh adapter trên đúng cùng validation; giữ nguyên các lượt huấn luyện."""
import argparse
import csv
import json
from pathlib import Path

from src.training.a1_evaluation import selection_score, summarize
from src.training.a1_runtime import file_hash, write_json


def compare(runs, output):
    if output.exists():
        raise ValueError("Không ghi đè bản so sánh đã có")
    candidates, predictions = [], []
    reference = None
    reference_settings = None
    for folder in runs:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        runtime = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed" or summary["test_evaluated"]:
            raise ValueError("Chỉ so sánh lượt validation đã hoàn tất, test chưa chấm")
        settings = ({k:config[k] for k in ("base_model","revision","generation")}, runtime["packages"],
                    {p:h for p,h in runtime["code_sha256"].items() if Path(p).name in {"a1_runtime.py","a1_evaluation.py","a1_dataset.py"}})
        if reference_settings is None:
            reference_settings = settings
        if settings != reference_settings:
            raise ValueError("Model, generation, thư viện hoặc bộ chấm khác nhau")
        epoch = summary["best_finetuned_epoch"]
        adapter = Path(summary["best_adapter_dir"])
        rows = [json.loads(line) for line in (adapter.parent/"validation/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = [{k:row[k] for k in ("id","group_id","category","input","expected")} for row in rows]
        if reference is None:
            reference = keys
        if keys != reference:
            raise ValueError("Không trùng thứ tự, câu hỏi, ngữ cảnh hoặc nhãn validation")
        metrics = summarize(rows)
        saved = next(e["validation"] for e in summary["epochs"] if e["epoch"] == epoch)
        if metrics != saved:
            raise ValueError("Điểm tổng hợp không khớp đầu ra nguyên bản")
        manifest = json.loads((adapter.parent/"adapter_manifest.json").read_text(encoding="utf-8"))
        for name, checksum in manifest["files_sha256"].items():
            if file_hash(adapter/name) != checksum:
                raise ValueError("Adapter checksum sai")
        integrity = json.loads((folder/"input_integrity.json").read_text(encoding="utf-8"))
        candidates.append({"run":str(folder.resolve()),"adapter_dir":str(adapter),"epoch":epoch,
                           "train_count":integrity["dataset_check"]["by_split"]["train"],
                           "metrics":metrics,"selection_score":selection_score(metrics)})
        predictions.append(rows)
    winner = max(candidates,key=lambda c:c["selection_score"])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"selection.json",{"selected":winner,"candidates":candidates,"test_evaluated":False,
                                       "runtime_integrated":False,"scope":"Validation development selection; text semantics not graded."})
    differences = []
    for old,new in zip(predictions[0],predictions[-1]):
        transition = "improved" if new["control_exact"] and not old["control_exact"] else "regressed" if old["control_exact"] and not new["control_exact"] else "unchanged"
        differences.append({"id":old["id"],"category":old["category"],"question":old["input"]["message"],
                            "old_control_exact":old["control_exact"],"new_control_exact":new["control_exact"],
                            "transition":transition,"old_raw":old["raw_output"],"new_raw":new["raw_output"]})
    with (output/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    lines=["# So sánh các lượt fine-tune A1", "", "Cùng model nền, thư viện, prompt, bộ chấm, cách sinh và cùng các mẫu validation. Giữ riêng test; không chấm nội dung y khoa.", "",
           "| Lượt | Mẫu train | Đúng điều phối | Đúng decision | Đúng schema | Macro F1 |", "|---|---:|---:|---:|---:|---:|"]
    for c in candidates:
        m=c["metrics"]; n=m["count"]
        lines.append(f"| {Path(c['run']).name} | {c['train_count']} | {m['correct']['control_exact']}/{n} | {m['correct']['decision_correct']}/{n} | {m['correct']['schema_valid']}/{n} | {m['decision_macro_f1']:.4f} |")
    lines += ["", f"Ứng viên theo chỉ tiêu điều phối: `{winner['adapter_dir']}`.", "",
              f"So lượt cuối với lượt đầu: {sum(d['transition']=='improved' for d in differences)} ca cải thiện, {sum(d['transition']=='regressed' for d in differences)} ca suy giảm.", "",
              "Validation đã định hướng bổ sung dữ liệu và lựa chọn mô hình, nên đây là đánh giá phát triển. Thêm mẫu cũng tăng số bước cập nhật trong cùng một epoch; chưa tách riêng hiệu ứng nội dung dữ liệu và số bước. Dữ liệu tổng hợp chưa có người duyệt độc lập. Cần rà ngữ nghĩa resolved_request/message và thử capability trước khi tích hợp."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"selected":winner["adapter_dir"],"report":str(output/"report.md")},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if len(args.runs)<2:
        parser.error("Cần ít nhất hai lượt để so sánh")
    compare(args.runs,args.output_dir)


if __name__ == "__main__":
    main()
