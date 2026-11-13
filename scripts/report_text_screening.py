"""Report matched base-model screening without declaring a fine-tuned winner."""
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median

from src.training.a1_runtime import write_json
from src.training.model_candidates import DEFAULT_CATALOG, read_candidates
from src.training.text_screening import grade


def remove_outer_fence(raw):
    """Diagnostic only: tolerate one complete Markdown fence, no JSON repair."""
    match = re.fullmatch(r"```(?:json)?\n(.*)\n```", raw.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    destination = args.run_dir / "report"
    if destination.exists():
        parser.error("Report already exists")
    records, runtimes, comparisons, reviews, diagnostics, identity = [], [], [], [], [], None
    for spec in read_candidates(args.catalog):
        directory = args.run_dir / "base" / spec["id"]
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        runtime = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            raise ValueError("Incomplete candidate: " + spec["id"])
        current_identity = {key: runtime[key] for key in ("suite_manifest", "catalog_sha256", "max_new_tokens", "max_input_tokens", "seed", "thinking", "generation", "quantization", "source_code_sha256", "versions", "gpu")}
        if identity is not None and current_identity != identity:
            raise ValueError("Unmatched screening protocol: " + spec["id"])
        identity = current_identity
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != runtime["suite_manifest"]["case_count"]:
            raise ValueError("Missing predictions: " + spec["id"])
        records.append(summary)
        runtimes.append(runtime)
        diagnostic_by_id = {}
        for row in rows:
            diagnostic = {"model": spec["id"], "id": row["id"], "agent": row["agent"],
                "outer_fence_removed": remove_outer_fence(row["raw_output"]) != row["raw_output"],
                **grade(row, remove_outer_fence(row["raw_output"]))}
            diagnostic_by_id[row["id"]] = diagnostic
            diagnostics.append(diagnostic)
            reviews.append({"model": spec["id"], "id": row["id"], "agent": row["agent"],
                "input": json.dumps(row["input"], ensure_ascii=False),
                "expected_control": json.dumps(row["expected"], ensure_ascii=False),
                "raw_output": row["raw_output"], "schema_valid": row["schema_valid"],
                "task_control_pass": row["task_control_pass"],
                **{key: "" for key in ("content_faithful", "complete", "vietnamese_fluency", "invented_facts", "reviewer", "notes")}})
        for agent, metrics in summary["agents"].items():
            subset = [row for row in rows if row["agent"] == agent]
            comparisons.append({"model": spec["id"], "repo": spec["repo"], "agent": agent,
                **{key: metrics[key] for key in ("count", "schema_valid", "task_control_pass")},
                "fence_tolerant_schema_valid": sum(diagnostic_by_id[row["id"]]["schema_valid"] for row in subset),
                "fence_tolerant_control_pass": sum(diagnostic_by_id[row["id"]]["task_control_pass"] for row in subset),
                "median_generation_seconds": round(median(row["generation_seconds"] for row in subset), 2),
                "token_limit_cases": sum(row["hit_token_limit"] for row in subset),
                "human_content_review_completed": False, "selected_for_production": False})
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "comparison.json", {"models": records, "runtime": runtimes,
        "comparison": comparisons, "winner_by_agent": {agent: None for agent in ("A1", "A2", "A3", "A4")},
        "task_finetuning_completed": False, "production_changed": False})
    write_json(destination / "fence_diagnostics.json", {"scope": "Secondary parsing diagnostic only: one complete outer Markdown fence removed; no other repair. Original predictions and primary strict scores unchanged.", "rows": diagnostics})
    with (destination / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (destination / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    text = ["# Sàng lọc model văn bản — 05/11/2026", "",
        f"Đã chạy cùng {runtimes[0]['suite_manifest']['case_count']} tình huống phát triển cho {len(records)} model gốc, không nạp adapter A1 và chưa fine-tune A2/A3/A4.",
        "Tất cả dùng CUDA NF4, BF16, greedy, tắt thinking, tối đa 384 token đầu ra và không sửa JSON.", "",
        "| Model | Agent | Đúng schema | Qua kiểm tra điều khiển | Trung vị sinh (giây) |", "|---|---|---:|---:|---:|"]
    for row in comparisons:
        text.append(f"| {row['repo']} | {row['agent']} | {row['schema_valid']}/{row['count']} | {row['task_control_pass']}/{row['count']} | {row['median_generation_seconds']} |")
    text.extend(["", "Chẩn đoán phụ: chỉ bỏ một lớp Markdown fence bao trọn đầu ra, rồi chấm lại cùng schema/nhãn. Không sửa JSON trùng khóa, sai trường hoặc nhãn; không thay số chính phía trên.", "",
        "| Model | Agent | Schema sau bỏ fence | Điều khiển sau bỏ fence |", "|---|---|---:|---:|"])
    for row in comparisons:
        text.append(f"| {row['repo']} | {row['agent']} | {row['fence_tolerant_schema_valid']}/{row['count']} | {row['fence_tolerant_control_pass']}/{row['count']} |")
    text.extend(["", "Kiểm tra điều khiển A1: decision/targets/service_topics/image_ids; chưa chấm nghĩa resolved_request.",
        "A2: trạng thái và mã nguồn; CHƯA chấm độ đúng, đủ và trung thành của nội dung nhận định.",
        "A3: nhãn quan hệ với đoạn giả lập và câu trích nguyên văn. Đây là thử logic trên ví dụ tổng hợp, chưa là độ chính xác kiểm chứng y khoa.",
        "A4: trạng thái và ID nhận định được chọn. CHƯA chấm nội dung câu trả lời có giữ đúng ý hay tự thêm thông tin. Hợp đồng A4 ở đây là đề xuất LLM mới, chưa tích hợp.", "",
        "Nguồn bộ thử: 5 mẫu validation A1 đã dùng trước đây; 3 mẫu A2, 6 mẫu A3 và 4 mẫu A4 tổng hợp. Không đọc test A1. Không dùng bộ thử này làm dữ liệu huấn luyện.",
        "Các file human_review.csv chưa có người chấm. Không dùng số qua kiểm tra điều khiển để tuyên bố model tốt nhất.", "",
        "| Model | Tham số LoRA thử gradient | Gradient hợp lệ | Đỉnh cấp phát suy luận (MiB) | Đỉnh gồm probe (MiB) |", "|---|---:|---|---:|---:|"])
    for runtime in runtimes:
        probe = runtime.get("lora_feasibility", {})
        text.append(f"| {runtime['candidate']['repo']} | {probe.get('trainable_parameters', '')} | {probe.get('finite_gradients', False) and probe.get('nonzero_gradient', False)} | {runtime['inference_peak_allocated_mib']} | {runtime.get('including_probe_peak_allocated_mib', '')} |")
        if probe.get("error"):
            text.append("\nLỗi probe " + runtime["candidate"]["repo"] + ": " + probe["error"].replace("\n", " ") + "\n")
    text.extend(["", "Probe LoRA chỉ tính backward trên một câu kỹ thuật riêng, không optimizer step và không lưu adapter. Nó xác nhận khả năng tính gradient ở độ dài ngắn; không chứng minh đủ bộ nhớ cho mọi độ dài huấn luyện.",
        "Số thời gian là chạy model trên GPU sau khi nạp, không phải thời gian toàn hệ thống RAG hoặc CPU hiện tại. Đỉnh bộ nhớ là PyTorch tensor allocation, không phải tổng VRAM toàn hệ thống.", "",
        "Bước tiếp: chuẩn bị dữ liệu riêng A2/A3/A4 và rubric chấm, chọn ứng viên sau sàng lọc có người đánh giá, fine-tune thử cùng dữ liệu/ngân sách, so sánh validation sau fine-tune, rồi đánh giá test độc lập một lần sau khi chốt. Agent ảnh hoãn theo yêu cầu người dùng."])
    (destination / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Report: {destination / 'report.md'}")


if __name__ == "__main__":
    main()
