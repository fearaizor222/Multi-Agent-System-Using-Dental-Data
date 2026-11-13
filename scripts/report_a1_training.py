"""Tổng hợp so sánh theo cùng mẫu validation từ một lượt A1 đã hoàn tất."""
import argparse
import csv
import json
from pathlib import Path


def rows_by_id(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Trùng ID kết quả")
    return result


def report(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["test_evaluated"]:
        raise ValueError("Báo cáo này cần lượt validation đã hoàn tất và chưa chấm test")
    epoch = summary["best_finetuned_epoch"]
    baseline = rows_by_id(directory / "baseline_validation/predictions.jsonl")
    candidate = rows_by_id(directory / f"epoch_{epoch:02}/validation/predictions.jsonl")
    if set(baseline) != set(candidate):
        raise ValueError("Hai lượt không dùng cùng danh sách mẫu")
    comparisons = []
    for sample_id, before in baseline.items():
        after = candidate[sample_id]
        if before["input"] != after["input"] or before["expected"] != after["expected"]:
            raise ValueError("Đầu vào/nhãn đã thay đổi giữa trước và sau")
        transition = "improved" if after["control_exact"] and not before["control_exact"] else "regressed" if before["control_exact"] and not after["control_exact"] else "unchanged_correct" if after["control_exact"] else "unchanged_wrong"
        comparisons.append({"id": sample_id, "category": before["category"], "message": before["input"]["message"],
            "expected_decision": before["expected"]["decision"],
            "before_decision": before["predicted"]["decision"] if before["schema_valid"] else "invalid",
            "after_decision": after["predicted"]["decision"] if after["schema_valid"] else "invalid",
            "before_control_exact": before["control_exact"], "after_control_exact": after["control_exact"],
            "transition": transition, "before_raw": before["raw_output"], "after_raw": after["raw_output"]})
    with (directory / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    b = summary["baseline_validation"]
    after = next(item["validation"] for item in summary["epochs"] if item["epoch"] == epoch)
    labels = [("JSON hợp lệ", "raw_json_valid"), ("Đúng schema", "schema_valid"),
              ("Đúng ràng buộc khả năng/ảnh", "context_valid"), ("Đúng quyết định", "decision_correct"),
              ("Đúng toàn bộ trường điều phối", "control_exact")]
    table = "\n".join(f"| {label} | {b['correct'][key]}/{b['count']} | {after['correct'][key]}/{after['count']} |" for label, key in labels)
    improved = sum(row["transition"] == "improved" for row in comparisons)
    regressed = sum(row["transition"] == "regressed" for row in comparisons)
    memory = summary["memory"]
    modules = ", ".join(config["target_modules"])
    baseline_note = ("Baseline được tái sử dụng từ lượt " + summary["baseline_reused_from"] + "; đã kiểm tra checkpoint, prompt, holdout, bộ chấm, phiên bản, raw score và replay ba mẫu."
                     if summary.get("baseline_reused_from") else "Baseline được sinh đầy đủ trong lượt này.")
    integrity = json.loads((directory / "input_integrity.json").read_text(encoding="utf-8"))
    train_count = integrity["dataset_check"]["by_split"]["train"]
    strategy_note=""
    if config.get("sampling"):
        strategy_note=(f"Mỗi epoch có {config['sampling']['samples_per_epoch']} lượt trình bày từ {train_count} mẫu train, phủ mọi mẫu rồi replay cân bằng decision và nhóm. "
                       f"Trọng số loss: {config['token_weighting']}. Các giá trị loss không so trực tiếp với lượt dùng loss đều token. "
                       f"Quy tắc chọn checkpoint: {summary['selection_rule']}. Ngưỡng recall/định dạng đã đặt đạt: {summary['critical_checkpoint_gate_passed']}; đây chưa phải xác nhận sẵn sàng triển khai.")
    text = f"""# Kết quả fine-tune thử A1

Model nền: {config['base_model']}. Huấn luyện QLoRA trên {train_count} mẫu tổng hợp, {config['epochs']} epoch, chỉ cập nhật adapter ở {modules}. So sánh trước/sau dùng cùng {b['count']} mẫu validation, cùng NF4/BF16, system prompt, greedy decoding và giới hạn token. Tập test chưa được dùng để đánh giá.

{baseline_note}

{strategy_note}

| Chỉ tiêu | Model nền | Adapter epoch {epoch} |
|---|---:|---:|
{table}

Macro F1 quyết định: {b['decision_macro_f1']:.4f} → {after['decision_macro_f1']:.4f}.
Có {improved} mẫu chuyển từ sai sang đúng và {regressed} mẫu chuyển từ đúng sang sai theo chỉ tiêu điều phối.

Adapter tốt nhất trong các epoch đã thử: `{summary['best_adapter_dir']}`.
Kết quả tốt hơn model nền theo quy tắc đã chốt: **{summary['finetuned_improves_over_baseline']}**.
Ứng viên được đề xuất từ lượt thử: `{summary['recommended_candidate']}`. Không tự tích hợp hoặc triển khai vào hệ thống.

Đã nạp lại adapter đã lưu và kiểm tra ba mẫu validation có tái hiện nguyên đầu ra: {summary['adapter_reload_probe_matches']}.
GPU đạt đỉnh {memory['peak_allocated_mib']:.1f} MiB tensor, dự trữ {memory['peak_reserved_mib']:.1f} MiB.

## Cách đọc và giới hạn

“Đúng toàn bộ trường điều phối” yêu cầu JSON đúng schema, đúng decision, tập targets, service_topics, image_ids và không gọi tác tử/công cụ không khả dụng hoặc ảnh không có. Đầu ra lỗi vẫn nằm trong mẫu số. Không sửa JSON, không thử lại hoặc thay bằng câu trả lời mẫu.

Chỉ tiêu này chưa tự đánh giá ngữ nghĩa của `resolved_request` và câu `message`. Cần rà những trường đó bằng đầu ra nguyên bản ở `comparison.csv`, nhất là các ca hỏi lại, hội thoại tiếp nối và nhiều nhu cầu.

Dữ liệu là tổng hợp, chưa có người duyệt nhãn độc lập. Validation đã dùng chọn checkpoint nên kết quả không phải đánh giá cuối hoặc bằng chứng tổng quát hóa sang hội thoại thực tế. Không diễn giải các tỷ lệ này thành độ chính xác y khoa.

Đầu ra chi tiết: `baseline_validation/`, các thư mục `epoch_*/validation/`, `training_log.jsonl`, `epochs.json`, `summary.json`, `comparison.csv`.
"""
    (directory / "report.md").write_text(text, encoding="utf-8")
    return {"report": str(directory / "report.md"), "improved_cases": improved, "regressed_cases": regressed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
