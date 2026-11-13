"""Publication figures from completed A1 logs only; no model inference/training."""
import csv
import json
import math
from pathlib import Path

from src.training.a1_evaluation import DECISIONS, score_response, summarize
from src.training.a1_runtime import file_hash, write_json

FLAGS = ("raw_json_valid", "schema_valid", "context_valid", "decision_correct", "targets_correct",
         "service_topics_correct", "image_ids_correct", "control_exact")
METRICS = [("JSON\nhợp lệ", "raw_json_valid"), ("Đúng\nschema", "schema_valid"),
           ("Đúng\nràng buộc", "context_valid"), ("Đúng\nquyết định", "decision_correct"),
           ("Đúng toàn bộ\nđiều phối", "control_exact"), ("Macro-F1\nquyết định", "decision_macro_f1")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rolling_mean(values, window):
    if not isinstance(window, int) or window < 1:
        raise ValueError("Smoothing window must be a positive integer")
    result, total = [], 0.0
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError("Non-finite training loss")
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def confusion_counts(rows):
    labels = [*DECISIONS, "invalid"]
    matrix = [[0 for _ in labels] for _ in DECISIONS]
    for row in rows:
        actual = row["expected"]["decision"]
        predicted = row["predicted"]["decision"] if row["schema_valid"] else "invalid"
        matrix[DECISIONS.index(actual)][labels.index(predicted)] += 1
    return matrix


def checked_predictions(path, expected_metrics):
    rows = read_lines(path)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty/duplicate prediction IDs: " + str(path))
    for row in rows:
        if row["split"] != "validation":
            raise ValueError("Only validation prediction logs are accepted")
        actual = score_response(row["input"], row["expected"], row["raw_output"])
        if any(row[key] != actual[key] for key in FLAGS) or row.get("predicted") != actual["predicted"]:
            raise ValueError("Saved scores disagree with raw output: " + row["id"])
    measured = summarize(rows)
    if (not expected_metrics["completed"] or measured["count"] != expected_metrics["count"]
            or measured["correct"] != expected_metrics["correct"]
            or not math.isclose(measured["decision_macro_f1"], expected_metrics["decision_macro_f1"], abs_tol=1e-10)):
        raise ValueError("Summary metrics disagree with predictions: " + str(path))
    return rows, measured


def load_run(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    integrity = read_json(run_dir / "input_integrity.json")
    if summary["status"] != "completed" or summary["test_evaluated"]:
        raise ValueError("Need a completed validation-only run; no fabricated/in-progress results")
    sources = [run_dir / name for name in ("summary.json", "config.json", "input_integrity.json", "training_log.jsonl")]
    baseline_path = run_dir / "baseline_validation/predictions.jsonl"
    baseline_rows, baseline_metrics = checked_predictions(baseline_path, summary["baseline_validation"])
    sources.append(baseline_path)
    by_id = {row["id"]: row for row in baseline_rows}
    stages = [{"epoch": 0, "metrics": baseline_metrics, "rows": baseline_rows}]
    epoch_numbers = [item["epoch"] for item in summary["epochs"]]
    if not epoch_numbers or epoch_numbers != list(range(1, len(epoch_numbers) + 1)):
        raise ValueError("Epochs must be complete, unique and sequential")
    for item in summary["epochs"]:
        prediction_path = run_dir / f"epoch_{item['epoch']:02}/validation/predictions.jsonl"
        rows, metrics = checked_predictions(prediction_path, item["validation"])
        if set(by_id) != {row["id"] for row in rows}:
            raise ValueError("Baseline and adapter do not use identical validation IDs")
        if any(by_id[row["id"]]["input"] != row["input"] or by_id[row["id"]]["expected"] != row["expected"] for row in rows):
            raise ValueError("Inputs/gold labels changed between model stages")
        stages.append({"epoch": item["epoch"], "metrics": metrics, "rows": rows})
        sources.append(prediction_path)
    if summary["best_finetuned_epoch"] not in epoch_numbers:
        raise ValueError("Selected adapter epoch is missing")
    logs = read_lines(run_dir / "training_log.jsonl")
    if not logs:
        raise ValueError("No real training loss log available")
    previous = 0
    for row in logs:
        if (row["optimizer_step"] <= previous or row["epoch"] not in epoch_numbers
                or not math.isfinite(row["mean_loss"]) or row["mean_loss"] < 0):
            raise ValueError("Invalid/non-finite or non-increasing training log")
        previous = row["optimizer_step"]
    if {row["epoch"] for row in logs} != set(epoch_numbers):
        raise ValueError("Training logs do not cover the recorded epochs")
    if logs[-1]["optimizer_step"] != summary["epochs"][-1]["optimizer_steps"]:
        raise ValueError("Incomplete optimizer-step log")
    return {"run_dir": run_dir, "summary": summary, "config": config, "integrity": integrity,
            "stages": stages, "logs": logs, "source_sha256": {str(path.relative_to(run_dir)): file_hash(path) for path in sources}}


def render(run_dir, output_dir=None, smoothing_window=20, dpi=300):
    data = load_run(run_dir)
    if not isinstance(dpi, int) or dpi < 150:
        raise ValueError("Report PNG requires dpi >= 150")
    logs, stages, summary = data["logs"], data["stages"], data["summary"]
    losses = [row["mean_loss"] for row in logs]
    smooth = rolling_mean(losses, smoothing_window)
    output = Path(output_dir) if output_dir else data["run_dir"] / "figures"
    if output.exists():
        raise FileExistsError("Figure directory exists; choose --output-dir to preserve previous figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    best = next(stage for stage in stages if stage["epoch"] == summary["best_finetuned_epoch"])
    before, after = stages[0]["metrics"], best["metrics"]
    dataset = Path(data["config"]["dataset_dir"]).name
    n = before["count"]
    blue, teal, grey = "#2468A0", "#008C82", "#8D99A8"
    files, captions = [], []
    scope = f"{dataset} · Validation n={n} · Chưa đánh giá test"

    def finish(fig, stem, caption):
        fig.text(.015, .015, scope, fontsize=9, color="#526071", ha="left")
        for extension in ("png", "pdf", "svg"):
            filename = stem + "." + extension
            fig.savefig(output / filename, dpi=dpi, bbox_inches="tight", facecolor="white")
            files.append(filename)
        captions.append({"stem": stem, "caption": caption})
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(9.4, 5.3))
        steps = [row["optimizer_step"] for row in logs]
        ax.plot(steps, losses, color="#BCD1E4", linewidth=.8, label="Loss mỗi optimizer step")
        ax.plot(steps, smooth, color=blue, linewidth=1.8, label=f"Trung bình trượt {smoothing_window} step")
        for epoch in range(1, len(stages) - 1):
            boundary = max(row["optimizer_step"] for row in logs if row["epoch"] == epoch)
            ax.axvline(boundary, color=grey, linestyle="--", linewidth=.9)
            ax.text(boundary, .95, f" Hết epoch {epoch}", transform=ax.get_xaxis_transform(), fontsize=9, va="top")
        ax.set(title="Loss trong quá trình huấn luyện A1", xlabel="Optimizer step", ylabel="Weighted completion-only loss")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False)
        fig.subplots_adjust(bottom=.17)
        finish(fig, "01_training_loss", "Loss huấn luyện theo optimizer step, kèm trung bình trượt. Đây là weighted completion-only loss; không phải validation loss hay độ chính xác.")

        fig, ax = plt.subplots(figsize=(9.4, 5.3))
        epochs = [stage["epoch"] for stage in stages]
        for key, label, color, marker in (("control_exact", "Đúng toàn bộ điều phối", teal, "o"),
                ("decision_correct", "Đúng quyết định", blue, "s"), ("decision_macro_f1", "Macro-F1 quyết định", "#986B3F", "^")):
            values = [stage["metrics"][key] if key == "decision_macro_f1" else stage["metrics"]["rates"][key] for stage in stages]
            ax.plot(epochs, values, marker=marker, color=color, linewidth=1.8, label=label)
        ax.axvline(best["epoch"], linestyle=":", color=grey, label=f"Epoch được chọn: {best['epoch']}")
        ax.set_xticks(epochs, ["Model nền" if e == 0 else f"Epoch {e}" for e in epochs])
        ax.set(title="Kết quả validation qua các epoch", ylabel="Giá trị chỉ tiêu", ylim=(0, 1.08))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.subplots_adjust(bottom=.17)
        finish(fig, "02_validation_progress", "So sánh model nền và từng epoch trên cùng validation. Epoch được chọn theo quy tắc trong summary.json; không nhất thiết là epoch có accuracy cao nhất. Macro-F1 được biểu diễn trên thang phần trăm để dễ so sánh.")

        fig, ax = plt.subplots(figsize=(10.6, 5.7))
        positions = np.arange(len(METRICS))
        for offset, metrics, label, color in ((-.19, before, "Model nền", grey), (.19, after, f"Adapter epoch {best['epoch']}", teal)):
            values = [metrics[key] if key == "decision_macro_f1" else metrics["rates"][key] for _, key in METRICS]
            bars = ax.bar(positions + offset, values, width=.36, label=label, color=color)
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(positions, [label for label, _ in METRICS])
        ax.set(title="So sánh trước và sau fine-tune A1", ylabel="Giá trị chỉ tiêu", ylim=(0, 1.15))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.02))
        fig.subplots_adjust(bottom=.22)
        finish(fig, "03_before_after", "Các chỉ tiêu validation trước fine-tune và tại adapter được chọn. Control exact yêu cầu đúng schema, ràng buộc và toàn bộ trường điều phối; không tự chấm nghĩa resolved_request/message. Không diễn giải thành độ chính xác y khoa.")

        matrices = [confusion_counts(stages[0]["rows"]), confusion_counts(best["rows"])]
        vmax = max(max(row) for matrix in matrices for row in matrix) or 1
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
        labels = [*DECISIONS, "invalid"]
        image = None
        for ax, matrix, title in zip(axes, matrices, ("Model nền", f"Adapter epoch {best['epoch']}")):
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
            for row in range(len(DECISIONS)):
                for col in range(len(labels)):
                    value = matrix[row][col]
                    ax.text(col, row, str(value) if value else "·", ha="center", va="center", fontsize=11,
                            color="white" if value > vmax * .55 else "#263444")
            ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(len(DECISIONS)), DECISIONS, fontsize=10)
            ax.set(title=title, xlabel="Quyết định dự đoán")
        axes[0].set_ylabel("Nhãn kỳ vọng")
        fig.suptitle("Ma trận nhầm lẫn quyết định A1", fontsize=16)
        fig.subplots_adjust(left=.11, right=.89, bottom=.25, top=.84, wspace=.34)
        cax = fig.add_axes([.92, .25, .018, .59])
        fig.colorbar(image, cax=cax, label="Số mẫu")
        finish(fig, "04_confusion_matrix", "Ma trận số lượng mẫu: hàng là nhãn kỳ vọng, cột là nhãn dự đoán. Cột invalid giữ các đầu ra JSON/schema lỗi trong mẫu số. Cùng thang màu cho trước và sau; mỗi ma trận chứa toàn bộ validation.")

        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        positions = np.arange(len(DECISIONS))
        for offset, metrics, label, color in ((-.18, before, "Model nền", grey), (.18, after, f"Adapter epoch {best['epoch']}", teal)):
            values = [metrics["per_decision"][key]["f1"] for key in DECISIONS]
            bars = ax.barh(positions + offset, values, height=.33, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, fontsize=10)
        ax.set_yticks(positions, [f"{key} (n={after['per_decision'][key]['support']})" for key in DECISIONS])
        ax.invert_yaxis()
        ax.set(title="F1 theo từng loại quyết định", xlabel="F1", xlim=(0, 1.15))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
        fig.subplots_adjust(left=.24, bottom=.27)
        finish(fig, "05_f1_by_decision", "F1 từng loại quyết định trước/sau fine-tune; n là số mẫu kỳ vọng của lớp trong validation. Các lớp ít mẫu có kết quả chưa ổn định. Precision và recall tương ứng được xuất trong CSV.")

        counts = data["integrity"]["dataset_check"]["by_split"]
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        values = [counts[split] for split in ("train", "validation", "test")]
        bars = ax.bar(["Train", "Validation", "Test"], values, color=[blue, teal, grey], width=.58)
        ax.bar_label(bars, padding=5, fontsize=13)
        ax.set(title="Phân chia dữ liệu gốc", ylabel="Số mẫu duy nhất", ylim=(0, max(values) * 1.17))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        fig.subplots_adjust(bottom=.18)
        finish(fig, "06_dataset_splits", "Số mẫu duy nhất theo tập dữ liệu, lấy từ kiểm tra dữ liệu đã lưu trong run. Không bao gồm lượt replay/oversampling. Việc thống kê số mẫu test không phải đánh giá model trên test.")

    figure_data = {"run_dir": str(data["run_dir"]), "model": data["config"]["base_model"], "dataset": dataset,
        "validation_count": n, "selected_epoch": best["epoch"], "split": "validation", "test_evaluated": False,
        "smoothing_window": smoothing_window,
        "training": [{"optimizer_step": row["optimizer_step"], "epoch": row["epoch"], "raw_loss": row["mean_loss"], "smoothed_loss": avg}
                     for row, avg in zip(logs, smooth)],
        "stages": [{"epoch": stage["epoch"], "metrics": stage["metrics"]} for stage in stages],
        "confusion_labels": {"rows": DECISIONS, "columns": [*DECISIONS, "invalid"]},
        "confusion_matrices": {"before": matrices[0], "after": matrices[1]}, "data_split_counts": counts}
    write_json(output / "figure_data.json", figure_data)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["epoch", "metric", "value", "validation_n"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            metrics = stage["metrics"]
            for key in [*FLAGS, "decision_macro_f1"]:
                writer.writerow({"epoch": stage["epoch"], "metric": key,
                    "value": metrics[key] if key == "decision_macro_f1" else metrics["rates"][key], "validation_n": n})
            for key in DECISIONS:
                for metric in ("precision", "recall", "f1", "support"):
                    writer.writerow({"epoch": stage["epoch"], "metric": key + "." + metric,
                                     "value": metrics["per_decision"][key][metric], "validation_n": n})
    text = ["# Chú thích hình kết quả A1", "", f"Run: `{data['run_dir'].name}`. Dữ liệu: `{dataset}`. Model: `{data['config']['base_model']}`.",
            f"Validation: {n} mẫu. Adapter được chọn: epoch {best['epoch']}. Test chưa được đánh giá.", "",
            f"Ảnh PNG {dpi} dpi dùng cho Word; PDF/SVG là bản vector. Số liệu được đối chiếu lại với raw predictions, không sinh số minh họa.", ""]
    for index, item in enumerate(captions, 1):
        text.append(f"**Hình {index}.** {item['caption']}\n\nTệp: `{item['stem']}.png`.\n")
    text.extend(["Nhãn quyết định: route = chuyển tác tử/công cụ; clarify = hỏi làm rõ; handoff = chuyển người phụ trách; out_of_scope = ngoài phạm vi; respond = phản hồi xã giao.",
                 "", "Validation đã dùng để chọn checkpoint, nên chưa phải kiểm thử độc lập. Dữ liệu tổng hợp chưa được chuyên gia duyệt độc lập. Chỉ tiêu điều phối chưa đánh giá đầy đủ nghĩa resolved_request/message hoặc chất lượng y khoa."])
    (output / "captions_vi.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_json(output / "figures_manifest.json", {"run_dir": str(data["run_dir"]), "dataset": dataset,
        "selected_epoch": best["epoch"], "validation_count": n, "png_dpi": dpi, "matplotlib": matplotlib.__version__,
        "source_sha256": data["source_sha256"], "scope": "Completed real training logs, validation only; no model loaded or run by plotting.",
        "files_sha256": {name: file_hash(output / name) for name in [*files, "figure_data.json", "metrics.csv", "captions_vi.md"]}})
    return {"figures_dir": str(output.resolve()), "figures": len(captions), "formats": ["png", "pdf", "svg"],
            "validation_count": n, "selected_epoch": best["epoch"], "dataset": dataset}
