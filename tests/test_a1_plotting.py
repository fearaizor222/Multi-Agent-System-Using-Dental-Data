"""Plot integrity and invalid-output accounting using labelled test fixtures only."""
import json
from copy import deepcopy

import pytest

from src.training.a1_evaluation import score_response, summarize
from src.training.a1_plotting import confusion_counts, load_run, render, rolling_mean
from src.training.a1_runtime import write_json


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "TEST_FIXTURE_NOT_A_REAL_RESULT"
    run.mkdir()
    route = {"decision": "route", "targets": ["A2"], "service_topics": [], "image_ids": [], "resolved_request": "Giải thích máng nhai.", "message": None}
    handoff = {"decision": "handoff", "targets": [], "service_topics": [], "image_ids": [], "resolved_request": None, "message": "Cần người phụ trách đánh giá."}
    respond = {**handoff, "decision": "respond", "message": "Xin chào."}

    def row(number, expected, raw):
        context = {"message": "Máng nhai là gì?" if number == 1 else "Quyết định điều trị riêng cho tôi.",
                   "history": [], "attachments": [], "capabilities": {"A2": True, "A5": False, "lookup_clinic_service": False}}
        return {"id": f"fixture_{number}", "group_id": f"fixture_group_{number}", "category": "knowledge",
                "split": "validation", "input": context, "expected": expected, "raw_output": raw,
                "elapsed_seconds": 1.0, **score_response(context, expected, raw)}

    before = [row(1, route, "{broken"), row(2, handoff, json.dumps(respond, ensure_ascii=False))]
    after = [row(1, route, json.dumps(route, ensure_ascii=False)), row(2, handoff, json.dumps(handoff, ensure_ascii=False))]
    for relative, rows in (("baseline_validation", before), ("epoch_01/validation", after)):
        directory = run / relative
        directory.mkdir(parents=True)
        (directory / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(run / "config.json", {"base_model": "fixture-model", "dataset_dir": "data/fixture_dataset"})
    write_json(run / "input_integrity.json", {"dataset_check": {"by_split": {"train": 4, "validation": 2, "test": 1}}})
    write_json(run / "summary.json", {"status": "completed", "test_evaluated": False, "baseline_validation": summarize(before),
        "epochs": [{"epoch": 1, "optimizer_steps": 2, "validation": summarize(after)}], "best_finetuned_epoch": 1})
    (run / "training_log.jsonl").write_text(''.join(json.dumps({"epoch": 1, "optimizer_step": i, "mean_loss": loss})+'\n'
        for i, loss in ((1, .9), (2, .7))), encoding="utf-8")
    return run


def test_invalid_json_remains_in_confusion_denominator(completed_run):
    data = load_run(completed_run)
    matrix = confusion_counts(data["stages"][0]["rows"])
    assert len(matrix) == 5 and len(matrix[0]) == 6
    assert matrix[0][-1] == 1
    assert sum(sum(row) for row in matrix) == 2


def test_moving_average_uses_only_available_past_points():
    assert rolling_mean([1., 2., 3., 4.], 2) == [1., 1.5, 2.5, 3.5]
    with pytest.raises(ValueError):
        rolling_mean([float("nan")], 2)


def test_changed_input_between_stages_is_rejected(completed_run):
    p = completed_run / "epoch_01/validation/predictions.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["input"]["message"] += "changed"
    p.write_text("".join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Inputs/gold"):
        load_run(completed_run)


def test_forged_scores_are_not_plotted(completed_run):
    p = completed_run / "baseline_validation/predictions.jsonl"
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    rows[0]["control_exact"] = True
    p.write_text("".join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Saved scores"):
        load_run(completed_run)


def test_failed_run_cannot_create_report_figures(completed_run, tmp_path):
    p = completed_run / "summary.json"
    summary = json.loads(p.read_text(encoding="utf-8"))
    summary["status"] = "failed"
    write_json(p, summary)
    destination = tmp_path / "figures"
    with pytest.raises(ValueError, match="completed"):
        render(completed_run, destination)
    assert not destination.exists()


def test_render_exports_six_figures_all_formats_and_exact_data(completed_run, tmp_path):
    destination = tmp_path / "figures"
    result = render(completed_run, destination, smoothing_window=2, dpi=150)
    assert result["figures"] == 6 and result["dataset"] == "fixture_dataset"
    for extension in ("png", "pdf", "svg"):
        assert len(list(destination.glob("*."+extension))) == 6
    figure_data = json.loads((destination / "figure_data.json").read_text(encoding="utf-8"))
    assert figure_data["validation_count"] == 2 and figure_data["confusion_matrices"]["before"][0][-1] == 1
    assert figure_data["training"][1]["smoothed_loss"] == pytest.approx(.8)
    assert "150 dpi" in (destination / "captions_vi.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        render(completed_run, destination)


def test_modal_can_export_only_png(completed_run, tmp_path):
    destination=tmp_path/'png_only'
    result=render(completed_run,destination,dpi=150,formats=('png',))
    assert result['formats']==['png']
    assert len(list(destination.glob('*.png')))==6
    assert not list(destination.glob('*.pdf')) and not list(destination.glob('*.svg'))
