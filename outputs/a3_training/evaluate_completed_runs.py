"""Read-only local verification of downloaded A3 run results; never loads weights."""
import json
import math
from pathlib import Path
from itertools import combinations
from scripts.train_a3 import parse_sample, score_answer, summarize_results

ROOT = Path(__file__).resolve().parent
report = {"models": {}, "pairwise": {}}
all_predictions = {}
identities = []
validation_reference = None

for folder in sorted(ROOT.glob("a3_*")):
    if not folder.is_dir() or not (folder / "metrics.json").exists():
        continue
    read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    status, metrics, info, config = [read(name) for name in (
        "run_status.json", "metrics.json", "dataset_info.json", "config.json")]
    assert status["status"] == "completed", folder
    records = [json.loads(line) for line in (folder / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    metas = [json.loads(line) for line in (folder / "validation.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(metas) == 191
    identity = [(row["messages"], meta) for row, meta in zip(records, metas)]
    if validation_reference is None:
        validation_reference = identity
    else:
        assert identity == validation_reference, "Different validation contents/order"
    identities.append(info["source_sha256"])
    scores = {}
    token_stats = {}
    for stage in ("before", "after"):
        predictions = [json.loads(line) for line in (folder / stage / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(predictions) == 191 and len({r["sample_id"] for r in predictions}) == 191
        for record, meta, pred in zip(records, metas, predictions):
            assert pred["sample_id"] == meta["id"]
            assert pred["expected"] == parse_sample(record)[1]
        scored = [score_answer(record, pred["raw_output"]) for record, pred in zip(records, predictions)]
        recomputed = summarize_results(scored)
        assert recomputed == metrics[stage], (folder.name, stage)
        scores[stage] = recomputed
        tokens = [r["generation_tokens"] for r in predictions]
        token_stats[stage] = {"mean": sum(tokens) / len(tokens), "max": max(tokens), "at_limit_512": sum(n == 512 for n in tokens)}
        if stage == "after":
            all_predictions[config["model_id"]] = scored
    logs = read("training_log.json")
    assert all(math.isfinite(float(row[key])) for row in logs for key in ("loss", "eval_loss", "grad_norm") if key in row)
    report["models"][config["model_id"]] = {
        "run": folder.name, "scores": scores, "token_stats": token_stats,
        "train_seconds": metrics["train_seconds"],
        "validation_losses": [{k: row[k] for k in ("epoch", "eval_loss", "step") if k in row} for row in logs if "eval_loss" in row],
        "best_checkpoint": metrics["best_checkpoint"],
    }
assert len(report["models"]) == 3
assert all(identity == identities[0] for identity in identities)
for a, b in combinations(all_predictions, 2):
    rows = zip(all_predictions[a], all_predictions[b])
    pair = list(rows)
    report["pairwise"][a + " vs " + b] = {
        "assessment_a_only_correct": sum(x["assessment_exact_match"] and not y["assessment_exact_match"] for x, y in pair),
        "assessment_b_only_correct": sum(y["assessment_exact_match"] and not x["assessment_exact_match"] for x, y in pair),
        "both_assessments_correct": sum(x["assessment_exact_match"] and y["assessment_exact_match"] for x, y in pair),
    }
report["scope"] = "Same supplied 191-example synthetic validation, used for checkpoint selection; known label and evidence-exposure limitations remain. No independent clinical test."
(ROOT / "comparison_verified_20261108.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
for model, row in report["models"].items():
    after = row["scores"]["after"]
    print(model, {key: round(after[key] * 100, 2) for key in (
        "schema_valid", "draft_support_correct", "public_support_correct", "quote_set_match", "assessment_exact_match")})
    print("Macro F1", {field: round(value["macro_f1"] * 100, 2) for field, value in after["decisions"].items()})
    print("Validation losses", row["validation_losses"], "Train seconds", row["train_seconds"])
    print("Generation", row["token_stats"])
print("Pairwise", report["pairwise"])
