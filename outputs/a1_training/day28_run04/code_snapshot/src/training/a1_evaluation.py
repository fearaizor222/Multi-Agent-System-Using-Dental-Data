"""Đánh giá đầu ra A1 nguyên bản; không sửa JSON hoặc thay nhãn lỗi."""
import csv
import json
import time
from collections import Counter
from pathlib import Path

from src.data.a1_dataset import A1Decision, validate_pair

DECISIONS = ["route", "clarify", "handoff", "out_of_scope", "respond"]


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Trùng khóa JSON: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Hằng số không thuộc JSON: " + value)

    return json.loads(text.strip(), object_pairs_hook=pairs, parse_constant=constant)


def score_response(context, expected, raw):
    result = {"raw_json_valid": False, "schema_valid": False, "context_valid": False,
              "decision_correct": False, "targets_correct": False, "service_topics_correct": False,
              "image_ids_correct": False, "control_exact": False, "predicted": None, "error": None}
    try:
        payload = strict_json(raw)
        result["raw_json_valid"] = True
        predicted = A1Decision.model_validate(payload).model_dump()
        result.update(schema_valid=True, predicted=predicted)
        result["decision_correct"] = predicted["decision"] == expected["decision"]
        for name in ("targets", "service_topics", "image_ids"):
            result[name + "_correct"] = set(predicted[name]) == set(expected[name])
        validate_pair(context, predicted)
        result["context_valid"] = True
        result["control_exact"] = all(result[key] for key in ("decision_correct", "targets_correct",
                                                              "service_topics_correct", "image_ids_correct"))
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def summarize(rows, completed=True):
    count = len(rows)
    keys = ("raw_json_valid", "schema_valid", "context_valid", "decision_correct", "targets_correct",
            "service_topics_correct", "image_ids_correct", "control_exact")
    correct = {key: sum(bool(row[key]) for row in rows) for key in keys}
    rates = {key: (correct[key]/count if count else 0.0) for key in keys}
    confusion = Counter((row["expected"]["decision"], row["predicted"]["decision"] if row["schema_valid"] else "invalid") for row in rows)
    per_class = {}
    for label in DECISIONS:
        tp = confusion[(label, label)]
        actual = sum(value for (truth, pred), value in confusion.items() if truth == label)
        predicted = sum(value for (truth, pred), value in confusion.items() if pred == label)
        precision, recall = (tp/predicted if predicted else 0.0), (tp/actual if actual else 0.0)
        per_class[label] = {"support": actual, "precision": precision, "recall": recall,
                            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {"count": len(subset), "control_exact": sum(row["control_exact"] for row in subset),
                                "schema_valid": sum(row["schema_valid"] for row in subset)}
    return {"completed": completed, "count": count, "correct": correct, "rates": rates,
            "decision_macro_f1": sum(item["f1"] for item in per_class.values())/len(DECISIONS),
            "per_decision": per_class, "by_category": categories,
            "confusion": [{"expected": truth, "predicted": pred, "count": value} for (truth, pred), value in sorted(confusion.items())],
            "total_generation_seconds": sum(row["elapsed_seconds"] for row in rows),
            "scope": "Synthetic validation control decisions; response wording and resolved_request semantics not automatically judged."}


def selection_score(metrics):
    return (metrics["rates"]["control_exact"], metrics["decision_macro_f1"],
            metrics["rates"]["context_valid"], metrics["rates"]["schema_valid"])


def evaluate(model, tokenizer, records, directory, generation, dtype, progress=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    was_training, old_cache = model.training, model.config.use_cache
    model.eval()
    model.config.use_cache = True
    results = []
    try:
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for number, record in enumerate(records, 1):
                # Chỉ system/user đi vào model; target assistant chỉ dùng sau sinh để chấm.
                prompt = tokenizer.apply_chat_template(record["messages"][:-1], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        max_new_tokens=generation["max_new_tokens"], do_sample=False,
                        temperature=None, top_p=None, top_k=None, use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                torch.cuda.synchronize()
                raw = tokenizer.decode(output[0, prompt.shape[1]:], skip_special_tokens=True)
                scored = score_response(record["input"], record["target"], raw)
                row = {"id": record["id"], "group_id": record["group_id"], "category": record["category"],
                       "split": record["split"], "input": record["input"], "expected": record["target"],
                       "raw_output": raw, "elapsed_seconds": round(time.perf_counter()-start, 4),
                       "generated_tokens": int(output.shape[1]-prompt.shape[1]), **scored}
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if progress:
                    progress(number, len(records), summarize(results, completed=False))
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    metrics = summarize(results)
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (directory / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["id", "category", "message", "expected_decision", "predicted_decision", "control_exact", "schema_valid", "context_valid", "error", "raw_output"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({"id": row["id"], "category": row["category"], "message": row["input"]["message"],
                "expected_decision": row["expected"]["decision"],
                "predicted_decision": row["predicted"]["decision"] if row["schema_valid"] else "invalid",
                **{key: row[key] for key in ("control_exact", "schema_valid", "context_valid", "error", "raw_output")}})
    return metrics, results
