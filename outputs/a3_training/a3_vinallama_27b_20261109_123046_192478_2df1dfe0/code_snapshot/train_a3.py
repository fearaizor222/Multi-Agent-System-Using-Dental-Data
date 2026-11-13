"""A3 QLoRA: shared training/evaluation for three Modal entrypoints.

The local --check-data command never loads model weights or submits a GPU job.
The existing train/val split and reference labels are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/a3_verification_v2"
LABELS = ["supported", "contradicted", "insufficient"]
FIELDS = ["draft_support", "public_support"]
MAX_LENGTH = 3072
MODELS = {
    "qwen3_17b": {
        "repo": "Qwen/Qwen3-1.7B", "name": "Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "vinallama_27b": {
        "repo": "vilm/vinallama-2.7b-chat", "name": "VinaLLaMA-2.7B-chat",
        "revision": "b31d5f1306494b2bf10ecb0c6031077af3f5b39a",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "mediphi_38b": {
        "repo": "microsoft/MediPhi-Instruct", "name": "MediPhi-Instruct 3.8B",
        "revision": "8a52461577949784b8c447d557fa6b98a7d272df",
        "targets": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    },
}
CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' }}"
    "{% if not loop.last or m['role'] != 'assistant' %}{{ '\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class SelectedAssessment(BaseModel):
    """Same output contract as the A3 quote-selection runtime; no RAG imports needed."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    draft_support: Literal["supported", "contradicted", "insufficient"]
    public_support: Literal["supported", "contradicted", "insufficient"]
    quote_ids: list[str] = Field(max_length=3)
    reason: str = Field(max_length=240)

    @model_validator(mode="after")
    def has_quotes(self):
        if self.public_support != "insufficient" and not self.quote_ids:
            raise ValueError("Kết luận hỗ trợ hoặc mâu thuẫn cần chọn quote_ids")
        if len(self.quote_ids) != len(set(self.quote_ids)):
            raise ValueError("quote_ids không được trùng")
        return self


def parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Invalid JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def valid_quote_selection(predicted, payload):
    allowed = {q["quote_id"] for q in payload["public_quote_options"]}
    return (set(predicted["quote_ids"]) <= allowed
            and (predicted["public_support"] != "insufficient" or not predicted["quote_ids"]))


def parse_sample(record):
    messages = record["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Mỗi mẫu cần đúng system/user/assistant")
    if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError("Nội dung messages bị rỗng")
    payload = parse_json(messages[1]["content"].split("\nJSON schema:\n", 1)[0])
    gold = SelectedAssessment.model_validate(parse_json(messages[2]["content"])).model_dump()
    if not payload["question"].strip() or not payload["claim"].strip():
        raise ValueError("Câu hỏi/nhận định bị rỗng")
    internal = payload["internal_evidence"]
    if not internal or any(not item["content"].strip() for item in internal):
        raise ValueError("Thiếu bằng chứng nội bộ")
    options = payload["public_quote_options"]
    ids = [q["quote_id"] for q in options]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Thiếu hoặc trùng mã trích dẫn")
    for q in options:
        if (not q["text"].strip() or not q["evidence_id"].strip()
                or not q["quote_id"].startswith(q["evidence_id"] + ".Q")):
            raise ValueError("Lựa chọn trích dẫn không hợp lệ")
    if "public_sources" in payload:
        sources = payload["public_sources"]
        source_ids = [s["evidence_id"] for s in sources]
        if (len(source_ids) != len(set(source_ids)) or set(source_ids) != {q["evidence_id"] for q in options}
                or any(not s["title"].strip() or not isinstance(s["heading_path"], list) for s in sources)):
            raise ValueError("Ngữ cảnh tiêu đề/mục nguồn không khớp lựa chọn trích dẫn")
    if not valid_quote_selection(gold, payload):
        raise ValueError("Đáp án trích dẫn ngoài context hoặc insufficient có quote_ids")
    return payload, gold


def load_data(directory=DEFAULT_DATA):
    """Take a stable snapshot without relabeling or resplitting the user's dataset."""
    directory = Path(directory)
    names = ["train.jsonl", "val.jsonl", "train.metadata.jsonl", "val.metadata.jsonl"]
    raw = {name: (directory / name).read_bytes() for name in names}
    parsed = {name: [parse_json(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
              for name, content in raw.items()}
    packet = {"train": [], "validation": []}
    seen = set()
    for source, destination in (("train", "train"), ("val", "validation")):
        records, metas = parsed[source + ".jsonl"], parsed[source + ".metadata.jsonl"]
        if not records or len(records) != len(metas):
            raise ValueError(f"{source}: dữ liệu rỗng hoặc metadata chưa khớp")
        for line, (record, meta) in enumerate(zip(records, metas), 1):
            try:
                payload, gold = parse_sample(record)
                if (any(meta[key] != gold[key] for key in FIELDS + ["quote_ids", "reason"])
                        or any(meta[key] != payload[key] for key in ("question", "claim"))
                        or not meta["public_chunk_id"] or not meta["public_document_id"]):
                    raise ValueError("Metadata không khớp nội dung/nhãn")
                key = json.dumps(record["messages"][:2], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    raise ValueError("Đầu vào trùng trong train/validation")
                seen.add(key)
                packet[destination].append({"messages": record["messages"], "metadata": meta})
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{source}.jsonl dòng {line}: {exc}") from exc
    if any((directory / name).read_bytes() != content for name, content in raw.items()):
        raise ValueError("Dữ liệu thay đổi trong lúc đọc")
    train_chunks = {row["metadata"][key] for row in packet["train"]
                    for key in ("public_chunk_id", "distractor_chunk_id") if row["metadata"].get(key)}
    exposed = sum(row["metadata"]["public_chunk_id"] in train_chunks for row in packet["validation"])
    packet["info"] = {
        "dataset_version": directory.name,
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()},
        "split_method": "Preserve supplied train.jsonl / val.jsonl; no resplit or relabeling",
        "counts": {s: len(packet[s]) for s in packet},
        "labels": {s: {field: dict(Counter(parse_sample(r)[1][field] for r in packet[s]))
                       for field in FIELDS} for s in packet},
        "validation_main_chunks_seen_in_train_including_distractors": exposed,
        "scope": "Synthetic reference labels; validation used for checkpoint selection, not independent test or clinical validation.",
    }
    return packet


def configure_tokenizer(tokenizer, model_id):
    if model_id not in MODELS:
        raise ValueError("Unsupported model: " + model_id)
    if model_id == "vinallama_27b":
        tokenizer.chat_template = CHATML
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<pad>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("Model thiếu chat template")
    for token in (tokenizer.eos_token, tokenizer.pad_token):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError("Invalid special token: " + str(token))
    tokenizer.padding_side = "right"
    return tokenizer


def encode_sample(tokenizer, record, max_length=MAX_LENGTH):
    messages = record["messages"]
    kwargs = {"tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template không khớp prefix/đáp án")
    if len(full) > max_length:
        raise ValueError(f"Mẫu dài {len(full)} token, vượt {max_length}; không tự cắt")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


def score_answer(record, raw):
    payload, gold = parse_sample(record)
    result = {"json_valid": False, "schema_valid": False, "quote_ids_valid": False,
              "quote_set_match": False, "both_labels_correct": False,
              "assessment_exact_match": False, "full_json_exact_match": False,
              "expected": gold, "predicted": None, "error": None}
    for field in FIELDS:
        result[field + "_correct"] = False
    try:
        decoded = parse_json(raw)
        result["json_valid"] = True
        predicted = SelectedAssessment.model_validate(decoded).model_dump()
        result.update(schema_valid=True, predicted=predicted,
                      quote_ids_valid=valid_quote_selection(predicted, payload))
        for field in FIELDS:
            result[field + "_correct"] = predicted[field] == gold[field]
        result["both_labels_correct"] = all(result[f + "_correct"] for f in FIELDS)
        result["quote_set_match"] = result["quote_ids_valid"] and set(predicted["quote_ids"]) == set(gold["quote_ids"])
        result["assessment_exact_match"] = result["both_labels_correct"] and result["quote_set_match"]
        result["full_json_exact_match"] = predicted == gold
    except (ValueError, TypeError, KeyError) as exc:
        result["error"] = str(exc)[:300]
    return result


def summarize_results(results):
    if not results:
        raise ValueError("Không có kết quả validation")
    keys = ["json_valid", "schema_valid", "quote_ids_valid", "quote_set_match", "both_labels_correct",
            "assessment_exact_match", "full_json_exact_match"] + [f + "_correct" for f in FIELDS]
    metrics = {key: sum(row[key] for row in results) / len(results) for key in keys}
    metrics.update(count=len(results), decisions={})
    for field in FIELDS:
        columns = LABELS + ["invalid"]
        matrix = [[0] * len(columns) for _ in LABELS]
        for row in results:
            predicted = row["predicted"][field] if row["schema_valid"] else "invalid"
            matrix[LABELS.index(row["expected"][field])][columns.index(predicted)] += 1
        per_label = {}
        for i, label in enumerate(LABELS):
            support, count = sum(matrix[i]), sum(r[i] for r in matrix)
            precision = matrix[i][i] / count if count else 0.
            recall = matrix[i][i] / support if support else 0.
            per_label[label] = {"support": support, "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}
        metrics["decisions"][field] = {
            "confusion_matrix": matrix, "confusion_labels": {"rows": LABELS, "columns": columns},
            "per_label": per_label, "macro_f1": sum(r["f1"] for r in per_label.values()) / len(LABELS)}
    return metrics


def generation_stop_ids(tokenizer, model_id):
    ids = [tokenizer.eos_token_id]
    if model_id == "mediphi_38b":
        end = tokenizer.encode("<|end|>", add_special_tokens=False)
        if len(end) != 1 or end[0] == tokenizer.unk_token_id:
            raise ValueError("MediPhi thiếu token kết thúc lượt")
        ids += end
    return list(dict.fromkeys(ids))


def load_evaluation_cache(records, path):
    """Reuse a contiguous prefix; only an interrupted final JSON line may be discarded."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    results = []
    for index, line in enumerate(lines):
        try:
            row = parse_json(line.decode("utf-8"))
        except ValueError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"Cache có JSON hỏng ở dòng {index + 1}: {path}")
        if index >= len(records):
            raise ValueError("Cache dài hơn validation")
        record = records[index]
        payload, gold = parse_sample(record)
        if (row.get("sample_id") != record["metadata"]["id"] or row.get("expected") != gold
                or row.get("question") != payload["question"] or not isinstance(row.get("raw_output"), str)):
            raise ValueError(f"Cache không khớp mẫu validation thứ {index + 1}")
        # Recompute scores instead of trusting saved metric fields.
        results.append({**row, **score_answer(record, row["raw_output"])})
    return results


def evaluate(model, tokenizer, records, directory, model_id, resume=False):
    import torch
    directory.mkdir(parents=True, exist_ok=resume)
    model.eval()
    model.config.use_cache = True
    path = directory / "predictions.jsonl"
    results = load_evaluation_cache(records, path) if resume else []
    if results:
        print(f"{directory.name}: reuse {len(results)}/{len(records)} saved predictions", flush=True)
    # Atomically remove only an incomplete final line before appending new answers.
    temp = directory / "predictions.resume.tmp"
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    temp.replace(path)
    with path.open("a", encoding="utf-8") as stream:
        for i, record in enumerate(records[len(results):], len(results) + 1):
            ids = tokenizer.apply_chat_template(record["messages"][:2], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=512, do_sample=False, temperature=None, top_p=None, top_k=None,
                    eos_token_id=generation_stop_ids(tokenizer, model_id), pad_token_id=tokenizer.pad_token_id)
            raw = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            row = {"sample_id": record["metadata"]["id"],
                   "question": parse_sample(record)[0]["question"], "raw_output": raw,
                   "generation_tokens": output.shape[1] - ids.shape[1], **score_answer(record, raw)}
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if i == 1 or i % 10 == 0 or i == len(records):
                print(f"{directory.name}: {i}/{len(records)}", flush=True)
    metrics = summarize_results(results)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def plot_results(logs, before, after, directory, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=False)
    train, val = [r for r in logs if "loss" in r], [r for r in logs if "eval_loss" in r]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], label="Train loss")
    ax.plot([r["step"] for r in val], [r["eval_loss"] for r in val], "o-", label="Validation loss")
    ax.set(xlabel="Optimizer step", ylabel="Assistant-token cross-entropy", title=f"A3 - {model_name}: loss")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=300); plt.close(fig)

    keys = ["schema_valid", "draft_support_correct", "public_support_correct", "quote_set_match", "assessment_exact_match"]
    names = ["Valid schema", "Correct draft label", "Correct public label", "Exact quote set", "Labels + quotes"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, metrics, name in ((-.18, before, "Before"), (.18, after, "After")):
        bars = ax.bar([i + offset for i in range(len(keys))], [metrics[k] * 100 for k in keys], width=.35, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(range(len(keys)), names, fontsize=9)
    ax.set(ylabel="Validation (%)", ylim=(0, 118), title=f"A3 - {model_name}: before / after")
    ax.legend(); fig.tight_layout()
    fig.savefig(directory / "before_after.png", dpi=300); plt.close(fig)

    maximum = max(v for m in (before, after) for f in FIELDS
                  for row in m["decisions"][f]["confusion_matrix"] for v in row) or 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for i, field in enumerate(FIELDS):
        for j, (metrics, stage) in enumerate(((before, "Before"), (after, "After"))):
            ax = axes[i, j]
            matrix = metrics["decisions"][field]["confusion_matrix"]
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum, aspect="auto")
            for a, row in enumerate(matrix):
                for b, value in enumerate(row):
                    ax.text(b, a, str(value), ha="center", va="center", color="white" if value > maximum / 2 else "black")
            ax.set_xticks(range(4), LABELS + ["invalid"], rotation=20, ha="right", fontsize=8)
            ax.set_yticks(range(3), LABELS, fontsize=9)
            ax.set(title=f"{field} - {stage}", xlabel="Predicted", ylabel="Reference")
    fig.colorbar(image, ax=axes, label="Samples", shrink=.8)
    fig.suptitle(f"A3 - {model_name}: confusion matrices (validation)")
    fig.savefig(directory / "04_confusion_matrix.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, field in zip(axes, FIELDS):
        for offset, metrics, name in ((-.18, before, "Before"), (.18, after, "After")):
            scores = metrics["decisions"][field]["per_label"]
            bars = ax.bar([i + offset for i in range(3)], [scores[s]["f1"] * 100 for s in LABELS], width=.35, label=name)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(range(3), [f"{s}\n(n={after['decisions'][field]['per_label'][s]['support']})" for s in LABELS], fontsize=9)
        ax.set(title=field, ylabel="F1 (%)", ylim=(0, 118))
        ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.suptitle(f"A3 - {model_name}: F1 by label (validation)")
    fig.savefig(directory / "05_f1_by_decision.png", dpi=300); plt.close(fig)


def validate_resume_name(name, model_id):
    if name and (not re.fullmatch(r"a3_[A-Za-z0-9_]+", name) or not name.startswith("a3_" + model_id + "_")):
        raise ValueError("resume-run phải là tên run của đúng model, không phải đường dẫn")
    return name


def resume_state(output, packet, spec, epochs):
    """Check the saved run identity before reusing any weights or predictions."""
    config = parse_json((output / "config.json").read_text(encoding="utf-8"))
    info = parse_json((output / "dataset_info.json").read_text(encoding="utf-8"))
    if (config["model"] != spec["repo"] or config["revision"] != spec["revision"]
            or config["epochs"] != epochs or config["max_length"] != MAX_LENGTH
            or info["source_sha256"] != packet["info"]["source_sha256"]):
        raise ValueError("Run cũ khác model/revision/epochs/dữ liệu; không thể tiếp tục")
    status = parse_json((output / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") == "completed":
        return {"completed": True, "adapter": False, "checkpoint": None}
    adapter = output / "adapter"
    # This file is written only after train() + saving the selected adapter succeed.
    trained = ((output / "train_metrics.json").exists()
               and (adapter / "adapter_model.safetensors").exists()
               and (adapter / "adapter_config.json").exists())
    checkpoints = []
    for path in (output / "checkpoints").glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "adapter_config.json", "adapter_model.safetensors")
        if match and all((path / name).is_file() for name in required):
            state = parse_json((path / "trainer_state.json").read_text(encoding="utf-8"))
            if state["global_step"] == int(match[1]):
                checkpoints.append((int(match[1]), path))
    return {"completed": False, "adapter": trained,
            "checkpoint": str(max(checkpoints)[1]) if checkpoints else None}


def train(packet, output, epochs=2, commit=lambda: None, model_id="qwen3_17b", resume=False):
    import shutil
    import time
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if epochs < 1:
        raise ValueError("epochs phải dương")
    spec = MODELS[model_id]
    output = Path(output)
    if resume:
        recovery = resume_state(output, packet, spec, epochs)
        if recovery["completed"]:
            print("Run already completed:", output, flush=True)
            return str(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        recovery = {"adapter": False, "checkpoint": None}
    started = time.monotonic()
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    def phase(name):
        save("run_status.json", {"status": "running", "phase": name})
        print("A3 phase:", name, flush=True)
    try:
        phase("tokenize")
        set_seed(42)
        save("dataset_info.json", packet["info"])
        for split in ("train", "validation"):
            with (output / (split + ".jsonl")).open("w", encoding="utf-8") as f, \
                 (output / (split + ".metadata.jsonl")).open("w", encoding="utf-8") as meta:
                for row in packet[split]:
                    f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
                    meta.write(json.dumps(row["metadata"], ensure_ascii=False) + "\n")
        from datetime import datetime
        code_dir = output / ("code_snapshot_resume_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") if resume else "code_snapshot")
        code_dir.mkdir()
        shutil.copy2(__file__, code_dir / "train_a3.py")
        for path in Path(__file__).parent.glob("modal_train_a3*.py"):
            shutil.copy2(path, code_dir / path.name)
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(spec["repo"],
            revision=spec["revision"], trust_remote_code=False), model_id)
        encoded = {}
        lengths = {}
        for split in ("train", "validation"):
            encoded[split] = []
            for row in packet[split]:
                parse_sample(row)
                try:
                    encoded[split].append(encode_sample(tokenizer, row))
                except ValueError as exc:
                    raise ValueError(f"{split}/{row['metadata']['id']}: {exc}") from exc
            sizes = sorted(len(row["input_ids"]) for row in encoded[split])
            lengths[split] = {"min": sizes[0], "max": sizes[-1], "count": len(sizes), "truncated": 0}
        save("token_lengths.json", lengths)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Cần GPU CUDA hỗ trợ BF16")
        config = {"model": spec["repo"], "revision": spec["revision"], "model_id": model_id,
            "chat_template": tokenizer.chat_template, "eos_token_id": tokenizer.eos_token_id,
            "generation_stop_ids": generation_stop_ids(tokenizer, model_id), "epochs": epochs,
            "learning_rate": 1e-4, "max_length": MAX_LENGTH, "max_new_tokens": 512,
            "batch_size": 1, "gradient_accumulation_steps": 8, "effective_batch_size": 8,
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": .05, "target_modules": spec["targets"],
            "seed": 42, "quantization": "NF4 double quantization / BF16", "assistant_only_loss": True,
            "checkpoint_selection": "lowest validation loss", "gpu": torch.cuda.get_device_name(0)}
        save("config.json", config)
        commit()
        phase("load_base_model")
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["revision"], trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
        phase("baseline_validation")
        before = evaluate(model, tokenizer, packet["validation"], output / "before", model_id, resume=resume)
        commit()
        if recovery["adapter"]:
            print("Saved adapter found: skip training, finish validation only", flush=True)
            # Match the original post-training evaluation: k-bit preparation casts
            # frozen non-quantized weights to FP32. Do not enable checkpointing here.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            model = PeftModel.from_pretrained(model, str(output / "adapter"), is_trainable=False)
            logs = parse_json((output / "training_log.json").read_text(encoding="utf-8"))
            training = parse_json((output / "train_metrics.json").read_text(encoding="utf-8"))
            train_seconds = training.get("train_runtime")
            best_checkpoint = None
            for path in sorted((output / "checkpoints").glob("checkpoint-*/trainer_state.json")):
                state = parse_json(path.read_text(encoding="utf-8"))
                if state.get("best_model_checkpoint"):
                    best_checkpoint = state["best_model_checkpoint"]
            phase("adapter_validation")
            after = evaluate(model, tokenizer, packet["validation"], output / "after", model_id, resume=True)
            save("metrics.json", {"before": before, "after": after, "model": spec,
                "best_checkpoint": best_checkpoint, "train_seconds": train_seconds,
                "elapsed_seconds_this_attempt": time.monotonic() - started,
                "scope": packet["info"]["scope"], "recovered_without_retraining": True,
                "metric_notes": "Reference-label agreement; not independent test or clinical correctness. Assessment match ignores reason wording."})
            figures = output / "figures"
            if figures.exists():
                figures.rename(output / ("figures_interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
            plot_results(logs, before, after, figures, spec["name"])
            save("run_status.json", {"status": "completed", "recovered_without_retraining": True})
            return str(output)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias="none",
            target_modules=spec["targets"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        class Persist(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                save("training_log.json", state.log_history)
                commit()
        args = TrainingArguments(output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, warmup_ratio=.05, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
            report_to="none", seed=42, data_seed=42, optim="adamw_torch", label_names=["labels"],
            prediction_loss_only=True)
        trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
            data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
            processing_class=tokenizer, callbacks=[Persist()])
        phase("training")
        training_started = time.monotonic()
        if recovery["checkpoint"]:
            print("Resume optimizer checkpoint:", recovery["checkpoint"], flush=True)
        result = trainer.train(resume_from_checkpoint=recovery["checkpoint"])
        train_seconds = time.monotonic() - training_started
        trainer.save_model(str(output / "adapter"))
        tokenizer.save_pretrained(output / "adapter")
        logs = trainer.state.log_history
        save("training_log.json", logs)
        save("train_metrics.json", result.metrics)
        commit()
        phase("adapter_validation")
        after = evaluate(trainer.model, tokenizer, packet["validation"], output / "after", model_id, resume=resume)
        save("metrics.json", {"before": before, "after": after, "model": spec,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "train_seconds": train_seconds, "elapsed_seconds": time.monotonic() - started,
            "scope": packet["info"]["scope"],
            "metric_notes": "Reference-label agreement, not clinical correctness. Quote-ID validity does not prove entailment. Assessment match ignores reason wording; invalid schema remains in confusion matrices. Macro-F1 averages all three fixed labels, including zero-support labels."})
        phase("plot_png")
        figures = output / "figures"
        if figures.exists():
            figures.rename(output / ("figures_interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        plot_results(logs, before, after, figures, spec["name"])
        save("run_status.json", {"status": "completed", "elapsed_seconds": time.monotonic() - started})
        return str(output)
    except BaseException as exc:
        save("run_status.json", {"status": "failed", "error": str(exc)[:1000]})
        raise
    finally:
        commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu A3; không gọi Modal/model")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--check-data", action="store_true", required=True)
    print(json.dumps(load_data(parser.parse_args().dataset_dir)["info"], ensure_ascii=False, indent=2))
