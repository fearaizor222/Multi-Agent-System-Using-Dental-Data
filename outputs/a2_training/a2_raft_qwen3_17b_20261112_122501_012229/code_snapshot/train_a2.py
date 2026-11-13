"""Shared A2 QLoRA loop for Qwen3, VinaLLaMA and MedScholar on Modal."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from src.agents.contracts import GeneratedDraft

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODELS = {
    "qwen3_17b": {"repo": MODEL, "revision": REVISION, "name": "Qwen3-1.7B"},
    "vinallama_27b": {"repo": "vilm/vinallama-2.7b-chat",
        "revision": "b31d5f1306494b2bf10ecb0c6031077af3f5b39a", "name": "VinaLLaMA-2.7B-chat"},
    "medscholar_15b": {"repo": "yasserrmd/MedScholar-1.5B",
        "revision": "c4a837a196bbf0fb08396fd8176bfc6bf10b647f", "name": "MedScholar-1.5B"},
}
CHATML = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' }}"
          "{% if not loop.last or m['role'] != 'assistant' %}{{ '\\n' }}{% endif %}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
MEDSCHOLAR_PROMPT = ("{% for m in messages %}{% if m['role'] == 'system' %}"
    "{{ '### Question:\\n' + m['content'] + '\\n\\n' }}{% elif m['role'] == 'user' %}"
    "{{ m['content'] + '\\n\\n### Answer:\\n' }}{% elif m['role'] == 'assistant' %}"
    "{{ m['content'] + eos_token }}{% endif %}{% endfor %}")
DEFAULT_DATA = ROOT / "data/processed/a2_drafting_v3"
STATUSES = ["draft", "insufficient_evidence"]


def configure_tokenizer(tokenizer, model_id):
    if model_id not in MODELS:
        raise ValueError("Unsupported model: " + model_id)
    if model_id == "vinallama_27b":
        tokenizer.chat_template = CHATML
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<pad>"
    elif model_id == "medscholar_15b":
        # The model author's format ends the prompt with ### Answer:, not ChatML.
        tokenizer.chat_template = MEDSCHOLAR_PROMPT
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = "<|endoftext|>"
    if not tokenizer.chat_template:
        raise ValueError("Missing reviewed chat template")
    for token in (tokenizer.eos_token, tokenizer.pad_token):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError("Invalid special token: " + str(token))
    tokenizer.padding_side = "right"
    return tokenizer


def parse_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key: " + k)
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("Invalid JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def output_model(record):
    schema = record['messages'][1]['content'].split('\nJSON schema:\n',1)
    if len(schema)==2 and '"GroundedClaim"' in schema[1]:
        from src.training.a2_raft_protocol import GroundedDraft
        if parse_json(schema[1]) != GroundedDraft.model_json_schema():
            raise ValueError('Unknown grounded output schema')
        return GroundedDraft
    return GeneratedDraft


def parse_sample(record):
    messages = record["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Mỗi mẫu cần đúng system/user/assistant")
    if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError("Nội dung messages bị rỗng")
    payload = parse_json(messages[1]["content"].split("\nJSON schema:\n", 1)[0])
    model_class = output_model(record)
    gold = model_class.model_validate(parse_json(messages[2]["content"]))
    evidence = payload["evidence"]
    ids = [item["evidence_id"] for item in evidence]
    allowed = payload["allowed_evidence_ids"]
    if (not payload["question"].strip() or not 1 <= len(evidence) <= 3
            or len(ids) != len(set(ids)) or len(allowed) != len(set(allowed)) or set(ids) != set(allowed)
            or any(not item["content"].strip() or not item["title"].strip() for item in evidence)):
        raise ValueError("Câu hỏi hoặc evidence không hợp lệ")
    if len(gold.claims) > 2 or any(not set(c.evidence_ids) <= set(ids) for c in gold.claims):
        raise ValueError("Đáp án có quá 2 claims hoặc trích dẫn ngoài context")
    if model_class is not GeneratedDraft:
        from scripts.a2_raft_io import validate_quotes
        validate_quotes(gold,payload)
    return payload, gold.model_dump()


def load_data(directory=DEFAULT_DATA):
    """Read a stable snapshot; regroup by document to avoid overlapping-context validation."""
    directory = Path(directory)
    if (directory/'manifest.json').exists() and json.loads((directory/'manifest.json').read_text(encoding='utf-8')).get('version','').startswith('a2-raft-pilot-'):
        from scripts.a2_raft_io import load_raft_data
        return load_raft_data(directory)
    if (directory / 'manifest.json').exists() and json.loads((directory / 'manifest.json').read_text(encoding='utf-8')).get('version', '').startswith(('a2-v2-grounded-','a2-v3-repaired-','a2-v4-audited-')):
        from scripts.a2_dataset_v2 import load_v2
        return load_v2(directory, parse_sample)
    names = ["train.jsonl", "val.jsonl", "train.metadata.jsonl", "val.metadata.jsonl"]
    raw = {name: (directory / name).read_bytes() for name in names}
    parsed = {name: [parse_json(l) for l in content.decode("utf-8-sig").splitlines() if l.strip()]
              for name, content in raw.items()}
    if not parsed["train.jsonl"] or not parsed["val.jsonl"]:
        raise ValueError("Train hoặc val còn rỗng. Đợi sinh dữ liệu hoàn tất trước khi train.")
    groups, seen, chunk_docs = {}, set(), {}
    for split in ("train", "val"):
        records, metadata = parsed[split + ".jsonl"], parsed[split + ".metadata.jsonl"]
        if len(records) != len(metadata):
            raise ValueError("JSONL và metadata chưa khớp; có thể dữ liệu vẫn đang được ghi")
        for record, meta in zip(records, metadata):
            payload, gold = parse_sample(record)
            doc, chunks = meta["document_id"], meta["chunk_ids"]
            if (not doc or not chunks or len(chunks) != len(payload["evidence"])
                    or meta["question"] != payload["question"] or meta["status"] != gold["status"]):
                raise ValueError("Metadata không khớp câu hỏi/nhãn")
            key = json.dumps(record["messages"][:2], ensure_ascii=False, sort_keys=True)
            if key in seen:
                raise ValueError("Có mẫu đầu vào trùng; cần lọc trước khi train")
            seen.add(key)
            for chunk in chunks:
                if chunk in chunk_docs and chunk_docs[chunk] != doc:
                    raise ValueError("Một chunk thuộc nhiều document_id")
                chunk_docs[chunk] = doc
            groups.setdefault(doc, []).append({"messages": record["messages"], "metadata": meta})
    if any((directory / name).read_bytes() != content for name, content in raw.items()):
        raise ValueError("Dữ liệu thay đổi trong lúc đọc; hãy đợi sinh xong")
    docs = sorted(groups)
    if len(docs) < 2:
        raise ValueError("Cần ít nhất hai document_id để tách train/validation")
    random.Random(42).shuffle(docs)
    held_out = set(docs[:max(1, round(len(docs) * .15))])
    packet = {"train": [], "validation": []}
    for doc, rows in groups.items():
        packet["validation" if doc in held_out else "train"].extend(rows)
    packet["info"] = {"source_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in raw.items()},
                      "split_method": "document_id disjoint; seed=42; approximately 15% documents held out",
                      "validation_documents": sorted(held_out),
                      "counts": {s: len(packet[s]) for s in ("train", "validation")},
                      "labels": {s: dict(Counter(parse_sample(r)[1]["status"] for r in packet[s]))
                                 for s in ("train", "validation")}}
    return packet


def encode_sample(tokenizer, record, max_length=3072):
    messages = record["messages"]
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template không khớp prefix/đáp án")
    if len(full) > max_length:
        raise ValueError(f"Mẫu dài {len(full)} token, vượt {max_length}; không tự cắt")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


def score_answer(record, raw):
    payload, gold = parse_sample(record)
    model_class = output_model(record)
    result = {"schema_valid": False, "status_correct": False, "citation_ids_valid": False,
              "exact_match": False, "error": None, "expected_status": gold["status"], "predicted_status": None}
    if model_class is not GeneratedDraft:
        count = sum(len(c['supporting_quotes']) for c in gold['claims'])
        result.update(quotes_verbatim=False,reference_quotes_count=count,reference_quotes_covered=0,
                      reference_proof_coverage=0. if count else None,reference_evidence_match=False)
    try:
        predicted = model_class.model_validate(parse_json(raw))
        if len(predicted.claims) > 2:
            raise ValueError("Quá 2 nhận định")
        result.update(schema_valid=True, predicted_status=predicted.status, status_correct=predicted.status == gold["status"],
                      citation_ids_valid=all(bool(c.evidence_ids) and set(c.evidence_ids) <= set(payload["allowed_evidence_ids"]) for c in predicted.claims),
                      exact_match=predicted.model_dump() == gold)
        if model_class is not GeneratedDraft:
            from scripts.a2_raft_io import score_quotes
            result.update(score_quotes(predicted,gold,payload))
    except (ValueError, TypeError, KeyError) as exc:
        result["error"] = str(exc)[:300]
    return result


def summarize_results(results):
    if not results:
        raise ValueError("Không có kết quả validation")
    keys = ("schema_valid", "status_correct", "citation_ids_valid", "exact_match")
    metrics = {k: sum(r[k] for r in results) / len(results) for k in keys}
    if all('quotes_verbatim' in r for r in results):
        drafts = [r for r in results if r['schema_valid'] and r['predicted_status']=='draft']
        count = sum(r['reference_quotes_count'] for r in results)
        metrics.update(quoted_draft_count=len(drafts),
            draft_quotes_verbatim_fraction=sum(r['quotes_verbatim'] for r in drafts)/len(drafts) if drafts else None,
            reference_evidence_match=sum(r['reference_evidence_match'] for r in results)/len(results),
            reference_proof_coverage=sum(r['reference_quotes_covered'] for r in results)/count if count else None,
            grounding_metric_scope='Verbatim quote and reference-span checks only; semantic support requires source-reading.')
    columns = STATUSES + ["invalid"]
    matrix = [[0] * len(columns) for _ in STATUSES]
    for row in results:
        prediction = row["predicted_status"] if row["schema_valid"] else "invalid"
        matrix[STATUSES.index(row["expected_status"])][columns.index(prediction)] += 1
    per_status = {}
    for i, label in enumerate(STATUSES):
        support = sum(matrix[i])
        predicted = sum(r[i] for r in matrix)
        precision = matrix[i][i] / predicted if predicted else 0.
        recall = matrix[i][i] / support if support else 0.
        per_status[label] = {"support": support, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    return {**metrics, "count": len(results), "per_status": per_status,
            "status_macro_f1": sum(r["f1"] for r in per_status.values()) / len(STATUSES),
            "confusion_matrix": matrix, "confusion_labels": {"rows": STATUSES, "columns": columns}}


def evaluate(model, tokenizer, records, directory, max_new_tokens=384):
    import torch
    directory.mkdir(parents=True, exist_ok=False)
    model.eval(); model.config.use_cache = True
    results = []
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, record in enumerate(records, 1):
            ids = tokenizer.apply_chat_template(record["messages"][:2], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            row = {"sample_id": record.get("metadata", {}).get("id"),
                   "question": parse_sample(record)[0]["question"], "raw_output": text,
                   "expected": parse_sample(record)[1],
                   "generation_tokens": int(output.shape[1] - ids.shape[1]),
                   "generation_limit": max_new_tokens, **score_answer(record, text)}
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            if i == 1 or i % 10 == 0 or i == len(records):
                print(f"{directory.name}: {i}/{len(records)}", flush=True)
    metrics = summarize_results(results)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def plot_results(logs, before, after, directory, model_name="Qwen3-1.7B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=False)
    train = [r for r in logs if "loss" in r]
    val = [r for r in logs if "eval_loss" in r]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], label="Train loss")
    ax.plot([r["step"] for r in val], [r["eval_loss"] for r in val], "o-", label="Validation loss")
    ax.set(xlabel="Optimizer step", ylabel="Assistant-token cross-entropy", title=f"A2 - {model_name}: loss")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=300); plt.close(fig)
    keys = ["schema_valid", "status_correct", "citation_ids_valid", "exact_match"]
    labels = ["Valid JSON/schema", "Correct status", "Valid citation IDs", "Exact reference match"]
    if 'reference_proof_coverage' in before and 'reference_proof_coverage' in after:
        keys = ["schema_valid", "status_correct", "citation_ids_valid", "draft_quotes_verbatim_fraction", "reference_proof_coverage"]
        labels = ["Valid schema", "Correct status", "Valid citation IDs", "Verbatim quotes\n(predicted drafts)", "Reference proof\ncoverage (lexical)"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, metrics, name in ((-.18, before, "Before"), (.18, after, "After")):
        bars = ax.bar([i + offset for i in range(len(keys))], [(metrics[k] or 0) * 100 for k in keys], width=.35, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(range(len(keys)), labels, fontsize=9)
    ax.set(ylabel="Validation (%)", ylim=(0, 118), title=f"A2 - {model_name}: before / after")
    ax.legend(); fig.tight_layout()
    fig.savefig(directory / "before_after.png", dpi=300); plt.close(fig)

    matrices = [before["confusion_matrix"], after["confusion_matrix"]]
    maximum = max(value for matrix in matrices for row in matrix for value in row) or 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, matrix, name in zip(axes, matrices, ("Before", "After")):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum, aspect="auto")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > maximum * .5 else "black")
        ax.set_xticks(range(3), STATUSES + ["invalid"], rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(2), STATUSES, fontsize=9)
        ax.set(title=name, xlabel="Predicted status")
    axes[0].set_ylabel("Expected status")
    fig.suptitle(f"A2 - {model_name}: confusion matrix (validation)")
    fig.subplots_adjust(left=.18, right=.89, bottom=.28, top=.84, wspace=.65)
    cax = fig.add_axes([.92, .28, .018, .56])
    fig.colorbar(image, cax=cax, label="Samples")
    fig.savefig(directory / "04_confusion_matrix.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, metrics, name in ((-.18, before, "Before"), (.18, after, "After")):
        bars = ax.bar([i + offset for i in range(2)],
            [metrics["per_status"][s]["f1"] * 100 for s in STATUSES], width=.35, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(range(2), [f"{s}\n(n={after['per_status'][s]['support']})" for s in STATUSES])
    ax.set(title=f"A2 - {model_name}: F1 by response status (validation)", ylabel="F1 (%)", ylim=(0, 118))
    ax.legend(); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); fig.tight_layout()
    fig.savefig(directory / "05_f1_by_decision.png", dpi=300); plt.close(fig)


def train(packet, output, epochs=2, commit=lambda: None, model_id="qwen3_17b"):
    import shutil
    import time
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if epochs < 1:
        raise ValueError("epochs phải dương")
    spec = MODELS[model_id]
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    save("run_status.json", {"status": "running"})
    try:
        set_seed(42)
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(spec["repo"],
            revision=spec["revision"], trust_remote_code=False), model_id)
        encoded = {s: [encode_sample(tokenizer, r) for r in packet[s]] for s in ("train", "validation")}
        save("dataset_info.json", packet["info"])
        snapshot = output / 'code_snapshot'
        snapshot.mkdir()
        for name in ('train_a2.py', 'a2_dataset_v2.py', 'a2_raft_io.py', 'modal_train_a2.py'):
            source = Path(__file__).parent / name
            if source.exists(): shutil.copy2(source, snapshot / name)
        if packet['info'].get('output_contract') == 'a2_grounded_quotes_v1':
            shutil.copy2(ROOT/'src/training/a2_raft_protocol.py',snapshot/'a2_raft_protocol.py')
        for split in ("train", "validation"):
            with (output / (split + ".jsonl")).open("w", encoding="utf-8") as f:
                for row in packet[split]: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Cần GPU CUDA hỗ trợ BF16")
        config = {"model": spec["repo"], "revision": spec["revision"], "model_id": model_id,
                  "chat_template": tokenizer.chat_template, "eos_token_id": tokenizer.eos_token_id,
                  "epochs": epochs, "learning_rate": 1e-4,
                  "max_length": 3072, "batch_size": 2, "gradient_accumulation_steps": 4,
                  "lora_r": 16, "lora_alpha": 32, "seed": 42, "quantization": "NF4 double quantization / BF16",
                  "checkpoint_selection": "lowest validation loss", "gpu": torch.cuda.get_device_name(0)}
        generation_tokens = 512 if packet['info'].get('dataset_version', '').startswith(('a2-v2-','a2-v3-','a2-v4-','a2-raft-')) else 384
        config.update(max_new_tokens=generation_tokens, assistant_only_loss=True, test_used_for_selection=False)
        save("config.json", config)
        commit()
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["revision"], trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
        print("Evaluate baseline", flush=True)
        before = evaluate(model, tokenizer, packet["validation"], output / "before", generation_tokens)
        commit()
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
        class Persist(TrainerCallback):
            def on_save(self, args, state, control, **kwargs): commit()
        args = TrainingArguments(output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
            per_device_train_batch_size=2, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-4, warmup_ratio=.05, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
            report_to="none", seed=42, data_seed=42, optim="adamw_torch", label_names=["labels"])
        trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
            data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
            processing_class=tokenizer, callbacks=[Persist()])
        training_started = time.monotonic()
        result = trainer.train()
        train_seconds = time.monotonic() - training_started
        trainer.save_model(str(output / "adapter")); tokenizer.save_pretrained(output / "adapter")
        logs = trainer.state.log_history
        save("training_log.json", logs)
        save('train_metrics.json', result.metrics)
        commit()
        after = evaluate(trainer.model, tokenizer, packet["validation"], output / "after", generation_tokens)
        save("metrics.json", {"before": before, "after": after,
            "train_seconds": train_seconds,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "scope": "Synthetic validation. Citation-ID validity does not prove factual support; exact match is strict wording agreement. No clinical correctness or independent test evaluation."})
        plot_results(logs, before, after, output / "figures", model_name=spec["name"])
        save("run_status.json", {"status": "completed"})
        return str(output)
    except BaseException as exc:
        save("run_status.json", {"status": "failed", "error": str(exc)[:1000]})
        raise
    finally:
        commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu A2 cục bộ; không gọi Modal/model")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--check-data", action="store_true", required=True)
    print(json.dumps(load_data(parser.parse_args().dataset_dir)["info"], ensure_ascii=False, indent=2))
