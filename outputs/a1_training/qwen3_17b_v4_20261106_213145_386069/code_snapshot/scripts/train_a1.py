"""Baseline validation → QLoRA A1 → validation từng epoch; không chấm test."""
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.check_a1_environment import encode_assistant_sample, completion_only_loss
from src.training.a1_evaluation import evaluate, selection_score
from src.training.a1_runtime import ROOT, file_hash, load_base, package_versions, verify_inputs, write_json
from src.training.a1_strategy import plan_epoch, assistant_weights, training_score


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_config(config):
    if config["validation_split"] != "validation" or config["evaluate_test_split"]:
        raise ValueError("Lượt pilot chỉ dùng validation, giữ test riêng")
    if config["epochs"] < 1 or config["gradient_accumulation_steps"] < 1:
        raise ValueError("epochs và gradient_accumulation_steps phải dương")
    if config["generation"]["do_sample"] or config["generation"]["enable_thinking"]:
        raise ValueError("So sánh này dùng greedy, thinking tắt")
    if not config["completion_logits_only"] or not config["frozen_embeddings_compute_dtype"] or config["save_embedding_layers"]:
        raise ValueError("Cần giữ cấu hình tiết kiệm VRAM và chỉ lưu adapter")
    if config.get("selection_profile","control_exact") not in ("control_exact","critical_recall"):
        raise ValueError("selection_profile không hợp lệ")


def run(config, output, baseline_from=None):
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    validate_config(config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if "cuda_memory_fraction" in config:
        if not 0 < config["cuda_memory_fraction"] <= 1:
            raise ValueError("cuda_memory_fraction phải thuộc (0,1]")
        torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], 0)
    status = {"started_at": now(), "status": "running", "phase": "initializing",
              "test_evaluated": False, "output_dir": str(output.resolve())}
    status_path = output / "run_status.json"

    def update(**kwargs):
        status.update(kwargs, updated_at=now())
        write_json(status_path, status)

    update()
    try:
        integrity = verify_inputs(config)
        write_json(output / "input_integrity.json", integrity)
        write_json(output / "config.json", config)
        versions = package_versions()
        code_files = [Path(__file__), ROOT / "src/training/a1_runtime.py", ROOT / "src/training/a1_evaluation.py",
                      ROOT / "src/data/a1_dataset.py", ROOT / "scripts/check_a1_environment.py",
                      ROOT / "scripts/check_a1_dataset.py", ROOT / "scripts/prepare_a1_dataset.py",
                      ROOT / "src/training/a1_strategy.py"]
        if baseline_from:
            code_files.append(ROOT / "src/training/a1_baseline_cache.py")
        write_json(output / "runtime.json", {"python": sys.executable, "packages": versions,
            "gpu": torch.cuda.get_device_name(0), "gpu_mib": torch.cuda.get_device_properties(0).total_memory/2**20,
            "torch_cuda": torch.version.cuda,
            "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in code_files}})
        for path in code_files:
            destination = output / "code_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
        dataset_dir = ROOT / config["dataset_dir"]
        training_rows = read_rows(dataset_dir / "train.metadata.jsonl")
        validation = read_rows(dataset_dir / "validation.metadata.jsonl")
        if any(row["split"] != "train" for row in training_rows) or any(row["split"] != "validation" for row in validation):
            raise ValueError("Metadata split không đúng")
        # Reject impossible sampling quotas before loading weights or running baseline.
        plans=[plan_epoch(training_rows,config["seed"]+epoch,config.get("sampling")) for epoch in range(1,config["epochs"]+1)]
        write_json(output/"sampling_plan.json",[detail for _,detail in plans])
        model, tokenizer, dtype = load_base(config)
        torch.cuda.reset_peak_memory_stats()

        def evaluation_progress(phase):
            def callback(index, total, metrics):
                update(phase=phase, evaluated_cases=index, evaluation_total=total,
                       control_correct=metrics["correct"]["control_exact"])
                if index == 1 or index % 5 == 0 or index == total:
                    print(f"{phase}: {index}/{total}, control={metrics['correct']['control_exact']}/{index}", flush=True)
            return callback

        if baseline_from:
            from src.training.a1_baseline_cache import checked_baseline
            baseline, baseline_rows, provenance = checked_baseline(baseline_from, config, integrity, validation, versions,
                allow_train_only_change=config.get("baseline_identical_holdout_only",False))
            print("Replaying three baseline examples before reusing saved validation baseline...", flush=True)
            _, probe = evaluate(model, tokenizer, validation[:3], output / "baseline_replay_probe", config["generation"], dtype,
                                evaluation_progress("baseline_replay_probe"))
            if any(row["raw_output"] != saved["raw_output"] for row, saved in zip(probe, baseline_rows[:3])):
                raise RuntimeError("Baseline replay không trùng; không được dùng cache")
            provenance["three_raw_replays_match"] = True
            write_json(output / "baseline_provenance.json", provenance)
            shutil.copytree(Path(baseline_from) / "baseline_validation", output / "baseline_validation")
        else:
            update(phase="baseline_validation")
            print("Baseline: same local NF4 checkpoint, validation only.", flush=True)
            baseline, baseline_rows = evaluate(model, tokenizer, validation, output / "baseline_validation",
                                               config["generation"], dtype, evaluation_progress("baseline_validation"))
        print("Baseline complete:", baseline["correct"], flush=True)
        encoded = []
        for row in training_rows:
            ids, labels, prompt_length = encode_assistant_sample(tokenizer, row["messages"])
            if len(ids) > config["max_length"]:
                raise ValueError("Không cắt mẫu quá dài: " + row["id"])
            weights=assistant_weights(tokenizer,row["messages"],ids,prompt_length,config.get("token_weighting"))
            encoded.append((ids, labels, prompt_length, weights))
        model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none", task_type="CAUSAL_LM"))
        trainable = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
            raise RuntimeError("Chỉ tham số LoRA được huấn luyện")
        steps_per_epoch = math.ceil(len(plans[0][0])/config["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * config["epochs"]
        warmup_steps = max(1, math.ceil(config["warmup_ratio"] * total_steps))
        optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"])

        def lr_scale(step):
            if step < warmup_steps:
                return (step+1)/warmup_steps
            return max(0.0, (total_steps-step)/(total_steps-warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        best = None
        epochs = []
        global_step = 0
        start_training = time.perf_counter()
        with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(1, config["epochs"]+1):
                order = plans[epoch-1][0]
                draws=len(order)
                epoch_losses = []
                epoch_dir = output / f"epoch_{epoch:02}"
                epoch_dir.mkdir()
                update(phase="training", epoch=epoch, epochs=config["epochs"], samples_completed=0,
                       train_samples=draws, unique_train_samples=len(encoded), optimizer_steps=global_step, optimizer_steps_total=total_steps)
                model.train()
                model.config.use_cache = False
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(order), config["gradient_accumulation_steps"]):
                    chunk = order[offset:offset+config["gradient_accumulation_steps"]]
                    step_losses = []
                    for index in chunk:
                        ids, labels, prompt_length, weights = encoded[index]
                        batch = {"input_ids": torch.tensor([ids], device="cuda"),
                            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
                            "labels": torch.tensor([labels], device="cuda")}
                        with torch.autocast("cuda", dtype=dtype):
                            token_weights=torch.tensor([weights],device="cuda") if config.get("token_weighting") else None
                            loss = completion_only_loss(model, batch, prompt_length, token_weights)
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Loss không hữu hạn: " + training_rows[index]["id"])
                        value = float(loss.detach())
                        (loss/len(chunk)).backward()
                        step_losses.append(value)
                        epoch_losses.append(value)
                        del loss, batch, token_weights
                    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"], error_if_nonfinite=True)
                    used_lr = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    entry = {"epoch": epoch, "optimizer_step": global_step,
                        "samples_completed": offset+len(chunk), "mean_loss": sum(step_losses)/len(step_losses),
                        "learning_rate": used_lr, "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": round(time.perf_counter()-start_training, 2),
                        "allocated_mib": round(torch.cuda.memory_allocated()/2**20, 1)}
                    log.write(json.dumps(entry)+"\n")
                    log.flush()
                    update(**entry, phase="training", optimizer_steps=global_step)
                    if global_step%10==0 or offset+len(chunk)==draws:
                        print(f"Train epoch {epoch}: {entry['samples_completed']}/{draws}, step {global_step}/{total_steps}, loss={entry['mean_loss']:.4f}", flush=True)
                adapter_dir = epoch_dir / "adapter"
                model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
                write_json(epoch_dir / "adapter_manifest.json", {"base_model": config["base_model"], "revision": config["revision"],
                    "epoch": epoch, "optimizer_steps": global_step, "dataset_manifest_sha256": integrity["dataset_manifest_sha256"],
                    "files_sha256": {path.name: file_hash(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()}})
                metrics, predictions = evaluate(model, tokenizer, validation, epoch_dir / "validation",
                    config["generation"], dtype, evaluation_progress(f"epoch_{epoch:02}_validation"))
                result = {"epoch": epoch, "mean_training_loss": sum(epoch_losses)/len(epoch_losses),
                          "optimizer_steps": global_step, "adapter_dir": str(adapter_dir.resolve()), "validation": metrics}
                epochs.append(result)
                if best is None or training_score(metrics,config.get("selection_profile","control_exact")) > training_score(best["validation"],config.get("selection_profile","control_exact")):
                    best = result
                write_json(output / "epochs.json", epochs)
                print(f"Epoch {epoch} validation complete:", metrics["correct"], flush=True)
        del optimizer, scheduler
        # Nạp adapter đã lưu vào tên riêng để kiểm tra artifact, không chỉ model trong RAM.
        update(phase="reload_probe")
        best_dir = Path(best["adapter_dir"])
        model.load_adapter(best_dir, adapter_name="reloaded", is_trainable=False)
        model.set_adapter("reloaded")
        best_predictions = read_rows(best_dir.parent / "validation/predictions.jsonl")
        probe_metrics, probe_rows = evaluate(model, tokenizer, validation[:3], output / "reload_probe",
            config["generation"], dtype, evaluation_progress("reload_probe"))
        reload_matches = all(row["raw_output"] == expected["raw_output"] for row, expected in zip(probe_rows, best_predictions[:3]))
        if not reload_matches:
            raise RuntimeError("Adapter nạp lại không tái hiện ba kết quả validation đã lưu")
        improved = training_score(best["validation"],config.get("selection_profile","control_exact")) > training_score(baseline,config.get("selection_profile","control_exact"))
        critical_gate=(bool(training_score(best["validation"],"critical_recall")[0])
                       if config.get("selection_profile")=="critical_recall" else None)
        memory = {"peak_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20, 1),
                  "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20, 1),
                  "dedicated_vram_mib": torch.cuda.get_device_properties(0).total_memory/2**20}
        if memory["peak_allocated_mib"] >= memory["dedicated_vram_mib"]:
            raise RuntimeError("Lượt chạy vượt mức cấp phát VRAM riêng")
        summary = {"status": "completed", "baseline_validation": baseline, "epochs": epochs,
            "best_finetuned_epoch": best["epoch"], "best_adapter_dir": best["adapter_dir"],
            "adapter_reload_probe_matches": reload_matches, "finetuned_improves_over_baseline": improved,
            "recommended_candidate": ("research_only" if critical_gate is False else "adapter" if improved else "base_model"),
            "critical_checkpoint_gate_passed":critical_gate,
            "selection_rule": ("critical_recall gate, min(clarify,handoff) recall, macro F1, control_exact, context/schema; ties keep earlier"
                               if config.get("selection_profile")=="critical_recall" else "control_exact, decision_macro_f1, context_valid, schema_valid; ties keep earlier epoch/base"),
            "selection_profile": config.get("selection_profile","control_exact"),
            "memory": memory, "test_evaluated": False, "human_reviewed_training_records": 0,
            "baseline_reused_from": str(Path(baseline_from).resolve()) if baseline_from else None,
            "scope": "Pilot on synthetic A1 ingress data; not a deployed model or independent clinical/generalization evaluation.",
            "finished_at": now()}
        write_json(output / "summary.json", summary)
        update(status="completed", phase="completed", best_finetuned_epoch=best["epoch"],
               finetuned_improves_over_baseline=improved, best_adapter_dir=best["adapter_dir"])
        print(json.dumps({key: summary[key] for key in ("status", "best_finetuned_epoch", "best_adapter_dir", "finetuned_improves_over_baseline", "memory")}, ensure_ascii=False, indent=2), flush=True)
        return summary
    except BaseException as exc:
        update(status="failed", error=type(exc).__name__+": "+str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/a1_training_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-from", type=Path, help="Nguồn baseline đã hoàn tất; phải trùng dữ liệu/model và qua replay")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Thư mục đầu ra đã tồn tại; chọn một lượt mới")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(config, args.output_dir, args.baseline_from)


if __name__ == "__main__":
    main()
