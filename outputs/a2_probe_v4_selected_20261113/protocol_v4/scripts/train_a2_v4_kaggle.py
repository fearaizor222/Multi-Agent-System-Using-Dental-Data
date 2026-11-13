"""Standalone Stage 2 (v4) QLoRA trainer & 0.5-epoch evaluator for Kaggle.

Key improvements over Stage 1:
  - Trains on cleaned `a2_sentence_selection_v4` (1,076 train / 84 val / 80 challenge)
  - Uses `max_length=4096` so zero samples are truncated or dropped
  - Pins single-GPU execution before torch import to avoid Kaggle 2xT4 DataParallel errors
  - Evaluates saved LoRA adapters every 0.5 epoch (0.5, 1.0, 1.5, 2.0) on BOTH
    validation and challenge, selecting the best adapter by composite Val+Challenge
    decision + selection F1/exact metrics (never `eval_loss` or `test`).
"""
from __future__ import annotations

import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

from scripts.a2_sentence_io import parse_sample, plot_results, score_answer, summarize_results
from scripts.train_a2 import MODEL, REVISION, configure_tokenizer, encode_sample, inference_dtype_signature

ROOT = Path(__file__).resolve().parents[1]
INITIAL_ADAPTER_SHA256 = '17de0c132134552a4d3ee3b824397207a5bb9b0a059ce56f54d1df0efc786be5'
DEFAULT_DATASET_DIR = ROOT / 'data/processed/a2_sentence_selection_v4'
DEFAULT_INITIAL_ADAPTER = ROOT / 'outputs/a2_colab_raft_v2/results/adapter'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs/a2_kaggle_sentence_v4/results'
MAX_LENGTH = 4096
MAX_NEW_TOKENS = 384


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_split(dataset_dir: Path, split: str) -> list[dict]:
    records = [
        json.loads(line)
        for line in (dataset_dir / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    meta_path = dataset_dir / f'{split}.metadata.jsonl'
    if meta_path.exists():
        metas = [
            json.loads(line)
            for line in meta_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        if len(metas) != len(records):
            raise ValueError(f'{split}: jsonl/metadata length mismatch ({len(records)} vs {len(metas)})')
        for rec, meta in zip(records, metas):
            rec['metadata'] = meta
    return records


def _load_tokenizer(model_path: str | Path, local_only: bool):
    from transformers import AutoTokenizer

    kwargs = {'trust_remote_code': False, 'local_files_only': local_only}
    if Path(model_path).exists():
        tok = AutoTokenizer.from_pretrained(str(model_path), **kwargs)
    else:
        tok = AutoTokenizer.from_pretrained(str(model_path), revision=REVISION, **kwargs)
    return configure_tokenizer(tok, 'qwen3_17b')


def preflight(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    initial_adapter: Path = DEFAULT_INITIAL_ADAPTER,
    model_path: str | Path | None = None,
    max_length: int = MAX_LENGTH,
    local_only: bool = True,
    expected_adapter_sha256: str = INITIAL_ADAPTER_SHA256,
) -> dict:
    """Validate dataset, adapter SHA, and full-sequence Qwen3 token budgets on CPU."""
    weights = initial_adapter / 'adapter_model.safetensors'
    if not weights.exists():
        raise FileNotFoundError(f'Missing initial adapter weights: {weights}')
    actual_sha = _sha256(weights)
    if actual_sha != expected_adapter_sha256:
        raise ValueError(f'Initial adapter SHA mismatch: {actual_sha} != {expected_adapter_sha256}')

    local_model = ROOT / 'data/models/Qwen--Qwen3-1.7B' / REVISION
    resolved_model = Path(model_path) if model_path is not None else (local_model if local_model.exists() else MODEL)
    tokenizer = _load_tokenizer(resolved_model, local_only=local_only and Path(resolved_model).exists())

    seen_prompts: set[str] = set()
    length_stats: dict[str, dict] = {}
    splits = {sp: _load_split(dataset_dir, sp) for sp in ('train', 'val', 'challenge')}

    for split_name, records in splits.items():
        full_lens: list[int] = []
        asst_lens: list[int] = []
        for rec in records:
            parse_sample(rec)
            p_key = json.dumps(rec['messages'][:2], ensure_ascii=False, sort_keys=True)
            if p_key in seen_prompts:
                raise ValueError(f'Duplicate prompt found across/within splits at {split_name}')
            seen_prompts.add(p_key)

            enc = encode_sample(tokenizer, rec, max_length=max_length)
            full_len = len(enc['input_ids'])
            asst_len = sum(1 for x in enc['labels'] if x != -100)
            full_lens.append(full_len)
            asst_lens.append(asst_len)

        sorted_lens = sorted(full_lens)
        length_stats[split_name] = {
            'count': len(records),
            'max_tokens': max(full_lens),
            'p95_tokens': sorted_lens[min(len(sorted_lens) - 1, int(0.95 * len(sorted_lens)))],
            'max_assistant_tokens': max(asst_lens),
        }

    return {
        'ready': True,
        'dataset_version': 'a2_sentence_selection_v4',
        'max_length': max_length,
        'max_new_tokens': MAX_NEW_TOKENS,
        'initial_adapter_sha256': actual_sha,
        'lengths': length_stats,
        'truncated_samples': 0,
        'weights_loaded': False,
        'training_executed': False,
        'test_used': False,
    }


def native_bf16_supported(cuda) -> bool:
    return (cuda.is_available() and cuda.get_device_capability(0)[0] >= 8
            and cuda.is_bf16_supported(including_emulation=False))


def _load_fresh_quantized_base(
    model_ref: str | Path,
    adapter_path: Path | None = None,
    *,
    is_trainable: bool = False,
):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # T4 can allocate BF16 tensors through emulation but has no native BF16
    # Tensor Cores. Never choose the emulated path for training or evaluation.
    use_bf16 = native_bf16_supported(torch.cuda)
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(json.dumps({'gpu': torch.cuda.get_device_name(0),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'compute_dtype': str(compute_dtype), 'native_bf16': use_bf16,
        'training': is_trainable}, ensure_ascii=False), flush=True)
    load_kwargs = {
        'trust_remote_code': False,
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        'device_map': {'': 0},
        'attn_implementation': 'sdpa',
    }
    if 'dtype' in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        load_kwargs['dtype'] = compute_dtype
    else:
        load_kwargs['torch_dtype'] = compute_dtype
    if not Path(model_ref).exists():
        load_kwargs['revision'] = REVISION

    base = AutoModelForCausalLM.from_pretrained(str(model_ref), **load_kwargs)
    if is_trainable:
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
        )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=is_trainable)
    else:
        model = base

    if is_trainable:
        # PyTorch AMP GradScaler (FP16 on T4/P100) requires trainable LoRA tensors in FP32.
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(True)
                if param.dtype != torch.float32:
                    param.data = param.data.float()
    else:
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = True
    return model, use_bf16


def _breakdown_by_slice(results: list[dict], records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row, rec in zip(results, records):
        meta = rec.get('metadata', {})
        kind = meta.get('kind', 'unknown')
        groups[f'kind:{kind}'].append(row)
        if meta.get('partial_direction') in ('pos_first', 'neg_first'):
            groups[f'partial:{meta["partial_direction"]}'].append(row)
        ev_n = meta.get('evidence_count')
        if ev_n is not None:
            groups[f'ev_count:{ev_n}'].append(row)

    out = {}
    for key, subset in sorted(groups.items()):
        m = summarize_results(subset)
        out[key] = {
            'count': len(subset),
            'schema_valid': m['schema_valid'],
            'status_correct': m['status_correct'],
            'selection_set_exact': m.get('selection_set_exact'),
            'selection_f1_on_answerable': m.get('selection_f1_on_answerable'),
            'valid_sentence_ids': m.get('valid_sentence_ids'),
        }
    return out


def evaluate_adapter_on_splits(
    *,
    model_ref: str | Path,
    tokenizer,
    adapter_path: Path,
    val_records: list[dict],
    challenge_records: list[dict],
    out_dir: Path,
) -> dict:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = _load_fresh_quantized_base(model_ref, adapter_path=adapter_path, is_trainable=False)
    sig = inference_dtype_signature(model)
    if sig['trainable_parameter_tensors'] != 0:
        raise ValueError(f'Expected frozen weights for evaluation, got {sig}')

    metrics_by_split: dict[str, dict] = {}
    try:
        for split_name, records in (('val', val_records), ('challenge', challenge_records)):
            split_dir = out_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            with (split_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                for idx, rec in enumerate(records, 1):
                    ids = tokenizer.apply_chat_template(
                        rec['messages'][:2],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_tensors='pt',
                    ).to('cuda')
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    raw_text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
                    payload, gold = parse_sample(rec)
                    row = {
                        'sample_id': rec.get('metadata', {}).get('id'),
                        'kind': rec.get('metadata', {}).get('kind'),
                        'partial_direction': rec.get('metadata', {}).get('partial_direction'),
                        'evidence_count': len(payload['evidence']),
                        'question': payload['question'],
                        'raw_output': raw_text,
                        'expected': gold,
                        'generation_tokens': int(generated.shape[1] - ids.shape[1]),
                        'generation_limit': MAX_NEW_TOKENS,
                        **score_answer(rec, raw_text),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    if idx == 1 or idx % 20 == 0 or idx == len(records):
                        print(f'[{out_dir.name}/{split_name}] {idx}/{len(records)}', flush=True)

            summary = summarize_results(rows)
            summary['slice_breakdown'] = _breakdown_by_slice(rows, records)
            (split_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            metrics_by_split[split_name] = summary
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return metrics_by_split


def checkpoint_composite(val_m: dict, chal_m: dict) -> float:
    """Compute composite selection score from Validation + Challenge only (never test)."""
    def _split(m: dict) -> float:
        if (m.get('schema_valid') or 0.0) < 0.98 or (m.get('valid_sentence_ids') or 0.0) < 0.98:
            return -1.0
        return (
            0.45 * (m.get('status_macro_f1') or 0.0)
            + 0.35 * (m.get('selection_f1_on_answerable') or 0.0)
            + 0.20 * (m.get('selection_set_exact') or 0.0)
        )

    sv, sc = _split(val_m), _split(chal_m)
    if sv < 0 or sc < 0:
        return min(sv, sc)
    return 0.5 * sv + 0.5 * sc


def train_and_evaluate(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    initial_adapter: Path = DEFAULT_INITIAL_ADAPTER,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path | None = None,
    epochs: float = 2.0,
    learning_rate: float = 8e-5,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_length: int = MAX_LENGTH,
    eval_before: bool = False,
    expected_adapter_sha256: str = INITIAL_ADAPTER_SHA256,
) -> dict:
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for Stage 2 QLoRA training')

    output_dir.mkdir(parents=True, exist_ok=True)
    pf = preflight(
        dataset_dir=dataset_dir,
        initial_adapter=initial_adapter,
        model_path=model_path,
        max_length=max_length,
        local_only=False,
        expected_adapter_sha256=expected_adapter_sha256,
    )
    (output_dir / 'preflight.json').write_text(json.dumps(pf, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'run_status.json').write_text(json.dumps({'status': 'running'}, indent=2), encoding='utf-8')

    resolved_model = Path(model_path) if model_path is not None else MODEL
    tokenizer = _load_tokenizer(resolved_model, local_only=Path(resolved_model).exists())

    train_records = _load_split(dataset_dir, 'train')
    val_records = _load_split(dataset_dir, 'val')
    challenge_records = _load_split(dataset_dir, 'challenge')

    try:
        set_seed(42)
        if eval_before:
            before = evaluate_adapter_on_splits(
                model_ref=resolved_model,
                tokenizer=tokenizer,
                adapter_path=initial_adapter,
                val_records=val_records,
                challenge_records=challenge_records,
                out_dir=output_dir / 'before',
            )
        else:
            before = None

        encoded_train = [encode_sample(tokenizer, r, max_length=max_length) for r in train_records]
        encoded_val = [encode_sample(tokenizer, r, max_length=max_length) for r in val_records]

        model, use_bf16 = _load_fresh_quantized_base(
            resolved_model, adapter_path=initial_adapter, is_trainable=True
        )

        steps_per_epoch = math.ceil(len(encoded_train) / (batch_size * grad_accum))
        total_steps = math.ceil(steps_per_epoch * epochs)
        warmup_steps = max(1, int(total_steps * 0.05))

        milestones = (0.5, 1.0, 1.5, 2.0)
        target_steps: dict[int, str] = {}
        for frac in milestones:
            if frac <= epochs + 1e-6:
                step = min(total_steps, max(1, math.ceil(steps_per_epoch * frac)))
                target_steps[step] = f'epoch_{frac:.1f}'

        ckpt_root = output_dir / 'checkpoint_adapters'
        ckpt_root.mkdir(parents=True, exist_ok=True)
        saved_tags: list[str] = []

        class HalfEpochAdapterSaver(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                self.speed_started = time.monotonic()

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if state.global_step == 5:
                    seconds_per_step = (time.monotonic()-self.speed_started)/5
                    speed = {'steps':5, 'seconds_per_step':seconds_per_step,
                        'estimated_training_hours':seconds_per_step*total_steps/3600,
                        'precision':'bf16' if use_bf16 else 'fp16'}
                    (output_dir/'speed_check.json').write_text(json.dumps(speed,indent=2),encoding='utf-8')
                    print('TRAINING_SPEED_CHECK '+json.dumps(speed),flush=True)
                    if seconds_per_step > 60:
                        model.save_pretrained(str(output_dir/'slow_run_partial_adapter'))
                        raise RuntimeError('Training exceeds 60 seconds per optimizer step; stopped to avoid wasting GPU quota.')
                tag = target_steps.get(state.global_step)
                if tag and tag not in saved_tags:
                    dest = ckpt_root / tag
                    model.save_pretrained(str(dest))
                    tokenizer.save_pretrained(str(dest))
                    saved_tags.append(tag)
                    print(f'Saved LoRA checkpoint {tag} at step {state.global_step}', flush=True)

        ta_params = {
            'output_dir': str(output_dir / 'trainer_work'),
            'num_train_epochs': epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': grad_accum,
            'learning_rate': learning_rate,
            'lr_scheduler_type': 'cosine',
            'warmup_steps': warmup_steps,
            'weight_decay': 0.01,
            'bf16': use_bf16,
            'fp16': not use_bf16,
            'gradient_checkpointing': True,
            'gradient_checkpointing_kwargs': {'use_reentrant': False},
            'logging_steps': 5,
            'eval_strategy': 'steps',
            'eval_steps': max(1, round(steps_per_epoch * 0.5)),
            'save_strategy': 'steps',
            'save_steps': 25,
            'save_total_limit': 2,
            'report_to': 'none',
            'seed': 42,
            'data_seed': 42,
            'optim': 'adamw_torch',
            'label_names': ['labels'],
        }
        train_args = TrainingArguments(**ta_params)
        trainer_kwargs = {
            'model': model,
            'args': train_args,
            'train_dataset': encoded_train,
            'eval_dataset': encoded_val,
            'data_collator': DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
            'callbacks': [HalfEpochAdapterSaver()],
        }
        sig_params = inspect.signature(Trainer.__init__).parameters
        if 'processing_class' in sig_params:
            trainer_kwargs['processing_class'] = tokenizer
        else:
            trainer_kwargs['tokenizer'] = tokenizer

        trainer = Trainer(**trainer_kwargs)
        started = time.monotonic()
        train_result = trainer.train()
        train_seconds = time.monotonic() - started
        logs = trainer.state.log_history

        final_tag = f'epoch_{epochs:.1f}'
        if final_tag not in saved_tags:
            dest = ckpt_root / final_tag
            model.save_pretrained(str(dest))
            tokenizer.save_pretrained(str(dest))
            saved_tags.append(final_tag)

        (output_dir / 'training_log.json').write_text(json.dumps(logs, indent=2), encoding='utf-8')
        (output_dir / 'train_metrics.json').write_text(
            json.dumps({**train_result.metrics, 'train_seconds': train_seconds}, indent=2),
            encoding='utf-8',
        )

        del trainer, model, train_result
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate every 0.5-epoch checkpoint cleanly on a fresh NF4 base.
        ckpt_summary: dict[str, dict] = {}
        best_tag = None
        best_score = -999.0
        for tag in saved_tags:
            res = evaluate_adapter_on_splits(
                model_ref=resolved_model,
                tokenizer=tokenizer,
                adapter_path=ckpt_root / tag,
                val_records=val_records,
                challenge_records=challenge_records,
                out_dir=output_dir / 'checkpoint_evals' / tag,
            )
            score = checkpoint_composite(res['val'], res['challenge'])
            ckpt_summary[tag] = {
                'composite_score': score,
                'val': res['val'],
                'challenge': res['challenge'],
            }
            if score > best_score:
                best_score = score
                best_tag = tag

        # Export selected best adapter & canonical result views.
        best_adapter_dir = output_dir / 'adapter'
        if best_adapter_dir.exists():
            shutil.rmtree(best_adapter_dir)
        shutil.copytree(ckpt_root / best_tag, best_adapter_dir)

        best_eval = output_dir / 'checkpoint_evals' / best_tag
        for sub, target_sub in (('val', 'after'), ('challenge', 'challenge_after')):
            dst = output_dir / target_sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(best_eval / sub, dst)

        summary_payload = {
            'selected_checkpoint': best_tag,
            'selected_composite_score': best_score,
            'precision': 'bf16' if use_bf16 else 'fp16',
            'max_length': max_length,
            'learning_rate': learning_rate,
            'train_seconds': train_seconds,
            'before': before,
            'checkpoints': ckpt_summary,
            'test_evaluated': False,
            'initial_adapter_sha256': expected_adapter_sha256,
            'optimizer_state_resumed': False,
            'training_epochs_this_run': epochs,
        }
        (output_dir / 'checkpoint_summary.json').write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (output_dir / 'metrics.json').write_text(
            json.dumps(ckpt_summary[best_tag]['val'], ensure_ascii=False, indent=2), encoding='utf-8'
        )

        # Clean up trainer scratch checkpoints directory so output stays compact.
        if (output_dir / 'trainer_work').exists():
            shutil.rmtree(output_dir / 'trainer_work')

        fig_dir = output_dir / 'figures'
        if before is not None and not fig_dir.exists():
            plot_results(
                logs,
                before['val'],
                ckpt_summary[best_tag]['val'],
                fig_dir,
                model_name='Qwen3-1.7B-A2-v4',
            )
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_checkpoint_progression(ckpt_summary, fig_dir)

        (output_dir / 'run_status.json').write_text(
            json.dumps({'status': 'completed', 'selected_checkpoint': best_tag, 'test_evaluated': False}, indent=2),
            encoding='utf-8',
        )
        return summary_payload
    except BaseException as exc:
        (output_dir / 'run_status.json').write_text(
            json.dumps({'status': 'failed', 'error': str(exc)[:1000]}, indent=2), encoding='utf-8'
        )
        raise


def _plot_checkpoint_progression(ckpt_summary: dict[str, dict], directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tags = list(ckpt_summary.keys())
    if not tags:
        return
    xs = list(range(len(tags)))
    val_f1 = [(ckpt_summary[t]['val'].get('status_macro_f1') or 0) * 100 for t in tags]
    chal_f1 = [(ckpt_summary[t]['challenge'].get('status_macro_f1') or 0) * 100 for t in tags]
    val_sel = [(ckpt_summary[t]['val'].get('selection_f1_on_answerable') or 0) * 100 for t in tags]
    chal_sel = [(ckpt_summary[t]['challenge'].get('selection_f1_on_answerable') or 0) * 100 for t in tags]
    comp = [(ckpt_summary[t].get('composite_score') or 0) * 100 for t in tags]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, val_f1, 'o-', label='Val Status Macro-F1')
    ax.plot(xs, chal_f1, 's-', label='Challenge Status Macro-F1')
    ax.plot(xs, val_sel, 'o--', label='Val Answerable Sel-F1')
    ax.plot(xs, chal_sel, 's--', label='Challenge Answerable Sel-F1')
    ax.plot(xs, comp, 'D-', linewidth=2, label='Composite Score')
    ax.set_xticks(xs, tags)
    ax.set(ylabel='Score (%)', ylim=(0, 105), title='Stage 2 (v4): 0.5-Epoch Checkpoint Progression')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'checkpoint_progression.png', dpi=250)
    plt.close(fig)


def package_results_zip(results_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(results_dir.rglob('*')):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(results_dir).as_posix())
    return zip_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 (v4) Qwen3-1.7B A2 trainer/evaluator')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--initial-adapter', type=Path, default=DEFAULT_INITIAL_ADAPTER)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--eval-before', action='store_true')
    parser.add_argument('--expected-adapter-sha256', default=INITIAL_ADAPTER_SHA256)
    parser.add_argument('--results-zip', type=Path, default=None)
    args = parser.parse_args()

    if args.preflight:
        print(json.dumps(preflight(
            dataset_dir=args.dataset_dir,
            initial_adapter=args.initial_adapter,
            model_path=args.model_path,
            max_length=args.max_length,
            expected_adapter_sha256=args.expected_adapter_sha256,
        ), ensure_ascii=False, indent=2))
    else:
        summary = train_and_evaluate(
            dataset_dir=args.dataset_dir,
            initial_adapter=args.initial_adapter,
            output_dir=args.output_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_length=args.max_length,
            eval_before=args.eval_before,
            expected_adapter_sha256=args.expected_adapter_sha256,
        )
        if args.results_zip is not None:
            package_results_zip(args.output_dir, args.results_zip)
        print(json.dumps({'selected': summary['selected_checkpoint'], 'score': summary['selected_composite_score']}, indent=2))
