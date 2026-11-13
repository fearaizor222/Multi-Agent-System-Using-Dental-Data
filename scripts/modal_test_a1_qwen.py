"""Frozen epoch-2 A1 Qwen adapter, held-out test only; no training."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a3 import image as base_image, volume, ROOT

SOURCE = ROOT / 'outputs/a1_training/qwen3_17b_v4_20261106_213145_386069'
ADAPTER = SOURCE / 'epoch_02/adapter'
app = modal.App('dental-a1-qwen-heldout-test')
image = base_image.add_local_dir(ADAPTER, '/app/a1_adapter')
for name in ('scripts/modal_test_a1_qwen.py', 'src/data/a1_dataset.py', 'src/training/a1_evaluation.py'):
    image = image.add_local_file(ROOT / name, '/app/' + name)


def packet():
    from src.data.a1_dataset import validate_pair
    directory = ROOT / 'data/processed/a1_supervisor_v4'
    manifest = json.loads((directory / 'manifest.json').read_text())
    raw = (directory / 'test.metadata.jsonl').read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    assert len(records) == 59 and len({r['id'] for r in records}) == 59
    assert hashlib.sha256(raw).hexdigest() == manifest['files_sha256']['test.metadata.jsonl']
    config = json.loads((SOURCE / 'config.json').read_text())
    summary = json.loads((SOURCE / 'summary.json').read_text())
    assert summary['best_finetuned_epoch'] == 2 and summary['adapter_reload_probe_matches']
    for record in records:
        assert record['split'] == 'test'
        validate_pair(record['input'], record['target'])
    groups = {r['group_id'] for r in records}
    for split in ('train', 'validation'):
        seen = [json.loads(line) for line in (directory / (split + '.metadata.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert not groups.intersection(r['group_id'] for r in seen)
    return {'records': records, 'config': config, 'test_sha256': hashlib.sha256(raw).hexdigest(),
            'adapter_sha256': hashlib.sha256((ADAPTER / 'adapter_model.safetensors').read_bytes()).hexdigest()}


@app.function(image=image, gpu='L4', cpu=4, memory=24576, timeout=7200, retries=0, volumes={'/vol': volume})
def run_test(data):
    import os
    os.environ['HF_HOME'] = '/vol/hf/qwen3_17b'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from src.training.a1_evaluation import evaluate
    from datetime import datetime, timezone, timedelta
    config = data['config']
    assert hashlib.sha256(Path('/app/a1_adapter/adapter_model.safetensors').read_bytes()).hexdigest() == data['adapter_sha256']
    directory = Path('/vol/a1_tests') / ('qwen3_v4_epoch02_' + datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('identity.json', {k: v for k, v in data.items() if k != 'records'})
    save('run_status.json', {'status': 'running', 'training_executed': False})
    save('test_records.json', data['records'])
    volume.commit()
    try:
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'], revision=config['revision'], trust_remote_code=False)
        for record in data['records']:
            ids = tokenizer.apply_chat_template(record['messages'], tokenize=True, enable_thinking=False)
            assert len(ids) <= config['max_length'], 'Test sample exceeds original max_length'
        model = AutoModelForCausalLM.from_pretrained(config['base_model'], revision=config['revision'], trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # Match original A1 load_base: frozen embeddings/output head use BF16.
        model.get_input_embeddings().to(dtype=torch.bfloat16)
        model.get_output_embeddings().to(dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, '/app/a1_adapter', is_trainable=False)
        assert not any(p.requires_grad for p in model.parameters())
        metrics, rows = evaluate(model, tokenizer, data['records'], directory / 'test', config['generation'], torch.bfloat16,
            progress=lambda n, total, _: print(f'A1 test: {n}/{total}', flush=True) if n == 1 or n % 10 == 0 or n == total else None)
        metrics['scope'] = 'Held-out synthetic A1 test control decisions; response wording and resolved_request semantics not automatically judged.'
        save('metrics.json', metrics)
        save('errors.json', [r for r in rows if not r['control_exact']])
        plot(metrics, directory / 'figures')
        save('run_status.json', {'status': 'completed', 'training_executed': False})
        print('Completed:', directory, flush=True)
        return str(directory)
    except BaseException as exc:
        save('run_status.json', {'status': 'failed', 'error': str(exc)[:1000]})
        raise
    finally:
        volume.commit()


def plot(metrics, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.training.a1_evaluation import DECISIONS
    directory.mkdir()
    columns = DECISIONS + ['invalid']
    matrix = [[0] * len(columns) for _ in DECISIONS]
    for row in metrics['confusion']:
        matrix[DECISIONS.index(row['expected'])][columns.index(row['predicted'])] = row['count']
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(6), columns, rotation=25); ax.set_yticks(range(5), DECISIONS)
    ax.set(title='A1 Qwen3: held-out test (n=59)', xlabel='Predicted', ylabel='Reference')
    for i, row in enumerate(matrix):
        for j, val in enumerate(row): ax.text(j, i, str(val), ha='center', va='center')
    fig.savefig(directory / '04_confusion_matrix.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    bars = ax.bar(DECISIONS, [metrics['per_decision'][s]['f1'] * 100 for s in DECISIONS])
    ax.bar_label(bars, fmt='%.1f'); ax.set(title='A1 Qwen3: held-out test (n=59)', ylabel='F1 (%)', ylim=(0, 110))
    fig.savefig(directory / '05_f1_by_decision.png', dpi=300); plt.close(fig)


@app.local_entrypoint()
def main():
    data = packet()
    print('A1 frozen epoch 2; test count:', len(data['records']), flush=True)
    print(run_test.remote(data))
