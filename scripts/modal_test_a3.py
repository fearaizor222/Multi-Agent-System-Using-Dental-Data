"""Evaluate frozen v2 A3 adapters on held-out test; never train or select checkpoints."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a3 import image as training_image, volume, ROOT

RUNS = {
    'qwen3_17b': 'a3_qwen3_17b_20261109_122845_028798_50aa903b',
    'vinallama_27b': 'a3_vinallama_27b_20261109_123046_192478_2df1dfe0',
    'mediphi_38b': 'a3_mediphi_38b_20261109_122937_227148_1d0bf5fc',
}
app = modal.App('dental-a3-heldout-test')
image = training_image.add_local_file(ROOT / 'scripts/modal_test_a3.py', '/app/scripts/modal_test_a3.py')


def read_test(directory):
    from scripts.train_a3 import parse_sample
    files = {name: (directory / name).read_bytes() for name in ('test.jsonl', 'test.metadata.jsonl')}
    records, metadata = [[json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
                         for data in files.values()]
    if len(records) != 167 or len(metadata) != len(records):
        raise ValueError('Expected unchanged 167-example v2 test')
    if len({m['id'] for m in metadata}) != len(metadata):
        raise ValueError('Duplicate test IDs')
    for record, meta in zip(records, metadata):
        parse_sample(record)
        if meta['split'] != 'test':
            raise ValueError('Non-test metadata')
        record['metadata'] = meta
    return {'records': records, 'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}


@app.function(image=image, gpu='L4', cpu=4, memory=24576, timeout=7200, retries=0,
              volumes={'/vol': volume})
def run_test(model_id, packet):
    import os
    os.environ['HF_HOME'] = '/vol/hf/' + model_id
    import time
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from scripts.train_a3 import MODELS, configure_tokenizer, encode_sample, evaluate
    from datetime import datetime, timezone, timedelta
    source = Path('/vol/runs') / RUNS[model_id]
    read = lambda name: json.loads((source / name).read_text())
    config, info, status = read('config.json'), read('dataset_info.json'), read('run_status.json')
    spec = MODELS[model_id]
    if status['status'] != 'completed' or info.get('dataset_version') != 'a3_verification_v2':
        raise ValueError('Require completed v2 run')
    if (config['model'], config['revision']) != (spec['repo'], spec['revision']):
        raise ValueError('Base model mismatch')
    # Freeze identity of the adapter chosen using validation, never choose using test.
    weights = source / 'adapter/adapter_model.safetensors'
    adapter_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    output = Path('/vol/tests') / (RUNS[model_id] + '_' + datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    save('run_status.json', {'status': 'running'})
    save('identity.json', {'model_id': model_id, 'source_run': RUNS[model_id], 'test_sha256': packet['sha256'],
         'adapter_sha256': adapter_sha, 'training_executed': False, 'selection': 'existing lowest-validation-loss adapter'})
    started = time.monotonic()
    try:
        set_seed(42)
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision']), model_id)
        for record in packet['records']:
            encode_sample(tokenizer, record)  # Reject overlength rather than truncate.
        with (output / 'test.jsonl').open('w', encoding='utf-8') as stream:
            for record in packet['records']:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        volume.commit()
        model = AutoModelForCausalLM.from_pretrained(spec['repo'], revision=spec['revision'], trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, str(source / 'adapter'), is_trainable=False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Evaluation must have all parameters frozen')
        metrics = evaluate(model, tokenizer, packet['records'], output / 'test', model_id)
        save('metrics.json', {'test': metrics, 'elapsed_seconds': time.monotonic() - started,
            'scope': 'Held-out v2 reference-label agreement, not expert clinical validation. No training or checkpoint selection.'})
        plot_test(metrics, output / 'figures', spec['name'])
        save('run_status.json', {'status': 'completed'})
        print('Test completed:', output, flush=True)
        return str(output)
    except BaseException as exc:
        save('run_status.json', {'status': 'failed', 'error': str(exc)[:1000]})
        raise
    finally:
        volume.commit()


def plot_test(metrics, directory, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.train_a3 import FIELDS, LABELS
    directory.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, field in zip(axes, FIELDS):
        matrix = metrics['decisions'][field]['confusion_matrix']
        ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(4), LABELS + ['invalid'], rotation=25)
        ax.set_yticks(range(3), LABELS)
        ax.set(title=field, xlabel='Predicted', ylabel='Reference')
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha='center', va='center')
    fig.suptitle(f'{name}: held-out test (n=167)')
    fig.savefig(directory / '04_confusion_matrix.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, field in zip(axes, FIELDS):
        bars = ax.bar(LABELS, [metrics['decisions'][field]['per_label'][s]['f1'] * 100 for s in LABELS])
        ax.bar_label(bars, fmt='%.1f')
        ax.set(title=field, ylabel='F1 (%)', ylim=(0, 110))
    fig.suptitle(f'{name}: held-out test (n=167)')
    fig.savefig(directory / '05_f1_by_decision.png', dpi=300); plt.close(fig)


@app.local_entrypoint()
def main(model_id: str = 'qwen3_17b'):
    if model_id not in RUNS:
        raise ValueError('Unknown model')
    packet = read_test(ROOT / 'data/processed/a3_verification_v2')
    print(model_id, len(packet['records']), packet['sha256'], flush=True)
    print(run_test.remote(model_id, packet))
