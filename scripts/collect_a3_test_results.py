"""Download and independently rescore completed held-out A3 tests; no inference."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.train_a3 import score_answer, summarize_results

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/a3_test'


def main():
    volume = modal.Volume.from_name('dental-a3-training')
    report = {}
    reference = None
    for entry in volume.listdir('tests'):
        name = Path(entry.path).name
        folder = DEST / name
        for file in volume.iterdir(entry.path, recursive=True):
            rel = file.path.lstrip('/').removeprefix(entry.path.lstrip('/') + '/')
            if not Path(rel).suffix:
                continue
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = b''.join(volume.read_file(file.path))
            target.write_bytes(data)
        read = lambda path: json.loads((folder / path).read_text(encoding='utf-8'))
        assert read('run_status.json')['status'] == 'completed'
        identity = read('identity.json')
        for filename, sha in identity['test_sha256'].items():
            assert hashlib.sha256((ROOT / 'data/processed/a3_verification_v2' / filename).read_bytes()).hexdigest() == sha
        records = [json.loads(l) for l in (folder / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
        preds = [json.loads(l) for l in (folder / 'test/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records) == len(preds) == 167
        if reference is None:
            reference = records
        assert records == reference
        assert len({p['sample_id'] for p in preds}) == 167
        for row, pred in zip(records, preds):
            assert row['metadata']['id'] == pred['sample_id']
        scored = [score_answer(row, pred['raw_output']) for row, pred in zip(records, preds)]
        metrics = summarize_results(scored)
        assert metrics == read('metrics.json')['test']
        model = identity['model_id']
        val = json.loads((ROOT / 'outputs/a3_training' / identity['source_run'] / 'metrics.json').read_text())['after']
        report[model] = {'folder': name, 'identity': identity, 'test': metrics,
            'validation_assessment': val['assessment_exact_match'],
            'generation_limit_count': sum(p['generation_tokens'] == 512 for p in preds),
            'elapsed_seconds': read('metrics.json')['elapsed_seconds']}
        errors = [{**p, 'messages': row['messages']} for row, p in zip(records, preds) if not p['assessment_exact_match']]
        (folder / 'errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
        print(model, json.dumps(report[model], ensure_ascii=False), flush=True)
    assert len(report) == 3
    (DEST / 'comparison_verified.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
