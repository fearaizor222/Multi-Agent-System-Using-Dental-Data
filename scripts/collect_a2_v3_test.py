"""Fetch the completed frozen A2 v3 test and independently rescore all 93 outputs."""
import hashlib
import json
from pathlib import Path

import modal
from scripts.train_a2 import parse_sample, score_answer, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    volume = modal.Volume.from_name('dental-a2-training')
    runs = sorted(e.path.strip('/') for e in volume.iterdir('tests')
                  if Path(e.path).name.startswith('a2_v3_qwen_epoch02_'))
    remote = runs[-1]
    read_remote = lambda name: json.loads(b''.join(volume.read_file(remote + '/' + name)))
    status = read_remote('run_status.json')
    if status['status'] != 'completed':
        print(json.dumps({'remote': remote, **status}, ensure_ascii=False))
        return
    folder = ROOT / 'outputs/a2_test' / Path(remote).name
    for entry in volume.iterdir(remote, recursive=True):
        rel = entry.path.lstrip('/').removeprefix(remote + '/')
        if not Path(rel).suffix:
            continue
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(entry.path)))
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    identity = read('identity.json')
    assert identity['selected_epoch'] == 2 and identity['weights_equal_to_epoch2']
    assert identity['training_executed'] is False and identity['test_used_for_selection'] is False
    for name, sha in identity['dataset_sha256'].items():
        assert hashlib.sha256((ROOT / 'data/processed/a2_drafting_v3' / name).read_bytes()).hexdigest() == sha
    records = [json.loads(l) for l in (folder / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions = [json.loads(l) for l in (folder / 'test/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    source = [json.loads(l) for l in (ROOT / 'data/processed/a2_drafting_v3/test.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == len(predictions) == len(source) == 93
    assert len({p['sample_id'] for p in predictions}) == 93
    results = []
    for record, prediction, original in zip(records, predictions, source):
        assert {k: v for k, v in record.items() if k != 'metadata'} == original
        assert record['metadata']['split'] == 'test'
        assert prediction['sample_id'] == record['metadata']['id']
        assert prediction['expected'] == parse_sample(record)[1]
        results.append(score_answer(record, prediction['raw_output']))
    metrics = read('metrics.json')
    assert summarize_results(results) == metrics['test']
    verification = {'verified': True, 'samples': 93, 'identity': identity, **metrics,
                    'at_token_limit': sum(p['generation_tokens'] == p['generation_limit'] for p in predictions)}
    (folder / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'folder': str(folder), **verification}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
