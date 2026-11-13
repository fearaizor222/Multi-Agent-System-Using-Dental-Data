"""Fetch frozen A1 test outputs and independently recompute metrics."""
from pathlib import Path
import json
import hashlib
import modal
from src.training.a1_evaluation import score_response, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    volume = modal.Volume.from_name('dental-a3-training')
    remote = 'a1_tests/qwen3_v4_epoch02_20261109_200124_424060'
    folder = ROOT / 'outputs/a1_test' / Path(remote).name
    for entry in volume.iterdir(remote, recursive=True):
        rel = entry.path.lstrip('/').removeprefix(remote + '/')
        if not Path(rel).suffix:
            continue
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(entry.path)))
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    assert read('run_status.json')['status'] == 'completed'
    identity = read('identity.json')
    data = ROOT / 'data/processed/a1_supervisor_v4/test.metadata.jsonl'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == identity['test_sha256']
    records = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    assert records == read('test_records.json')
    preds = [json.loads(line) for line in (folder / 'test/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(preds) == len(records) == 59
    rescored = []
    for record, pred in zip(records, preds):
        assert pred['id'] == record['id'] and pred['expected'] == record['target'] and pred['input'] == record['input']
        rescored.append({**pred, **score_response(record['input'], record['target'], pred['raw_output'])})
    metrics = summarize(rescored)
    assert metrics == read('test/metrics.json')
    metrics['scope'] = read('metrics.json')['scope']
    assert metrics == read('metrics.json')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print('ERRORS', json.dumps([r for r in rescored if not r['control_exact']], ensure_ascii=False, indent=2))
    (folder / 'verification.json').write_text(json.dumps({'verified': True, 'count': 59, 'metrics': metrics}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
