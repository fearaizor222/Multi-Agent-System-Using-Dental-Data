"""Convert audited RAFT v2 into a separate sentence-selection experiment; no GPU."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.train_a2 import parse_sample
from scripts.check_a2_raft_v2 import check
from src.training.a2_sentence_protocol import make_messages, selection_from_gold, resolve_selection

ROOT = Path(__file__).resolve().parents[1]


def build(source, output):
    if output.exists():
        raise ValueError('Refusing to overwrite existing experiment')
    check(source, write_reports=False)
    records, metadata, hashes = {}, {}, {}
    for split in ('train', 'val'):
        path = source / (split + '.jsonl')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        metas = [json.loads(x) for x in (source/(split+'.metadata.jsonl')).read_text(encoding='utf-8').splitlines()]
        records[split], metadata[split] = [], []
        for row, meta in zip(rows, metas):
            payload, gold = parse_sample(row)
            selected = selection_from_gold(gold, payload)
            resolved = resolve_selection(selected, payload)
            # Selection conversion must not rewrite the answer or status.
            assert resolved.status == gold['status']
            assert [c.text for c in resolved.claims] == [c['text'] for c in gold['claims']]
            messages = make_messages(payload) + [dict(role='assistant', content=json.dumps(selected.model_dump(), ensure_ascii=False))]
            records[split].append(dict(messages=messages))
            metadata[split].append({**meta, 'parent_messages_sha256': meta['messages_sha256'],
                'messages_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                'contract': 'a2_sentence_selection_v1'})
    # Same source families remain in their original splits, never moving holdout into train.
    assert not ({m['family_id'] for m in metadata['train']} & {m['family_id'] for m in metadata['val']})
    output.mkdir(parents=True)
    for split in records:
        for suffix, rows in [('.jsonl', records[split]), ('.metadata.jsonl', metadata[split])]:
            (output/(split+suffix)).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest = dict(contract='a2_sentence_selection_v1', counts={s:len(r) for s,r in records.items()},
        source=str(source), source_sha256=hashes, test_used_for_examples=False,
        scope='Converted existing audited facts; no new independent clinical facts. Validation has same questions/sources, new input/output format. Old metrics are not directly comparable.',
        training_executed=False, inference_executed=False)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/processed/a2_raft_pilot_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/a2_sentence_selection_v1')
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output),ensure_ascii=False,indent=2))
