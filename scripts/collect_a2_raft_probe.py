"""Download the current inference-only prompt probe without submitting another job."""
import json
import modal
from scripts.modal_probe_a2_raft import OUT


def main():
    volume = modal.Volume.from_name('dental-a2-training')
    folders = sorted(e.path.strip('/') for e in volume.iterdir('comparisons')
        if e.path.strip('/').split('/')[-1].startswith('a2_raft_prompt_'))
    if not folders:
        print('Waiting for GPU startup; no remote output directory yet.')
        return
    remote = folders[-1]
    for entry in volume.iterdir(remote):
        name = entry.path.split('/')[-1]
        if name in ('run_status.json','identity.json','predictions.jsonl'):
            (OUT/name).write_bytes(b''.join(volume.read_file(entry.path)))
    status = json.loads((OUT/'run_status.json').read_text(encoding='utf-8'))
    predictions = []
    if (OUT/'predictions.jsonl').exists():
        predictions = [json.loads(x) for x in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    keys = {(p['id'],p['variant']) for p in predictions}
    assert len(keys)==len(predictions)
    if status['status']=='completed':
        cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
        assert keys=={(c['id'],v) for c in cases for v in ('old_prompt','quote_prompt')}
    collection={'remote':remote,'status':status,'generations':len(predictions),'complete_pairs':sum(
        (ident,'old_prompt') in keys and (ident,'quote_prompt') in keys for ident in {p['id'] for p in predictions})}
    (OUT/'collection.json').write_text(json.dumps(collection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(collection,ensure_ascii=False))


if __name__=='__main__':main()
