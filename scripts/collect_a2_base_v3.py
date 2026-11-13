"""Collect inference, verify paired prompts and prepare blinded source-reading packets."""
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/evaluation/a2_base_vs_v3_20261111'
OUT=ROOT/'outputs/a2_base_vs_v3_20261111'


def main():
    volume=modal.Volume.from_name('dental-a2-training')
    runs=sorted(e.path.strip('/') for e in volume.iterdir('comparisons') if Path(e.path).name.startswith('a2_base_v3_'))
    if not runs:
        print('No inference output yet; job may be queued.')
        return
    remote=runs[-1];OUT.mkdir(exist_ok=True)
    for entry in volume.iterdir(remote):
        name=Path(entry.path).name
        if '.' in name:
            (OUT/name).write_bytes(b''.join(volume.read_file(entry.path)))
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    blob=(DATA/'cases.jsonl').read_bytes();assert hashlib.sha256(blob).hexdigest()==manifest['cases_sha256']
    cases=list(map(json.loads,blob.decode('utf-8').splitlines()))
    (OUT/'cases.jsonl').write_bytes(blob)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    status=json.loads((OUT/'run_status.json').read_text(encoding='utf-8'))
    if not (OUT/'predictions.jsonl').exists():
        print(status)
        return
    predictions=list(map(json.loads,(OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()))
    keyed={}
    for row in predictions:
        assert (row['id'],row['variant']) not in keyed
        keyed[row['id'],row['variant']]=row
    blind=[];key={}
    for i,case in enumerate(cases):
        if not all((case['id'],v) in keyed for v in ('base','v3')):
            continue
        a,b=keyed[case['id'],'base'],keyed[case['id'],'v3']
        assert a['prompt_ids_sha256']==b['prompt_ids_sha256'] and a['input_tokens']==b['input_tokens']
        labels=('base','v3') if int(hashlib.sha256((case['id']+'source-review-20261111').encode()).hexdigest(),16)%2 else ('v3','base')
        key[case['id']]={'A':labels[0],'B':labels[1]}
        blind.append({'index':i,'id':case['id'],'question':case['payload']['question'],
            'conversation':case['payload'].get('conversation'), 'sources':case['payload']['evidence'],
            'reference':case['expected'],'reference_note':case.get('grading_note'),
            'outputs':{label:keyed[case['id'],variant]['raw_output'] for label,variant in zip(('A','B'),labels)}})
    (OUT/'blind_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in blind),encoding='utf-8')
    (OUT/'blinding_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    (OUT/'collection.json').write_text(json.dumps({'remote':remote,'status':status,'paired_cases':len(blind),
        'generations':len(predictions),'pair_prompt_hashes_verified':True},indent=2),encoding='utf-8')
    if status['status']=='completed':
        assert len(predictions)==104 and len(blind)==52
    print(json.dumps({'remote':remote,'status':status,'paired_cases':len(blind),'generations':len(predictions)},ensure_ascii=False))


if __name__=='__main__':main()
