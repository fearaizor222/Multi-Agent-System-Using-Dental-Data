"""Verify the completed pilot and prepare source-reading packets for the 32 paired cases."""
import hashlib
import json
import math
from pathlib import Path

from scripts.train_a2 import parse_sample,score_answer,summarize_results

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'outputs/a2_raft_training_20261112'


def main():
    collection=json.loads((WORK/'collection.json').read_text(encoding='utf-8'))
    folder=Path(collection['folder'])
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    assert read('run_status.json')['status']=='completed'
    assert read('experiment_status.json')['status']=='completed'
    info=read('dataset_info.json')
    for name,sha in info['source_sha256'].items():
        assert hashlib.sha256((ROOT/'data/processed/a2_raft_pilot_v1'/name).read_bytes()).hexdigest()==sha
    records=[json.loads(x) for x in (folder/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==72 and len({r['metadata']['id'] for r in records})==72
    metrics=read('metrics.json')
    for stage in ('before','after'):
        predictions=[json.loads(x) for x in (folder/stage/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(predictions)==72
        scored=[]
        for record,prediction in zip(records,predictions):
            assert record['metadata']['id']==prediction['sample_id']
            assert parse_sample(record)[1]==prediction['expected']
            scored.append(score_answer(record,prediction['raw_output']))
        assert summarize_results(scored)==metrics[stage]
    adapter_sha=hashlib.sha256((folder/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
    selected_sha=hashlib.sha256((folder/'selected_checkpoint/adapter_model.safetensors').read_bytes()).hexdigest()
    if adapter_sha!=selected_sha:
        import torch
        from safetensors.torch import load_file
        exported=load_file(str(folder/'adapter/adapter_model.safetensors'))
        selected=load_file(str(folder/'selected_checkpoint/adapter_model.safetensors'))
        assert set(exported)==set(selected) and all(torch.equal(exported[k],selected[k]) for k in exported)
    expected=json.loads((ROOT/'outputs/a2_raft_prompt_probe_20261112/cases.json').read_text(encoding='utf-8'))
    probe=[json.loads(x) for x in (folder/'paired_probe/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    by_key={(r['id'],r['variant']):r for r in probe}
    assert len(by_key)==len(probe)==64
    assert set(by_key)=={(r['id'],v) for r in expected for v in ('v3','pilot')}
    identity=read('paired_probe/identity.json')
    assert identity['variants']['pilot']==adapter_sha
    assert identity['variants']['v3']=='47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    aggregate={}
    for variant in ('v3','pilot'):
        rescored=[]
        for case in expected:
            prediction=by_key[case['id'],variant]
            score=score_answer(case['quote_prompt'],prediction['raw_output'])
            assert all(prediction[k]==v for k,v in score.items())
            rescored.append(score)
        aggregate[variant]=summarize_results(rescored)
    assert aggregate==read('paired_probe/metrics.json')
    review=[]
    previous=[json.loads(x) for x in (ROOT/'outputs/a2_raft_prompt_probe_20261112/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    previous_by_key={(r['id'],r['variant']):r for r in previous}
    previous_reviews=json.loads((ROOT/'outputs/a2_raft_prompt_probe_20261112/manual_review.json').read_text(encoding='utf-8'))
    inherited={}
    for index,case in enumerate(expected):
        base,new=by_key[case['id'],'v3'],by_key[case['id'],'pilot']
        assert base['prompt_ids_sha256']==new['prompt_ids_sha256']
        payload,gold=parse_sample(case['quote_prompt'])
        same_old=base['raw_output']==previous_by_key[case['id'],'quote_prompt']['raw_output']
        if same_old:
            inherited[str(index)]={'v3':previous_reviews[str(index)]['quote_prompt']}
        review.append({'index':index,'id':case['id'],'metadata':case['metadata'],
            'question':payload['question'],'sources':payload['evidence'],'gold':gold,
            'v3_matches_previous_output':same_old,
            'outputs':{variant:by_key[case['id'],variant]['raw_output'] for variant in ('v3','pilot')}})
    (WORK/'source_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review),encoding='utf-8')
    if not (WORK/'manual_review.json').exists():
        (WORK/'manual_review.json').write_text(json.dumps(inherited,ensure_ascii=False,indent=2),encoding='utf-8')
    losses=[{k:r[k] for k in ('step','epoch','eval_loss') if k in r} for r in read('training_log.json') if 'eval_loss' in r]
    assert all(math.isfinite(r['eval_loss']) for r in losses)
    best=min(losses,key=lambda r:r['eval_loss'])
    assert Path(metrics['best_checkpoint']).name=='checkpoint-'+str(best['step'])
    verification={'verified':True,'run':collection['run'],'counts':{'train':144,'validation':72,'paired_cases':32},
        'adapter_sha256':adapter_sha,'exported_weights_equal_selected_checkpoint':True,
        'best_checkpoint':metrics['best_checkpoint'],'selected_epoch':best['epoch'],'epoch_losses':losses,
        'train_seconds':metrics['train_seconds'],'same_v3_outputs_as_previous':sum(r['v3_matches_previous_output'] for r in review),
        'validation_metrics':metrics,'paired_metrics':aggregate,'test_evaluated':False}
    (WORK/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('validation_metrics','paired_metrics')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
