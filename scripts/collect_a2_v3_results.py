"""Download A2 v3 evaluation artifacts and verify metrics without inference."""
from pathlib import Path
import hashlib
import json
import math
import modal
from scripts.train_a2 import parse_sample, score_answer, summarize_results

ROOT=Path(__file__).resolve().parents[1]
RUN='a2_qwen3_17b_20261110_194806_350795'


def main():
    volume=modal.Volume.from_name('dental-a2-training')
    folder=ROOT/'outputs/a2_training'/RUN
    prefix='runs/'+RUN+'/'
    for entry in volume.iterdir('runs/'+RUN,recursive=True):
        rel=entry.path.lstrip('/').removeprefix(prefix)
        if rel.startswith(('checkpoints/','adapter/')) or not Path(rel).suffix:
            continue
        target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(entry.path)))
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    assert read('run_status.json')['status']=='completed'
    info=read('dataset_info.json'); metrics=read('metrics.json')
    for name,sha in info['source_sha256'].items():
        assert hashlib.sha256((ROOT/'data/processed/a2_drafting_v3'/name).read_bytes()).hexdigest()==sha
    records=[json.loads(l) for l in (folder/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==107
    results={}
    for stage in ('before','after'):
        preds=[json.loads(l) for l in (folder/stage/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(preds)==107 and len({p['sample_id'] for p in preds})==107
        rescored=[]
        for record,pred in zip(records,preds):
            assert pred['sample_id']==record['metadata']['id'] and pred['expected']==parse_sample(record)[1]
            rescored.append(score_answer(record,pred['raw_output']))
        assert summarize_results(rescored)==metrics[stage]
        results[stage]={'metrics':metrics[stage],'at_token_limit':sum(p['generation_tokens']==p['generation_limit'] for p in preds)}
    losses=[{k:r[k] for k in ('epoch','step','eval_loss') if k in r} for r in read('training_log.json') if 'eval_loss' in r]
    assert all(math.isfinite(r['eval_loss']) for r in losses)
    report={'verified':True,'run':RUN,'results':results,'losses':losses,'train_seconds':metrics['train_seconds'],
            'best_checkpoint':metrics['best_checkpoint'],'test_evaluated':False}
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

