"""Fetch pilot training artifacts and paired inference without submitting GPU work."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'outputs/a2_raft_training_20261112'


def main():
    volume=modal.Volume.from_name('dental-a2-training')
    runs=sorted(e.path.strip('/') for e in volume.iterdir('runs')
        if Path(e.path).name.startswith('a2_raft_qwen3_17b_'))
    if not runs:
        print('Waiting for GPU startup.')
        return
    remote=runs[-1]
    status=json.loads(b''.join(volume.read_file(remote+'/run_status.json')))
    folder=ROOT/'outputs/a2_training'/Path(remote).name
    folder.mkdir(parents=True,exist_ok=True)
    for entry in volume.iterdir(remote,recursive=True):
        rel=entry.path.lstrip('/').removeprefix(remote+'/')
        if not Path(rel).suffix or (rel.startswith('checkpoints/') and not rel.endswith('/trainer_state.json')):
            continue
        target=folder/rel
        # Immutable artifacts, especially weights, need downloading only once.
        if target.exists() and (rel.startswith(('adapter/','code_snapshot/','figures/'))
                or rel in ('config.json','dataset_info.json','train.jsonl','validation.jsonl')):
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(entry.path)))
    experiment=None
    if (folder/'experiment_status.json').exists():
        experiment=json.loads((folder/'experiment_status.json').read_text(encoding='utf-8'))
    if status['status']=='completed' and (folder/'metrics.json').exists():
        metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        checkpoint=Path(metrics['best_checkpoint']).name
        selected=folder/'selected_checkpoint';selected.mkdir(exist_ok=True)
        for name in ('adapter_model.safetensors','adapter_config.json'):
            target=selected/name
            if not target.exists():
                target.write_bytes(b''.join(volume.read_file(remote+'/checkpoints/'+checkpoint+'/'+name)))
        (selected/'origin.json').write_text(json.dumps({'remote_checkpoint':metrics['best_checkpoint']}),encoding='utf-8')
    counts={}
    for stage in ('before','after','paired_probe'):
        path=folder/stage/'predictions.jsonl'
        counts[stage]=len(path.read_text(encoding='utf-8').splitlines()) if path.exists() else 0
    progress={'run':Path(remote).name,'training':status,'experiment':experiment,
        'predictions':counts,'folder':str(folder)}
    states=list((folder/'checkpoints').glob('checkpoint-*/trainer_state.json'))
    if states:
        latest=max(states,key=lambda p:int(p.parent.name.split('-')[-1]))
        state=json.loads(latest.read_text(encoding='utf-8'))
        progress['saved_epoch']=state.get('epoch')
        progress['saved_global_step']=state.get('global_step')
    (WORK/'collection.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(progress,ensure_ascii=False))


if __name__=='__main__':main()
