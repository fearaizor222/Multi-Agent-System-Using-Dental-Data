"""Modal-env fetch helper. Exit42 means the existing inference is not yet complete."""
from pathlib import Path
import json
import modal

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/evaluation/a2_paired_20261110'
OUT=ROOT/'outputs/a2_paired_20261110'


def main():
    volume=modal.Volume.from_name('dental-a2-training')
    expected=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['benchmark_sha256']
    matches=[]
    try:entries=volume.listdir('comparisons')
    except modal.exception.NotFoundError:raise SystemExit(42)
    for entry in entries:
        prefix=entry.path.rstrip('/')
        identity=json.loads(b''.join(volume.read_file(prefix+'/identity.json')))
        if identity['benchmark_manifest']['benchmark_sha256']!=expected:continue
        status=json.loads(b''.join(volume.read_file(prefix+'/run_status.json')))
        if status['status']=='failed':raise RuntimeError('Paired inference failed: '+status.get('error',''))
        if status['status']=='completed':matches.append(prefix)
    if not matches:raise SystemExit(42)
    if len(matches)!=1:raise ValueError('Multiple completed runs require explicit selection')
    remote=matches[0];folder=OUT/Path(remote).name
    for entry in volume.iterdir(remote,recursive=True):
        rel=entry.path.lstrip('/').removeprefix(remote.lstrip('/')+'/')
        if not Path(rel).suffix:continue
        target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b''.join(volume.read_file(entry.path)))
    print(json.dumps({'folder':str(folder)}))


if __name__=='__main__':main()
