"""Build a private Kaggle input bundle; preserve and evaluate both full epochs."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/a2_kaggle_sentence_v3'


def blob(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def runner_source():
    source = (ROOT / 'scripts/train_a2_sentence_colab.py').read_text(encoding='utf-8')
    def replace(old, new):
        nonlocal source
        if old not in source:
            raise ValueError('Upstream runner changed: ' + old[:100])
        source = source.replace(old, new)
    replace("eval_strategy='steps',save_strategy='steps',eval_steps=25,save_steps=25,logging_steps=5,",
            "eval_strategy='epoch',save_strategy='epoch',logging_steps=5,")
    replace('save_total_limit=2,load_best_model_at_end=True',
            'save_total_limit=2,load_best_model_at_end=False')
    replace("checkpoint_manifest(Path(args.output_dir)/f'checkpoint-{state.global_step}',state.global_step)",
            """directory = Path(args.output_dir)/f'checkpoint-{state.global_step}'
            checkpoint_manifest(directory,state.global_step)
            epoch = int(round(state.epoch))
            if abs(state.epoch-epoch) > 0.01 or epoch not in (1,2):
                raise ValueError('Expected a complete epoch checkpoint')
            import shutil
            target = run/'epoch_adapters'/f'epoch_{epoch}'
            target.mkdir(parents=True,exist_ok=True)
            for name in ('adapter_config.json','adapter_model.safetensors'):
                shutil.copyfile(directory/name,target/name)
            tokenizer.save_pretrained(target)
            save(target/'epoch_identity.json',{'epoch':state.epoch,'global_step':state.global_step,
                'adapter_sha256':sha(target/'adapter_model.safetensors')})""")
    replace("'best_checkpoint':trainer.state.best_model_checkpoint,",
            "'best_loss_checkpoint':trainer.state.best_model_checkpoint,'exported_epoch':2,'export_policy':'final epoch, no best-loss reload',")
    replace("if stage.endswith('after') and not", "if stage.endswith('after') and not")
    replace("directory=run/stage;directory.mkdir(parents=True,exist_ok=True)",
            """if 'epoch1' in stage:
        adapter=run/'epoch_adapters/epoch_1'
    directory=run/stage;directory.mkdir(parents=True,exist_ok=True)""")
    replace("'challenge_after','export'],required=True)",
            "'challenge_after','epoch1_after','challenge_epoch1_after','export'],required=True)")
    replace("'challenge_before/metrics.json','challenge_after/metrics.json')):",
            "'challenge_before/metrics.json','challenge_after/metrics.json','epoch1_after/metrics.json','challenge_epoch1_after/metrics.json')):")
    replace("'training_complete.json','before/metrics.json','after/metrics.json','challenge_before/metrics.json','challenge_after/metrics.json','epoch1_after/metrics.json','challenge_epoch1_after/metrics.json'",
            "'training_complete.json','after/metrics.json','challenge_after/metrics.json','epoch1_after/metrics.json','challenge_epoch1_after/metrics.json'")
    replace("and (run/'challenge_before/metrics.json').exists():", ":")
    replace("if not torch.cuda.is_available():raise RuntimeError('Chọn Runtime > Change runtime type > GPU trong Colab.')",
            "if not torch.cuda.is_available():raise RuntimeError('Enable GPU in Kaggle notebook settings.')")
    # One visible GPU: the source loader explicitly places the quantized model on CUDA:0.
    replace('import os\n', "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\nos.environ['USE_TF'] = '0'\nos.environ['USE_FLAX'] = '0'\n")
    ast.parse(source)
    return source.encode('utf-8')


def notebook():
    cells = []
    def add(kind, text):
        cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# A2 Qwen3-1.7B — full 2 epochs on Kaggle
Attach the ZIP bundle as a **private** dataset. Enable Internet and GPU.
Same 1054 training samples, 84 validation, 56 development challenge; no reserved test.
Start again from the original RAFT v2 adapter, not checkpoint-75 of sentence v3.
Save both complete epochs. Evaluate both on validation and challenge; no loss-based reload.
Use one GPU even when T4 x2 is allocated. Results and resumable checkpoints: /kaggle/working.
''')
    add('code', '''from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, os, subprocess, sys
matches = list(Path('/kaggle/input').rglob('a2_sentence_v3_kaggle_bundle.bundle'))
if not matches:
    matches = list(Path('/kaggle/input').rglob('a2_sentence_v3_kaggle_bundle.zip'))
if len(matches) != 1:
    raise RuntimeError('Attach the private dataset containing a2_sentence_v3_kaggle_bundle.zip first.')
WORK = Path('/kaggle/working/a2_code')
WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(matches[0]) as z:
    manifest = json.loads(z.read('bundle_manifest.json'))
    assert set(z.namelist()) == set(manifest['files_sha256']) | {'bundle_manifest.json'}
    for name, digest in manifest['files_sha256'].items():
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    z.extractall(WORK)
print('Verified bundle:', matches[0])
''')
    add('code', "%pip install -q -r /kaggle/working/a2_code/requirements-colab-a2.txt")
    add('code', '''RUN_DIR = Path('/kaggle/working/a2_sentence_v3_kaggle_01')
env = dict(os.environ, PYTHONPATH=str(WORK), HF_HOME='/tmp/a2_hf_cache',
           CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
def run_stage(stage):
    command = [sys.executable, '-u', '-m', 'scripts.train_a2_sentence_kaggle',
               '--stage', stage, '--output-dir', str(RUN_DIR)]
    with subprocess.Popen(command, cwd=WORK, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout: print(line, end='', flush=True)
        if process.wait(): raise RuntimeError('Failed stage: '+stage)
subprocess.run(['nvidia-smi'], check=True)
''')
    add('code', '''# One run; do not launch a second GPU session in parallel.
for stage in ['preflight', 'train',
              'epoch1_after', 'challenge_epoch1_after', 'after', 'challenge_after', 'export']:
    print('\\nSTAGE:', stage, flush=True)
    run_stage(stage)
''')
    add('code', '''summary = {}
for name in ['epoch1_after','after','challenge_epoch1_after','challenge_after']:
    summary[name] = json.loads((RUN_DIR/name/'metrics.json').read_text())
(RUN_DIR/'checkpoint_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
figures = RUN_DIR/'figures'
figures.mkdir(exist_ok=True)
logs = json.loads((RUN_DIR/'training_log.json').read_text())
fig, ax = plt.subplots()
for key, label in [('loss','Train loss'),('eval_loss','Validation loss')]:
    points = [(r['step'],r[key]) for r in logs if key in r]
    if points: ax.plot(*zip(*points),marker='o',label=label)
ax.set(xlabel='Optimizer step',ylabel='Loss'); ax.legend()
fig.tight_layout(); fig.savefig(figures/'loss.png',dpi=180); plt.close(fig)
for stage in ['epoch1_after','after']:
    m = summary[stage]
    matrix = m['confusion_matrix']
    if isinstance(matrix,dict):
        raise ValueError('Unexpected confusion matrix format; retain metrics for inspection')
    fig, ax = plt.subplots()
    ax.imshow(matrix,cmap='Blues')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row): ax.text(j,i,str(value),ha='center',va='center')
    ax.set(xticks=range(3),xticklabels=['draft','insufficient','invalid'],yticks=range(2),
           yticklabels=['draft','insufficient'],xlabel='Predicted',ylabel='Expected',title=stage)
    fig.tight_layout(); fig.savefig(figures/(stage+'_04_confusion_matrix.png'),dpi=180); plt.close(fig)
    labels = ['Status macro-F1','Sentence micro-F1','Sentence F1 answerable']
    values = [m['status_macro_f1'],m['sentence_micro_f1'],m['selection_f1_on_answerable']]
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(labels,values); ax.bar_label(bars,fmt='%.3f')
    ax.set(ylim=(0,1.15),title=stage+' (reference matching, not semantics)')
    fig.tight_layout(); fig.savefig(figures/(stage+'_f1.png'),dpi=180); plt.close(fig)
    fig, ax = plt.subplots()
    decisions = ['draft','insufficient_evidence']
    bars = ax.bar(decisions,[m['per_status'][d]['f1'] for d in decisions])
    ax.bar_label(bars,fmt='%.3f'); ax.set(ylim=(0,1.15),ylabel='F1',title=stage)
    fig.tight_layout(); fig.savefig(figures/(stage+'_05_f1_by_decision.png'),dpi=180); plt.close(fig)
run_stage('export')
from IPython.display import FileLink, display
display(FileLink(str(RUN_DIR.parent/(RUN_DIR.name+'_results.zip'))))
print('Epoch 1 adapter:', RUN_DIR/'epoch_adapters/epoch_1')
print('Epoch 2 adapter:', RUN_DIR/'adapter')
print('PNG charts:', RUN_DIR/'figures')
''')
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        'kernelspec': {'name':'python3','display_name':'Python 3'},
        'kaggle': {'isInternetEnabled':True, 'isGpuEnabled':True, 'language':'python'}})


def main():
    original = ROOT/'outputs/a2_colab_sentence_v3/a2_sentence_v3_bundle.zip'
    with zipfile.ZipFile(original) as z:
        files = {n:z.read(n) for n in z.namelist() if n != 'bundle_manifest.json'}
    plan = json.loads(files['plan.json'])
    plan.update(checkpoint_selection='Export final epoch 2; retain and evaluate epoch 1 and 2 on validation. Challenge reporting only.',
                platform='kaggle', save_eval_strategy='epoch', initialization='restart original RAFT v2 adapter with fresh optimizer; train full 2 epochs')
    files['plan.json'] = blob(plan)
    files['scripts/train_a2_sentence_kaggle.py'] = runner_source()
    files['bundle_manifest.json'] = blob({'version':'a2-sentence-v3-kaggle-full-epochs',
        'files_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}, 'test_examples_included':False})
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT/'a2_sentence_v3_kaggle_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items(): z.writestr(name,content)
    (OUT/'A2_Qwen3_Sentence_v3_Kaggle.ipynb').write_bytes(blob(notebook()))
    print(OUT)


if __name__ == '__main__': main()
