"""Package the reviewed sentence dataset and tokenizer-only CPU preflight."""
import json
import hashlib
import zipfile
import tempfile
from pathlib import Path
from scripts.build_a2_hard_v3 import ROOT,OUTPUT as DATA,rows,sha
from scripts.check_a2_hard_v3 import check
from scripts.train_a2 import MODEL,REVISION

OUTPUT=ROOT/'outputs/a2_colab_sentence_v3'
BASE=ROOT/'outputs/a2_colab_raft_v2/results/adapter'
BASE_SHA='17de0c132134552a4d3ee3b824397207a5bb9b0a059ce56f54d1df0efc786be5'


def blob(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def package_files():
    check(write_report=False)
    if sha(BASE/'adapter_model.safetensors')!=BASE_SHA:raise ValueError('Wrong initial RAFT v2 adapter')
    config=json.loads((BASE/'adapter_config.json').read_text(encoding='utf-8'))
    if config['base_model_name_or_path']!=MODEL or config['r']!=16:raise ValueError('Unexpected initial model/LoRA')
    files={}
    for name in ('adapter_config.json','adapter_model.safetensors'):files['initial_adapter/'+name]=(BASE/name).read_bytes()
    for source,name in [('train','train'),('val','validation'),('challenge','challenge')]:
        records=rows(DATA/(source+'.jsonl'));meta=rows(DATA/(source+'.metadata.jsonl'))
        files['data/'+name+'.jsonl']=''.join(json.dumps({**r,'metadata':{'id':m['id']}},ensure_ascii=False)+'\n' for r,m in zip(records,meta)).encode('utf-8')
    files['data/provenance.json']=blob({'dataset_manifest':json.loads((DATA/'manifest.json').read_text(encoding='utf-8')),
        'semantic_review':'assistant source-reading, not clinical expert','test_examples_included':False})
    files['plan.json']=blob(dict(model=MODEL,revision=REVISION,epochs=2,learning_rate=3e-5,seed=42,
        initial_adapter_sha256=BASE_SHA,initialization='continue saved RAFT v2 checkpoint-75; new optimizer and output format',
        max_length=4096,max_new_tokens=256,precision='fp16',quantization='NF4 double quantization',
        batch_size=1,gradient_accumulation_steps=8,save_eval_steps=25,
        replay_draws=0,train_count=1054,validation_count=84,challenge_count=56,
        checkpoint_selection='lowest validation loss; challenge never used for checkpoint selection',
        challenge_exposure='Historical corpus training exposure possible; not a blind test',test_used=False))
    for name in ('scripts/train_a2.py','scripts/a2_sentence_io.py','scripts/train_a2_sentence_colab.py',
                 'src/agents/contracts.py','src/training/a2_raft_protocol.py','src/training/a2_sentence_protocol.py','requirements-colab-a2.txt'):
        files[name]=(ROOT/name).read_bytes()
    for name in ('scripts/__init__.py','src/__init__.py','src/agents/__init__.py','src/training/__init__.py'):files[name]=b''
    files['bundle_manifest.json']=blob({'version':'a2-sentence-v3-colab',
        'files_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},'test_examples_included':False})
    return files


def notebook():
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)))
    md('''# A2 Qwen3 sentence selection v3 — Colab GPU
1054 train, 84 validation, 56 development challenge. Tiếp tục adapter RAFT v2, không phải model nền ngẫu nhiên.
Đổi Runtime → Change runtime type → T4 GPU. Tải a2_sentence_v3_bundle.zip. Không cần Modal.
Challenge không dùng chọn checkpoint và có thể từng xuất hiện trong kho train lịch sử.
Notebook chỉ được kiểm tra CPU/token khi chuẩn bị; chưa chạy GPU.''')
    code('''from google.colab import drive, files
from pathlib import Path, PurePosixPath
import json, hashlib, zipfile, shutil
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/DentalA2')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE = DRIVE_ROOT / 'a2_sentence_v3_bundle.zip'
if not BUNDLE.exists():
    local = Path('/content/a2_sentence_v3_bundle.zip')
    if not local.exists():
        uploaded = files.upload()
        if set(uploaded) != {'a2_sentence_v3_bundle.zip'}:
            raise ValueError('Chọn đúng a2_sentence_v3_bundle.zip')
        local.write_bytes(uploaded['a2_sentence_v3_bundle.zip'])
        del uploaded
    shutil.copyfile(local, BUNDLE)
WORK = Path('/content/dental_a2_sentence_v3')
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    manifest = json.loads(z.read('bundle_manifest.json'))
    assert set(z.namelist()) == set(manifest['files_sha256']) | {'bundle_manifest.json'}
    for name, digest in manifest['files_sha256'].items():
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    z.extractall(WORK)
print('Đã giải nén:', WORK)''')
    code('%pip install -q -r /content/dental_a2_sentence_v3/requirements-colab-a2.txt')
    code('''import os, sys, subprocess
RUN_NAME = 'a2_sentence_v3_colab_01'
RUN_DIR = DRIVE_ROOT / 'runs' / RUN_NAME
env = dict(os.environ, PYTHONPATH=str(WORK), HF_HOME='/content/hf_cache_dental',
           TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
def run_stage(stage):
    command = [sys.executable, '-u', '-X', 'utf8', '-m', 'scripts.train_a2_sentence_colab',
               '--stage', stage, '--output-dir', str(RUN_DIR)]
    with subprocess.Popen(command, cwd=WORK, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'Stage {stage} failed; xem log phía trên.')
subprocess.run(['nvidia-smi'], check=True)
print('Kết quả:', RUN_DIR)''')
    for stage,title in [('preflight','Kiểm tra token, không nạp trọng số'),('before','Validation trước train'),
                        ('challenge_before','Challenge trước train'),('train','Huấn luyện'),
                        ('after','Validation sau train và biểu đồ PNG'),('challenge_after','Challenge sau train')]:
        md('## '+title)
        if stage=='train':md('Giữ RUN_NAME khi resume. Kết nối lại GPU và chạy lại setup/preflight rồi cell train; chỉ nhận checkpoint hoàn chỉnh trên Drive. Không đổi tên để vô tình tạo lượt mới.')
        code(f"run_stage('{stage}')")
    md('## Đóng gói kết quả\nAdapter ở adapter/, biểu đồ ở figures/. Metrics khớp mã câu không thay thế chấm đúng nội dung.')
    code("run_stage('export')\nRESULT_ZIP = RUN_DIR.parent / (RUN_NAME + '_results.zip')\nfiles.download(str(RESULT_ZIP))")
    for i,c in enumerate(cells):c['id']=f'sentence-{i:02d}'
    return dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3')))


def main():
    from transformers import AutoTokenizer
    from scripts.train_a2 import configure_tokenizer
    from scripts.train_a2_sentence_colab import preflight
    OUTPUT.mkdir(parents=True,exist_ok=True);files=package_files()
    # Tokenizer files only; no AutoModel/PeftModel/GPU calls in this preparation.
    local=ROOT/'data/models/Qwen--Qwen3-1.7B'/REVISION
    tokenizer=configure_tokenizer(AutoTokenizer.from_pretrained(local,local_files_only=True,trust_remote_code=False),'qwen3_17b')
    with tempfile.TemporaryDirectory(prefix='a2_sentence_preflight_') as temp:
        root=Path(temp)
        for name,data in files.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        report=preflight(root,tokenizer)[0]
    (OUTPUT/'preflight_local.json').write_bytes(blob(report))
    archive=OUTPUT/'a2_sentence_v3_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Bad ZIP')
    (OUTPUT/'A2_Qwen3_Sentence_v3_Colab.ipynb').write_bytes(blob(notebook()))
    print(json.dumps(dict(preflight=report,zip=str(archive),size_mb=round(archive.stat().st_size/1024**2,1)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
