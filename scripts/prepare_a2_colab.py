"""Create an uploadable notebook and minimal ZIP; does not load or train an LLM."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.train_a2 import ROOT,MODEL,REVISION,load_data,parse_sample

OUTPUT=ROOT/'outputs/a2_colab_raft_v2'
BASE=ROOT/'outputs/a2_training/a2_qwen3_17b_20261110_194806_350795/adapter'
EXPECTED_ADAPTER='47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'


def blob(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def lines(records):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records).encode('utf-8')
def read_lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def package_files():
    packet=load_data(ROOT/'data/processed/a2_raft_pilot_v2')
    previous=load_data(ROOT/'data/processed/a2_drafting_v4')
    audit=read_lines(ROOT/'outputs/a2_data_audit_20261112/manual_review.jsonl')
    changes=read_lines(ROOT/'data/processed/a2_drafting_v4/changes.jsonl')
    kept={r['id'] for r in audit if r['decision']=='keep_supported_in_sample'}
    repaired={r['id'] for r in changes if r['action']=='repair'}
    ids=kept|repaired
    replay=[r for r in previous['train'] if r['metadata']['id'] in ids]
    assert len(replay)==len(ids)==31
    forbidden={m['document_id'] for m in read_lines(ROOT/'data/processed/a2_drafting_v4/test.metadata.jsonl')}
    forbidden|={o['document_id'] for r in packet['validation'] for o in r['metadata']['source_origins']}
    for record in replay:
        meta=record['metadata'];payload,gold=parse_sample(record)
        assert meta['document_id'] not in forbidden
        evidence={e['evidence_id']:' '.join(e['content'].split()) for e in payload['evidence']}
        assert len(meta['proofs'])==len(gold['claims'])
        for claim,proof in zip(gold['claims'],meta['proofs']):
            assert claim['text']==proof['text'] and set(claim['evidence_ids'])==set(proof['evidence_ids'])
            assert proof['proofs'] and all(' '.join(q['quote'].split()) in evidence[q['evidence_id']] for q in proof['proofs'])
    files={}
    for name in ('adapter_config.json','adapter_model.safetensors'):
        files['initial_adapter/'+name]=(BASE/name).read_bytes()
    assert hashlib.sha256(files['initial_adapter/adapter_model.safetensors']).hexdigest()==EXPECTED_ADAPTER
    config=json.loads(files['initial_adapter/adapter_config.json'])
    assert config['base_model_name_or_path']==MODEL and config['r']==16 and config['lora_alpha']==32
    files['data/raft_train.jsonl']=lines(packet['train'])
    files['data/replay.jsonl']=lines(replay)
    files['data/validation.jsonl']=lines(packet['validation'])
    files['data/provenance.json']=blob({'raft_info':packet['info'],
        'replay_source':'a2_drafting_v4; 14 source-read keeps + 17 source-read repairs',
        'replay_ids':sorted(ids),'replay_keep_ids':sorted(kept),'replay_repair_ids':sorted(repaired),
        'replay_source_file_sha256':hashlib.sha256((ROOT/'data/processed/a2_drafting_v4/train.jsonl').read_bytes()).hexdigest(),
        'replay_statuses':dict(Counter(parse_sample(r)[1]['status'] for r in replay)),
        'test_examples_included':False,'clinical_expert_review':False})
    files['plan.json']=blob({'model':MODEL,'revision':REVISION,'initialization':'continue LoRA A2 v3 epoch 2; new optimizer',
        'initial_adapter_sha256':EXPECTED_ADAPTER,'epochs':2,'learning_rate':3e-5,'seed':42,
        'replay_draws':220,'raft_samples':880,'unique_replay_samples':31,'samples_per_epoch':1100,
        'precision':'fp16','quantization':'NF4 double quantization','batch_size':1,'gradient_accumulation_steps':8,
        'max_length':3072,'max_new_tokens':512,'save_eval_steps':25,
        'checkpoint_selection':'lowest validation loss','validation':'same 72 development variants; not an independent test',
        'test_used':False})
    for name in ('scripts/train_a2.py','scripts/a2_raft_io.py','scripts/train_a2_colab.py',
                 'src/agents/contracts.py','src/training/a2_raft_protocol.py','requirements-colab-a2.txt'):
        files[name]=(ROOT/name).read_bytes()
    for name in ('scripts/__init__.py','src/__init__.py','src/agents/__init__.py','src/training/__init__.py'):
        files[name]=b''
    files['bundle_manifest.json']=blob({'version':'a2-colab-raft-v2-20261112',
        'files_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        'training_executed':False,'inference_executed':False,'test_examples_included':False})
    return files


def notebook():
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})
    md('''# Fine-tune A2 trên Google Colab free

**Runtime → Change runtime type → GPU**. Nếu Colab cấp T4, notebook dùng NF4 + FP16. Chạy các cell theo thứ tự.

Chuẩn bị file `a2_colab_bundle.zip` đã được tạo trên máy. ZIP chứa dữ liệu, code và adapter A2 v3 khoảng 70 MB; model nền Qwen3-1.7B sẽ tải riêng từ Hugging Face.

Lượt này **tiếp tục từ A2 v3**, không tạo adapter ngẫu nhiên: 880 mẫu RAFT + 220 lượt ôn tập lấy từ 31 mẫu cũ đã kiểm tra, learning rate 3e-5, 2 epoch, batch hiệu dụng 8. Có 72 mẫu validation; không dùng 93 test.

Colab free không bảo đảm GPU/thời lượng phiên. Không dùng thủ thuật giữ phiên. Checkpoint lưu trên Drive mỗi 25 optimizer step; mất phiên có thể mất phần việc kể từ checkpoint hoàn chỉnh gần nhất.''')
    code('''from google.colab import drive, files
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile

drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/DentalA2')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE = DRIVE_ROOT / 'a2_colab_bundle.zip'
if not BUNDLE.exists():
    print('Chọn a2_colab_bundle.zip từ máy tính (chỉ cần tải một lần).')
    uploaded = files.upload()
    if len(uploaded) != 1 or next(iter(uploaded)) != 'a2_colab_bundle.zip':
        raise ValueError('Cần chọn đúng một file a2_colab_bundle.zip')
    BUNDLE.write_bytes(uploaded['a2_colab_bundle.zip'])
    del uploaded

WORK = Path('/content/dental_a2')
WORK.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError('ZIP có tên trùng')
    manifest = json.loads(z.read('bundle_manifest.json'))
    if set(names) != set(manifest['files_sha256']) | {'bundle_manifest.json'}:
        raise ValueError('ZIP không khớp manifest')
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\\\' in name:
            raise ValueError('Đường dẫn ZIP không hợp lệ')
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256(z.read(name)).hexdigest() != digest:
            raise ValueError('Sai checksum: ' + name)
    z.extractall(WORK)
print('Đã kiểm tra và giải nén:', WORK)''')
    code('''%pip install -q -r /content/dental_a2/requirements-colab-a2.txt''')
    md('''## Cấu hình phiên

Giữ nguyên `RUN_NAME` để chạy tiếp sau khi Colab ngắt. Nếu đổi dữ liệu/cấu hình cho thí nghiệm khác, dùng tên mới. Lệnh chạy bằng tiến trình Python riêng để nhận đúng thư viện vừa cài; không cần cài Modal.

Thư mục kết quả: `MyDrive/DentalA2/runs/<RUN_NAME>/`. Checkpoint có cả optimizer, scheduler và RNG để resume; adapter cuối cùng không đủ để khôi phục đầy đủ tiến trình train.''')
    code('''import os, subprocess, sys
RUN_NAME = 'a2_v3_raft_v2_colab_01'
RUN_DIR = DRIVE_ROOT / 'runs' / RUN_NAME
env = dict(os.environ, PYTHONPATH=str(WORK), HF_HOME='/content/hf_cache_dental',
           TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
def run_stage(stage):
    subprocess.run([sys.executable, '-u', '-X', 'utf8', '-m', 'scripts.train_a2_colab',
                    '--stage', stage, '--output-dir', str(RUN_DIR)], cwd=WORK, env=env, check=True)
subprocess.run(['nvidia-smi'], check=True)
print('Kết quả lưu tại:', RUN_DIR)''')
    md('## 1. Kiểm tra dữ liệu và token — chưa tải trọng số model')
    code("run_stage('preflight')")
    md('''## 2. Đánh giá A2 v3 trước khi train

72 câu, prompt yêu cầu quote như lượt sau train, cùng NF4/FP16 và greedy decoding. Không so trực tiếp các tỷ lệ này với lượt Modal BF16 cũ. Chạy lại cell sẽ tiếp tục các câu chưa có kết quả.''')
    code("run_stage('before')")
    md('''## 3. Huấn luyện tiếp từ A2 v3

Nếu bị ngắt: kết nối lại GPU, chạy lại các cell thiết lập phía trên, giữ `RUN_NAME`, rồi chạy lại cell này. Chương trình chỉ nhận checkpoint có dấu hoàn tất và checksum đúng. Khôi phục từ checkpoint gần nhất, không phải từ từng batch cuối cùng.

Checkpoint nằm trực tiếp trên Drive, giữ tối đa 2 checkpoint theo cơ chế Trainer. Nếu đổi GPU/thư viện sau khi ngắt, tính toán có thể không giống tuyệt đối dù trạng thái optimizer và bộ sinh số được khôi phục.''')
    code("run_stage('train')")
    md('''## 4. Đánh giá sau train và tạo 4 hình PNG

Nạp mới nền Qwen3 + adapter được chọn theo validation loss, cùng cấu hình FP16 với lượt trước train. Các chỉ số tự động đo schema, status, mã dẫn và quote; vẫn cần đọc nguồn để chấm đúng nội dung.

Các hình: `loss.png`, `before_after.png`, `04_confusion_matrix.png`, `05_f1_by_decision.png`. Before là **A2 v3**, After là **adapter mới**.''')
    code("run_stage('after')")
    md('## 5. Đóng gói và tải kết quả — checkpoint đầy đủ vẫn ở Drive')
    code('''run_stage('export')
RESULT_ZIP = RUN_DIR.parent / (RUN_NAME + '_results.zip')
print(RESULT_ZIP)
files.download(str(RESULT_ZIP))''')
    md('''Nguồn tham khảo: [Giới hạn Colab](https://research.google.com/colaboratory/faq.html), [PEFT và lượng tử hóa](https://huggingface.co/docs/peft/developer_guides/quantization), [Trainer resume](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train). Notebook chưa được chạy trên một phiên Colab thật khi được chuẩn bị.''')
    for index,cell in enumerate(cells):cell['id']=f'cell-{index:02d}'
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'A2_Qwen3_RAFT_v2_Colab.ipynb'},
        'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'}},'cells':cells}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    files=package_files()
    archive=OUTPUT/'a2_colab_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    (OUTPUT/'A2_Qwen3_RAFT_v2_Colab.ipynb').write_bytes(blob(notebook()))
    print(json.dumps({'zip':str(archive),'size_mb':round(archive.stat().st_size/1024**2,1),
        'notebook':str(OUTPUT/'A2_Qwen3_RAFT_v2_Colab.ipynb'),'training_executed':False},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
