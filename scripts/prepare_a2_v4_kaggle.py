"""Package Stage 2 (v4) dataset, code, and notebook into a standalone Kaggle bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from scripts.check_a2_sentence_v4 import check as check_v4
from scripts.train_a2 import REVISION
from scripts.train_a2_v4_kaggle import INITIAL_ADAPTER_SHA256

ROOT = Path(__file__).resolve().parents[1]
V4_DIR = ROOT / 'data/processed/a2_sentence_selection_v4'
INITIAL_ADAPTER = ROOT / 'outputs/a2_colab_raft_v2/results/adapter'
MODEL_DIR = ROOT / 'data/models/Qwen--Qwen3-1.7B' / REVISION
OUT_DIR = ROOT / 'outputs/a2_kaggle_sentence_v4'
BUNDLE_NAME = 'a2_sentence_v4_kaggle_bundle.zip'
NOTEBOOK_NAME = 'A2_Qwen3_Sentence_v4_Kaggle.ipynb'

BUNDLE_DATA_FILES = (
    'data/processed/a2_sentence_selection_v4/train.jsonl',
    'data/processed/a2_sentence_selection_v4/train.metadata.jsonl',
    'data/processed/a2_sentence_selection_v4/val.jsonl',
    'data/processed/a2_sentence_selection_v4/val.metadata.jsonl',
    'data/processed/a2_sentence_selection_v4/challenge.jsonl',
    'data/processed/a2_sentence_selection_v4/challenge.metadata.jsonl',
    'data/processed/a2_sentence_selection_v4/manifest.json',
    'data/processed/a2_sentence_selection_v4/source_review.json',
    'data/processed/a2_sentence_selection_v4/verification.json',
)

BUNDLE_CODE_FILES = (
    'src/__init__.py',
    'src/agents/__init__.py',
    'src/agents/contracts.py',
    'src/agents/sentence_drafting.py',
    'src/training/__init__.py',
    'src/training/a2_raft_protocol.py',
    'src/training/a2_sentence_protocol.py',
    'scripts/__init__.py',
    'scripts/train_a2.py',
    'scripts/a2_sentence_io.py',
    'scripts/a2_hard_specs.py',
    'scripts/a2_v4_cleaning.py',
    'scripts/a2_v4_specs.py',
    'scripts/check_a2_sentence_v4.py',
    'scripts/train_a2_v4_kaggle.py',
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_cell(cell_type: str, source: str) -> dict:
    lines = [line + '\n' for line in source.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    cell = {'cell_type': cell_type, 'metadata': {}, 'source': lines,
            'id': hashlib.sha256(source.encode('utf-8')).hexdigest()[:12]}
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell


def build_notebook() -> dict:
    md_intro = (
        '# A2 Qwen3-1.7B Stage 2 (v4) Training & 0.5-Epoch Checkpoint Evaluation (Kaggle)\n\n'
        '- Dataset: `a2_sentence_selection_v4` (`1,076` train, `84` val, `80` challenge; zero test data touched)\n'
        '- Starting weights: clean `RAFT v2` initial adapter (`outputs/a2_colab_raft_v2`)\n'
        '- Sequence budget: `max_length=4096` (0 samples dropped/truncated)\n'
        '- Checkpoints saved & evaluated on both Val + Challenge at `epoch_0.5`, `epoch_1.0`, `epoch_1.5`, `epoch_2.0`'
    )
    cell_setup = """import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

WORK = Path('/kaggle/working/a2_v4_work')
RESULTS = Path('/kaggle/working/results_v4')
ZIP_OUT = Path('/kaggle/working/a2_sentence_v4_kaggle_results.zip')
WORK.mkdir(parents=True, exist_ok=True)

# Locate uploaded Kaggle bundle (zip file OR auto-extracted directory under /kaggle/input)
bundle_zip = next(Path('/kaggle/input').rglob('a2_sentence_v4_kaggle_bundle.bundle'), None)
if bundle_zip is None:
    bundle_zip = next(Path('/kaggle/input').rglob('a2_sentence_v4_kaggle_bundle.zip'), None)
if bundle_zip is not None:
    print('Extracting bundle:', bundle_zip)
    with zipfile.ZipFile(bundle_zip, 'r') as zf:
        zf.extractall(WORK)
else:
    found_train = next(Path('/kaggle/input').rglob('data/processed/a2_sentence_selection_v4/train.jsonl'), None)
    if found_train is None:
        raise FileNotFoundError('Upload a2_sentence_v4_kaggle_bundle.zip as a Kaggle Dataset first.')
    src_root = found_train.parents[3]
    print('Copying extracted dataset from:', src_root)
    for sub in ('data', 'initial_adapter', 'src', 'scripts'):
        dst = WORK / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_root / sub, dst)

os.chdir(WORK)
sys.path.insert(0, str(WORK))
print('Workspace ready:', WORK)"""

    cell_deps = """import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(WORK/'requirements-colab-a2.txt')])
subprocess.check_call(['nvidia-smi'])"""

    cell_preflight = """import subprocess, sys
env = dict(os.environ, PYTHONPATH=str(WORK))
subprocess.check_call([
    sys.executable, '-m', 'scripts.train_a2_v4_kaggle',
    '--preflight',
    '--dataset-dir', str(WORK / 'data/processed/a2_sentence_selection_v4'),
    '--initial-adapter', str(WORK / 'initial_adapter'),
], env=env)"""

    cell_train = """import subprocess, sys
env = dict(
    os.environ,
    PYTHONPATH=str(WORK),
    CUDA_VISIBLE_DEVICES='0',
    USE_TF='0', USE_FLAX='0', PYTHONUNBUFFERED='1',
    TOKENIZERS_PARALLELISM='false',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
)
subprocess.check_call([
    sys.executable, '-m', 'scripts.train_a2_v4_kaggle',
    '--dataset-dir', str(WORK / 'data/processed/a2_sentence_selection_v4'),
    '--initial-adapter', str(WORK / 'initial_adapter'),
    '--output-dir', str(RESULTS),
    '--epochs', '2.0',
    '--lr', '8e-5',
    '--batch-size', '1',
    '--grad-accum', '8',
    '--max-length', '4096',
    '--results-zip', str(ZIP_OUT),
], env=env)
print('Saved results zip:', ZIP_OUT, 'exists=', ZIP_OUT.exists())"""

    cell_inspect = """summary = json.loads((RESULTS / 'checkpoint_summary.json').read_text(encoding='utf-8'))
print('Selected checkpoint:', summary['selected_checkpoint'], 'score:', round(summary['selected_composite_score'], 4))
for tag, row in summary['checkpoints'].items():
    v, c = row['val'], row['challenge']
    print(
        f"{tag:10s} comp={row['composite_score']:.4f} | "
        f"val(status={v['status_correct']:.3f}, f1={v['selection_f1_on_answerable']:.3f}, exact={v['selection_set_exact']:.3f}) | "
        f"chal(status={c['status_correct']:.3f}, f1={c['selection_f1_on_answerable']:.3f}, exact={c['selection_set_exact']:.3f})"
    )
sel = summary['selected_checkpoint']
print('\\nChallenge slice breakdown for', sel)
print(json.dumps(summary['checkpoints'][sel]['challenge']['slice_breakdown'], ensure_ascii=False, indent=2))"""

    return {
        'cells': [
            _notebook_cell('markdown', md_intro),
            _notebook_cell('code', cell_setup),
            _notebook_cell('code', cell_deps),
            _notebook_cell('code', cell_preflight),
            _notebook_cell('code', cell_train),
            _notebook_cell('code', cell_inspect),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def prepare(output_dir: Path = OUT_DIR) -> dict:
    v4_summary = check_v4(write_report=True)

    weights = INITIAL_ADAPTER / 'adapter_model.safetensors'
    if not weights.exists():
        raise FileNotFoundError(f'Missing initial adapter: {weights}')
    if _sha256(weights) != INITIAL_ADAPTER_SHA256:
        raise ValueError('Initial adapter SHA mismatch')

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / BUNDLE_NAME
    notebook_path = output_dir / NOTEBOOK_NAME

    with zipfile.ZipFile(bundle_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in BUNDLE_DATA_FILES + BUNDLE_CODE_FILES:
            src = ROOT / rel
            if not src.exists():
                if src.name == '__init__.py':
                    zf.writestr(rel, '')
                    continue
                raise FileNotFoundError(f'Missing required bundle file: {src}')
            zf.write(src, rel)

        for adapter_file in sorted(INITIAL_ADAPTER.iterdir()):
            if adapter_file.is_file():
                zf.write(adapter_file, f'initial_adapter/{adapter_file.name}')
        zf.write(ROOT/'requirements-colab-a2.txt', 'requirements-colab-a2.txt')

    notebook_path.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding='utf-8'
    )

    # Verify extracted standalone bundle in an isolated temporary directory via subprocess CPU preflight.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(bundle_path, 'r') as zf:
            zf.extractall(tmp)
        model_arg = ['--model-path', str(MODEL_DIR)] if MODEL_DIR.exists() else []
        cmd = [
            sys.executable,
            '-m',
            'scripts.train_a2_v4_kaggle',
            '--preflight',
            '--dataset-dir',
            str(tmp / 'data/processed/a2_sentence_selection_v4'),
            '--initial-adapter',
            str(tmp / 'initial_adapter'),
            *model_arg,
        ]
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, check=True)
        preflight_data = json.loads(res.stdout)

    (output_dir / 'preflight.json').write_text(
        json.dumps(preflight_data, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    manifest = {
        'bundle_path': str(bundle_path),
        'bundle_sha256': _sha256(bundle_path),
        'bundle_size_mb': round(bundle_path.stat().st_size / (1024 * 1024), 2),
        'notebook_path': str(notebook_path),
        'initial_adapter_sha256': INITIAL_ADAPTER_SHA256,
        'v4_dataset_counts': {k: v['count'] for k, v in v4_summary.items()},
        'preflight': preflight_data,
        'test_examples_used': False,
    }
    (output_dir / 'bundle_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
