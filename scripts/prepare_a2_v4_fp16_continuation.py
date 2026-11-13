"""Continue recovered v4 half-epoch weights with native GPU precision, fresh optimizer."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from scripts.prepare_a2_v4_kaggle import BUNDLE_CODE_FILES,BUNDLE_DATA_FILES,build_notebook,MODEL_DIR
from scripts.train_a2_v4_kaggle import preflight

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/a2_kaggle_sentence_v4_fp16'
PARENT=ROOT/'outputs/a2_kaggle_sentence_v4/cancelled_v1/results_v4/checkpoint_adapters/epoch_0.5'


def main():
    OUT.mkdir(exist_ok=True)
    digest=hashlib.sha256((PARENT/'adapter_model.safetensors').read_bytes()).hexdigest()
    report=preflight(initial_adapter=PARENT,model_path=MODEL_DIR,expected_adapter_sha256=digest)
    plan={'dataset':'a2_sentence_selection_v4','parent_run':'vunguyen25552/dental-a2-qwen3-v4/1',
          'parent_checkpoint':'epoch_0.5','parent_adapter_sha256':digest,
          'additional_epochs':1.5,'optimizer_state_resumed':False,
          'note':'Adapter warm start with fresh optimizer and shuffle. Not an exact continuation of the interrupted sample order or LR schedule. Checkpoint tags refer to this additional training stage.',
          'precision_policy':'Native BF16 only on supported hardware; T4 uses FP16',
          'speed_guard':'Stop if first five optimizer steps average more than 60 seconds',
          'test_used':False}
    files={n:(ROOT/n).read_bytes() if (ROOT/n).exists() else b'' for n in BUNDLE_CODE_FILES+BUNDLE_DATA_FILES}
    for p in PARENT.iterdir():
        if p.is_file():files['initial_adapter/'+p.name]=p.read_bytes()
    files['requirements-colab-a2.txt']=(ROOT/'requirements-colab-a2.txt').read_bytes()
    files['continuation_plan.json']=json.dumps(plan,ensure_ascii=False,indent=2).encode('utf-8')
    archive=OUT/'a2_sentence_v4_fp16_bundle.bundle'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,data in files.items():z.writestr(n,data)
    nb=build_notebook()
    nb['cells'][0]['source']=['# A2 v4 FP16 continuation\n',
        'Recovered ~0.5-epoch adapter; train 1.5 additional epochs with fresh optimizer.\n',
        'This is a warm start, not an exact optimizer/sample-order resume.\n',
        'T4 uses FP16. Speed guard after 5 steps. No before-evaluation, no reserved test.\n',
        'Checkpoint labels 0.5/1.0/1.5 refer to this additional stage.\n']
    for c in nb['cells'][1:]:
        s=''.join(c['source'])
        s=s.replace('a2_sentence_v4_kaggle_bundle.bundle','a2_sentence_v4_fp16_bundle.bundle')
        s=s.replace('a2_sentence_v4_kaggle_bundle.zip','a2_sentence_v4_fp16_bundle.zip')
        s=s.replace('/kaggle/working/a2_v4_work','/kaggle/working/a2_v4_fp16_work')
        s=s.replace('/kaggle/working/results_v4','/kaggle/working/results_v4_fp16')
        s=s.replace('/kaggle/working/a2_sentence_v4_kaggle_results.zip','/kaggle/working/a2_sentence_v4_fp16_results.zip')
        s=s.replace("'--epochs', '2.0'","'--epochs', '1.5'")
        if 'scripts.train_a2_v4_kaggle' in s:
            needle="'--initial-adapter', str(WORK / 'initial_adapter'),"
            assert needle in s
            s=s.replace(needle,needle+f"\n    '--expected-adapter-sha256', '{digest}',")
        if 'print(\'Saved results zip:' in s:
            s="RESULTS.mkdir(parents=True,exist_ok=True)\n(RESULTS/'continuation_plan.json').write_text((WORK/'continuation_plan.json').read_text())\n"+s
        if c['cell_type']=='code':ast.parse(s)
        c['source']=s.splitlines(keepends=True)
        c['id']=hashlib.sha256(s.encode()).hexdigest()[:12]
    dataset=OUT/'upload_dataset';dataset.mkdir(exist_ok=True)
    shutil.copyfile(archive,dataset/archive.name)
    (dataset/'dataset-metadata.json').write_text(json.dumps({'id':'vunguyen25552/dental-a2-v4-fp16-continuation',
        'title':'Dental A2 v4 FP16 continuation','licenses':[{'name':'other'}]},indent=2))
    kernel=OUT/'kernel';kernel.mkdir(exist_ok=True)
    (kernel/'train.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':'vunguyen25552/dental-a2-qwen3-v4-fp16',
        'title':'Dental A2 Qwen3 v4 FP16','code_file':'train.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
        'dataset_sources':['vunguyen25552/dental-a2-v4-fp16-continuation'],'competition_sources':[],'kernel_sources':[]},indent=2))
    (OUT/'continuation_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'ready':report['ready'],'parent_sha256':digest,'additional_epochs':1.5,'bundle':str(archive)},indent=2))


if __name__=='__main__':main()
