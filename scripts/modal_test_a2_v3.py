"""A2 v3 held-out test with the frozen epoch2 adapter. No training/model selection."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a2 import image as training_image,volume,ROOT

RUN='a2_qwen3_17b_20261110_194806_350795'
DATA=ROOT/'data/processed/a2_drafting_v3'
app=modal.App('dental-a2-v3-heldout-test')
image=(training_image.add_local_file(ROOT/'scripts/modal_test_a2_v3.py','/app/scripts/modal_test_a2_v3.py')
       .add_local_file(ROOT/'scripts/modal_train_a2.py','/app/scripts/modal_train_a2.py'))


def load_test():
    from scripts.train_a2 import load_data,parse_sample
    info=load_data(DATA)['info']  # Validate current split boundaries and review provenance.
    records=[json.loads(l) for l in (DATA/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    metadata=[json.loads(l) for l in (DATA/'test.metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==len(metadata)==93 and len({m['id'] for m in metadata})==93
    for record,meta in zip(records,metadata):
        parse_sample(record)
        assert meta['split']=='test'
        record['metadata']=meta
    for name,sha in info['source_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==sha
    return {'records':records,'dataset_version':info['dataset_version'],'dataset_sha256':info['source_sha256']}


@app.function(image=image,gpu=['L4','A10G'],cpu=4,memory=24576,timeout=7200,retries=0,volumes={'/vol':volume})
def test_frozen_adapter(packet):
    import time
    from datetime import datetime,timezone,timedelta
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from scripts.train_a2 import MODEL,REVISION,configure_tokenizer,encode_sample,evaluate
    source=Path('/vol/runs')/RUN
    read=lambda name:json.loads((source/name).read_text())
    config,info,metrics,status=read('config.json'),read('dataset_info.json'),read('metrics.json'),read('run_status.json')
    assert status['status']=='completed'
    assert config['model']==MODEL and config['revision']==REVISION
    assert info['dataset_version']==packet['dataset_version']=='a2-v3-repaired-20261110-r1'
    assert info['source_sha256']==packet['dataset_sha256']
    selected=Path(metrics['best_checkpoint'])
    assert selected==source/'checkpoints/checkpoint-252'
    state=json.loads((selected/'trainer_state.json').read_text())
    assert state['global_step']==252 and abs(state['epoch']-2.0)<1e-6
    weights=source/'adapter/adapter_model.safetensors'
    checkpoint_weights=selected/'adapter_model.safetensors'
    adapter_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    checkpoint_sha=hashlib.sha256(checkpoint_weights.read_bytes()).hexdigest()
    if adapter_sha!=checkpoint_sha:
        from safetensors.torch import load_file
        a,b=load_file(str(weights)),load_file(str(checkpoint_weights))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'Saved adapter differs from selected epoch2'
        del a,b
    output=Path('/vol/tests')/('a2_v3_qwen_epoch02_'+datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('identity.json',{'source_run':RUN,'selected_epoch':2,'selected_checkpoint':str(selected),
        'adapter_sha256':adapter_sha,'checkpoint_sha256':checkpoint_sha,'weights_equal_to_epoch2':True,
        'dataset_sha256':packet['dataset_sha256'],'dataset_version':packet['dataset_version'],
        'model':MODEL,'revision':REVISION,'training_executed':False,'test_used_for_selection':False,
        'max_new_tokens':config['max_new_tokens'],'gpu':torch.cuda.get_device_name(0)})
    save('run_status.json',{'status':'running'})
    with (output/'test.jsonl').open('w',encoding='utf-8') as stream:
        for row in packet['records']:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    volume.commit();started=time.monotonic()
    try:
        set_seed(config['seed'])
        tokenizer=configure_tokenizer(AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False),'qwen3_17b')
        assert tokenizer.chat_template==config['chat_template'] and tokenizer.eos_token_id==config['eos_token_id']
        for row in packet['records']:encode_sample(tokenizer,row,config['max_length'])
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model=PeftModel.from_pretrained(model,str(source/'adapter'),is_trainable=False)
        assert not any(p.requires_grad for p in model.parameters())
        result=evaluate(model,tokenizer,packet['records'],output/'test',config['max_new_tokens'])
        save('metrics.json',{'test':result,'elapsed_seconds':time.monotonic()-started,
            'scope':'Held-out v3 reference-label agreement; not clinical correctness. Frozen epoch2, no training/selection on test.'})
        plot_test(result,output/'figures')
        save('run_status.json',{'status':'completed'})
        print('Completed A2 v3 test:',output,flush=True)
        return str(output)
    except BaseException as exc:
        save('run_status.json',{'status':'failed','error':str(exc)[:1000]});raise
    finally:volume.commit()


def plot_test(metrics,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.train_a2 import STATUSES
    folder.mkdir()
    matrix=metrics['confusion_matrix']
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ax.imshow(matrix,cmap='Blues')
    ax.set_xticks(range(3),STATUSES+['invalid'],rotation=20)
    ax.set_yticks(range(2),STATUSES)
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center')
    ax.set(title='A2 Qwen3 v3 epoch2: held-out test (n=93)',xlabel='Predicted',ylabel='Reference')
    fig.savefig(folder/'04_confusion_matrix.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    bars=ax.bar(STATUSES,[metrics['per_status'][s]['f1']*100 for s in STATUSES])
    ax.bar_label(bars,fmt='%.1f');ax.set(title='A2 Qwen3 v3 epoch2: held-out test (n=93)',ylabel='F1 (%)',ylim=(0,110))
    fig.savefig(folder/'05_f1_by_decision.png',dpi=300);plt.close(fig)


@app.local_entrypoint()
def main():
    packet=load_test()
    print('A2 Qwen3 v3 epoch2; frozen held-out test:',len(packet['records']),'samples',flush=True)
    print(test_frozen_adapter.remote(packet))
