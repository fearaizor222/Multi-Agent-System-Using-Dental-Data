"""Paired A2 adapter inference on a frozen common benchmark. No training or gold upload."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a2 import image as training_image, volume, ROOT

RUNS={'old':'a2_qwen3_17b_20261107_101037_528370','new':'a2_qwen3_17b_20261109_213712_836549'}
app=modal.App('dental-a2-paired-comparison')
image=(training_image.add_local_file(ROOT/'scripts/modal_compare_a2.py','/app/scripts/modal_compare_a2.py')
       .add_local_file(ROOT/'scripts/modal_train_a2.py','/app/scripts/modal_train_a2.py'))


@app.function(image=image,gpu=['L4','A10G'],cpu=4,memory=24576,timeout=7200,retries=0,volumes={'/vol':volume})
def compare(packet):
    import time
    from datetime import datetime, timezone, timedelta
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from scripts.train_a2 import MODEL, REVISION, configure_tokenizer
    from src.agents.contracts import GeneratedDraft
    stamp=datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output=Path('/vol/comparisons')/('paired_'+stamp)
    output.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    identities={}
    for label,run in RUNS.items():
        folder=Path('/vol/runs')/run
        config=json.loads((folder/'config.json').read_text())
        status=json.loads((folder/'run_status.json').read_text())
        assert status['status']=='completed'
        assert config['model']==MODEL and config['revision']==REVISION
        identities[label]={'run':run,'weights_sha256':hashlib.sha256((folder/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()}
    save('identity.json',{'adapters':identities,'benchmark_manifest':packet['manifest'],
        'base':MODEL,'revision':REVISION,'quantization':'NF4 double / BF16','max_new_tokens':512,
        'do_sample':False,'training_executed':False,'gpu':torch.cuda.get_device_name(0),
        'prompt_policy':'same current production system for both adapters'})
    save('run_status.json',{'status':'running'})
    save('inputs.json',packet['cases'])
    volume.commit()
    started=time.monotonic()
    try:
        set_seed(42)
        tokenizer=configure_tokenizer(AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False),'qwen3_17b')
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model=PeftModel.from_pretrained(model,str(Path('/vol/runs')/RUNS['old']/'adapter'),adapter_name='old',is_trainable=False)
        model.load_adapter(str(Path('/vol/runs')/RUNS['new']/'adapter'),adapter_name='new',is_trainable=False)
        model.eval();model.config.use_cache=True
        with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
            for i,case in enumerate(packet['cases']):
                user=json.dumps(case['payload'],ensure_ascii=False)+'\nJSON schema:\n'+json.dumps(GeneratedDraft.model_json_schema(),ensure_ascii=False)
                messages=[{'role':'system','content':case['system']},{'role':'user','content':user}]
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
                if ids.shape[1]+512>4096:raise ValueError('Benchmark exceeds runtime token budget')
                for label in (('old','new') if i%2==0 else ('new','old')):
                    model.set_adapter(label); model.requires_grad_(False)
                    assert not any(p.requires_grad for p in model.parameters())
                    before=time.monotonic()
                    with torch.inference_mode():
                        result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=512,
                            do_sample=False,temperature=None,top_p=None,top_k=None,
                            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                    row={'id':case['id'],'adapter':label,'raw_output':tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip(),
                        'tokens':int(result.shape[1]-ids.shape[1]),'elapsed_seconds':time.monotonic()-before}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if i==0 or (i+1)%6==0:
                    print(f'Paired comparison: {i+1}/{len(packet["cases"])} cases, both adapters',flush=True)
                    volume.commit()
        save('run_status.json',{'status':'completed','elapsed_seconds':time.monotonic()-started})
        print('Completed:',output,flush=True)
        return str(output)
    except BaseException as exc:
        save('run_status.json',{'status':'failed','error':str(exc)[:1000]});raise
    finally:volume.commit()


@app.local_entrypoint()
def main():
    folder=ROOT/'data/evaluation/a2_paired_20261110'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((folder/'cases.jsonl').read_bytes()).hexdigest()==manifest['benchmark_sha256']
    cases=[json.loads(l) for l in (folder/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    stripped=[{k:c[k] for k in ('id','family','case','language','payload','system')} for c in cases]
    assert len(stripped)==manifest['count']==72
    print('72 identical contexts/questions, frozen old/new adapters; no gold sent to GPU.',flush=True)
    print(compare.remote({'manifest':manifest,'cases':stripped}))
