"""A2 only: same frozen inputs, base disabled adapter versus v3 epoch2. Inference only."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a2 import image as training_image, volume, ROOT

RUN='a2_qwen3_17b_20261110_194806_350795'
DATA=ROOT/'data/evaluation/a2_base_vs_v3_20261111'
app=modal.App('dental-a2-base-v3-diagnostic')
image=(training_image.add_local_file(ROOT/'scripts/modal_compare_a2_base_v3.py','/app/scripts/modal_compare_a2_base_v3.py')
       .add_local_file(ROOT/'scripts/modal_train_a2.py','/app/scripts/modal_train_a2.py'))


@app.function(image=image,gpu=['L4','A10G'],cpu=4,memory=24576,timeout=7200,retries=0,volumes={'/vol':volume})
def compare(packet):
    import time
    from contextlib import nullcontext
    from datetime import datetime,timezone,timedelta
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from peft.tuners.tuners_utils import BaseTunerLayer
    from scripts.train_a2 import MODEL,REVISION,configure_tokenizer
    source=Path('/vol/runs')/RUN
    read=lambda n:json.loads((source/n).read_text())
    assert read('run_status.json')['status']=='completed'
    cfg=read('config.json');assert cfg['model']==MODEL and cfg['revision']==REVISION
    assert read('metrics.json')['best_checkpoint']==str(source/'checkpoints/checkpoint-252')
    weights=source/'adapter/adapter_model.safetensors'
    sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert sha==packet['adapter_sha256']=='47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    adapter_config=json.loads((source/'adapter/adapter_config.json').read_text())
    assert not adapter_config.get('modules_to_save') and adapter_config['peft_type']=='LORA'
    output=Path('/vol/comparisons')/('a2_base_v3_'+datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    save=lambda n,v:(output/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    save('run_status.json',{'status':'running'})
    save('inputs.json',packet['cases'])
    save('identity.json',{'base_model':MODEL,'revision':REVISION,'adapter_run':RUN,'selected_epoch':2,
        'adapter_sha256':sha,'benchmark_manifest':packet['manifest'],'gpu':torch.cuda.get_device_name(0),
        'variants':{'base':'same frozen checkpoint with all LoRA layers disabled','v3':'A2 v3 epoch2 enabled'},
        'quantization':'NF4 double / BF16 compute; same shared weights','max_new_tokens':512,'do_sample':False,
        'thinking_enabled':False,'training_executed':False,'other_agents_executed':False,
        'rag_executed':False,'nli_executed':False,'format_repair_attempts':0})
    volume.commit();started=time.monotonic()
    try:
        set_seed(42)
        tokenizer=configure_tokenizer(AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False),'qwen3_17b')
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model=PeftModel.from_pretrained(model,source/'adapter',adapter_name='a2',is_trainable=False)
        model.eval();model.config.use_cache=True
        with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
            for i,case in enumerate(packet['cases']):
                messages=[{'role':'system','content':case['system']},{'role':'user','content':case['user']}]
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
                assert ids.shape[1]<=4096,'Input exceeds production limit'
                prompt_sha=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                for label in (('base','v3') if i%2==0 else ('v3','base')):
                    model.set_adapter('a2');model.requires_grad_(False)
                    assert not any(p.requires_grad for p in model.parameters())
                    before=time.monotonic()
                    with (model.disable_adapter() if label=='base' else nullcontext()),torch.inference_mode():
                        layers=[m for m in model.modules() if isinstance(m,BaseTunerLayer)]
                        assert layers and all(bool(m.disable_adapters)==(label=='base') for m in layers)
                        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=512,
                            do_sample=False,temperature=None,top_p=None,top_k=None,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                    row={'id':case['id'],'variant':label,'prompt_ids_sha256':prompt_sha,
                         'input_tokens':int(ids.shape[1]),'output_tokens':int(generated.shape[1]-ids.shape[1]),
                         'raw_output':tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip(),
                         'seconds':time.monotonic()-before}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if i==0 or (i+1)%4==0 or i+1==len(packet['cases']):
                    print(f'A2 base/v3: {i+1}/{len(packet["cases"])} paired cases',flush=True);volume.commit()
        save('run_status.json',{'status':'completed','paired_cases':len(packet['cases']),
                              'generations':2*len(packet['cases']),'seconds':time.monotonic()-started})
        print('Completed:',output,flush=True)
        return str(output)
    except BaseException as exc:
        save('run_status.json',{'status':'failed','error':str(exc)[:1000]});raise
    finally:volume.commit()


@app.local_entrypoint()
def main():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    blob=(DATA/'cases.jsonl').read_bytes();assert hashlib.sha256(blob).hexdigest()==manifest['cases_sha256']
    cases=list(map(json.loads,blob.decode('utf-8').splitlines()))
    assert len(cases)==manifest['count']==52
    # No reference answers or grading proofs sent to GPU inference.
    inputs=[{k:c[k] for k in ('id','system','user')} for c in cases]
    print(compare.remote({'manifest':manifest,'cases':inputs,
        'adapter_sha256':'47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'}))
