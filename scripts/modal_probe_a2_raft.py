"""Inference-only paired prompt trial on the existing A2 v3 adapter."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a2 import ROOT, image as base_image, volume

DATA = ROOT/'data/processed/a2_raft_pilot_v1'
OUT = ROOT/'outputs/a2_raft_prompt_probe_20261112'
app = modal.App('dental-a2-raft-prompt-probe')
image = (base_image.add_local_file(ROOT/'scripts/modal_probe_a2_raft.py','/app/scripts/modal_probe_a2_raft.py')
         .add_local_file(ROOT/'scripts/modal_train_a2.py','/app/scripts/modal_train_a2.py'))
RUN = 'a2_qwen3_17b_20261110_194806_350795'


def prepare():
    import ast
    from scripts.train_a2 import load_data
    from src.agents.contracts import GeneratedDraft
    drafting = ast.parse((ROOT/'src/agents/drafting.py').read_text(encoding='utf-8'))
    draft_system = next(ast.literal_eval(node.value) for node in drafting.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DRAFT_SYSTEM' for t in node.targets))
    packet = load_data(DATA)
    selected = []
    for record in packet['validation']:
        meta = record['metadata']
        single = meta['question_id'].endswith(('_q0','_q1'))
        wanted = (single and meta['mode'] in ('clean','noise_middle','missing_support_same_topic')) or (
            not single and meta['mode'] in ('noise_middle','partial_answer'))
        if not wanted:
            continue
        body = record['messages'][1]['content'].split('\nJSON schema:\n')[0]
        gold = json.loads(record['messages'][2]['content'])
        old_gold = {'status':gold['status'],'claims':[{k:c[k] for k in ('text','evidence_ids')} for c in gold['claims']]}
        old = {'messages':[{'role':'system','content':draft_system},
            {'role':'user','content':body+'\nJSON schema:\n'+json.dumps(GeneratedDraft.model_json_schema(),ensure_ascii=False)},
            {'role':'assistant','content':json.dumps(old_gold,ensure_ascii=False)}]}
        selected.append({'id':meta['id'],'metadata':meta,
                         'old_prompt':old,'quote_prompt':{'messages':record['messages']}})
    selected.sort(key=lambda r:r['id'])
    assert len(selected)==32
    OUT.mkdir(exist_ok=True)
    if (OUT/'cases.json').exists():
        assert json.loads((OUT/'cases.json').read_text(encoding='utf-8'))==selected
    else:
        (OUT/'cases.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'count':32,'generations':64,'dataset_files_sha256':packet['info']['source_sha256'],
        'scope':'32 selected validation variants from 4 families and 12 distinct questions; paired same A2 v3 weights, contexts, questions and decoding; prompt/schema differs. No training.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return [{ 'id':r['id'], **{v:r[v]['messages'][:2] for v in ('old_prompt','quote_prompt')}} for r in selected], manifest


@app.function(image=image,gpu=['L4','A10G'],cpu=4,memory=16384,timeout=3600,retries=0,volumes={'/vol':volume})
def probe(cases, manifest):
    import time
    from datetime import datetime,timezone,timedelta
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel
    from scripts.train_a2 import MODEL,REVISION
    source=Path('/vol/runs')/RUN
    assert json.loads((source/'run_status.json').read_text())['status']=='completed'
    weights=source/'adapter/adapter_model.safetensors'
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest=='47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    config=json.loads((source/'config.json').read_text());assert config['model']==MODEL and config['revision']==REVISION
    output=Path('/vol/comparisons')/('a2_raft_prompt_'+datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    save=lambda n,v:(output/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    save('run_status.json',{'status':'loading'});volume.commit()
    started=time.monotonic()
    try:
        set_seed(42)
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
        model=PeftModel.from_pretrained(model,source/'adapter',adapter_name='a2',is_trainable=False)
        model.set_adapter('a2');model.requires_grad_(False);model.eval();model.config.use_cache=True
        assert not any(p.requires_grad for p in model.parameters())
        save('identity.json',{'model':MODEL,'revision':REVISION,'adapter_run':RUN,'adapter_sha256':digest,
            'gpu':torch.cuda.get_device_name(0),'quantization':'NF4 double, BF16 compute','max_new_tokens':512,
            'do_sample':False,'thinking':False,'training_executed':False,'manifest':manifest})
        save('run_status.json',{'status':'running'});volume.commit()
        with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
            for i,case in enumerate(cases):
                for variant in (('old_prompt','quote_prompt') if i%2==0 else ('quote_prompt','old_prompt')):
                    ids=tokenizer.apply_chat_template(case[variant],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
                    assert ids.shape[1]<=3072
                    before=time.monotonic()
                    with torch.inference_mode():
                        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=512,
                            do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    row={'id':case['id'],'variant':variant,'input_tokens':int(ids.shape[1]),
                        'output_tokens':int(generated.shape[1]-ids.shape[1]),'seconds':time.monotonic()-before,
                        'raw_output':tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if i==0 or (i+1)%4==0:
                    print(f'Completed {i+1}/{len(cases)} pairs',flush=True);volume.commit()
        save('run_status.json',{'status':'completed','pairs':len(cases),'generations':2*len(cases),'seconds':time.monotonic()-started})
        print('Results:',output,flush=True)
        return str(output)
    except BaseException as exc:
        save('run_status.json',{'status':'failed','error':str(exc)[:1000]});raise
    finally:volume.commit()


@app.local_entrypoint()
def main():
    cases,manifest=prepare()
    print(probe.remote(cases,manifest))
