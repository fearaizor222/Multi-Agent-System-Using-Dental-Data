"""Train one Qwen3 RAFT pilot and compare it with A2 v3 on fixed validation cases."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a2 import ROOT, image as base_image, volume

app = modal.App('dental-a2-raft-pilot-training')
image = (base_image
    .add_local_file(ROOT/'scripts/modal_train_a2.py','/app/scripts/modal_train_a2.py')
    .add_local_file(ROOT/'scripts/modal_train_a2_raft_pilot.py','/app/scripts/modal_train_a2_raft_pilot.py'))
LOCAL = ROOT/'outputs/a2_raft_training_20261112'


def paired_probe(output, records):
    import time
    import torch
    from transformers import AutoTokenizer
    from scripts.train_a2 import MODEL, REVISION, MODELS, configure_tokenizer, load_inference_model, inference_dtype_signature, score_answer, summarize_results
    old = Path('/vol/runs/a2_qwen3_17b_20261110_194806_350795/adapter')
    assert hashlib.sha256((old/'adapter_model.safetensors').read_bytes()).hexdigest() == '47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False),'qwen3_17b')
    model = load_inference_model(MODELS['qwen3_17b'],old,adapter_name='v3')
    model.load_adapter(output/'adapter',adapter_name='pilot',is_trainable=False)
    model.eval(); model.config.use_cache=True
    model.requires_grad_(False)
    directory=output/'paired_probe';directory.mkdir()
    identity={'model':MODEL,'revision':REVISION,'gpu':torch.cuda.get_device_name(0),
        'dtype_signature':inference_dtype_signature(model),
        'variants':{v:hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()
                    for v,path in [('v3',old),('pilot',output/'adapter')]},
        'scope':'Same quote prompt, inputs, shared NF4/BF16 base, GPU and greedy decoding for both adapters. 32 validation variants, not independent test.'}
    (directory/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    metrics={v:[] for v in ('v3','pilot')}
    with (directory/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for index,record in enumerate(records):
            ids=tokenizer.apply_chat_template(record['messages'][:2],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_tensors='pt').to('cuda')
            digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            for variant in (('v3','pilot') if index%2==0 else ('pilot','v3')):
                model.set_adapter(variant);model.requires_grad_(False)
                assert not any(p.requires_grad for p in model.parameters())
                start=time.monotonic()
                with torch.inference_mode():
                    result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                        max_new_tokens=512,do_sample=False,temperature=None,top_p=None,top_k=None,
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                raw=tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
                row={'id':record['metadata']['id'],'variant':variant,'prompt_ids_sha256':digest,
                    'raw_output':raw,'seconds':time.monotonic()-start,
                    'input_tokens':int(ids.shape[1]),'output_tokens':int(result.shape[1]-ids.shape[1]),
                    **score_answer(record,raw)}
                metrics[variant].append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            if index==0 or (index+1)%4==0:
                print(f'Paired v3/pilot: {index+1}/{len(records)}',flush=True);volume.commit()
    summary={v:summarize_results(rows) for v,rows in metrics.items()}
    (directory/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    volume.commit()


@app.function(image=image,gpu=['L4','A10G'],cpu=4,memory=24576,timeout=10800,retries=0,
              volumes={'/vol':volume})
def run(packet, probes, epochs):
    import gc
    import torch
    from datetime import datetime,timezone,timedelta
    from scripts.train_a2 import train
    name='a2_raft_qwen3_17b_'+datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output=Path('/vol/runs')/name
    print('Run:',name,flush=True)
    train(packet,output,epochs=epochs,commit=volume.commit)
    status=output/'experiment_status.json'
    status.write_text(json.dumps({'status':'paired_probe_running'}),encoding='utf-8');volume.commit()
    try:
        gc.collect();torch.cuda.empty_cache()
        paired_probe(output,probes)
        status.write_text(json.dumps({'status':'completed'}),encoding='utf-8')
    except BaseException as exc:
        status.write_text(json.dumps({'status':'paired_probe_failed','error':str(exc)[:1000]}),encoding='utf-8')
        raise
    finally:volume.commit()
    print('Experiment completed:',output,flush=True)
    return str(output)


@app.local_entrypoint()
def main(epochs: int = 3):
    from scripts.train_a2 import load_data
    packet=load_data(ROOT/'data/processed/a2_raft_pilot_v1')
    assert len(packet['train'])==144 and len(packet['validation'])==72
    previous=json.loads((ROOT/'outputs/a2_raft_prompt_probe_20261112/cases.json').read_text(encoding='utf-8'))
    by_id={r['metadata']['id']:r for r in packet['validation']}
    probes=[by_id[c['id']] for c in previous]
    assert len(probes)==32
    for record,case in zip(probes,previous):
        assert record['messages']==case['quote_prompt']['messages']
    LOCAL.mkdir(exist_ok=False)
    plan={'epochs':epochs,'initialization':'fresh Qwen3-1.7B base, new LoRA',
        'dataset_info':packet['info'],'probe_ids':[r['metadata']['id'] for r in probes],
        'checkpoint_selection':'lowest validation loss','production_adapter_replaced':False}
    (LOCAL/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    result=run.remote(packet,probes,epochs)
    (LOCAL/'remote_path.txt').write_text(result,encoding='utf-8')
    print(result)
