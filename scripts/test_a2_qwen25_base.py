"""Small source-reading diagnostic: base Qwen2.5 only, no adapters or training."""
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/a2_qwen25_base_20261112'
    out.mkdir(exist_ok=False)
    source=ROOT/'data/evaluation/a2_base_vs_v3_20261111/cases.jsonl'
    cases=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    indices=[0,11,13,17,22,26,27,32,37,38,39,47]
    base=ROOT/'data/models/Qwen--Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    (out/'identity.json').write_text(json.dumps({'model':'Qwen/Qwen2.5-1.5B-Instruct','revision':base.name,'adapter':None,'quantization':'NF4 double, BF16 compute','indices':indices,'scope':'12 selected diagnostic cases, not an independent benchmark; identical A2 system/user text as previous comparison; model-specific chat template','gpu':torch.cuda.get_device_name(0),'training':False},indent=2),encoding='utf-8')
    with (out/'predictions.jsonl').open('w',encoding='utf-8',buffering=1) as f:
        for index in indices:
            c=cases[index]
            ids=tokenizer.apply_chat_template([{'role':'system','content':c['system']},{'role':'user','content':c['user']}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
            start=time.perf_counter()
            with torch.inference_mode():
                result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,temperature=None,top_p=None,top_k=None,max_new_tokens=512,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            row={'index':index,'id':c['id'],'question':c['payload']['question'],'raw_output':tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip(),'seconds':time.perf_counter()-start,'input_tokens':ids.shape[1],'output_tokens':result.shape[1]-ids.shape[1]}
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(row,ensure_ascii=False),flush=True)
    print('Completed',out,flush=True)

if __name__=='__main__': main()
