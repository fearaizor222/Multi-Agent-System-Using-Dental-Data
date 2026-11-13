"""Source-context replay of completed A2 v4; eight diagnostic questions only."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import json,sys,time,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'protocol_v4'))
from src.training.a2_sentence_protocol import make_messages,resolve_selection,normalize_payload_evidence
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
torch.set_num_threads(6)
identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8'))
assert hashlib.sha256((OUT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==identity['adapter_sha256']
base=ROOT/'data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
prepared=[]
for i,case in enumerate(cases):
    payload=normalize_payload_evidence({'question':case['question'],'evidence':[
        {k:e[k] for k in ['evidence_id','title','content']} for e in case['evidence']]})
    for e in payload['evidence']:e['content']=' '.join(e['content'].split())
    messages=make_messages(payload)
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
    (OUT/f'messages_{i}.json').write_text(json.dumps(messages,ensure_ascii=False,indent=2),encoding='utf-8')
    prepared.append((payload,ids))
print('Token counts:',[int(ids.shape[1]) for _,ids in prepared],flush=True)
model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),
    device_map={'':0},dtype=torch.float16,attn_implementation='sdpa')
model=PeftModel.from_pretrained(model,str(OUT/'adapter')).eval();model.requires_grad_(False)
for i,(case,(payload,ids)) in enumerate(zip(cases,prepared)):
    target=OUT/f'result_{i}.json'
    if target.exists():continue
    result=dict(question=case['question'],kind=case['kind'],input_tokens=int(ids.shape[1]),adapter_sha256=identity['adapter_sha256'],
        checkpoint=identity['selected_checkpoint'],context_replay=True,whitespace_normalized=True,v4_ocr_normalization=True,
        original_retrieval=case['retrieval_reused_from'],all_retrieved_chunks_retained=True)
    if ids.shape[1]>4096:result['blocked']='Context exceeds 4096-token runtime budget; no chunks dropped'
    elif not payload['evidence']:result.update(model_invoked=False,resolved_output={'status':'insufficient_evidence','claims':[]})
    else:
        ids=ids.to('cuda');start=time.perf_counter()
        try:
            with torch.inference_mode():
                generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                    temperature=None,top_p=None,top_k=None,max_new_tokens=384,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
            raw=tok.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
            result.update(raw_output=raw,seconds=time.perf_counter()-start,generation_tokens=int(generated.shape[1]-ids.shape[1]),
                          generation_limit_hit=int(generated.shape[1]-ids.shape[1])>=384)
            try:result['resolved_output']=resolve_selection(json.loads(raw),payload).model_dump()
            except Exception as exc:result['resolution_error']=str(exc)
            del generated
        except torch.OutOfMemoryError as exc:result['generation_error']=str(exc)
        del ids;torch.cuda.empty_cache()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
