import os
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import sys,json,time,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'protocol_v3'))
from src.training.a2_sentence_protocol import make_messages, resolve_selection
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
torch.set_num_threads(6)
retrieval=json.loads((OUT/'retrieval.json').read_text(encoding='utf-8'))
payload={'question':retrieval['question'],'evidence':[
    {k:e[k] for k in ('evidence_id','title','content')} for e in retrieval['evidence']]}
normalized='--normalize-whitespace' in sys.argv
suffix='_whitespace' if normalized else ''
if normalized:
    for e in payload['evidence']: e['content']=' '.join(e['content'].split())
messages=make_messages(payload)
(OUT/('messages'+suffix+'.json')).write_text(json.dumps(messages,ensure_ascii=False,indent=2),encoding='utf-8')
base=ROOT/'data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
print('Input tokens:',ids.shape[1],flush=True)
print('Diagnostic full-context override: no truncation; training/runtime budget was 4096.',flush=True)
digest=hashlib.sha256((OUT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
assert digest=='f80acf2d73be13e3ff2f8a6ccb18fca22a79e24baa73eda901a683f7fcb4a067'
model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),
    device_map={'':0},dtype=torch.float16,attn_implementation='sdpa')
if ids.shape[1]+256>model.config.max_position_embeddings:raise ValueError('Exceeds native model context')
model=PeftModel.from_pretrained(model,str(OUT/'adapter')).eval();model.requires_grad_(False)
ids=ids.to('cuda');start=time.perf_counter()
with torch.inference_mode():
    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
        temperature=None,top_p=None,top_k=None,max_new_tokens=256,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
raw=tok.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
result={'question':payload['question'],'adapter_sha256':digest,'epoch':2,'raw_output':raw,
    'whitespace_normalized':normalized,
    'training_runtime_token_budget':4096,'diagnostic_full_context_override':True,
    'seconds':time.perf_counter()-start,'input_tokens':int(ids.shape[1]),
    'generated_tokens':int(generated.shape[1]-ids.shape[1]),'generation_limit_hit':int(generated.shape[1]-ids.shape[1])>=256,
    'scope':'A2 only; production internal RAG; frozen v3 training protocol; no other agents or rule-based drafting'}
try:result['resolved_output']=resolve_selection(json.loads(raw),payload).model_dump()
except Exception as e:result['resolution_error']=str(e)
(OUT/('result'+suffix+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
