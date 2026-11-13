"""Batch A2-only definition probes, real internal RAG and completed v3 epoch 2."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/a2_probe_v3_definitions_20261113'
QUESTIONS=['Máng nhai là gì?','Viêm nướu là gì?','Viêm nha chu là gì?','Cao răng là gì?']


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def retrieve():
    from src.rag.local_embeddings import LocalE5Embeddings
    from src.rag.reranker import LocalReranker
    from src.rag.reranked_retriever import RerankedRetriever
    from src.rag.bilingual_queries import BilingualQueries,NllbTranslator
    from src.agents.contracts import evidence_bundle
    OUT.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'config/assistant_local.json').read_text(encoding='utf-8'))['knowledge']
    print('Loading retrieval models',flush=True)
    r=RerankedRetriever.from_disk(ROOT/cfg['internal_index'],LocalReranker(),LocalE5Embeddings(),candidate_k=cfg['candidate_k'])
    if cfg.get('bilingual_queries'):r.query_expander=BilingualQueries(NllbTranslator(ROOT/cfg['query_translation_dir']))
    r.context_neighbors=cfg.get('context_neighbors',0);r.min_rerank_logit=cfg.get('internal_min_rerank_logit')
    for i,q in enumerate(QUESTIONS):
        path=OUT/f'retrieval_{i}.json'
        if path.exists():continue
        start=time.perf_counter();hits=r.expand_context(r.retrieve(q,top_k=3))
        evidence=evidence_bundle(hits,'I','internal')
        dump(path,dict(question=q,evidence=[e.model_dump() for e in evidence],retrieval=r.last_run,
                       seconds=time.perf_counter()-start,config=cfg))
        print(q,len(evidence),'chunks',round(time.perf_counter()-start,1),'seconds',flush=True)


def infer(wait_for_retrieval=False):
    import os,sys,hashlib
    os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    frozen=ROOT/'outputs/a2_probe_v3_rag_caries_20261113'
    sys.path.insert(0,str(frozen/'protocol_v3'))
    from src.training.a2_sentence_protocol import make_messages,resolve_selection
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    torch.set_num_threads(6)
    base=ROOT/'data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    digest=hashlib.sha256((frozen/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
    assert digest=='f80acf2d73be13e3ff2f8a6ccb18fca22a79e24baa73eda901a683f7fcb4a067'
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),
        device_map={'':0},dtype=torch.float16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(model,str(frozen/'adapter')).eval();model.requires_grad_(False)
    for i,q in enumerate(QUESTIONS):
        if (OUT/f'result_{i}.json').exists():continue
        path=OUT/f'retrieval_{i}.json'
        if wait_for_retrieval and not path.exists():
            print('Waiting for retrieval:',q,flush=True)
            deadline=time.monotonic()+300
            while not path.exists() and time.monotonic()<deadline:time.sleep(1)
        retrieval=json.loads(path.read_text(encoding='utf-8'))
        if not retrieval['evidence']:
            result=dict(question=q,epoch=2,adapter_sha256=digest,
                resolved_output={'status':'insufficient_evidence','claims':[]},
                model_invoked=False,reason='Production A2 stops when internal RAG returns no evidence')
            dump(OUT/f'result_{i}.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
            continue
        payload={'question':q,'evidence':[dict(evidence_id=e['evidence_id'],title=e['title'],content=' '.join(e['content'].split())) for e in retrieval['evidence']]}
        messages=make_messages(payload);dump(OUT/f'messages_{i}.json',messages)
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt')
        result=dict(question=q,epoch=2,adapter_sha256=digest,input_tokens=int(ids.shape[1]),
            whitespace_normalized=True,all_retrieved_chunks_retained=True,training=False)
        if ids.shape[1]>4096:
            result['blocked']='Retrieved context exceeds 4096-token runtime budget; no truncation or manual answer selection'
        else:
            ids=ids.to('cuda');start=time.perf_counter()
            try:
                with torch.inference_mode():
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,temperature=None,top_p=None,top_k=None,
                        max_new_tokens=256,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
                raw=tok.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
                result.update(raw_output=raw,seconds=time.perf_counter()-start,generation_limit_hit=int(generated.shape[1]-ids.shape[1])>=256)
                try:result['resolved_output']=resolve_selection(json.loads(raw),payload).model_dump()
                except Exception as exc:result['resolution_error']=str(exc)
                del generated
            except torch.OutOfMemoryError as exc:result['generation_error']=str(exc)
            del ids;torch.cuda.empty_cache()
        dump(OUT/f'result_{i}.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['retrieve','infer'])
    p.add_argument('--cases',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--wait-for-retrieval',action='store_true');a=p.parse_args()
    if a.cases:QUESTIONS=[r['question'] for r in json.loads(a.cases.read_text(encoding='utf-8'))]
    if a.output_dir:OUT=a.output_dir.resolve()
    if a.stage=='retrieve':retrieve()
    else:infer(wait_for_retrieval=a.wait_for_retrieval)
