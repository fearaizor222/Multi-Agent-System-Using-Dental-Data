"""Structural/provenance audit; semantic review remains a source-reading judgment."""
import json
import hashlib
import re
from collections import Counter,defaultdict
from pathlib import Path
from scripts.build_a2_hard_v3 import ROOT,OUTPUT,BASE,rows,sha
from src.training.a2_sentence_protocol import prepare_payload,resolve_selection,SYSTEM


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def ngrams(text):
    words=re.findall(r'\w+',text.casefold())
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}


def check(directory=OUTPUT,write_report=True):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('Dataset changed: '+name)
    if sha(ROOT/'src/training/a2_sentence_protocol.py')!=manifest['protocol_sha256']:
        raise ValueError('Protocol changed')
    if sha(directory/'val.jsonl')!=sha(BASE/'val.jsonl') or (directory/'val.metadata.jsonl').read_bytes()!=(BASE/'val.metadata.jsonl').read_bytes():
        raise ValueError('Frozen validation changed')
    seeds=json.loads((ROOT/'data/processed/a2_raft_pilot_v2/seed_review.json').read_text(encoding='utf-8'))
    seeds={s['id']:s for s in seeds}
    allids=set();prompts={};documents=defaultdict(set);summary={}
    contexts=defaultdict(set);addition_groups=defaultdict(list)
    for split in ('train','val','challenge'):
        records=rows(directory/(split+'.jsonl'));meta=rows(directory/(split+'.metadata.jsonl'))
        if len(records)!=len(meta) or len(records)!=manifest['counts'][split]:raise ValueError('Count mismatch')
        status=Counter();kinds=Counter();langs=Counter()
        for r,m in zip(records,meta):
            if m['id'] in allids:raise ValueError('Duplicate sample ID')
            allids.add(m['id'])
            if [x['role'] for x in r['messages']]!=['system','user','assistant'] or r['messages'][0]['content']!=SYSTEM:raise ValueError('Invalid messages')
            u=json.JSONDecoder().raw_decode(r['messages'][1]['content'])[0]
            evidence=[dict(evidence_id='I1',title=m['source']['title'],content=m['source']['content'])] if 'source' in m else m['original_evidence']
            if prepare_payload(dict(question=u['question'],evidence=evidence))!=u:raise ValueError('Source/input mismatch')
            signature=fingerprint(r['messages'][:2])
            if signature in prompts:raise ValueError('Duplicate prompt in '+split+' and '+prompts[signature])
            prompts[signature]=split
            gold=json.loads(r['messages'][2]['content'])
            resolve_selection(gold,dict(question=u['question'],evidence=evidence))
            status[gold['status']]+=1;kinds[m.get('kind','base_reviewed')]+=1
            if 'source' in m:
                if hashlib.sha256(m['source']['content'].encode()).hexdigest()!=m['context_sha256']:raise ValueError('Context checksum mismatch')
                documents[split].add(m['document_id']);addition_groups[(split,m['chunk_id'])].append((r,m))
                langs['vi' if m['document_id'].startswith('RHM') else 'en']+=1
            else:
                seed=seeds[m['family_id']];partner=seeds[m['distractor']['owner']]
                documents[split].update(v['document_id'] for v in seed['sources'].values())
                documents[split].update(v['document_id'] for v in partner['sources'].values())
            for e in evidence:contexts[split].add(e['content'])
        summary[split]=dict(count=len(records),statuses=dict(status),kinds=dict(kinds),addition_source_languages=dict(langs),source_documents=sorted(documents[split]))
    for a,b in [('train','val'),('train','challenge'),('val','challenge')]:
        if documents[a]&documents[b]:raise ValueError('Source document overlap '+a+'/'+b)
    for key,group in addition_groups.items():
        if len(group)!=7 or len({m['context_sha256'] for _,m in group})!=1:raise ValueError('Intact family changed')
        if Counter(m['kind'] for _,m in group)!=Counter(single_aspect=3,two_supported_aspects=2,intact_same_entity_missing_aspect=1,intact_partial_support=1):raise ValueError('Family task coverage mismatch')
        positives=[json.loads(r['messages'][-1]['content']) for r,m in group if m['kind']=='single_aspect']
        if len({g['claims'][0]['sentence_ids'][0] for g in positives})!=3:raise ValueError('Different aspects share target')
    old_ngrams=[ngrams(e['content']) for s in seeds.values() for e in s['sources'].values()]
    catalog=json.loads((ROOT/'data/curated/a2_hard_v3_sources.json').read_text(encoding='utf-8'))
    overlap=[]
    for source in catalog.values():
        if source['document_id'] in documents['challenge']:continue
        ns=ngrams(source['content'])
        maximum=max((len(ns&old)/len(ns|old) if ns|old else 0) for old in old_ngrams)
        if maximum>=.5:overlap.append(dict(chunk_id=source['chunk_id'],maximum_word_5gram_jaccard=round(maximum,3)))
    summary['new_train_contexts_near_old_seed_contexts']=overlap
    summary['scope']='Structure/provenance checks only; not model inference or automatic semantic grading. Challenge is new development data, not the reserved independent test.'
    summary['passed']=True
    if write_report:(directory/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
