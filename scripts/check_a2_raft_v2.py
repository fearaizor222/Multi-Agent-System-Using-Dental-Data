"""Check v2 sources, two-sided partial answers, frozen holdout and token budgets."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from scripts.build_a2_raft_pilot import ROOT,SOURCE,read_rows,write_json,sha,fingerprint
from scripts.build_a2_raft_v2 import OUTPUT,PREVIOUS
from scripts.train_a2 import parse_sample,encode_sample
from src.training.a2_raft_protocol import SYSTEM


def check(directory=OUTPUT,tokenizer=None,*,write_reports=True):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['version'].startswith('a2-raft-pilot-v2-'):raise ValueError('Not a v2 pilot')
    for name,digest in manifest['source_files_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('Parent dataset changed: '+name)
    for name,digest in manifest['files_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('Pilot file changed: '+name)
    for name,digest in manifest['frozen_validation']['files'].items():
        if sha(PREVIOUS/name)!=digest or sha(directory/name)!=digest:
            raise ValueError('Validation must remain byte-identical to v1: '+name)
    seeds={s['id']:s for s in json.loads((directory/'seed_review.json').read_text(encoding='utf-8'))}
    parents={};parent_docs={}
    for split in ('train','val'):
        for row,meta in zip(read_rows(SOURCE/(split+'.jsonl')),read_rows(SOURCE/(split+'.metadata.jsonl'))):
            payload,_=parse_sample(row)
            parent_docs[split,meta['id']]=meta['document_id']
            for e,chunk in zip(payload['evidence'],meta['chunk_ids']):
                parents[split,meta['id'],e['evidence_id']]=(chunk,meta['document_id'],e['title'],' '.join(e['content'].split()))
    for seed in seeds.values():
        for eid,source in seed['sources'].items():
            if parents.get((seed['split'],seed['parent_id'],eid))!=(source['chunk_id'],source['document_id'],source['title'],source['content']):
                raise ValueError('Seed source is outside its parent split')
        for fact in seed['claims']:
            if seed['sources'][fact['source_key']]['content'][fact['quote_start']:fact['quote_end']]!=fact['quote']:
                raise ValueError('Seed source offset mismatch')
    test_documents={m['document_id'] for m in read_rows(SOURCE/'test.metadata.jsonl')}
    boundaries={};summaries={};seen_ids=set();seen_inputs=set();examples=[]
    for split in ('train','val'):
        rows=read_rows(directory/(split+'.jsonl'));metas=read_rows(directory/(split+'.metadata.jsonl'))
        if len(rows)!=len(metas) or len(rows)!=manifest['counts'][split]:raise ValueError('Count mismatch')
        groups=defaultdict(dict);docs=set();chunks=set();lengths=[];target_lengths=[];positions=Counter()
        for row,meta in zip(rows,metas):
            if meta['id'] in seen_ids or meta['split']!=split:raise ValueError('Duplicate ID or wrong split')
            seen_ids.add(meta['id'])
            if row['messages'][0]['content']!=SYSTEM or fingerprint(row['messages'])!=meta['messages_sha256']:
                raise ValueError('Changed prompt, answer or review fingerprint')
            payload,gold=parse_sample(row)
            if set(payload)!={'question','evidence','allowed_evidence_ids'}:raise ValueError('Metadata leaked into model input')
            signature=fingerprint(payload)
            if signature in seen_inputs:raise ValueError('Duplicate model input')
            seen_inputs.add(signature)
            seed=seeds[meta['family_id']]
            if seed['split']!=split or meta['parent_id']!=seed['parent_id']:raise ValueError('Seed split mismatch')
            indices=meta['question_fact_indices']
            if indices not in ([0],[1],[0,1]):raise ValueError('Unknown question facts')
            q_index=indices[0] if len(indices)==1 else 2
            if meta['question_id']!=f'{seed["id"]}_q{q_index}':raise ValueError('Question ID mismatch')
            if payload['question']!=(seed['questions']+[seed['combined_question']])[q_index]:raise ValueError('Question changed across contrasts')
            mode=meta['mode'];supported=meta['supported_fact_indices']
            if mode in ('clean','noise_before','noise_middle','noise_after','focused'):required=indices
            elif mode in ('without_evidence','missing_support_same_topic'):required=[]
            elif mode in ('partial_answer','partial_answer_second'):required=[1]
            elif mode=='partial_answer_first':required=[0]
            else:raise ValueError('Unknown mode: '+mode)
            if mode.startswith('partial_answer') and indices!=[0,1]:raise ValueError('Partial must retain a two-part question')
            if mode=='missing_support_same_topic' and len(indices)!=1:raise ValueError('Hard negative must ask one fact')
            if supported!=required or len(gold['claims'])!=len(required):raise ValueError('Incomplete or incorrect answer coverage')
            if gold['status']!=meta['status'] or gold['status']!=('draft' if required else 'insufficient_evidence'):
                raise ValueError('Wrong expected status')
            eids=[f'I{i}' for i in range(1,len(payload['evidence'])+1)]
            if [e['evidence_id'] for e in payload['evidence']]!=eids or payload['allowed_evidence_ids']!=eids:
                raise ValueError('Evidence position mismatch')
            if len(meta['source_origins'])!=len(eids):raise ValueError('Missing source provenance')
            for evidence,origin in zip(payload['evidence'],meta['source_origins']):
                owner=seeds[origin['owner_family']];source=owner['sources'][origin['source_key']]
                if (owner['split']!=split or source['document_id']!=origin['document_id'] or source['chunk_id']!=origin['chunk_id']
                    or source['title']!=evidence['title'] or origin['evidence_id']!=evidence['evidence_id']
                    or source['content'][origin['excerpt_start']:origin['excerpt_end']]!=evidence['content']):
                    raise ValueError('Supplied source excerpt not from declared split/offset')
                docs.add(origin['document_id']);chunks.add(origin['chunk_id'])
            for claim,index in zip(gold['claims'],required):
                fact=seed['claims'][index]
                matches=[e['evidence_id'] for e,o in zip(payload['evidence'],meta['source_origins'])
                    if o['owner_family']==seed['id'] and o['source_key']==fact['source_key'] and fact['quote'] in e['content']]
                if (claim['text']!=fact['text'] or len(matches)!=1 or claim['evidence_ids']!=matches
                    or claim['supporting_quotes']!=[{'evidence_id':matches[0],'quote':fact['quote']}]):
                    raise ValueError('Answer not bound to its source-read fact and exact quote')
                positions.update(matches)
            if mode=='without_evidence' and any(o['owner_family']==seed['id'] for o in meta['source_origins']):
                raise ValueError('Oracle remains in removed-evidence case')
            if mode=='missing_support_same_topic' or mode.startswith('partial_answer'):
                removed=indices if mode=='missing_support_same_topic' else [i for i in indices if i not in required]
                if not any(o['owner_family']==seed['id'] for o in meta['source_origins']):raise ValueError('Contrast lost same-topic source')
                if any(seed['claims'][i]['quote'] in e['content'] for i in removed for e in payload['evidence']):
                    raise ValueError('Removed proof remains visible in negative/partial contrast')
            group=groups[meta['question_id']]
            if mode in group:raise ValueError('Duplicate question/mode')
            group[mode]=(payload,gold)
            if tokenizer:
                encoded=encode_sample(tokenizer,row)
                lengths.append(len(encoded['input_ids']))
                target_lengths.append(sum(x!=-100 for x in encoded['labels']))
                if lengths[-1]>3072 or target_lengths[-1]>512:raise ValueError('Token budget exceeded')
            if seed['id'] in ('hypotheses','soft_hard_end_feel') and mode in ('focused','missing_support_same_topic','partial_answer_first','partial_answer_second'):
                examples.append({'id':meta['id'],'input':payload,'expected':gold})
        if split=='train':
            for qid,group in groups.items():
                required={'clean','noise_before','noise_middle','noise_after','focused','without_evidence'}
                required|={'partial_answer_first','partial_answer_second'} if qid.endswith('_q2') else {'missing_support_same_topic'}
                if set(group)!=required:raise ValueError('Missing contrast for '+qid)
            for family in {m['family_id'] for m in metas}:
                clean=[groups[f'{family}_q{i}']['clean'][0]['evidence'] for i in range(3)]
                if len({fingerprint(e) for e in clean})!=1:raise ValueError('Same-source questions no longer share clean context')
        if docs & test_documents:raise ValueError('Test chapter used in pilot')
        boundaries[split]={'docs':docs,'chunks':chunks}
        summaries[split]=dict(count=len(rows),families=len({m['family_id'] for m in metas}),
            distinct_questions=len(groups),documents=len(docs),source_chunks=len(chunks),
            languages=dict(Counter(m['language'] for m in metas)),statuses=dict(Counter(m['status'] for m in metas)),
            modes=dict(Counter(m['mode'] for m in metas)),citation_positions=dict(positions),
            max_tokens=max(lengths) if lengths else None,max_assistant_tokens=max(target_lengths) if target_lengths else None)
    if any(boundaries['train'][k] & boundaries['val'][k] for k in ('docs','chunks')):raise ValueError('Cross-split source overlap')
    result=dict(status='passed',splits=summaries,validation_byte_identical_to_v1=True,test_chapter_overlap=0,
        train_validation_chapter_chunk_overlap=0,exact_source_offsets_and_quotes_checked=True,
        question_and_full_two_claim_coverage_checked=True,both_partial_directions_checked=True,
        tokenizer_checked=tokenizer is not None,training_executed=False,inference_executed=False,
        review_scope='Assistant source-reading, 80 train facts across 40 groups; no independent clinical review. Validation is the unchanged 72-case diagnostic set already examined, not a fresh test.')
    if write_reports:
        write_json(directory/'verification.json',result);write_json(directory/'examples.json',examples)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tokenizer',action='store_true')
    args=parser.parse_args();tokenizer=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',local_files_only=True)
    print(json.dumps(check(tokenizer=tokenizer),ensure_ascii=False,indent=2))
