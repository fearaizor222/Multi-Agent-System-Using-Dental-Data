"""Build a larger, source-read train pilot while freezing the v1 validation set."""
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from scripts.build_a2_raft_pilot import ROOT,SOURCE,read_rows,write_json,write_rows,sha,fingerprint
from scripts.a2_raft_v2_seeds import specifications
from scripts.train_a2 import parse_sample
from src.training.a2_raft_protocol import GroundedDraft,SYSTEM

PREVIOUS=ROOT/'data/processed/a2_raft_pilot_v1'
OUTPUT=ROOT/'data/processed/a2_raft_pilot_v2'


def make_seeds():
    original=json.loads((PREVIOUS/'seed_review.json').read_text(encoding='utf-8'))
    seeds=[dict(s) for s in original]
    rows=read_rows(SOURCE/'train.jsonl');metas=read_rows(SOURCE/'train.metadata.jsonl')
    parents={m['id']:(r,m) for r,m in zip(rows,metas)}
    for spec in specifications():
        row,meta=parents[spec['parent_id']]
        payload,_=parse_sample(row)
        source_map={e['evidence_id']:{'document_id':meta['document_id'],'chunk_id':chunk,
            'title':e['title'],'content':' '.join(e['content'].split())}
            for e,chunk in zip(payload['evidence'],meta['chunk_ids'])}
        claims=[]
        for text,eid,first,last in spec['facts']:
            content=source_map[eid]['content']
            try:
                start=content.index(first);end=content.index(last,start)+len(last)
            except ValueError as exc:
                raise ValueError(f'Source span not found: {spec["id"]} / {eid} / {first}') from exc
            claims.append(dict(text=text,source_key=eid,quote=content[start:end],quote_start=start,quote_end=end))
        used={c['source_key'] for c in claims}
        seeds.append(dict(id=spec['id'],parent_id=spec['parent_id'],parent_family=meta['family_id'],
            split='train',language=spec['language'],questions=spec['questions'],
            combined_question=spec['combined_question'],claims=claims,
            sources={k:v for k,v in source_map.items() if k in used},
            review='assistant source-read two fact/claim pairs and missing-support contrasts; not expert clinical adjudication'))
    assert len({s['id'] for s in seeds})==44
    assert sum(s['split']=='train' for s in seeds)==40
    # Distractors come from another training chapter in the same source language.
    for lang in ('en','vi'):
        group=sorted([s for s in seeds if s['split']=='train' and s['language']==lang],key=lambda s:s['id'])
        assert len(group)==20
        for i,seed in enumerate(group):
            own_docs={v['document_id'] for v in seed['sources'].values()}
            partner=next(p for p in group[i+1:]+group[:i] if not own_docs & {v['document_id'] for v in p['sources'].values()})
            seed['partner']=partner['id']
    return seeds


def build(output=OUTPUT):
    seeds=make_seeds();by_id={s['id']:s for s in seeds}
    output.mkdir(exist_ok=False)
    records=[];metadata=[]

    def excerpt(seed,index):
        fact=seed['claims'][index]
        return dict(owner=seed['id'],source_key=fact['source_key'],content=fact['quote'],
            excerpt_start=fact['quote_start'],excerpt_end=fact['quote_end'])

    for seed in seeds:
        if seed['split']!='train':continue
        partner=by_id[seed['partner']]
        oracle=[dict(owner=seed['id'],source_key=eid,content=source['content'],
            excerpt_start=0,excerpt_end=len(source['content'])) for eid,source in seed['sources'].items()]
        noise=[excerpt(partner,i) for i in range(2 if len(oracle)==1 else 1)]
        variants={'clean':oracle,'noise_before':noise+oracle,
            'noise_middle':([noise[0],*oracle,*noise[1:]] if len(oracle)==1 else [oracle[0],*noise,*oracle[1:]]),
            'noise_after':oracle+noise,'focused':[excerpt(seed,0),excerpt(seed,1)],
            'without_evidence':[excerpt(partner,0),excerpt(partner,1)]}
        questions=seed['questions']+[seed['combined_question']]

        def add(q_index,mode,contexts,supported):
            evidence=[];origins=[]
            for i,context in enumerate(contexts,1):
                eid=f'I{i}';owner=by_id[context['owner']];source=owner['sources'][context['source_key']]
                evidence.append(dict(evidence_id=eid,title=source['title'],content=context['content']))
                origins.append(dict(evidence_id=eid,owner_family=owner['id'],source_key=context['source_key'],
                    document_id=source['document_id'],chunk_id=source['chunk_id'],
                    excerpt_start=context['excerpt_start'],excerpt_end=context['excerpt_end']))
            claims=[]
            for index in supported:
                fact=seed['claims'][index]
                matches=[f'I{i}' for i,c in enumerate(contexts,1) if c['owner']==seed['id']
                    and c['source_key']==fact['source_key'] and fact['quote'] in c['content']]
                assert len(matches)==1,(seed['id'],mode,index)
                claims.append(dict(text=fact['text'],evidence_ids=matches,
                    supporting_quotes=[dict(evidence_id=matches[0],quote=fact['quote'])]))
            gold=dict(status='draft' if claims else 'insufficient_evidence',claims=claims)
            GroundedDraft.model_validate(gold)
            payload=dict(question=questions[q_index],evidence=evidence,
                allowed_evidence_ids=[e['evidence_id'] for e in evidence])
            record={'messages':[dict(role='system',content=SYSTEM),dict(role='user',
                content=json.dumps(payload,ensure_ascii=False)+'\nJSON schema:\n'+json.dumps(GroundedDraft.model_json_schema(),ensure_ascii=False)),
                dict(role='assistant',content=json.dumps(gold,ensure_ascii=False))]}
            meta=dict(id=f'raft2_{seed["id"]}_q{q_index}_{mode}',split='train',family_id=seed['id'],
                parent_id=seed['parent_id'],parent_family=seed['parent_family'],language=seed['language'],
                question_id=f'{seed["id"]}_q{q_index}',mode=mode,status=gold['status'],
                supported_fact_indices=supported,question_fact_indices=[q_index] if q_index<2 else [0,1],
                source_origins=origins,messages_sha256=fingerprint(record['messages']),
                review_origin='assistant source-read seed facts and context removals; deterministic variants')
            records.append(record);metadata.append(meta)

        for q in range(3):
            supported=[q] if q<2 else [0,1]
            for mode,contexts in variants.items():
                add(q,mode,contexts,[] if mode=='without_evidence' else supported)
            if q<2:
                remaining=[excerpt(seed,1-q),excerpt(partner,0)]
                if q:remaining.reverse()
                add(q,'missing_support_same_topic',remaining,[])
        for keep in (0,1):
            add(2,'partial_answer_first' if keep==0 else 'partial_answer_second',
                [excerpt(seed,keep),excerpt(partner,0),excerpt(partner,1)],[keep])
    paired=list(zip(records,metadata));random.Random(20261113).shuffle(paired)
    write_rows(output/'train.jsonl',[r for r,m in paired]);write_rows(output/'train.metadata.jsonl',[m for r,m in paired])
    for name in ('val.jsonl','val.metadata.jsonl','schema.json'):
        shutil.copyfile(PREVIOUS/name,output/name)
    write_json(output/'seed_review.json',seeds)
    manifest=dict(version='a2-raft-pilot-v2-20261112',source_dataset='a2_drafting_v4',
        source_files_sha256={name:sha(SOURCE/name) for name in ('manifest.json','train.jsonl','train.metadata.jsonl','val.jsonl','val.metadata.jsonl','test.jsonl','test.metadata.jsonl')},
        frozen_validation={'directory':PREVIOUS.name,'files':{name:sha(PREVIOUS/name) for name in ('val.jsonl','val.metadata.jsonl')}},
        counts={'train':len(records),'val':72},families={'train':40,'val':4},
        distinct_questions={'train':120,'val':12},
        modes={'train':dict(Counter(m['mode'] for m in metadata))},
        scope='Source-read task-specific RAFT pilot; 120 train questions expanded into correlated context variants. No CoT, no external clinical adjudication.',
        split_policy='New seed facts and all distractors use v4 training chapters only. V1 validation bytes frozen; test data not used to construct examples.',
        output_contract='unchanged a2_grounded_quotes_v1; same system prompt and schema as v1',
        training_executed=False,inference_executed=False,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
