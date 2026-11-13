"""Source-read intra-chunk additions, unchanged contexts, separate challenge set."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from scripts.a2_hard_specs import specs,evaluation_specs
from scripts.train_a2 import parse_sample
from src.training.a2_sentence_protocol import make_messages,unit_index,resolve_selection,SentenceDraft

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/a2_sentence_selection_v2_reviewed'
OUTPUT=ROOT/'data/processed/a2_sentence_selection_v3_hard'


def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def make_family(spec, source):
    eid='I1';payload=dict(question='',evidence=[dict(evidence_id=eid,title=source['title'],content=source['content'])],allowed_evidence_ids=[eid])
    index=unit_index(payload);facts=[]
    for f in spec['facts']:
        matches=[sid for sid,(_,u) in index.items() if f['needle'] in u['text']]
        if len(matches)!=1:raise ValueError(f'{spec["pool_index"]}: ambiguous/missing {f["needle"]}: {matches}')
        sid=matches[0]
        if len(index[sid][1]['text'])>600:raise ValueError(f'{spec["pool_index"]}: target too broad {sid}')
        facts.append(dict(text=f['text'],evidence_ids=[eid],sentence_ids=[sid]))
    if len({f['sentence_ids'][0] for f in facts})!=3:raise ValueError(f'{spec["pool_index"]}: facts share target unit')
    tasks=[(f['question'],[i],'single_aspect') for i,f in enumerate(spec['facts'])]
    for a,b in [(0,1),(1,2)]:
        tasks.append((spec['facts'][a]['question']+' Đồng thời, '+spec['facts'][b]['question'],[a,b],'two_supported_aspects'))
    tasks.append((spec['negative_question'],[],'intact_same_entity_missing_aspect'))
    tasks.append((spec['facts'][0]['question']+' Đồng thời, '+spec['negative_question'],[0],'intact_partial_support'))
    result=[]
    for n,(question,supported,kind) in enumerate(tasks):
        p={**payload,'question':question}
        gold=dict(status='draft' if supported else 'insufficient_evidence',claims=[facts[i] for i in supported])
        SentenceDraft.model_validate(gold);resolve_selection(gold,p)
        result.append(dict(messages=make_messages(p)+[dict(role='assistant',content=json.dumps(gold,ensure_ascii=False))],
            metadata=dict(id=f'hard_{source["chunk_id"][:20]}_{n}',family_id=source['chunk_id'],kind=kind,
                document_id=source['document_id'],chunk_id=source['chunk_id'],parent_id=source['parent_id'],
                context_sha256=hashlib.sha256(source['content'].encode()).hexdigest(),
                supported_fact_indices=supported,status=gold['status'],source=source,
                source_review='assistant read full chunk, three positive spans, missing-aspect question; not clinical expert',
                negative_review=spec['negative_review'],source_intact=True)))
    return result


def build(output=OUTPUT):
    if output.exists():raise ValueError('Refusing to overwrite')
    catalog=ROOT/'data/curated/a2_hard_v3_sources.json'
    pool=json.loads(catalog.read_text(encoding='utf-8'))
    base_manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in base_manifest['files_sha256'].items():
        if sha(BASE/name)!=digest:raise ValueError('Reviewed base changed: '+name)
    # Rebind every selected source to actual v4 training evidence. A forged pool
    # cannot manufacture source passages or move old holdout evidence to train.
    originals={}
    parent=ROOT/'data/processed/a2_drafting_v4'
    for record,meta in zip(rows(parent/'train.jsonl'),rows(parent/'train.metadata.jsonl')):
        p,_=parse_sample(record)
        for e,k in zip(p['evidence'],meta['chunk_ids']):
            originals.setdefault(k,(meta['document_id'],' '.join(e['content'].split())))
    old_seeds=json.loads((ROOT/'data/processed/a2_raft_pilot_v2/seed_review.json').read_text(encoding='utf-8'))
    used_chunks={v['chunk_id'] for s in old_seeds for v in s['sources'].values()}
    challenge_docs={'OHA-CH01','DPH-CH23'}
    old_used_docs={v['document_id'] for s in old_seeds for v in s['sources'].values()}
    assert not challenge_docs & old_used_docs
    records={};reviews=[]
    for split,definitions in [('train',specs()),('challenge',evaluation_specs())]:
        result=[]
        for spec in definitions:
            source=pool[str(spec['pool_index'])]
            if originals.get(source['chunk_id'])!=(source['document_id'],source['content']):raise ValueError('Source differs from original training corpus')
            assert source['chunk_id'] not in used_chunks
            assert (source['document_id'] in challenge_docs)==(split=='challenge')
            family=make_family(spec,source);result+=family
            reviews.append(dict(split=split,**spec,source=source))
        records[split]=result
    train_chunks={r['metadata']['chunk_id'] for r in records['train']}
    challenge_chunks={r['metadata']['chunk_id'] for r in records['challenge']}
    assert len(train_chunks)==40 and len(challenge_chunks)==8 and not train_chunks & challenge_chunks
    # No changes to frozen baseline validation. New challenge has its own name.
    output.mkdir(parents=True)
    for split,result in records.items():
        (output/(split+'.additions.jsonl')).write_text(''.join(json.dumps({'messages':r['messages']},ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
        (output/(split+'.additions.metadata.jsonl')).write_text(''.join(json.dumps(r['metadata'],ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    for suffix in ('.jsonl','.metadata.jsonl'):
        (output/('challenge'+suffix)).write_bytes((output/('challenge.additions'+suffix)).read_bytes())
    for name in ('val.jsonl','val.metadata.jsonl'):(output/name).write_bytes((BASE/name).read_bytes())
    for suffix in ('.jsonl','.metadata.jsonl'):
        (output/('train'+suffix)).write_bytes((BASE/('train'+suffix)).read_bytes()+(output/('train.additions'+suffix)).read_bytes())
    write(output/'source_review.json',reviews)
    manifest=dict(version='a2_sentence_selection_v3_hard',counts=dict(train=774+len(records['train']),val=84,challenge=len(records['challenge'])),
        new_train_source_families=40,new_challenge_chunks=8,challenge_document_groups=2,
        addition_statuses={s:dict(Counter(r['metadata']['status'] for r in rr)) for s,rr in records.items()},
        statuses={s:dict(Counter(json.loads(r['messages'][-1]['content'])['status'] for r in rows(output/(s+'.jsonl')))) for s in ('train','val','challenge')},
        validation_frozen_sha256=sha(BASE/'val.jsonl'),base_manifest_sha256=sha(BASE/'manifest.json'),
        protocol_sha256=sha(ROOT/'src/training/a2_sentence_protocol.py'),specs_sha256=sha(ROOT/'scripts/a2_hard_specs.py'),catalog_sha256=sha(catalog),
        context_policy='All additions retain complete original context, whitespace-normalized only. No answer-sentence removal, no added external text.',
        challenge_scope='56 new development challenge records from 8 overlapping chunks / 2 source-held-out documents; English sources only; not 56 independent questions or a clinical test.',
        historical_exposure='Challenge sources originate in the historical v4 training corpus. Earlier adapters may have encountered them; this is not a blind test of all prior training.',
        review='Assistant source-reading, not clinical expert. Questions may still have lexical cues; no guarantee of model performance.',
        source_review_complete=True,data_ready_for_pilot=True,runner_compatible=False,training_ready=False,training_executed=False,test_examples_used=False,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    write(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
