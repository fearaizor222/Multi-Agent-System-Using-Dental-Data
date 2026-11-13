"""Build review-gated full-context candidates, not an automatically trainable dataset."""
import json
import hashlib
import random
from collections import Counter
from pathlib import Path
from src.training.a2_sentence_protocol import source_units, selection_from_gold, make_messages
from scripts.a2_sentence_review_policy import allowed_keep, additional_removal

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/a2_raft_pilot_v2/seed_review.json'
OUT=ROOT/'data/processed/a2_sentence_selection_v2_review06'


def build(output=OUT):
    if output.exists():
        raise ValueError('Refusing to overwrite')
    seeds=json.loads(SOURCE.read_text(encoding='utf-8'))
    candidates=[]; excluded=[]
    for seed in seeds:
        # Reject poor positive supervision before making any variants.
        original={k:dict(evidence_id=k,title=v['title'],content=v['content']) for k,v in seed['sources'].items()}
        payload=dict(question=seed['combined_question'],evidence=list(original.values()))
        gold=dict(status='draft',claims=[dict(text=f['text'],evidence_ids=[f['source_key']],
            supporting_quotes=[dict(evidence_id=f['source_key'],quote=f['quote'])]) for f in seed['claims']])
        try:
            selected=selection_from_gold(gold,payload)
            units={f'{eid}:S{i}':u for eid,e in original.items() for i,u in enumerate(source_units(e['content']),1)}
            issues=[]
            if selected.claims[0].sentence_ids==selected.claims[1].sentence_ids:
                issues.append('two_facts_share_selection')
            for c,f in zip(selected.claims,seed['claims']):
                length=sum(len(units[s]['text']) for s in c.sentence_ids)
                if length>max(300,1.8*len(f['quote'])):
                    issues.append('selected_span_too_broad')
            if issues:
                excluded.append(dict(family=seed['id'],reasons=sorted(set(issues))));continue
        except ValueError as exc:
            excluded.append(dict(family=seed['id'],reasons=[str(exc)]));continue
        partners=[s for s in seeds if s['split']==seed['split'] and s['language']==seed['language'] and s['id']!=seed['id']]
        documents={s['document_id'] for s in seed['sources'].values()}
        partners.sort(key=lambda s:(not bool(documents & {v['document_id'] for v in s['sources'].values()}),s['id']))
        partner=partners[0]
        for kept in ([0,1],[0],[1],[]):
            if not allowed_keep(seed['id'], kept):
                continue
            # Source-read exceptions: the following paragraph still discusses
            # smoking cessation; extensor fragment needs the two-bone antecedent.
            if seed['id']=='global_care' and 0 not in kept:
                continue
            if seed['id']=='flexors_extensors' and kept==[1]:
                continue
            # Remove whole units, not just known quote strings. Keep surrounding
            # full context. Alternative supporting paraphrases require review.
            sources=[]; provenance=[]
            for eid,e in original.items():
                removed={sid for i,c in enumerate(selected.claims) if i not in kept for sid in c.sentence_ids if sid.startswith(eid+':')}
                retained=[u for i,u in enumerate(source_units(e['content']),1) if f'{eid}:S{i}' not in removed
                          and not additional_removal(seed['id'], kept, u['text'])]
                content=' '.join(u['text'] for u in retained)
                if content:
                    sources.append(dict(title=e['title'],content=content,owner=seed['id'],key=eid))
                    provenance.append(dict(owner=seed['id'],source_key=eid,retained_spans=[[u['start'],u['end']] for u in retained]))
            # Full distractor, never an extracted single-answer sentence.
            key,next_source=next(iter(partner['sources'].items()))
            sources.append(dict(title=next_source['title'],content=next_source['content'],owner=partner['id'],key=key))
            for position in range(min(len(sources),3)):
                contexts=sources[position:]+sources[:position]
                evidence=[dict(evidence_id=f'I{i}',title=e['title'],content=e['content']) for i,e in enumerate(contexts,1)]
                remap={e['key']:f'I{i}' for i,e in enumerate(contexts,1) if e['owner']==seed['id']}
                for q in range(3):
                    supported=[i for i in kept if q==2 or i==q]
                    asked=[q] if q<2 else [0,1]
                    if any(seed['claims'][i]['quote'] in e['content']
                           for i in asked if i not in supported for e in evidence):
                        continue  # known answer still present: never label absent
                    question=(seed['questions']+[seed['combined_question']])[q]
                    p=dict(question=question,evidence=evidence,allowed_evidence_ids=[e['evidence_id'] for e in evidence])
                    claims=[]
                    for i in supported:
                        f=seed['claims'][i];eid=remap[f['source_key']]
                        claims.append(dict(text=f['text'],evidence_ids=[eid],supporting_quotes=[dict(evidence_id=eid,quote=f['quote'])]))
                    g=dict(status='draft' if claims else 'insufficient_evidence',claims=claims)
                    try: target=selection_from_gold(g,p).model_dump()
                    except ValueError as exc:
                        excluded.append(dict(family=seed['id'],reasons=['variant_mapping: '+str(exc)]));continue
                    ident=f'{seed["id"]}_q{q}_keep'+''.join(map(str,kept))+f'_rotate{position}'
                    candidates.append(dict(id=ident,split=seed['split'],family_id=seed['id'],
                        language=seed['language'],kept_fact_indices=kept,question_fact_indices=[q] if q<2 else [0,1],
                        messages=make_messages(p)+[dict(role='assistant',content=json.dumps(target,ensure_ascii=False))],
                        proposed_status=g['status'],requires_source_review=True,
                        review_reasons=['check_remaining_context_and_distractor_for_alternative_support','check_claim_and_selected_units'],
                        source_provenance=provenance,distractor=dict(owner=partner['id'],source_key=key),
                        original_evidence=evidence))
    # Remove identical prompts produced by rotation or unchanged kept sets.
    unique={}
    for c in candidates:
        key=json.dumps(c['messages'][:2],ensure_ascii=False,sort_keys=True)
        if key in unique and unique[key]['messages'][-1]!=c['messages'][-1]:
            raise ValueError('Conflicting labels')
        unique.setdefault(key,c)
    candidates=list(unique.values());random.Random(42).shuffle(candidates)
    output.mkdir(parents=True)
    (output/'candidates.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in candidates),encoding='utf-8')
    summary=dict(version='a2_sentence_selection_v2_review_candidates',training_ready=False,
        candidates=len(candidates),counts=dict(Counter((c['split']+'/'+c['proposed_status']) for c in candidates)),
        excluded=excluded,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'src/training/a2_sentence_protocol.py').read_bytes()).hexdigest(),
        limitations=['Not manually adjudicated; proposed missing-evidence labels may have alternative support.',
                    'Validation is a new development version, not the old 72-sample benchmark.',
                    'No new independent facts; same-topic full distractors still need curation.'],
        training_executed=False,test_used=False)
    (output/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
