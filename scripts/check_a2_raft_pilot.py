"""Validate pilot provenance, source quotes, contrasts and Qwen3 token budgets."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.build_a2_raft_pilot import (
    ROOT, SOURCE, OUTPUT, SYSTEM, GroundedDraft, fingerprint, read_rows, sha, write_json,
)
from scripts.train_a2 import parse_json


def check(directory, tokenizer=None, *, write_reports=True):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name, checksum in manifest['source_files_sha256'].items():
        if sha(SOURCE/name) != checksum:
            raise ValueError('Parent dataset changed: '+name)
    for name, checksum in manifest['files_sha256'].items():
        if sha(directory/name) != checksum:
            raise ValueError('Pilot file changed: '+name)
    seeds = {seed['id']:seed for seed in json.loads((directory/'seed_review.json').read_text(encoding='utf-8'))}
    parent_sources = {}
    # Read train/validation only. Test text is not used in this pilot.
    for split in ('train','val'):
        for row, meta in zip(read_rows(SOURCE/(split+'.jsonl')),read_rows(SOURCE/(split+'.metadata.jsonl'))):
            payload = parse_json(row['messages'][1]['content'].split('\nJSON schema:\n')[0])
            for evidence, chunk_id in zip(payload['evidence'],meta['chunk_ids']):
                parent_sources[split,meta['id'],evidence['evidence_id']] = (
                    chunk_id, meta['document_id'], ' '.join(evidence['content'].split()))
    for seed in seeds.values():
        for key, source in seed['sources'].items():
            actual = parent_sources[seed['split'],seed['parent_id'],key]
            if actual != (source['chunk_id'],source['document_id'],source['content']):
                raise ValueError('Seed not from specified parent source')
        for fact in seed['claims']:
            source = seed['sources'][fact['source_key']]['content']
            if source[fact['quote_start']:fact['quote_end']] != fact['quote']:
                raise ValueError('Seed quote offsets do not match source')
    seen_ids, seen_inputs = set(), set()
    boundaries, summaries, examples = {}, {}, []
    for split in ('train','val'):
        records = read_rows(directory/(split+'.jsonl'))
        metas = read_rows(directory/(split+'.metadata.jsonl'))
        if len(records) != len(metas) or len(records) != manifest['counts'][split]:
            raise ValueError('Count mismatch')
        groups = defaultdict(list)
        documents, chunks, families, lengths = set(), set(), set(), []
        for record, meta in zip(records,metas):
            if meta['id'] in seen_ids or meta['split'] != split:
                raise ValueError('Duplicate ID or incorrect split')
            seen_ids.add(meta['id'])
            if [m['role'] for m in record['messages']] != ['system','user','assistant']:
                raise ValueError('Incorrect chat roles')
            if record['messages'][0]['content'] != SYSTEM or fingerprint(record['messages']) != meta['messages_sha256']:
                raise ValueError('Prompt or answer changed after review')
            body, schema = record['messages'][1]['content'].split('\nJSON schema:\n')
            if parse_json(schema) != GroundedDraft.model_json_schema():
                raise ValueError('Output schema mismatch')
            payload = parse_json(body)
            if set(payload) != {'question','evidence','allowed_evidence_ids'}:
                raise ValueError('Metadata must not be visible to the answering model')
            if fingerprint(payload) in seen_inputs:
                raise ValueError('Duplicate input')
            seen_inputs.add(fingerprint(payload))
            expected_ids = [f'I{i}' for i in range(1,len(payload['evidence'])+1)]
            if not 1 <= len(expected_ids) <= 3 or payload['allowed_evidence_ids'] != expected_ids:
                raise ValueError('Incorrect supplied evidence IDs')
            if [e['evidence_id'] for e in payload['evidence']] != expected_ids:
                raise ValueError('Evidence ID order mismatch')
            if len(meta['source_origins']) != len(expected_ids):
                raise ValueError('Missing source provenance')
            for evidence, origin in zip(payload['evidence'],meta['source_origins']):
                owner = seeds[origin['owner_family']]
                source = owner['sources'][origin['source_key']]
                if (owner['split'] != split or source['document_id'] != origin['document_id']
                        or source['chunk_id'] != origin['chunk_id']
                        or source['title'] != evidence['title']
                        or origin['evidence_id'] != evidence['evidence_id']
                        or source['content'][origin['excerpt_start']:origin['excerpt_end']] != evidence['content']):
                    raise ValueError('Source excerpt provenance mismatch')
                documents.add(origin['document_id']); chunks.add(origin['chunk_id'])
            gold = GroundedDraft.model_validate(parse_json(record['messages'][2]['content']))
            seed = seeds[meta['family_id']]
            if gold.status != meta['status'] or len(gold.claims) != len(meta['supported_fact_indices']):
                raise ValueError('Status/claim metadata mismatch')
            sources = {e['evidence_id']:e['content'] for e in payload['evidence']}
            origins = {o['evidence_id']:o for o in meta['source_origins']}
            for claim, fact_index in zip(gold.claims,meta['supported_fact_indices']):
                fact = seed['claims'][fact_index]
                if claim.text != fact['text']:
                    raise ValueError('Answer is different from source-read seed')
                for quote in claim.supporting_quotes:
                    if quote.quote != fact['quote'] or quote.quote not in sources[quote.evidence_id]:
                        raise ValueError('Claim lacks its exact supporting quote')
                    origin = origins[quote.evidence_id]
                    if origin['owner_family'] != seed['id'] or origin['source_key'] != fact['source_key']:
                        raise ValueError('Claim cites distractor instead of supporting source')
            if meta['mode'] == 'without_evidence':
                if gold.claims or any(o['owner_family']==seed['id'] for o in origins.values()):
                    raise ValueError('Removed-evidence case still contains an oracle')
            if meta['mode'] == 'missing_support_same_topic':
                requested = meta['question_fact_indices']
                if gold.claims or len(requested)!=1 or not any(o['owner_family']==seed['id'] for o in origins.values()):
                    raise ValueError('Invalid same-topic missing-support contrast')
                if any(seed['claims'][requested[0]]['quote'] in e['content'] for e in payload['evidence']):
                    raise ValueError('Same-topic negative still contains the requested proof')
            if meta['mode'] == 'partial_answer':
                if meta['supported_fact_indices'] != [1] or meta['question_fact_indices'] != [0,1]:
                    raise ValueError('Partial-answer contrast mismatch')
                # This cannot prove semantic independence; all 12 such pairs were source-read.
                if any(seed['claims'][0]['quote'] in e['content'] for e in payload['evidence']):
                    raise ValueError('Partial case still contains the other supporting quote')
            families.add(meta['family_id'])
            groups[meta['question_id']].append((meta,payload,gold))
            if tokenizer:
                length = len(tokenizer.apply_chat_template(record['messages'],tokenize=True,enable_thinking=False))
                if length > 3072:
                    raise ValueError('Pilot exceeds training token budget: '+meta['id'])
                lengths.append(length)
            if meta['mode'] in ('noise_middle','without_evidence','partial_answer','missing_support_same_topic') and seed['id'] in ('sealants','splint_duration'):
                examples.append({'id':meta['id'],'input':payload,'expected':gold.model_dump()})
        for group in groups.values():
            clean = next((p for m,p,g in group if m['mode']=='clean'), None)
            modes = {m['mode'] for m,p,g in group}
            if not {'clean','noise_before','noise_middle','noise_after','without_evidence'} <= modes:
                raise ValueError('Question lacks planned paired variants')
            if any(p['question']!=clean['question'] for m,p,g in group):
                raise ValueError('Question changes across paired evidence cases')
        # Three distinct questions per family share exactly the same positive context.
        for family in families:
            clean = [p for group in groups.values() for m,p,g in group if m['family_id']==family and m['mode']=='clean']
            if len(clean)!=3 or len({fingerprint(p['evidence']) for p in clean})!=1 or len({p['question'] for p in clean})!=3:
                raise ValueError('Missing same-context/different-question group')
        boundaries[split] = {'documents':documents,'chunks':chunks,'families':families}
        summaries[split] = {'count':len(records),'families':len(families),'distinct_questions':len(groups),
            'languages':dict(Counter(m['language'] for m in metas)),
            'statuses':dict(Counter(m['status'] for m in metas)),
            'modes':dict(Counter(m['mode'] for m in metas)),
            'max_tokens':max(lengths) if lengths else None,
            'citation_positions':dict(Counter(q['evidence_id'] for r in records for c in parse_json(r['messages'][2]['content'])['claims'] for q in c['supporting_quotes']))}
    for key in ('documents','chunks','families'):
        if boundaries['train'][key] & boundaries['val'][key]:
            raise ValueError('Cross-split overlap: '+key)
    result = {'status':'passed','splits':summaries,'cross_split_document_chunk_family_overlap':0,
        'exact_quotes_and_source_offsets_checked':True,'paired_question_context_variants_checked':True,
        'semantic_review':'Assistant reviewed 24 seed fact/claim pairs and paired distractor excerpts. Variants checked deterministically; no independent clinical review.',
        'token_limit':3072,'tokenizer_checked':tokenizer is not None,
        'test_used':False,'training_executed':False,'inference_executed':False}
    if write_reports:
        write_json(directory/'verification.json',result)
        write_json(directory/'examples.json',examples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=OUTPUT)
    parser.add_argument('--tokenizer',action='store_true')
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'data/models/Qwen--Qwen3-1.7B/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',local_files_only=True)
    print(json.dumps(check(args.directory,tokenizer),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
