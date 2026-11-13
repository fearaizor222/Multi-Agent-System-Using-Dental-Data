"""Read-only independent audit of updated A2 data; reports only under outputs."""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from scripts.train_a2 import parse_sample
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/a2_drafting_v2'
OUT=ROOT/'outputs/a2_data_audit/20261110_updated'
READ=lambda p:[json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
norm=lambda s:' '.join(unicodedata.normalize('NFC',s).casefold().split())
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files={p.name:hash_file(p) for p in DATA.glob('*.jsonl')}
    catalog={c['chunk_id']:c for c in READ(ROOT/'data/raw/internal_imports/2026-11-02/chunks.jsonl')}
    report={'sha256':files,'splits':{},'errors':[],'overlap':{},'changed_from_trained_v2':{},'review_provenance':{}}
    flat=[]; sets={}; questionmap=defaultdict(list)
    for split in ('train','val','test'):
        rows=READ(DATA/(split+'.jsonl'));metas=READ(DATA/(split+'.metadata.jsonl'))
        assert len(rows)==len(metas)
        state={k:set() for k in ('ids','documents','chunks','families','inputs','contents')}
        for row,meta in zip(rows,metas):
            id=meta['id'];payload,gold=parse_sample(row)
            assert meta['question']==payload['question'] and meta['status']==gold['status'] and meta['split']==split
            assert row['messages'][0]['content']==DRAFT_SYSTEM+(CONVERSATION_POLICY if 'conversation' in payload else '')
            state['ids'].add(id);state['documents'].add(meta['document_id']);state['chunks'].update(meta['chunk_ids'])
            state['families'].add(meta.get('family_id',id));state['inputs'].add(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            questionmap[norm(payload['question'])].append({'id':id,'split':split,'family':meta.get('family_id',id)})
            for ev,chunk_id in zip(payload['evidence'],meta['chunk_ids']):
                source=catalog[chunk_id]
                if source['document_id']!=meta['document_id'] or norm(source['content'])!=norm(ev['content']):
                    report['errors'].append({'id':id,'kind':'source_mismatch','chunk_id':chunk_id})
                state['contents'].add(norm(ev['content']))
            evidence={e['evidence_id']:e['content'] for e in payload['evidence']}
            for claim,proofclaim in zip(gold['claims'],meta['proofs']):
                if claim!={k:proofclaim[k] for k in ('text','evidence_ids')}:
                    report['errors'].append({'id':id,'kind':'proof_claim_mismatch'})
                for proof in proofclaim['proofs']:
                    if norm(proof['quote']) not in norm(evidence[proof['evidence_id']]):
                        report['errors'].append({'id':id,'kind':'non_source_quote'})
                nums=lambda text:set(re.findall(r'\d+(?:[.,]\d+)?',text))
                if not nums(claim['text']) <= nums(' '.join(p['quote'] for p in proofclaim['proofs'])):
                    report['errors'].append({'id':id,'kind':'numeric_literal_not_in_proof'})
            flat.append({'split':split,'row':row,'meta':meta,'payload':payload,'gold':gold})
        sets[split]=state
        subset=[r for r in flat if r['split']==split]
        report['splits'][split]={'count':len(rows),'unique_ids':len(state['ids']),'unique_inputs':len(state['inputs']),
            'documents':len(state['documents']),'unique_chunks':len(state['chunks']),
            'labels':dict(Counter(m['status'] for m in metas)),'modes':dict(Counter(m['mode'] for m in metas)),
            'requested_modes':dict(Counter(m.get('requested_mode') for m in metas)),
            'source_languages':dict(Counter(catalog[m['chunk_ids'][0]]['language'] for m in metas)),
            'has_history':sum('conversation' in r['payload'] for r in subset),
            'claims_per_draft':dict(Counter(len(r['gold']['claims']) for r in subset if r['gold']['status']=='draft')),
            'actually_uses_multiple_citations':sum(len({k for c in r['gold']['claims'] for k in c['evidence_ids']})>=2 for r in subset),
            'documents_by_count':dict(Counter(m['document_id'] for m in metas)),
            'review_status':dict(Counter(m['review_status'] for m in metas))}
    for a,b in combinations(sets,2):report['overlap'][a+'/'+b]={k:len(sets[a][k]&sets[b][k]) for k in sets[a]}
    report['duplicate_questions']=[{'question':q,'rows':values} for q,values in questionmap.items() if len(values)>1]
    oldfolder=ROOT/'outputs/a2_training/a2_qwen3_17b_20261109_213712_836549'
    changed=[]
    for split,oldsplit in (('train','train'),('val','validation')):
        old=READ(oldfolder/(oldsplit+'.jsonl')); oldmap={r['metadata']['id']:r for r in old}
        stats=Counter()
        for item in [r for r in flat if r['split']==split]:
            id=item['meta']['id']
            if id not in oldmap:stats['added']+=1;continue
            previous=oldmap[id];oldpayload=json.loads(previous['messages'][1]['content'].split('\nJSON schema:\n')[0])
            if item['payload']['question']!=oldpayload['question']:
                stats['question_changed']+=1
                if item['meta']['review']==previous['metadata']['review']:stats['question_changed_same_review']+=1
                changed.append({'id':id,'split':split,'old_question':oldpayload['question'],'new_question':item['payload']['question'],
                                'unchanged_review':item['meta']['review']==previous['metadata']['review']})
            if item['gold']!=json.loads(previous['messages'][2]['content']):stats['answer_changed']+=1
        report['changed_from_trained_v2'][split]=dict(stats)
    negs=[r for r in flat if r['gold']['status']=='insufficient_evidence']
    blind=READ(ROOT/'outputs/a2_data_upgrade_v2/blind.jsonl')
    real_blind_ids={r['id'] for r in blind if 'decision_unambiguous' in r.get('result',{})}
    missing=[{'id':r['meta']['id'],'split':r['split'],'review_status':r['meta']['review_status'],
              'blind_reason_equals_review_reason':r['meta'].get('blind_answerability',{}).get('reason')==r['meta']['review'].get('reason')}
             for r in negs if r['meta']['id'] not in real_blind_ids]
    report['review_provenance']['negative_without_original_blind_log']=missing
    report['previous_manifest_and_token_reports_stale']={}
    for filename in ('README.md','token_lengths.json'):
        report['previous_manifest_and_token_reports_stale'][filename]=(DATA/filename).read_text(encoding='utf-8')[:500]
    # The separately frozen common72-case benchmark must remain outside future training.
    benchmark=READ(ROOT/'data/evaluation/a2_paired_20261110/cases.jsonl')
    report['paired72_overlap']={'document_ids':len({d for r in benchmark for d in r['source_documents']} & sets['train']['documents']),
                                'chunk_ids':len({d for r in benchmark for d in r['chunk_ids']} & sets['train']['chunks'])}
    assert files=={p.name:hash_file(p) for p in DATA.glob('*.jsonl')},'Dataset changed during audit'
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'changed_questions.json').write_text(json.dumps(changed,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','duplicate_questions','previous_manifest_and_token_reports_stale')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
