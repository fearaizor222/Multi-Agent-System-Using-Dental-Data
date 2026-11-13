"""Freeze A2-only diagnostic cases before reading either model's predictions."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY
from src.agents.contracts import GeneratedDraft

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/evaluation/a2_base_vs_v3_20261111'


def main():
    if OUT.exists():
        raise FileExistsError('Frozen comparison already exists')
    source={r['chunk_id']:r for r in map(json.loads,(ROOT/'data/raw/internal_imports/2026-11-02/chunks.jsonl').read_text(encoding='utf-8').splitlines())}
    data=ROOT/'data/processed/a2_drafting_v3'
    records=list(map(json.loads,(data/'test.jsonl').read_text(encoding='utf-8').splitlines()))
    metadata=list(map(json.loads,(data/'test.metadata.jsonl').read_text(encoding='utf-8').splitlines()))
    assert len(records)==len(metadata)==93
    pairs=list(zip(records,metadata));rng=random.Random(20261111)
    selected=[pair for pair in pairs if pair[1]['mode']!='direct']
    for lang in ('vi','en'):
        pool=[pair for pair in pairs if pair[1]['mode']=='direct' and {source[c]['language'] for c in pair[1]['chunk_ids']}=={lang}]
        rng.shuffle(pool)
        selected.extend(pool[:15])
    assert len(selected)==48
    cases=[]
    for record,meta in selected:
        payload=json.loads(record['messages'][1]['content'].split('\nJSON schema:\n')[0])
        assert all(e['evidence_id'].startswith('I') for e in payload['evidence'])
        cases.append({'id':meta['id'],'group':'heldout_subset','mode':meta['mode'],
            'source_languages':sorted({source[c]['language'] for c in meta['chunk_ids']}),
            'family_id':meta['family_id'],'source_chunk_ids':meta['chunk_ids'],
            'payload':payload,'expected':json.loads(record['messages'][2]['content']),
            'reference_proofs':meta['proofs']})
    ids={'caries':'chunk-0cb2cd547fcd6eba308deeeb43fdad982d7bab4c4fbbe7c4a5ccb6d90497a2dc',
         'caries_next':'chunk-dd7f3d328a2c1287a3822b01b31df4d24a3b3d3212bd2386a4c3ddcb112e632a',
         'splint':'chunk-f9fd0cd95e12eddcdf9b4bea5ae8bcec173204bcc598d48c62359f99a3f1fdd2'}
    def add(key,question,chunk_keys,answer,cited,focused=False):
        evidence=[];spans=[]
        for i,chunk_key in enumerate(chunk_keys,1):
            row=source[ids[chunk_key]];text=row['content'];start,end=0,len(text)
            if focused:
                start=text.index('Dental Caries\n') if chunk_key=='caries' else text.index('subsequently dissolve')
                end=len(text) if chunk_key=='caries' else text.index('EPIDEMIOLOGY')
            spans.append({'chunk_id':row['chunk_id'],'start':start,'end':end})
            evidence.append({'evidence_id':f'I{i}','title':' > '.join(row['heading_path']) or row['document_id'],'content':text[start:end]})
        cases.append({'id':key,'group':'known_regression','mode':'focus_distractor' if len(chunk_keys)>1 else 'focus',
            'source_languages':sorted({source[ids[c]]['language'] for c in chunk_keys}),
            'family_id':key.split('_')[0],'source_chunk_ids':[ids[c] for c in chunk_keys],
            'payload':{'question':question,'evidence':evidence,'allowed_evidence_ids':[e['evidence_id'] for e in evidence]},
            'expected':{'status':'draft','claims':[{'text':answer,'evidence_ids':cited}]},'source_spans':spans,
            'grading_note':'Definition/mechanism required; prevalence and age alone do not answer what-is. Reference is semantic, not exact-match.'})
    add('caries_full','Sâu răng là gì?',['caries'],
        'Sâu răng liên quan đến axit do vi khuẩn trong mảng bám tạo ra, làm mất khoáng men răng.',['I1'])
    add('caries_focused','Sâu răng là gì?',['caries','caries_next'],
        'Sâu răng liên quan đến axit do vi khuẩn trong mảng bám tạo ra, làm mất khoáng men răng và tiến triển thành lỗ sâu.',['I1','I2'],True)
    add('splint_full','Máng nhai là gì?',['splint'],
        'Máng nhai là khí cụ bằng nhựa đặt giữa hai cung răng, dùng trong chẩn đoán và điều trị rối loạn cắn khớp.',['I1'])
    add('splint_distractor','Máng nhai là gì?',['caries','splint'],
        'Máng nhai là khí cụ bằng nhựa đặt giữa hai cung răng, dùng trong chẩn đoán và điều trị rối loạn cắn khớp.',['I2'])
    rng.shuffle(cases)
    for case in cases:
        case['system']=DRAFT_SYSTEM+(CONVERSATION_POLICY if 'conversation' in case['payload'] else '')
        case['user']=json.dumps(case['payload'],ensure_ascii=False)+'\nJSON schema:\n'+json.dumps(GeneratedDraft.model_json_schema(),ensure_ascii=False)
    OUT.mkdir(parents=True)
    blob=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases).encode('utf-8')
    (OUT/'cases.jsonl').write_bytes(blob)
    manifest={'count':52,'heldout_count':48,'known_regression_count':4,'seed':20261111,
        'cases_sha256':hashlib.sha256(blob).hexdigest(),'system_sha256':hashlib.sha256(DRAFT_SYSTEM.encode()).hexdigest(),
        'heldout_files_sha256':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ('test.jsonl','test.metadata.jsonl')},
        'modes':dict(Counter(c['mode'] for c in cases)),
        'scope':'A2-only frozen diagnostic comparison. Selected heldout subset previously evaluated, plus known failures; not a new blind clinical benchmark. All source contexts from internal corpus.',
        'grading':'Source-reading for faithfulness, question focus, translation, omissions and abstention; exact wording not required.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
