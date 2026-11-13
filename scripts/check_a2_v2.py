"""Check v2 references against actual imported source and runtime prompts."""
from collections import Counter
import json
from pathlib import Path
import unicodedata
from scripts.train_a2 import load_data, parse_sample, DEFAULT_DATA
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY


def main():
    directory=DEFAULT_DATA
    root=Path(__file__).resolve().parents[1]
    packet=load_data(directory)
    norm=lambda s:' '.join(unicodedata.normalize('NFC',s).casefold().split())
    catalog={r['chunk_id']:r for r in [json.loads(l) for l in (root/'data/raw/internal_imports/2026-11-02/chunks.jsonl').read_text(encoding='utf-8').splitlines()]}
    report={'counts':packet['info']['counts'],'splits':{},'source_and_prompt_errors':[]}
    families={}
    for split in ('train','val','test'):
        rows=[json.loads(l) for l in (directory/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        metas=[json.loads(l) for l in (directory/(split+'.metadata.jsonl')).read_text(encoding='utf-8').splitlines()]
        cases=Counter(); history=0; docs=set(); families[split]=set()
        for row,meta in zip(rows,metas):
            payload,gold=parse_sample(row)
            expected=DRAFT_SYSTEM+(CONVERSATION_POLICY if 'conversation' in payload else '')
            assert row['messages'][0]['content']==expected
            history+=int('conversation' in payload)
            families[split].add(meta['family_id']); docs.add(meta['document_id']); cases[meta['mode']]+=1
            for evidence,chunk_id in zip(payload['evidence'],meta['chunk_ids']):
                chunk=catalog[chunk_id]
                assert chunk['document_id']==meta['document_id']
                assert norm(evidence['content'])==norm(chunk['content'])
        report['splits'][split]={'documents':len(docs),'history_samples':history,'cases':dict(cases)}
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not families[a]&families[b]
    report.update(status='passed', source_document_chunk_family_overlap=0,
                  scope='Source checks and AI review do not replace expert clinical adjudication.')
    (directory/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
