"""Real-index regression checks for bilingual retrieval; no answer-model training."""
import json
from pathlib import Path
import time
from src.rag.bilingual_queries import BilingualQueries, NllbTranslator
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256
from src.rag.reranker import LocalReranker
from src.rag.reranked_retriever import RerankedRetriever

ROOT=Path(__file__).resolve().parents[1]
CASES=[('caries_vi','Sâu răng là gì?', 'chunk-0cb2cd547fcd6eba308deeeb43fdad982d7bab4c4fbbe7c4a5ccb6d90497a2dc'),
       ('splint_vi','Máng nhai là gì?', 'chunk-f9fd0cd95e12eddcdf9b4bea5ae8bcec173204bcc598d48c62359f99a3f1fdd2'),
       ('caries_en','What is dental caries?', 'chunk-0cb2cd547fcd6eba308deeeb43fdad982d7bab4c4fbbe7c4a5ccb6d90497a2dc')]


def main():
    config=json.loads((ROOT/'config/assistant_local.json').read_text(encoding='utf-8'))['knowledge']
    output=ROOT/'outputs/rag_bilingual_20261111'
    output.mkdir(exist_ok=True)
    directories={'internal':ROOT/config['internal_index'],'public':ROOT/config['public_index']}
    before={w:{name:file_sha256(d/name) for name in ('manifest.json','records.jsonl','vectors.npy')} for w,d in directories.items()}
    encoder,ranker=LocalE5Embeddings(),LocalReranker()
    planner=BilingualQueries(NllbTranslator(ROOT/config['query_translation_dir']))
    internal=RerankedRetriever.from_disk(directories['internal'],ranker,encoder,config['candidate_k'])
    internal.query_expander=planner
    internal.context_neighbors=config['context_neighbors']
    internal.min_rerank_logit=config.get('internal_min_rerank_logit')
    rows=[]
    for key,query,expected in [*CASES,('unrelated','Thủ đô nước Pháp là gì?',None)]:
        start=time.monotonic()
        hits=internal.retrieve(query,top_k=3)
        expanded=internal.expand_context(hits)
        assert all(d.metadata['warehouse']=='internal' for d in expanded)
        ids=[d.id for d in hits]
        row={'id':key,'query':query,'expected_source_read_anchor':expected,
             'anchor_rank':ids.index(expected)+1 if expected in ids else None,
             'passed': (expected in ids) if expected is not None else not hits,
             'retrieval':internal.last_run.copy(),'evidence':[d.model_dump() for d in expanded],
             'seconds':time.monotonic()-start}
        rows.append(row)
        (output/'retrieval_checks.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
        print(key,'reference rank:',row['anchor_rank'],'context chunks:',len(expanded),flush=True)
    assert before=={w:{name:file_sha256(d/name) for name in before[w]} for w,d in directories.items()}
    result={'scope':'Three source-read positive cases plus one unrelated-query guard; not independent clinical accuracy',
            'indices_unchanged':True,'indices_sha256':before,'all_internal_only':True,
            'source_anchor_found_in_top3':sum(r['anchor_rank'] is not None for r in rows),'rows':rows}
    (output/'retrieval_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(r['passed'] for r in rows),'Bilingual retrieval regression failed'


if __name__=='__main__':main()
