"""Read-only stage diagnosis on the actual internal index; no LLM or training."""
import json
from pathlib import Path
import time
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256
from src.rag.reranker import LocalReranker
from src.rag.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
GOLD = {'caries': ['chunk-0cb2cd547fcd6eba308deeeb43fdad982d7bab4c4fbbe7c4a5ccb6d90497a2dc'],
        'splint': ['chunk-f9fd0cd95e12eddcdf9b4bea5ae8bcec173204bcc598d48c62359f99a3f1fdd2']}


def main():
    config = json.loads((ROOT/'config/assistant_local.json').read_text())
    directory = ROOT/config['knowledge']['internal_index']
    output = ROOT/'outputs/a2_rag_audit_20261111'
    output.mkdir(exist_ok=True)
    hashes = {n: file_sha256(directory/n) for n in ('records.jsonl','vectors.npy','manifest.json')}
    encoder, reranker = LocalE5Embeddings(), LocalReranker()
    hybrid = HybridRetriever.from_disk(directory, encoder=encoder, candidate_k=10)
    corpus = {d.id: d for d in hybrid.documents}
    imported = {r['chunk_id']: r for r in map(json.loads, (ROOT/'data/raw/internal_imports/2026-11-02/chunks.jsonl').read_text(encoding='utf-8').splitlines())}
    assert set(corpus) == set(imported)
    assert all(d.content == imported[cid]['content'] for cid,d in corpus.items())
    rows = []
    for key, query in [('caries','Sâu răng là gì?'), ('caries','What is dental caries?'),
                       ('splint','Máng nhai là gì?')]:
        start = time.perf_counter()
        dense = hybrid.dense_retriever.retrieve(query, top_k=len(corpus))
        sparse = hybrid.sparse_retriever.retrieve(query, top_k=len(corpus))
        fused = hybrid.fuse(dense[:50], sparse[:50], top_k=50)
        row = {'query': query, 'reference_note': 'Source-read diagnostic anchors, not exhaustive relevance labels or a clinical benchmark',
               'reference_ids': GOLD[key], 'dense_reference_ranks': {cid: next((i for i,d in enumerate(dense,1) if d.id==cid), None) for cid in GOLD[key]},
               'bm25_reference_ranks': {cid: next((i for i,d in enumerate(sparse,1) if d.id==cid), None) for cid in GOLD[key]}, 'stages': []}
        for k in (10,30,50):
            # Match production: only candidate_k candidates from each branch, then RRF top candidate_k.
            candidates = hybrid.fuse(dense[:k], sparse[:k], top_k=k)
            ranked = reranker.rerank(query, candidates, top_k=len(candidates))
            row['stages'].append({'candidate_k': k, 'reference_rrf_ranks': {cid:next((i for i,d in enumerate(candidates,1) if d.id==cid),None) for cid in GOLD[key]},
                'reference_rerank_ranks': {cid:next((i for i,d in enumerate(ranked,1) if d.id==cid),None) for cid in GOLD[key]},
                'top3': [d.model_dump() for d in ranked[:3]]})
        # Is the answer passage recognizable if the candidate stage actually provides it?
        oracle_candidates = list({d.id:d for d in [*fused,*(corpus[c] for c in GOLD[key])]}.values())
        oracle = reranker.rerank(query, oracle_candidates, top_k=len(oracle_candidates))
        row['reference_injected_rerank_ranks'] = {cid: next(i for i,d in enumerate(oracle,1) if d.id==cid) for cid in GOLD[key]}
        row['reference_documents'] = [corpus[c].model_dump() for c in GOLD[key]]
        row['seconds'] = time.perf_counter()-start
        rows.append(row)
        (output/'retrieval_stages.json').write_text(json.dumps({'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
        print(query, 'dense',row['dense_reference_ranks'],'stages',[(s['candidate_k'],s['reference_rrf_ranks'],s['reference_rerank_ranks']) for s in row['stages']],flush=True)
    assert hashes == {n: file_sha256(directory/n) for n in hashes}
    result = {'internal_chunks': len(corpus), 'index_matches_imported_corpus': True, 'index_unchanged': True,
              'indices_sha256': hashes, 'llm_calls':0,'training_executed':False,'rows':rows}
    (output/'retrieval_stages.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Read-only RAG audit complete',flush=True)


if __name__ == '__main__':
    main()
