"""One A2-only diagnostic with current internal retrieval and frozen v3 protocol."""
import argparse
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def retrieve(out):
    from src.rag.local_embeddings import LocalE5Embeddings
    from src.rag.reranker import LocalReranker
    from src.rag.reranked_retriever import RerankedRetriever
    from src.rag.bilingual_queries import BilingualQueries, NllbTranslator
    from src.agents.contracts import evidence_bundle
    config = json.loads((ROOT/'config/assistant_local.json').read_text(encoding='utf-8'))['knowledge']
    started = time.perf_counter()
    retriever = RerankedRetriever.from_disk(ROOT/config['internal_index'], LocalReranker(),
        LocalE5Embeddings(), candidate_k=config['candidate_k'])
    if config.get('bilingual_queries'):
        retriever.query_expander = BilingualQueries(NllbTranslator(ROOT/config['query_translation_dir']))
    retriever.context_neighbors = config.get('context_neighbors',0)
    retriever.min_rerank_logit = config.get('internal_min_rerank_logit')
    question = 'Sâu răng là gì?'
    hits = retriever.expand_context(retriever.retrieve(question,top_k=3))
    evidence = evidence_bundle(hits,'I','internal')
    result = dict(question=question, evidence=[e.model_dump() for e in evidence],
        retrieval= retriever.last_run, seconds=time.perf_counter()-started, config=config)
    (out/'retrieval.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Retrieved',len(evidence),'chunks in',round(result['seconds'],1),'seconds',flush=True)
    # Reproduce the completed run's protocol, which differs from the newly updated v4.
    with zipfile.ZipFile(ROOT/'outputs/a2_kaggle_sentence_v3/a2_sentence_v3_kaggle_bundle.zip') as z:
        for name in z.namelist():
            if name.startswith(('scripts/','src/')):
                p=out/'protocol_v3'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    with zipfile.ZipFile(ROOT/'outputs/a2_kaggle_sentence_v3/results/a2_sentence_v3_kaggle_01_results.zip') as z:
        for name in z.namelist():
            if name.startswith('adapter/'):
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    retrieve(args.output_dir)
