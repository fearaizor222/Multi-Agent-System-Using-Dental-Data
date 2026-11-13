"""Measure the unchanged local assistant, including warmup and nested CPU/GPU calls."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import functools
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    directory = args.output_dir or ROOT/'outputs/local_latency_20261112'/datetime.now().strftime('%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    events, stack = [], []
    phase = 'imports'
    stream = (directory/'events.jsonl').open('w', encoding='utf-8', buffering=1)

    @contextmanager
    def span(name, detail=None):
        event = {'id': len(events), 'parent': stack[-1] if stack else None, 'phase': phase,
                 'name': name, 'start_seconds': time.perf_counter()-started, 'detail': detail or {}}
        events.append(event); stack.append(event['id'])
        stream.write(json.dumps({'event':'start', **event}, ensure_ascii=False)+'\n')
        try:
            yield event
        except BaseException as exc:
            event['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            event['seconds'] = time.perf_counter()-started-event['start_seconds']
            stack.pop()
            stream.write(json.dumps({'event':'end', **event}, ensure_ascii=False)+'\n')
            print(f'{phase}: {name} {event["seconds"]:.2f}s', flush=True)

    def wrap(cls, method, name):
        original = getattr(cls, method)
        @functools.wraps(original)
        def measured(self, *a, **kw):
            detail = {}
            if name == 'gpu.generate': detail['role'] = a[0] if a else kw.get('role')
            if name == 'reranker': detail['candidates'] = len(a[1]) if len(a)>1 else len(kw.get('documents', []))
            with span(name, detail) as event:
                result = original(self, *a, **kw)
                if name == 'gpu.generate':
                    event['detail'].update({k:result.get(k) for k in ('generation_seconds','input_tokens','output_tokens','allocated_mib')})
                return result
        setattr(cls, method, measured)

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1')
    with span('imports'):
        from src.graph.assistant_runtime import AssistantBundle
        from src.core.shared_qwen import SharedQwenClient
        from src.rag.local_embeddings import LocalE5Embeddings
        from src.rag.reranker import LocalReranker
        from src.rag.entailment import LocalEntailment
        from src.rag.bilingual_queries import NllbTranslator, BilingualQueries
        from src.rag.reranked_retriever import RerankedRetriever
    for cls, method, name in [
        (SharedQwenClient,'ensure_started','gpu.ensure_started'),
        (SharedQwenClient,'generate','gpu.generate'),
        (LocalE5Embeddings,'__init__','load.embedding'),
        (LocalReranker,'__init__','load.reranker'),
        (LocalEntailment,'__init__','load.nli'),
        (NllbTranslator,'__init__','load.translation'),
        (LocalE5Embeddings,'encode_inputs','embedding'),
        (LocalReranker,'rerank','reranker'),
        (LocalEntailment,'assess','nli'),
        (NllbTranslator,'translate','translation'),
        (BilingualQueries,'__call__','query_expansion'),
        (RerankedRetriever,'retrieve','retrieve'),
        (RerankedRetriever,'retrieve_excluding','retrieve_excluding'),
        (RerankedRetriever,'rank_candidates','rank_candidates'),
    ]: wrap(cls, method, name)
    config_path = ROOT/'config/assistant_local.json'
    (directory/'config.json').write_bytes(config_path.read_bytes())
    bundle = AssistantBundle(config_path)
    results = []
    try:
        phase = 'warmup'
        with span('warmup'):
            identity = bundle.warmup()
        (directory/'identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
        for number, question in enumerate(['Sâu răng là gì?', 'Máng nhai là gì?', 'Máng nhai là gì?'], 1):
            phase = f'turn_{number:03d}'
            bundle.on_progress = lambda node, status: print(f'PROGRESS {phase} {node}: {status}', flush=True)
            with span('turn', {'question':question}) as turn:
                result = {'input':{'message':question, 'history':[]}}
                for event in bundle.graph.stream(result, stream_mode='updates'):
                    for node, update in event.items():
                        result.update(update)
                        print(f'PROGRESS {phase} {node}: {update.get("status")}', flush=True)
            result['profile_seconds'] = turn['seconds']
            (directory/f'{phase}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            results.append({'turn':phase,'question':question,'seconds':turn['seconds'],
                            'status':result.get('status'),'answer':result.get('message'),
                            'trace':result.get('trace',[])})
            (directory/'turn_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        bundle.close()
        stream.close()
        (directory/'spans.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
        print('OUTPUT', directory, flush=True)


if __name__ == '__main__':
    main()
