"""Sentence-catalog data validation and lexical reference scoring (not NLI)."""
import json
from itertools import permutations
from scripts.train_a2 import parse_json, summarize_results as summarize_decisions, plot_results as plot_decisions
from src.training.a2_sentence_protocol import SentenceDraft,SYSTEM
from src.training.a2_raft_protocol import GroundedDraft


def catalog_index(payload):
    evidence=payload['evidence'];ids=[e['evidence_id'] for e in evidence]
    if not payload['question'].strip() or not 1<=len(ids)<=3 or len(ids)!=len(set(ids)) or ids!=payload['allowed_evidence_ids']:
        raise ValueError('Invalid sentence catalog')
    index={}
    for e in evidence:
        last=0
        if not e['title'].strip() or not e['sentences']:raise ValueError('Empty evidence')
        for i,u in enumerate(e['sentences'],1):
            if u['sentence_id']!=f'{e["evidence_id"]}:S{i}' or u['start']<last or u['end']-u['start']!=len(u['text']) or not u['text'].strip():
                raise ValueError('Invalid unit ID or offsets')
            index[u['sentence_id']]=(e['evidence_id'],u['text']);last=u['end']
    return index


def resolve_catalog(selection,payload):
    selection=SentenceDraft.model_validate(selection);index=catalog_index(payload)
    claims=[]
    for c in selection.claims:
        if any(s not in index for s in c.sentence_ids):raise ValueError('Unknown sentence ID')
        if set(c.evidence_ids)!={index[s][0] for s in c.sentence_ids}:raise ValueError('Sentence/evidence mismatch')
        claims.append(dict(text=c.text,evidence_ids=c.evidence_ids,
            supporting_quotes=[dict(evidence_id=index[s][0],quote=index[s][1]) for s in c.sentence_ids]))
    return GroundedDraft.model_validate(dict(status=selection.status,claims=claims))


def parse_sample(record):
    messages=record['messages']
    if [m['role'] for m in messages]!=['system','user','assistant'] or messages[0]['content']!=SYSTEM:
        raise ValueError('Wrong sentence-selection prompt')
    data,schema=messages[1]['content'].split('\nJSON schema:\n',1)
    if parse_json(schema)!=SentenceDraft.model_json_schema():raise ValueError('Wrong output schema')
    payload=parse_json(data);gold=SentenceDraft.model_validate(parse_json(messages[2]['content']))
    resolve_catalog(gold,payload)
    return payload,gold.model_dump()


def score_answer(record,raw):
    payload,gold=parse_sample(record)
    gs=[set(c['sentence_ids']) for c in gold['claims']]
    result=dict(schema_valid=False,status_correct=False,citation_ids_valid=False,exact_match=False,
        expected_status=gold['status'],predicted_status=None,error=None,
        valid_sentence_ids=False,selection_set_exact=False,selection_tp=0,
        selection_expected=sum(map(len,gs)),selection_predicted=0,
        selection_precision=0.,selection_recall=0.,selection_f1=0.,resolved_output=None)
    try:
        pred=SentenceDraft.model_validate(parse_json(raw))
        ps=[set(c.sentence_ids) for c in pred.claims]
        result.update(schema_valid=True,predicted_status=pred.status,status_correct=pred.status==gold['status'],
                      exact_match=pred.model_dump()==gold,selection_predicted=sum(map(len,ps)))
        resolved=resolve_catalog(pred,payload)
        result.update(citation_ids_valid=True,valid_sentence_ids=True,resolved_output=resolved.model_dump())
        size=max(len(ps),len(gs));pg=ps+[set()]*(size-len(ps));gg=gs+[set()]*(size-len(gs))
        tp=max((sum(len(a&b) for a,b in zip(order,gg)) for order in permutations(pg)),default=0)
        np,ng=sum(map(len,ps)),sum(map(len,gs))
        precision=tp/np if np else float(not ng);recall=tp/ng if ng else float(not np)
        result.update(selection_tp=tp,selection_precision=precision,selection_recall=recall,
            selection_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
            selection_set_exact=sorted(map(sorted,ps))==sorted(map(sorted,gs)))
    except (ValueError,KeyError,TypeError) as exc:result['error']=str(exc)[:300]
    return result


def summarize_results(results):
    metrics=summarize_decisions(results)
    answerable=[r for r in results if r['expected_status']=='draft']
    tp=sum(r['selection_tp'] for r in results);np=sum(r['selection_predicted'] for r in results);ng=sum(r['selection_expected'] for r in results)
    p=tp/np if np else 0.;r=tp/ng if ng else 0.
    if all('generation_limit_hit' in r for r in results):
        metrics['generation_limit_hit_fraction']=sum(r['generation_limit_hit'] for r in results)/len(results)
    metrics.update(valid_sentence_ids=sum(r['valid_sentence_ids'] for r in results)/len(results),
        selection_set_exact=sum(r['selection_set_exact'] for r in answerable)/len(answerable) if answerable else None,
        selection_f1_on_answerable=sum(r['selection_f1'] for r in answerable)/len(answerable) if answerable else None,
        sentence_micro_precision=p,sentence_micro_recall=r,sentence_micro_f1=2*p*r/(p+r) if p+r else 0.,
        metric_scope='Lexical reference sentence matching, best claim permutation by IDs. Does NOT measure claim semantics, relevance beyond reference, or clinical correctness. Verbatim copying is by code, not a model score.')
    return metrics


def plot_results(logs,before,after,directory,model_name='Qwen3-1.7B sentence selection'):
    import tempfile
    from pathlib import Path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # Old plotter expects a new directory. Rebuild known images in private temp
    # storage; do not remove other user files when resuming an evaluation cell.
    with tempfile.TemporaryDirectory(prefix='a2_plots_') as temp:
        staged=Path(temp)/'figures'
        plot_decisions(logs,before,after,staged,model_name)
        import shutil
        for p in staged.iterdir():shutil.copy2(p,directory/p.name)
    import matplotlib.pyplot as plt
    keys=['status_correct','valid_sentence_ids','selection_set_exact','selection_f1_on_answerable']
    labels=['Correct decision','Valid sentence IDs','Exact reference set\n(answerable)','Sentence F1\n(answerable)']
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,m,name in [(-.18,before,'Before'),(.18,after,'After')]:
        bars=ax.bar([i+offset for i in range(4)],[100*(m[k] or 0) for k in keys],width=.35,label=name)
        ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_xticks(range(4),labels);ax.set(ylim=(0,118),ylabel='Validation (%)',title='A2: reference selection (not semantic correctness)');ax.legend()
    fig.tight_layout();fig.savefig(directory/'before_after.png',dpi=250);plt.close(fig)
