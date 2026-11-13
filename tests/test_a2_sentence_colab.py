import ast
import json
from pathlib import Path
import pytest
from scripts.a2_sentence_io import parse_sample,score_answer,summarize_results,plot_results
from scripts.prepare_a2_sentence_colab import notebook
from src.training.a2_sentence_protocol import make_messages


def record(two=False):
    p=dict(question='Hỏi hai ý' if two else 'Hỏi ý đầu',evidence=[dict(evidence_id='I1',title='Nguồn',content='Ý đầu đúng. Ý thứ hai khác. Câu mồi khác ý.')])
    claims=[dict(text='Ý đầu đúng.',evidence_ids=['I1'],sentence_ids=['I1:S1'])]
    if two:claims.append(dict(text='Ý thứ hai khác.',evidence_ids=['I1'],sentence_ids=['I1:S2']))
    g=dict(status='draft',claims=claims)
    return dict(messages=make_messages(p)+[dict(role='assistant',content=json.dumps(g,ensure_ascii=False))])


def test_wrong_sentence_is_not_rewarded_for_being_verbatim():
    r=record();g=json.loads(r['messages'][-1]['content']);g['claims'][0]['sentence_ids']=['I1:S3']
    s=score_answer(r,json.dumps(g))
    assert s['status_correct'] and s['valid_sentence_ids']
    assert s['selection_f1']==0 and not s['selection_set_exact']
    assert s['resolved_output']['claims'][0]['supporting_quotes'][0]['quote']=='Câu mồi khác ý.'


def test_merged_or_missing_claim_does_not_get_full_reference_credit():
    r=record(True);g=json.loads(r['messages'][-1]['content']);g['claims']=g['claims'][:1]
    s=score_answer(r,json.dumps(g));assert s['selection_recall']==.5
    g['claims'][0]['sentence_ids']=['I1:S1','I1:S2']
    s=score_answer(r,json.dumps(g));assert s['selection_f1']<1


def test_swapped_claim_order_preserves_id_alignment_but_not_semantic_certification():
    r=record(True);g=json.loads(r['messages'][-1]['content']);g['claims'].reverse()
    s=score_answer(r,json.dumps(g));assert s['selection_f1']==1 and s['selection_set_exact']
    g['claims'][0]['text']='Nội dung bịa hoàn toàn.'
    assert score_answer(r,json.dumps(g))['selection_f1']==1  # IDs alone are not NLI


def test_unknown_ids_and_duplicate_json_keys_fail():
    r=record();g=json.loads(r['messages'][-1]['content']);g['claims'][0]['sentence_ids']=['I1:S999']
    s=score_answer(r,json.dumps(g));assert s['schema_valid'] and not s['valid_sentence_ids'] and s['selection_f1']==0
    assert not score_answer(r,'{"status":"draft","status":"insufficient_evidence","claims":[]}')['schema_valid']


def test_notebook_syntax_progress_and_challenge_order():
    sources=[''.join(c['source']) for c in notebook()['cells'] if c['cell_type']=='code']
    for source in sources:
        if not source.startswith('%pip'):ast.parse(source)
    stages=[s for s in sources if s.startswith('run_stage(')]
    assert stages[:6]==[f"run_stage('{s}')" for s in ('preflight','before','challenge_before','train','after','challenge_after')]
    assert any('stdout=subprocess.PIPE' in s for s in sources)


def test_png_figures_can_be_rebuilt_without_removing_other_files(tmp_path):
    r=record();score=score_answer(r,r['messages'][-1]['content']);metrics=summarize_results([score])
    directory=tmp_path/'figures';directory.mkdir();(directory/'keep.txt').write_text('keep')
    plot_results([dict(step=1,loss=.2),dict(step=1,eval_loss=.3)],metrics,metrics,directory)
    assert (directory/'keep.txt').read_text()=='keep'
    assert {p.name for p in directory.glob('*.png')}=={'loss.png','before_after.png','04_confusion_matrix.png','05_f1_by_decision.png'}
