import json
import pytest

from scripts.train_a2 import parse_sample, plot_results, score_answer, summarize_results
from src.training.a2_raft_protocol import GroundedDraft, SYSTEM


def sample():
    payload={'question':'A là gì?','evidence':[
        {'evidence_id':'I1','title':'Nguồn','content':'A là khái niệm. B là ví dụ.'},
        {'evidence_id':'I2','title':'Nhiễu','content':'C là thông tin khác.'}],
        'allowed_evidence_ids':['I1','I2']}
    gold={'status':'draft','claims':[{'text':'A là khái niệm.','evidence_ids':['I1'],
        'supporting_quotes':[{'evidence_id':'I1','quote':'A là khái niệm.'}]}]}
    return {'messages':[{'role':'system','content':SYSTEM},
        {'role':'user','content':json.dumps(payload,ensure_ascii=False)+'\nJSON schema:\n'+json.dumps(GroundedDraft.model_json_schema(),ensure_ascii=False)},
        {'role':'assistant','content':json.dumps(gold,ensure_ascii=False)}]}


def test_grounded_record_and_prediction_validate():
    record=sample()
    _,gold=parse_sample(record)
    result=score_answer(record,record['messages'][2]['content'])
    assert gold['claims'][0]['supporting_quotes']
    assert result['quotes_verbatim'] and result['reference_proof_coverage']==1.


def test_wrong_source_quote_in_gold_rejected():
    record=sample();gold=json.loads(record['messages'][2]['content'])
    gold['claims'][0]['evidence_ids']=['I2']
    gold['claims'][0]['supporting_quotes'][0]['evidence_id']='I2'
    record['messages'][2]['content']=json.dumps(gold)
    with pytest.raises(ValueError,match='verbatim'):parse_sample(record)


def test_schema_validity_does_not_imply_valid_quote():
    record=sample();gold=json.loads(record['messages'][2]['content'])
    gold['claims'][0]['supporting_quotes'][0]['quote']='A là một thông tin bịa.'
    result=score_answer(record,json.dumps(gold))
    assert result['schema_valid'] and result['citation_ids_valid']
    assert not result['quotes_verbatim'] and result['reference_proof_coverage']==0.


def test_verbatim_quote_can_still_be_irrelevant():
    record=sample();gold=json.loads(record['messages'][2]['content'])
    gold['claims'][0]['supporting_quotes'][0]['quote']='B là ví dụ.'
    result=score_answer(record,json.dumps(gold))
    assert result['quotes_verbatim'] and result['reference_evidence_match']
    assert result['reference_proof_coverage']==0.


def test_missing_quote_for_a_cited_source_rejected():
    record=sample();gold=json.loads(record['messages'][2]['content'])
    gold['claims'][0]['evidence_ids'].append('I2')
    assert not score_answer(record,json.dumps(gold))['schema_valid']


def test_invalid_output_included_in_reference_coverage_denominator():
    record=sample()
    results=[score_answer(record,record['messages'][2]['content']),score_answer(record,'invalid')]
    metrics=summarize_results(results)
    assert metrics['reference_proof_coverage']==.5
    assert metrics['quoted_draft_count']==1 and metrics['draft_quotes_verbatim_fraction']==1.


def test_refusal_has_no_quotes_but_cannot_match_answerable_status():
    result=score_answer(sample(),'{"status":"insufficient_evidence","claims":[]}')
    assert result['schema_valid'] and result['quotes_verbatim']
    assert not result['status_correct'] and result['reference_proof_coverage']==0.


def test_quote_metrics_plot_handles_no_valid_draft(tmp_path):
    record=sample()
    before=summarize_results([score_answer(record,'invalid')])
    after=summarize_results([score_answer(record,record['messages'][2]['content'])])
    plot_results([{'step':1,'loss':1.},{'step':2,'eval_loss':.8}],before,after,tmp_path/'figures')
    assert {p.suffix for p in (tmp_path/'figures').iterdir()}=={'.png'}
