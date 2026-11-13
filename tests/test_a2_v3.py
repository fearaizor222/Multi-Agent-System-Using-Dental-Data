from copy import deepcopy
import json
import pytest
from scripts.a2_dataset_v2 import canonical_sha, validate_v3_provenance


def reference():
    payload={'question':'A là gì?','evidence':[{'evidence_id':'I1','title':'A','content':'A là một khái niệm.'}],
             'allowed_evidence_ids':['I1']}
    claims=[{'text':'A là một khái niệm.','evidence_ids':['I1'],
             'proofs':[{'evidence_id':'I1','quote':'A là một khái niệm.'}]}]
    row={'messages':[{'role':'system','content':'Policy'}, {'role':'user','content':json.dumps(payload,ensure_ascii=False)},
                     {'role':'assistant','content':json.dumps({'status':'draft','claims':[{k:claims[0][k] for k in ('text','evidence_ids')}]},ensure_ascii=False)}]}
    packet={'id':'x','question':payload['question'],'history':[],'evidence':payload['evidence'],'proposed_claims':claims}
    review={'id':'x','status':'draft','decision_unambiguous':True,'question_natural':True,
            **{k:True for k in ('all_claims_supported','answers_current_question','supported_part_covered','citations_correct','no_source_conflict')}}
    record={'id':'x','model':'gemini-3.1-flash-lite','request_sha256':canonical_sha(packet),
            'binding_sha256':canonical_sha(row['messages']),'result':review}
    meta={'id':'x','status':'draft','proofs':claims,'binding_sha256':canonical_sha(row['messages']),'review_record':record}
    return row,meta,payload,{canonical_sha(record)},set()


def test_current_review_matches_exact_training_input():
    validate_v3_provenance(*reference())


def test_question_edit_invalidates_previous_review():
    row,meta,payload,reviewlog,blindlog=reference()
    row['messages'][1]['content']=row['messages'][1]['content'].replace('A là gì?','A dùng để làm gì?')
    with pytest.raises(ValueError,match='Stale'):validate_v3_provenance(row,meta,payload,reviewlog,blindlog)


def test_editing_metadata_binding_cannot_replace_real_review_log():
    row,meta,payload,reviewlog,blindlog=reference()
    row['messages'][2]['content']='{"status":"insufficient_evidence","claims":[]}'
    meta['binding_sha256']=canonical_sha(row['messages']);meta['review_record']['binding_sha256']=meta['binding_sha256']
    with pytest.raises(ValueError,match='log entry'):validate_v3_provenance(row,meta,payload,reviewlog,blindlog)


def test_review_for_different_claims_rejected_even_if_binding_reused():
    row,meta,payload,reviewlog,blindlog=reference()
    meta['proofs'][0]['text']='B là một khái niệm.'
    with pytest.raises(ValueError,match='request'):validate_v3_provenance(row,meta,payload,reviewlog,blindlog)


def test_blind_metadata_without_real_request_log_rejected():
    row,meta,payload,reviewlog,blindlog=reference()
    meta['status']='insufficient_evidence';meta['review_record']['result']['status']='insufficient_evidence'
    reviewlog={canonical_sha(meta['review_record'])}
    meta['blind_answerability']={'status':'insufficient_evidence','decision_unambiguous':True,'answerable_quotes':[]}
    with pytest.raises(ValueError,match='real blind'):validate_v3_provenance(row,meta,payload,reviewlog,blindlog)
