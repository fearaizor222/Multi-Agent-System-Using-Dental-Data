import json
import pytest
from scripts.a2_dataset_v2 import load_v2
from scripts.train_a2 import parse_sample, load_data


def fixture_data(path):
    (path / 'manifest.json').write_text(json.dumps({'version':'a2-v2-grounded-test',
        'split_method':'fixed documents','scope':'synthetic'}), encoding='utf-8')
    for split in ('train','val','test'):
        payload={'question':'Nội dung ' + split + ' là gì?', 'evidence':[{'evidence_id':'I1','title':split,
                 'content':'Tài liệu ' + split + ' có thông tin.'}],'allowed_evidence_ids':['I1']}
        claim={'text':payload['evidence'][0]['content'],'evidence_ids':['I1']}
        gold={'status':'draft','claims':[claim]}
        record={'messages':[{'role':'system','content':'Policy'},{'role':'user','content':json.dumps(payload,ensure_ascii=False)},
                            {'role':'assistant','content':json.dumps(gold,ensure_ascii=False)}]}
        meta={'id':split,'split':split,'document_id':split,'chunk_ids':[split],'question':payload['question'],
              'status':'draft','proofs':[{**claim,'proofs':[{'evidence_id':'I1','quote':claim['text']}]}],
              'review':{'status':'draft', **{k:True for k in ('clear','all_claims_supported','answers_current_question',
                         'supported_part_covered','citations_correct')}}}
        for suffix,data in (('.jsonl',record),('.metadata.jsonl',meta)):
            (path/(split+suffix)).write_text(json.dumps(data,ensure_ascii=False)+'\n', encoding='utf-8')


def mutate(path, split, change):
    p=path/(split+'.metadata.jsonl'); row=json.loads(p.read_text(encoding='utf-8')); change(row)
    p.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')


def test_preserve_supplied_splits_and_never_return_test_to_trainer(tmp_path):
    fixture_data(tmp_path)
    packet=load_data(tmp_path)
    assert set(packet)=={'train','validation','info'}
    assert packet['train'][0]['metadata']['document_id']=='train'
    assert packet['validation'][0]['metadata']['document_id']=='val'
    assert packet['info']['counts']=={'train':1,'validation':1,'test':1}
    assert packet['info']['test_reserved']


def test_distractor_source_overlap_rejected(tmp_path):
    fixture_data(tmp_path)
    mutate(tmp_path,'test',lambda m:m.update(chunk_ids=['train']))
    with pytest.raises(ValueError,match='overlap'): load_data(tmp_path)


def test_whole_document_overlap_rejected(tmp_path):
    fixture_data(tmp_path)
    mutate(tmp_path,'val',lambda m:m.update(document_id='train'))
    with pytest.raises(ValueError,match='overlap'): load_data(tmp_path)


def test_unverified_reference_rejected(tmp_path):
    fixture_data(tmp_path)
    mutate(tmp_path,'train',lambda m:m['review'].update(all_claims_supported=False))
    with pytest.raises(ValueError,match='Unapproved'): load_data(tmp_path)


def test_phantom_source_quote_rejected(tmp_path):
    fixture_data(tmp_path)
    mutate(tmp_path,'train',lambda m:m['proofs'][0]['proofs'][0].update(quote='Không có trong nguồn'))
    with pytest.raises(ValueError,match='Non-source'): load_data(tmp_path)


def test_missing_evidence_has_no_positive_claims_to_score(tmp_path):
    fixture_data(tmp_path)
    p=tmp_path/'test.jsonl'; record=json.loads(p.read_text(encoding='utf-8'))
    record['messages'][2]['content']=json.dumps({'status':'insufficient_evidence','claims':[]})
    p.write_text(json.dumps(record,ensure_ascii=False)+'\n',encoding='utf-8')
    mutate(tmp_path,'test',lambda m:m.update(status='insufficient_evidence',proofs=[],review={
        'status':'insufficient_evidence','clear':True,'actual_case':'missing_evidence',
        'all_claims_supported':False,'answers_current_question':False,'supported_part_covered':False,'citations_correct':False},
        blind_answerability={'decision_unambiguous':True,'status':'insufficient_evidence','answerable_quotes':[]}))
    assert load_data(tmp_path)['info']['labels']['test']=={'insufficient_evidence':1}
