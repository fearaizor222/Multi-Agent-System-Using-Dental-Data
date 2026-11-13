import json
from pathlib import Path

import pytest

from scripts.train_a2 import (ROOT, MODELS, configure_tokenizer, encode_sample, load_data,
                             parse_sample, plot_results, score_answer, summarize_results)


def sample(doc):
    payload={"question":"Câu hỏi " + doc, "evidence":[{"evidence_id":"I1","title":doc,"content":"Nội dung " + doc}],
             "allowed_evidence_ids":["I1"]}
    gold={"status":"draft","claims":[{"text":"Nội dung " + doc,"evidence_ids":["I1"]}]}
    return {"messages":[{"role":"system","content":"Policy"},
                        {"role":"user","content":json.dumps(payload,ensure_ascii=False)},
                        {"role":"assistant","content":json.dumps(gold,ensure_ascii=False)}]}


def write_data(path):
    for split,docs in [("train",["A","B"]),("val",["A","C"])]:
        records=[];metas=[]
        for index,doc in enumerate(docs):
            record=sample(doc + str(index))
            records.append(record)
            payload,gold=parse_sample(record)
            metas.append({"document_id":doc,"chunk_ids":[doc+str(index)],
                          "question":payload['question'],"status":gold['status']})
        for suffix,rows in [(".jsonl",records),(".metadata.jsonl",metas)]:
            (path/(split+suffix)).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')


def test_split_holds_entire_documents_and_preserves_all_rows(tmp_path):
    write_data(tmp_path)
    # Make the two samples for document A distinct while preserving their document grouping.
    p=tmp_path/'val.jsonl';rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
    rows[0]=sample('A-other')
    p.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    m=tmp_path/'val.metadata.jsonl';meta=[json.loads(l) for l in m.read_text(encoding='utf-8').splitlines()]
    meta[0]['question']=parse_sample(rows[0])[0]['question'];meta[0]['chunk_ids']=['A-other']
    m.write_text('\n'.join(json.dumps(r) for r in meta),encoding='utf-8')
    packet=load_data(tmp_path)
    a={r['metadata']['document_id'] for r in packet['train']}
    b={r['metadata']['document_id'] for r in packet['validation']}
    assert not a&b and sum(packet['info']['counts'].values())==4
    assert load_data(tmp_path)==packet


def test_empty_validation_and_partial_metadata_are_rejected(tmp_path):
    write_data(tmp_path)
    (tmp_path/'val.jsonl').write_text('',encoding='utf-8')
    with pytest.raises(ValueError,match='rỗng'):load_data(tmp_path)
    write_data(tmp_path)
    (tmp_path/'train.metadata.jsonl').write_text('',encoding='utf-8')
    with pytest.raises(ValueError,match='chưa khớp'):load_data(tmp_path)


def test_out_of_context_gold_citation_rejected():
    record=sample('A');gold=json.loads(record['messages'][2]['content'])
    gold['claims'][0]['evidence_ids']=['I9']
    record['messages'][2]['content']=json.dumps(gold)
    with pytest.raises(ValueError,match='trích dẫn'):parse_sample(record)


def test_invalid_prediction_not_repaired_and_schema_not_confused_with_citations():
    record=sample('A')
    assert not score_answer(record,'not json')['schema_valid']
    assert not score_answer(record,'{"status":"draft","status":"draft","claims":[]}')['schema_valid']
    gold=json.loads(record['messages'][2]['content']);gold['claims'][0]['evidence_ids']=['I9']
    scores=score_answer(record,json.dumps(gold))
    assert scores['schema_valid'] and not scores['citation_ids_valid'] and not scores['exact_match']
    assert score_answer(record,record['messages'][2]['content'])['exact_match']


def test_assistant_mask_and_no_truncation():
    class Tokens:
        def apply_chat_template(self,messages,**kwargs):
            return [1,2,3] if len(messages)==2 else [1,2,3,4,5]
    encoded=encode_sample(Tokens(),sample('A'),5)
    assert encoded['labels']==[-100,-100,-100,4,5]
    with pytest.raises(ValueError,match='không tự cắt'):encode_sample(Tokens(),sample('A'),4)


def test_figures_are_png_only(tmp_path):
    record=sample('A')
    metrics=summarize_results([score_answer(record,record['messages'][2]['content']),score_answer(record,'invalid')])
    plot_results([{'step':1,'loss':1.},{'step':2,'eval_loss':.8}],metrics,metrics,tmp_path/'figures')
    assert {p.name for p in (tmp_path/'figures').iterdir()}=={'loss.png','before_after.png','04_confusion_matrix.png','05_f1_by_decision.png'}


def test_invalid_output_remains_in_confusion_matrix_and_reduces_recall():
    record=sample('A')
    valid=score_answer(record,record['messages'][2]['content'])
    invalid=score_answer(record,'not JSON')
    metrics=summarize_results([valid,invalid])
    assert metrics['confusion_matrix']==[[1,0,1],[0,0,0]]
    assert metrics['per_status']['draft']['support']==2
    assert metrics['per_status']['draft']['precision']==1.
    assert metrics['per_status']['draft']['recall']==.5
    assert metrics['per_status']['draft']['f1']==pytest.approx(2/3)


def test_wrong_status_is_counted_as_off_diagonal():
    record=sample('A')
    row=score_answer(record,'{"status":"insufficient_evidence","claims":[]}')
    metrics=summarize_results([row])
    assert metrics['confusion_matrix']==[[0,1,0],[0,0,0]]
    assert metrics['status_correct']==0 and metrics['per_status']['draft']['f1']==0


@pytest.mark.parametrize('model_id',list(MODELS))
def test_real_tokenizers_have_exact_prompt_masks_and_correct_end_tokens(model_id):
    from transformers import AutoTokenizer
    spec=MODELS[model_id]
    path=ROOT/'data/models'/spec['repo'].replace('/','--')/spec['revision']
    tokenizer=configure_tokenizer(AutoTokenizer.from_pretrained(path,local_files_only=True),model_id)
    record=sample('A')
    prompt=tokenizer.apply_chat_template(record['messages'][:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if model_id=='medscholar_15b':
        assert prompt.startswith('### Question:\n') and prompt.endswith('### Answer:\n')
        assert tokenizer.eos_token=='<|endoftext|>'
    elif model_id=='vinallama_27b':
        assert prompt.endswith('<|im_start|>assistant\n') and tokenizer.eos_token=='<|im_end|>'
    encoded=encode_sample(tokenizer,record)
    supervised=[value for value in encoded['labels'] if value!=-100]
    assert supervised and tokenizer.eos_token_id in supervised
    assert encoded['labels'][0]==-100


def test_modal_entrypoints_isolate_model_output_and_cache_paths():
    import ast
    entries=[('modal_train_a2.py','qwen3_17b'),('modal_train_a2_vinallama.py','vinallama_27b'),
             ('modal_train_a2_medscholar.py','medscholar_15b')]
    for filename,model_id in entries:
        source=(ROOT/'scripts'/filename).read_text(encoding='utf-8')
        tree=ast.parse(source)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='train']
        assert len(calls)==1
        key=next((kw.value.value for kw in calls[0].keywords if kw.arg=='model_id'),'qwen3_17b')
        assert key==model_id
        assert f'/vol/hf/{model_id}' in source and f'a2_{model_id}_' in source
