import pytest
from src.training.a2_sentence_protocol import prepare_payload, selection_from_gold, resolve_selection
from src.agents.sentence_drafting import draft_with_sentences


def fixture():
    payload={'question':'Khi nào phải ghi fluoride?', 'evidence':[{'evidence_id':'I1','title':'Nguồn',
        'content':'Giới hạn 1.4 mg/L. Chỉ khi thêm fluoride thì phải ghi nhãn. Không bắt buộc ghi nồng độ.'}]}
    gold={'status':'draft','claims':[{'text':'Phải ghi khi thêm fluoride.', 'evidence_ids':['I1'],
        'supporting_quotes':[{'evidence_id':'I1','quote':'Chỉ khi thêm fluoride thì phải ghi nhãn.'}]}]}
    return payload,gold


def test_exact_source_offsets_and_decimal():
    p,g=fixture(); units=prepare_payload(p)['evidence'][0]['sentences']
    assert units[0]['text']=='Giới hạn 1.4 mg/L.'
    assert all(p['evidence'][0]['content'][u['start']:u['end']]==u['text'] for u in units)
    result=resolve_selection(selection_from_gold(g,p),p)
    assert result.claims[0].supporting_quotes[0].quote==g['claims'][0]['supporting_quotes'][0]['quote']


@pytest.mark.parametrize('ids,eids',[(['I1:S999'],['I1']),(['I1:S2'],['I2']),(['I1:S2','I1:S2'],['I1'])])
def test_bad_references_fail_closed(ids,eids):
    p,_=fixture()
    with pytest.raises(ValueError):
        resolve_selection({'status':'draft','claims':[{'text':'X','sentence_ids':ids,'evidence_ids':eids}]},p)


def test_units_do_not_depend_on_question_or_gold():
    p,_=fixture(); a=prepare_payload(p)['evidence'];p['question']='Câu khác'
    assert a==prepare_payload(p)['evidence']


def test_wrong_focus_is_not_certified_by_resolver():
    p,_=fixture()
    # Structurally valid selection, semantically irrelevant: leave to semantic review.
    out=draft_with_sentences(lambda _: '{"status":"draft","claims":[{"text":"Giới hạn 1.4 mg/L.","evidence_ids":["I1"],"sentence_ids":["I1:S1"]}]}',p)
    assert out.claims[0].supporting_quotes[0].quote=='Giới hạn 1.4 mg/L.'


def test_duplicate_json_keys_are_rejected():
    p,_=fixture()
    with pytest.raises(ValueError):
        draft_with_sentences(lambda _: '{"status":"draft","status":"insufficient_evidence","claims":[]}',p)


def test_footnotes_and_honorifics():
    from src.training.a2_sentence_protocol import source_units
    text='Ms. McKinney worked here.19 Operant conditioning rewards behavior.20 Modeling supports imitation.'
    units=source_units(text)
    assert units[0]['text']=='Ms. McKinney worked here.19'
    assert units[-1]['text']=='Modeling supports imitation.'


def test_numbered_items_and_bullets_keep_their_own_labels():
    from src.training.a2_sentence_protocol import source_units
    text='CHỈ ĐỊNH 1. Có tiếp xúc sớm tại TQTT, 2. Có tiếp xúc quá mức tại LMTĐ. - Mềm do cơ. - Cứng do khớp.'
    units=source_units(text)
    assert any(u['text']=='2. Có tiếp xúc quá mức tại LMTĐ.' for u in units)
    assert not any('TQTT' in u['text'] and 'LMTĐ' in u['text'] for u in units)
    assert not any('Mềm' in u['text'] and 'Cứng' in u['text'] for u in units)


def test_note_is_not_merged_with_numeric_table():
    from src.training.a2_sentence_protocol import source_units
    text='Table 2–3 maximum 53.7 2.4 1.7 Note: FDA requires labeling only if fluoride is added.'
    units=source_units(text)
    assert units[-1]['text']=='Note: FDA requires labeling only if fluoride is added.'


def test_independent_label_clauses_do_not_share_a_unit():
    from src.training.a2_sentence_protocol import source_units
    text='Note: FDA regulations require labeling only if added, but the bottler is not required to list concentration.'
    units=source_units(text)
    assert len(units)==2
    assert units[1]['text'].startswith('but the bottler is not required')


def test_table_rows_and_spaced_footnotes():
    from src.training.a2_sentence_protocol import source_units
    text='Positive reversal Change in a logical direction Negative reversal Change in an illogical direction Sources: Book.'
    units=source_units(text)
    assert units[0]['text']=='Positive reversal Change in a logical direction'
    assert units[1]['text']=='Negative reversal Change in an illogical direction'
    assert len(source_units('WHO collaboration.40, 42 Mobile care transports facilities.'))==2
