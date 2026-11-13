"""Tests for Stage 2 (v4) protocol upgrades, normalization, and dataset invariants."""
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from scripts.a2_sentence_io import parse_sample, score_answer
from scripts.check_a2_sentence_v4 import check as check_v4
from scripts.train_a2_v4_kaggle import checkpoint_composite
from src.agents.sentence_drafting import draft_with_sentences
from src.training.a2_sentence_protocol import (
    SentenceDraft,
    normalize_source_content,
    source_units,
)


def test_selected_claim_field_order_and_regex_constraints():
    schema = SentenceDraft.model_json_schema()
    props = schema['$defs']['SelectedClaim']['properties']
    assert list(props.keys()) == ['evidence_ids', 'sentence_ids', 'text']
    assert schema['$defs']['SelectedClaim']['required'] == ['evidence_ids', 'sentence_ids', 'text']
    assert props['evidence_ids']['items']['pattern'] == r'^I\d+$'
    assert props['sentence_ids']['items']['pattern'] == r'^I\d+:S\d+$'

    ok = SentenceDraft.model_validate({
        'status': 'draft',
        'claims': [{'evidence_ids': ['I1'], 'sentence_ids': ['I1:S2'], 'text': 'Hợp lệ.'}],
    })
    assert list(ok.model_dump()['claims'][0].keys()) == ['evidence_ids', 'sentence_ids', 'text']

    for bad_eid in ('Hàm lượng hóa chất', 'E1', '1'):
        with pytest.raises(ValidationError):
            SentenceDraft.model_validate({
                'status': 'draft',
                'claims': [{'evidence_ids': [bad_eid], 'sentence_ids': ['I1:S1'], 'text': 'Sai ID.'}],
            })

    for bad_sid in ('I1:S', 'S1', 'I1:1', 'I1:S2a'):
        with pytest.raises(ValidationError):
            SentenceDraft.model_validate({
                'status': 'draft',
                'claims': [{'evidence_ids': ['I1'], 'sentence_ids': [bad_sid], 'text': 'Sai câu.'}],
            })


def test_source_units_v4_fixes_decimals_bullets_and_tables():
    # Decimals preceded by operators or ranges must stay intact; footnote splits stay active.
    bottle = (
        'The FDA does not allow imported bottled water with no added fluoride to contain >1.4 mg fluoride/L '
        'or with added fluoride to contain >0.8 mg/L. Range 53.8–58.3 applies here. Worked here.19 Next sentence.'
    )
    texts = [u['text'] for u in source_units(bottle)]
    assert any('>1.4 mg fluoride/L' in t and '>0.8 mg/L.' in t for t in texts)
    assert any('53.8–58.3 applies here.' in t for t in texts)
    assert 'Next sentence.' in texts

    # Black square bullets without spaces must split individually.
    oha = 'Intro.■Encourage self-care. ■Uses a small head.'
    assert [u['text'] for u in source_units(oha)] == [
        'Intro.',
        '■Encourage self-care.',
        '■Uses a small head.',
    ]

    # Single-header payment_systems layout must still separate trailing prose.
    pay = 'Two-party system: Dentist and patient directly. years, many dental companies emerged.'
    assert [u['text'] for u in source_units(pay)] == [
        'Two-party system: Dentist and patient directly.',
        'years, many dental companies emerged.',
    ]

    # Standalone punctuation-only slices must be dropped.
    punct = 'First item.\n.\n- \nSecond item.'
    assert [u['text'] for u in source_units(punct)] == ['First item.', 'Second item.']


def test_normalize_and_runtime_draft_with_sentences():
    raw = 'fluor\xadide \x07 Mức p\ue0230.05 và cƣơng độ.'
    norm = normalize_source_content(raw)
    assert '\xad' not in norm and '\x07' not in norm and '\ue023' not in norm and 'ƣ' not in norm
    assert 'fluoride' in norm and '≤0.05' in norm and 'cương độ' in norm

    captured = {}

    def fake_generate(messages):
        captured['user'] = messages[1]['content']
        return json.dumps({
            'status': 'draft',
            'claims': [{
                'evidence_ids': ['I1'],
                'sentence_ids': ['I1:S1'],
                'text': 'Ngưỡng có ý nghĩa thống kê là p ≤ 0,05.',
            }],
        }, ensure_ascii=False)

    payload = {
        'question': 'Ngưỡng p là bao nhiêu?',
        'evidence': [{
            'evidence_id': 'I1',
            'title': 'Biostatistics',
            'content': 'The accepted level is p \ue023 .05\xad.',
        }],
    }
    draft = draft_with_sentences(fake_generate, payload)
    assert '\ue023' not in captured['user'] and '\xad' not in captured['user']
    assert draft.status == 'draft'
    assert draft.claims[0].supporting_quotes[0].quote == 'The accepted level is p ≤ .05.'


def test_v4_dataset_invariants_and_v3_preserved():
    root = Path(__file__).resolve().parents[1]
    v3_manifest = json.loads((root / 'data/processed/a2_sentence_selection_v3_hard/manifest.json').read_text(encoding='utf-8'))
    assert v3_manifest['counts'] == {'train': 1054, 'val': 84, 'challenge': 56}

    summary = check_v4(write_report=False)
    assert summary['train']['count'] == 1076
    assert summary['val']['count'] == 84
    assert summary['challenge']['count'] == 80
    assert summary['train']['partial_directions'] == {'pos_first': 40, 'neg_first': 40}
    assert summary['challenge']['partial_directions'] == {'pos_first': 8, 'neg_first': 8}

    # Verify sample parsing & gold self-scoring on v4 validation and challenge records.
    for split in ('val', 'challenge'):
        first = json.loads((root / f'data/processed/a2_sentence_selection_v4/{split}.jsonl').read_text(encoding='utf-8').splitlines()[0])
        payload, gold = parse_sample(first)
        scored = score_answer(first, first['messages'][-1]['content'])
        assert scored['schema_valid'] is True
        assert scored['selection_set_exact'] is True
        assert payload['question'] and gold['status'] in ('draft', 'insufficient_evidence')


def test_checkpoint_composite_penalizes_invalid_and_ranks_cleanly():
    bad = {'schema_valid': 0.90, 'valid_sentence_ids': 1.0, 'status_macro_f1': 0.99, 'selection_f1_on_answerable': 0.99, 'selection_set_exact': 0.99}
    ok = {'schema_valid': 1.0, 'valid_sentence_ids': 1.0, 'status_macro_f1': 0.90, 'selection_f1_on_answerable': 0.85, 'selection_set_exact': 0.80}
    better = {'schema_valid': 1.0, 'valid_sentence_ids': 1.0, 'status_macro_f1': 0.94, 'selection_f1_on_answerable': 0.90, 'selection_set_exact': 0.86}
    assert checkpoint_composite(bad, ok) < 0
    assert checkpoint_composite(better, better) > checkpoint_composite(ok, ok)
