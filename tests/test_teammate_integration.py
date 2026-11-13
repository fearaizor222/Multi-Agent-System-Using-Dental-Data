"""Integration contracts: source preservation, image upload and explicit failure."""
import base64
from io import BytesIO
from types import SimpleNamespace
import pytest
from PIL import Image
from tests.test_knowledge_pipeline import ScriptedLLM
from tests.test_synthesis import supported_state
from src.agents.teammate_synthesis import TeammateSynthesisAgent
from scripts.run_web import save_upload, WebApplication


class SynthesisLLM(ScriptedLLM):
    identity = {'repo': 'Qwen/Qwen2.5-1.5B-Instruct', 'variant': 'finetuned'}


def test_synthesis_uses_adapter_contract_and_renders_only_audited_claims():
    state = supported_state()
    claim = state['draft']['claims'][0]['text']
    llm = SynthesisLLM({'key_claims': [{'claim_id': 'c1', 'text': claim, 'citations': ['I1']}],
                        'draft_response': 'An extra unsupported sentence must never appear.'})
    result = TeammateSynthesisAgent(llm, lambda: None).run(state)
    assert result['model_used'] and result['strategy'] == 'verified_claims_qwen25_adapter_v1'
    assert 'unsupported sentence' not in result['message']
    assert result['claims'][0]['reference_ids'] == ['1', '2']


@pytest.mark.parametrize('claim_id,citations,text', [('invented', ['I1'], 'Sai'),
    ('c1', ['P99'], 'Sai'), ('c1', ['I1'], 'Uống 500 mg mỗi ngày')])
def test_synthesis_falls_back_on_changed_claim_source_or_numbers(claim_id, citations, text):
    reply = {'key_claims': [{'claim_id': claim_id, 'text': text, 'citations': citations}], 'draft_response': text}
    llm = SynthesisLLM(reply, reply)
    result = TeammateSynthesisAgent(llm, lambda: None).run(supported_state())
    assert not result['model_used'] and result['rewrite_error']
    assert '500 mg' not in result['message']


def test_upload_uses_generated_name_and_decodes_real_image(tmp_path, monkeypatch):
    import scripts.run_web as web
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    buffer = BytesIO()
    Image.new('RGB', (12, 12)).save(buffer, format='PNG')
    path = save_upload({'name': '../../escape.png', 'data': base64.b64encode(buffer.getvalue()).decode()}, 'abc')
    assert path == tmp_path / 'outputs/web_uploads/abc.png'
    assert Image.open(path).size == (12, 12)


def test_upload_rejects_invalid_image(tmp_path, monkeypatch):
    import scripts.run_web as web
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        save_upload({'data': base64.b64encode(b'not an image').decode()}, 'abc')


def test_vision_failure_is_explicit_not_a_fabricated_finding():
    from src.agents.local_vision import analyze_image
    fake = SimpleNamespace(generate=lambda *a, **kw: {'raw_output': 'not json'})
    with pytest.raises(ValueError):
        analyze_image(fake, 'image.png', 'Mô tả ảnh')


def test_missing_image_urgency_is_unknown_instead_of_assumed():
    from src.agents.local_vision import VisionFindings
    result = VisionFindings.model_validate({'image_type': 'other', 'anatomical_regions': [],
        'teeth_involved': [], 'detected_pathologies': [], 'preliminary_impression': 'Không rõ.'})
    assert result.clinical_urgency is None
