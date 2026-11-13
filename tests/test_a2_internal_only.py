"""A2 warehouse boundaries hold for every graph exit, not just normal answers."""
import pytest
from tests.test_knowledge_pipeline import FixedRetriever, ScriptedLLM, doc
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from src.agents.drafting import DraftingAgent
from src.agents.contracts import evidence_bundle


@pytest.mark.parametrize('documents,reply', [([], None),
    ([doc()], {'status': 'insufficient_evidence', 'claims': []})])
def test_missing_internal_answer_never_queries_public_or_retries_a2(documents, reply):
    public = FixedRetriever([doc('public', 'public')])
    model = ScriptedLLM(*([reply] if reply else []))
    state = create_knowledge_pipeline(FixedRetriever(documents), public, model, prefer_model=True).invoke({'question': 'Sâu răng là gì?'})
    assert state['status'] == 'insufficient_evidence' and state['draft']['claims'] == []
    assert public.calls == [] and model.calls == int(bool(documents))
    assert not state['repair_attempted'] and state['draft_origin'] == 'internal'
    assert all(t['step'] != 'a2_revise' for t in state['trace'])


def test_failed_public_verification_never_becomes_a2_public_drafting():
    public = FixedRetriever([])
    a2 = ScriptedLLM({'status': 'draft', 'claims': [{'text': 'Khí cụ bằng nhựa.', 'evidence_ids': ['I1']}]})
    a3 = ScriptedLLM()
    state = create_knowledge_pipeline(FixedRetriever([doc()]), public, a2, verification_llm=a3,
                                     prefer_model=True).invoke({'question': 'Máng nhai là gì?'})
    assert a2.calls == 1 and state['draft_origin'] == 'internal' and not state['repair_attempted']
    assert state['status'] == 'needs_review'
    assert {e['warehouse'] for e in state['draft_evidence']} == {'internal'}


def test_public_chunk_from_misconfigured_internal_retriever_is_rejected_before_llm():
    model = ScriptedLLM()
    state = create_knowledge_pipeline(FixedRetriever([doc('bad', 'public')]), FixedRetriever([]), model).invoke({'question': 'Câu hỏi'})
    assert state['status'] == 'error' and model.calls == 0


def test_direct_public_override_cannot_bypass_boundary():
    model = ScriptedLLM()
    data = evidence_bundle([doc('public', 'public')], 'P', 'public')
    with pytest.raises(ValueError, match='nội bộ'):
        DraftingAgent(model, prefer_model=True).run('Câu hỏi', data, warehouse='public')
    assert model.calls == 0


def test_relabeling_outer_warehouse_cannot_hide_public_metadata():
    model = ScriptedLLM()
    data = evidence_bundle([doc('public', 'public')], 'P', 'public')
    forged = data[0].model_copy(update={'warehouse': 'internal'})
    with pytest.raises(ValueError, match='nội bộ'):
        DraftingAgent(model).run('Câu hỏi', [forged])
    assert model.calls == 0

def test_v4_pipeline_sends_sentence_catalog_and_preserves_citations():
    class CapturingLLM(ScriptedLLM):
        def invoke(self, messages):
            self.messages = messages
            return super().invoke(messages)

    model = CapturingLLM({"status": "draft", "claims": [{
        "evidence_ids": ["I1"], "sentence_ids": ["I1:S1"],
        "text": "Máng nhai là khí cụ bằng nhựa."}]})
    state = create_knowledge_pipeline(
        FixedRetriever([doc()]), FixedRetriever([]), model,
        prefer_model=True, draft_sentence_selection=True
    ).invoke({"question": "Máng nhai là gì?"})
    assert model.calls == 1
    assert '"sentence_id": "I1:S1"' in model.messages[1].content
    assert state["draft"]["claims"][0]["citations"][0]["chunk_id"] == "internal-1"
    assert state["draft_grounding"][0]["resolved_selection"]["claims"][0]["supporting_quotes"][0]["quote"] == doc().content
    assert state["model_calls"][0]["calls"][0]["protocol"] == "sentence_selection_v4"


def test_v4_invalid_sentence_never_becomes_a_draft():
    model = ScriptedLLM({"status": "draft", "claims": [{
        "evidence_ids": ["I1"], "sentence_ids": ["I1:S999"],
        "text": "Máng nhai là khí cụ bằng nhựa."}]})
    state = create_knowledge_pipeline(
        FixedRetriever([doc()]), FixedRetriever([]), model,
        prefer_model=True, draft_sentence_selection=True
    ).invoke({"question": "Máng nhai là gì?"})
    assert state["status"] == "error"
    assert state["draft"]["claims"] == []
