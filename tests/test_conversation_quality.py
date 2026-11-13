from copy import deepcopy

import pytest

from src.agents.question_context import prepare_routing_context, compact_history, parse_request, verification_question
from src.agents.drafting import DraftingAgent
from src.agents.contracts import evidence_bundle, citation_of, Evidence
from src.agents.contextual_drafting import atomic_clauses
from src.agents.public_retrieval import retrieve_public_evidence
from src.rag.reranked_retriever import RerankedRetriever
from src.agents.synthesis import SynthesisAgent
from src.rag.entailment import aggregate_predictions
from src.graph.assistant_pipeline import create_assistant_pipeline
from src.rag.base import DentalDocument
from tests.test_knowledge_pipeline import ScriptedLLM
from tests.test_synthesis import supported_state, nli


def context(message="chữa thế nào"):
    return {"message": message, "history": [
        {"role":"user","content":"sâu răng là gì"},
        {"role":"assistant","content":"Định nghĩa và giải thích đã kiểm chứng. [1, 2]\n\nNguồn công khai:\n[1] MSD — https://example.org/a\n[2] Bộ Y tế — https://example.org/b"}],
        "attachments":[],"capabilities":{"A2":True,"A5":False,"lookup_clinic_service":False}}


def test_followup_uses_previous_user_topic_and_retains_original_context():
    raw=context();saved=deepcopy(raw)
    model,detail=prepare_routing_context(raw)
    assert raw==saved
    assert model["message"]=="Điều trị sâu răng như thế nào?"
    assert model["history"][1]["content"]=="Định nghĩa và giải thích đã kiểm chứng."
    assert detail["strategy"]=="previous_user_topic" and detail["bibliography_removed"]


@pytest.mark.parametrize("message",["Tôi đau răng, kê thuốc cho tôi", "Giá bao nhiêu", "viêm lợi là gì"])
def test_explicit_new_or_personal_request_is_not_rewritten(message):
    raw=context(message);projected,_=prepare_routing_context(raw)
    assert projected["message"]==message


def test_followup_does_not_reuse_topic_across_a_change_of_goal_or_attachments():
    raw=context();raw["history"].append({"role":"user","content":"Tôi muốn hỏi giá dịch vụ"})
    assert prepare_routing_context(raw)[0]["message"]=="chữa thế nào"
    raw=context();raw["attachments"]=[{"id":"x"}]
    assert prepare_routing_context(raw)[0]["message"]=="chữa thế nào"
    raw=context();raw["history"]=[]
    assert prepare_routing_context(raw)[0]["message"]=="chữa thế nào"


def test_compaction_only_removes_a_complete_generated_reference_tail():
    raw=[{"role":"assistant","content":"Nội dung chính.\nNguồn tham khảo:\nThông tin người dùng còn cần đọc."}]
    assert compact_history(raw)==raw


def test_normalization_never_bypasses_a1_validation():
    class Client:
        def plan(self,data):
            assert data["message"]=="Điều trị sâu răng như thế nào?"
            return {"raw_output":'{"decision":"route","targets":["A5"],"service_topics":[],"image_ids":[],"resolved_request":"x","message":null}'}
    def unused(request):raise AssertionError("Invalid A1 must not execute knowledge")
    result=create_assistant_pipeline(Client(),unused).invoke({"input":context()})
    assert result["status"]=="blocked" and result["executed_targets"]==[]
    assert result["context"]["message"]=="chữa thế nào"


@pytest.mark.parametrize("question,expected",[("Nguyên nhân sâu răng là gì?",("sâu răng","cause")),
    ("Điều trị sâu răng như thế nào?",("sâu răng","treatment")),("Sâu răng là gì?",("Sâu răng","definition"))])
def test_aspect_questions_are_not_mistaken_for_definition_subjects(question,expected):
    assert parse_request(question)==expected


def doc(key,content):
    return DentalDocument(id=key,title="Sâu răng",content=content,
        metadata={"warehouse":"internal","document_id":"same-topic-doc","source":"textbook"})


def test_definition_adds_separately_cited_cause_and_symptoms_from_same_topic():
    items=evidence_bundle([doc("intro","Sâu răng là hiện tượng răng bị phá hủy, thường được gọi là lỗ sâu. Các triệu chứng như đau, nhạy cảm thường xuất hiện muộn."),
        doc("cause","Một số loại vi khuẩn tạo ra axit có thể làm mất khoáng chất ở cấu trúc cứng của răng.")],"I","internal")
    model=ScriptedLLM()
    result=DraftingAgent(model).run("sâu răng là gì",items)
    assert len(result.claims)==3 and model.calls==0
    assert result.claims[1].citations[0].chunk_id=="cause"
    assert verification_question("sâu răng là gì",result.claims[1].text)=="Nguyên nhân sâu răng là gì?"
    assert verification_question("sâu răng là gì",result.claims[0].text)=="sâu răng là gì"


def test_treatment_is_extracted_only_from_matching_topic_sources():
    body="Điều trị bao gồm loại bỏ cấu trúc răng bị ảnh hưởng và phục hồi lại bằng các vật liệu khác nhau."
    matching=evidence_bundle([doc("treatment",body)],"I","internal")
    result=DraftingAgent(ScriptedLLM()).run("Điều trị sâu răng như thế nào?",matching)
    assert result.claims[0].text==body
    wrong=matching[0].model_copy(update={"title":"Sâu răng sữa"})
    assert not DraftingAgent(ScriptedLLM()).run("Điều trị sâu răng như thế nào?",[wrong]).claims


def test_long_mechanism_is_split_without_removing_modality():
    sentence="Một số loại vi khuẩn trong mảng bám răng có khả năng lên men carbohydrate, tạo ra axit có thể làm mất khoáng chất ở cấu trúc cứng của răng."
    assert atomic_clauses(sentence,"cause")==[
        "Một số loại vi khuẩn trong mảng bám răng có thể tạo ra axit.",
        "Axit có thể làm mất khoáng chất ở cấu trúc cứng của răng."]
    conditional="Nếu đủ điều kiện, "+sentence
    assert atomic_clauses(conditional,"cause")==[conditional]


def test_independent_retrieval_filters_subtype_before_ranking():
    broad=doc("broad","Điều trị phù hợp với chủ đề chung.")
    broad.metadata["source"]="byt"
    narrow=doc("narrow","Điều trị chỉ cho phân loại hẹp.")
    narrow.title="Sâu răng sữa";narrow.metadata["source"]="byt"
    class Pool:
        def retrieve(self,query,top_k):return [narrow,broad]
    class Ranker:
        def rerank(self,query,documents,top_k):
            assert [d.id for d in documents]==["broad"]
            return documents
    retriever=RerankedRetriever(Pool(),Ranker())
    result=retriever.retrieve_excluding("Điều trị sâu răng",sources={"msd"},topic="sâu răng")
    assert [d.id for d in result]==["broad"]


def test_scoped_multiquery_ranks_by_claim_after_topic_filter():
    broad=doc("broad","Điều trị phù hợp.");broad.metadata["source"]="byt"
    class Retriever:
        def retrieve_excluding(self,query,top_k,**options):return [broad]
        def rank_candidates(self,query,documents,top_k):
            assert query=="Một nhận định cụ thể về điều trị."
            return documents
    result,trace=retrieve_public_evidence(Retriever(),"Điều trị sâu răng như thế nào?",
        "Một nhận định cụ thể về điều trị.",sources={"msd"},topic="sâu răng")
    assert result and trace["required_topic"]=="sâu răng"


def test_multiple_chunks_of_one_document_are_one_reference_with_all_chunk_ids():
    state=supported_state();check=state["verification"]["checks"][0]
    extra=deepcopy(check["public_evidence"][0]);extra.update(evidence_id="P2",chunk_id="second-chunk")
    check["public_evidence"].append(extra);check["public_evidence_ids"].append("P2")
    check["public_quotes"].append({"evidence_id":"P2","text":state["draft"]["claims"][0]["text"]})
    check["public_citations"].append(citation_of(Evidence.model_validate(extra)).model_dump())
    check["nli_assessment"]["public"]=aggregate_predictions(nli("P1")["predictions"]+nli("P2")["predictions"])
    result=SynthesisAgent().run(state)
    assert result["status"]=="answered" and len(result["sources"])==2
    assert result["sources"][1]["chunk_ids"]==["chunk-a","second-chunk"]
    assert result["claims"][0]["reference_ids"]==["1","2"]


def test_displayed_unicode_url_keeps_original_url_in_json():
    state=supported_state();url="https://example.org/s%C3%A2u-r%C4%83ng"
    state["verification"]["checks"][0]["public_evidence"][0]["metadata"]["source_url"]=url
    result=SynthesisAgent().run(state)
    assert "https://example.org/sâu-răng" in result["message"]
    assert result["sources"][1]["url"]==url


def test_adjacent_claims_with_same_references_share_one_display_citation():
    state=supported_state("Thông tin thứ nhất. Thông tin thứ hai.")
    state["draft"]["claims"][0]["text"]="Thông tin thứ nhất."
    other=deepcopy(state["draft"]["claims"][0]);other.update(claim_id="c2",text="Thông tin thứ hai.")
    state["draft"]["claims"].append(other)
    second=deepcopy(state["verification"]["checks"][0]);second["claim_id"]="c2"
    state["verification"]["checks"].append(second)
    result=SynthesisAgent().run(state)
    assert result["status"]=="answered"
    assert "Thông tin thứ nhất. Thông tin thứ hai. [1, 2]" in result["message"]
    assert result["message"].count("[1, 2]")==1
    assert all(c["reference_ids"]==["1","2"] for c in result["claims"])
