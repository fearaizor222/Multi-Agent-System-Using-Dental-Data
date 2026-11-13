"""Kiểm tra luồng và chặn thực thi; phản hồi A1 trong test là dữ liệu dựng sẵn."""
import json

import pytest

from src.graph.assistant_pipeline import create_assistant_pipeline
from src.graph.assistant_runtime import select_retrieval_query,LazyKnowledgeBackend
from tests.test_knowledge_pipeline import ScriptedLLM,FixedRetriever,doc,PUBLIC_QUOTE
from src.graph.knowledge_pipeline import create_knowledge_pipeline


def decision(kind="route",targets=None):
    return {"decision":kind,"targets":(["A2"] if targets is None else targets) if kind=="route" else [],
            "service_topics":[],"image_ids":[],"resolved_request":"Gợi ý có thể viết sai thuật ngữ" if kind=="route" else None,
            "message":None if kind=="route" else "Thông báo thử nghiệm"}


class ScriptedA1:
    def __init__(self,raw):
        self.raw=raw
        self.inputs=[]
    def plan(self,context):
        self.inputs.append(context)
        return {"raw_output":self.raw,"identity":{"type":"test_double"}}


class Backend:
    def __init__(self):
        self.requests=[]
    def __call__(self,request):
        self.requests.append(request)
        return {"status":"needs_review","trace":[{"step":"a2_retrieve","status":"retrieved","milliseconds":1},
                {"step":"a2_draft","status":"draft","milliseconds":1},{"step":"a3_verify","status":"needs_review","milliseconds":1}]}


def test_route_preserves_original_and_history_instead_of_replacing_with_hint():
    a1=ScriptedA1(json.dumps(decision()))
    backend=Backend()
    history=[{"role":"user","content":"Tôi đang tìm hiểu về enamel."},{"role":"assistant","content":"Bạn muốn tài liệu nhập môn?"}]
    result=create_assistant_pipeline(a1,backend).invoke({"input":{"message":"Đúng, tìm tài liệu giúp mình.","history":history}})
    assert result["status"]=="knowledge_result" and result["executed_targets"]==["A2"]
    assert backend.requests==[{"question":"Đúng, tìm tài liệu giúp mình.","history":history,
                               "routing_hint":"Gợi ý có thể viết sai thuật ngữ"}]
    assert [t["step"] for t in result["trace"]]==["a1_plan","a1_validate","a2_retrieve","a2_draft","a3_verify","a4_synthesize"]
    assert result["final_answer"]["status"]=="insufficient_evidence"


@pytest.mark.parametrize("kind",["clarify","handoff","respond","out_of_scope"])
def test_non_route_decisions_never_initialize_knowledge(kind):
    backend=Backend()
    result=create_assistant_pipeline(ScriptedA1(json.dumps(decision(kind))),backend).invoke({"input":{"message":"Câu thử"}})
    assert result["status"]==kind and result["executed_targets"]==[] and backend.requests==[]
    assert result["final_answer"]=={} and not any(t["step"]=="a4_synthesize" for t in result["trace"])
    if kind=="handoff":
        assert result["message_origin"]=="application_policy"
        assert "liên hệ trực tiếp" in result["message"]


@pytest.mark.parametrize("raw",["not json",'```json\n{}\n```','{"decision":"route","decision":"respond"}',
                               json.dumps({**decision(),"targets":["A3"]})])
def test_invalid_plans_stop_before_any_backend_call(raw):
    backend=Backend()
    result=create_assistant_pipeline(ScriptedA1(raw),backend).invoke({"input":{"message":"Câu thử"}})
    assert result["status"]=="blocked" and result["executed_targets"]==[] and backend.requests==[]


def test_forged_capabilities_cannot_enable_a5_or_partially_execute_route():
    payload={**decision(),"targets":["A5","A2"],"image_ids":["img"]}
    a1=ScriptedA1(json.dumps(payload));backend=Backend()
    context={"message":"Đọc ảnh rồi tìm kiến thức.","capabilities":{"A2":True,"A5":True,"lookup_clinic_service":True},
             "attachments":[{"id":"img","filename":"a.png","kind":"document","readable":True}]}
    result=create_assistant_pipeline(a1,backend).invoke({"input":context})
    assert a1.inputs[0]["capabilities"]=={"A2":True,"A5":False,"lookup_clinic_service":False}
    assert result["status"]=="blocked" and backend.requests==[]


def test_service_target_is_blocked_even_when_caller_claims_it_is_available():
    payload={**decision(),"targets":["lookup_clinic_service"],"service_topics":["pricing"]}
    backend=Backend()
    result=create_assistant_pipeline(ScriptedA1(json.dumps(payload)),backend).invoke({"input":{"message":"Hỏi giá",
            "capabilities":{"A2":True,"A5":True,"lookup_clinic_service":True}}})
    assert result["status"]=="blocked" and backend.requests==[]


def test_missing_backend_makes_a2_unavailable():
    a1=ScriptedA1(json.dumps(decision()))
    result=create_assistant_pipeline(a1).invoke({"input":{"message":"Tìm kiến thức"}})
    assert a1.inputs[0]["capabilities"]["A2"] is False
    assert result["status"]=="blocked" and result["executed_targets"]==[]


def test_worker_failure_does_not_fall_back_to_unrouted_knowledge():
    class Broken:
        def plan(self,context):
            raise RuntimeError("worker timeout")
    backend=Backend()
    result=create_assistant_pipeline(Broken(),backend).invoke({"input":{"message":"Câu thử"}})
    assert result["status"]=="error" and backend.requests==[]


def test_long_request_errors_are_explained_and_not_silently_truncated():
    class Long:
        def plan(self,context):
            raise RuntimeError("A1_INPUT_TOO_LONG: limit")
    result=create_assistant_pipeline(Long(),Backend()).invoke({"input":{"message":"Câu dài"}})
    assert result["status"]=="error" and "rút gọn" in result["message"]


def test_backend_failure_is_explicit_without_synthetic_answer():
    def broken(request):
        raise RuntimeError("model unavailable")
    result=create_assistant_pipeline(ScriptedA1(json.dumps(decision())),broken).invoke({"input":{"message":"Câu thử"}})
    assert result["status"]=="error" and result["knowledge"]=={}
    assert result["executed_targets"]==["A2"] and "model unavailable" in result["error"]


def test_retrieval_uses_original_and_latest_user_not_assistant_or_hint():
    history=[{"role":"user","content":"Chủ đề trước."},{"role":"assistant","content":"Không dùng tôi làm truy vấn."}]
    query,strategy=select_retrieval_query("Câu hiện tại.",history,lambda q:len(q)<100)
    assert query=="Câu hiện tại.\nChủ đề trước." and strategy=="current_and_previous_user"
    query,strategy=select_retrieval_query("Câu hiện tại.",history,lambda q:len(q)<20)
    assert query=="Câu hiện tại." and strategy=="current_user"
    with pytest.raises(ValueError,match="KNOWLEDGE_INPUT_TOO_LONG"):
        select_retrieval_query("Câu dài",[],lambda q:False)


def test_a2_and_a3_receive_original_question_and_structured_conversation():
    class CapturingLLM(ScriptedLLM):
        def __init__(self,*responses):
            super().__init__(*responses)
            self.payloads=[]
        def invoke(self,messages):
            self.payloads.append(json.loads(messages[-1].content.split("\nJSON schema:")[0]))
            return super().invoke(messages)
    llm=CapturingLLM({"status":"draft","claims":[{"text":"Khí cụ bằng nhựa.","evidence_ids":["I1"]}]},
        {"draft_support":"supported","public_support":"supported","public_evidence_ids":["P1"],"public_quotes":[PUBLIC_QUOTE]})
    internal=FixedRetriever([doc()]);public=FixedRetriever([doc("public-1","public")])
    conversation={"history":[{"role":"user","content":"Câu cũ"}],"routing_hint":"Gợi ý sai"}
    result=create_knowledge_pipeline(internal,public,llm).invoke({"question":"Câu hỏi gốc", "retrieval_query":"Truy vấn có ngữ cảnh", "conversation":conversation})
    assert result["status"]=="supported"
    assert internal.calls[0][0]=="Truy vấn có ngữ cảnh"
    assert len(llm.payloads)==2
    assert all(p["question"]=="Câu hỏi gốc" and p["conversation"]==conversation for p in llm.payloads)


def test_lazy_backend_loads_once_and_keeps_bad_routing_hint_out_of_llm_input(tmp_path):
    from types import SimpleNamespace
    for warehouse in ("internal","public"):
        folder=tmp_path/warehouse;folder.mkdir()
        for name in ("manifest.json","records.jsonl","vectors.npy"):
            (folder/name).touch()
    calls=[];states=[]
    class Pipeline:
        def invoke(self,state):
            states.append(state)
            return {**state,"status":"insufficient_evidence","internal_evidence":[]}
    runtime=SimpleNamespace(pipeline=Pipeline(),internal=SimpleNamespace(retriever=SimpleNamespace(documents=[])),
        public=SimpleNamespace(retriever=SimpleNamespace(documents=[])),
        reranker=SimpleNamespace(tokenizer=lambda text,**kw:{"input_ids":text.split()}),
        encoder=SimpleNamespace(token_lengths=lambda texts:[len(t.split()) for t in texts]),models={},indices={})
    def loader(*args):
        calls.append(args)
        return runtime
    backend=LazyKnowledgeBackend(tmp_path,loader=loader)
    assert calls==[]
    request={"question":"Câu gốc", "history":[],"routing_hint":"Gợi ý bị viết sai"}
    backend(request)
    backend({**request,"history":[{"role":"user","content":"Ngữ cảnh cũ"}]})
    assert len(calls)==1 and len(states)==2
    assert states[0]["conversation"] is None
    assert states[1]["conversation"]=={"history":[{"role":"user","content":"Ngữ cảnh cũ"}]}
    assert all("Gợi ý bị viết sai" not in str(state) for state in states)
