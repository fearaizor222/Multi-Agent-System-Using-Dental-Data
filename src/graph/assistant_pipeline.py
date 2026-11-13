"""A1 tiếp nhận → kiểm tra kế hoạch → A2/A3 → A4 tổng hợp nếu được phép."""
import time
from typing import TypedDict

from langgraph.graph import END,START,StateGraph

from src.data.a1_dataset import A1Input,validate_pair
from src.training.a1_evaluation import strict_json
from src.agents.synthesis import SynthesisAgent
from src.agents.question_context import prepare_routing_context


class AssistantState(TypedDict,total=False):
    input:dict
    context:dict
    status:str
    message:str|None
    message_origin:str
    a1:dict
    decision:dict
    knowledge_request:dict
    knowledge:dict
    final_answer:dict
    executed_targets:list[str]
    trace:list[dict]
    error:str
    a1_context:dict
    question_resolution:dict


def create_assistant_pipeline(a1_client,knowledge_runner=None,knowledge_available=True,synthesis_agent=None):
    capabilities={"A2":knowledge_runner is not None and knowledge_available,"A5":False,"lookup_clinic_service":False}

    def trace(state,step,status,started):
        return [*state.get("trace",[]),{"step":step,"status":status,"milliseconds":(time.perf_counter()-started)*1000}]

    def plan(state):
        start=time.perf_counter()
        context={}
        base={"executed_targets":[],"message":None,"trace":[],"decision":{},"knowledge":{},"error":"",
              "a1":{},"context":{},"knowledge_request":{},"final_answer":{}}
        try:
            # Khả năng thuộc ứng dụng, không tin capabilities do caller truyền vào.
            data={**state["input"],"capabilities":capabilities.copy()}
            context=A1Input.model_validate(data).model_dump()
            a1_context,resolution=prepare_routing_context(context)
            prediction=a1_client.plan(a1_context)
            if not isinstance(prediction,dict) or not isinstance(prediction.get("raw_output"),str):
                raise ValueError("A1 không trả raw output")
            return {**base,"context":context,"a1_context":a1_context,"question_resolution":resolution,
                    "a1":prediction,"status":"planned",
                    "trace":trace(base,"a1_plan","planned",start)}
        except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
            message=("Nội dung quá dài. Bạn rút gọn câu hỏi hoặc bắt đầu cuộc trò chuyện mới nhé."
                     if "A1_INPUT_TOO_LONG" in str(exc) else "Chưa xử lý được yêu cầu tiếp nhận. Bạn vui lòng thử lại sau.")
            return {**base,"context":context,"a1":{"diagnostics":getattr(exc,"diagnostics","")},
                    "status":"error","message":message,"message_origin":"application",
                    "error":type(exc).__name__+": "+str(exc),"trace":trace(base,"a1_plan","error",start)}

    def gate(state):
        start=time.perf_counter()
        try:
            _,decision=validate_pair(state["context"],strict_json(state["a1"]["raw_output"]))
            if decision["decision"]=="route":
                if decision["targets"]!=["A2"] or knowledge_runner is None:
                    raise ValueError("Runtime chỉ có nhánh A2; không thực thi một phần kế hoạch")
                model_context=state.get("a1_context",state["context"])
                request={"question":model_context["message"],"history":model_context["history"],
                         "routing_hint":decision["resolved_request"]}
                if model_context!=state["context"]:
                    request["original_question"]=state["context"]["message"]
                    request["original_history"]=state["context"]["history"]
                return {"decision":decision,"knowledge_request":request,"status":"route_a2",
                        "trace":trace(state,"a1_validate","accepted",start)}
            message=decision["message"]
            origin="a1"
            if decision["decision"]=="handoff":
                message="Bạn vui lòng liên hệ trực tiếp người phụ trách phòng khám để được tiếp nhận yêu cầu."
                origin="application_policy"
            return {"decision":decision,"status":decision["decision"],"message":message,"message_origin":origin,
                    "trace":trace(state,"a1_validate","accepted",start)}
        except (ValueError,TypeError,KeyError) as exc:
            return {"status":"blocked","message":"Kế hoạch tiếp nhận chưa hợp lệ. Bạn vui lòng diễn đạt lại yêu cầu.",
                    "message_origin":"application","error":str(exc),
                    "trace":trace(state,"a1_validate","blocked",start)}

    def knowledge(state):
        start=time.perf_counter()
        try:
            result=knowledge_runner(state["knowledge_request"])
            if not isinstance(result,dict) or not isinstance(result.get("status"),str):
                raise ValueError("Luồng tri thức không trả trạng thái hợp lệ")
            return {"status":"error" if result["status"]=="error" else "knowledge_result",
                    "message":("Chưa hoàn tất xử lý kiến thức. Bạn vui lòng thử lại sau." if result["status"]=="error" else None),
                    "knowledge":result,"executed_targets":["A2"],
                    "trace":[*state["trace"],*result.get("trace",[])]}
        except (ValueError,RuntimeError,OSError,KeyError,TypeError,IndexError) as exc:
            message=("Câu hỏi quá dài để tìm tài liệu. Bạn rút gọn câu hỏi nhé."
                     if "KNOWLEDGE_INPUT_TOO_LONG" in str(exc) else "Chưa hoàn tất xử lý kiến thức. Bạn vui lòng thử lại sau.")
            return {"status":"error","message":message,
                    "message_origin":"application","executed_targets":["A2"],"error":type(exc).__name__+": "+str(exc),
                    "trace":trace(state,"a2_runtime","error",start)}

    def synthesize(state):
        started=time.perf_counter()
        final=(synthesis_agent if synthesis_agent is not None else SynthesisAgent()).run(state["knowledge"])
        return {"final_answer":final,"message":final["message"],
                "message_origin":"a4_model" if final.get("model_used") else "a4_policy",
                "trace":trace(state,"a4_synthesize",final["status"],started)}

    graph=StateGraph(AssistantState)
    graph.add_node("a1_plan",plan)
    graph.add_node("a1_validate",gate)
    graph.add_node("a2_a3",knowledge)
    graph.add_node("a4_synthesize",synthesize)
    graph.add_edge(START,"a1_plan")
    graph.add_conditional_edges("a1_plan",lambda s:"a1_validate" if s["status"]=="planned" else END)
    graph.add_conditional_edges("a1_validate",lambda s:"a2_a3" if s["status"]=="route_a2" else END)
    graph.add_conditional_edges("a2_a3",lambda s:"a4_synthesize" if s["status"]=="knowledge_result" else END)
    graph.add_edge("a4_synthesize",END)
    return graph.compile()
