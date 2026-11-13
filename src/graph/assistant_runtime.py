"""Four text agents share Qwen3 on GPU; retrieval and NLI run on CPU."""
import json
from pathlib import Path

from src.core.a1_process import A1ProcessClient,ROOT
from src.graph.assistant_pipeline import create_assistant_pipeline


def select_retrieval_query(question,history,accepts):
    if not accepts(question):
        raise ValueError("KNOWLEDGE_INPUT_TOO_LONG: cần rút gọn câu hỏi để truy xuất")
    from src.agents.question_context import parse_request
    if parse_request(question)[0]:
        return question, 'explicit_current_topic'
    previous=next((turn["content"] for turn in reversed(history) if turn["role"]=="user"),None)
    if previous and previous!=question:
        candidate=question+"\n"+previous
        if accepts(candidate):
            return candidate,"current_and_previous_user"
    return question,"current_user"


class LazyKnowledgeBackend:
    def __init__(self,index_dir,internal_index=None,public_index=None,candidate_k=10,loader=None):
        from src.graph.runtime import index_paths,load_runtime
        self.paths=index_paths(index_dir,internal_index,public_index)
        self.candidate_k=candidate_k
        self.loader=loader or load_runtime
        self.runtime=None
        self.catalogs=None
        self.progress_callback=None

    def __call__(self,request):
        self.warmup()
        def accepts(text):
            return (len(self.runtime.reranker.tokenizer(text,add_special_tokens=False)["input_ids"])<=128
                    and self.runtime.encoder.token_lengths(["query: "+text])[0]<=512)
        query,strategy=select_retrieval_query(request["question"],request["history"],accepts)
        # Giữ hint trong log tiếp nhận, không dùng làm câu hỏi hay bằng chứng cho A2/A3.
        conversation={"history":request["history"]} if request["history"] else None
        initial={"question":request["question"],"conversation":conversation,
                 "retrieval_query":query,"trace":[],"model_calls":[]}
        if self.progress_callback is None:
            state=self.runtime.pipeline.invoke(initial)
        else:
            state=initial.copy()
            for event in self.runtime.pipeline.stream(initial,stream_mode="updates"):
                for node,update in event.items():
                    state.update(update)
                    self.progress_callback(node,update.get("status"))
        from src.graph.pipeline_review import review_run
        state["review"]=review_run(state,*self.catalogs)
        state["models"]=self.runtime.models
        state["indices"]=self.runtime.indices
        state["retrieval_context"]={"query":query,"strategy":strategy,"history_messages":len(request["history"])}
        if not state["review"]["citation_integrity_ok"]:
            state["status"]="needs_review"
        return state

    def warmup(self):
        if self.runtime is None:
            self.runtime=self.loader(self.paths,self.candidate_k,"huggingface")
            self.catalogs=({doc.id:doc for doc in self.runtime.internal.retriever.documents},
                           {doc.id:doc for doc in self.runtime.public.retriever.documents})
        return self.runtime


def load_assistant(config_path):
    config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if {"text_models","knowledge"} <= set(config):
        bundle=AssistantBundle(config_path,config)
        return bundle.graph,config
    if set(config)!={"a1","knowledge"}:
        raise ValueError("Cấu hình assistant chỉ nhận a1 và knowledge")
    a1=config["a1"]
    client=A1ProcessClient(ROOT/a1["python"],ROOT/a1["run_dir"],
                           a1.get("timeout_seconds",180),a1.get("max_input_tokens",1536))
    knowledge=config["knowledge"]
    backend=LazyKnowledgeBackend(ROOT/knowledge["index_dir"],
        ROOT/knowledge["internal_index"] if knowledge.get("internal_index") else None,
        ROOT/knowledge["public_index"] if knowledge.get("public_index") else None,
        knowledge.get("candidate_k",10))
    return create_assistant_pipeline(client,backend),config


class AssistantBundle:
    def __init__(self,config_path,config=None):
        from src.core.shared_qwen import SharedQwenClient,AgentChatModel
        from src.agents.teammate_synthesis import TeammateSynthesisAgent
        from src.graph.runtime import load_runtime
        self.config=config or json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.on_progress=None
        if not {"text_models","knowledge"} <= set(self.config):
            raise ValueError("Web dùng cấu hình text_models và knowledge")
        self.client=SharedQwenClient(config_path)
        self.models={role:AgentChatModel(self.client,role) for role in ("A1","A2","A3","A4")}
        knowledge=self.config["knowledge"]
        def loader(paths,candidate_k,provider):
            planner = None
            if knowledge.get('bilingual_queries'):
                from src.rag.bilingual_queries import BilingualQueries, NllbTranslator
                planner = BilingualQueries(NllbTranslator(ROOT/knowledge['query_translation_dir']))
            return load_runtime(paths,candidate_k,provider,draft_llm=self.models["A2"],
                verification_llm=self.models["A3"],prefer_model=True,verification_quote_selection=True,
                query_expander=planner,context_neighbors=knowledge.get('context_neighbors',0),
                internal_min_rerank_logit=knowledge.get('internal_min_rerank_logit'),
                draft_sentence_selection=self.config['text_models'].get('a2_protocol') == 'sentence_selection_v4')
        self.backend=LazyKnowledgeBackend(ROOT/knowledge["index_dir"],
            ROOT/knowledge["internal_index"],ROOT/knowledge["public_index"],
            knowledge.get("candidate_k",10),loader=loader)
        self.backend.progress_callback=lambda node,status:self.on_progress(node,status) if self.on_progress else None
        synthesis=TeammateSynthesisAgent(self.models["A4"],lambda:self.backend.warmup().entailment)
        self.graph=create_assistant_pipeline(self.client,self.backend,synthesis_agent=synthesis)

    def warmup(self):
        self.client.ensure_started()
        self.backend.warmup()
        return self.client.identity

    def close(self):
        self.client.close()

    def analyze_image(self, path, question):
        from src.agents.local_vision import analyze_image
        return analyze_image(self.client, path, question)
