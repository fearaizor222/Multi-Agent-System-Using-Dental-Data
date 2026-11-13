"""Nạp model một lần, dùng chung cho chạy một câu hoặc một lô câu hỏi."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.llm import get_llm
from src.graph.knowledge_pipeline import create_knowledge_pipeline
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256
from src.rag.reranked_retriever import RerankedRetriever
from src.rag.reranker import LocalReranker
from src.rag.entailment import LocalEntailment


def index_paths(index_dir, internal_index=None, public_index=None):
    paths = {"internal": Path(internal_index) if internal_index else Path(index_dir) / "internal",
             "public": Path(public_index) if public_index else Path(index_dir) / "public"}
    for name, path in paths.items():
        for filename in ("manifest.json", "records.jsonl", "vectors.npy"):
            if not (path / filename).is_file():
                raise ValueError(f"Index {name} thiếu {filename}: {path}")
    return {name: path.resolve() for name, path in paths.items()}


@dataclass
class KnowledgeRuntime:
    pipeline: Any
    internal: Any
    public: Any
    llm: Any
    entailment: Any
    reranker: Any
    encoder: Any
    indices: dict
    verification_llm: Any = None

    @property
    def models(self):
        planner=getattr(self.internal,'query_expander',None)
        return {"embedding": self.encoder.identity, "reranker": self.reranker.identity,
                "query_translation": getattr(getattr(planner,'translator',None),'identity',None),
                "llm": getattr(self.llm, "identity", {"type": self.llm._llm_type}),
                "verification_llm": getattr(self.verification_llm, "identity", None),
                "entailment": self.entailment.identity, "nli_threshold": self.entailment.threshold}


def load_runtime(paths, candidate_k=10, provider="huggingface", draft_llm=None, verification_llm=None,
                 prefer_model=False,verification_quote_selection=False,query_expander=None,context_neighbors=0,
                 internal_min_rerank_logit=None):
    if candidate_k < 3:
        raise ValueError("candidate-k phải >= 3")
    encoder, reranker = LocalE5Embeddings(), LocalReranker()
    internal = RerankedRetriever.from_disk(paths["internal"], reranker, encoder, candidate_k)
    public = RerankedRetriever.from_disk(paths["public"], reranker, encoder, candidate_k)
    internal.query_expander = query_expander
    internal.context_neighbors = context_neighbors
    internal.min_rerank_logit = internal_min_rerank_logit
    # Kiểm tra model/index trước khi sinh nội dung.
    for retriever in (internal, public):
        retriever.retriever.dense_retriever._disk_index.encoder
    llm = draft_llm if draft_llm is not None else get_llm(streaming=False, override_provider=provider)
    entailment = LocalEntailment()
    indices = {name: {"path": str(path), "sha256": {filename: file_sha256(path / filename)
               for filename in ("manifest.json", "records.jsonl", "vectors.npy")}} for name, path in paths.items()}
    graph = create_knowledge_pipeline(internal, public, llm, entailment=entailment,
        verification_llm=verification_llm, prefer_model=prefer_model,
        verification_quote_selection=verification_quote_selection)
    return KnowledgeRuntime(graph, internal, public, llm, entailment, reranker, encoder, indices, verification_llm)
