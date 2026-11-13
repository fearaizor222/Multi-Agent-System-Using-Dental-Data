"""Dùng chung một reranker cho hai kho; giữ nguyên bộ lọc của HybridRetriever."""
from src.rag.base import BaseRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.source_scope import scoped_to_topic


class RerankedRetriever(BaseRetriever):
    def __init__(self, retriever, reranker, candidate_k=30, query_expander=None):
        if candidate_k < 1:
            raise ValueError("candidate_k phải dương")
        self.retriever, self.reranker, self.candidate_k = retriever, reranker, candidate_k
        self.query_expander = query_expander
        self.last_run = {}
        self.context_neighbors = 0
        self.min_rerank_logit = None

    def _filter_relevance(self, documents):
        if self.min_rerank_logit is None:
            return documents
        kept, dropped = [], []
        for doc in documents:
            logit = doc.metadata.get('retrieval',{}).get('rerank_logit',doc.score)
            if logit >= self.min_rerank_logit:
                kept.append(doc)
            else:
                dropped.append({'chunk_id':doc.id,'rerank_logit':logit})
        self.last_run.setdefault('relevance_filter',{'minimum_logit':self.min_rerank_logit,'dropped':[]})['dropped'].extend(dropped)
        return kept

    def expand_context(self, hits):
        if not self.context_neighbors:
            return hits
        from src.rag.adjacent_context import expand_adjacent
        expanded = expand_adjacent(self.retriever.documents, hits, self.context_neighbors)
        self.last_run['context_ids'] = [d.id for d in expanded]
        return expanded

    @classmethod
    def from_disk(cls, directory, reranker, encoder=None, candidate_k=30):
        return cls(HybridRetriever.from_disk(directory, encoder=encoder, candidate_k=candidate_k), reranker, candidate_k)

    def add_documents(self, documents):
        raise ValueError("Index lưu trên đĩa chỉ đọc")

    def retrieve(self, query, top_k=5, source=None, language=None):
        if top_k < 1 or top_k > self.candidate_k:
            raise ValueError("Cần 1 <= top_k <= candidate_k")
        if self.query_expander is not None:
            return self._bilingual(query, top_k, self.candidate_k, source=source, language=language)
        documents = self.retriever.retrieve(query, top_k=self.candidate_k, source=source, language=language)
        return self._filter_relevance(self.reranker.rerank(query, documents, top_k=top_k))

    def _variants(self, query):
        expanded = self.query_expander(query)
        queries, seen, errors = [], set(), []
        for variant in expanded['queries'][:2]:
            text = variant['text'].strip()
            if text in seen:
                continue
            if len(self.reranker.tokenizer(text, add_special_tokens=False)['input_ids']) > 128:
                errors.append('Bỏ truy vấn dịch vượt 128 token reranker')
                continue
            seen.add(text)
            queries.append({'text': text, 'language': variant['language']})
        if not queries or queries[0]['text'] != query.strip():
            from src.rag.reranker import RetrievalQueryTooLong
            raise RetrievalQueryTooLong('Truy vấn gốc vượt giới hạn reranker')
        self.last_run = {'query_expansion': expanded, 'translation_warnings': errors}
        return queries

    def _rank_languages(self, queries, documents, top_k):
        """Rank each document with a query in its language; never compare cross-query logits."""
        groups = [[] for _ in queries]
        for doc in documents:
            lang = doc.metadata.get('language')
            branch = next((i for i,q in enumerate(queries) if q['language'] == lang), 0)
            groups[branch].append(doc)
        combined = []
        for branch, docs in enumerate(groups):
            if not docs:
                continue
            previous = {d.id: d.metadata.get('retrieval', {}).copy() for d in docs}
            ranked = self._filter_relevance(self.reranker.rerank(queries[branch]['text'], docs, top_k=len(docs)))
            for rank, doc in enumerate(ranked, 1):
                # RRF here fuses within-language ranks; logits are diagnostic values only.
                score = 1. / (60 + rank)
                doc.metadata['retrieval'].update({k:v for k,v in previous[doc.id].items() if k in ('rrf_rank','rrf_score')})
                doc.metadata['retrieval'].update(method='bilingual_hybrid_rerank_v1',
                    query_variants=queries, rerank_query=queries[branch]['text'], language_rank=rank,
                    candidate_k_per_query=self.candidate_k, language_rank_score=score)
                doc.score = score
                combined.append((rank, branch, doc))
        combined.sort(key=lambda item: (item[0], item[1], item[2].id))
        return [doc for _,_,doc in combined[:top_k]]

    def _bilingual(self, query, top_k, candidate_k, *, source=None, language=None,
                   document_ids=(), sources=(), topic=None):
        queries = self._variants(query)
        documents, seen, branches = [], set(), []
        for variant in queries:
            hits = self.retriever.retrieve(variant['text'], top_k=candidate_k, source=source, language=language)
            hits = [doc for doc in hits
                    if (doc.metadata.get('document_id') or doc.metadata.get('parent_id')) not in document_ids
                    and doc.metadata.get('source') not in sources
                    and (not topic or scoped_to_topic(doc,topic))]
            branches.append({'query':variant, 'candidate_ids':[d.id for d in hits]})
            for doc in hits:
                if doc.id not in seen:
                    seen.add(doc.id)
                    documents.append(doc)
        selected = self._rank_languages(queries, documents, top_k)
        self.last_run.update(strategy='bilingual_hybrid_rerank_v1', branches=branches,
            candidate_count=len(documents), selected_ids=[d.id for d in selected])
        return selected

    def rank_candidates(self, query, documents, top_k=2):
        """Xếp hạng tập ứng viên nhiều truy vấn theo cùng một câu hỏi."""
        if top_k < 1:
            raise ValueError('top_k phải dương')
        if self.query_expander is not None:
            return self._rank_languages(self._variants(query), documents, top_k)
        previous = {doc.id: doc.metadata.get("retrieval", {}).copy() for doc in documents}
        ranked = self.reranker.rerank(query, documents, top_k=top_k)
        for doc in ranked:
            # Đầu vào đã rerank: không ghi nhầm logit cũ thành điểm RRF.
            for key in ("rrf_score", "rrf_rank"):
                if key in previous[doc.id]:
                    doc.metadata["retrieval"][key] = previous[doc.id][key]
            doc.metadata["retrieval"]["multi_query_rerank"] = True
        return ranked

    def retrieve_excluding(self, query, top_k=2, *, document_ids=(), sources=(), topic=None):
        # Tập tìm mở rộng có giới hạn. Lọc trước rerank, không lấy chính nguồn
        # soạn nháp (hoặc chunk khác của cùng nhà cung cấp) để tự kiểm chứng.
        if top_k < 1:
            raise ValueError('top_k phải dương')
        if self.query_expander is not None:
            return self._bilingual(query, top_k, max(50,self.candidate_k),
                document_ids=document_ids, sources=sources, topic=topic)
        documents = self.retriever.retrieve(query, top_k=max(50, self.candidate_k))
        documents = [doc for doc in documents
                     if (doc.metadata.get("document_id") or doc.metadata.get("parent_id")) not in document_ids
                     and doc.metadata.get("source") not in sources
                     and (not topic or scoped_to_topic(doc,topic))]
        return self.reranker.rerank(query, documents, top_k=top_k)
