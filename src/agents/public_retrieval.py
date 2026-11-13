"""Tìm độc lập theo câu hỏi/nhận định, giữ ứng viên từ cả hai hướng tìm."""
from src.rag.source_scope import scoped_to_topic


def retrieve_public_evidence(retriever, question, claim, *, document_ids=(), sources=(), topic=None):
    # Không nối nhận định dài vào câu hỏi: từ phụ có thể lấn át chủ đề gốc.
    # Điểm reranker giữa hai truy vấn không so sánh trực tiếp được với nhau.
    queries = list(dict.fromkeys([question.strip(), claim.strip()]))
    branches = []
    for query in queries:
        if document_ids or sources:
            if not hasattr(retriever, "retrieve_excluding"):
                raise ValueError("Retriever chưa hỗ trợ đối chiếu nguồn công khai độc lập")
            options={"document_ids":document_ids,"sources":sources}
            if topic: options["topic"]=topic
            hits = retriever.retrieve_excluding(query, top_k=2, **options)
            # Kiểm tra lại bộ lọc cả khi retriever triển khai sai.
            hits = [doc for doc in hits
                    if (doc.metadata.get("document_id") or doc.metadata.get("parent_id")) not in document_ids
                    and doc.metadata.get("source") not in sources
                    and (not topic or scoped_to_topic(doc,topic))]
        else:
            hits = retriever.retrieve(query, top_k=2)
        branches.append((query, hits))
    candidates, seen = [], set()
    for _, hits in branches:
        for doc in hits:
            if doc.id not in seen:
                seen.add(doc.id)
                candidates.append(doc)
    # Cùng một câu hỏi để các điểm cuối có thể so sánh. Không ghép lại claim.
    # Bộ đoạn cố định dùng trong kiểm tra không có model xếp hạng lại.
    ranker = getattr(retriever, "rank_candidates", None)
    ranking_query=claim if topic else question
    selected = ranker(ranking_query, candidates, top_k=2) if ranker and candidates else candidates[:2]
    return selected, {
        "strategy": "question_claim_independent_v2",
        "excluded_document_ids": sorted(document_ids),
        "excluded_sources": sorted(sources),
        "branch_top_k": 2,
        "max_selected": 2,
        "selection": "rerank_by_question" if ranker else "fixed_evidence_question_first",
        "rerank_query": ranking_query,
        "required_topic":topic,
        "candidate_chunk_ids": [doc.id for doc in candidates],
        "branches": [{"query": query, "hits": [
            {"chunk_id": doc.id, "rank": rank, "score": doc.score,
             "retrieval": doc.metadata.get("retrieval", {}).copy()}
            for rank, doc in enumerate(hits, 1)]} for query, hits in branches],
        "selected_chunk_ids": [doc.id for doc in selected],
        "selected_scores": [doc.score for doc in selected],
    }
