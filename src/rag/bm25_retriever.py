"""Sparse Keyword Retriever using BM25 with Vietnamese text processing."""

import re
from typing import List
from src.rag.base import BaseRetriever, DentalDocument
from src.core.logger import get_logger

logger = get_logger("rag.bm25")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("Thư viện rank-bm25 chưa được cài đặt. Sử dụng BM25 fallback.")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/terms."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1]


class BM25Retriever(BaseRetriever):
    """Sparse keyword search based on BM25 Okapi algorithm."""

    def __init__(self, documents: List[DentalDocument] = None):
        self.documents: List[DentalDocument] = documents or []
        self.bm25 = None
        if self.documents:
            self._build_index()

    def _build_index(self) -> None:
        """Build or re-build the BM25 inverted index."""
        if not self.documents:
            return

        corpus = [
            tokenize(f"{doc.title} {doc.content} {' '.join(doc.keywords)}")
            for doc in self.documents
        ]

        if HAS_BM25:
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = corpus

    def add_documents(self, documents: List[DentalDocument]) -> None:
        """Add new documents and refresh index."""
        self.documents.extend(documents)
        self._build_index()

    def retrieve(self, query: str, top_k: int = 4) -> List[DentalDocument]:
        """Perform sparse keyword retrieval."""
        if not self.documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        if HAS_BM25 and isinstance(self.bm25, BM25Okapi):
            scores = self.bm25.get_scores(tokens)
        else:
            # Simple keyword matching fallback
            token_set = set(tokens)
            scores = []
            for doc in self.documents:
                doc_tokens = set(tokenize(f"{doc.title} {doc.content} {' '.join(doc.keywords)}"))
                match_count = len(token_set.intersection(doc_tokens))
                scores.append(float(match_count))

        scored_docs = []
        for doc, score in zip(self.documents, scores):
            if score > 0:
                doc_copy = doc.model_copy()
                doc_copy.score = float(score)
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
        return scored_docs[:top_k]
