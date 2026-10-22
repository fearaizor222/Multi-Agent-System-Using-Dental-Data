"""BM25 trên cùng các chunk đã tạo embedding, giữ nguyên metadata nguồn."""
import math
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.base import BaseRetriever, DentalDocument
from src.rag.local_embeddings import passage_body


def tokenize(text: str) -> list[str]:
    # Giữ dấu tiếng Việt, số và từ một ký tự; không bỏ từ phủ định như "không".
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())


class _PositiveIDFBM25(BM25Okapi):
    def _calc_idf(self, document_frequencies):
        # IDF log(1 + (N-df+0.5)/(df+0.5)), cũng được Lucene dùng.
        # IDF luôn dương, kể cả khi kho chỉ có một tài liệu.
        self.idf = {term: math.log1p((self.corpus_size - count + 0.5) / (count + 0.5))
                    for term, count in document_frequencies.items()}


class BM25Retriever(BaseRetriever):
    def __init__(self, documents=None, k1=1.5, b=0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 cần k1 > 0 và 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.documents = list(documents) if documents is not None else []
        self._read_only = False
        self._build_index()

    @classmethod
    def from_disk(cls, directory):
        from src.rag.embedding_store import DenseIndex
        result = cls(DenseIndex(directory).documents)
        result._read_only = True
        return result

    def _build_index(self):
        ids = [doc.id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Trùng ID chunk trong BM25")
        corpus = [tokenize(passage_body(doc)) for doc in self.documents]
        self.bm25 = _PositiveIDFBM25(corpus, k1=self.k1, b=self.b) if any(corpus) else None

    def add_documents(self, documents):
        if self._read_only:
            raise ValueError("Index lưu trên đĩa chỉ đọc; tạo index mới khi dữ liệu đổi")
        additions = list(documents)
        ids = [doc.id for doc in self.documents + additions]
        if len(ids) != len(set(ids)):
            raise ValueError("Trùng ID chunk trong BM25")
        self.documents.extend(additions)
        self._build_index()

    def retrieve(self, query, top_k=4, source=None, language=None) -> list[DentalDocument]:
        if top_k < 1:
            raise ValueError("top_k phải dương")
        if not query.strip():
            raise ValueError("Câu hỏi rỗng")
        tokens = list(dict.fromkeys(tokenize(query)))
        if self.bm25 is None or not tokens:
            return []
        eligible = [i for i, doc in enumerate(self.documents)
                    if (source is None or doc.metadata.get("source") == source)
                    and (language is None or doc.metadata.get("language") == language)]
        if not eligible:
            return []
        scores = np.asarray(self.bm25.get_batch_scores(tokens, eligible))
        best = np.argsort(-scores, kind="stable")
        results = []
        for i in best:
            if scores[i] <= 0:
                continue
            doc = self.documents[eligible[int(i)]].model_copy(deep=True, update={"score": float(scores[i])})
            doc.metadata["retrieval"] = {"method": "bm25", "bm25_score": doc.score}
            results.append(doc)
            if len(results) == top_k:
                break
        return results
