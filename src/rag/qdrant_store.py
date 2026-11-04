"""Qdrant Vector Database Integration for Dental Knowledge Bases.

Supports both Docker Qdrant (http://localhost:6333) and embedded local disk (./data/qdrant_db)
with automatic fallback and singleton client management.
"""

import json
import logging
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.base import BaseRetriever, DentalDocument
from src.rag.local_embeddings import LocalE5Embeddings, passage_body

logger = get_logger("rag.qdrant_store")

ROOT_DIR = Path(__file__).resolve().parents[2]
COLLECTION_INTERNAL = "dental_internal"
COLLECTION_PUBLIC = "dental_public"
VECTOR_DIMENSION = 384

_CLIENT_INSTANCE: Optional[QdrantClient] = None
_ENCODER_INSTANCE: Optional[LocalE5Embeddings] = None


def is_docker_qdrant_online(url: str, timeout: float = 0.6) -> bool:
    """Kiểm tra xem Docker Qdrant server có đang hoạt động trên URL chỉ định không."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/collections",
            headers={"User-Agent": "DentalQdrantProbe"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_qdrant_client(
    url: Optional[str] = None,
    storage_path: Optional[str] = None,
    force_new: bool = False,
) -> QdrantClient:
    """Trả về singleton QdrantClient.
    
    Tự động kết nối Qdrant Docker (http://localhost:6333) nếu đang chạy;
    nếu không thì tự động fallback về lưu trữ trực tiếp trên đĩa (./data/qdrant_db).
    """
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is not None and not force_new:
        return _CLIENT_INSTANCE

    target_url = url or getattr(settings, "QDRANT_URL", "http://localhost:6333")
    target_path = storage_path or getattr(settings, "QDRANT_STORAGE_PATH", "./data/qdrant_db")

    if is_docker_qdrant_online(target_url):
        logger.info(f"[Qdrant] Kết nối thành công đến Qdrant Server (Docker): {target_url}")
        client = QdrantClient(url=target_url)
    else:
        db_dir = Path(target_path)
        if not db_dir.is_absolute():
            db_dir = ROOT_DIR / target_path
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Qdrant] Sử dụng Qdrant Local Engine trên đĩa: {db_dir}")
        client = QdrantClient(path=str(db_dir))

    _CLIENT_INSTANCE = client
    return _CLIENT_INSTANCE


def get_shared_encoder() -> LocalE5Embeddings:
    """Singleton embedding encoder để tiết kiệm RAM và thời gian khởi tạo."""
    global _ENCODER_INSTANCE
    if _ENCODER_INSTANCE is None:
        _ENCODER_INSTANCE = LocalE5Embeddings(batch_size=64, threads=8)
    return _ENCODER_INSTANCE


class QdrantDentalStore:
    """Quản lý lưu trữ và tìm kiếm vector cho Hệ thống RAG Nha khoa bằng Qdrant."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        encoder: Optional[LocalE5Embeddings] = None,
    ):
        self.client = client or get_qdrant_client()
        self.encoder = encoder or get_shared_encoder()

    def ensure_collection(
        self,
        collection_name: str,
        dimension: int = VECTOR_DIMENSION,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Tạo collection nếu chưa tồn tại."""
        try:
            if not self.client.collection_exists(collection_name):
                logger.info(f"[Qdrant] Tạo mới collection: {collection_name} (dim={dimension}, {distance})")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=distance),
                )
        except Exception as e:
            logger.error(f"[Qdrant] Lỗi khi tạo collection {collection_name}: {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """Kiểm tra collection có tồn tại hay không."""
        try:
            return self.client.collection_exists(collection_name)
        except Exception:
            return False

    def count(self, collection_name: str) -> int:
        """Đếm số lượng điểm (chunks) trong collection."""
        try:
            if not self.collection_exists(collection_name):
                return 0
            info = self.client.get_collection(collection_name)
            return info.points_count or 0
        except Exception as e:
            logger.warning(f"[Qdrant] Không thể lấy số lượng điểm của {collection_name}: {e}")
            return 0

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[DentalDocument],
        batch_size: int = 64,
        progress_callback=None,
    ) -> int:
        """Nhúng vector và nạp danh sách tài liệu vào Qdrant."""
        if not documents:
            return 0

        self.ensure_collection(collection_name)
        total = len(documents)
        logger.info(f"[Qdrant] Bắt đầu nạp {total} tài liệu vào collection '{collection_name}' (batch_size={batch_size})...")

        upserted_count = 0
        for i in range(0, total, batch_size):
            batch_docs = documents[i : i + batch_size]
            texts_to_embed = [passage_body(doc) for doc in batch_docs]

            try:
                vectors = self.encoder.embed_documents(texts_to_embed)
            except Exception as e:
                logger.error(f"[Qdrant] Lỗi tạo vector cho batch {i}-{i+len(batch_docs)}: {e}")
                raise RuntimeError(f"Embedding failed at batch {i}: {e}") from e

            points = []
            for doc, vec in zip(batch_docs, vectors):
                # Qdrant yêu cầu point_id là unsigned int hoặc UUID hợp lệ
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id))
                payload = {
                    "doc_id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "category": doc.category or "",
                    "keywords": doc.keywords or [],
                    "metadata": doc.metadata or {},
                    "source": doc.metadata.get("source", ""),
                    "warehouse": doc.metadata.get("warehouse", ""),
                    "language": doc.metadata.get("language", "vi"),
                }
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            self.client.upsert(collection_name=collection_name, points=points)
            upserted_count += len(points)
            if progress_callback:
                progress_callback(upserted_count, total)
            logger.info(f"[Qdrant] [{collection_name}] Đã nạp {upserted_count}/{total} chunks.")

        return upserted_count

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 4,
        source: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[DentalDocument]:
        """Truy vấn ngữ nghĩa vector qua Qdrant."""
        if not query.strip():
            return []

        if not self.collection_exists(collection_name):
            logger.warning(f"[Qdrant] Collection '{collection_name}' chưa tồn tại hoặc rỗng.")
            return []

        try:
            query_vector = self.encoder.embed_query(query)
        except Exception as e:
            logger.error(f"[Qdrant] Không thể tạo vector câu hỏi: {e}")
            raise RuntimeError("Embedding query failed") from e

        # Xây dựng bộ lọc payload
        filter_conditions = []
        if source is not None:
            filter_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )
        if language is not None:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
            hits = results.points
        except Exception as e:
            logger.error(f"[Qdrant] Lỗi khi thực hiện query_points trên '{collection_name}': {e}")
            return []

        retrieved_docs: List[DentalDocument] = []
        for point in hits:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata", {}))
            if collection_name == COLLECTION_INTERNAL or payload.get("warehouse") == "internal":
                metadata["warehouse"] = "internal"
            elif collection_name == COLLECTION_PUBLIC or payload.get("warehouse") == "public":
                metadata["warehouse"] = "public"

            if not metadata.get("document_id") and payload.get("document_id"):
                metadata["document_id"] = payload.get("document_id")

            doc = DentalDocument(
                id=payload.get("doc_id", str(point.id)),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                category=payload.get("category", "Chung"),
                keywords=payload.get("keywords", []),
                metadata=metadata,
                score=float(point.score) if point.score is not None else 0.0,
            )
            retrieved_docs.append(doc)

        return retrieved_docs


class QdrantDenseRetriever(BaseRetriever):
    """Dense Vector Retriever bọc ngoài QdrantDentalStore, tương thích hoàn toàn với BaseRetriever và HybridRetriever."""

    def __init__(
        self,
        collection_name: str = COLLECTION_INTERNAL,
        store: Optional[QdrantDentalStore] = None,
    ):
        self.collection_name = collection_name
        self.store = store or QdrantDentalStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        source: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[DentalDocument]:
        return self.store.search(
            collection_name=self.collection_name,
            query=query,
            top_k=top_k,
            source=source,
            language=language,
        )

    def add_documents(self, documents: List[DentalDocument]) -> None:
        self.store.upsert_documents(self.collection_name, documents)
