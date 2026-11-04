"""Nạp toàn bộ Corpus chuẩn hóa vào Qdrant Vector Database.

Hỗ trợ nạp Kho Nội bộ (internal_v1: 1,392 chunks) và Kho Công khai (public_v1: BYT, MSD, PMC - 23,411 chunks).
Có tính năng Resume / Bỏ qua các chunk đã nạp để tiết kiệm thời gian.
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Set

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.core.logger import get_logger
from src.rag.embedding_store import load_corpus
from src.rag.qdrant_store import (
    COLLECTION_INTERNAL,
    COLLECTION_PUBLIC,
    QdrantDentalStore,
    get_qdrant_client,
    get_shared_encoder,
)

logger = get_logger("scripts.import_qdrant")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse",
        choices=["internal", "public", "both"],
        default="both",
        help="Kho dữ liệu cần nạp vào Qdrant (mặc định: both)",
    )
    parser.add_argument(
        "--public-sources",
        choices=["byt_msd", "all", "byt", "msd", "pmc"],
        default="byt_msd",
        help="Nguồn dữ liệu công khai cần nạp: byt_msd (4,290 chunks), all (23,411 chunks gồm cả PMC), pmc (19,121 chunks)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Kích thước batch khi tạo embedding và upsert (mặc định: 64)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa collection cũ và tạo lại từ đầu nếu đã tồn tại",
    )
    return parser.parse_args()


def get_all_existing_ids(client, collection_name: str) -> Set[str]:
    """Lấy danh sách toàn bộ point IDs đã tồn tại trong collection để hỗ trợ resume."""
    if not client.collection_exists(collection_name):
        return set()
    existing_ids = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            offset=next_offset,
            limit=2000,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            existing_ids.add(str(p.id))
        if next_offset is None or not points:
            break
    return existing_ids


def import_internal(store: QdrantDentalStore, batch_size: int, recreate: bool = False):
    print("\n" + "=" * 60)
    print(">>> 1. NẠP KHO NỘI BỘ (INTERNAL_V1) VÀO QDRANT...")
    print("=" * 60)

    if recreate and store.client.collection_exists(COLLECTION_INTERNAL):
        print(f"[*] Xóa collection cũ '{COLLECTION_INTERNAL}'...")
        store.client.delete_collection(COLLECTION_INTERNAL)

    store.ensure_collection(COLLECTION_INTERNAL)
    existing_count = store.count(COLLECTION_INTERNAL)
    print(f"[*] Trạng thái hiện tại của '{COLLECTION_INTERNAL}': {existing_count} chunks.")

    t0 = time.time()
    docs, sources = load_corpus("internal")
    print(f"[*] Đã tải {len(docs)} chunks từ internal_v1 ({len(sources)} nguồn tài liệu).")

    if not recreate and existing_count > 0:
        existing_ids = get_all_existing_ids(store.client, COLLECTION_INTERNAL)
        missing_docs = [
            d for d in docs
            if str(uuid.uuid5(uuid.NAMESPACE_DNS, d.id)) not in existing_ids
        ]
        if not missing_docs:
            print(f"[✓] Collection '{COLLECTION_INTERNAL}' đã đầy đủ ({existing_count} chunks). Bỏ qua upsert.")
            return existing_count
        print(f"[*] Đã có {len(existing_ids)} chunks, cần nạp thêm {len(missing_docs)} chunks mới...")
        docs = missing_docs

    upserted = store.upsert_documents(COLLECTION_INTERNAL, docs, batch_size=batch_size)
    elapsed = time.time() - t0
    final_count = store.count(COLLECTION_INTERNAL)
    print(f"[✓] Hoàn thành nạp Kho Nội bộ: {final_count} chunks (thêm {upserted} chunks trong {elapsed:.1f}s).")
    return final_count


def import_public(
    store: QdrantDentalStore,
    batch_size: int,
    source_filter: str = "byt_msd",
    recreate: bool = False,
):
    print("\n" + "=" * 60)
    print(f">>> 2. NẠP KHO CÔNG KHAI (PUBLIC_V1, filter={source_filter}) VÀO QDRANT...")
    print("=" * 60)

    if recreate and store.client.collection_exists(COLLECTION_PUBLIC):
        print(f"[*] Xóa collection cũ '{COLLECTION_PUBLIC}'...")
        store.client.delete_collection(COLLECTION_PUBLIC)

    store.ensure_collection(COLLECTION_PUBLIC)
    existing_count = store.count(COLLECTION_PUBLIC)
    print(f"[*] Trạng thái hiện tại của '{COLLECTION_PUBLIC}': {existing_count} chunks.")

    t0 = time.time()
    docs, sources = load_corpus("public")
    print(f"[*] Đã tải tổng cộng {len(docs)} chunks từ public_v1.")

    # Lọc nguồn công khai theo cấu hình
    if source_filter == "byt_msd":
        selected_docs = [d for d in docs if d.metadata.get("source") in ("byt", "msd")]
    elif source_filter == "byt":
        selected_docs = [d for d in docs if d.metadata.get("source") == "byt"]
    elif source_filter == "msd":
        selected_docs = [d for d in docs if d.metadata.get("source") == "msd"]
    elif source_filter == "pmc":
        selected_docs = [d for d in docs if d.metadata.get("source") == "pmc"]
    else:
        selected_docs = docs

    print(f"[*] Số chunks chọn nạp theo bộ lọc '{source_filter}': {len(selected_docs)} chunks.")

    if not recreate and existing_count > 0:
        print("[*] Đang kiểm tra các chunks đã có trong Qdrant để bỏ qua...")
        existing_ids = get_all_existing_ids(store.client, COLLECTION_PUBLIC)
        missing_docs = [
            d for d in selected_docs
            if str(uuid.uuid5(uuid.NAMESPACE_DNS, d.id)) not in existing_ids
        ]
        if not missing_docs:
            print(f"[✓] Collection '{COLLECTION_PUBLIC}' đã có đầy đủ {len(selected_docs)} chunks theo bộ lọc. Bỏ qua.")
            return existing_count
        print(f"[*] Đã có {len(existing_ids)} chunks, cần nạp thêm {len(missing_docs)} chunks mới...")
        selected_docs = missing_docs

    upserted = store.upsert_documents(COLLECTION_PUBLIC, selected_docs, batch_size=batch_size)
    elapsed = time.time() - t0
    final_count = store.count(COLLECTION_PUBLIC)
    print(f"[✓] Hoàn thành nạp Kho Công khai: {final_count} chunks (thêm {upserted} chunks trong {elapsed:.1f}s).")
    return final_count


def main():
    args = parse_args()
    print("============================================================")
    print("   KHỞI ĐỘNG CÔNG CỤ NẠP DỮ LIỆU CORPUS VÀO QDRANT")
    print("============================================================")

    client = get_qdrant_client()
    encoder = get_shared_encoder()
    store = QdrantDentalStore(client=client, encoder=encoder)

    if args.warehouse in ("internal", "both"):
        import_internal(store, batch_size=args.batch_size, recreate=args.recreate)

    if args.warehouse in ("public", "both"):
        import_public(
            store,
            batch_size=args.batch_size,
            source_filter=args.public_sources,
            recreate=args.recreate,
        )

    print("\n" + "=" * 60)
    print("[✓] TOÀN BỘ QUÁ TRÌNH NẠP VECTOR DATABASE QDRANT ĐÃ HOÀN TẤT!")
    print(f"    - Collection Internal: {store.count(COLLECTION_INTERNAL)} chunks")
    print(f"    - Collection Public  : {store.count(COLLECTION_PUBLIC)} chunks")
    print("============================================================\n")


if __name__ == "__main__":
    main()
