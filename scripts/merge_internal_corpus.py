"""Script hợp nhất toàn bộ chunks và documents nội bộ vào chung một file duy nhất:
- data/processed/internal_v1/corpus/chunks.jsonl (RHM1 + Oral Health & Aging + Dental Public Health)
- data/processed/internal_v1/corpus/documents.jsonl (Tất cả Master Documents)
- Giữ rhm_documents.jsonl để tương thích ngược
- Dọn dẹp các file chunk riêng lẻ theo yêu cầu người dùng
- Cập nhật manifest.json tổng hợp toàn bộ kho nội bộ
"""

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CORPUS_DIR = ROOT / "data/processed/internal_v1/corpus"

from src.data.public_chunks import PublicChunk
from src.data.public_schema import PublicDocument


def main():
    print("=" * 80)
    print("HỢP NHẤT TOÀN BỘ CHUNKS VÀ DOCUMENTS NỘI BỘ VÀO CHUNG CHUNKS.JSONL VÀ DOCUMENTS.JSONL")
    print("=" * 80)

    # 1. HỢP NHẤT DOCUMENTS
    doc_sources = [
        CORPUS_DIR / "rhm_documents.jsonl",
        CORPUS_DIR / "oral_health_aging_documents.jsonl",
        CORPUS_DIR / "dental_public_health_documents.jsonl"
    ]

    all_docs = {}
    for src in doc_sources:
        if not src.exists():
            print(f"Bỏ qua (chưa tồn tại): {src.name}")
            continue
        count = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc = PublicDocument.model_validate_json(line)
            all_docs[doc.document_id] = doc
            count += 1
        print(f" -> Đã đọc {count:2d} documents từ {src.name}")

    unified_docs_file = CORPUS_DIR / "documents.jsonl"
    with open(unified_docs_file, "w", encoding="utf-8") as f:
        for doc in all_docs.values():
            f.write(doc.model_dump_json() + "\n")
    print(f"\n✅ Đã lưu {len(all_docs)} documents hợp nhất vào: {unified_docs_file.name}")

    # 2. HỢP NHẤT CHUNKS
    chunk_sources = [
        ("rhm1", CORPUS_DIR / "chunks.jsonl"),  # chứa RHM1 ban đầu
        ("nidcr", CORPUS_DIR / "oral_health_aging_chunks.jsonl"),
        ("dph", CORPUS_DIR / "dental_public_health_chunks.jsonl")
    ]

    all_chunks = {}
    source_stats = Counter()

    for name, src in chunk_sources:
        if not src.exists():
            print(f"Bỏ qua chunk source (chưa tồn tại): {src.name}")
            continue
        count = 0
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = PublicChunk.model_validate_json(line)
            # Chỉ lấy các chunk thuộc document hợp lệ
            if chunk.document_id in all_docs:
                all_chunks[chunk.chunk_id] = chunk
                source_stats[chunk.source] += 1
                count += 1
        print(f" -> Đã đọc {count:4d} chunks từ {src.name}")

    unified_chunks_file = CORPUS_DIR / "chunks.jsonl"
    with open(unified_chunks_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks.values():
            f.write(chunk.model_dump_json() + "\n")
    print(f"\n✅ Đã lưu {len(all_chunks)} chunks hợp nhất vào file duy nhất: {unified_chunks_file.name}")

    # 3. DỌN DẸP CÁC FILE CHUNK RIÊNG LẺ
    separate_chunk_files = [
        CORPUS_DIR / "oral_health_aging_chunks.jsonl",
        CORPUS_DIR / "dental_public_health_chunks.jsonl"
    ]
    for scf in separate_chunk_files:
        if scf.exists():
            scf.unlink()
            print(f" -> Đã xóa file chunk riêng lẻ: {scf.name} (đã gộp vào chunks.jsonl)")

    # 4. TẠO MANIFEST TỔNG HỢP TOÀN BỘ KHO NỘI BỘ INTERNAL_V1
    token_counts = [c.token_count for c in all_chunks.values()]
    manifest = {
        "dataset": "internal_v1",
        "description": "Kho ngữ liệu nội bộ hợp nhất (RHM1 Cắn khớp học + NIDCR Oral Health & Aging + Dental Public Health & Research)",
        "total_documents": len(all_docs),
        "total_chunks": len(all_chunks),
        "source_breakdown": dict(source_stats),
        "token_statistics": {
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0
        },
        "files": {
            "documents": "data/processed/internal_v1/corpus/documents.jsonl",
            "chunks": "data/processed/internal_v1/corpus/chunks.jsonl",
            "rhm_documents_compat": "data/processed/internal_v1/corpus/rhm_documents.jsonl"
        }
    }

    manifest_file = CORPUS_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Đã cập nhật manifest hợp nhất: {manifest_file.name}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
