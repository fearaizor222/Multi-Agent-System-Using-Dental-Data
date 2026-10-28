"""Script xử lý tài liệu NIDCR Oral Health & Aging theo đúng chuẩn internal_v1 (giáo trình RHM1).
Thực hiện:
1. Bóc tách 10 hình ảnh lâm sàng, kỹ thuật, dụng cụ hỗ trợ chất lượng cao vào data/images/oral_health_aging/.
2. Đọc trực tiếp lớp chữ nguyên bản từ PDF gốc.
3. Tạo file Master document: data/processed/oral_health_aging_chapters.json (tương tự khop_can_hoc_chapters.json).
4. Chunking trượt theo token (max_tokens=380, overlap_words=16, tokenizer=intfloat/multilingual-e5-small)
   xuất ra file data/processed/oral_health_aging_sections.json (chuẩn khop_can_hoc_sections.json)
   kèm liên kết figures_data, page_start, page_end, parent_id.
5. Xuất các bản ghi corpus internal_v1 (oral_health_aging_documents.jsonl, oral_health_aging_chunks.jsonl).
"""

import argparse
import json
import sys
from pathlib import Path

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data.sources.oral_health_aging import (
    DOC_ID,
    FULL_ID,
    MODEL_NAME,
    build_internal_corpus_records,
    build_master_document,
    chunk_document_internal,
    extract_figures,
    read_pdf_pages,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "data/raw/oral-health-aging-brushing.pdf",
        help="Đường dẫn file PDF gốc cần xử lý"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "data/images/oral_health_aging",
        help="Thư mục xuất ảnh minh họa bóc tách"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed",
        help="Thư mục xuất các file JSON processed (chapters & sections)"
    )
    parser.add_argument(
        "--internal-corpus-dir",
        type=Path,
        default=ROOT / "data/processed/internal_v1/corpus",
        help="Thư mục xuất corpus internal_v1 (jsonl)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=380,
        help="Giới hạn token tối đa cho mỗi chunk (mặc định: 380)"
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=16,
        help="Số từ gối đầu (overlap) khi chia đoạn (mặc định: 16)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHUẨN HÓA DỮ LIỆU ORAL HEALTH & AGING THEO ĐÚNG CHUẨN INTERNAL_V1")
    print("=" * 80)

    if not args.pdf.exists():
        print(f"LỖI: Không tìm thấy file PDF tại {args.pdf}", file=sys.stderr)
        return 1

    # 1. BÓC TÁCH HÌNH ẢNH
    print("\n[BƯỚC 1/5] Đang bóc tách 10 hình ảnh minh họa lâm sàng & kỹ thuật...")
    all_figures = extract_figures(args.pdf, args.images_dir)
    print(f" -> Đã trích xuất {len(all_figures)} hình ảnh chất lượng cao vào: {args.images_dir}")
    for item in all_figures:
        print(f"    * [Trang {item['page']}] {item['label']} ({item['dimensions']}, {item['filesize_bytes']} bytes)")
        print(f"      File: {item['local_path']}")

    img_catalog_path = args.images_dir / "images_catalog.json"
    img_catalog_path.write_text(json.dumps(all_figures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" -> Đã lưu manifest hình ảnh: {img_catalog_path}")

    # 2. ĐỌC LỚP CHỮ GỐC TỪ PDF
    print("\n[BƯỚC 2/5] Đang trích xuất trực tiếp lớp chữ từ các trang PDF gốc...")
    pages, page_spans, source_sha = read_pdf_pages(args.pdf)
    total_chars = sum(len(txt) for txt in pages.values())
    print(f" -> Đã đọc 4 trang PDF ({total_chars} ký tự).")
    print(f" -> SHA-256 bản gốc: {source_sha}")

    # 3. TẠO MASTER DOCUMENT (doc_level='master')
    print("\n[BƯỚC 3/5] Đang tạo Master Document (chuẩn khop_can_hoc_chapters.json)...")
    master_doc = build_master_document(pages, page_spans, source_sha, all_figures)
    chapters_file = args.output_dir / "oral_health_aging_chapters.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(chapters_file, "w", encoding="utf-8") as f:
        json.dump([master_doc.model_dump()], f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu Master Document: {chapters_file} (ID: {master_doc.id})")

    # 4. CHUNKING SLIDING WINDOW THEO CHUẨN INTERNAL_V1 (khop_can_hoc_sections.json)
    print(f"\n[BƯỚC 4/5] Đang nạp Tokenizer ({MODEL_NAME}) và thực hiện chunking sliding window...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    chunks = chunk_document_internal(
        master_doc=master_doc,
        page_spans=page_spans,
        all_figures=all_figures,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        overlap_words=args.overlap_words
    )
    print(f" -> Tạo thành công {len(chunks)} chunks chuẩn hóa (mỗi chunk <= {args.max_tokens} tokens).")

    sections_file = args.output_dir / "oral_health_aging_sections.json"
    with open(sections_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu Chunks/Sections vào: {sections_file}")

    for c in chunks:
        meta = c.metadata
        figs_count = len(meta.get("figures_data", []))
        print(f"    * [{c.id}] Trang {meta['page_start']}-{meta['page_end']} | {meta['token_count']} tokens | Figures: {figs_count}")
        print(f"      Tiêu đề: {c.title}")

    # 5. XUẤT CORPUS INTERNAL_V1 (DOCUMENTS & CHUNKS JSONL)
    print("\n[BƯỚC 5/5] Đang xuất corpus định dạng internal_v1 (jsonl)...")
    args.internal_corpus_dir.mkdir(parents=True, exist_ok=True)
    pub_doc, pub_chunks = build_internal_corpus_records(
        master_doc=master_doc,
        chunks=chunks,
        pdf_path=args.pdf,
        source_sha=source_sha,
        output_dir=args.internal_corpus_dir
    )

    doc_file = args.internal_corpus_dir / "oral_health_aging_documents.jsonl"
    with open(doc_file, "w", encoding="utf-8") as f:
        f.write(pub_doc.model_dump_json() + "\n")
    print(f" -> Đã lưu Document JSONL: {doc_file}")

    chunk_file = args.internal_corpus_dir / "oral_health_aging_chunks.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for pc in pub_chunks:
            f.write(pc.model_dump_json() + "\n")
    print(f" -> Đã lưu Chunks JSONL: {chunk_file}")

    # Tạo manifest
    token_counts = [c.metadata["token_count"] for c in chunks]
    manifest = {
        "dataset": "internal_v1",
        "source": "oral-health-aging-brushing.pdf",
        "document_id": master_doc.id,
        "master_title": master_doc.title,
        "total_pdf_pages": 4,
        "chunks_count": len(chunks),
        "figures_count": len(all_figures),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        "files": {
            "master_chapters": str(chapters_file.relative_to(ROOT)),
            "sections_chunks": str(sections_file.relative_to(ROOT)),
            "corpus_documents": str(doc_file.relative_to(ROOT)),
            "corpus_chunks": str(chunk_file.relative_to(ROOT)),
            "images_catalog": str(img_catalog_path.relative_to(ROOT))
        }
    }

    manifest_file = args.internal_corpus_dir / "oral_health_aging_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Tự động đồng bộ và hợp nhất vào file chunks.jsonl và documents.jsonl duy nhất
    from scripts.merge_internal_corpus import main as merge_corpus
    merge_corpus()

    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT CHUẨN HÓA KHO NỘI BỘ THEO ĐÚNG CHUẨN INTERNAL_V1:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
