"""Script xử lý sách Dental Public Health & Research (Christine Nathe) theo chuẩn internal_v1.
Thực hiện:
1. Bóc tách toàn bộ hình ảnh lâm sàng, lịch sử & sơ đồ vào data/images/dental_public_health/.
2. Đọc trực tiếp lớp chữ nguyên bản từ PDF gốc.
3. Tạo file Master documents: data/processed/dental_public_health_chapters.json.
4. Chunking trượt theo token (max_tokens=380, overlap_words=16, tokenizer=intfloat/multilingual-e5-small)
   xuất ra file data/processed/dental_public_health_sections.json.
5. Xuất các bản ghi corpus internal_v1 (dental_public_health_documents.jsonl, dental_public_health_chunks.jsonl).
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data.sources.dental_public_health import (
    CATEGORY,
    DPH_METADATA,
    IMAGES_DIR,
    MODEL_NAME,
    PDF_PATH,
    build_dph_corpus_records,
    build_master_document,
    chunk_chapter_internal,
    load_figures_catalog,
    read_pdf_book,
)
from scripts.extract_dph_images import extract_all_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PDF_PATH,
        help="Đường dẫn file PDF gốc cần xử lý"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Thư mục xuất ảnh minh họa"
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
    print("CHUẨN HÓA DỮ LIỆU DENTAL PUBLIC HEALTH & RESEARCH THEO CHUẨN INTERNAL_V1")
    print("=" * 80)

    if not args.pdf.exists():
        print(f"LỖI: Không tìm thấy file PDF tại {args.pdf}", file=sys.stderr)
        return 1

    start_time = time.time()

    # 1. BÓC TÁCH HÌNH ẢNH
    print("\n[BƯỚC 1/5] Kiểm tra và trích xuất hình ảnh lâm sàng & sơ đồ minh họa...")
    all_figures = extract_all_figures()
    print(f" -> Đã có {len(all_figures)} hình ảnh chất lượng cao trong catalog.")

    # 2. ĐỌC LỚP CHỮ GỐC TỪ PDF
    print("\n[BƯỚC 2/5] Đang đọc toàn bộ lớp chữ từ file PDF gốc...")
    pages, source_sha = read_pdf_book(args.pdf)
    total_chars = sum(len(txt) for txt in pages.values())
    print(f" -> Đã đọc {len(pages)} trang PDF ({total_chars:,} ký tự).")
    print(f" -> SHA-256: {source_sha}")

    # 3. NẠP TOKENIZER
    print(f"\n[BƯỚC 3/5] Đang nạp Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. TẠO MASTER DOCUMENTS VÀ CHUNKS
    print("\n[BƯỚC 4/5] Đang xử lý từng chương (Master Document & Chunking sliding window)...")
    all_master_docs = []
    all_section_chunks = []
    all_public_docs = []
    all_public_chunks = []

    for item in DPH_METADATA:
        master_doc, spans = build_master_document(item, pages, source_sha, all_figures)
        all_master_docs.append(master_doc)

        chunks = chunk_chapter_internal(
            master_doc=master_doc,
            page_spans=spans,
            all_figures=all_figures,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            overlap_words=args.overlap_words
        )
        all_section_chunks.extend(chunks)

        pub_doc, pub_chunks = build_dph_corpus_records(master_doc, chunks, source_sha)
        all_public_docs.append(pub_doc)
        all_public_chunks.extend(pub_chunks)

        ch_num = item["num"]
        print(f"   [{ch_num:02d}/25] {item['id']} ({item['start_page']:3d}-{item['end_page']:3d}): {len(chunks):3d} chunks | {master_doc.title[:45]}...")

    print(f"\n -> Tổng cộng: {len(all_master_docs)} Master Documents.")
    print(f" -> Tổng cộng: {len(all_section_chunks)} Section Chunks chuẩn hóa (mỗi chunk <= {args.max_tokens} tokens).")

    # Lưu Master Documents & Sections vào data/processed/
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chapters_file = args.output_dir / "dental_public_health_chapters.json"
    with open(chapters_file, "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in all_master_docs], f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu: {chapters_file}")

    sections_file = args.output_dir / "dental_public_health_sections.json"
    with open(sections_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in all_section_chunks], f, ensure_ascii=False, indent=2)
    print(f" -> Đã lưu: {sections_file}")

    # 5. XUẤT CORPUS INTERNAL_V1 (DOCUMENTS & CHUNKS)
    print("\n[BƯỚC 5/5] Đang ghi nhận các bản ghi vào corpus internal_v1...")
    args.internal_corpus_dir.mkdir(parents=True, exist_ok=True)
    
    doc_file = args.internal_corpus_dir / "dental_public_health_documents.jsonl"
    with open(doc_file, "w", encoding="utf-8") as f:
        for pd in all_public_docs:
            f.write(pd.model_dump_json() + "\n")
    print(f" -> Đã lưu: {doc_file} ({len(all_public_docs)} documents)")

    chunk_file = args.internal_corpus_dir / "dental_public_health_chunks.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for pc in all_public_chunks:
            f.write(pc.model_dump_json() + "\n")
    print(f" -> Đã lưu: {chunk_file} ({len(all_public_chunks)} chunks)")

    # Manifest riêng của DPH
    token_counts = [c.token_count for c in all_public_chunks]
    manifest = {
        "dataset": "internal_v1",
        "source": "Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
        "documents_count": len(all_public_docs),
        "chunks_count": len(all_public_chunks),
        "figures_count": len(all_figures),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "files": {
            "chapters": str(chapters_file.relative_to(ROOT)),
            "sections": str(sections_file.relative_to(ROOT)),
            "documents": str(doc_file.relative_to(ROOT)),
            "chunks": str(chunk_file.relative_to(ROOT))
        }
    }
    manifest_file = args.internal_corpus_dir / "dental_public_health_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" -> Đã lưu manifest DPH: {manifest_file}")

    # Tự động đồng bộ và hợp nhất vào file chunks.jsonl và documents.jsonl duy nhất
    from scripts.merge_internal_corpus import main as merge_corpus
    merge_corpus()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ XỬ LÝ DENTAL PUBLIC HEALTH HOÀN TẤT TRONG {elapsed:.1f} GIÂY!")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
