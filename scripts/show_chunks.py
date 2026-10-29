"""Hiển thị chi tiết dữ liệu chunks chuẩn hóa theo định dạng internal_v1.
Hỗ trợ xem toàn bộ file hợp nhất data/processed/internal_v1/corpus/chunks.jsonl
hoặc xem từng bộ sách cụ thể (DPH, OHA, RHM1).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INTERNAL_CORPUS_DIR = ROOT / "data/processed/internal_v1/corpus"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["all", "dph", "oha", "rhm1"],
        default="all",
        help="Nguồn chunk cần xem: all (chunks.jsonl hợp nhất), dph (Dental Public Health), oha (Oral Health Aging), rhm1 (Khớp cắn học)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Số lượng chunks tối đa hiển thị chi tiết (mặc định: 10, đặt 0 để xem tất cả)"
    )
    args = parser.parse_args()

    print("=" * 85)
    print("DỮ LIỆU CHUNKING KHO NỘI BỘ (CHUẨN INTERNAL_V1):")
    print("=" * 85)

    if args.source == "all":
        chunks_file = INTERNAL_CORPUS_DIR / "chunks.jsonl"
        docs_file = INTERNAL_CORPUS_DIR / "documents.jsonl"
        manifest_file = INTERNAL_CORPUS_DIR / "manifest.json"

        if not chunks_file.exists():
            print(f"LỖI: Chưa có file {chunks_file}. Hãy chạy các script xử lý dữ liệu trước.")
            return 1

        doc_catalog = {}
        if docs_file.exists():
            for line in docs_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    doc_catalog[item["document_id"]] = item["title"]

        chunks = []
        source_counts = Counter()
        for line in chunks_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                chunks.append(c)
                source_counts[c.get("source", "unknown")] += 1

        print(f"📁 Tệp nguồn: {chunks_file.relative_to(ROOT)}")
        print(f"📊 Tổng số chunks hợp nhất: {len(chunks)}")
        print(f"   * Phân bổ theo nguồn: {dict(source_counts)}")
        if manifest_file.exists():
            mf = json.loads(manifest_file.read_text(encoding="utf-8"))
            ts = mf.get("token_statistics", {})
            print(f"   * Thống kê tokens: Min={ts.get('min_tokens')}, Max={ts.get('max_tokens')}, Avg={ts.get('avg_tokens')}")

        display_chunks = chunks[:args.limit] if args.limit > 0 else chunks
        for idx, c in enumerate(display_chunks, 1):
            doc_id = c.get("document_id")
            doc_title = doc_catalog.get(doc_id, doc_id)
            locs = c.get("locators", [])
            page_info = f"Trang {locs[0].get('pdf_page')}" if locs and locs[0].get("pdf_page") else "N/A"
            source = c.get("source", "")

            print(f"\n🔹 CHUNK #{idx:02d} | Nguồn: [{source}] | Doc: {doc_id} | {page_info} | Tokens: {c.get('token_count')}/380")
            print(f"   Tiêu đề tài liệu: {doc_title}")
            print(f"   Chunk ID: {c.get('chunk_id')}")
            print(f"   Nội dung trích đoạn:")
            print("-" * 50)
            lines = c.get("content", "").splitlines()
            for l in lines[:8]:
                if l.strip():
                    print(f"   {l}")
            if len(lines) > 8:
                print(f"   ... ({len(lines) - 8} dòng nữa) ...")
            print("-" * 85)

        if args.limit > 0 and len(chunks) > args.limit:
            print(f"\nℹ️ Đang hiển thị {args.limit}/{len(chunks)} chunks. Sử dụng `--limit 0` để xem toàn bộ.")

    elif args.source == "dph":
        sec_file = ROOT / "data/processed/dental_public_health_sections.json"
        if not sec_file.exists():
            print(f"LỖI: Chưa có file {sec_file}.")
            return 1
        with open(sec_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        display_chunks = chunks[:args.limit] if args.limit > 0 else chunks
        for idx, c in enumerate(display_chunks, 1):
            meta = c.get("metadata", {})
            figs = meta.get("figures_data", [])
            print(f"\n🔹 CHUNK #{idx:02d} | ID: {c['id']} | Trang: {meta.get('page_start')}-{meta.get('page_end')} | Tokens: {meta.get('token_count')}/380")
            print(f"   Tiêu đề: {c['title']}")
            if figs:
                print(f"   📸 HÌNH ẢNH MINH HỌA ({len(figs)} ảnh):")
                for fig in figs:
                    print(f"      * [{fig['label']}] - {fig['local_path']}")
            print("-" * 50)
            lines = c.get("content", "").splitlines()
            for l in lines[:6]:
                if l.strip():
                    print(f"   {l}")
            print("-" * 85)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
