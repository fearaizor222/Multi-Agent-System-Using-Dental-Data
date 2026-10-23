"""Chuẩn hóa bộ dữ liệu đóng (sách RHM1 Khớp cắn học) theo đúng định dạng corpus của public_v1.
Tạo:
- data/processed/internal_v1/corpus/rhm_documents.jsonl (các Document có sections, blocks, locators)
- data/processed/internal_v1/corpus/chunks.jsonl (các Chunk chuẩn hóa, spans, locators, token_count <= 380)
Bỏ qua hoàn toàn file 2hdrnghmmt.pdf.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data/raw/RHM1 khop can hoc.pdf"
CHAPTERS_JSON = ROOT / "data/processed/khop_can_hoc_chapters.json"
OUTPUT_DIR = ROOT / "data/processed/internal_v1/corpus"
MODEL_NAME = "intfloat/multilingual-e5-small"

from src.data.public_schema import (
    DocumentBuilder,
    Locator,
    Processing,
    RawSource,
    make_document,
)
from src.data.public_chunks import make_chunks, validate_chunk_sources


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


def is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    # Tiêu đề số la mã, chữ số, chữ cái, hoặc chữ in hoa ngắn
    patterns = [
        r"^(?:[I|V|X]+)\.\s+\S+",
        r"^(?:\d+)\.\s+\S+",
        r"^(?:[A-Z])\.\s+\S+",
        r"^(?:MỤC TIÊU|ĐẠI CƯƠNG|ĐỊNH NGHĨA|CHỈ ĐỊNH|CHỐNG CHỈ ĐỊNH|PHÂN LOẠI|KẾT LUẬN)\b",
    ]
    for p in patterns:
        if re.match(p, line, re.IGNORECASE):
            return True
    return False


def build_internal_documents(chapters, pdf_sha: str):
    documents = []
    
    for ch in chapters:
        num = ch["metadata"].get("chapter_number", 0)
        doc_id = f"RHM1-CH{num:02d}" if num > 0 else ch["id"].replace("-FULL", "")
        title = ch["title"].strip()
        page_start = ch["metadata"].get("page_start", 1)
        page_end = ch["metadata"].get("page_end", page_start)
        
        builder = DocumentBuilder(doc_id)
        current_section = builder.section(
            title=title,
            locator=Locator(path="data/raw/RHM1 khop can hoc.pdf", pdf_page=page_start)
        )
        
        content = ch["content"]
        paragraphs = re.split(r"\n\s*\n+", content)
        
        total_p = max(1, len(paragraphs))
        for idx, raw_p in enumerate(paragraphs):
            p_text = raw_p.strip()
            if not p_text:
                continue
            
            # Ước lượng số trang PDF tuyến tính theo vị trí đoạn trong chương
            est_page = page_start + int((idx / total_p) * (page_end - page_start))
            loc = Locator(path="data/raw/RHM1 khop can hoc.pdf", pdf_page=est_page)
            
            lines = p_text.splitlines()
            first_line = lines[0].strip()
            
            if is_heading_candidate(first_line):
                # Nếu đoạn bắt đầu bằng một tiêu đề, mở section con mới
                sub_title = first_line
                current_section = builder.section(title=sub_title, locator=loc, parent=None)
                remaining_text = "\n".join(lines[1:]).strip()
                if remaining_text:
                    builder.block(current_section, kind="paragraph", text=remaining_text, locator=loc)
            else:
                builder.block(current_section, kind="paragraph", text=p_text, locator=loc)
        
        sections = builder.finish_sections()
        if not sections:
            continue
            
        doc = make_document(
            document_id=doc_id,
            title=title,
            source="rhm1",
            source_type="internal",
            language="vi",
            source_url="data/raw/RHM1 khop can hoc.pdf",
            raw=RawSource(
                path="data/raw/RHM1 khop can hoc.pdf",
                sha256=pdf_sha,
                media_type="application/pdf"
            ),
            processing=Processing(
                parser="rhm-chapter-parser",
                parser_version="1.0.1",
                method="pdf-ocr-chapter-blocks"
            ),
            sections=sections
        )
        documents.append(doc)
        
    return documents


def main():
    print("=" * 70)
    print("CHUẨN HÓA KHO NỘI BỘ RHM1 SANG CHUẨN DATASET INTERNAL_V1")
    print("=" * 70)
    
    if not PDF_PATH.exists():
        print(f"LỖI: Không tìm thấy file {PDF_PATH}", file=sys.stderr)
        sys.exit(1)
        
    if not CHAPTERS_JSON.exists():
        print(f"LỖI: Không tìm thấy file {CHAPTERS_JSON}", file=sys.stderr)
        sys.exit(1)
        
    print(f"1. Đang tính SHA-256 cho {PDF_PATH.name}...")
    pdf_sha = compute_sha256(PDF_PATH)
    print(f"   -> SHA-256: {pdf_sha}")
    
    print(f"2. Đang đọc dữ liệu chương từ {CHAPTERS_JSON.name}...")
    with open(CHAPTERS_JSON, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    print(f"   -> Đã tải {len(chapters)} chương.")
    
    print("3. Đang xây dựng danh sách Document chuẩn theo schema...")
    documents = build_internal_documents(chapters, pdf_sha)
    print(f"   -> Đã tạo {len(documents)} Document thành công.")
    
    print("4. Đang nạp tokenizer E5 (intfloat/multilingual-e5-small)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("5. Đang cắt chunks theo chuẩn (max_tokens=380, overlap_words=16)...")
    all_chunks = []
    for doc in documents:
        chunks = make_chunks(doc, tokenizer, max_tokens=380, overlap_words=16, tokenizer_name=MODEL_NAME)
        validate_chunk_sources(doc, chunks)
        all_chunks.extend(chunks)
        print(f"   - {doc.document_id} ({doc.title[:40]}...): {len(chunks)} chunks")
        
    print(f"   -> Tổng cộng: {len(all_chunks)} chunks.")
    
    # 6. Ghi ra thư mục data/processed/internal_v1/corpus/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = OUTPUT_DIR / "rhm_documents.jsonl"
    chunk_path = OUTPUT_DIR / "chunks.jsonl"
    
    print(f"6. Đang lưu kết quả vào {OUTPUT_DIR}...")
    with open(doc_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
    print(f"   -> Đã lưu documents: {doc_path} ({len(documents)} records)")
    
    with open(chunk_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")
    print(f"   -> Đã lưu chunks: {chunk_path} ({len(all_chunks)} records)")
    
    token_counts = [chunk.token_count for chunk in all_chunks]
    summary = {
        "dataset": "internal_v1",
        "source": "RHM1 khop can hoc.pdf",
        "documents_count": len(documents),
        "chunks_count": len(all_chunks),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "files": {
            "documents": str(doc_path.relative_to(ROOT)),
            "chunks": str(chunk_path.relative_to(ROOT))
        }
    }
    
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n✅ HOÀN TẤT CHUẨN HÓA KHO NỘI BỘ:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
