"""Trích lớp chữ OCR của giáo trình Cắn khớp học, giữ nguồn và chia theo token."""
import argparse
import bisect
import hashlib
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from src.rag.base import DentalDocument
from src.rag.ocr_review import (
    DEFAULT_REVIEW, NORMALIZATION_VERSION, apply_reviewed_corrections, normalize_ocr_text,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "intfloat/multilingual-e5-small"

# Số trang PDF đã đối chiếu với bản scan; chương 1 bị đảo trang 3 và 4.
CHAPTER_METADATA = [
    {"num": 1, "title": "Mở đầu - Giới thiệu môn học Cắn khớp", "start_page": 10, "end_page": 17,
     "page_order": [11, 10, 12, 13, 14, 15, 16, 17]},
    {"num": 2, "title": "Giải phẫu chức năng hệ thống nhai", "start_page": 18, "end_page": 40},
    {"num": 3, "title": "Khớp cắn và quan niệm về khớp cắn", "start_page": 41, "end_page": 72},
    {"num": 4, "title": "Động học cắn khớp", "start_page": 73, "end_page": 89},
    {"num": 5, "title": "Hoạt động chức năng và cận chức năng", "start_page": 90, "end_page": 109},
    {"num": 6, "title": "Vận động và các vị trí của hàm dưới, khía cạnh thực hành", "start_page": 110, "end_page": 122},
    {"num": 7, "title": "Khám cắn khớp", "start_page": 124, "end_page": 146},
    {"num": 8, "title": "Khám các tiếp xúc cắn khớp", "start_page": 147, "end_page": 180},
    {"num": 9, "title": "Đại cương điều trị cắn khớp và điều chỉnh khớp cắn", "start_page": 181, "end_page": 206},
    {"num": 10, "title": "Giá khớp và sử dụng giá khớp", "start_page": 208, "end_page": 235},
    {"num": 11, "title": "Cung mặt và lên giá khớp", "start_page": 236, "end_page": 257},
    {"num": 12, "title": "Máng nhai", "start_page": 259, "end_page": 268},
    {"num": 13, "title": "Thực hành máng nhai", "start_page": 269, "end_page": 277},
]

GLOSSARY_METADATA = [
    {"id": "CKH-GLOSSARY-DEFINITIONS", "title": "Giải nghĩa thuật ngữ cắn khớp",
     "start_page": 278, "end_page": 283, "kind": "definitions"},
    {"id": "CKH-GLOSSARY-TRANSLATIONS", "title": "Đối chiếu thuật ngữ Việt - Anh - Pháp",
     "start_page": 284, "end_page": 292, "kind": "translation_table"},
]

# Chỉ đánh dấu các dạng OCR đáng ngờ; việc xác nhận phải đối chiếu bản scan.
OCR_SUSPECT = re.compile(
    r"\b(?:uà|uù|uê|rỡng|bhớp|bhóp|hhớp|hhop|lỗi\s+cầu|lôi\s+cầu|thái\s+đương)\b", re.I
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clean_rhm_ocr_text(text):
    """Chuẩn hóa cơ học, giữ nguyên từ OCR để có thể đối chiếu với nguồn."""
    return normalize_ocr_text(text)


def extract_keywords_rhm(title, text):
    vocabulary = (
        "khớp cắn", "cắn khớp", "hệ thống nhai", "khớp thái dương hàm", "tmd",
        "lồi cầu", "đĩa khớp", "cơ nhai", "vận động hàm dưới", "tương quan trung tâm",
        "lồng múi tối đa", "cản trở cắn khớp", "mài chỉnh khớp cắn", "máng nhai",
        "nghiến răng", "giá khớp", "cung mặt", "lên giá khớp",
    )
    lower = (title + " " + text).lower()
    return sorted({word for word in vocabulary if word in lower})


def embedding_text(document):
    return "passage: " + document.title + "\n" + document.content


def chapter_pages(chapter):
    return chapter.get("page_order", list(range(chapter["start_page"], chapter["end_page"] + 1)))


def make_parent(identifier, title, numbers, pages, source_sha, *, page_texts=None, ocr_review=None, **extra):
    page_texts = page_texts or {number: clean_rhm_ocr_text(pages[number]) for number in numbers}
    review_pages = (ocr_review or {}).get("pages", {})
    parts, spans, position = [], [], 0
    for number in numbers:
        text = page_texts[number]
        if parts:
            position += 2
        spans.append({"pdf_page": number, "start": position, "end": position + len(text),
                      "raw_text_sha256": sha256(pages[number].encode("utf-8")),
                      "page_text_sha256": sha256(text.encode("utf-8")),
                      "ocr_correction_count": len(review_pages.get(number, {}).get("changes", []))})
        parts.append(text)
        position += len(text)
    content = "\n\n".join(parts)
    if not content.strip():
        raise ValueError(f"Không có lớp chữ OCR trong {identifier}")
    flags = ["ocr_text_requires_review"]
    if numbers != sorted(numbers):
        flags.append("source_page_order_corrected")
    if any(not clean_rhm_ocr_text(pages[number]) for number in numbers):
        flags.append("empty_ocr_page")
    layout_pages = [number for number in numbers if review_pages.get(number, {}).get("layout_flags")]
    if layout_pages:
        flags.append("table_layout_requires_review")
    revision = sha256(json.dumps([source_sha, numbers, NORMALIZATION_VERSION,
                                  [span["page_text_sha256"] for span in spans]], ensure_ascii=False).encode("utf-8"))
    metadata = {
        "source": "RHM1 khop can hoc.pdf", "source_name": "Giáo trình Cắn khớp học",
        "source_type": "internal", "language": "vi", "doc_level": "master",
        "source_sha256": source_sha, "source_revision": revision,
        "normalization_version": NORMALIZATION_VERSION,
        "extraction_method": "existing_pdf_ocr_text_layer", "page_start": min(numbers),
        "page_end": max(numbers), "pdf_pages": numbers, "page_spans": spans,
        "quality_flags": flags, "layout_review_pages": layout_pages,
        "page_text_basis": "normalized_ocr_with_reviewed_corrections",
        "ocr_review": {"sha256": (ocr_review or {}).get("sha256"), "scope": (ocr_review or {}).get("scope", "none"),
                       "corrected_pdf_pages": [number for number in numbers if review_pages.get(number, {}).get("changes")],
                       "applied_corrections": sum(span["ocr_correction_count"] for span in spans)}, **extra,
    }
    return DentalDocument(id=identifier, title=title, content=content, category="Cắn khớp học & TMJ",
                          keywords=extract_keywords_rhm(title, content), metadata=metadata)


def table_units(parent, ocr_review):
    units = []
    review_pages = (ocr_review or {}).get("pages", {})
    for page in parent.metadata["page_spans"]:
        for change in review_pages.get(page["pdf_page"], {}).get("changes", []):
            if change.get("kind") != "table_rows":
                continue
            offset = page["start"] + change["corrected_start"]
            for index, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\n|\Z)", change["after"]), 1):
                units.append({"id": f"CKH-TABLE-{page['pdf_page']}-{index}",
                              "start": offset + match.start(), "end": offset + match.end()})
    return units


def chunk_chapter_content(parent, tokenizer, max_tokens=380, overlap_words=16, ocr_review=None):
    """Chia theo token, lưu offset chính xác trong chương và từng trang PDF."""
    if not 64 <= max_tokens <= 500 or overlap_words < 0:
        raise ValueError("max_tokens cần trong khoảng 64–500, overlap_words không âm")
    words = list(re.finditer(r"\S+", parent.content))
    starts = [word.start() for word in words]
    tables = table_units(parent, ocr_review)
    for unit in tables:
        unit["word_start"] = bisect.bisect_left(starts, unit["start"])
        unit["word_end"] = bisect.bisect_left(starts, unit["end"])
    chunks, start = [], 0
    while start < len(words):
        index = len(chunks) + 1
        title = f"{parent.title} - Đoạn {index}"
        begin = words[start].start()
        low, high, end = start + 1, len(words), start
        while low <= high:
            middle = (low + high) // 2
            content = parent.content[begin:words[middle - 1].end()]
            count = len(tokenizer.encode("passage: " + title + "\n" + content, verbose=False))
            if count <= max_tokens:
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError(f"Một từ hoặc tiêu đề vượt giới hạn token: {parent.id}")
        for unit in tables:
            if unit["word_start"] < end < unit["word_end"]:
                end = unit["word_start"]
                if end <= start:
                    raise ValueError(f"Hàng bảng vượt giới hạn token: {unit['id']}")
        stop = words[end - 1].end()
        source_spans = []
        for span in parent.metadata["page_spans"]:
            a, b = max(begin, span["start"]), min(stop, span["end"])
            if a < b:
                source_spans.append({"pdf_page": span["pdf_page"],
                                     "start": a - span["start"], "end": b - span["start"],
                                     "chapter_start": a, "chapter_end": b})
        numbers = [span["pdf_page"] for span in source_spans]
        content = parent.content[begin:stop]
        metadata = {key: value for key, value in parent.metadata.items() if key != "page_spans"}
        metadata["layout_review_pages"] = [number for number in numbers if number in parent.metadata["layout_review_pages"]]
        metadata["quality_flags"] = [flag for flag in parent.metadata["quality_flags"]
                                     if flag != "table_layout_requires_review" or metadata["layout_review_pages"]]
        review_pages = (ocr_review or {}).get("pages", {})
        applied = [(span["pdf_page"], change) for span in source_spans
                   for change in review_pages.get(span["pdf_page"], {}).get("changes", [])
                   if change["corrected_start"] < span["end"] and change["corrected_end"] > span["start"]]
        metadata["ocr_review"] = {"sha256": parent.metadata["ocr_review"]["sha256"],
                                  "scope": parent.metadata["ocr_review"]["scope"],
                                  "corrected_pdf_pages": list(dict.fromkeys(number for number, _ in applied)),
                                  "applied_corrections": len(applied),
                                  "correction_ids": [change["id"] for _, change in applied]}
        metadata.update(parent_id=parent.id, doc_level="section", chunk_index=index,
                        chapter_span={"start": begin, "end": stop}, source_spans=source_spans,
                        pdf_pages=numbers, page_start=min(numbers), page_end=max(numbers),
                        tokenizer=MODEL_NAME, max_tokens=max_tokens, overlap_words=overlap_words,
                        token_count=len(tokenizer.encode("passage: " + title + "\n" + content, verbose=False)),
                        ocr_suspects=sorted(set(match.group(0) for match in OCR_SUSPECT.finditer(content))))
        metadata["table_row_ids"] = [unit["id"] for unit in tables if begin <= unit["start"] and stop >= unit["end"]]
        metadata["embedding_ready"] = not metadata["ocr_suspects"] and not metadata["layout_review_pages"]
        signature = sha256(json.dumps([parent.metadata["source_revision"], begin, stop, max_tokens, overlap_words]).encode())[:10]
        chunks.append(DentalDocument(id=f"CKH-CH{metadata['chapter_number']:02d}-{index:04d}-{signature}",
                                     title=title, content=content, category=parent.category,
                                     keywords=extract_keywords_rhm(title, content), metadata=metadata))
        if end == len(words):
            break
        next_start = max(start + 1, end - min(overlap_words, end - start - 1))
        for unit in tables:
            if unit["word_start"] < next_start < unit["word_end"]:
                next_start = unit["word_start"] if unit["word_start"] > start else end
        start = next_start
    return chunks


class RHMDataPreprocessor:
    def __init__(self, pdf_path, output_dir, *, page_cache=None, max_tokens=380, overlap_words=16, ocr_review_path=DEFAULT_REVIEW):
        self.pdf_path, self.output_dir = Path(pdf_path), Path(output_dir)
        self.page_cache = Path(page_cache) if page_cache else None
        self.max_tokens, self.overlap_words = max_tokens, overlap_words
        self.ocr_review_path = ocr_review_path

    def read_pages(self):
        source_sha = sha256(self.pdf_path.read_bytes())
        reader = PdfReader(self.pdf_path)
        if len(reader.pages) != 314:
            raise ValueError("Cần rà lại khoảng trang: bản giáo trình này phải có 314 trang PDF")
        if self.page_cache:
            info = json.loads(self.page_cache.with_name("source.json").read_text(encoding="utf-8"))
            if info["sha256"] != source_sha:
                raise ValueError("Cache lớp chữ không khớp SHA-256 của PDF")
            records = [json.loads(line) for line in self.page_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
            pages = {item["pdf_page"]: item["text"] for item in records}
            if len(records) != len(pages) or set(pages) != set(range(1, 315)):
                raise ValueError("Cache thiếu trang hoặc trùng trang")
        else:
            pages = {}
            for number, page in enumerate(reader.pages, 1):
                pages[number] = page.extract_text() or ""
                if number % 50 == 0:
                    print(f"Đã đọc {number}/314 trang", flush=True)
        return pages, source_sha

    def run(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        pages, source_sha = self.read_pages()
        page_texts, ocr_review = apply_reviewed_corrections(pages, source_sha, self.ocr_review_path)
        chapters, chunks, glossary = [], [], []
        for chapter in CHAPTER_METADATA:
            number = chapter["num"]
            numbers = chapter_pages(chapter)
            if not re.search(rf"chương\s+{number}\s*:", pages[numbers[0]][:350], re.I):
                raise ValueError(f"Trang mở đầu không khớp chương {number}")
            parent = make_parent(f"CKH-CH{number:02d}-FULL", f"Chương {number}: {chapter['title']}",
                                 numbers, pages, source_sha, chapter_number=number,
                                 chapter_title=chapter["title"], page_texts=page_texts, ocr_review=ocr_review)
            pieces = chunk_chapter_content(parent, tokenizer, self.max_tokens, self.overlap_words, ocr_review)
            chapters.append(parent)
            chunks.extend(pieces)
            print(f"Chương {number}: {len(numbers)} trang, {len(pieces)} chunk", flush=True)
        for item in GLOSSARY_METADATA:
            numbers = list(range(item["start_page"], item["end_page"] + 1))
            parent = make_parent(item["id"], item["title"], numbers, pages, source_sha,
                                 doc_level="appendix", appendix_kind=item["kind"], retrieval_ready=False,
                                 page_texts=page_texts, ocr_review=ocr_review)
            parent.metadata["quality_flags"].append("multi_column_layout_requires_review")
            glossary.append(parent)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        names = {"chapters": "khop_can_hoc_chapters.json", "sections": "khop_can_hoc_sections.json",
                 "glossary_review": "khop_can_hoc_glossary_review.json"}
        for key, documents in [("chapters", chapters), ("sections", chunks), ("glossary_review", glossary)]:
            target = self.output_dir / names[key]
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps([document.model_dump() for document in documents],
                                            ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
        return {"total_chapters": len(chapters), "total_search_chunks": len(chunks),
                "chapter_pdf_pages": sum(len(chapter_pages(item)) for item in CHAPTER_METADATA),
                "glossary_documents_for_review": len(glossary), "max_tokens": self.max_tokens,
                "reviewed_corrections": sum(len(item["changes"]) for item in ocr_review["pages"].values()),
                "pages_with_reviewed_corrections": len(ocr_review["pages"]),
                "files_saved": {key: str(self.output_dir / name) for key, name in names.items()}}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/raw/RHM1 khop can hoc.pdf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--page-cache", type=Path)
    parser.add_argument("--max-tokens", type=int, default=380)
    parser.add_argument("--overlap-words", type=int, default=16)
    parser.add_argument("--ocr-review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    try:
        result = RHMDataPreprocessor(args.pdf, args.output_dir, page_cache=args.page_cache,
                                     max_tokens=args.max_tokens, overlap_words=args.overlap_words,
                                     ocr_review_path=args.ocr_review).run()
    except Exception as exc:
        print(f"Không xử lý được giáo trình: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
