"""Chuẩn hóa tài liệu Oral Health & Aging theo đúng cấu trúc và thuật toán của internal_v1 (sách RHM1).
Bao gồm:
- Bóc tách hình ảnh lâm sàng & dụng cụ hỗ trợ chất lượng cao vào data/images/oral_health_aging/
- Trích xuất trực tiếp lớp chữ từ các trang PDF gốc (không tóm tắt hay tự ý viết lại)
- Tạo Master Document (doc_level='master') như khop_can_hoc_chapters.json
- Cắt chunks theo thuật toán sliding window của internal_v1 (max_tokens=380, overlap_words=16,
  tokenizer=intfloat/multilingual-e5-small) tương đồng với khop_can_hoc_sections.json
- Tích hợp metadata: chunk_index, doc_level='section', page_start, page_end, parent_id, figures_data
- Xuất PublicDocument & PublicChunk theo chuẩn corpus internal_v1
"""

import bisect
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from transformers import AutoTokenizer

from src.data.public_chunks import TextSpan, PublicChunk
from src.data.public_schema import (
    DocumentBuilder,
    Locator,
    Processing,
    PublicDocument,
    RawSource,
    digest,
    make_document,
    save_raw,
)
from src.rag.base import DentalDocument

DOC_ID = "OHA-CH01"
FULL_ID = "OHA-CH01-FULL"
DOC_TITLE = "Oral Health & Aging: Information for Caregivers - Brushing"
SOURCE_NAME = "NIDCR - Oral Health & Aging: Information for Caregivers"
SOURCE_URL = "data/raw/oral-health-aging-brushing.pdf"
CATEGORY = "Chăm sóc Răng miệng & Lão khoa"
MODEL_NAME = "intfloat/multilingual-e5-small"

VOCABULARY = [
    "oral health", "aging", "caregivers", "older adults", "brushing",
    "toothbrush", "fluoride", "toothpaste", "dental plaque", "tooth decay",
    "gum disease", "foam tubing", "ergonomic grip", "utensil holder",
    "rubber band", "power toothbrush", "multiple-sided toothbrush",
    "mouth rest", "accidental biting", "two-toothbrush technique",
    "dentures", "denture care", "dentist", "tell-show-do", "flossing"
]

FIGURE_SPECS = [
    {
        "page": 1,
        "xref": 406,
        "filename": "oral_health_aging_banner.jpg",
        "label": "Banner: Oral Health & Aging",
        "caption": "Oral health and aging header banner showing older adult and caregiver.",
        "category": "Banner"
    },
    {
        "page": 1,
        "xref": 408,
        "filename": "caregiver_guiding_toothbrush.jpg",
        "label": "Hình 1: Hướng dẫn chải răng (Guide the Toothbrush)",
        "caption": "Help brush by placing your hand very gently over the person's hand and guiding the toothbrush.",
        "category": "Technique"
    },
    {
        "page": 2,
        "xref": 5,
        "filename": "toothbrush_foam_tubing_handle.jpg",
        "label": "Hình 2: Ống xốp bọc cán bàn chải (Foam Tubing)",
        "caption": "Foam tubing placed over toothbrush handle to make it bigger and easier to grip.",
        "category": "Adaptation"
    },
    {
        "page": 2,
        "xref": 7,
        "filename": "toothbrush_ergonomic_grip.jpg",
        "label": "Hình 3: Tay cầm công thái học (Ergonomic Grip)",
        "caption": "Toothbrush fitted with an ergonomic ball/cushion grip for reduced hand fatigue.",
        "category": "Adaptation"
    },
    {
        "page": 2,
        "xref": 9,
        "filename": "toothbrush_utensil_holder_aid.jpg",
        "label": "Hình 4: Đai giữ bàn chải (Utensil Holder)",
        "caption": "Utensil holder strap securing the toothbrush to hand for individuals with weak grip.",
        "category": "Adaptation"
    },
    {
        "page": 2,
        "xref": 11,
        "filename": "toothbrush_rubber_band_aid.jpg",
        "label": "Hình 5: Dây thun trợ lực giữ bàn chải (Rubber Band)",
        "caption": "Wide rubber band wrapped around hand and toothbrush to stabilize holding.",
        "category": "Adaptation"
    },
    {
        "page": 2,
        "xref": 13,
        "filename": "power_toothbrush.jpg",
        "label": "Hình 6: Bàn chải điện (Power Toothbrush)",
        "caption": "Electric power toothbrush providing motorized circular motions for effortless plaque removal.",
        "category": "Device"
    },
    {
        "page": 2,
        "xref": 15,
        "filename": "multiple_sided_toothbrush.jpg",
        "label": "Hình 7: Bàn chải đa diện (Multiple-Sided Toothbrush)",
        "caption": "Three-sided surround toothbrush cleaning outer, chewing, and inner tooth surfaces simultaneously.",
        "category": "Device"
    },
    {
        "page": 2,
        "xref": 17,
        "filename": "foam_mouth_rest_usage.jpg",
        "label": "Hình 8: Dụng cụ chặn miệng bằng xốp (Foam Mouth Rest)",
        "caption": "Foam mouth rest placed between back teeth to safely keep mouth open and prevent accidental biting.",
        "category": "Safety"
    },
    {
        "page": 3,
        "xref": 22,
        "filename": "two_toothbrush_technique.jpg",
        "label": "Hình 9: Kỹ thuật hai bàn chải (Two-Toothbrush Technique)",
        "caption": "Two-toothbrush technique: Using handle of one brush to retract cheek and rest teeth while brushing with second brush.",
        "category": "Technique"
    }
]


def extract_keywords(text: str) -> List[str]:
    lower = text.lower()
    return sorted({word for word in VOCABULARY if word in lower})


def extract_figures(pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)

    catalog = []
    for spec in FIGURE_SPECS:
        page_no = spec["page"]
        page = doc[page_no - 1]
        xref = spec["xref"]
        base_img = doc.extract_image(xref)

        img_bytes = base_img["image"]
        ext = base_img["ext"]
        out_file = output_dir / spec["filename"]
        out_file.write_bytes(img_bytes)

        rects = page.get_image_rects(xref)
        bbox = list(rects[0]) if rects else None
        sha = hashlib.sha256(img_bytes).hexdigest()

        meta = {
            "label": spec["label"],
            "caption": spec["caption"],
            "local_path": f"data/images/oral_health_aging/{spec['filename']}",
            "page": page_no,
            "category": spec["category"],
            "dimensions": f"{base_img['width']}x{base_img['height']}",
            "filesize_bytes": len(img_bytes),
            "sha256": sha,
            "bbox": bbox
        }
        catalog.append(meta)

    return catalog


def read_pdf_pages(pdf_path: Path) -> Tuple[Dict[int, str], List[Dict[str, Any]], str]:
    """Đọc nguyên bản lớp chữ của từng trang PDF (tương đồng với read_pages trong internal_v1)."""
    pdf_bytes = pdf_path.read_bytes()
    source_sha = hashlib.sha256(pdf_bytes).hexdigest()

    doc = pymupdf.open(pdf_path)
    pages = {}
    page_spans = []
    current_offset = 0

    for idx, page in enumerate(doc, 1):
        text = page.get_text() or ""
        text = text.replace("\r\n", "\n").strip()
        if current_offset > 0:
            current_offset += 2  # Ngăn cách giữa các trang bằng 2 dấu xuống dòng \n\n
        start = current_offset
        end = start + len(text)
        page_spans.append({
            "pdf_page": idx,
            "start": start,
            "end": end
        })
        pages[idx] = text
        current_offset = end

    return pages, page_spans, source_sha


def build_master_document(
    pages: Dict[int, str],
    page_spans: List[Dict[str, Any]],
    source_sha: str,
    all_figures: List[Dict[str, Any]]
) -> DentalDocument:
    """Tạo Document master (doc_level='master') như khop_can_hoc_chapters.json."""
    full_content = "\n\n".join(pages[idx] for idx in sorted(pages.keys()))
    numbers = sorted(pages.keys())

    metadata = {
        "source": "oral-health-aging-brushing.pdf",
        "source_name": SOURCE_NAME,
        "source_type": "internal",
        "warehouse": "internal",
        "language": "en",
        "doc_level": "master",
        "chapter_number": 1,
        "chapter_title": DOC_TITLE,
        "page_start": min(numbers),
        "page_end": max(numbers),
        "pdf_pages": numbers,
        "source_sha256": source_sha,
        "figures_data": all_figures
    }

    return DentalDocument(
        id=FULL_ID,
        title=DOC_TITLE,
        content=full_content,
        category=CATEGORY,
        keywords=extract_keywords(DOC_TITLE + " " + full_content),
        metadata=metadata
    )


def chunk_document_internal(
    master_doc: DentalDocument,
    page_spans: List[Dict[str, Any]],
    all_figures: List[Dict[str, Any]],
    tokenizer,
    max_tokens: int = 380,
    overlap_words: int = 16
) -> List[DentalDocument]:
    """Cắt chunks chuẩn xác theo thuật toán sliding window của internal_v1 (chunk_chapter_content)."""
    words = list(re.finditer(r"\S+", master_doc.content))
    chunks = []
    start = 0

    while start < len(words):
        chunk_idx = len(chunks) + 1
        begin = words[start].start()
        low, high, end = start + 1, len(words), start

        # Tìm ranh giới token tối đa không vượt quá max_tokens
        while low <= high:
            middle = (low + high) // 2
            cand_text = master_doc.content[begin:words[middle - 1].end()]
            cand_tokens = len(tokenizer.encode("passage: " + master_doc.title + "\n" + cand_text, verbose=False))
            if cand_tokens <= max_tokens:
                end = middle
                low = middle + 1
            else:
                high = middle - 1

        if end == start:
            raise ValueError(f"Từ tại vị trí {start} vượt giới hạn max_tokens!")

        stop = words[end - 1].end()
        chunk_content = master_doc.content[begin:stop]

        # Xác định các trang PDF chứa chunk này
        pages_covered = []
        for span in page_spans:
            if max(begin, span["start"]) < min(stop, span["end"]):
                pages_covered.append(span["pdf_page"])

        p_start = min(pages_covered) if pages_covered else 1
        p_end = max(pages_covered) if pages_covered else p_start

        # Lấy dòng tiêu đề đầu tiên của đoạn làm sub-title (như trong internal_v1)
        first_line = chunk_content.splitlines()[0].strip()
        first_line_clean = re.sub(r"^[■\s\-\.]+", "", first_line)
        if len(first_line_clean) > 60:
            first_line_clean = first_line_clean[:57] + "..."
        chunk_title = f"{master_doc.title} - {first_line_clean}"

        # Gán figures_data cho chunk dựa theo trang xuất hiện
        chunk_figures = [f for f in all_figures if f["page"] in pages_covered]

        token_count = len(tokenizer.encode("passage: " + master_doc.title + "\n" + chunk_content, verbose=False))

        chunk_meta = {
            "source": "oral-health-aging-brushing.pdf",
            "source_name": SOURCE_NAME,
            "source_type": "internal",
            "warehouse": "internal",
            "chapter_number": 1,
            "chapter_title": DOC_TITLE,
            "chunk_index": chunk_idx,
            "doc_level": "section",
            "page_start": p_start,
            "page_end": p_end,
            "parent_id": master_doc.id,
            "token_count": token_count,
            "tokenizer": MODEL_NAME,
            "max_tokens": max_tokens,
            "overlap_words": overlap_words,
            "figures_data": chunk_figures
        }

        chunk_doc = DentalDocument(
            id=f"{DOC_ID}-{chunk_idx:02d}",
            title=chunk_title,
            content=chunk_content,
            category=master_doc.category,
            keywords=extract_keywords(chunk_title + " " + chunk_content),
            metadata=chunk_meta,
            score=0.0
        )
        chunks.append(chunk_doc)

        if end == len(words):
            break
        # Dịch chuyển start theo overlap_words
        start = max(start + 1, end - min(overlap_words, end - start - 1))

    return chunks


def build_internal_corpus_records(
    master_doc: DentalDocument,
    chunks: List[DentalDocument],
    pdf_path: Path,
    source_sha: str,
    output_dir: Path
) -> Tuple[PublicDocument, List[PublicChunk]]:
    """Xây dựng PublicDocument và PublicChunk theo chuẩn internal_v1/corpus/ (rhm_documents.jsonl, chunks.jsonl)."""
    builder = DocumentBuilder(DOC_ID)

    # Tạo section cho document
    sec = builder.section(
        title=master_doc.title,
        locator=Locator(path="data/raw/oral-health-aging-brushing.pdf", pdf_page=1)
    )

    # Mỗi chunk DentalDocument tương ứng một Block chuẩn hóa trong PublicDocument
    for c in chunks:
        loc = Locator(
            path="data/raw/oral-health-aging-brushing.pdf",
            pdf_page=c.metadata.get("page_start", 1)
        )
        builder.block(sec, kind="paragraph", text=c.content, locator=loc)

    sections = builder.finish_sections()

    doc = make_document(
        document_id=DOC_ID,
        title=master_doc.title,
        source="internal",
        source_type="internal",
        language="en",
        source_url="data/raw/oral-health-aging-brushing.pdf",
        raw=RawSource(
            path="data/raw/oral-health-aging-brushing.pdf",
            sha256=source_sha,
            media_type="application/pdf"
        ),
        processing=Processing(
            parser="internal-pdf-chunk-parser",
            parser_version="1.0.1",
            method="pdf-sliding-window-tokens-internal"
        ),
        sections=sections
    )

    # Chuyển đổi thành PublicChunk có source spans và locators
    public_chunks = []
    chunk_config = {
        "algorithm": "blocks-with-source-spans-v2",
        "max_tokens": 380,
        "overlap_words": 16,
        "tokenizer": MODEL_NAME,
        "passage_prefix": "passage: "
    }

    for b, c in zip(sections[0].blocks, chunks):
        context = doc.title + "\n" + " > ".join(sec.heading_path)
        content = context + "\n" + b.text
        tokens = c.metadata.get("token_count", 0)
        source_span = [TextSpan(block_id=b.block_id, start=0, end=len(b.text))]

        identity = {
            "revision": doc.revision_id,
            "section": sec.section_id,
            "spans": [s.model_dump() for s in source_span],
            "chunking": chunk_config
        }
        chunk_id = "chunk-" + digest(identity)

        public_chunks.append(PublicChunk(
            schema_version="public-chunk-v1",
            chunk_id=chunk_id,
            document_id=doc.document_id,
            document_revision=doc.revision_id,
            section_id=sec.section_id,
            heading_path=sec.heading_path,
            source="internal",
            source_url=doc.source_url,
            language=doc.language,
            content=content,
            token_count=tokens,
            spans=source_span,
            locators=b.locators,
            chunking=chunk_config,
            quality_flags=[]
        ))

    return doc, public_chunks
