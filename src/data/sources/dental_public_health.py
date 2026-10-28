"""Chuẩn hóa tài liệu Dental Public Health & Research (Christine Nathe) theo đúng chuẩn internal_v1.
Bao gồm:
- Bóc tách hình ảnh lâm sàng, lịch sử & sơ đồ minh họa chất lượng cao vào data/images/dental_public_health/
- Trích xuất trực tiếp lớp chữ từ các trang PDF gốc (không tóm tắt hay tự ý viết lại)
- Tạo Master Documents (doc_level='master') lưu vào data/processed/dental_public_health_chapters.json
- Cắt chunks theo thuật toán sliding window của internal_v1 (max_tokens=380, overlap_words=16,
  tokenizer=intfloat/multilingual-e5-small) lưu vào data/processed/dental_public_health_sections.json
- Tích hợp metadata: chunk_index, doc_level='section', page_start, page_end, parent_id, figures_data, token_count
- Xuất PublicDocument & PublicChunk theo chuẩn corpus internal_v1 (nguồn 'dph', source_type 'internal')
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

from src.data.public_chunks import PublicChunk, TextSpan
from src.data.public_schema import (
    DocumentBuilder,
    Locator,
    Processing,
    PublicDocument,
    RawSource,
    digest,
    make_document,
)
from src.rag.base import DentalDocument

ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = ROOT / "data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf"
IMAGES_DIR = ROOT / "data/images/dental_public_health"
MODEL_NAME = "intfloat/multilingual-e5-small"
CATEGORY = "Nha khoa Công cộng & Nghiên cứu"

DPH_METADATA = [
    {"num": 1, "id": "DPH-CH01", "title": "Chapter 1: Dental Public Health - An Overview", "start_page": 19, "end_page": 33},
    {"num": 2, "id": "DPH-CH02", "title": "Chapter 2: The Prevention Movement", "start_page": 34, "end_page": 50},
    {"num": 3, "id": "DPH-CH03", "title": "Chapter 3: Dental Care Delivery in the United States", "start_page": 51, "end_page": 63},
    {"num": 4, "id": "DPH-CH04", "title": "Chapter 4: Dental Hygiene Care Delivery in the Global Community", "start_page": 64, "end_page": 81},
    {"num": 5, "id": "DPH-CH05", "title": "Chapter 5: Financing of Dental Care", "start_page": 82, "end_page": 98},
    {"num": 6, "id": "DPH-CH06", "title": "Chapter 6: Federal and State Legislation Affecting Dental Hygiene Practice", "start_page": 99, "end_page": 108},
    {"num": 7, "id": "DPH-CH07", "title": "Chapter 7: Advocacy for Dental Care", "start_page": 109, "end_page": 124},
    {"num": 8, "id": "DPH-CH08", "title": "Chapter 8: Dental Health Education and Promotion", "start_page": 125, "end_page": 135},
    {"num": 9, "id": "DPH-CH09", "title": "Chapter 9: Lesson Plan Development", "start_page": 136, "end_page": 146},
    {"num": 10, "id": "DPH-CH10", "title": "Chapter 10: Target Populations", "start_page": 147, "end_page": 160},
    {"num": 11, "id": "DPH-CH11", "title": "Chapter 11: Cultural Competency", "start_page": 161, "end_page": 170},
    {"num": 12, "id": "DPH-CH12", "title": "Chapter 12: Program Planning", "start_page": 171, "end_page": 180},
    {"num": 13, "id": "DPH-CH13", "title": "Chapter 13: Program Evaluation", "start_page": 181, "end_page": 196},
    {"num": 14, "id": "DPH-CH14", "title": "Chapter 14: Research in Dental Hygiene", "start_page": 197, "end_page": 203},
    {"num": 15, "id": "DPH-CH15", "title": "Chapter 15: Ethical Principles in Research", "start_page": 204, "end_page": 211},
    {"num": 16, "id": "DPH-CH16", "title": "Chapter 16: The Research Process", "start_page": 212, "end_page": 226},
    {"num": 17, "id": "DPH-CH17", "title": "Chapter 17: Biostatistics", "start_page": 227, "end_page": 249},
    {"num": 18, "id": "DPH-CH18", "title": "Chapter 18: Oral Epidemiology", "start_page": 250, "end_page": 279},
    {"num": 19, "id": "DPH-CH19", "title": "Chapter 19: Current Oral Epidemiological Findings", "start_page": 280, "end_page": 299},
    {"num": 20, "id": "DPH-CH20", "title": "Chapter 20: Evaluation of Scientific Literature and Dental Products", "start_page": 300, "end_page": 310},
    {"num": 21, "id": "DPH-CH21", "title": "Chapter 21: Careers in Dental Public Health", "start_page": 311, "end_page": 318},
    {"num": 22, "id": "DPH-CH22", "title": "Chapter 22: Strategies for Creating Dental Hygiene Positions in Dental Public Health Settings", "start_page": 319, "end_page": 326},
    {"num": 23, "id": "DPH-CH23", "title": "Chapter 23: Dental Public Health Review", "start_page": 327, "end_page": 336},
    {"num": 24, "id": "DPH-APP-C", "title": "Appendix C: Dental Terms and Phrases Translated into Spanish and Vietnamese", "start_page": 340, "end_page": 341},
    {"num": 25, "id": "DPH-GLOSSARY", "title": "Glossary: Dental Public Health & Research", "start_page": 354, "end_page": 361},
]

VOCABULARY = [
    "dental public health", "community dental health", "oral health", "dental hygiene",
    "prevention", "primary prevention", "secondary prevention", "tertiary prevention",
    "water fluoridation", "fluoride", "dental fluorosis", "sealants", "mouthguards",
    "epidemiology", "oral epidemiology", "incidence", "prevalence", "index", "indices",
    "biostatistics", "mean", "median", "mode", "standard deviation", "variance",
    "null hypothesis", "p-value", "t-test", "anova", "chi-square", "correlation",
    "research", "clinical trial", "observational study", "cohort", "case-control",
    "ethics", "autonomy", "beneficence", "nonmaleficence", "justice", "informed consent",
    "caries", "dental caries", "early childhood caries", "periodontal disease", "gingivitis",
    "oral cancer", "pharyngeal cancer", "squamous cell carcinoma", "candidiasis",
    "cleft lip", "cleft palate", "malocclusion", "tooth loss", "edentulism",
    "health promotion", "health education", "program planning", "evaluation",
    "medicaid", "medicare", "who", "cdc", "nih", "nidcr", "hrsa", "target population"
]


def extract_keywords(text: str) -> List[str]:
    lower = text.lower()
    return sorted({word for word in VOCABULARY if word in lower})


def clean_page_text(text: str, page_num: int) -> str:
    # 1. Normalize spaces
    text = text.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ").replace("\t", " ")
    
    # 2. Filter running headers / footers
    lines = text.splitlines()
    cleaned_lines = []
    for l in lines:
        s = l.strip()
        # Skip standalone page numbers matching page_num +/- 2
        if re.match(r"^\d+$", s) and abs(int(s) - page_num) <= 2:
            continue
        # Skip unit / chapter running banners
        if re.match(r"^UNIT\s+[IVXLCDM]+\s*[:–-]", s, re.I):
            continue
        if re.match(r"^CHAPTER\s+\d+\s+.*", s, re.I):
            continue
        cleaned_lines.append(l)
        
    text = "\n".join(cleaned_lines)
    
    # 3. De-hyphenate words broken across lines: e.g., pre-\nvention -> prevention\n
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2\n", text)
    
    # 4. Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_book(pdf_path: Path) -> Tuple[Dict[int, str], str]:
    if pymupdf is None:
        raise ImportError("Cần thư viện pymupdf để đọc file PDF.")
    doc = pymupdf.open(pdf_path)
    source_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    
    pages = {}
    for pno in range(len(doc)):
        page_num = pno + 1
        raw_text = doc[pno].get_text() or ""
        cleaned = clean_page_text(raw_text, page_num)
        pages[page_num] = cleaned
        
    return pages, source_sha


def load_figures_catalog(images_dir: Path) -> List[Dict[str, Any]]:
    catalog_path = images_dir / "images_catalog.json"
    if catalog_path.exists():
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    return []


def build_master_document(
    item: Dict[str, Any],
    pages: Dict[int, str],
    source_sha: str,
    all_figures: List[Dict[str, Any]]
) -> Tuple[DentalDocument, List[Dict[str, Any]]]:
    """Tạo DentalDocument cấp độ master và danh sách page_spans."""
    start_p, end_p = item["start_page"], item["end_page"]
    page_numbers = list(range(start_p, end_p + 1))
    
    parts = []
    spans = []
    curr_pos = 0
    
    for p in page_numbers:
        p_text = pages.get(p, "")
        if parts:
            curr_pos += 2  # tính "\n\n"
        start_idx = curr_pos
        end_idx = curr_pos + len(p_text)
        spans.append({
            "pdf_page": p,
            "start": start_idx,
            "end": end_idx,
            "char_count": len(p_text)
        })
        parts.append(p_text)
        curr_pos = end_idx
        
    full_content = "\n\n".join(parts)
    
    # Lấy figures trong khoảng trang của chapter
    ch_figures = [f for f in all_figures if start_p <= f["page"] <= end_p]
    
    metadata = {
        "source": "Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
        "source_name": "Dental Public Health & Research (Christine Nathe)",
        "source_type": "internal",
        "warehouse": "internal",
        "chapter_number": item["num"],
        "chapter_title": item["title"],
        "doc_level": "master",
        "page_start": start_p,
        "page_end": end_p,
        "pdf_pages": page_numbers,
        "source_sha256": source_sha,
        "figures_data": ch_figures,
        "page_spans": spans
    }
    
    master_doc = DentalDocument(
        id=f"{item['id']}-FULL",
        title=item["title"],
        content=full_content,
        category=CATEGORY,
        keywords=extract_keywords(item["title"] + " " + full_content),
        metadata=metadata
    )
    
    return master_doc, spans


def chunk_chapter_internal(
    master_doc: DentalDocument,
    page_spans: List[Dict[str, Any]],
    all_figures: List[Dict[str, Any]],
    tokenizer,
    max_tokens: int = 380,
    overlap_words: int = 16
) -> List[DentalDocument]:
    """Cắt chunks chuẩn xác theo thuật toán sliding window của internal_v1."""
    words = list(re.finditer(r"\S+", master_doc.content))
    if not words:
        return []
        
    chunks = []
    start = 0
    base_id = master_doc.id.replace("-FULL", "")

    while start < len(words):
        chunk_idx = len(chunks) + 1
        begin = words[start].start()
        low, high, end = start + 1, len(words), start

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
            # Fallback nếu 1 từ đơn lẻ dài vượt mức
            end = start + 1

        stop = words[end - 1].end()
        chunk_content = master_doc.content[begin:stop]

        # Xác định các trang PDF chứa chunk này
        pages_covered = []
        for span in page_spans:
            if max(begin, span["start"]) < min(stop, span["end"]):
                pages_covered.append(span["pdf_page"])

        p_start = min(pages_covered) if pages_covered else master_doc.metadata.get("page_start", 1)
        p_end = max(pages_covered) if pages_covered else p_start

        # Sub-title dựa theo dòng đầu tiên của chunk
        first_line = chunk_content.splitlines()[0].strip()
        first_line_clean = re.sub(r"^[■•\s\-\.]+", "", first_line)
        if len(first_line_clean) > 60:
            first_line_clean = first_line_clean[:57] + "..."
        chunk_title = f"{master_doc.title} - {first_line_clean}" if first_line_clean else master_doc.title

        chunk_figures = [f for f in all_figures if f["page"] in pages_covered]
        token_count = len(tokenizer.encode("passage: " + master_doc.title + "\n" + chunk_content, verbose=False))

        chunk_meta = {
            "source": "Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
            "source_name": "Dental Public Health & Research (Christine Nathe)",
            "source_type": "internal",
            "warehouse": "internal",
            "chapter_number": master_doc.metadata.get("chapter_number", 0),
            "chapter_title": master_doc.title,
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
            id=f"{base_id}-{chunk_idx:04d}",
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
        start = max(start + 1, end - min(overlap_words, end - start - 1))

    return chunks


def build_dph_corpus_records(
    master_doc: DentalDocument,
    chunks: List[DentalDocument],
    source_sha: str
) -> Tuple[PublicDocument, List[PublicChunk]]:
    """Xây dựng PublicDocument và PublicChunk theo chuẩn dataset internal_v1."""
    doc_id = master_doc.id.replace("-FULL", "")
    builder = DocumentBuilder(doc_id)

    sec = builder.section(
        title=master_doc.title,
        locator=Locator(path="data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf", pdf_page=master_doc.metadata["page_start"])
    )

    for c in chunks:
        loc = Locator(
            path="data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
            pdf_page=c.metadata.get("page_start", 1)
        )
        builder.block(sec, kind="paragraph", text=c.content, locator=loc)

    sections = builder.finish_sections()

    doc = make_document(
        document_id=doc_id,
        title=master_doc.title,
        source="dph",
        source_type="internal",
        language="vi" if "Vietnamese" in master_doc.title else "en",
        source_url="data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
        raw=RawSource(
            path="data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf",
            sha256=source_sha,
            media_type="application/pdf"
        ),
        processing=Processing(
            parser="dph-chapter-parser",
            parser_version="1.0.1",
            method="pdf-sliding-window-tokens-internal"
        ),
        sections=sections
    )

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
            source="dph",
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
