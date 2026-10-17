"""Data Preprocessing Pipeline for Gnathology & Occlusion Textbook (RHM1 khop can hoc.pdf).

This module extracts, cleans OCR artifacts, structures, and converts all 13 educational chapters
from the Dental Occlusion & Temporomandibular Joint (TMJ) textbook into standardized
DentalDocument objects ready for Hybrid RAG.
"""

import os
import re
import sys
import json
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import pypdf

from src.rag.base import DentalDocument
from src.core.logger import get_logger

logger = get_logger("rag.preprocess_rhm")

# Table of 13 Chapters and their page boundaries
CHAPTER_METADATA = [
    {
        "num": 1,
        "title": "Mở đầu - Giới thiệu môn học Cắn khớp",
        "start_page": 11,
        "end_page": 17,
        "topics": ["định nghĩa cắn khớp", "hệ thống nhai", "lịch sử cắn khớp học"]
    },
    {
        "num": 2,
        "title": "Giải phẫu chức năng hệ thống nhai",
        "start_page": 18,
        "end_page": 40,
        "topics": ["xương hàm trên", "xương hàm dưới", "cơ nhai", "cơ chân bướm", "khớp thái dương hàm", "lồi cầu", "đĩa khớp", "bao khớp"]
    },
    {
        "num": 3,
        "title": "Khớp cắn và quan niệm về khớp cắn",
        "start_page": 41,
        "end_page": 72,
        "topics": ["khớp cắn răng sữa", "khớp cắn răng vĩnh viễn", "lồng múi tối đa", "phân loại Angle", "cung răng"]
    },
    {
        "num": 4,
        "title": "Động học cắn khớp",
        "start_page": 73,
        "end_page": 89,
        "topics": ["động học hàm dưới", "vận động biên", "sơ đồ Posselt", "góc Bennett", "vận động sang bên", "vận động ra trước"]
    },
    {
        "num": 5,
        "title": "Hoạt động chức năng và cận chức năng",
        "start_page": 90,
        "end_page": 109,
        "topics": ["động tác nhai", "động tác nuốt", "cận chức năng", "nghiến răng", "siết chặt răng", "mòn răng do nghiến"]
    },
    {
        "num": 6,
        "title": "Vận động và các vị trí của hàm dưới trong thực hành",
        "start_page": 110,
        "end_page": 122,
        "topics": ["tương quan trung tâm CR", "khớp cắn trung tâm CO", "kích thước dọc cắn khớp VDO", "tư thế nghỉ sinh lý"]
    },
    {
        "num": 7,
        "title": "Khám cắn khớp và hệ thống nhai",
        "start_page": 123,
        "end_page": 146,
        "topics": ["hỏi bệnh sử cắn khớp", "khám khớp thái dương hàm", "tiếng kêu khớp", "khám cơ nhai", "sờ nắn cơ"]
    },
    {
        "num": 8,
        "title": "Khám các tiếp xúc cắn khớp",
        "start_page": 147,
        "end_page": 180,
        "topics": ["tiếp xúc cắn khớp", "tiếp xúc sớm", "cản trở cắn khớp", "giấy cắn", "hướng dẫn răng nanh", "hướng dẫn nhóm"]
    },
    {
        "num": 9,
        "title": "Đại cương điều trị cắn khớp và mài điều chỉnh khớp cắn",
        "start_page": 181,
        "end_page": 206,
        "topics": ["điều trị cắn khớp", "mài chỉnh khớp cắn", "quy tắc mài chỉnh", "mài chỉnh ở tương quan trung tâm", "loại bỏ cản trở cắn khớp"]
    },
    {
        "num": 10,
        "title": "Giá khớp và sử dụng giá khớp",
        "start_page": 207,
        "end_page": 235,
        "topics": ["giá khớp", "phân loại giá khớp", "giá khớp bản lề", "giá khớp bán thích ứng", "giá khớp thích ứng hoàn toàn", "cài đặt giá khớp"]
    },
    {
        "num": 11,
        "title": "Cung mặt và lên giá khớp",
        "start_page": 236,
        "end_page": 257,
        "topics": ["cung mặt", "ghi dấu cung mặt", "lên giá khớp hàm trên", "vô giá khớp", "sai lầm khi lên giá khớp"]
    },
    {
        "num": 12,
        "title": "Máng nhai",
        "start_page": 258,
        "end_page": 268,
        "topics": ["máng nhai", "phân loại máng nhai", "máng nhai thư giãn", "máng nhai định vị lại", "chỉ định máng nhai"]
    },
    {
        "num": 13,
        "title": "Thực hành máng nhai và điều trị rối loạn khớp thái dương hàm",
        "start_page": 269,
        "end_page": 284,
        "topics": ["điều trị rối loạn tmd", "chế tạo máng nhai", "chỉnh máng nhai", "vật lý trị liệu tmd", "tiên lượng tmd"]
    },
    {
        "num": 14,
        "title": "Bảng thuật ngữ Cắn khớp học chuyên sâu",
        "start_page": 285,
        "end_page": 294,
        "topics": ["thuật ngữ cắn khớp", "tương quan trung tâm", "khớp cắn lồng múi", "gysi", "bennett", "posselt"]
    }
]

# Vocabulary for Dental Occlusion keywords extraction
RHM_KEYWORD_VOCAB = [
    "khớp cắn", "cắn khớp", "hệ thống nhai", "khớp thái dương hàm", "tmd",
    "lồi cầu", "đĩa khớp", "bao khớp", "dây chằng", "cơ cắn", "cơ thái dương", "cơ chân bướm",
    "sơ đồ posselt", "góc bennett", "vận động biên", "vận động sang bên", "vận động ra trước",
    "tương quan trung tâm", "khớp cắn trung tâm", "lồng múi tối đa", "kích thước dọc", "tư thế nghỉ",
    "hướng dẫn răng nanh", "hướng dẫn nhóm", "cản trở cắn khớp", "tiếp xúc sớm",
    "mài chỉnh khớp cắn", "giấy cắn", "máng nhai", "máng nhai thư giãn", "nghiến răng",
    "giá khớp", "giá khớp bản lề", "giá khớp bán thích ứng", "cung mặt", "lên giá khớp"
]


def clean_rhm_ocr_text(text: str) -> str:
    """Normalize OCR text artifacts and correct known Vietnamese OCR misrecognitions."""
    if not text:
        return ""

    # Common OCR typo corrections
    replacements = [
        (r'\buù\b', 'và'),
        (r'\buà\b', 'và'),
        (r'\buê\b', 'về'),
        (r'\bcúa\b', 'của'),
        (r'\bbể\b', 'kể'),
        (r'\brỡng\b', 'răng'),
        (r'\bbhóp\b', 'khớp'),
        (r'\bshép\b', 'khớp'),
        (r'\bthái đương\b', 'thái dương'),
        (r'\bđương hàm\b', 'dương hàm'),
        (r'\buì\b', 'vi'),
        (r'\bhỳ\b', 'kỳ'),
        (r'\bchúe\b', 'chức'),
        (r'\bdưới\b', 'dưới'),
        (r'\bđướt\b', 'dưới'),
        (r'\bhàm đướt\b', 'hàm dưới'),
        (r'\bkháo cắn\b', 'khớp cắn'),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Normalize Unicode NFC
    text = unicodedata.normalize('NFC', text)

    # Fix horn characters
    text = text.replace('ƣ', 'ư').replace('Ƣ', 'Ư')

    # Remove repeated blank lines & multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_keywords_rhm(title: str, text: str) -> List[str]:
    """Extract relevant gnathology keywords from text."""
    keywords = set()

    # Add words from title
    clean_title = re.sub(r'^\d+\.\s*', '', title).strip().lower()
    keywords.add(clean_title)
    for w in re.split(r'[\s\-\(\)\/,:]+', clean_title):
        if len(w) >= 3 and w not in {"các", "của", "cho", "trong", "với", "những"}:
            keywords.add(w)

    lower_text = text.lower()
    for kw in RHM_KEYWORD_VOCAB:
        if kw in lower_text:
            keywords.add(kw)

    keywords.add("cắn khớp học & tmj")
    return sorted(list(keywords))


def chunk_chapter_content(
    chapter_num: int,
    chapter_title: str,
    chapter_text: str,
    start_page: int,
    end_page: int,
    target_chunk_size: int = 1800,
    overlap: int = 200
) -> List[DentalDocument]:
    """Split chapter text into cohesive, readable chunks of 500-800 tokens."""
    # Split paragraphs by double newline
    paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
    if not paragraphs:
        # Fallback to single newline split
        paragraphs = [p.strip() for p in chapter_text.split('\n') if len(p.strip()) > 30]

    chunks = []
    current_chunk = []
    current_len = 0
    chunk_index = 1

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len > target_chunk_size and current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            # Find subheader if any in chunk
            first_line = current_chunk[0].split('\n')[0].strip()
            sub_title = first_line[:50] if len(first_line) > 5 else f"Phần {chunk_index}"

            doc_id = f"CKH-CH{chapter_num:02d}-{chunk_index:02d}"
            title = f"Chương {chapter_num}: {chapter_title} - {sub_title}"
            keywords = extract_keywords_rhm(title, chunk_content)

            doc = DentalDocument(
                id=doc_id,
                title=title,
                content=chunk_content,
                category="Cắn khớp học & TMJ",
                keywords=keywords,
                metadata={
                    "source": "RHM1 khop can hoc.pdf",
                    "source_name": "Giáo trình Cắn khớp học - ĐH Y Dược TP.HCM",
                    "chapter_number": chapter_num,
                    "chapter_title": chapter_title,
                    "chunk_index": chunk_index,
                    "doc_level": "section",
                    "page_start": start_page,
                    "page_end": end_page
                }
            )
            chunks.append(doc)
            chunk_index += 1

            # Keep last paragraph for overlap
            current_chunk = [current_chunk[-1]] if len(current_chunk[-1]) < overlap else []
            current_len = sum(len(x) for x in current_chunk)

        current_chunk.append(p)
        current_len += p_len

    # Add final remaining chunk
    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        first_line = current_chunk[0].split('\n')[0].strip()
        sub_title = first_line[:50] if len(first_line) > 5 else f"Phần {chunk_index}"

        doc_id = f"CKH-CH{chapter_num:02d}-{chunk_index:02d}"
        title = f"Chương {chapter_num}: {chapter_title} - {sub_title}"
        keywords = extract_keywords_rhm(title, chunk_content)

        doc = DentalDocument(
            id=doc_id,
            title=title,
            content=chunk_content,
            category="Cắn khớp học & TMJ",
            keywords=keywords,
            metadata={
                "source": "RHM1 khop can hoc.pdf",
                "source_name": "Giáo trình Cắn khớp học - ĐH Y Dược TP.HCM",
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "chunk_index": chunk_index,
                "doc_level": "section",
                "page_start": start_page,
                "page_end": end_page
            }
        )
        chunks.append(doc)

    return chunks


class RHMDataPreprocessor:
    """Preprocessor for Gnathology & Occlusion Textbook (RHM1 khop can hoc.pdf)."""

    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute extraction, cleaning and chunking across all 13 chapters."""
        logger.info(f"Đang đọc file Giáo trình Cắn khớp học: {self.pdf_path}...")
        reader = pypdf.PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"Tổng số trang: {total_pages}. Đang trích xuất theo 13 chương...")

        all_section_docs: List[DentalDocument] = []
        all_chapter_docs: List[DentalDocument] = []

        for ch in CHAPTER_METADATA:
            ch_num = ch["num"]
            ch_title = ch["title"]
            start_p = ch["start_page"]
            end_p = min(ch["end_page"], total_pages)

            # Collect text for this chapter
            ch_pages_text = []
            for p_idx in range(start_p - 1, end_p):
                raw_text = reader.pages[p_idx].extract_text() or ""
                cleaned = clean_rhm_ocr_text(raw_text)
                if cleaned:
                    ch_pages_text.append(cleaned)

            full_chapter_text = "\n\n".join(ch_pages_text)
            if not full_chapter_text.strip():
                logger.warning(f"Chương {ch_num} không có text!")
                continue

            # 1. Full Chapter Document (Parent DocStore)
            master_id = f"CKH-CH{ch_num:02d}-FULL"
            master_title = f"Chương {ch_num}: {ch_title}"
            ch_keywords = extract_keywords_rhm(master_title, full_chapter_text)

            chapter_doc = DentalDocument(
                id=master_id,
                title=master_title,
                content=full_chapter_text,
                category="Cắn khớp học & TMJ",
                keywords=ch_keywords,
                metadata={
                    "source": "RHM1 khop can hoc.pdf",
                    "source_name": "Giáo trình Cắn khớp học - ĐH Y Dược TP.HCM",
                    "chapter_number": ch_num,
                    "chapter_title": ch_title,
                    "doc_level": "master",
                    "page_start": start_p,
                    "page_end": end_p
                }
            )
            all_chapter_docs.append(chapter_doc)

            # 2. Granular Section Chunks (Search Index)
            sec_chunks = chunk_chapter_content(
                chapter_num=ch_num,
                chapter_title=ch_title,
                chapter_text=full_chapter_text,
                start_page=start_p,
                end_page=end_p
            )
            for sc in sec_chunks:
                sc.metadata["parent_id"] = master_id
            all_section_docs.extend(sec_chunks)

            logger.info(f"Đã xử lý Chương {ch_num:2d}: '{ch_title}' -> Tạo {len(sec_chunks)} Chunks.")

        # Save to JSON
        sections_output_path = os.path.join(self.output_dir, "khop_can_hoc_sections.json")
        chapters_output_path = os.path.join(self.output_dir, "khop_can_hoc_chapters.json")

        with open(sections_output_path, "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in all_section_docs], f, ensure_ascii=False, indent=2)

        with open(chapters_output_path, "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in all_chapter_docs], f, ensure_ascii=False, indent=2)

        stats = {
            "total_chapters": len(all_chapter_docs),
            "total_search_chunks": len(all_section_docs),
            "files_saved": {
                "sections": sections_output_path,
                "chapters_docstore": chapters_output_path
            }
        }
        logger.info(f"Hoàn thành tiền xử lý Cắn khớp học: {len(all_section_docs)} chunks tìm kiếm lưu tại {sections_output_path}")
        return stats


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_file = os.path.join(base_dir, "data", "raw", "RHM1 khop can hoc.pdf")
    output_directory = os.path.join(base_dir, "data", "processed")

    preprocessor = RHMDataPreprocessor(pdf_file, output_directory)
    results = preprocessor.run()

    print("\n" + "=" * 60)
    print(" BÁO CÁO TIỀN XỬ LÝ GIÁO TRÌNH CẮN KHỚP HỌC (RHM1)")
    print("=" * 60)
    print(f"- Tổng số chương lý thuyết đã bóc tách: {results['total_chapters']} chương")
    print(f"- Tổng số Section Chunks tạo cho RAG: {results['total_search_chunks']} chunks")
    print(f"- File Chunks tìm kiếm: {results['files_saved']['sections']}")
    print(f"- File DocStore toàn văn: {results['files_saved']['chapters_docstore']}")
    print("=" * 60)
