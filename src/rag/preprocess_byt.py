"""Data Preprocessing Pipeline for Ministry of Health Dental Clinical Guidelines (2hdrnghmmt.pdf).

This module extracts, normalizes, structures, and converts all 57 clinical disease guidelines
from the Ministry of Health (Decision 3108/QĐ-BYT) into standardized DentalDocument objects
ready for Hybrid RAG (ChromaDB Vector Store + BM25).
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

logger = get_logger("rag.preprocess_byt")

# 57 Specialty classification mapping based on Vietnamese Dental Curriculum
SPECIALTY_MAPPING: Dict[int, str] = {
    1: "Phẫu thuật miệng & Tiểu phẫu",          # Răng khôn mọc lệch
    2: "Phục hình răng",                         # Mất răng toàn bộ
    3: "Phục hình răng",                         # Mất răng từng phần
    4: "Nha khoa trẻ em",                        # Sâu răng sữa
    5: "Nha khoa trẻ em & Nội nha",              # Viêm tủy răng sữa
    6: "Bệnh lý nha chu",                        # Viêm lợi do mảng bám
    7: "Bệnh lý nha chu",                        # Viêm quanh răng tiến triển chậm
    8: "Bệnh lý nha chu (Cấp tính)",             # Viêm lợi loét hoại tử cấp
    9: "Bệnh lý sâu răng",                       # Bệnh sâu răng
    10: "Tổn thương mô cứng răng",               # Tổn thương mô cứng không do sâu
    11: "Bệnh lý tủy răng & Nội nha",            # Viêm tủy răng
    12: "Bệnh lý cuống răng & Nội nha",          # Viêm quanh cuống răng
    13: "Chỉnh hình răng mặt (Khớp cắn loại I)", # Sai khớp cắn loại I
    14: "Chỉnh hình răng mặt (Khớp cắn loại II)",# Sai khớp cắn loại II quá phát HT
    15: "Chỉnh hình răng mặt (Khớp cắn loại II)",# Sai khớp cắn loại II kém PT HD
    16: "Chỉnh hình răng mặt (Khớp cắn loại II)",# Sai khớp cắn loại II do xương hai hàm
    17: "Chỉnh hình răng mặt (Khớp cắn loại II)",# Sai khớp cắn loại II tiểu loại I do răng
    18: "Chỉnh hình răng mặt (Khớp cắn loại II)",# Sai khớp cắn loại II tiểu loại II do răng
    19: "Chỉnh hình răng mặt (Khớp cắn loại III)",# Giả khớp cắn loại III
    20: "Chỉnh hình răng mặt (Khớp cắn hở)",     # Khớp cắn hở
    21: "Chỉnh hình răng mặt (Cắn chéo)",        # Cắn chéo
    22: "Chỉnh hình răng mặt (Khớp cắn loại III)",# Sai khớp cắn loại III kém PT HT
    23: "Dị tật bẩm sinh & Phẫu thuật hàm mặt",  # Khe hở môi
    24: "Dị tật bẩm sinh & Phẫu thuật hàm mặt",  # Khe hở vòm miệng
    25: "Nang vùng hàm mặt",                     # Nang thân răng
    26: "Nang vùng hàm mặt & Tuyến nước bọt",    # Nang nhái sàn miệng
    27: "Bệnh lý tuyến nước bọt",                # Nang tuyến nước bọt dưới hàm & dưới lưỡi
    28: "Nang vùng hàm mặt",                     # Nang khe mang
    29: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe vùng cơ cắn
    30: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe má
    31: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe vùng dưới hàm
    32: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe vùng sàn miệng
    33: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe vùng mang tai
    34: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Áp xe thành bên họng
    35: "Nhiễm trùng hàm mặt (Cấp cứu)",         # Viêm tấy lan tỏa vùng hàm mặt
    36: "Khối u xương hàm",                      # U men xương hàm
    37: "Khối u do răng",                        # U răng
    38: "Khối u xương hàm",                      # U xương răng
    39: "Khối u lành tính hàm mặt",              # U máu ở trẻ em
    40: "Khối u lành tính hàm mặt",              # U bạch mạch
    41: "Khối u lành tính hàm mặt",              # U xơ thần kinh
    42: "Bệnh lý tuyến nước bọt",                # Viêm tuyến nước bọt mang tai do virus (Quai bị)
    43: "Bệnh lý tuyến nước bọt",                # Viêm tuyến nước bọt mang tai mạn tính
    44: "Bệnh lý tuyến nước bọt",                # Viêm tuyến nước bọt dưới hàm do sỏi
    45: "Khối u tuyến nước bọt",                 # U hỗn hợp tuyến nước bọt mang tai
    46: "Khối u tuyến nước bọt",                 # U tuyến nước bọt dưới hàm
    47: "Khối u tuyến nước bọt",                 # U tuyến nước bọt vòm miệng
    48: "Thần kinh hàm mặt",                     # Đau dây thần kinh V
    49: "Ung thư hàm mặt (Ung bướu)",            # Ung thư lưỡi
    50: "Ung thư hàm mặt (Ung bướu)",            # Ung thư sàn miệng
    51: "Ung thư hàm mặt (Ung bướu)",            # Ung thư tuyến nước bọt mang tai
    52: "Cấy ghép Implant & Phục hình",          # Viêm quanh Implant
    53: "Chấn thương hàm mặt (Cấp cứu)",         # Chấn thương phần mềm vùng hàm mặt
    54: "Chấn thương hàm mặt (Cấp cứu)",         # Gãy xương hàm dưới
    55: "Chấn thương hàm mặt (Cấp cứu)",         # Gãy xương hàm trên
    56: "Chấn thương hàm mặt (Cấp cứu)",         # Gãy xương gò má cung tiếp
    57: "Bệnh lý khớp thái dương hàm (TMJ)",     # Dính khớp thái dương hàm
}

EMERGENCY_DISEASES = {8, 29, 30, 31, 32, 33, 34, 35, 48, 53, 54, 55, 56}


def normalize_vietnamese_text(text: str) -> str:
    """Normalize font quirks and repair fragmented syllables from PDF text extraction."""
    # 1. Horn characters replacement (old Word font export artifact)
    text = text.replace('ƣ', 'ư').replace('Ƣ', 'Ư')
    text = unicodedata.normalize('NFC', text)

    # 2. Repair non-word consonant prefix followed by vowel syllable (e.g. 'tr ạng' -> 'trạng', 'm ất' -> 'mất')
    non_words = r'(?:ngh|ng|nh|th|ch|ph|kh|tr|gi|qu|[bcdđghklmnprstvxBCĐDGHIKLMNPQRSTVX])'
    vowels_start = r'[aăâeêioôơuưyàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹáắấéếíóốớúứýạặậẹệịọộợụựỵ]'
    pattern = r'\b(' + non_words + r')\s+(' + vowels_start + r'[a-zA-Zà-ỹÀ-Ỹ]*)\b'
    text = re.sub(pattern, r'\1\2', text)

    # 3. Repair common split vowel pairs
    vowel_prefixes = r'(?:thư|phư|dư|trư|nguy|ho|tho|quy|khuy|li|ti|chuy|nghi|hi|đi|chu|to|ho|đo|bản|bi|giải)'
    text = re.sub(r'\b(' + vowel_prefixes + r')\s+([a-zA-Zà-ỹÀ-Ỹ]+)\b', r'\1\2', text, flags=re.IGNORECASE)

    # 4. Standardize whitespace & line endings
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_keywords_from_text(title: str, text: str, category: str) -> List[str]:
    """Extract dental domain keywords and clinical entities for BM25 keyword matching."""
    keywords = set()

    # Add words from title
    clean_title = re.sub(r'^\d+\.\s*', '', title).strip().lower()
    keywords.add(clean_title)
    for word in re.split(r'[\s\-\(\)\/,]+', clean_title):
        if len(word) >= 3 and word not in {"các", "của", "cho", "trong", "loại", "tiểu", "dạng"}:
            keywords.add(word)

    # Dental domain terminology vocabulary to look for
    domain_vocab = [
        "răng khôn", "răng số 8", "răng số 7", "răng số 6", "răng cửa", "răng nanh", "răng hàm sữa",
        "viêm tủy", "chữa tủy", "buồng tủy", "tủy hoại tử", "viêm quanh cuống", "áp xe", "lợi trùm",
        "viêm nha chu", "viêm lợi", "chảy máu chân răng", "tụt nướu", "mảng bám", "cao răng",
        "mòn cổ răng", "ê buốt", "lỗ sâu", "tái khoáng", "chống ê buốt", "sâu men", "sâu ngà",
        "implant", "mất răng", "hàm tháo lắp", "cầu răng", "chụp sứ", "mão răng",
        "sai khớp cắn", "khớp cắn hở", "cắn chéo", "khớp cắn loại ii", "khớp cắn loại iii", "niềng răng",
        "khớp thái dương hàm", "há miệng hạn chế", "cơ cắn", "dây chằng",
        "u men", "u răng", "nang xương hàm", "nang nhái", "nang khe mang", "viêm tấy lan tỏa",
        "gãy xương hàm", "chấn thương", "x quang", "panorama", "conebeam ct", "cephalometrics",
        "fluor", "mta", "gic", "composite", "canxi hydroxit", "kháng sinh", "giảm đau"
    ]

    lower_text = text.lower()
    for term in domain_vocab:
        if term in lower_text:
            keywords.add(term)

    # Add category
    keywords.add(category.lower())

    return sorted(list(keywords))


def parse_disease_sections(full_disease_text: str) -> Dict[str, str]:
    """Parse standard Ministry of Health disease sections:

    I. ĐỊNH NGHĨA
    II. NGUYÊN NHÂN
    III. CHẨN ĐOÁN (Lâm sàng, Cận lâm sàng, Chẩn đoán phân biệt)
    IV. ĐIỀU TRỊ (Nguyên tắc, Điều trị cụ thể)
    V. TIÊN LƯỢNG VÀ BIẾN CHỨNG
    VI. PHÒNG BỆNH
    """
    sections = {}

    # Define standard Roman numeral headers
    section_patterns = [
        ("I", r"I\.\s*ĐỊNH\s*NGHĨA"),
        ("II", r"II\.\s*NGUYÊN\s*NHÂN"),
        ("III", r"III\.\s*CHẨN\s*ĐOÁN"),
        ("IV", r"IV\.\s*ĐIỀU\s*TRỊ"),
        ("V", r"V\.\s*TIÊN\s*LƯỢNG(?:\s*VÀ\s*BIẾN\s*CHỨNG)?"),
        ("VI", r"VI\.\s*PHÒNG\s*BỆNH"),
    ]

    # Find start indices of each section
    matches = []
    for key, pattern in section_patterns:
        m = re.search(pattern, full_disease_text, re.IGNORECASE)
        if m:
            matches.append((m.start(), key, m.group(0)))

    matches.sort(key=lambda x: x[0])

    for idx, (start_idx, key, header_str) in enumerate(matches):
        end_idx = matches[idx + 1][0] if idx + 1 < len(matches) else len(full_disease_text)
        sec_content = full_disease_text[start_idx:end_idx].strip()
        sections[key] = sec_content

    return sections


class BYTDataPreprocessor:
    """Preprocessor for Ministry of Health Dental Clinical Guidelines PDF."""

    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_full_text_with_pages(self) -> List[Tuple[int, str]]:
        """Read PDF pages 8 to 202 (main clinical guidelines content)."""
        logger.info(f"Đang đọc file PDF: {self.pdf_path}...")
        reader = pypdf.PdfReader(self.pdf_path)
        pages_data = []

        # Pages 8 to 202 are 0-indexed 7 to 201
        for p_idx in range(7, 202):
            if p_idx < len(reader.pages):
                raw_text = reader.pages[p_idx].extract_text() or ""
                cleaned = normalize_vietnamese_text(raw_text)
                pages_data.append((p_idx + 1, cleaned))

        logger.info(f"Đã trích xuất thành công {len(pages_data)} trang hướng dẫn lâm sàng.")
        return pages_data

    def detect_diseases(self, pages_data: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Detect boundaries of all 57 diseases across the pages."""
        # Join text with page markers
        combined_text = ""
        for p_num, p_text in pages_data:
            # Strip isolated page numbers at the beginning
            lines = p_text.split('\n')
            filtered_lines = [l for l in lines if not re.match(r'^\s*\d+\s*$', l)]
            combined_text += f"\n[PAGE_{p_num}]\n" + "\n".join(filtered_lines)

        disease_header_regex = re.compile(
            r'^\s*(\d{1,2})\s*\.\s*([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ\s\-\(\)\/]{4,})',
            re.MULTILINE
        )

        matches = []
        for m in disease_header_regex.finditer(combined_text):
            num = int(m.group(1))
            title_candidate = m.group(2).strip()
            # Clean title
            title_clean = re.sub(r'\s+', ' ', title_candidate)
            # Filter out false matches
            if 1 <= num <= 57 and not any(title_clean.startswith(x) for x in ["NGUYÊN", "ĐIỀU", "CHẨN", "TIÊN"]):
                matches.append((m.start(), num, title_clean))

        # De-duplicate by keeping the first occurrence of each disease number
        unique_matches = []
        seen_nums = set()
        for pos, num, title in matches:
            if num not in seen_nums:
                unique_matches.append((pos, num, title))
                seen_nums.add(num)

        unique_matches.sort(key=lambda x: x[0])
        logger.info(f"Đã xác định được {len(unique_matches)}/57 bệnh lý nha khoa chuyên khoa.")

        diseases = []
        for i, (pos, num, title) in enumerate(unique_matches):
            next_pos = unique_matches[i + 1][0] if i + 1 < len(unique_matches) else len(combined_text)
            raw_disease_text = combined_text[pos:next_pos].strip()

            # Find page range
            page_tags = re.findall(r'\[PAGE_(\d+)\]', raw_disease_text)
            pages = [int(p) for p in page_tags]
            start_page = min(pages) if pages else 8
            end_page = max(pages) if pages else start_page

            # Remove page tags from disease content
            clean_content = re.sub(r'\[PAGE_\d+\]\s*', '', raw_disease_text)
            clean_content = normalize_vietnamese_text(clean_content)

            # Clean Title format: e.g. "Răng khôn mọc lệch"
            # Title usually comes before "I. ĐỊNH NGHĨA"
            title_match = re.search(r'^\s*\d+\.\s*(.*?)(?=\n\s*I\.|\n\s*\(|\n\s*[A-Z]|\n\n)', clean_content, re.DOTALL)
            if title_match:
                extracted_title = title_match.group(1).strip()
                extracted_title = re.sub(r'\s+', ' ', extracted_title)
            else:
                extracted_title = title

            category = SPECIALTY_MAPPING.get(num, "Chuyên khoa Răng Hàm Mặt")
            is_emergency = num in EMERGENCY_DISEASES

            diseases.append({
                "index": num,
                "title": extracted_title.title(),
                "raw_title": title,
                "category": category,
                "is_emergency": is_emergency,
                "start_page": start_page,
                "end_page": end_page,
                "content": clean_content
            })

        return diseases

    def generate_datasets(self, diseases: List[Dict[str, Any]]) -> Tuple[List[DentalDocument], List[DentalDocument]]:
        """Generate both Master Disease Documents and Granular Section Chunks."""
        master_docs: List[DentalDocument] = []
        section_docs: List[DentalDocument] = []

        for d in diseases:
            idx = d["index"]
            title = d["title"]
            category = d["category"]
            content = d["content"]
            is_emergency = d["is_emergency"]
            start_p = d["start_page"]
            end_p = d["end_page"]

            keywords = extract_keywords_from_text(title, content, category)

            # 1. Master Disease Document (Complete Guideline)
            master_id = f"BYT-DIS-{idx:03d}"
            master_doc = DentalDocument(
                id=master_id,
                title=f"{idx}. {title}",
                content=content,
                category=category,
                keywords=keywords,
                metadata={
                    "source": "2hdrnghmmt.pdf",
                    "source_name": "Bộ Y Tế - Hướng dẫn Chẩn đoán và Điều trị RHM (QĐ 3108/QĐ-BYT)",
                    "disease_index": idx,
                    "disease_title": title,
                    "doc_level": "master",
                    "is_emergency": is_emergency,
                    "page_start": start_p,
                    "page_end": end_p
                }
            )
            master_docs.append(master_doc)

            # 2. Granular Section Chunks
            sections = parse_disease_sections(content)
            section_meta_map = {
                "I": ("Định nghĩa & Nguyên nhân", ["I", "II"]),
                "III": ("Chẩn đoán lâm sàng & Phân biệt", ["III"]),
                "IV": ("Phác đồ điều trị & Quy trình", ["IV"]),
                "V": ("Tiên lượng, Biến chứng & Phòng bệnh", ["V", "VI"])
            }

            for sec_key, (sec_label, include_keys) in section_meta_map.items():
                sec_text_parts = [sections[k] for k in include_keys if k in sections]
                if not sec_text_parts:
                    continue

                sec_content = "\n\n".join(sec_text_parts)
                sec_id = f"BYT-SEC-{idx:03d}-{sec_key}"
                sec_title = f"{title} - {sec_label}"
                sec_keywords = extract_keywords_from_text(sec_title, sec_content, category)

                has_differential = "chẩn đoán phân biệt" in sec_content.lower()

                sec_doc = DentalDocument(
                    id=sec_id,
                    title=sec_title,
                    content=sec_content,
                    category=category,
                    keywords=sec_keywords,
                    metadata={
                        "source": "2hdrnghmmt.pdf",
                        "source_name": "Bộ Y Tế - Hướng dẫn Chẩn đoán và Điều trị RHM",
                        "parent_id": master_id,
                        "disease_index": idx,
                        "disease_title": title,
                        "doc_level": "section",
                        "section_code": sec_key,
                        "section_name": sec_label,
                        "is_emergency": is_emergency,
                        "has_differential_diagnosis": has_differential,
                        "page_start": start_p,
                        "page_end": end_p
                    }
                )
                section_docs.append(sec_doc)

        logger.info(f"Đã tạo {len(master_docs)} tài liệu Bệnh Toàn văn và {len(section_docs)} Chunks Chuyên mục.")
        return master_docs, section_docs

    def run(self) -> Dict[str, Any]:
        """Execute full preprocessing pipeline and save JSON files."""
        pages_data = self.extract_full_text_with_pages()
        diseases = self.detect_diseases(pages_data)
        master_docs, section_docs = self.generate_datasets(diseases)

        # Save paths
        master_path = os.path.join(self.output_dir, "byt_diseases_full.json")
        sections_path = os.path.join(self.output_dir, "byt_diseases_sections.json")

        # Serializing to JSON
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in master_docs], f, ensure_ascii=False, indent=2)

        with open(sections_path, "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in section_docs], f, ensure_ascii=False, indent=2)

        stats = {
            "total_diseases": len(diseases),
            "master_documents": len(master_docs),
            "section_documents": len(section_docs),
            "files_saved": {
                "full_docstore": master_path,
                "sections_search_index": sections_path
            }
        }

        logger.info(f"Hoàn thành tiền xử lý: Đã lưu {len(section_docs)} chunks tìm kiếm và {len(master_docs)} phác đồ toàn văn tại {self.output_dir}")
        return stats


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_file = os.path.join(base_dir, "data", "raw", "2hdrnghmmt.pdf")
    output_directory = os.path.join(base_dir, "data", "processed")

    preprocessor = BYTDataPreprocessor(pdf_file, output_directory)
    results = preprocessor.run()

    print("\n" + "=" * 60)
    print(" BÁO CÁO KẾT QUẢ TIỀN XỬ LÝ DỮ LIỆU BỘ Y TẾ (2hdrnghmmt.pdf)")
    print("=" * 60)
    print(f"- Tổng số bệnh lý bóc tách thành công: {results['total_diseases']} / 57 bệnh")
    print(f"- File DocStore toàn văn (57 bệnh): {results['files_saved']['full_docstore']}")
    print(f"- File Index tìm kiếm RAG ({results['section_documents']} Chunks): {results['files_saved']['sections_search_index']}")
    print("=" * 60)

