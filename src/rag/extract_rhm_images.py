"""Extract clinical diagrams and anatomical figures from Gnathology Textbook (RHM1 khop can hoc.pdf).

Since RHM1 is a scanned book with an OCR text layer, this module locates figure captions
(e.g., 'Hình 2-3', 'Hình 4-1') and automatically crops the high-resolution diagram region
between text blocks, saving individual figures to data/images/rhm1/ and linking them to RAG metadata.
"""

import os
import re
import sys
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pymupdf

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT_DIR / "data" / "raw" / "RHM1 khop can hoc.pdf"
OUTPUT_IMAGES_DIR = ROOT_DIR / "data" / "images" / "rhm1"
SECTIONS_JSON_PATH = ROOT_DIR / "data" / "processed" / "khop_can_hoc_sections.json"
CHAPTERS_JSON_PATH = ROOT_DIR / "data" / "processed" / "khop_can_hoc_chapters.json"

FIG_REGEX = re.compile(
    r'(?:Hình|H\.)\s*(\d+[-–\.]\d+[a-zA-Z]?)(?:[\.:\s]\s*([^\n\r]*)|$)',
    re.IGNORECASE
)

CHAPTER_METADATA = [
    {"num": 1, "title": "Mở đầu - Giới thiệu môn học Cắn khớp", "start_page": 11, "end_page": 17},
    {"num": 2, "title": "Giải phẫu chức năng hệ thống nhai", "start_page": 18, "end_page": 40},
    {"num": 3, "title": "Khớp cắn và quan niệm về khớp cắn", "start_page": 41, "end_page": 72},
    {"num": 4, "title": "Động học cắn khớp", "start_page": 73, "end_page": 89},
    {"num": 5, "title": "Hoạt động chức năng và cận chức năng", "start_page": 90, "end_page": 109},
    {"num": 6, "title": "Vận động và các vị trí của hàm dưới trong thực hành", "start_page": 110, "end_page": 122},
    {"num": 7, "title": "Khám cắn khớp và hệ thống nhai", "start_page": 123, "end_page": 146},
    {"num": 8, "title": "Khám các tiếp xúc cắn khớp", "start_page": 147, "end_page": 180},
    {"num": 9, "title": "Đại cương điều trị cắn khớp và mài điều chỉnh khớp cắn", "start_page": 181, "end_page": 206},
    {"num": 10, "title": "Giá khớp và sử dụng giá khớp", "start_page": 207, "end_page": 235},
    {"num": 11, "title": "Cung mặt và lên giá khớp", "start_page": 236, "end_page": 257},
    {"num": 12, "title": "Máng nhai", "start_page": 258, "end_page": 268},
    {"num": 13, "title": "Thực hành máng nhai và điều trị rối loạn TMD", "start_page": 269, "end_page": 284},
    {"num": 14, "title": "Bảng thuật ngữ Cắn khớp học chuyên sâu", "start_page": 285, "end_page": 294}
]


def sanitize_slug(text: str) -> str:
    """Create a URL/filename-friendly slug from Vietnamese text."""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_').lower()
    return text[:45]


def extract_figures_from_page(
    page: pymupdf.Page,
    page_num: int,
    chapter_num: int,
    dpi: int = 200
) -> List[Dict[str, Any]]:
    """Detect and crop all figure diagrams on a scanned PDF page."""
    blocks = page.get_text("blocks")
    page_w = page.rect.width
    page_h = page.rect.height

    # Find caption blocks
    captions = []
    for b_idx, b in enumerate(blocks):
        text = b[4].strip()
        match = FIG_REGEX.search(text)
        if match:
            fig_no = match.group(1).replace('–', '-').replace('.', '-')
            raw_caption = match.group(2).strip()
            # If caption spans multiple lines in the block, collect them
            caption_lines = [l.strip() for l in text.split('\n') if l.strip()]
            full_caption = " ".join(caption_lines)
            captions.append({
                "block_idx": b_idx,
                "fig_no": fig_no,
                "label": f"Hình {fig_no}",
                "caption": full_caption,
                "bbox": pymupdf.Rect(b[0], b[1], b[2], b[3]),
                "raw_text": text
            })

    if not captions:
        return []

    # Sort captions from top to bottom
    captions.sort(key=lambda c: c["bbox"].y0)

    extracted = []
    for idx, cap in enumerate(captions):
        cap_rect = cap["bbox"]
        fig_no = cap["fig_no"]

        # Determine diagram boundaries
        # 1. Top boundary: look for preceding text blocks
        prev_blocks = [
            b for b in blocks
            if b[3] <= cap_rect.y0 - 8 and (b[2] - b[0] > 180 or len(b[4].strip()) > 50)
        ]

        if idx > 0:
            # If multiple figures on page, top is above this caption but below previous caption
            prev_cap = captions[idx - 1]
            candidate_tops = [b[3] for b in prev_blocks if b[3] > prev_cap["bbox"].y1]
            fig_top = max(candidate_tops) + 5 if candidate_tops else prev_cap["bbox"].y1 + 5
        else:
            fig_top = max([b[3] for b in prev_blocks]) + 5 if prev_blocks else 30.0

        # 2. Bottom boundary: includes the caption block
        fig_bottom = min(cap_rect.y1 + 10, page_h - 25)

        # 3. Horizontal bounds: standard page margins
        fig_left = 35.0
        fig_right = min(page_w - 30.0, 485.0)

        # Validate crop region height
        if fig_bottom - fig_top < 60:
            # Fallback height if text blocks are tight
            fig_top = max(30.0, cap_rect.y0 - 200.0)

        crop_rect = pymupdf.Rect(fig_left, fig_top, fig_right, fig_bottom)

        # Render high-resolution crop
        pix = page.get_pixmap(clip=crop_rect, dpi=dpi)

        slug = sanitize_slug(cap["caption"][:40])
        filename = f"rhm1_ch{chapter_num:02d}_p{page_num:03d}_fig{fig_no}_{slug}.jpg"
        dest_path = OUTPUT_IMAGES_DIR / filename
        rel_path = f"data/images/rhm1/{filename}"

        pix.save(str(dest_path))

        extracted.append({
            "fig_no": fig_no,
            "label": cap["label"],
            "caption": cap["caption"],
            "local_path": rel_path,
            "page": page_num,
            "chapter": chapter_num,
            "width": pix.width,
            "height": pix.height,
            "filesize_bytes": dest_path.stat().st_size
        })

    return extracted


def update_rhm_sections_metadata(all_figures: List[Dict[str, Any]]) -> int:
    """Link extracted figures to corresponding sections in khop_can_hoc_sections.json."""
    if not SECTIONS_JSON_PATH.exists():
        return 0

    with open(SECTIONS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Group figures by chapter
    figs_by_chap: Dict[int, List[Dict[str, Any]]] = {}
    for fig in all_figures:
        ch = fig["chapter"]
        figs_by_chap.setdefault(ch, []).append(fig)

    updated_count = 0
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        ch_num = meta.get("chapter_number")
        chunk_content = chunk.get("content", "")

        if ch_num in figs_by_chap:
            matched_figs = []
            for fig in figs_by_chap[ch_num]:
                fig_no = fig.get("fig_no", fig["label"].split()[-1])
                # Strict match: "Hình 4-1" or "H. 4-1", not followed by digits or letters
                pattern = rf'(?:Hình|H\.)\s*{re.escape(fig_no)}(?![0-9a-zA-Z])'
                if re.search(pattern, chunk_content, re.IGNORECASE):
                    matched_figs.append({
                        "label": fig["label"],
                        "caption": fig["caption"],
                        "local_path": fig["local_path"],
                        "page": fig["page"]
                    })

            if matched_figs:
                meta["figures_data"] = matched_figs
                updated_count += 1

    with open(SECTIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return updated_count


def update_rhm_chapters_metadata(all_figures: List[Dict[str, Any]]) -> int:
    """Link extracted figures to corresponding master chapter documents in khop_can_hoc_chapters.json."""
    if not CHAPTERS_JSON_PATH.exists():
        return 0

    with open(CHAPTERS_JSON_PATH, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    figs_by_chap: Dict[int, List[Dict[str, Any]]] = {}
    for fig in all_figures:
        ch = fig["chapter"]
        figs_by_chap.setdefault(ch, []).append({
            "label": fig["label"],
            "caption": fig["caption"],
            "local_path": fig["local_path"],
            "page": fig["page"]
        })

    updated_count = 0
    for doc in chapters:
        meta = doc.get("metadata", {})
        ch_num = meta.get("chapter_number")
        if ch_num in figs_by_chap:
            meta["figures_data"] = figs_by_chap[ch_num]
            meta["total_figures"] = len(figs_by_chap[ch_num])
            updated_count += 1

    with open(CHAPTERS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False, indent=2)

    return updated_count


def run_extraction(start_page: int = 11, end_page: int = 294) -> Dict[str, Any]:
    """Execute figure extraction pipeline for selected page ranges."""
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(PDF_PATH))
    total_pages = len(doc)

    print(f"Bắt đầu bóc tách hình ảnh từ: {PDF_PATH.name} (Trang {start_page} đến {end_page})...")
    all_extracted_figs = []

    for p_num in range(start_page, min(end_page + 1, total_pages + 1)):
        # Determine chapter
        ch_num = 1
        for ch in CHAPTER_METADATA:
            if ch["start_page"] <= p_num <= ch["end_page"]:
                ch_num = ch["num"]
                break

        page = doc[p_num - 1]
        figs = extract_figures_from_page(page, page_num=p_num, chapter_num=ch_num)
        for f in figs:
            all_extracted_figs.append(f)
            print(f"  [+] Trang {p_num:03d} (Chương {ch_num:02d}) -> {f['label']} ({f['width']}x{f['height']} px) - {f['caption'][:60]}")

    print(f"\n[+] Tổng số hình ảnh bóc tách thành công: {len(all_extracted_figs)} ảnh.")

    # Update sections and chapters metadata
    updated_sections = update_rhm_sections_metadata(all_extracted_figs)
    print(f"[+] Đã liên kết hình ảnh vào {updated_sections} search chunks trong {SECTIONS_JSON_PATH.name}")

    updated_chapters = update_rhm_chapters_metadata(all_extracted_figs)
    print(f"[+] Đã liên kết hình ảnh vào {updated_chapters} master chapters trong {CHAPTERS_JSON_PATH.name}")

    return {
        "total_extracted": len(all_extracted_figs),
        "updated_sections": updated_sections,
        "updated_chapters": updated_chapters,
        "figures": all_extracted_figs
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" BÓC TÁCH TOÀN BỘ HÌNH ẢNH 13 CHƯƠNG GIÁO TRÌNH CẮN KHỚP HỌC (RHM1)")
    print("=" * 70)
    res = run_extraction(start_page=11, end_page=294)

    print("\n" + "=" * 70)
    print(" BÁO CÁO TỔNG KẾT HOÀN TẤT:")
    print(f"- Tổng số hình ảnh chuyên khoa đã bóc tách: {res['total_extracted']} ảnh")
    print(f"- Thư mục lưu trữ: {OUTPUT_IMAGES_DIR}")
    print(f"- Chunks tìm kiếm được cập nhật: {res['updated_sections']} chunks")
    print(f"- Chapters docstore được cập nhật: {res['updated_chapters']} chương")
    print("=" * 70)
