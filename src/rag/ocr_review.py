"""Áp dụng các chỗ sửa OCR đã đối chiếu bản scan, theo đúng trang và offset."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

NORMALIZATION_VERSION = "nfc-whitespace-glyphs-v1"
DEFAULT_REVIEW = Path(__file__).resolve().parents[2] / "data/curated/rhm_ocr_corrections.json"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_ocr_text(text):
    text = unicodedata.normalize("NFC", text.replace("ƣ", "ư").replace("Ƣ", "Ư"))
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def apply_reviewed_corrections(raw_pages, source_sha, review_path=DEFAULT_REVIEW):
    """Dừng nếu nguồn hoặc chữ trước khi sửa không còn khớp bản đã rà."""
    pages = {number: normalize_ocr_text(text) for number, text in raw_pages.items()}
    if review_path is None:
        return pages, {"sha256": None, "pages": {}, "scope": "none"}
    data_bytes = Path(review_path).read_bytes()
    review = json.loads(data_bytes)
    if review.get("schema_version") != 1 or review.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError("Phiên bản bảng sửa OCR không phù hợp")
    if review.get("source_sha256") != source_sha:
        raise ValueError("Bảng sửa OCR không khớp PDF nguồn")
    reviewed_pages = {}
    for record in review["pages"]:
        number = record["pdf_page"]
        if number not in pages or number in reviewed_pages:
            raise ValueError(f"Trang OCR không hợp lệ hoặc trùng: {number}")
        text = pages[number]
        if digest(text) != record["normalized_text_sha256"]:
            raise ValueError(f"Chữ OCR đã thay đổi ở trang {number}; cần rà lại chỗ sửa")
        parts, cursor, shift, changes = [], 0, 0, []
        for change in record["changes"]:
            start, end = change["start"], change["end"]
            if not isinstance(start, int) or not isinstance(end, int) or not cursor <= start < end <= len(text):
                raise ValueError(f"Offset sửa OCR trùng hoặc không hợp lệ ở trang {number}")
            if text[start:end] != change["before"] or not change["after"].strip() or change["before"] == change["after"]:
                raise ValueError(f"Nội dung sửa OCR không khớp ở trang {number}, offset {start}")
            parts.extend([text[cursor:start], change["after"]])
            changes.append({**change, "id": f"p{number}-{start}", "corrected_start": start + shift,
                            "corrected_end": start + shift + len(change["after"])})
            cursor = end
            shift += len(change["after"]) - (end - start)
        parts.append(text[cursor:])
        pages[number] = "".join(parts)
        reviewed_pages[number] = {"changes": changes, "layout_flags": record.get("layout_flags", [])}
    return pages, {"sha256": hashlib.sha256(data_bytes).hexdigest(), "pages": reviewed_pages,
                   "scope": review["review_scope"]}
