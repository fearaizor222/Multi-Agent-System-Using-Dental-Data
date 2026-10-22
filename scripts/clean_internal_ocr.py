"""Sửa các mẫu OCR lặp lại đã đối chiếu; không thay số liệu bằng quy tắc chung."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from src.rag.ocr_review import DEFAULT_REVIEW, NORMALIZATION_VERSION, normalize_ocr_text

ROOT = Path(__file__).resolve().parents[1]
# Chỉ các mẫu đã đối chiếu trên bản scan; mỗi lần thay được ghi thành một patch.
RULES = [
    ("condyle", r"\b(lỗi|lôi|lổi|lãi|li)(\s+)(cầu|câu)\b|\b(lồi)(\s+)câu\b", None, [29, 30, 33, 83, 222, 249]),
    ("and", r"\b(?:uà|uù)\b", "và", [11, 216]),
    ("about", r"\b(?:uê|uề)\b", "về", [11]),
    ("joint", r"\b(?:bhớp|bhóp|hhớp|hhop|khóp)\b", "khớp", [11, 216]),
    ("temporal", r"\bthái(\s+)đương\b", None, [29, 30]),
    ("motion", r"\buận(\s+)động\b", None, [83]),
    ("with", r"\buới\b", "với", [216]),
    ("slope", r"\bđộ(\s+)đốc\b", None, [222, 249]),
    ("articular_surface", r"\bđiện(\s+)khớp\b", None, [30, 33]),
    ("articular_eminence", r"\b(?:lỗi|lôi|lổi|lãi)(\s+)khớp\b", None, [30]),
    ("coronoid", r"\bmổỏm(\s+)quạ\b", None, [29]),
    ("survival", r"\bsinh(\s+)tổn\b", None, [11]),
]


def replacement(rule, match, value):
    if rule == "condyle":
        value = "lồi" + (match.group(2) or match.group(5)) + "cầu"
    elif rule == "temporal":
        value = "thái" + match.group(1) + "dương"
    elif rule == "motion":
        value = "vận" + match.group(1) + "động"
    elif rule == "slope":
        value = "độ" + match.group(1) + "dốc"
    elif rule in {"articular_surface", "articular_eminence", "coronoid", "survival"}:
        head, tail = {"articular_surface": ("diện", "khớp"), "articular_eminence": ("lồi", "khớp"),
                      "coronoid": ("mỏm", "quạ"), "survival": ("sinh", "tồn")}[rule]
        value = head + match.group(1) + tail
    if match.group(0).isupper():
        return value.upper()
    return value[0].upper() + value[1:] if match.group(0)[0].isupper() else value


def clean(cache, path):
    data = json.loads(path.read_text(encoding="utf-8"))
    info = json.loads(cache.with_name("source.json").read_text(encoding="utf-8"))
    if data["source_sha256"] != info["sha256"] or data["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Nguồn hoặc phiên bản chuẩn hóa không khớp")
    pages = {item["pdf_page"]: normalize_ocr_text(item["text"]) for item in
             map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
    records = {item["pdf_page"]: item for item in data["pages"]}
    added = []
    # Bỏ phần đầu sách, trang phân phần và câu hỏi tự lượng giá.
    included = set(range(10, 293)) - {123, 207, 258}
    for number in sorted(included):
        text = pages[number]
        record = records.get(number, {"pdf_page": number,
                                     "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                     "layout_flags": [], "changes": []})
        if record["normalized_text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError(f"Trang {number}: chữ gốc không khớp bảng sửa")
        for key, pattern, value, references in RULES:
            for match in re.finditer(pattern, text, re.I):
                if any(match.start() < item["end"] and match.end() > item["start"] for item in record["changes"]):
                    continue
                change = {"start": match.start(), "end": match.end(), "before": match.group(0),
                          "after": replacement(key, match, value), "method": "repeated_ocr_pattern",
                          "rule": key, "reference_pdf_pages": references,
                          "reason": "Mẫu lỗi lặp lại đã đối chiếu trên các trang tham chiếu"}
                record["changes"].append(change)
                added.append({"pdf_page": number, **change})
        if record["changes"]:
            record["changes"].sort(key=lambda item: item["start"])
            records[number] = record
    for item in records.values():
        for change in item["changes"]:
            change.setdefault("method", "visual_comparison_with_pdf_scan")
    data["pages"] = [records[number] for number in sorted(records)]
    data["review_scope"] = "selected_visual_edits_and_repeated_ocr_patterns"
    data["notes"] = [
        "Phân biệt các chỗ đối chiếu trực tiếp và các mẫu OCR lặp lại; không khẳng định đã rà từng chữ của toàn sách.",
        "Không dùng quy tắc chung để sửa số liệu, công thức hoặc nội dung chuyên môn.",
        "Đã sắp lại bảng ARCON/NON-ARCON ở trang 216–217 theo từng cột.",
        "Phần thuật ngữ nhiều cột được giữ riêng, chưa đưa vào kho embedding.",
    ]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return {"added_corrections": len(added), "pages_affected": len({x["pdf_page"] for x in added}),
            "total_patches": sum(len(x["changes"]) for x in records.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/internal_rhm_check/source_pages.jsonl")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    print(json.dumps(clean(args.page_cache, args.review), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
