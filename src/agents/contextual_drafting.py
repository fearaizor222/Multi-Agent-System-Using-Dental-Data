"""Chọn thêm câu giải thích từ đúng chủ đề, không viết thêm kiến thức ngoài nguồn."""
import re

from src.agents.evidence_review import body_start
from src.rag.source_scope import scoped_to_topic


PATTERNS = {
    "cause": [r"vi khuẩn.*(?:axit|acid)|(?:hủy|mất) khoáng.*vi khuẩn", r"nguyên nhân|kết quả của"],
    "symptoms": [r"triệu chứng|biểu hiện|ê buốt"],
    "treatment": [r"điều trị bao gồm|nguyên tắc điều trị", r"phục hồi|trám|hàn răng|điều trị tủy|lấy tủy|nhổ răng|tái khoáng|fluor"],
    "prevention": [r"ngăn ngừa|phòng ngừa|phòng tránh"],
}


def atomic_clauses(sentence, aspect):
    if aspect == "cause" and not re.search(r"\b(không|chỉ khi|trừ khi|nếu)\b", sentence, re.I):
        # Tách chủ thể -> tác nhân -> tác động trong một câu ghép của nguồn.
        # Giữ tính khả năng, không biến "có thể" thành khẳng định tuyệt đối.
        match = re.fullmatch(r"(.+?)\s+có khả năng\s+[^,]+,\s*(tạo ra\s+(axit|acid))\s+(có thể\s+.+?)\.", sentence, re.I)
        if match:
            return [match[1] + " có thể " + match[2] + ".", match[3].capitalize() + " " + match[4] + "."]
    return [sentence]


def extract_aspect(topic, aspect, evidence, limit=2, source_id=None):
    pool = []
    for item in evidence:
        if not scoped_to_topic(item, topic) or (source_id and item.source_id != source_id):
            continue
        body = re.sub(r"\s+", " ", item.content[body_start(item):]).strip()
        for match in re.finditer(r"[^.!?]+[.!?]|[^.!?]+$", body):
            sentence = match[0].strip()
            if not 7 <= len(sentence.split()) <= 55 or len(sentence) > 480:
                continue
            # Liều/dạng dùng thuốc không thuộc phần giải thích thông tin chung này.
            if re.search(r"\b(?:mg|ml|ppm|tiêm|uống)\b|\d+\s*%", sentence, re.I):
                continue
            rank = next((i for i, pattern in enumerate(PATTERNS[aspect]) if re.search(pattern, sentence, re.I)), None)
            if rank is not None:
                pool.extend((rank, clause, item, sentence) for clause in atomic_clauses(sentence, aspect))
    selected, seen = [], set()
    for _, sentence, item, original in sorted(pool, key=lambda entry: entry[0]):
        if sentence.casefold() in seen:
            continue
        seen.add(sentence.casefold())
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        selected.append((sentence, item, {"mode": "extractive_context", "subject": topic,
                                         "aspect": aspect, "original_sentence": original,
                                         "selected_text": sentence, "evidence_id": item.evidence_id}))
        if len(selected) == limit:
            break
    return selected
