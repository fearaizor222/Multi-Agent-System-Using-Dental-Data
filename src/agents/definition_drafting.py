"""Trích định nghĩa tường minh, không suy ra định nghĩa từ từ khóa cùng chủ đề."""
import re

from src.agents.evidence_review import body_start


def definition_subject(question):
    question = re.sub(r"\s+", " ", question).strip().rstrip("?!. ")
    match = re.fullmatch(r"(.{2,120}?)\s+là\s+gì", question, re.I)
    if not match:
        match = re.fullmatch(r"(?:thế nào là|định nghĩa)\s+(.{2,120})", question, re.I)
    return match[1].strip() if match else None


def extract_definition(subject, evidence):
    for item in evidence:
        body = item.content[body_start(item):]
        pattern = r"(?:^|(?<=[.!?])\s+)\s*(" + re.escape(subject) + r"\s+là\s+[^.!?]{2,790}[.!?]?)"
        for match in re.finditer(pattern, body, re.I | re.M):
            # Không bỏ điều kiện/phủ định ở dòng ngay trước một câu bị xuống dòng.
            before = re.split(r"[.!?]", body[:match.start(1)])[-1]
            if re.search(r"\b(nếu|không|chưa|giả sử|trừ khi)\b", before, re.I):
                continue
            original = re.sub(r"\s+", " ", match[1]).strip()
            text = original
            # Tách phần mục đích nối thêm; giữ định nghĩa là một nhận định ngắn.
            if not re.search(r"\b(không|chỉ khi|trừ khi|nếu)\b", original, re.I):
                # Tên gọi khác, vị trí đặt và mục đích là các ý nối thêm.
                # Giữ câu định nghĩa chính để A3 không phải xác nhận nhiều ý cùng lúc.
                boundary = (r"\s+(?:được thực hiện\s+với mục đích|nhằm|(?:được\s+)?đặt\s+(?:giữa|trên|trong))\s+"
                            r"|,\s*(?:thường\s+được\s+gọi\s+là|còn\s+(?:được\s+)?gọi\s+là|hay\s+còn\s+gọi\s+là)\s+")
                parts = re.split(boundary, original, maxsplit=1, flags=re.I)
                if len(parts[0].split()) >= 8:
                    text = parts[0].rstrip(" ,;")
            if not text.endswith((".", "!", "?")):
                text += "."
            return text, item, {"mode": "extractive_definition", "subject": subject,
                                "original_sentence": original, "selected_text": text,
                                "evidence_id": item.evidence_id}
    return None
