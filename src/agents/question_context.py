"""Làm rõ câu nối ngắn từ câu hỏi của người dùng, giữ nguyên đầu vào gốc."""
from copy import deepcopy
import re
import unicodedata

from src.agents.definition_drafting import definition_subject


def plain(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip().rstrip("?!. ")


ASPECTS = {
    "treatment": ("Điều trị {topic} như thế nào?", {"chua the nao", "chua nhu the nao", "dieu tri the nao", "dieu tri ra sao", "chua ra sao"}),
    "cause": ("Nguyên nhân {topic} là gì?", {"nguyen nhan la gi", "tai sao lai bi", "vi sao bi"}),
    "symptoms": ("Triệu chứng {topic} là gì?", {"trieu chung la gi", "bieu hien the nao", "co bieu hien gi"}),
    "prevention": ("Phòng ngừa {topic} như thế nào?", {"phong ngua the nao", "phong tranh the nao", "ngan ngua ra sao"}),
}


def short_aspect(question):
    value = plain(question)
    return next((key for key, (_, forms) in ASPECTS.items() if value in forms), None)


def parse_request(question):
    text = re.sub(r"\s+", " ", question).strip().rstrip("?!. ")
    for aspect, prefix in [("treatment", r"(?:cách\s+)?(?:điều trị|chữa)"),
                           ("cause", r"nguyên nhân(?:\s+gây)?"),
                           ("symptoms", r"(?:triệu chứng|biểu hiện)(?:\s+của)?"),
                           ("prevention", r"(?:cách\s+)?(?:phòng ngừa|phòng tránh|ngăn ngừa)")]:
        match = re.fullmatch(prefix + r"\s+(.{2,120}?)(?:\s+(?:như thế nào|thế nào|ra sao|là gì))?", text, re.I)
        if match:
            return match[1].strip(), aspect
    topic = definition_subject(question)
    if topic:
        return topic, "definition"
    return None, None


def compact_history(history):
    result = deepcopy(history)
    for turn in result:
        if turn["role"] != "assistant":
            continue
        lines = turn["content"].splitlines()
        for i, line in enumerate(lines):
            if line.strip() not in ("Nguồn công khai:", "Nguồn tham khảo:"):
                continue
            tail = [s for s in lines[i + 1:] if s.strip()]
            if tail and all(re.match(r"^\[\d+\]\s+", s) for s in tail):
                # Chỉ bỏ danh mục nguồn do ứng dụng sinh; không rút gọn nội dung.
                content = "\n".join(lines[:i]).strip()
                turn["content"] = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", content)
                break
    return result


def prepare_routing_context(context):
    projected = deepcopy(context)
    projected["history"] = compact_history(context["history"])
    detail = {"original_question": context["message"], "resolved_question": context["message"],
              "strategy": "current_user", "bibliography_removed": projected["history"] != context["history"]}
    aspect = short_aspect(context["message"])
    if aspect and not context["attachments"]:
        for turn in reversed(context["history"]):
            if turn["role"] != "user":
                continue
            topic, _ = parse_request(turn["content"])
            if topic:
                projected["message"] = ASPECTS[aspect][0].format(topic=topic)
                detail.update(resolved_question=projected["message"], strategy="previous_user_topic",
                              source_user_question=turn["content"])
                break
            if not short_aspect(turn["content"]):
                break
    return projected, detail


def verification_question(question, claim):
    topic, aspect = parse_request(question)
    if not topic or aspect != "definition" or claim.casefold().startswith(topic.casefold() + " là "):
        return question
    for key, expression in [("cause", r"vi khuẩn.*(?:axit|acid)|(?:hủy|mất) khoáng|nguyên nhân"),
                            ("symptoms", r"triệu chứng|biểu hiện"),
                            ("treatment", r"điều trị|phục hồi|trám|hàn răng"),
                            ("prevention", r"phòng ngừa|ngăn ngừa")]:
        if re.search(expression, claim, re.I):
            return ASPECTS[key][0].format(topic=topic)
    return question
