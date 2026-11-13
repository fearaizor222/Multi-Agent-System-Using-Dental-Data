"""Đối chiếu chủ đề tường minh trong tiêu đề/đề mục, không mở rộng phân loại."""
import re


def topic_key(text):
    text = re.sub(r"^\s*(?:chương\s+)?\d+[\s.:]+", "", text, flags=re.I)
    text = re.sub(r"\s*-\s*đoạn\s+\d+\s*$", "", text, flags=re.I)
    text = re.sub(r"^bệnh\s+", "", text, flags=re.I)
    return " ".join(text.casefold().split())


def scoped_to_topic(item, topic):
    labels = [item.title, *(item.metadata.get("heading_path") or [])[:1]]
    return any(topic_key(label) == topic_key(topic) for label in labels)
