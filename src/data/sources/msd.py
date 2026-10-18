"""Đọc bản HTML MSD đã lưu; giữ đề mục, danh sách, bảng và chú thích hình."""
import re
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup, Tag

from src.data.sources.msd_http import is_article_url, normalize_url
from src.data.public_schema import DocumentBuilder, Locator, Processing, make_document, save_raw, table_text
from src.data.sources.markup import clean, dated_month, parse_table, tag_text


def parse_msd(path, raw_directory, retrieved_at=None):
    path = Path(path)
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    title = soup.select_one("#topicHeaderTitle")
    body = soup.select_one('[data-testid="topic-main-content"]')
    canonical = soup.select_one('link[rel="canonical"]')
    url = normalize_url(canonical.get("href", "")) if canonical else None
    if not title or not body or not url or not is_article_url(url):
        raise ValueError("HTML không phải bài nha khoa MSD tiếng Việt có nội dung và URL nguồn")
    if not soup.html or soup.html.get("lang", "").split("-")[0] != "vi":
        raise ValueError("HTML MSD không có ngôn ngữ tiếng Việt")
    header = soup.select_one('[class*="TopicHead_topic__authors__container"]')
    header_text = clean(header.get_text(" ", strip=True)) if header else ""
    native_id = soup.select_one('meta[name="VasontID"]')
    article_match = re.search(r"\bv\d+_vi\b", native_id.get("content", "") if native_id else header_text)
    source_id = article_match[0] if article_match else None
    if not source_id:
        from src.data.public_schema import digest
        source_id = digest(url.encode("utf-8"))[:20]
    document_id = "MSD-" + source_id
    raw = save_raw(path, raw_directory, "text/html", url, retrieved_at)
    builder = DocumentBuilder(document_id)
    current = None
    heading_stack = []
    definition = soup.select_one('[data-testid="topicDefinition"]')

    def locator(tag):
        anchor = tag.get("id") or next((parent.get("id") for parent in tag.parents
                                        if isinstance(parent, Tag) and parent.get("id")), None)
        lineage = [node for node in reversed(list(tag.parents)) if isinstance(node, Tag) and node.name != "[document]"] + [tag]
        path = "/" + "/".join(f"{node.name}[{len(node.find_previous_siblings(node.name))+1}]" for node in lineage)
        return Locator(url=url + ("#" + quote(anchor, safe="") if anchor else ""), source_id=anchor, path=path)

    def ensure_section():
        nonlocal current
        if current is None:
            current = builder.section("", Locator(url=url))
        return current

    # Định nghĩa có thể nằm ngoài topic-main-content.
    if definition and body not in definition.parents:
        builder.block(ensure_section(), "paragraph", tag_text(definition), locator(definition))

    def visit(container):
        nonlocal current
        for child in container.children:
            if not isinstance(child, Tag):
                continue
            classes = " ".join(child.get("class", []))
            if (child.name in {"script", "style", "button", "nav"} or child.get("aria-hidden") == "true"
                    or "TopicSubNavigation_" in classes):
                continue
            if child.name in {"h2", "h3", "h4", "h5", "h6"}:
                level, text = int(child.name[1]), tag_text(child)
                if not text:
                    continue
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent = heading_stack[-1][1] if heading_stack else None
                current = builder.section(text, locator(child), parent)
                heading_stack.append((level, current))
            elif child.name == "table":
                table = parse_table(child)
                builder.block(ensure_section(), "table", table_text(table), locator(child), table=table)
            elif child.name in {"ul", "ol"}:
                items = [tag_text(item) for item in child.find_all("li", recursive=False)]
                items = [item for item in items if item]
                if items:
                    builder.block(ensure_section(), "list", "\n".join(items), locator(child), list_items=items)
            elif child.name in {"figure", "figcaption"} or "TopicImage_" in classes:
                caption = child.find("figcaption") or child.select_one('[class*="caption"], [class*="Caption"]')
                text = tag_text(caption) if caption else tag_text(child) if child.name == "figcaption" else ""
                if text:
                    builder.block(ensure_section(), "figure_caption", text, locator(child),
                                  quality_flags=["figure_image_not_interpreted"])
            elif child.name in {"p", "blockquote"}:
                # Một p có thể bọc bảng/list trong HTML lỗi; giữ cấu trúc trước.
                if child.find(["table", "ul", "ol"]):
                    visit(child)
                else:
                    builder.block(ensure_section(), "paragraph", tag_text(child), locator(child))
            else:
                visit(child)

    visit(body)
    reviewed = re.search(r"(?:Đánh giá toàn diện|Đã xem xét|Đã đánh giá)\s*:\s*(Thg\s*\d{1,2}\s+\d{4})", header_text)
    updated = re.search(r"(?:Cập nhật lần cuối|Sửa đổi nội dung lần cuối cùng)\s*:\s*(Thg\s*\d{1,2}\s+\d{4})", header_text)
    return make_document(document_id=document_id, title=tag_text(title), source="msd", language="vi", source_url=url,
                         updated_date=dated_month(updated[1]) if updated else None,
                         source_metadata={"article_id": article_match[0] if article_match else None,
                                          "edition": "professional", "reviewed_date": dated_month(reviewed[1]) if reviewed else None,
                                          "reviewed_date_raw": reviewed[1] if reviewed else None,
                                          "updated_date_raw": updated[1] if updated else None,
                                          "license": None},
                         raw=raw, processing=Processing(parser="msd_html", method="html_dom",
                             quality_flags=[] if retrieved_at else ["retrieval_time_unknown"]), sections=builder.finish_sections())
