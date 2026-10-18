"""Các phần HTML/XML dùng chung; không suy đoán nội dung y khoa."""
import re
import unicodedata

from src.data.public_schema import TableCell, TableData


def clean(text):
    return " ".join(unicodedata.normalize("NFC", text).replace("\u00ad", "").split())


def tag_text(tag):
    # Giữ từ có định dạng inline, đồng thời ngăn hai đoạn/list dính vào nhau.
    if tag is None:
        return ""
    from bs4 import Comment, NavigableString
    block_tags = {"p", "div", "li", "br", "tr", "td", "th", "sec", "title", "caption", "label"}

    def visit(node):
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return str(node)
        if hasattr(node, "children"):
            text = "".join(visit(child) for child in node.children)
            return text + ("\n" if node.name in block_tags else "")
        text = node.text or ""
        for child in node:
            text += visit(child) + (child.tail or "")
        return text + ("\n" if node.tag in block_tags else "")

    return clean(visit(tag))


def parse_table(table, label=None, caption=None, footnotes=None):
    from bs4 import BeautifulSoup
    table = BeautifulSoup(str(table), "html.parser").find("table")
    if table is None:
        raise ValueError("Bảng không có phần table dạng HTML/JATS")
    cells, occupied = [], set()
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (row_index, column) in occupied:
                column += 1
            rowspan = max(1, int(cell.get("rowspan") or 1))
            colspan = max(1, int(cell.get("colspan") or 1))
            cells.append(TableCell(row=row_index, column=column, text=tag_text(cell), rowspan=rowspan,
                                   colspan=colspan, is_header=cell.name == "th" or cell.find_parent("thead") is not None))
            occupied.update((r, c) for r in range(row_index, row_index + rowspan)
                            for c in range(column, column + colspan))
            column += colspan
    return TableData(label=label, caption=caption or tag_text(table.find("caption", recursive=False)) or None,
                     footnotes=footnotes or [], cells=cells, original_markup=str(table))


def dated_month(text):
    match = re.search(r"Thg\s*(\d{1,2})\s+(\d{4})", text or "")
    return f"{match[2]}-{int(match[1]):02}" if match else None
