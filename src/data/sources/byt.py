"""Đọc PDF Bộ Y tế có lớp chữ; giữ vị trí và cấu trúc bảng theo tọa độ."""
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from src.data.public_schema import DocumentBuilder, Locator, Processing, TableCell, TableData, make_document, save_raw, table_text
from src.data.sources.markup import clean

BYT_URL = "https://kcb.vn/phac-do/huong-dan-chan-doan-va-dieu-tri-mot-so-benh-ve-rang-ham-mat.html"


def fix_glyph(text):
    return clean(text.replace("Ƣ", "Ư").replace("ƣ", "ư"))


def heading_level(text, bold):
    if re.match(r"^\d+\s*\.\s*", text) and text.isupper() and bold:
        return 0
    if re.match(r"^[IVX]+\s*\.\s+", text) and bold:
        return 1
    # Các mục 1., 1.1., a. trong tài liệu này thường không in đậm.
    match = re.match(r"^(\d+(?:\s*\.\s*\d+)*)(?:\s*\.)?\s+[A-ZĐÀ-Ỹ]", text)
    if match and len(text) < 160:
        return 2 + match[1].count(".")
    if re.match(r"^[a-z]\.\s+[A-ZĐÀ-Ỹ]", text) and len(text) < 160:
        return 3
    return None


def page_entities(page):
    tables = page.find_tables()
    lines = []
    words = page.extract_words(x_tolerance=2, y_tolerance=3, extra_attrs=["fontname"])
    for word in sorted(words, key=lambda value: (value["top"], value["x0"])):
        x, y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
        if any(t.bbox[0] <= x <= t.bbox[2] and t.bbox[1] <= y <= t.bbox[3] for t in tables):
            continue
        if lines and abs(lines[-1]["top"] - word["top"]) < 3:
            lines[-1]["words"].append(word)
        else:
            lines.append({"top": word["top"], "words": [word]})
    entities = []
    for line in lines:
        words = sorted(line["words"], key=lambda word: word["x0"])
        text = fix_glyph(" ".join(word["text"] for word in words))
        if text.isdigit() and line["top"] > page.height * 0.85:
            continue
        entities.append({"type": "line", "top": line["top"], "text": text,
                         "bold": sum("bold" in word["fontname"].lower() for word in words) > len(words) / 2,
                         "bbox": [min(w["x0"] for w in words), min(w["top"] for w in words),
                                  max(w["x1"] for w in words), max(w["bottom"] for w in words)]})
    for table in tables:
        xs = sorted({round(x, 2) for cell in table.cells for x in (cell[0], cell[2])})
        ys = sorted({round(y, 2) for cell in table.cells for y in (cell[1], cell[3])})
        extracted, values = table.extract(), {}
        for row, texts in zip(table.rows, extracted):
            for cell, text in zip(row.cells, texts):
                if cell is not None:
                    values[tuple(cell)] = fix_glyph(text or "")
        cells = []
        for bbox in table.cells:
            x0, top, x1, bottom = [round(value, 2) for value in bbox]
            cells.append(TableCell(row=ys.index(top), column=xs.index(x0), text=values.get(tuple(bbox), ""),
                                   rowspan=ys.index(bottom)-ys.index(top), colspan=xs.index(x1)-xs.index(x0),
                                   is_header=ys.index(top) == 0))
        entities.append({"type": "table", "top": table.bbox[1], "bottom": table.bbox[3],
                         "bbox": list(table.bbox), "table": TableData(cells=cells, header_inferred=True)})
    return sorted(entities, key=lambda entity: entity["top"])


def parse_byt(path, raw_directory, page_numbers=None):
    path = Path(path)
    reader = PdfReader(path)
    identity = " ".join(page.extract_text() or "" for page in reader.pages[:5])
    if not all(value in identity for value in ("3108", "BỘ Y TẾ", "RĂNG HÀM MẶT")):
        raise ValueError("PDF không khớp tài liệu Bộ Y tế 3108/QĐ-BYT năm 2015")
    numbers = sorted(set(page_numbers or range(8, 202)))
    if not numbers or any(number < 8 or number > min(201, len(reader.pages)) for number in numbers):
        raise ValueError("Chỉ chọn trang PDF chuyên môn 8–201 trong tài liệu này")
    raw = save_raw(path, raw_directory, "application/pdf", BYT_URL)
    builder = DocumentBuilder("BYT-3108-2015")
    stack, previous_table = [], None
    flags = ["glyph_u_replaced", "paragraph_boundaries_inferred", "numbered_headings_inferred", "pdf_table_headers_require_review"]

    def heading(title, level, locator):
        while stack and stack[-1][0] >= level:
            stack.pop()
        section = builder.section(title, locator, stack[-1][1] if stack else None)
        stack.append((level, section))
        return section

    # Khôi phục đề mục cha khi mẫu bắt đầu giữa chương. Đọc text, không OCR.
    context = []
    if numbers[0] > 8:
        for number in range(8, numbers[0]):
            lines = [fix_glyph(line) for line in (reader.pages[number-1].extract_text() or "").splitlines() if line.strip()]
            for line in lines:
                level = 0 if re.match(r"^\d+\.\s*", line) and line.isupper() else 1 if re.match(r"^[IVX]+\s*\.\s+", line) else None
                if level is not None:
                    while context and context[-1][0] >= level:
                        context.pop()
                    context.append((level, line, number))
        for level, title, number in context:
            heading(title, level, Locator(url=BYT_URL, pdf_page=number))

    with pdfplumber.open(path) as pdf:
        for number in numbers:
            page = pdf.pages[number-1]
            text = (reader.pages[number-1].extract_text() or "").splitlines()
            printed = text[0].strip() if text and text[0].strip().isdigit() else None
            pending, boxes, pending_kind = [], [], "paragraph"
            open_chapter_heading = None

            def flush():
                nonlocal pending, boxes
                if not pending:
                    return
                if not stack:
                    heading("", 0, Locator(url=BYT_URL, pdf_page=number, printed_page=printed))
                bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
                extra = {"list_items": [" ".join(pending)]} if pending_kind == "list" else {}
                builder.block(stack[-1][1], pending_kind, " ".join(pending),
                              Locator(url=BYT_URL, pdf_page=number, printed_page=printed, bbox=bbox), **extra)
                pending, boxes = [], []

            for entity in page_entities(page):
                locator = Locator(url=BYT_URL, pdf_page=number, printed_page=printed, bbox=entity["bbox"])
                if entity["type"] == "table":
                    open_chapter_heading = None
                    flush()
                    if not stack:
                        heading("", 0, locator)
                    table, locators = entity["table"], [locator]
                    if (previous_table and previous_table["page"] == number-1
                            and previous_table["chapter"] == stack[0][1].section_id
                            and previous_table["bottom"] > previous_table["height"] * 0.85
                            and entity["top"] < page.height * 0.15
                            and max(c.column+c.colspan for c in table.cells) == previous_table["width"]):
                        header = [cell.model_copy(deep=True) for cell in previous_table["table"].cells if cell.is_header]
                        height = max((cell.row+cell.rowspan for cell in header), default=0)
                        # Trang tiếp theo bắt đầu bằng dữ liệu, chưa lặp lại tiêu đề cột.
                        if header and [c.text for c in table.cells if c.row == 0] != [c.text for c in header]:
                            shifted = [cell.model_copy(update={"row": cell.row+height, "is_header": False}) for cell in table.cells]
                            table = TableData(cells=header+shifted, header_inferred=True)
                            locators.extend(previous_table["header_locators"])
                    block = builder.block(stack[-1][1], "table", table_text(table), locators, table=table,
                                          quality_flags=["pdf_table_header_inferred"])
                    previous_table = {"page": number, "height": page.height, "bottom": entity["bottom"],
                                      "chapter": stack[0][1].section_id,
                                      "table": table, "width": max(c.column+c.colspan for c in table.cells),
                                      "header_locators": locators[1:] or [locator], "block_id": block.block_id}
                    continue
                line = entity["text"]
                level = heading_level(line, entity["bold"])
                if (open_chapter_heading is not None and level is None and entity["bold"]
                        and line.isupper() and not re.match(r"^[-+•]",line)):
                    open_chapter_heading.title += " " + line
                    open_chapter_heading.heading_path = [open_chapter_heading.title]
                    continue
                if level is not None and re.match(r"^[a-z]\.\s+", line):
                    numeric_parent = next((rank for rank, section in reversed(stack)
                                           if rank >= 2 and re.match(r"^\d", section.title)), 2)
                    level = numeric_parent + 1
                if level is not None:
                    flush()
                    created = heading(line, level, locator)
                    open_chapter_heading = created if level == 0 else None
                else:
                    open_chapter_heading = None
                    kind = "list" if re.match(r"^[-+•]\s+", line) else "paragraph"
                    if kind == "list" or (pending and pending_kind != kind and pending_kind != "list"):
                        flush()
                    if not pending:
                        pending_kind = kind
                    pending.append(line)
                    boxes.append(entity["bbox"])
            flush()
            page.close()
    return make_document(document_id=builder.document_id,
        title="Hướng dẫn chẩn đoán và điều trị một số bệnh về Răng Hàm Mặt", source="byt", language="vi",
        source_url=BYT_URL, publication_date="2015-07-28", content_scope="page_subset",
        source_metadata={"decision_number": "3108/QĐ-BYT", "original_filename": path.name,
                         "pdf_page_count": len(reader.pages), "selected_pdf_pages": numbers,
                         "indexed_scope": "clinical_pages", "license": None}, raw=raw,
        processing=Processing(parser="byt_pdf", method="native_pdf_layout", quality_flags=flags), sections=builder.finish_sections())
