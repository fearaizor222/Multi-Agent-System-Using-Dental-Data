"""Hiển thị cấu trúc tài liệu đã xử lý để rà soát."""
import html


def render_document(doc):
    escape = html.escape
    parts = [f'<article><h1>{escape(doc.title)}</h1><p>{escape(doc.source.upper())} · {escape(doc.document_id)}</p>',
             f'<p><a href="{escape(doc.source_url, quote=True)}">Nguồn gốc</a></p>']
    events = [(section.order, "heading", section) for section in doc.sections]
    events += [(block.order, "block", block) for section in doc.sections for block in section.blocks]
    for _, kind, item in sorted(events, key=lambda event: event[0]):
        if kind == "heading":
            if item.title:
                level = min(6, len(item.heading_path)+1)
                parts.append(f'<h{level}>{escape(item.title)}</h{level}>')
        else:
            label = " · ".join(f"PDF {loc.pdf_page} / trang in {loc.printed_page or '?'}" if loc.pdf_page else loc.source_id or loc.path or ""
                               for loc in item.locators)
            parts.append(f'<div class="block"><small>{escape(item.block_id)} · {escape(label)}</small>')
            if item.table:
                table = item.table
                if table.label or table.caption:
                    parts.append(f'<p><b>{escape(" ".join(filter(None,[table.label,table.caption])))}</b></p>')
                parts.append("<table>")
                for row in sorted({cell.row for cell in table.cells}):
                    parts.append("<tr>")
                    for cell in sorted((cell for cell in table.cells if cell.row == row), key=lambda cell: cell.column):
                        tag = "th" if cell.is_header else "td"
                        parts.append(f'<{tag} rowspan="{cell.rowspan}" colspan="{cell.colspan}">{escape(cell.text)}</{tag}>')
                    parts.append("</tr>")
                parts.append("</table>")
                parts += [f'<p class="note">{escape(note)}</p>' for note in table.footnotes]
            elif item.kind == "list":
                parts.append("<ul>"+"".join(f"<li>{escape(text)}</li>" for text in item.list_items)+"</ul>")
            else:
                parts.append(f'<p>{escape(item.text)}</p>')
            if item.quality_flags:
                parts.append(f'<small>Cần rà soát: {escape(", ".join(item.quality_flags))}</small>')
            parts.append("</div>")
    return "\n".join(parts)+"</article>"
