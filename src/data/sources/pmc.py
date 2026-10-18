"""Đọc toàn văn JATS/XML của PMC, giữ cấu trúc và vị trí nguồn."""
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from src.data.public_schema import DocumentBuilder, Locator, Processing, make_document, save_raw, table_text
from src.data.sources.markup import clean, parse_table, tag_text

def parse_pmc(path, raw_directory, acquisition_url=None, retrieved_at=None):
    path = Path(path)
    root = ET.fromstring(path.read_bytes())
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    error = root.find(".//error")
    if error is not None:
        raise ValueError("PMC OAI-PMH: " + tag_text(error))
    article = root if root.tag == "article" else root.find(".//article")
    if article is None or article.find("body") is None:
        raise ValueError("Không có article/body; không coi metadata hoặc abstract là toàn văn")
    meta = article.find("front/article-meta")
    if meta is None:
        raise ValueError("XML thiếu article-meta")
    identifiers = {node.get("pub-id-type"): tag_text(node) for node in meta.findall("article-id")}
    pmcid = identifiers.get("pmc") or identifiers.get("pmcid")
    if pmcid and pmcid.isdigit():
        pmcid = "PMC" + pmcid
    if not pmcid or not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError("XML thiếu PMCID hợp lệ")
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    raw = save_raw(path, raw_directory, "application/xml", acquisition_url or url, retrieved_at)
    builder = DocumentBuilder(pmcid)
    quality = [] if retrieved_at else ["retrieval_time_unknown"]
    language = article.get("{http://www.w3.org/XML/1998/namespace}lang") or "unknown"
    if language == "unknown":
        quality.append("language_unknown")

    node_paths={}

    def remember_paths(node,path):
        node_paths[id(node)]=path
        counts=Counter()
        for child in node:
            counts[child.tag]+=1
            remember_paths(child,f'{path}/{child.tag}[{counts[child.tag]}]')

    remember_paths(article,'/article')

    def locator(node, location):
        anchor = node.get("id")
        return Locator(url=url + ("#" + anchor if anchor else ""), source_id=anchor, path=node_paths.get(id(node),location))

    def visit(container, section, location):
        counts = Counter()
        for node in container:
            counts[node.tag] += 1
            node_path = f"{location}/{node.tag}[{counts[node.tag]}]"
            if node.tag in {"title", "label"}:
                continue
            if node.tag == "sec":
                nested = builder.section(tag_text(node.find("title")), locator(node, node_path), section)
                visit(node, nested, node_path)
            elif node.tag == "table-wrap":
                table_node = node.find(".//table")
                if table_node is None:
                    quality.append("table_without_parseable_cells:" + (node.get("id") or node_path))
                    text = tag_text(node)
                    builder.block(section, "note", text, locator(node, node_path),
                                  quality_flags=["table_structure_requires_review"])
                else:
                    table = parse_table(ET.tostring(table_node, encoding="unicode"),
                                        label=tag_text(node.find("label")) or None,
                                        caption=tag_text(node.find("caption")) or None,
                                        footnotes=[tag_text(fn) for fn in node.findall("table-wrap-foot/fn")])
                    builder.block(section, "table", table_text(table), locator(node, node_path), table=table)
            elif node.tag == "list":
                items = [tag_text(item) for item in node.findall("list-item") if tag_text(item)]
                if items:
                    builder.block(section, "list", "\n".join(items), locator(node, node_path), list_items=items)
            elif node.tag == "fig":
                text = " ".join(filter(None, [tag_text(node.find("label")), tag_text(node.find("caption"))]))
                builder.block(section, "figure_caption", text, locator(node, node_path),
                              quality_flags=["figure_image_not_interpreted"])
            elif node.tag == "p":
                # Nội dung display có thể được đặt bên trong p trong JATS.
                displays = {"table-wrap", "fig", "list", "boxed-text"}
                if any(child.tag in displays for child in node):
                    if node.text and clean(node.text):
                        builder.block(section, "paragraph", clean(node.text), locator(node, node_path))
                    for child_index, child in enumerate(node, 1):
                        if child.tag in displays:
                            wrapper = ET.Element("wrapper")
                            wrapper.append(child)
                            visit(wrapper, section, node_path)
                        else:
                            builder.block(section, "paragraph", tag_text(child), locator(node, node_path))
                        if child.tail and clean(child.tail):
                            builder.block(section, "paragraph", clean(child.tail), locator(node, node_path))
                else:
                    builder.block(section, "paragraph", tag_text(node), locator(node, node_path))
            elif node.tag in {"ref-list", "ack", "permissions"}:
                continue
            elif len(node):
                visit(node, section, node_path)
            elif tag_text(node):
                builder.block(section, "note", tag_text(node), locator(node, node_path))

    for index, abstract in enumerate(meta.findall("abstract"), 1):
        section = builder.section("Abstract", locator(abstract, f"/article/front/article-meta/abstract[{index}]"))
        visit(abstract, section, f"/article/front/article-meta/abstract[{index}]")
    body = article.find("body")
    body_section = builder.section("Body", locator(body, "/article/body"))
    visit(body, body_section, "/article/body")
    if not any(section.blocks for section in builder.sections if section is not body_section and "Body" in section.heading_path) and not body_section.blocks:
        raise ValueError("XML có body nhưng không lấy được nội dung toàn văn")
    dates = meta.findall("pub-date")
    publication = next((date for date in dates if date.get("pub-type") == "epub" or date.get("publication-format") == "electronic"), dates[0] if dates else None)
    parts = [tag_text(publication.find(name)) for name in ("year", "month", "day")] if publication is not None else []
    publication_date = "-".join(part.zfill(2) if index else part for index, part in enumerate(parts) if part) or None
    licenses = meta.findall("permissions/license")
    related = [{"type": node.get("related-article-type"), "text": tag_text(node), "attributes": dict(node.attrib)}
               for node in meta.findall("related-article")]
    return make_document(document_id=pmcid, title=tag_text(meta.find("title-group/article-title")), source="pmc",
        language=language, source_url=url, publication_date=publication_date,
        source_metadata={"pmcid": pmcid, "pmid": identifiers.get("pmid"), "doi": identifiers.get("doi"),
                         "article_type": article.get("article-type"),
                         "journal": tag_text(article.find("front/journal-meta/journal-title-group/journal-title")),
                         "licenses": [{"text": tag_text(node), "attributes": dict(node.attrib)} for node in licenses],
                         "related_articles": related, "references_indexed": False},
        raw=raw, processing=Processing(parser="pmc_jats", method="xml_jats", quality_flags=quality), sections=builder.finish_sections())
