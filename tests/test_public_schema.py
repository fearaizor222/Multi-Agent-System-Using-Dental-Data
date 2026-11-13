import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.public_chunks import make_chunks, validate_chunk_sources
from src.data.public_schema import PublicDocument, digest, table_text, make_document, save_raw, write_documents
from src.data.sources.msd import parse_msd
from src.data.sources.pmc import parse_pmc
from src.data.sources.byt import heading_level

FIXTURE = Path(__file__).parent/"fixtures/pmc_structure.xml"


class WordTokenizer:
    def encode(self, text, **kwargs):
        return [0] + text.split() + [1]


def test_pmc_preserves_ids_nested_order_tables_and_figures(tmp_path):
    doc = parse_pmc(FIXTURE, tmp_path/"raw")
    assert doc.document_id == "PMC999999999"
    assert doc.language == "en" and doc.publication_date == "2024-02-03"
    assert doc.content_type == "full_text"
    assert doc.source_metadata["doi"] == "10.0000/parser-test"
    blocks = [block for section in doc.sections for block in section.blocks]
    assert any(block.text == "alphabeta and source 1." for block in blocks)
    assert not any("Reference excluded" in block.text for block in blocks)
    nested = next(section for section in doc.sections if section.title == "Nested section")
    first = next(section for section in doc.sections if section.title == "First section")
    assert nested.parent_section_id == first.section_id
    before = next(block for block in first.blocks if "alphabeta" in block.text)
    after = next(block for block in first.blocks if "Paragraph after" in block.text)
    assert before.order < nested.order < nested.blocks[0].order < after.order
    assert after.locators[0].path == "/article/body[1]/sec[1]/p[2]"
    table = next(block for block in blocks if block.kind == "table")
    assert table.table.cells[0].rowspan == 2
    assert table.table.cells[1].colspan == 2
    assert table.table.cells[-1].column == 2
    assert table.table.caption == "Table caption." and table.table.footnotes == ["Units: mg."]
    assert table.table.original_markup and table.text == table_text(table.table)
    assert any(block.kind == "figure_caption" and "Caption kept." in block.text for block in blocks)
    assert Path(doc.raw.path).read_bytes() == FIXTURE.read_bytes()
    assert doc.raw.sha256 == digest(FIXTURE.read_bytes())


def test_revision_stable_across_run_time_and_cache_path_but_detects_content_changes(tmp_path):
    first = parse_pmc(FIXTURE, tmp_path/"first")
    second = parse_pmc(FIXTURE, tmp_path/"second", retrieved_at="2026-10-15T00:00:00+00:00")
    assert first.revision_id == second.revision_id
    data = json.loads(first.model_dump_json())
    data["sections"][0]["blocks"][0]["text"] += " changed"
    with pytest.raises(ValidationError, match="revision_id"):
        PublicDocument.model_validate(data)


def test_pmc_does_not_accept_metadata_only_as_full_text(tmp_path):
    path = tmp_path/"metadata.xml"
    path.write_text(FIXTURE.read_text(encoding="utf-8").replace("<body>", "<body-missing>").replace("</body>", "</body-missing>"), encoding="utf-8")
    with pytest.raises(ValueError, match="toàn văn"):
        parse_pmc(path, tmp_path/"raw")


def test_chunks_have_exact_source_offsets_and_table_context(tmp_path):
    doc = parse_pmc(FIXTURE, tmp_path/"raw")
    chunks = make_chunks(doc, WordTokenizer(), max_tokens=64)
    assert chunks and validate_chunk_sources(doc, chunks)
    assert all(chunk.token_count <= 64 for chunk in chunks)
    tables = [chunk for chunk in chunks if any("table-wrap" in (loc.path or "") for loc in chunk.locators)]
    assert tables and "Units: mg." in tables[0].content and "Group" in tables[0].content
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in make_chunks(doc, WordTokenizer(), 64)]
    chunks[0].content += " invented"
    with pytest.raises(ValueError, match="khớp nguồn"):
        validate_chunk_sources(doc, chunks)


def test_msd_keeps_definition_headings_table_spans_and_real_dom_path(tmp_path):
    path = tmp_path/"msd.html"
    path.write_text('''<html lang="vi"><head><link rel="canonical" href="https://www.msdmanuals.com/vi/professional/rối-loạn-nha-khoa/nhom/bai"/></head><body>
    <h1 id="topicHeaderTitle">Bài thử</h1><div class="TopicHead_topic__authors__container">v123_vi Đã xem xét: Thg 2 2024</div>
    <div data-testid="topicDefinition"><p>Định nghĩa tiếng Việt.</p></div>
    <div data-testid="topic-main-content"><h2 id="s1">Mục lớn</h2><p>Giá trị 0,5 mg.</p>
    <h3 id="s11">Mục nhỏ</h3><ul><li>Ý một</li><li>Ý hai</li></ul>
    <table><caption>Chú thích bảng</caption><tr><th colspan="2">Cột gộp</th></tr><tr><td>A</td><td>10</td></tr></table>
    <figure><figcaption>Chú thích hình</figcaption><img src="x.png"/></figure></div></body></html>''', encoding="utf-8")
    doc = parse_msd(path, tmp_path/"raw")
    assert doc.document_id == "MSD-v123_vi" and doc.source_metadata["reviewed_date"] == "2024-02"
    blocks = [block for section in doc.sections for block in section.blocks]
    assert blocks[0].text == "Định nghĩa tiếng Việt."
    assert any(block.text == "Giá trị 0,5 mg." for block in blocks)
    assert next(section for section in doc.sections if section.title == "Mục nhỏ").heading_path == ["Mục lớn", "Mục nhỏ"]
    table = next(block for block in blocks if block.kind == "table")
    assert table.table.cells[0].colspan == 2 and table.table.caption == "Chú thích bảng"
    assert table.locators[0].path == "/html[1]/body[1]/div[3]/table[1]"
    assert any(block.kind == "figure_caption" for block in blocks)


def test_pdf_numbered_headings_without_bold_and_decimal_units():
    assert heading_level("2. Chẩn đoán phân biệt", False) == 2
    assert heading_level("1.1. Chẩn đoán tổn thương sớm", False) == 3
    assert heading_level("a. Tổn thương sâu răng sớm", False) == 3
    assert heading_level("0.5 mg", False) is None
    assert heading_level("0,5 mg", False) is None
    assert heading_level("20 . KHỚP CẮN HỞ", True) == 0


def test_markup_separates_paragraphs_without_breaking_inline_words():
    from bs4 import BeautifulSoup
    from src.data.sources.markup import tag_text
    node = BeautifulSoup("<td><p>Perio<i>dontitis</i>.</p><p>Second paragraph.</p></td>", "html.parser").td
    assert tag_text(node) == "Periodontitis. Second paragraph."


def test_long_chunks_cover_text_and_ids_include_chunking_config(tmp_path):
    doc = parse_pmc(FIXTURE, tmp_path/"raw")
    values = doc.model_dump()
    values.pop("revision_id")
    original = " ".join(f"word{number}" for number in range(250))
    values["sections"][0]["blocks"][0]["text"] = original
    doc = make_document(**values)
    chunks = make_chunks(doc, WordTokenizer(), 64)
    validate_chunk_sources(doc, chunks)
    block_id = doc.sections[0].blocks[0].block_id
    relevant = [chunk for chunk in chunks if any(span.block_id == block_id for span in chunk.spans)]
    assert len(relevant) > 1 and all(chunk.token_count <= 64 for chunk in relevant)
    covered = set(" ".join(chunk.content for chunk in relevant).split())
    assert set(original.split()) <= covered
    assert chunks[0].chunk_id != make_chunks(doc, WordTokenizer(), 80)[0].chunk_id


def test_raw_cache_detects_corruption_and_normalized_revisions_are_retained(tmp_path):
    doc = parse_pmc(FIXTURE, tmp_path/"raw")
    path = tmp_path/"normalized/pmc_documents.jsonl"
    write_documents(path, [doc])
    assert (path.parent/"revisions"/doc.document_id/(doc.revision_id+".json")).exists()
    Path(doc.raw.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache gốc"):
        save_raw(FIXTURE, tmp_path/"raw", "application/xml")


def test_long_table_notes_are_split_without_losing_source_words(tmp_path):
    doc=parse_pmc(FIXTURE,tmp_path/'raw')
    table=next(block for section in doc.sections for block in section.blocks if block.kind=='table')
    table.table.footnotes=[' '.join(f'note{number}' for number in range(180))]
    table.text=table_text(table.table)
    values=doc.model_dump(exclude={'revision_id'})
    doc=make_document(**values)
    chunks=make_chunks(doc,WordTokenizer(),64)
    validate_chunk_sources(doc,chunks)
    relevant=[chunk for chunk in chunks if any(span.block_id==table.block_id for span in chunk.spans)]
    assert all(chunk.token_count<=64 for chunk in relevant)
    words=set(' '.join(chunk.content for chunk in relevant).split())
    assert set(table.text.split())<=words
    assert any('table_context_split_fetch_parent_for_full_notes' in chunk.quality_flags for chunk in relevant)


def test_multiple_display_elements_inside_one_paragraph_keep_distinct_xml_paths(tmp_path):
    text=FIXTURE.read_text(encoding='utf-8').replace('<p>Last paragraph.</p>',
        '<p>Text<fig id="inline1"><caption><p>First inline figure.</p></caption></fig>'
        '<fig id="inline2"><caption><p>Second inline figure.</p></caption></fig></p>')
    path=tmp_path/'inline.xml'
    path.write_text(text,encoding='utf-8')
    doc=parse_pmc(path,tmp_path/'raw')
    figures=[block for section in doc.sections for block in section.blocks
             if block.kind=='figure_caption' and block.locators[0].source_id in {'inline1','inline2'}]
    assert [block.locators[0].path for block in figures]==['/article/body[1]/sec[2]/p[1]/fig[1]','/article/body[1]/sec[2]/p[1]/fig[2]']
