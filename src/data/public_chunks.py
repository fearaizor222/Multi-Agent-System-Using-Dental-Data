"""Chia mẫu thành đoạn; mỗi phần văn bản giữ offset về khối nguồn."""
import re
from typing import Literal

from pydantic import Field

from src.data.public_schema import Locator, PublicDocument, Record, digest


class TextSpan(Record):
    block_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class PublicChunk(Record):
    schema_version: Literal["public-chunk-v1"] = "public-chunk-v1"
    chunk_id: str
    document_id: str
    document_revision: str
    section_id: str
    heading_path: list[str]
    source: str
    source_url: str
    language: str
    content: str
    token_count: int
    spans: list[TextSpan]
    locators: list[Locator]
    chunking: dict
    quality_flags: list[str] = Field(default_factory=list)


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, verbose=False))


def split_offsets(text, fits, overlap_words):
    words = list(re.finditer(r"\S+", text))
    start = 0
    while start < len(words):
        low, high, end = start+1, len(words), start
        while low <= high:
            middle = (low+high)//2
            if fits(words[start].start(), words[middle-1].end()):
                end, low = middle, middle+1
            else:
                high = middle-1
        if end == start:
            raise ValueError("Một từ hoặc ngữ cảnh vượt giới hạn token; cần kiểm tra khối nguồn")
        yield words[start].start(), words[end-1].end()
        if end == len(words):
            break
        start = max(start+1, end-min(overlap_words, end-start-1))


def make_chunks(document: PublicDocument, tokenizer, max_tokens=380, overlap_words=16, tokenizer_name="intfloat/multilingual-e5-small"):
    if not 64 <= max_tokens <= 500 or overlap_words < 0:
        raise ValueError("max_tokens cần trong khoảng 64–500, overlap_words không âm")
    chunks = []
    reserve_tokens = min(64,max_tokens//4)
    config = {"algorithm": "blocks-with-source-spans-v2", "max_tokens": max_tokens,
              "overlap_words": overlap_words, "tokenizer": tokenizer_name, "passage_prefix": "passage: "}
    for section in document.sections:
        context = document.title + "\n" + " > ".join(section.heading_path)
        for block in section.blocks:
            groups = []
            flags = list(block.quality_flags)
            if block.table:
                lines = block.text.splitlines(keepends=True)
                offsets, position = [], 0
                for line in lines:
                    offsets.append((position, position+len(line.rstrip("\n"))))
                    position += len(line)
                prefix_count = bool(block.table.label) + bool(block.table.caption)
                rows = sorted({cell.row for cell in block.table.cells})
                fixed_indices = list(range(prefix_count))
                for index, row in enumerate(rows):
                    if all(cell.is_header for cell in block.table.cells if cell.row == row):
                        fixed_indices.append(prefix_count+index)
                fixed_indices.extend(range(prefix_count+len(rows), len(lines)))
                fixed = [offsets[index] for index in fixed_indices if offsets[index][1]>offsets[index][0]]
                bodies = [offset for index, offset in enumerate(offsets) if index not in fixed_indices]
                # Chú thích bảng có thể dài hơn cả giới hạn embedding. Tách riêng khi cần,
                # vẫn giữ nguyên khối bảng và offset để truy lại toàn bộ ngữ cảnh.
                fixed_text = "\n".join(block.text[a:b] for a,b in fixed)
                if count_tokens(tokenizer,"passage: "+context+"\n"+fixed_text)>max_tokens-reserve_tokens:
                    retained,detached = [],[]
                    header_positions = [prefix_count+index for index,row in enumerate(rows)
                                        if all(cell.is_header for cell in block.table.cells if cell.row==row)]
                    priorities = [offsets[index] for index in header_positions]+fixed
                    for span in dict.fromkeys(priorities):
                        if span[1]<=span[0]:
                            continue
                        candidate = "\n".join(block.text[a:b] for a,b in retained+[span])
                        if count_tokens(tokenizer,"passage: "+context+"\n"+candidate)<=max_tokens-reserve_tokens:
                            retained.append(span)
                        else:
                            detached.append(span)
                    fixed=retained
                    for start,end in detached:
                        fits = lambda a,b: count_tokens(tokenizer,"passage: "+context+"\n"+block.text[a:b])<=max_tokens
                        for a,b in split_offsets(block.text[start:end],lambda a,b:fits(start+a,start+b),overlap_words):
                            groups.append([(start+a,start+b)])
                    flags.append("table_context_split_fetch_parent_for_full_notes")
                if not bodies and fixed:
                    groups.append(fixed)
                for start, end in bodies:
                    if end<=start or not block.text[start:end].strip():
                        continue
                    prefix = "\n".join(block.text[a:b] for a, b in fixed)
                    fits = lambda a, b: count_tokens(tokenizer, "passage: " + context + "\n" + prefix + "\n" + block.text[a:b]) <= max_tokens
                    pieces = list(split_offsets(block.text[start:end], lambda a,b: fits(start+a,start+b), 0))
                    groups.extend(fixed+[(start+a, start+b)] for a,b in pieces)
            else:
                fits = lambda a,b: count_tokens(tokenizer, "passage: " + context + "\n" + block.text[a:b]) <= max_tokens
                groups = [[(a,b)] for a,b in split_offsets(block.text, fits, overlap_words)]
            for spans in groups:
                # Header và chú thích bảng được lặp lại, nhưng từng phần vẫn trỏ về nguồn.
                spans = sorted(set(spans))
                content = context + "\n" + "\n".join(block.text[a:b] for a,b in spans)
                tokens = count_tokens(tokenizer, "passage: " + content)
                if tokens > max_tokens:
                    raise ValueError("Đoạn vượt giới hạn token; không tự cắt ngắn tài liệu")
                source_spans = [TextSpan(block_id=block.block_id, start=a, end=b) for a,b in spans]
                identity = {"revision": document.revision_id, "section": section.section_id,
                            "spans": [span.model_dump() for span in source_spans], "chunking": config}
                chunks.append(PublicChunk(chunk_id="chunk-"+digest(identity), document_id=document.document_id,
                    document_revision=document.revision_id, section_id=section.section_id, heading_path=section.heading_path,
                    source=document.source, source_url=document.source_url, language=document.language,
                    content=content, token_count=tokens, spans=source_spans, locators=block.locators,
                    chunking=config, quality_flags=flags))
    return chunks


def validate_chunk_sources(document, chunks):
    sections = {section.section_id: section for section in document.sections}
    blocks = {block.block_id: (section.section_id, block) for section in document.sections for block in section.blocks}
    for chunk in chunks:
        if chunk.document_id != document.document_id or chunk.document_revision != document.revision_id:
            raise ValueError("Đoạn trỏ đến sai tài liệu/phiên bản")
        section = sections[chunk.section_id]
        text_parts = []
        for span in chunk.spans:
            section_id, block = blocks[span.block_id]
            if section_id != section.section_id or not 0 <= span.start < span.end <= len(block.text):
                raise ValueError("Offset của đoạn không nằm trong khối nguồn")
            text_parts.append(block.text[span.start:span.end])
        context = document.title + "\n" + " > ".join(section.heading_path)
        if chunk.content != context + "\n" + "\n".join(text_parts):
            raise ValueError("Nội dung đoạn không khớp nguồn")
    return True
