"""Cấu trúc chung cho tài liệu công khai, độc lập với model và vector store."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "public-document-v1"
PARSER_VERSION = "1.0.1"


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Locator(Record):
    url: str | None = None
    source_id: str | None = None
    path: str | None = None
    pdf_page: int | None = Field(default=None, ge=1)
    printed_page: str | None = None
    bbox: list[float] | None = None


class RawSource(Record):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str
    acquisition_url: str | None = None
    retrieved_at: str | None = None


class TableCell(Record):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    text: str
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    is_header: bool = False


class TableData(Record):
    label: str | None = None
    caption: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    cells: list[TableCell]
    original_markup: str | None = None
    header_inferred: bool = False

    @model_validator(mode="after")
    def check_cells(self):
        occupied = set()
        for cell in self.cells:
            for row in range(cell.row, cell.row + cell.rowspan):
                for column in range(cell.column, cell.column + cell.colspan):
                    if (row, column) in occupied:
                        raise ValueError("Các ô trong bảng bị chồng lên nhau")
                    occupied.add((row, column))
        if not self.cells:
            raise ValueError("Bảng không có ô dữ liệu")
        return self


def table_text(table):
    parts = [value for value in (table.label, table.caption) if value]
    rows = sorted({cell.row for cell in table.cells})
    for row in rows:
        cells = sorted((cell for cell in table.cells if cell.row == row), key=lambda cell: cell.column)
        parts.append(" | ".join(cell.text for cell in cells))
    parts.extend(table.footnotes)
    return "\n".join(parts)


class Block(Record):
    block_id: str
    order: int = Field(ge=0)
    kind: Literal["paragraph", "list", "table", "figure_caption", "note"]
    text: str = Field(min_length=1)
    locators: list[Locator] = Field(min_length=1)
    list_items: list[str] = Field(default_factory=list)
    table: TableData | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_kind(self):
        if (self.kind == "table") != (self.table is not None):
            raise ValueError("Khối table phải có dữ liệu bảng; các khối khác không có table")
        if self.table and self.text != table_text(self.table):
            raise ValueError("Văn bản bảng không khớp các ô và chú thích")
        if self.kind == "list" and not self.list_items:
            raise ValueError("Danh sách thiếu các mục")
        return self


class Section(Record):
    section_id: str
    parent_section_id: str | None = None
    order: int = Field(ge=0)
    title: str
    heading_path: list[str]
    locator: Locator
    blocks: list[Block] = Field(default_factory=list)


class Processing(Record):
    parser: str
    parser_version: str = PARSER_VERSION
    ingested_at: str = Field(default_factory=now_utc)
    method: str
    quality_flags: list[str] = Field(default_factory=list)


class PublicDocument(Record):
    schema_version: Literal["public-document-v1"] = SCHEMA_VERSION
    document_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    revision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1)
    source: Literal["byt", "msd", "pmc", "rhm1", "internal", "nidcr", "dph"]
    source_type: Literal["public", "internal"] = "public"
    language: str
    source_url: str
    content_type: Literal["full_text", "abstract"] = "full_text"
    content_scope: Literal["full_document", "page_subset"] = "full_document"
    publication_date: str | None = None
    updated_date: str | None = None
    source_metadata: dict = Field(default_factory=dict)
    raw: RawSource
    processing: Processing
    sections: list[Section] = Field(min_length=1)

    @model_validator(mode="after")
    def check_structure(self):
        section_ids, block_ids, orders = {}, set(), set()
        for section in self.sections:
            if section.section_id in section_ids or section.order in orders:
                raise ValueError("Mã mục hoặc thứ tự đọc bị trùng")
            if section.parent_section_id:
                parent = section_ids.get(section.parent_section_id)
                expected_path = parent.heading_path + ([section.title] if section.title else []) if parent else None
                if parent is None or parent.order >= section.order or section.heading_path != expected_path:
                    raise ValueError("Mục con không khớp mục cha")
            elif section.heading_path != ([section.title] if section.title else []):
                raise ValueError("Đường dẫn mục gốc không khớp tiêu đề")
            section_ids[section.section_id] = section
            orders.add(section.order)
        for section in self.sections:
            for block in section.blocks:
                if block.block_id in block_ids or block.order in orders or block.order <= section.order:
                    raise ValueError("Mã khối/thứ tự đọc bị trùng hoặc khối đứng trước đề mục")
                block_ids.add(block.block_id)
                orders.add(block.order)
        if not block_ids:
            raise ValueError("Tài liệu không có nội dung")
        if self.revision_id != revision_digest(self.model_dump()):
            raise ValueError("Nội dung không khớp revision_id")
        return self


def revision_digest(data):
    # Ngày chạy và đường dẫn cache không làm đổi phiên bản nội dung.
    keys = ("schema_version", "document_id", "title", "source", "language", "source_url",
            "content_type", "content_scope", "publication_date", "updated_date", "source_metadata", "sections")
    payload = {key: data.get(key) for key in keys}
    payload["parser_version"] = data["processing"]["parser_version"]
    return digest(payload)


def make_document(**data):
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("content_type", "full_text")
    data.setdefault("content_scope", "full_document")
    data.setdefault("source_metadata", {})
    for key in ("publication_date", "updated_date"):
        data.setdefault(key, None)
    data["sections"] = [section.model_dump() if isinstance(section, Section) else section for section in data["sections"]]
    for key in ("raw", "processing"):
        if isinstance(data[key], BaseModel):
            data[key] = data[key].model_dump()
    data["revision_id"] = revision_digest(data)
    return PublicDocument.model_validate(data)


class DocumentBuilder:
    """Thứ tự đọc dùng chung cho cả đề mục và khối, kể cả mục lồng nhau."""
    def __init__(self, document_id):
        self.document_id = document_id
        self.sections = []
        self.order = 0

    def section(self, title, locator, parent=None):
        self.order += 1
        section = Section(section_id=f"{self.document_id}:s{len(self.sections)+1:04}",
                          parent_section_id=parent.section_id if parent else None,
                          order=self.order, title=title,
                          heading_path=(parent.heading_path if parent else []) + ([title] if title else []),
                          locator=locator)
        self.sections.append(section)
        return section

    def block(self, section, kind, text, locator, **extra):
        if not text.strip():
            return None
        self.order += 1
        locators = locator if isinstance(locator, list) else [locator]
        block = Block(block_id=f"{self.document_id}:b{self.order:05}", order=self.order,
                      kind=kind, text=text, locators=locators, **extra)
        section.blocks.append(block)
        return block

    def finish_sections(self):
        by_id = {section.section_id: section for section in self.sections}
        keep = {section.section_id for section in self.sections if section.blocks}
        for section_id in list(keep):
            parent = by_id[section_id].parent_section_id
            while parent:
                keep.add(parent)
                parent = by_id[parent].parent_section_id
        return [section for section in self.sections if section.section_id in keep]


def write_documents(path, documents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for document in documents:
        archive = path.parent/"revisions"/document.document_id/(document.revision_id+".json")
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        else:
            archived = PublicDocument.model_validate_json(archive.read_text(encoding="utf-8"))
            if archived.document_id != document.document_id or archived.revision_id != document.revision_id:
                raise ValueError("Bản tài liệu lưu trữ không khớp mã hoặc phiên bản")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(document.model_dump_json() + "\n" for document in documents), encoding="utf-8")
    temporary.replace(path)


def save_raw(input_path, directory, media_type, acquisition_url=None, retrieved_at=None):
    """Bản gốc lưu theo SHA-256; chạy lại không ghi đè phiên bản cũ."""
    input_path, directory = Path(input_path), Path(directory)
    data = input_path.read_bytes()
    identifier = digest(data)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (identifier + input_path.suffix.lower())
    if not target.exists():
        target.write_bytes(data)
    elif digest(target.read_bytes()) != identifier:
        raise ValueError(f"Bản cache gốc đã bị thay đổi: {target}")
    return RawSource(path=target.as_posix(), sha256=identifier, media_type=media_type,
                     acquisition_url=acquisition_url, retrieved_at=retrieved_at)
