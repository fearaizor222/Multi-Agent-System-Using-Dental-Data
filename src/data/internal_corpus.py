"""Gom các file chunk nội bộ đã xử lý, giữ nguồn PDF và xuất một phiên bản mới."""
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from src.rag.base import DentalDocument
from src.rag.local_embeddings import MODEL, file_sha256

SCHEMA = "internal-corpus-v1"


def validate_chunk(item):
    unknown = set(item) - set(DentalDocument.model_fields)
    if unknown:
        raise ValueError("Trường chunk chưa được hỗ trợ: " + ", ".join(sorted(unknown)))
    doc = DentalDocument.model_validate(item)
    if any(not value.strip() for value in (doc.id, doc.title, doc.content)):
        raise ValueError("Mã, tiêu đề và nội dung chunk không được rỗng")
    meta = doc.metadata
    if meta.get("warehouse", "internal") != "internal" or meta.get("source_type") != "internal":
        raise ValueError(f"{doc.id}: chunk phải thuộc kho nội bộ")
    if meta.get("embedding_ready") is not True or meta.get("doc_level") != "section":
        raise ValueError(f"{doc.id}: chunk chưa sẵn sàng cho embedding")
    if meta.get("tokenizer") != MODEL or type(meta.get("token_count")) is not int or not 0 < meta["token_count"] <= 380:
        raise ValueError(f"{doc.id}: sai tokenizer hoặc số token")
    for key in ("source", "source_name", "language"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f"{doc.id}: thiếu {key}")
    for key in ("source_sha256", "source_revision"):
        if not isinstance(meta.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", meta[key]):
            raise ValueError(f"{doc.id}: thiếu hoặc sai {key}")
    identity = meta.get("document_id") or meta.get("parent_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"{doc.id}: thiếu mã tài liệu/chương")
    pages = meta.get("pdf_pages")
    if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
        raise ValueError(f"{doc.id}: thiếu hoặc sai trang PDF")
    # Có bản scan bị đảo trang: pdf_pages giữ thứ tự đọc, khoảng start/end giữ min/max.
    if (len(pages) != len(set(pages)) or type(meta.get("page_start")) is not int
            or type(meta.get("page_end")) is not int
            or meta["page_start"] != min(pages) or meta["page_end"] != max(pages)):
        raise ValueError(f"{doc.id}: danh sách trang không khớp page_start/page_end")
    spans = meta.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{doc.id}: thiếu vị trí văn bản nguồn")
    for span in spans:
        if (not isinstance(span, dict) or type(span.get("pdf_page")) is not int or span["pdf_page"] not in pages
                or type(span.get("start")) is not int or type(span.get("end")) is not int
                or not 0 <= span["start"] < span["end"]):
            raise ValueError(f"{doc.id}: vị trí văn bản nguồn không hợp lệ")
    if {span["pdf_page"] for span in spans} != set(pages):
        raise ValueError(f"{doc.id}: trang và vị trí văn bản nguồn không khớp")
    doc.metadata["warehouse"] = "internal"
    doc.score = 0.0
    return doc


def read_chunks(path):
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        items = json.loads(text)
    else:
        raise ValueError("Chỉ nhận file chunk .json hoặc .jsonl")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: cần danh sách chunk không rỗng")
    documents = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}, bản ghi {index}: chunk phải là object")
        try:
            documents.append(validate_chunk(item))
        except ValueError as exc:
            raise ValueError(f"{path}, bản ghi {index}: {exc}") from exc
    return documents, hashlib.sha256(raw).hexdigest()


def prepare_internal_corpus(chunk_files, pdf_dir, output_dir):
    pdf_dir, output_dir = Path(pdf_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Thư mục đầu ra đã có; chọn phiên bản mới")
    if not chunk_files:
        raise ValueError("Cần ít nhất một file chunk")
    documents, by_id, inputs, revisions, pdfs = [], {}, [], {}, {}
    duplicate_ids = []
    for path in chunk_files:
        path = Path(path).resolve()
        incoming, checksum = read_chunks(path)
        inputs.append({"path": os.path.relpath(path, output_dir), "sha256": checksum, "records": len(incoming)})
        for doc in incoming:
            meta = doc.metadata
            pdf_path = (pdf_dir / meta["source"]).resolve()
            if not pdf_path.is_relative_to(pdf_dir) or pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"{doc.id}: đường dẫn PDF nằm ngoài thư mục nguồn")
            if pdf_path not in pdfs:
                checksum = file_sha256(pdf_path)
                with pdf_path.open("rb") as stream:
                    page_count = len(PdfReader(stream).pages)
                pdfs[pdf_path] = {"path": os.path.relpath(pdf_path, output_dir), "sha256": checksum,
                                  "page_count": page_count, "chunks": 0}
            source = pdfs[pdf_path]
            if source["sha256"] != meta["source_sha256"]:
                raise ValueError(f"{doc.id}: PDF không khớp mã băm nguồn")
            if max(meta["pdf_pages"]) > source["page_count"]:
                raise ValueError(f"{doc.id}: số trang vượt PDF gốc")
            key = (meta["source_sha256"], meta.get("document_id") or meta["parent_id"])
            if key in revisions and revisions[key] != meta["source_revision"]:
                raise ValueError(f"{doc.id}: trộn hai phiên bản của cùng tài liệu/chương")
            revisions[key] = meta["source_revision"]
            if doc.id in by_id:
                if by_id[doc.id].model_dump() != doc.model_dump():
                    raise ValueError(f"Trùng ID nhưng khác nội dung/metadata: {doc.id}")
                duplicate_ids.append(doc.id)
                continue
            by_id[doc.id] = doc
            documents.append(doc)
            source["chunks"] += 1
    payload = "".join(doc.model_dump_json() + "\n" for doc in documents).encode("utf-8")
    manifest = {"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(documents), "pdf_count": len(pdfs), "inputs": inputs, "pdf_sources": list(pdfs.values()),
        "chunks_sha256": hashlib.sha256(payload).hexdigest(), "duplicate_ids_skipped": duplicate_ids,
        "counts_by_source": dict(Counter(doc.metadata["source"] for doc in documents)),
        "counts_by_language": dict(Counter(doc.metadata["language"] for doc in documents)),
        "chunks_with_quality_flags": sum(bool(doc.metadata.get("quality_flags")) for doc in documents),
        "chunks_with_reordered_pdf_pages": sum(doc.metadata["pdf_pages"] != sorted(doc.metadata["pdf_pages"]) for doc in documents),
        "scope": "Kiểm tra cấu trúc, danh tính PDF và giới hạn trang; chưa xác nhận đúng OCR, nội dung bảng, ảnh hay chuyên môn.",
        "token_count_check": "Metadata only; embedding step must recalculate with the actual tokenizer."}
    # Chỉ tạo thư mục sau khi đã rà toàn bộ đầu vào. Không sửa file chunk hoặc PDF gốc.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "chunks.jsonl").write_bytes(payload)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_internal_corpus(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Sai phiên bản cấu trúc kho nội bộ")
    chunks_path = directory / "chunks.jsonl"
    documents, checksum = read_chunks(chunks_path)
    if checksum != manifest["chunks_sha256"] or len(documents) != manifest["count"]:
        raise ValueError("Chunk đã đổi sau lần gom dữ liệu")
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("Trùng ID trong kho đã gom")
    files = {str(chunks_path): checksum, str(directory / "manifest.json"): file_sha256(directory / "manifest.json")}
    for source in [*manifest["inputs"], *manifest["pdf_sources"]]:
        path = (directory / source["path"]).resolve()
        if file_sha256(path) != source["sha256"]:
            raise ValueError("Đầu vào đã đổi sau lần gom dữ liệu: " + str(path))
        files[str(path)] = source["sha256"]
    return documents, files
