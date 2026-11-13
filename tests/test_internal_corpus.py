"""Kiểm tra gom nhiều nguồn và chặn dữ liệu sai trước khi cập nhật vector."""
import json
from copy import deepcopy

import pytest
from pypdf import PdfWriter

from src.data.internal_corpus import load_internal_corpus, prepare_internal_corpus, validate_chunk
from src.rag.local_embeddings import MODEL, file_sha256


@pytest.fixture
def source_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for number in (1, 2):
        pdf = raw / f"book-{number}.pdf"
        writer = PdfWriter()
        writer.add_metadata({"/Title": f"Source {number}"})
        writer.add_blank_page(width=300, height=400)
        writer.add_blank_page(width=300, height=400)
        with pdf.open("wb") as stream:
            writer.write(stream)
        record = {"id": f"book-{number}-chunk-1", "title": "Mục thử", "content": "Văn bản để thử liên kết nguồn.",
            "metadata": {"source": pdf.name, "source_name": f"Sách thử {number}", "source_type": "internal",
                "source_sha256": file_sha256(pdf), "source_revision": str(number) * 64, "parent_id": f"book-{number}-ch1",
                "language": "vi", "doc_level": "section", "embedding_ready": True, "tokenizer": MODEL, "token_count": 10,
                "pdf_pages": [1], "page_start": 1, "page_end": 1,
                "source_spans": [{"pdf_page": 1, "start": 0, "end": 34}],
                "quality_flags": ["ocr_text_requires_review"], "images": [{"id": f"image-{number}", "pdf_page": 1}]}}
        chunk = tmp_path / f"book-{number}.json"
        chunk.write_text(json.dumps([record]), encoding="utf-8")
        records.append((chunk, record))
    return raw, records


def test_merge_two_books_preserves_ids_images_flags_and_original_inputs(tmp_path, source_files):
    raw, records = source_files
    before = {path: path.read_bytes() for path, _ in records}
    output = tmp_path / "corpus"
    manifest = prepare_internal_corpus([path for path, _ in records], raw, output)
    docs, sources = load_internal_corpus(output)
    assert manifest["pdf_count"] == 2 and manifest["count"] == 2
    assert [doc.id for doc in docs] == [record["id"] for _, record in records]
    for doc, (path, original) in zip(docs, records):
        assert doc.metadata["images"] == original["metadata"]["images"]
        assert doc.metadata["quality_flags"] == ["ocr_text_requires_review"]
        assert doc.metadata["warehouse"] == "internal"
        assert path.read_bytes() == before[path]
    assert str(output / "chunks.jsonl") in sources


def test_identical_id_is_only_kept_once_and_conflict_is_rejected(tmp_path, source_files):
    raw, records = source_files
    path, record = records[0]
    manifest = prepare_internal_corpus([path, path], raw, tmp_path / "deduplicated")
    assert manifest["count"] == 1
    assert manifest["duplicate_ids_skipped"] == [record["id"]]
    changed = deepcopy(record)
    changed["content"] = "Nội dung khác nhưng dùng nhầm ID."
    other = tmp_path / "conflict.jsonl"
    other.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Trùng ID nhưng khác"):
        prepare_internal_corpus([path, other], raw, output)
    assert not output.exists()


@pytest.mark.parametrize("change, message", [
    ({"source_sha256": "0" * 64}, "PDF không khớp"),
    ({"pdf_pages": [3], "page_start": 3, "page_end": 3,
      "source_spans": [{"pdf_page": 3, "start": 0, "end": 34}]}, "số trang vượt"),
    ({"source": "../outside.pdf"}, "nằm ngoài"),
    ({"warehouse": "public"}, "kho nội bộ"),
    ({"embedding_ready": False}, "chưa sẵn sàng"),
    ({"tokenizer": "other-model"}, "sai tokenizer"),
    ({"pdf_pages": [1, 1]}, "danh sách trang"),
    ({"source_spans": []}, "thiếu vị trí"),
])
def test_invalid_source_and_metadata_do_not_publish_corpus(tmp_path, source_files, change, message):
    raw, records = source_files
    path, record = records[0]
    record["metadata"].update(change)
    path.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        prepare_internal_corpus([path], raw, output)
    assert not output.exists()


def test_reordered_scan_pages_keep_reading_order(source_files):
    _, records = source_files
    record = deepcopy(records[0][1])
    record["metadata"].update(pdf_pages=[2, 1], page_start=1, page_end=2,
        source_spans=[{"pdf_page": 2, "start": 0, "end": 5}, {"pdf_page": 1, "start": 0, "end": 10}])
    assert validate_chunk(record).metadata["pdf_pages"] == [2, 1]


def test_mixed_revision_of_same_chapter_is_rejected(tmp_path, source_files):
    raw, records = source_files
    path, record = records[0]
    changed = deepcopy(record)
    changed["id"] += "-another"
    changed["metadata"]["source_revision"] = "a" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps([changed]), encoding="utf-8")
    with pytest.raises(ValueError, match="trộn hai phiên bản"):
        prepare_internal_corpus([path, other], raw, tmp_path / "rejected")


@pytest.mark.parametrize("target", ["chunks", "input", "pdf"])
def test_changed_files_are_detected_when_loading(tmp_path, source_files, target):
    raw, records = source_files
    path, record = records[0]
    output = tmp_path / "corpus"
    prepare_internal_corpus([path], raw, output)
    if target == "chunks":
        changed = output / "chunks.jsonl"
        with changed.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        changed = path if target == "input" else raw / record["metadata"]["source"]
        with changed.open("ab") as stream:
            stream.write(b"\n")
    with pytest.raises(ValueError, match="đã đổi"):
        load_internal_corpus(output)


def test_existing_corpus_is_not_overwritten(tmp_path, source_files):
    raw, records = source_files
    output = tmp_path / "corpus"
    prepare_internal_corpus([records[0][0]], raw, output)
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="đã có"):
        prepare_internal_corpus([records[1][0]], raw, output)
    assert (output / "manifest.json").read_bytes() == before
