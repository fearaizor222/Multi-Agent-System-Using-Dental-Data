import json

import pytest

from src.agents.contracts import evidence_bundle
from src.data.imported_internal_chunks import load_imported_internal_chunks
from src.rag.local_embeddings import MODEL, passage_body


def record():
    return {"schema_version": "public-chunk-v1", "chunk_id": "chunk-1", "document_id": "RHM1-CH12",
            "document_revision": "a" * 64, "section_id": "RHM1-CH12:s1", "heading_path": ["Máng nhai"],
            "source": "rhm1", "source_url": "data/raw/book.pdf", "language": "vi",
            "content": "Máng nhai\nNội dung kiểm thử.", "token_count": 20,
            "spans": [{"block_id": "b1", "start": 0, "end": 10}],
            "locators": [{"path": "data/raw/book.pdf", "pdf_page": 12}],
            "chunking": {"tokenizer": MODEL, "passage_prefix": "passage: "}, "quality_flags": ["ocr_review"]}


def load(tmp_path, rows):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return load_imported_internal_chunks(path)


def test_internal_mapping_preserves_content_and_source_without_duplicating_title(tmp_path):
    item = record()
    docs, sources = load(tmp_path, [item])
    doc = docs[0]
    assert doc.content == item["content"] == passage_body(doc)
    assert doc.id == item["chunk_id"]
    assert doc.metadata["locators"][0]["pdf_page"] == 12
    assert doc.metadata["quality_flags"] == ["ocr_review"]
    assert doc.metadata["source_type"] == doc.metadata["warehouse"] == "internal"
    assert "source_sha256" not in doc.metadata  # No invented original-PDF hash.
    assert doc.category == "Chung"
    assert len(sources) == 2
    evidence = evidence_bundle(docs, "I", "internal")
    assert evidence[0].source_id == item["document_id"]


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load(tmp_path, [record(), record()])


def test_mixed_document_revisions_are_rejected(tmp_path):
    other = record()
    other.update(chunk_id="chunk-2", document_revision="b" * 64)
    with pytest.raises(ValueError, match="Mixed"):
        load(tmp_path, [record(), other])


@pytest.mark.parametrize("change", [{"token_count": 381}, {"content": " "}, {"locators": []},
                                    {"chunking": {"tokenizer": "other"}}, {"heading_path": []}])
def test_invalid_or_incompatible_chunks_are_rejected(tmp_path, change):
    item = record()
    item.update(change)
    with pytest.raises(ValueError):
        load(tmp_path, [item])
