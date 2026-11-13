"""Adapt imported public-chunk-v1 records to the internal warehouse, preserving provenance."""
import re
from pathlib import Path

from src.data.public_chunks import PublicChunk
from src.rag.base import DentalDocument
from src.rag.local_embeddings import MODEL, file_sha256


def load_imported_internal_chunks(path):
    path = Path(path).resolve()
    documents, revisions, ids = [], {}, set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        chunk = PublicChunk.model_validate_json(line)
        if (not all(value.strip() for value in (chunk.chunk_id, chunk.document_id, chunk.section_id,
                                                chunk.source, chunk.source_url, chunk.language, chunk.content))
                or not chunk.heading_path or any(not value.strip() for value in chunk.heading_path)
                or not re.fullmatch(r"[0-9a-f]{64}", chunk.document_revision)
                or chunk.chunking.get("tokenizer") != MODEL
                or chunk.chunking.get("passage_prefix") != "passage: "
                or not 0 < chunk.token_count <= 380
                or not chunk.spans or not chunk.locators
                or any(span.start >= span.end for span in chunk.spans)):
            raise ValueError(f"Invalid internal import at line {number}: {chunk.chunk_id}")
        if chunk.chunk_id in ids:
            raise ValueError("Duplicate imported chunk ID: " + chunk.chunk_id)
        previous = revisions.setdefault(chunk.document_id, chunk.document_revision)
        if previous != chunk.document_revision:
            raise ValueError("Mixed document revisions: " + chunk.document_id)
        ids.add(chunk.chunk_id)
        metadata = chunk.model_dump(exclude={"content"})
        metadata.update(warehouse="internal", source_type="internal", doc_level="section",
                        parent_id=chunk.document_id, content_includes_title=True,
                        import_format="public-chunk-v1", source_name=chunk.heading_path[0],
                        source_validation="Imported locators retained; original PDFs not verified by this adapter.")
        documents.append(DentalDocument(id=chunk.chunk_id, title=chunk.heading_path[0],
                                        content=chunk.content, category="Chung", metadata=metadata))
    if not documents:
        raise ValueError("Empty internal import")
    return documents, {str(path): file_sha256(path), str(Path(__file__).resolve()): file_sha256(__file__)}
