"""Verify imported internal index against original records and run retrieval-only smoke checks."""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.agents.contracts import evidence_bundle
from src.data.imported_internal_chunks import load_imported_internal_chunks
from src.rag.embedding_store import DenseIndex
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256, passage_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report exists; use a new output path")
    expected, source_hashes = load_imported_internal_chunks(args.source)
    index = DenseIndex(args.index)
    if any(index.manifest["source_files"].get(name) != checksum for name, checksum in source_hashes.items()):
        raise ValueError("Import file/adapter changed after index build")
    if len(expected) != len(index.documents):
        raise ValueError("Index count differs from import")
    for incoming, stored in zip(expected, index.documents):
        # Only token counts are recalculated during embedding; all provenance/content must match.
        metadata = dict(stored.metadata)
        if not metadata.pop("token_count_recalculated", False):
            raise ValueError("Missing actual token audit")
        metadata["token_count"] = metadata.pop("imported_token_count")
        restored = stored.model_copy(update={"metadata": metadata})
        if restored.model_dump() != incoming.model_dump():
            raise ValueError("Imported content, ID or provenance changed: " + incoming.id)
    encoder = LocalE5Embeddings(batch_size=8, threads=2)
    if index.manifest["encoder"] != encoder.identity:
        raise ValueError("Wrong embedding model")
    lengths = encoder.token_lengths(["passage: " + passage_body(doc) for doc in index.documents])
    if any(length != doc.metadata["token_count"] or length > 512 for doc, length in zip(index.documents, lengths)):
        raise ValueError("Actual input token count differs")
    retriever = HybridRetriever.from_disk(args.index, encoder=encoder, candidate_k=10)
    checks = []
    for query, source, language, title_term in (("máng nhai", "rhm1", "vi", "máng nhai"),
                                    ("Careers in Dental Public Health", "dph", "en", "careers"),
                                    ("brushing caregivers", "internal", "en", "caregivers")):
        hits = retriever.retrieve(query, top_k=3, source=source, language=language)
        if not hits or any(hit.metadata["warehouse"] != "internal" or hit.metadata["source"] != source for hit in hits):
            raise ValueError("Retrieval source/warehouse mismatch")
        if not any(title_term in hit.title.casefold() for hit in hits):
            raise ValueError("Expected topic absent from top 3: " + query)
        evidence = evidence_bundle(hits, "I", "internal")
        checks.append({"query": query, "source_filter": source, "language_filter": language,
                       "ids": [hit.id for hit in hits], "titles": [hit.title for hit in hits],
                       "evidence_ids": [item.evidence_id for item in evidence]})
    result = {"status": "passed", "count": len(expected), "source_files_sha256": source_hashes,
              "index": str(args.index.resolve()), "index_manifest_sha256": file_sha256(args.index / "manifest.json"),
              "counts_by_source": dict(Counter(doc.metadata["source"] for doc in index.documents)),
              "counts_by_language": dict(Counter(doc.metadata["language"] for doc in index.documents)),
              "content_ids_provenance_preserved": True, "actual_max_tokens": max(lengths),
              "truncated_records": 0, "retrieval_checks": checks,
              "scope": "Exact import/index verification, real dense+BM25+RRF retrieval and A2 evidence mapping. No LLM, reranker, fine-tuning or clinical answer evaluation. Original PDF files not verified."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
