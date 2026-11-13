"""So sánh index nội bộ mới/cũ và thử cập nhật trên một tập con có kiểm soát."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.rag.embedding_store import DenseIndex, INDEX_ROOT, ROOT, build_index
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256, passage_body


def snapshot(directory):
    return {name: file_sha256(directory / name) for name in ("manifest.json", "records.jsonl", "vectors.npy")}


def input_key(doc):
    return hashlib.sha256(("passage: " + passage_body(doc)).encode("utf-8")).hexdigest()


def controlled_incremental_check(documents, encoder, output):
    if len(documents) < 6:
        raise ValueError("Cần ít nhất 6 chunk để thử cập nhật có kiểm soát")
    original = documents[:4]
    extended = [documents[index] for index in (4, 1, 0, 5, 3, 2)]
    reversed_order = list(reversed(extended))
    a, b, c = (output / name for name in ("initial_4", "added_2_reordered", "reordered_again"))
    build_index(original, {"scope": "controlled subset of existing book"}, a, encoder)
    added = build_index(extended, {"scope": "controlled subset of existing book"}, b, encoder, reuse_from=a)
    reordered = build_index(reversed_order, {"scope": "controlled subset of existing book"}, c, encoder, reuse_from=b)
    assert (added["reused_rows"], added["new_embedding_rows"]) == (4, 2)
    assert (reordered["reused_rows"], reordered["new_embedding_rows"]) == (6, 0)
    direct = encoder.encode_inputs(["passage: " + passage_body(doc) for doc in reversed_order])
    actual = DenseIndex(c).vectors
    np.testing.assert_allclose(actual, direct, rtol=1e-5, atol=1e-6)
    first, second = DenseIndex(a), DenseIndex(b)
    mapping = {doc.id: index for index, doc in enumerate(second.documents)}
    assert all(np.array_equal(first.vectors[i], second.vectors[mapping[doc.id]]) for i, doc in enumerate(first.documents))
    return {"scope": "Tập con của giáo trình hiện có; không phải tài liệu mới bổ sung.",
        "initial_rows": 4, "after_addition_rows": 6, "reused_rows_after_addition": added["reused_rows"],
        "new_embedding_rows_after_addition": added["new_embedding_rows"],
        "reused_rows_after_reordering": reordered["reused_rows"], "direct_encoding_matches": True,
        "unchanged_vectors_bitwise_equal": True}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-index", type=Path, default=INDEX_ROOT / "internal")
    parser.add_argument("--new-index", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "config/retrieval_smoke_cases.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-incremental", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Thư mục kết quả đã có; chọn thư mục mới")
    cases_bytes = args.cases.read_bytes()
    cases = [case for case in json.loads(cases_bytes)["cases"] if case["warehouse"] == "internal"]
    if not cases:
        parser.error("Không có câu hỏi nội bộ trong bộ thử")
    before = {"old": snapshot(args.old_index), "new": snapshot(args.new_index)}
    encoder = LocalE5Embeddings()
    old, new = DenseIndex(args.old_index, encoder), DenseIndex(args.new_index, encoder)
    if old.manifest["encoder"] != new.manifest["encoder"]:
        raise ValueError("Hai index dùng model khác nhau")
    old_ids, new_ids = {doc.id for doc in old.documents}, {doc.id for doc in new.documents}
    by_text = {input_key(doc): i for i, doc in enumerate(old.documents)}
    reused = [(i, by_text[input_key(doc)]) for i, doc in enumerate(new.documents) if input_key(doc) in by_text]
    matching_vectors = sum(np.array_equal(new.vectors[i], old.vectors[j]) for i, j in reused)
    if matching_vectors != len(reused):
        raise ValueError("Vector của nội dung không đổi không khớp index cũ")
    retrievers = {"old": HybridRetriever.from_disk(args.old_index, encoder=encoder),
                  "new": HybridRetriever.from_disk(args.new_index, encoder=encoder)}
    results = []
    for case in cases:
        result = {"case_id": case["id"], "question": case["query"], "reference_ids": case["relevant_ids"]}
        for name, retriever in retrievers.items():
            hits = retriever.retrieve(case["query"], top_k=5, source=case.get("source"), language=case.get("language"))
            result[name] = {"ids": [doc.id for doc in hits], "sources": [doc.metadata for doc in hits],
                "hit_at_1": bool(hits and hits[0].id in case["relevant_ids"]),
                "hit_at_5": any(doc.id in case["relevant_ids"] for doc in hits)}
        result["same_top5_order"] = result["old"]["ids"] == result["new"]["ids"]
        results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cases_used.json").write_bytes(cases_bytes)
    (args.output_dir / "retrieval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"run_at": datetime.now(timezone.utc).isoformat(), "old_count": len(old.documents), "new_count": len(new.documents),
        "added_ids": sorted(new_ids - old_ids), "removed_ids": sorted(old_ids - new_ids),
        "unchanged_input_rows": len(reused), "unchanged_vectors_bitwise_equal": matching_vectors == len(reused),
        "case_count": len(cases), "same_top5_order_count": sum(case["same_top5_order"] for case in results),
        "retrieval": {name: {"hit_at_1": sum(case[name]["hit_at_1"] for case in results),
                               "hit_at_5": sum(case[name]["hit_at_5"] for case in results)} for name in retrievers},
        "scope": "Kiểm tra cập nhật index và truy xuất trên kho hiện có; không đánh giá câu trả lời hay ảnh."}
    if args.check_incremental:
        summary["controlled_incremental"] = controlled_incremental_check(new.documents, encoder, args.output_dir / "controlled")
    after = {"old": snapshot(args.old_index), "new": snapshot(args.new_index)}
    assert before == after, "Index nguồn bị thay đổi trong khi kiểm tra"
    summary.update(source_indices_unchanged=True, source_hashes=after)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
