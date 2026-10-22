"""Kiểm tra vector đã lưu, nguồn dữ liệu và thử truy xuất tiếng Việt."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.rag.embedding_store import INDEX_ROOT, ROOT, DenseIndex, write_json
from src.rag.local_embeddings import LocalE5Embeddings, file_sha256, passage_body

EXAMPLES = {
    "internal": [
        {"query": "Máng nhai là gì?"},
        {"query": "Giá khớp ARCON và NON-ARCON khác nhau ở vị trí hướng dẫn lồi cầu như thế nào?"},
    ],
    "public": [
        {"query": "Triệu chứng và điều trị viêm tủy răng không hồi phục", "source": "byt"},
        {"query": "Nguyên nhân gây viêm lợi do mảng bám", "source": "msd"},
        {"query": "Hút thuốc lá có liên quan đến viêm nha chu như thế nào?", "source": "pmc", "language": "en"},
    ],
}


def check_one(warehouse, directory, encoder):
    index = DenseIndex(directory, encoder=encoder)
    errors = []
    if index.encoder.identity != index.manifest["encoder"]:
        errors.append("Model không khớp")
    for name, expected in index.manifest["source_files"].items():
        if file_sha256(ROOT / name) != expected:
            errors.append(f"Dữ liệu đầu vào đã thay đổi: {name}")
    ids = [doc.id for doc in index.documents]
    if len(ids) != len(set(ids)):
        errors.append("Trùng ID")
    for doc in index.documents:
        if doc.metadata.get("warehouse") != warehouse:
            errors.append(f"Sai kho: {doc.id}")
        if warehouse == "public" and not doc.metadata.get("locators"):
            errors.append(f"Thiếu nguồn công khai: {doc.id}")
        if warehouse == "internal" and (not doc.metadata.get("source_spans") or not doc.metadata.get("pdf_pages")):
            errors.append(f"Thiếu trang nội bộ: {doc.id}")
    # Nhúng lại các hàng ở đầu/giữa/cuối để phát hiện ghi nhầm thứ tự ID/vector.
    rows = sorted({0, len(index.documents) // 2, len(index.documents) - 1})
    recalculated = encoder.encode_inputs(["passage: " + passage_body(index.documents[row]) for row in rows])
    similarities = np.sum(np.asarray(index.vectors[rows]) * recalculated, axis=1)
    if np.min(similarities) < 0.9999:
        errors.append("Vector lưu không khớp văn bản/ID")
    examples = []
    for case in EXAMPLES[warehouse]:
        start = time.perf_counter()
        hits = index.search(case["query"], top_k=5, source=case.get("source"), language=case.get("language"))
        duration = time.perf_counter() - start
        if not hits:
            errors.append("Không có kết quả cho câu thử: " + case["query"])
        if any(case.get("source") and hit.metadata.get("source") != case["source"] for hit in hits):
            errors.append("Bộ lọc nguồn hoạt động sai")
        if any(case.get("language") and hit.metadata.get("language") != case["language"] for hit in hits):
            errors.append("Bộ lọc ngôn ngữ hoạt động sai")
        examples.append({**case, "seconds": duration, "hits": [doc.model_dump() for doc in hits]})
    return {"warehouse": warehouse, "passed": not errors, "count": len(index.documents),
            "shape": list(index.vectors.shape), "dtype": str(index.vectors.dtype),
            "source_files_unchanged": not any("đã thay đổi" in error for error in errors),
            "roundtrip_rows": rows, "roundtrip_cosines": similarities.tolist(),
            "examples": examples, "errors": errors,
            "scope": "Kiểm tra kỹ thuật và ví dụ truy xuất; chưa đo chất lượng RAG hoặc đánh giá chuyên môn."}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", choices=["internal", "public", "both"], default="both")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/embedding_checks")
    args = parser.parse_args()
    encoder = LocalE5Embeddings()
    reports = []
    for warehouse in (["internal", "public"] if args.warehouse == "both" else [args.warehouse]):
        report = check_one(warehouse, args.index_dir / warehouse, encoder)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / (warehouse + ".json"), report)
        reports.append(report)
        print(json.dumps({key: report[key] for key in ["warehouse", "passed", "count", "shape", "roundtrip_cosines", "errors"]}, ensure_ascii=False), flush=True)
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
