"""Kiểm tra JSON kho nội bộ với lớp chữ gốc và tokenizer E5 trên máy."""
import argparse
import csv
import hashlib
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from src.rag.base import DentalDocument

ROOT = Path(__file__).resolve().parents[1]
MODEL = "intfloat/multilingual-e5-small"
# Khoảng trang đối chiếu với mục lục và trang mở đầu trong bản scan.
CHAPTER_PAGES = {
    1: [11, 10, *range(12, 18)], 2: list(range(18, 41)),
    3: list(range(41, 73)), 4: list(range(73, 90)),
    5: list(range(90, 110)), 6: list(range(110, 123)),
    7: list(range(124, 147)), 8: list(range(147, 181)),
    9: list(range(181, 207)), 10: list(range(208, 236)),
    11: list(range(236, 258)), 12: list(range(259, 269)),
    13: list(range(269, 278)),
}
SUSPECT = re.compile(r"\b(?:uà|uù|uê|rỡng|bhớp|bhóp|hhớp|hhop|lỗi\s+cầu|lôi\s+cầu|thái\s+đương)\b", re.I)


def normalized(text):
    text = unicodedata.normalize("NFC", text.replace("ƣ", "ư").replace("Ƣ", "Ư"))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.strip() for line in text.splitlines())).strip()


def write_csv(path, records, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def check(processed, report_dir, cache, pdf, ocr_review_path):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    source_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    source_info = json.loads(cache.with_name("source.json").read_text(encoding="utf-8"))
    require(source_sha == source_info["sha256"], "Cache không khớp PDF")
    records = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_pages = {item["pdf_page"]: item["text"] for item in records}
    require(len(records) == len(raw_pages) == 314 and set(raw_pages) == set(range(1, 315)), "Cache không đủ 314 trang riêng biệt")
    pages = {number: normalized(text) for number, text in raw_pages.items()}
    original_pages = dict(pages)
    review_bytes = ocr_review_path.read_bytes()
    ocr_review = json.loads(review_bytes)
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    require(ocr_review["source_sha256"] == source_sha, "Bảng sửa OCR không khớp PDF")
    require(ocr_review["normalization_version"] == "nfc-whitespace-glyphs-v1", "Sai cơ sở offset sửa OCR")
    applied_rows, review_pages = [], {}
    for record in ocr_review["pages"]:
        number = record["pdf_page"]
        require(number not in review_pages, f"Bảng sửa OCR lặp trang {number}")
        require(hashlib.sha256(pages[number].encode()).hexdigest() == record["normalized_text_sha256"], f"Trang {number}: bảng sửa không khớp chữ OCR")
        cursor, shift = 0, 0
        for change in record["changes"]:
            a, b = change["start"], change["end"]
            require(cursor <= a < b <= len(pages[number]), f"Trang {number}: offset sửa trùng hoặc ngoài trang")
            require(pages[number][a:b] == change["before"], f"Trang {number}: chữ trước sửa không khớp")
            if change.get("method") == "repeated_ocr_pattern":
                require(re.findall(r"\d+(?:[.,]\d+)?", change["before"]) == re.findall(r"\d+(?:[.,]\d+)?", change["after"]),
                        f"Trang {number}: quy tắc OCR tự động đã đổi số liệu")
            applied_rows.append({"pdf_page": number, "id": f"p{number}-{a}", "start": a, "end": b,
                                 "before": change["before"], "after": change["after"],
                                 "method": change.get("method", "visual_comparison_with_pdf_scan"),
                                 "corrected_start": a + shift, "corrected_end": a + shift + len(change["after"])})
            cursor = b
            shift += len(change["after"]) - (b - a)
        for change in reversed(record["changes"]):
            pages[number] = pages[number][:change["start"]] + change["after"] + pages[number][change["end"]:]
        review_pages[number] = record
    numeric_spots = {33: ["5-9 mm"], 222: ["L = H / 8 + 12"], 230: ["70°"], 249: ["20 - 25°"]}
    for number, phrases in numeric_spots.items():
        for phrase in phrases:
            require(phrase in pages[number], f"Trang {number}: mất chỗ sửa số liệu đã đối chiếu ({phrase})")
    names = {
        "chapters": "khop_can_hoc_chapters.json",
        "chunks": "khop_can_hoc_sections.json",
        "glossary": "khop_can_hoc_glossary_review.json",
    }
    documents, files = {}, {}
    for kind, name in names.items():
        path = processed / name
        data = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(data, list), f"{name}: cần mảng JSON")
        documents[kind] = [DentalDocument.model_validate(item) for item in data]
        files[kind] = {"path": str(path.resolve()), "records": len(data),
                       "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    chapters, chunks, glossary = (documents[key] for key in names)
    ids = [doc.id for values in documents.values() for doc in values]
    require(len(ids) == len(set(ids)), "ID bị trùng")
    require(len(chapters) == 13, "Cần có 13 chương")
    require(sorted(doc.metadata["chapter_number"] for doc in chapters) == list(range(1, 14)), "Sai số chương")
    require(len(glossary) == 2, "Cần hai phần thuật ngữ để rà riêng")
    parents = {doc.id: doc for doc in chapters}
    table_data = json.loads(ocr_review_path.with_name("rhm_tables.json").read_text(encoding="utf-8"))
    require(table_data["source_sha256"] == source_sha, "Nguồn của bảng không khớp PDF")
    table_spans = {parent_id: [] for parent_id in parents}
    table_hits = {row["id"]: 0 for row in table_data["rows"]}
    require(len(table_hits) == 9, "Cần đủ 9 hàng bảng đã đối chiếu")
    for row in table_data["rows"]:
        body = f"{row['section']}\nARCON: {row['columns']['ARCON']}\nNON-ARCON: {row['columns']['NON-ARCON']}"
        parent = next(doc for doc in chapters if row["pdf_page"] in doc.metadata["pdf_pages"])
        page_span = next(span for span in parent.metadata["page_spans"] if span["pdf_page"] == row["pdf_page"])
        local = pages[row["pdf_page"]].find(body)
        require(local >= 0, f"{row['id']}: nội dung bảng không khớp bản đối chiếu")
        table_spans[parent.id].append({"id": row["id"], "start": page_span["start"] + local,
                                       "end": page_span["start"] + local + len(body)})
    all_main_pages, all_glossary_pages, rows = [], [], []

    for doc in [*chapters, *glossary]:
        meta = doc.metadata
        numbers = meta["pdf_pages"]
        require(meta["source_sha256"] == source_sha, f"{doc.id}: sai SHA-256 nguồn")
        require(meta["ocr_review"]["sha256"] == review_sha, f"{doc.id}: sai phiên bản bảng sửa OCR")
        require(meta["ocr_review"]["applied_corrections"] == sum(len(review_pages.get(number, {}).get("changes", [])) for number in numbers), f"{doc.id}: sai số chỗ đã sửa OCR")
        require(meta["source_type"] == "internal" and meta["language"] == "vi", f"{doc.id}: sai loại nguồn/ngôn ngữ")
        require(meta["page_start"] == min(numbers) and meta["page_end"] == max(numbers), f"{doc.id}: sai khoảng trang")
        require(doc.content == "\n\n".join(pages[number] for number in numbers), f"{doc.id}: nội dung không khớp các trang OCR")
        require(len(meta["page_spans"]) == len(numbers), f"{doc.id}: thiếu offset trang")
        position = 0
        for i, (number, span) in enumerate(zip(numbers, meta["page_spans"])):
            if i:
                position += 2
            require(span["pdf_page"] == number and span["start"] == position and span["end"] == position + len(pages[number]), f"{doc.id}: sai offset trang {number}")
            require(span["raw_text_sha256"] == hashlib.sha256(raw_pages[number].encode("utf-8")).hexdigest(), f"{doc.id}: sai hash chữ OCR trang {number}")
            require(span["page_text_sha256"] == hashlib.sha256(pages[number].encode()).hexdigest(), f"{doc.id}: sai hash chữ sau sửa trang {number}")
            require(span["ocr_correction_count"] == len(review_pages.get(number, {}).get("changes", [])), f"{doc.id}: sai số chỗ sửa trang {number}")
            position += len(pages[number])
        if doc.id in parents:
            number = meta["chapter_number"]
            require(numbers == CHAPTER_PAGES[number], f"{doc.id}: sai thứ tự/khoảng trang chương")
            require(meta["doc_level"] == "master", f"{doc.id}: sai cấp tài liệu")
            all_main_pages.extend(numbers)
        else:
            expected = list(range(278, 284)) if meta["appendix_kind"] == "definitions" else list(range(284, 293))
            require(numbers == expected, f"{doc.id}: sai trang thuật ngữ")
            require(meta["retrieval_ready"] is False and meta["doc_level"] == "appendix", f"{doc.id}: thuật ngữ chưa được tách khỏi phần truy xuất")
            all_glossary_pages.extend(numbers)

    require(len(all_main_pages) == len(set(all_main_pages)) == 265, "Trang chương bị trùng hoặc thiếu")
    require(len(all_glossary_pages) == len(set(all_glossary_pages)) == 15, "Trang thuật ngữ bị trùng hoặc thiếu")
    require(not set(all_main_pages) & set(all_glossary_pages), "Thuật ngữ bị lẫn vào chương")
    excluded = sorted(set(range(1, 315)) - set(all_main_pages) - set(all_glossary_pages))
    require(excluded == [*range(1, 10), 123, 207, 258, *range(293, 315)], "Sai danh sách trang không đưa vào truy xuất")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    counts, grouped = [], {key: [] for key in parents}
    for doc in chunks:
        meta = doc.metadata
        parent = parents.get(meta["parent_id"])
        require(parent is not None, f"{doc.id}: không có chương cha")
        if parent is None:
            continue
        grouped[parent.id].append(doc)
        a, b = meta["chapter_span"]["start"], meta["chapter_span"]["end"]
        require(0 <= a < b <= len(parent.content), f"{doc.id}: offset ngoài chương")
        require(doc.content == parent.content[a:b] and bool(doc.content.strip()), f"{doc.id}: nội dung không khớp offset chương")
        contained = []
        for row in table_spans[parent.id]:
            if row["start"] < b and row["end"] > a:
                require(a <= row["start"] and row["end"] <= b, f"{doc.id}: cắt giữa hàng bảng {row['id']}")
                if a <= row["start"] and row["end"] <= b:
                    contained.append(row["id"])
                    table_hits[row["id"]] += 1
        require(meta["table_row_ids"] == contained, f"{doc.id}: sai tham chiếu hàng bảng")
        require(meta["chapter_number"] == parent.metadata["chapter_number"] and meta["doc_level"] == "section", f"{doc.id}: sai chương/cấp tài liệu")
        require(meta["source_revision"] == parent.metadata["source_revision"] and meta["source_sha256"] == source_sha, f"{doc.id}: sai phiên bản nguồn")
        expected_spans = []
        for span in parent.metadata["page_spans"]:
            start, end = max(a, span["start"]), min(b, span["end"])
            if start < end:
                expected_spans.append({"pdf_page": span["pdf_page"], "start": start - span["start"],
                                       "end": end - span["start"], "chapter_start": start, "chapter_end": end})
        require(meta["source_spans"] == expected_spans, f"{doc.id}: sai offset trang nguồn")
        numbers = [span["pdf_page"] for span in expected_spans]
        require(numbers == meta["pdf_pages"] and numbers, f"{doc.id}: sai danh sách trang nguồn")
        require(meta["page_start"] == min(numbers) and meta["page_end"] == max(numbers), f"{doc.id}: sai khoảng trang chunk")
        relevant = [change for span in expected_spans for change in applied_rows
                    if change["pdf_page"] == span["pdf_page"] and change["corrected_start"] < span["end"]
                    and change["corrected_end"] > span["start"]]
        require(meta["ocr_review"]["sha256"] == review_sha and meta["ocr_review"]["correction_ids"] == [change["id"] for change in relevant], f"{doc.id}: sai lịch sử sửa OCR")
        require(meta["ocr_review"]["applied_corrections"] == len(relevant), f"{doc.id}: sai số chỗ sửa OCR")
        layout_pages = [number for number in numbers if review_pages.get(number, {}).get("layout_flags")]
        require(meta["layout_review_pages"] == layout_pages and ("table_layout_requires_review" in meta["quality_flags"]) == bool(layout_pages), f"{doc.id}: sai đánh dấu trang bảng cần rà")
        for span in expected_spans:
            require(pages[span["pdf_page"]][span["start"]:span["end"]] == parent.content[span["chapter_start"]:span["chapter_end"]], f"{doc.id}: chữ không khớp trang nguồn")
        count = len(tokenizer.encode("passage: " + doc.title + "\n" + doc.content, verbose=False))
        counts.append(count)
        require(meta["token_count"] == count and 0 < count <= meta["max_tokens"] <= 500, f"{doc.id}: sai token hoặc vượt giới hạn")
        require(meta["tokenizer"] == MODEL, f"{doc.id}: sai tokenizer")
        require(meta["embedding_ready"] == (not meta["ocr_suspects"] and not layout_pages), f"{doc.id}: sai trạng thái đầu vào embedding")

    require(all(table_hits.values()), "Có hàng bảng chưa được đưa đầy đủ vào chunk")

    for parent in chapters:
        pieces = sorted(grouped[parent.id], key=lambda doc: doc.metadata["chunk_index"])
        require([doc.metadata["chunk_index"] for doc in pieces] == list(range(1, len(pieces) + 1)), f"{parent.id}: thiếu số thứ tự chunk")
        covered, previous_start = 0, -1
        for piece in pieces:
            span = piece.metadata["chapter_span"]
            require(span["start"] > previous_start, f"{parent.id}: chunk không tiến về phía trước")
            require(not parent.content[covered:span["start"]].strip(), f"{parent.id}: có nội dung chưa nằm trong chunk")
            covered, previous_start = max(covered, span["end"]), span["start"]
        require(not parent.content[covered:].strip(), f"{parent.id}: thiếu nội dung cuối chương")
        rows.append({"chapter": parent.metadata["chapter_number"], "title": parent.title,
                     "pdf_pages": ",".join(map(str, parent.metadata["pdf_pages"])), "pages": len(parent.metadata["pdf_pages"]),
                     "chunks": len(pieces), "characters": len(parent.content),
                     "max_tokens": max((piece.metadata["token_count"] for piece in pieces), default=0)})

    review = []
    for number in sorted(set(all_main_pages + all_glossary_pages)):
        for match in SUSPECT.finditer(pages[number]):
            review.append({"pdf_page": number, "start": match.start(), "end": match.end(),
                           "text": match.group(0), "context": pages[number][max(0, match.start() - 70):match.end() + 70].replace("\n", " ")})
    duplicates = {text: count for text, count in Counter(doc.content for doc in chunks).items() if count > 1}
    empty = [number for number in all_main_pages if not pages[number]]
    require(not empty, "Có trang chương không trích được chữ")
    require(not duplicates, "Có chunk trùng hệt nội dung")
    require(all("\ufffd" not in doc.content for values in documents.values() for doc in values), "Có ký tự thay thế Unicode")
    methods = dict(Counter(row["method"] for row in applied_rows))
    layout_pending = sorted(number for number, item in review_pages.items() if item.get("layout_flags"))
    ready = not errors and not review and not layout_pending and all(doc.metadata["embedding_ready"] for doc in chunks)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "structure_passed": not errors,
        "ocr_review_complete": False, "medical_review_complete": False,
        "embedding_input_ready": ready,
        "embedding_readiness_basis": "JSON, nguồn, token, độ phủ chữ, hàng bảng và các mẫu OCR đã biết; chưa xác nhận toàn bộ nội dung chuyên môn",
        "source_pdf": str(pdf.resolve()), "source_sha256": source_sha, "files": files,
        "chapters": len(chapters), "chunks": len(chunks), "main_pdf_pages": len(all_main_pages),
        "glossary_pdf_pages_for_review": len(all_glossary_pages), "excluded_pdf_pages": excluded,
        "tokenizer": MODEL, "embedding_input": "passage: {title}\\n{content}",
        "min_tokens": min(counts), "max_tokens": max(counts),
        "chunks_over_380_tokens": sum(count > 380 for count in counts),
        "chunks_over_512_tokens": sum(count > 512 for count in counts),
        "empty_main_pages": empty, "duplicate_chunks": len(duplicates),
        "ocr_corrections_applied": len(applied_rows), "ocr_pages_with_corrections": sorted(review_pages),
        "ocr_correction_methods": methods, "table_rows_checked": len(table_hits), "table_row_chunk_coverage": table_hits,
        "numeric_spot_checks": numeric_spots,
        "ocr_review_sha256": review_sha,
        "ocr_suspect_occurrences_before_corrections": sum(len(SUSPECT.findall(original_pages[number])) for number in set(all_main_pages + all_glossary_pages)),
        "layout_review_pages": layout_pending,
        "ocr_suspect_occurrences": len(review), "ocr_suspect_pages": sorted({item["pdf_page"] for item in review}),
        "chapter_summary": rows, "errors": errors,
        "notes": ["Đã kiểm tra toàn bộ JSON, token, offset, trang nguồn và độ phủ chữ OCR của 13 chương.",
                  "Trang PDF 11 được đọc trước trang 10 theo số trang in trong sách.",
                  "Phần thuật ngữ nhiều cột lưu riêng; trang giới thiệu phần và câu hỏi tự lượng giá không đưa vào chunk.",
                  "Danh sách từ OCR đáng ngờ chỉ là gợi ý rà soát, không phải toàn bộ lỗi.",
                  "Phân biệt sửa trực tiếp theo bản scan và sửa theo mẫu OCR lặp lại đã đối chiếu trên các trang tham chiếu.",
                  "Đã sắp lại bảng ARCON/NON-ARCON; mỗi hàng giữ đầy đủ hai cột trong cùng chunk.",
                  "Giữ nguyên PDF và lớp chữ gốc; có thể dựng lại kết quả từ bảng sửa theo trang/offset.",
                  "OCR mới chỉ dùng thử để đối chiếu; không thay toàn bộ lớp chữ. Chưa trích hình hoặc kiểm tra chuyên môn.",
                  "embedding_input_ready chỉ xác nhận đầu vào kỹ thuật và các lỗi đã biết; không xác nhận đã rà từng chữ hoặc kiểm tra chuyên môn."]}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report_dir / "chapters.csv", rows, ["chapter", "title", "pdf_pages", "pages", "chunks", "characters", "max_tokens"])
    write_csv(report_dir / "ocr_review.csv", review, ["pdf_page", "start", "end", "text", "context"])
    write_csv(report_dir / "ocr_corrections_applied.csv", applied_rows, ["pdf_page", "id", "start", "end", "before", "after", "method", "corrected_start", "corrected_end"])
    table = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in ["chapter", "title", "pages", "chunks", "max_tokens"]) + "</tr>" for row in rows)
    examples = []
    for parent in chapters:
        piece = grouped[parent.id][0]
        examples.append(f"<details><summary>{html.escape(parent.title)} — ví dụ chunk đầu</summary><p>{html.escape(piece.id)} · Trang PDF {piece.metadata['pdf_pages']}</p><pre>{html.escape(piece.content)}</pre></details>")
    status = "Đạt kiểm tra cấu trúc" if not errors else "Có lỗi cấu trúc"
    page_html = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra kho nội bộ</title>
<style>body{{font:16px/1.6 Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#202b36}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#edf2f7}}pre{{white-space:pre-wrap;font:inherit;background:#f4f6f8;padding:16px}}details{{margin:12px 0}}.note{{background:#fff4d6;padding:16px}}</style>
<h1>Kiểm tra kho nội bộ — Cắn khớp học</h1><p><strong>{status}</strong>: {len(chapters)} chương, {len(chunks)} chunk, tối đa {max(counts)} token/chunk.</p>
<p>Kiểm tra toàn bộ chunk với chữ sau khi áp dụng bản sửa và vị trí trang. Giới hạn token tính cả tiêu đề, tiền tố passage và token đặc biệt của E5.</p>
<p>Đã áp dụng {len(applied_rows)} bản sửa trên {len(review_pages)} trang: {methods.get('visual_comparison_with_pdf_scan', 0)} chỗ đối chiếu trực tiếp và {methods.get('repeated_ocr_pattern', 0)} chỗ theo mẫu OCR lặp lại đã được đối chiếu trên các trang tham chiếu. Xem <a href="ocr_corrections_applied.csv">nội dung trước/sau và cách sửa</a>.</p>
<p>Đã sắp lại {len(table_hits)} hàng bảng ARCON/NON-ARCON ở trang 216–217. Kiểm tra từng hàng có đủ hai cột và được giữ nguyên trong chunk.</p>
<p class="note">Còn {len(review)} vị trí khớp danh sách mẫu OCR đang kiểm tra; con số này không khẳng định văn bản hoàn toàn không có lỗi. Chưa rà từng chữ hoặc kiểm tra chuyên môn. Thuật ngữ nhiều cột và hình ảnh chưa đưa vào kho embedding.</p>
<p><strong>Đầu vào kỹ thuật cho embedding: {'Đạt' if ready else 'Cần xử lý thêm'}.</strong></p>
<h2>Kết quả theo chương</h2><table><tr><th>Chương</th><th>Tên</th><th>Trang PDF</th><th>Chunk</th><th>Token lớn nhất</th></tr>{table}</table>
<h2>Nội dung mẫu</h2>{''.join(examples)}<h2>Chi tiết</h2><p><a href="checks.json">Kết quả kiểm tra JSON</a> · <a href="chapters.csv">Danh sách chương</a></p>
<p>Chương 1 đọc trang PDF 11 trước trang 10. Bỏ trang đầu sách, trang phân phần 123/207/258 và câu hỏi tự lượng giá 293–314. Hai phần thuật ngữ nằm ở trang 278–292.</p>
<pre>{html.escape(json.dumps(errors, ensure_ascii=False, indent=2)) if errors else 'Không phát hiện lỗi cấu trúc.'}</pre></html>"""
    (report_dir / "report.html").write_text(page_html, encoding="utf-8")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/internal_rhm_check")
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/internal_rhm_check/source_pages.jsonl")
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/raw/RHM1 khop can hoc.pdf")
    parser.add_argument("--ocr-review", type=Path, default=ROOT / "data/curated/rhm_ocr_corrections.json")
    args = parser.parse_args()
    report = check(args.processed, args.report_dir, args.page_cache, args.pdf, args.ocr_review)
    print(json.dumps({key: report[key] for key in ["structure_passed", "chapters", "chunks", "max_tokens", "chunks_over_512_tokens", "ocr_corrections_applied", "ocr_suspect_occurrences", "errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["structure_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
