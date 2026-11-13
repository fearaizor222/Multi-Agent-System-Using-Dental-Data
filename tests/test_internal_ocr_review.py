"""Ngăn áp nhầm bản sửa OCR khi nguồn hoặc offset thay đổi."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.rag.ocr_review import NORMALIZATION_VERSION, apply_reviewed_corrections, digest
from scripts.clean_internal_ocr import clean


class OCRReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "review.json"
        self.raw = {1: "Mô tả li cầu và uà.", 2: "Giữ nguyên trang này."}
        text = self.raw[1]
        self.data = {"schema_version": 1, "normalization_version": NORMALIZATION_VERSION,
                     "source_sha256": "source", "review_scope": "selected_spans_only", "pages": [
                         {"pdf_page": 1, "normalized_text_sha256": digest(text), "changes": [
                             {"start": text.index("li cầu"), "end": text.index("li cầu") + 6, "before": "li cầu", "after": "lồi cầu"},
                             {"start": text.index("uà"), "end": text.index("uà") + 2, "before": "uà", "after": "và"}]}]}

    def run_review(self, data=None, raw=None, source="source"):
        self.path.write_text(json.dumps(data if data is not None else self.data), encoding="utf-8")
        return apply_reviewed_corrections(raw if raw is not None else self.raw, source, self.path)

    def test_offsets_after_length_change_and_original_preserved(self):
        before = dict(self.raw)
        pages, review = self.run_review()
        self.assertEqual(pages[1], "Mô tả lồi cầu và và.")
        self.assertEqual(pages[2], self.raw[2])
        self.assertEqual(self.raw, before)
        for change in review["pages"][1]["changes"]:
            self.assertEqual(pages[1][change["corrected_start"]:change["corrected_end"]], change["after"])

    def test_reject_changed_pdf(self):
        with self.assertRaisesRegex(ValueError, "PDF nguồn"):
            self.run_review(source="another-pdf")

    def test_reject_changed_ocr_page(self):
        with self.assertRaisesRegex(ValueError, "Chữ OCR đã thay đổi"):
            self.run_review(raw={**self.raw, 1: "OCR mới"})

    def test_reject_wrong_original_word(self):
        data = copy.deepcopy(self.data)
        data["pages"][0]["changes"][0]["before"] = "sai"
        with self.assertRaisesRegex(ValueError, "Nội dung sửa OCR không khớp"):
            self.run_review(data=data)

    def test_reject_overlapping_edits(self):
        data = copy.deepcopy(self.data)
        data["pages"][0]["changes"].insert(1, dict(data["pages"][0]["changes"][0]))
        with self.assertRaisesRegex(ValueError, "Offset sửa OCR trùng"):
            self.run_review(data=data)

    def test_reject_duplicate_page(self):
        data = copy.deepcopy(self.data)
        data["pages"].append(copy.deepcopy(data["pages"][0]))
        with self.assertRaisesRegex(ValueError, "Trang OCR không hợp lệ hoặc trùng"):
            self.run_review(data=data)


class RepeatedPatternTests(unittest.TestCase):
    def test_cleaning_is_repeatable_preserves_manual_edits_and_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = "Lỗi cầu và điện khớp, uà; góc 20 - 281."
            records = [{"pdf_page": n, "text": text if n == 10 else ""} for n in range(1, 315)]
            cache = root / "pages.jsonl"
            cache.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records), encoding="utf-8")
            original = cache.read_bytes()
            (root / "source.json").write_text(json.dumps({"sha256": "source"}), encoding="utf-8")
            a = text.index("uà")
            review = root / "review.json"
            review.write_text(json.dumps({"schema_version": 1, "normalization_version": NORMALIZATION_VERSION,
                "source_sha256": "source", "review_scope": "selected_spans_only", "pages": [
                    {"pdf_page": 10, "normalized_text_sha256": digest(text), "changes": [
                        {"start": a, "end": a + 2, "before": "uà", "after": "và (đã kiểm tra)"}]}]}), encoding="utf-8")
            first = clean(cache, review)
            result, _ = apply_reviewed_corrections({row["pdf_page"]: row["text"] for row in records}, "source", review)
            self.assertEqual(result[10], "Lồi cầu và diện khớp, và (đã kiểm tra); góc 20 - 281.")
            self.assertEqual(first["added_corrections"], 2)
            stable = review.read_bytes()
            self.assertEqual(clean(cache, review)["added_corrections"], 0)
            self.assertEqual(review.read_bytes(), stable)
            self.assertEqual(cache.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
