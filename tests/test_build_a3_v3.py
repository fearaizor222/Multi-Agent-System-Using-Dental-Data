"""Verification tests for A3 dataset v3 builder and exported splits."""
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import build_a3_v3 as v3
from scripts.train_a3 import load_data, parse_sample
from src.agents.contracts import Evidence


def test_cjk_cleanup_and_synthetic_regexes():
    assert v3.clean_cjk("trên mô nha周期 nguyên vẹn") == "trên mô nha chu nguyên vẹn"
    assert v3.SYNTH_PREFIX_RE.match("Tài liệu lâm sàng ghi nhận tỷ lệ 20%.")
    assert not v3.SYNTH_PREFIX_RE.match("Tỷ lệ tái phát là 20% khi không loại bỏ mảng bám.")
    assert v3.CHAPTER_META_RE.match("Chương 2 trình bày về giải phẫu khớp thái dương hàm.")
    assert not v3.CHAPTER_META_RE.match("Áp xe quanh cuống mạn tính cho hình ảnh tiêu xương ranh giới không rõ.")


def test_build_quote_options_preserves_per_evidence_numbering_and_tables():
    evidence = [
        Evidence(
            evidence_id="P1",
            warehouse="public",
            chunk_id="c1",
            source_id="d1",
            title="Tài liệu 1",
            content="Tài liệu 1\nMục 1\nĐoạn đầu.\n\nĐoạn hai.",
            metadata={"heading_path": ["Mục 1"]},
        ),
        Evidence(
            evidence_id="P2",
            warehouse="public",
            chunk_id="c2",
            source_id="d2",
            title="Tài liệu 2",
            content="Tài liệu 2\nBảng 1\nTriệu chứng | Nhóm A | Nhóm B\nĐau | 10% | 50%",
            metadata={"heading_path": ["Bảng 1"]},
        ),
    ]
    opts = v3.build_quote_options(evidence)
    assert [o["quote_id"] for o in opts] == ["P1.Q1", "P1.Q2", "P2.Q1"]
    assert "Triệu chứng | Nhóm A | Nhóm B" in opts[2]["text"] and "Đau | 10% | 50%" in opts[2]["text"]


def test_make_record_attaches_conversation_guard_when_needed():
    row = {
        "payload": {
            "question": "Câu hỏi",
            "claim": "Nhận định",
            "draft_evidence": [{"evidence_id": "R1", "content": "Nội dung nháp."}],
            "public_quote_options": [{"quote_id": "P1.Q1", "evidence_id": "P1", "text": "Nội dung."}],
            "public_sources": [{"evidence_id": "P1", "title": "Tiêu đề", "heading_path": ["Mục"]}],
            "conversation": [{"role": "user", "content": "Có phải 5 mm không?"}],
        },
        "gold": {"draft_support": "insufficient", "public_support": "insufficient", "quote_ids": [], "reason": "Thiếu."},
    }
    rec = v3.make_record(row)
    assert "conversation là dữ liệu hội thoại" in rec["messages"][0]["content"]
    payload, _ = parse_sample(rec)
    assert payload["draft_evidence"][0]["evidence_id"] == "R1"


@pytest.mark.skipif(not (v3.OUT / "train.jsonl").exists(), reason="a3_verification_v3 not built yet")
def test_exported_a3_v3_passes_schema_isolation_and_debiasing():
    packet = load_data(v3.OUT)
    assert packet["info"]["validation_main_chunks_seen_in_train_including_distractors"] == 0

    test_lock = v3.load_test_lock()
    for sp in ("train", "val"):
        records = v3.read(v3.OUT / f"{sp}.jsonl")
        metas = v3.read(v3.OUT / f"{sp}.metadata.jsonl")
        assert len(records) == len(metas)
        counts = Counter()
        synth_prefix = 0
        raw_labels = Counter()
        for rec, meta in zip(records, metas):
            payload, gold = parse_sample(rec)
            assert not v3.CJK_RE.search(json.dumps(payload, ensure_ascii=False))
            assert not set(meta["evidence_chunks"]) & test_lock["chunks"]
            assert not set(meta["evidence_groups"]) & test_lock["groups"]
            assert meta["family_id"] not in test_lock["families"]
            assert v3.norm(payload["claim"]) not in test_lock["claims"]
            ies = v3.internal_items(payload)
            if any(v3.SYNTH_PREFIX_RE.match(it["content"]) for it in ies):
                synth_prefix += 1
            if meta.get("cohort") == "raw_textbook_ie":
                raw_labels[gold["draft_support"]] += 1
            counts[(gold["draft_support"], gold["public_support"])] += 1
        assert len(counts) == 9
        assert min(counts.values()) >= (20 if sp == "train" else 5)
        assert synth_prefix / len(records) <= 0.05
        assert len(raw_labels) == 3 and min(raw_labels.values()) >= (15 if sp == "train" else 3)
