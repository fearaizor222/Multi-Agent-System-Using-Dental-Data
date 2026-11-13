"""Kiểm tra lỗi dữ liệu có thể làm sai việc huấn luyện hoặc đánh giá A1."""
from copy import deepcopy
import json

import pytest

from scripts.prepare_a1_dataset import SOURCE, build_records, split_groups
from src.data.a1_dataset import validate_pair, validate_records


@pytest.fixture(scope="module")
def records():
    return build_records(json.loads(SOURCE.read_text(encoding="utf-8")))


def image_record(records):
    return next(row for row in records if row["target"]["targets"] == ["A5"])


def test_rejects_nonexistent_image_even_if_json_shape_is_valid(records):
    row = deepcopy(image_record(records))
    row["target"]["image_ids"] = ["not-attached"]
    with pytest.raises(ValueError, match="Chọn ảnh"):
        validate_pair(row["input"], row["target"])


def test_rejects_unavailable_agent(records):
    row = deepcopy(image_record(records))
    row["input"]["capabilities"]["A5"] = False
    with pytest.raises(ValueError, match="không khả dụng"):
        validate_pair(row["input"], row["target"])


def test_rejects_unknown_targets_and_invented_fields(records):
    row = deepcopy(records[0])
    row["target"]["targets"] = ["A3"]
    with pytest.raises(ValueError):
        validate_pair(row["input"], row["target"])
    row = deepcopy(records[0])
    row["target"]["confidence"] = .99
    with pytest.raises(ValueError):
        validate_pair(row["input"], row["target"])


def test_rejects_schema_valid_but_mismatched_sft_answer(records):
    rows = deepcopy(records[:2])
    rows[0]["messages"][-1]["content"] = rows[1]["messages"][-1]["content"]
    # Hai paraphrase cùng group có chung target; dùng câu trả lời khác để kiểm tra corruption.
    rows[0]["messages"][-1]["content"] = '{"decision":"respond"}'
    with pytest.raises(ValueError, match="messages không khớp"):
        validate_records(rows)


def test_family_never_crosses_splits_and_every_decision_has_holdout(records):
    stats = validate_records(records)
    assert stats["group_leakage"] == stats["current_message_leakage"] == 0
    for split in ("train", "validation", "test"):
        decisions = {row["target"]["decision"] for row in records if row["split"] == split}
        assert decisions == {"route", "clarify", "handoff", "out_of_scope", "respond"}


def test_capability_and_missing_image_counterfactuals_change_action(records):
    for group_id in ("a1_cap_image", "a1_cap_service", "a1_img_table", "a1_ctx_same_price"):
        group = [row for row in records if row["group_id"] == group_id]
        assert len({row["target"]["decision"] for row in group}) == 2
        assert len({row["split"] for row in group}) == 1


def test_exact_same_words_in_different_contexts_stay_together():
    groups = [{"id": f"g{i}", "category": "context", "cases": [{"input": {"message": text}}]}
              for i, text in enumerate(["Giá dịch vụ này bao nhiêu?", "gia dich vu nay bao nhieu?", "Câu hỏi khác"])]
    assignments = split_groups(groups, seed=9)
    assert assignments["g0"] == assignments["g1"]
