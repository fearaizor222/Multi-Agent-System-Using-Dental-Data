"""Chấm điều phối đúng cả schema, capability và tham số, không chỉ nhãn route."""
import json
from copy import deepcopy

import pytest

from src.training.a1_evaluation import strict_json, score_response, summarize, selection_score
from scripts.train_a1 import validate_config


CONTEXT = {"message": "Giá lấy cao răng tại phòng khám?", "history": [], "attachments": [],
           "capabilities": {"A2": True, "A5": True, "lookup_clinic_service": True}}
EXPECTED = {"decision": "route", "targets": ["lookup_clinic_service"], "service_topics": ["pricing"],
            "image_ids": [], "resolved_request": "Tra giá lấy cao răng tại phòng khám.", "message": None}


def test_duplicate_keys_nan_and_markdown_not_silently_repaired():
    for raw in ('{"decision":"route","decision":"handoff"}', '{"value":NaN}', '```json\n{}\n```'):
        with pytest.raises(ValueError):
            strict_json(raw)


def test_right_route_with_wrong_service_topic_not_counted_correct():
    prediction = deepcopy(EXPECTED)
    prediction["service_topics"] = ["opening_hours"]
    scored = score_response(CONTEXT, EXPECTED, json.dumps(prediction))
    assert scored["decision_correct"] and scored["context_valid"]
    assert not scored["control_exact"]


def test_unavailable_tool_invalidates_otherwise_valid_answer():
    context = deepcopy(CONTEXT)
    context["capabilities"]["lookup_clinic_service"] = False
    scored = score_response(context, EXPECTED, json.dumps(EXPECTED))
    assert scored["schema_valid"]
    assert not scored["context_valid"] and not scored["control_exact"]


def test_extra_fields_and_unknown_agent_are_not_successful_json():
    prediction = deepcopy(EXPECTED)
    prediction["confidence"] = 1
    scored = score_response(CONTEXT, EXPECTED, json.dumps(prediction))
    assert scored["raw_json_valid"] and not scored["schema_valid"] and not scored["decision_correct"]
    prediction = deepcopy(EXPECTED)
    prediction["targets"] = ["A3"]
    assert not score_response(CONTEXT, EXPECTED, json.dumps(prediction))["schema_valid"]


def test_rewording_is_not_penalized_but_metrics_include_invalid_rows():
    prediction = deepcopy(EXPECTED)
    prediction["resolved_request"] = "Tìm chi phí dịch vụ cạo vôi tại cơ sở."
    correct = score_response(CONTEXT, EXPECTED, json.dumps(prediction))
    bad = score_response(CONTEXT, EXPECTED, "không phải JSON")
    assert correct["control_exact"]
    rows = [{"expected": EXPECTED, "category": "service", "elapsed_seconds": 1, **item} for item in (correct, bad)]
    summary = summarize(rows)
    assert summary["count"] == 2
    assert summary["rates"]["control_exact"] == .5
    assert summary["per_decision"]["route"]["recall"] == .5
    assert selection_score(summary)[0] == .5


def test_pilot_cannot_enable_test_evaluation():
    with pytest.raises(ValueError, match="giữ test"):
        validate_config({"validation_split": "test", "evaluate_test_split": True})
