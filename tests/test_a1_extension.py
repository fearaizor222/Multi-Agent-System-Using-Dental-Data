"""Bảo vệ holdout và nhãn đối chứng khi bổ sung dữ liệu A1."""
from copy import deepcopy
import itertools
import json

import pytest

from scripts.build_a1_v2 import additions
from scripts.prepare_a1_dataset import SOURCE, build_records
from src.data.a1_dataset import validate_pair


def test_explicit_split_map_cannot_omit_or_invent_groups():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["split_assignments"] = {g["id"]: "train" for g in source["groups"]}
    source["split_assignments"].pop(next(iter(source["split_assignments"])))
    with pytest.raises(ValueError, match="split_assignments"):
        build_records(source)


def test_extension_preserves_every_original_record_and_split():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    old = build_records(source)
    new_source = deepcopy(source)
    new_source["groups"].extend(additions())
    new_source["split_assignments"] = {r["group_id"]: r["split"] for r in old}
    new_source["split_assignments"].update({g["id"]: "train" for g in additions()})
    new = build_records(new_source)
    new_by_id = {r["id"]: r for r in new}
    assert all(new_by_id[r["id"]] == r for r in old)
    assert all(r["split"] == "train" for r in new if r["id"].startswith("a1v2_"))
    frozen = json.loads((SOURCE.parent.parent / "a1_supervisor_v2/scenarios.json").read_text(encoding="utf-8"))
    assert [g for g in frozen["groups"] if g["id"].startswith("a1v2_")] == additions()


def test_capability_grid_has_positive_routes_despite_unrelated_disabled_agents():
    for group in [g for g in additions() if g["category"] == "capability"]:
        cases = group["cases"]
        assert {tuple(c["input"]["capabilities"].values()) for c in cases} == set(itertools.product((False, True), repeat=3))
        required = next(c["target"]["targets"] for c in cases if c["target"]["decision"] == "route")
        for c in cases:
            validate_pair(c["input"], c["target"])
            available = all(c["input"]["capabilities"][name] for name in required)
            assert c["target"]["decision"] == ("route" if available else "handoff")
        if len(required) < 3:
            assert any(c["target"]["decision"] == "route" and not all(c["input"]["capabilities"].values()) for c in cases)


def test_term_preservation_labels_do_not_rewrite_standalone_request():
    group = next(g for g in additions() if g["id"] == "a1v2_term_preservation")
    for case in group["cases"]:
        assert case["target"]["resolved_request"] == case["input"]["message"]
