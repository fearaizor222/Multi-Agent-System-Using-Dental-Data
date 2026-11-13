"""Kiểm tra ca mới giữ ngoài train và phép dò thuật ngữ có giới hạn rõ ràng."""
import json
import re

from scripts.build_a1_challenge_v2 import challenge
from scripts.evaluate_a1_challenge import literal_terms_preserved
from src.data.a1_dataset import chat_messages, normalized
from src.training.a1_runtime import ROOT


def test_challenge_not_in_training_or_original_holdout():
    source = json.loads((ROOT / "data/curated/a1_supervisor_v2/scenarios.json").read_text(encoding="utf-8"))
    existing = [normalized(c["input"]["message"]) for g in source["groups"] for c in g["cases"]]
    for row in challenge():
        assert row["split"] == "development_challenge"
        assert row["messages"] == chat_messages(row["input"], row["target"])
        text = normalized(row["input"]["message"])
        tokens = set(re.findall(r"\w+",text))
        for previous in existing:
            assert text != previous
            old = set(re.findall(r"\w+",previous))
            if min(len(tokens),len(old)) >= 5:
                assert len(tokens & old)/len(tokens | old) < .85


def test_literal_check_does_not_count_wrong_terms_or_invalid_plan():
    row = {"context_valid":True,"predicted":{"decision":"route","resolved_request":"Tìm bài đọc về pericoronitis và cheilitis."}}
    assert literal_terms_preserved(row,["pericoronitis","cheilitis"])
    assert not literal_terms_preserved(row,["pulpitis"])
    assert not literal_terms_preserved(row,["peri"])
    assert literal_terms_preserved(row,[]) is None
    row["context_valid"]=False
    assert not literal_terms_preserved(row,["cheilitis"])


def test_frozen_challenge_matches_authoring_source():
    path = ROOT / "data/curated/a1_supervisor_v2/development_challenge.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["records"] == challenge()
    assert frozen["used_for_training"] is False
