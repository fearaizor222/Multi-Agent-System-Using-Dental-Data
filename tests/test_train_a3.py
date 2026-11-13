import ast
import json

import pytest

from scripts.train_a3 import (DEFAULT_DATA, FIELDS, MAX_LENGTH, MODELS, ROOT, SelectedAssessment,
    configure_tokenizer, encode_sample, evaluate, generation_stop_ids, load_data,
    load_evaluation_cache, parse_sample, plot_results, resume_state, score_answer,
    summarize_results, validate_resume_name)


def sample():
    payload = {"question": "Câu hỏi", "claim": "Nhận định", "internal_evidence": [
        {"evidence_id": "I1", "content": "Bằng chứng nội bộ"}],
        "public_quote_options": [{"quote_id": "P1.Q1", "evidence_id": "P1", "text": "Nguồn công khai"}]}
    gold = {"draft_support": "supported", "public_support": "supported", "quote_ids": ["P1.Q1"], "reason": "Lý do"}
    return {"messages": [{"role": "system", "content": "Policy"},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}]}


def prediction(**changes):
    gold = parse_sample(sample())[1]
    return json.dumps({**gold, **changes}, ensure_ascii=False)


def test_source_split_and_labels_are_preserved_in_real_data():
    packet = load_data()
    for source, split in (("train", "train"), ("val", "validation")):
        path = DEFAULT_DATA / (source + ".jsonl")
        original = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [row["messages"] for row in packet[split]] == [row["messages"] for row in original]
        assert packet["info"]["counts"][split] == len(original)
    assert packet["info"]["validation_main_chunks_seen_in_train_including_distractors"] >= 0


def test_partial_or_stale_metadata_rejected(tmp_path):
    for split in ("train", "val"):
        record = sample()
        record["messages"][1]["content"] = record["messages"][1]["content"].replace("Câu hỏi", split)
        payload, gold = parse_sample(record)
        meta = {**gold, "question": payload["question"], "claim": payload["claim"],
                "public_document_id": split, "public_chunk_id": split}
        (tmp_path / (split + ".jsonl")).write_text(json.dumps(record), encoding="utf-8")
        (tmp_path / (split + ".metadata.jsonl")).write_text(json.dumps(meta), encoding="utf-8")
    assert load_data(tmp_path)["info"]["counts"] == {"train": 1, "validation": 1}
    (tmp_path / "val.metadata.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_data(tmp_path)


def test_gold_out_of_context_or_insufficient_with_quotes_rejected():
    for changes in ({"quote_ids": ["P9.Q9"]}, {"public_support": "insufficient"}):
        record = sample()
        record["messages"][2]["content"] = prediction(**changes)
        with pytest.raises(ValueError, match="trích dẫn"):
            parse_sample(record)
    draft_record = sample()
    payload = json.loads(draft_record["messages"][1]["content"])
    payload["draft_evidence"] = payload.pop("internal_evidence")
    draft_record["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    parsed_payload, _ = parse_sample(draft_record)
    assert parsed_payload["draft_evidence"][0]["evidence_id"] == "R1" or parsed_payload["draft_evidence"][0]["evidence_id"] == "I1"


def test_prediction_schema_and_context_are_separate_and_no_json_repair():
    record = sample()
    assert not score_answer(record, "```json\n" + prediction() + "\n```")["json_valid"]
    assert not score_answer(record, '{"draft_support":"supported","draft_support":"contradicted"}')["json_valid"]
    result = score_answer(record, prediction(quote_ids=["P9.Q9"]))
    assert result["schema_valid"] and result["both_labels_correct"]
    assert not result["quote_ids_valid"] and not result["assessment_exact_match"]
    result = score_answer(record, prediction(public_support="insufficient"))
    assert result["schema_valid"] and not result["quote_ids_valid"]


def test_explanation_wording_not_required_for_assessment_match():
    result = score_answer(sample(), prediction(reason="Một lời giải thích khác"))
    assert result["assessment_exact_match"] and not result["full_json_exact_match"]


def test_invalid_output_stays_in_both_matrices_and_wrong_labels_off_diagonal():
    record = sample()
    rows = [score_answer(record, prediction()), score_answer(record, "invalid"),
            score_answer(record, prediction(draft_support="contradicted"))]
    metrics = summarize_results(rows)
    draft, public = [metrics["decisions"][field] for field in FIELDS]
    assert draft["confusion_matrix"] == [[1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert public["confusion_matrix"] == [[2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert draft["per_label"]["supported"]["recall"] == pytest.approx(1 / 3)
    assert public["per_label"]["supported"]["recall"] == pytest.approx(2 / 3)
    assert metrics["count"] == 3


def test_masking_and_long_sample_rejected_without_truncation():
    class Tokens:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3] if len(messages) == 2 else [1, 2, 3, 4, 5]
    assert encode_sample(Tokens(), sample(), 5)["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match="không tự cắt"):
        encode_sample(Tokens(), sample(), 4)


@pytest.mark.parametrize("model_id", list(MODELS))
def test_all_real_samples_fit_and_only_assistant_is_supervised(model_id):
    from transformers import AutoTokenizer
    spec = MODELS[model_id]
    path = ROOT / "data/models" / spec["repo"].replace("/", "--") / spec["revision"]
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(path, local_files_only=True), model_id)
    packet = load_data()
    maximum = 0
    for split in ("train", "validation"):
        for record in packet[split]:
            encoded = encode_sample(tokenizer, record)
            masked = encoded["labels"].index(next(v for v in encoded["labels"] if v != -100))
            answer = tokenizer.decode(encoded["input_ids"][masked:], skip_special_tokens=True).strip()
            assert json.loads(answer) == json.loads(record["messages"][2]["content"])
            assert set(encoded["labels"][:masked]) == {-100}
            assert tokenizer.eos_token_id in encoded["labels"][masked:]
            maximum = max(maximum, len(encoded["input_ids"]))
    assert maximum <= MAX_LENGTH
    if model_id == "mediphi_38b":
        assert tokenizer.convert_tokens_to_ids("<|end|>") in generation_stop_ids(tokenizer, model_id)


def test_output_contract_matches_runtime_fields():
    # Inspect the real class without importing unrelated LangChain/RAG dependencies
    # into the intentionally separate training environment.
    from typing import Literal
    from pydantic import BaseModel, ConfigDict, Field, model_validator
    class StrictModel(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    tree = ast.parse((ROOT / "src/agents/verification_selection.py").read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SelectedAssessment")
    namespace = {"StrictModel": StrictModel, "Literal": Literal, "Field": Field, "model_validator": model_validator}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "runtime_contract", "exec"), namespace)
    RuntimeSchema = namespace["SelectedAssessment"]
    expected = RuntimeSchema.model_json_schema()
    actual = SelectedAssessment.model_json_schema()
    for key in ("properties", "required", "additionalProperties"):
        assert expected[key] == actual[key]


def test_three_launchers_have_separate_model_cache_and_run_names():
    for suffix, model_id in (("", "qwen3_17b"), ("_vinallama", "vinallama_27b"), ("_mediphi", "mediphi_38b")):
        source = (ROOT / "scripts" / f"modal_train_a3{suffix}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "train"]
        assert len(calls) == 1
        assert next(kw.value.value for kw in calls[0].keywords if kw.arg == "model_id") == model_id
        assert f"/vol/hf/{model_id}" in source and f"a3_{model_id}_" in source
        assert 'gpu="L4"' in source and 'retries=0' in source


def test_four_png_figures_include_both_decision_fields(tmp_path):
    record = sample()
    before = summarize_results([score_answer(record, "invalid")])
    after = summarize_results([score_answer(record, prediction())])
    plot_results([{"step": 1, "loss": 1.}, {"step": 2, "eval_loss": .8}],
                 before, after, tmp_path / "figures", "TEST FIXTURE")
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {
        "loss.png", "before_after.png", "04_confusion_matrix.png", "05_f1_by_decision.png"}


def cached_sample():
    record = sample()
    record["metadata"] = {"id": "row-1"}
    raw = prediction()
    return record, {"sample_id": "row-1", "question": "Câu hỏi", "raw_output": raw,
                    **score_answer(record, raw)}


def test_evaluation_resume_reuses_valid_prefix_and_drops_only_partial_tail(tmp_path):
    record, saved = cached_sample()
    path = tmp_path / "predictions.jsonl"
    saved["schema_valid"] = False  # Stored scores are recomputed.
    path.write_bytes(json.dumps(saved, ensure_ascii=False).encode() + b'\n{"raw_output":"\xe1\xbb')
    rows = load_evaluation_cache([record], path)
    assert len(rows) == 1 and rows[0]["schema_valid"]
    path.write_text('broken\n' + json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON hỏng"):
        load_evaluation_cache([record], path)


def test_wrong_sample_cache_is_rejected(tmp_path):
    record, saved = cached_sample()
    saved["sample_id"] = "wrong-id"
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="không khớp"):
        load_evaluation_cache([record], path)


def test_complete_cached_evaluation_never_generates_again(tmp_path):
    from types import SimpleNamespace
    record, saved = cached_sample()
    directory = tmp_path / "before"
    directory.mkdir()
    (directory / "predictions.jsonl").write_text(json.dumps(saved), encoding="utf-8")
    class Model:
        config = SimpleNamespace()
        def eval(self):
            pass
        def generate(self, **kwargs):
            raise AssertionError("Completed cached predictions must not be regenerated")
    metrics = evaluate(Model(), None, [record], directory, "qwen3_17b", resume=True)
    assert metrics["assessment_exact_match"] == 1
    assert (directory / "metrics.json").exists()


def test_resume_identity_and_selection_of_saved_adapter_or_complete_checkpoint(tmp_path):
    spec = MODELS["qwen3_17b"]
    packet = {"info": {"source_sha256": {"train.jsonl": "abc"}}}
    config = {"model": spec["repo"], "revision": spec["revision"], "epochs": 2, "max_length": MAX_LENGTH}
    for name, content in (("config.json", config), ("dataset_info.json", packet["info"]),
                          ("run_status.json", {"status": "failed"})):
        (tmp_path / name).write_text(json.dumps(content), encoding="utf-8")
    assert resume_state(tmp_path, packet, spec, 2)["checkpoint"] is None
    checkpoint = tmp_path / "checkpoints/checkpoint-132"
    checkpoint.mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"test fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step":132}', encoding="utf-8")
    assert resume_state(tmp_path, packet, spec, 2)["checkpoint"] == str(checkpoint)
    (tmp_path / "adapter").mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (tmp_path / "adapter" / name).write_bytes(b"test fixture")
    (tmp_path / "train_metrics.json").write_text('{}', encoding="utf-8")
    assert resume_state(tmp_path, packet, spec, 2)["adapter"]
    with pytest.raises(ValueError, match="Run cũ khác"):
        resume_state(tmp_path, packet, spec, 3)
    packet["info"]["source_sha256"]["train.jsonl"] = "changed"
    with pytest.raises(ValueError, match="Run cũ khác"):
        resume_state(tmp_path, packet, spec, 2)


def test_resume_run_name_cannot_escape_directory_or_mix_models():
    assert validate_resume_name("a3_qwen3_17b_20261108_123456", "qwen3_17b")
    for name in ("../other", "a3_qwen3_17b_../../other", "a3_mediphi_38b_123"):
        with pytest.raises(ValueError, match="resume-run"):
            validate_resume_name(name, "qwen3_17b")
