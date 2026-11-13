"""Evaluate completed A3 v3 adapter on the untouched 167-example held-out v2 test set."""
import hashlib
import json
from pathlib import Path
import modal
from scripts.modal_train_a3 import image as training_image, volume, ROOT
from scripts.modal_test_a3 import plot_test, read_test

DEFAULT_RUN = "a3_qwen3_17b_20261113_160558_423926_7236d85d"
app = modal.App("dental-a3-v3-heldout-test")
image = (
    training_image
    .add_local_file(ROOT / "scripts/modal_test_a3.py", "/app/scripts/modal_test_a3.py")
    .add_local_file(ROOT / "scripts/modal_test_a3_v3.py", "/app/scripts/modal_test_a3_v3.py")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24576,
    timeout=7200,
    retries=0,
    volumes={"/vol": volume},
)
def run_test(model_id: str, run_name: str, packet: dict):
    from datetime import datetime, timedelta, timezone
    import os
    import time
    os.environ["HF_HOME"] = "/vol/hf/" + model_id
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from scripts.train_a3 import MODELS, configure_tokenizer, encode_sample, evaluate, validate_resume_name

    run_name = validate_resume_name(run_name, model_id)
    source = Path("/vol/runs") / run_name
    read = lambda name: json.loads((source / name).read_text(encoding="utf-8"))
    config, info, status = read("config.json"), read("dataset_info.json"), read("run_status.json")
    spec = MODELS[model_id]
    if status["status"] != "completed" or info.get("dataset_version") != "a3_verification_v3":
        raise ValueError("Require completed v3 run")
    if (config["model"], config["revision"]) != (spec["repo"], spec["revision"]):
        raise ValueError("Base model mismatch")

    weights = source / "adapter/adapter_model.safetensors"
    adapter_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    output = Path("/vol/tests") / (
        run_name + "_" + datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    save("run_status.json", {"status": "running"})
    save(
        "identity.json",
        {
            "model_id": model_id,
            "source_run": run_name,
            "source_dataset": info.get("dataset_version"),
            "test_sha256": packet["sha256"],
            "adapter_sha256": adapter_sha,
            "training_executed": False,
            "selection": "existing lowest-validation-loss v3 adapter",
        },
    )
    started = time.monotonic()
    try:
        set_seed(42)
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"]), model_id)
        for record in packet["records"]:
            encode_sample(tokenizer, record)
        with (output / "test.jsonl").open("w", encoding="utf-8") as stream:
            for record in packet["records"]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        volume.commit()
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo"],
            revision=spec["revision"],
            trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(model, str(source / "adapter"), is_trainable=False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Evaluation must have all parameters frozen")
        metrics = evaluate(model, tokenizer, packet["records"], output / "test", model_id)
        save(
            "metrics.json",
            {
                "test": metrics,
                "elapsed_seconds": time.monotonic() - started,
                "scope": "Held-out v2 reference-label agreement (n=167), not expert clinical validation. No training or checkpoint selection.",
            },
        )
        plot_test(metrics, output / "figures", spec["name"] + " (v3)")
        save("run_status.json", {"status": "completed"})
        print("Test completed:", output.name, flush=True)
        return output.name
    except BaseException as exc:
        save("run_status.json", {"status": "failed", "error": str(exc)[:1000]})
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main(model_id: str = "qwen3_17b", run_name: str = DEFAULT_RUN):
    from scripts.train_a3 import MODELS
    if model_id not in MODELS:
        raise ValueError("Unknown model: " + model_id)
    packet = read_test(ROOT / "data/processed/a3_verification_v2")
    print(model_id, run_name, len(packet["records"]), packet["sha256"], flush=True)
    test_name = run_test.remote(model_id, run_name, packet)
    print("Completed test run:", test_name, flush=True)
