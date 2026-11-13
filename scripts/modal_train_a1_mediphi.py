"""A1 MediPhi-Instruct QLoRA on Modal. Running this script submits one paid L4 job."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dental-a1-mediphi")
volume = modal.Volume.from_name("dental-a1-training", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.6", "peft==0.18.1", "accelerate==1.12.0",
        "bitsandbytes==0.49.2", "pydantic==2.12.5", "numpy==2.2.6", "matplotlib==3.10.8",
        "trl==0.24.0", "datasets==4.4.2", "huggingface-hub==0.36.2")
    .env({"PYTHONPATH": "/app", "HF_HOME": "/vol/hf/mediphi", "TOKENIZERS_PARALLELISM": "false"}))
for name in ("scripts/train_a1.py", "scripts/check_a1_environment.py", "scripts/check_a1_dataset.py",
             "scripts/modal_train_a1_mediphi.py",
             "scripts/prepare_a1_dataset.py", "scripts/report_a1_training.py", "scripts/prepare_a1_mediphi_modal.py",
             "src/data/a1_dataset.py", "src/training/a1_runtime.py", "src/training/a1_evaluation.py",
             "src/training/a1_strategy.py", "src/training/a1_model_support.py", "src/training/a1_plotting.py"):
    image = image.add_local_file(ROOT / name, "/app/" + name)


@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=14400, retries=0,
              volumes={"/vol": volume})
def run_training(payload: dict):
    import json
    import sys
    import shutil
    from datetime import datetime, timedelta, timezone
    sys.path.insert(0, "/app")
    from huggingface_hub import snapshot_download
    from scripts.prepare_a1_mediphi_modal import materialize
    from scripts.check_a1_environment import encode_assistant_sample
    from scripts.train_a1 import run
    from scripts.report_a1_training import report
    from src.training.a1_runtime import file_hash
    from src.training.a1_model_support import load_tokenizer
    from src.training.a1_strategy import assistant_weights
    from src.training.a1_plotting import render
    config = materialize(payload, Path("/app"))
    print("Download pinned MediPhi checkpoint...", flush=True)
    snapshot = Path(snapshot_download(config["base_model"], revision=config["revision"],
        allow_patterns=list(payload["model_manifest"]["sha256"])))
    # Same SHA-256 expectations as the previously verified local checkpoint.
    for name, expected in payload["model_manifest"]["sha256"].items():
        if file_hash(snapshot / name) != expected:
            raise ValueError("Downloaded checkpoint checksum mismatch: " + name)
    (snapshot / "download_manifest.json").write_text(json.dumps(payload["model_manifest"]), encoding="utf-8")
    volume.commit()
    config["model_dir"] = str(snapshot)
    tokenizer = load_tokenizer(config, Path("/app"))
    rows = [json.loads(l) for l in (Path("/app") / config["dataset_dir"] / "train.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        ids, labels, prefix = encode_assistant_sample(tokenizer, row["messages"])
        if len(ids) > config["max_length"]:
            raise ValueError("Train sample exceeds max_length: " + row["id"])
        assistant_weights(tokenizer, row["messages"], ids, prefix, config["token_weighting"])
    name = "a1_mediphi_38b_" + datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    output = Path("/vol/runs") / name
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(config, output, on_checkpoint=volume.commit)
        report(output)
        render(output, formats=("png",))
        shutil.copy2("/app/scripts/modal_train_a1_mediphi.py", output / "code_snapshot/modal_train_a1_mediphi.py")
        print("Completed:", output, flush=True)
        return name
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.prepare_a1_mediphi_modal import load_payload
    payload = load_payload()
    print("A1:", payload["dataset_check"]["by_split"], payload["config"]["base_model"])
    name = run_training.remote(payload)
    print(f"Download: python -m modal volume get dental-a1-training runs/{name} outputs/a1_training/{name}")
