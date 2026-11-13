"""A3 vinallama_27b QLoRA on Modal; modal run submits one L4 training job."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dental-a3-vinallama")
volume = modal.Volume.from_name("dental-a3-training", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.6", "peft==0.18.1", "accelerate==1.12.0",
                 "bitsandbytes==0.49.2", "pydantic==2.12.5", "numpy==2.2.6", "matplotlib==3.10.8")
    .env({"PYTHONPATH": "/app", "HF_HOME": "/vol/hf/vinallama_27b", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(ROOT / "scripts/train_a3.py", "/app/scripts/train_a3.py")
    .add_local_file(ROOT / "scripts/modal_train_a3_vinallama.py", "/app/scripts/modal_train_a3_vinallama.py"))


@app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=14400, retries=0,
              volumes={"/vol": volume})
def run_training(packet: dict, epochs: int, resume_run: str = ""):
    import sys
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4
    sys.path.insert(0, "/app")
    from scripts.train_a3 import train, validate_resume_name
    name = validate_resume_name(resume_run, "vinallama_27b") or ("a3_vinallama_27b_" + datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8])
    output = train(packet, Path("/vol/runs") / name, epochs=epochs,
                   commit=volume.commit, model_id="vinallama_27b", resume=bool(resume_run))
    print("Completed:", output, flush=True)
    return name


@app.local_entrypoint()
def main(dataset_dir: str = str(ROOT / "data/processed/a3_verification_v3"), epochs: int = 2, resume_run: str = ""):
    import json
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.train_a3 import load_data, validate_resume_name
    if epochs < 1:
        raise ValueError("epochs phải dương")
    validate_resume_name(resume_run, "vinallama_27b")
    packet = load_data(Path(dataset_dir))
    print(json.dumps(packet["info"], ensure_ascii=False, indent=2))
    name = run_training.remote(packet, epochs, resume_run)
    print("Download results:")
    print(f"python -m modal volume get dental-a3-training runs/{name} outputs/a3_training/{name}")

