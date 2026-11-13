"""Run: python -m modal run scripts/modal_train_a2_medscholar.py (starts one paid GPU job)."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("dental-a2-medscholar")
volume = modal.Volume.from_name("dental-a2-training", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==4.57.6", "peft==0.18.1", "accelerate==1.12.0",
                 "bitsandbytes==0.49.2", "pydantic==2.12.5", "numpy==2.2.6", "matplotlib==3.10.8")
    .env({"PYTHONPATH": "/app", "HF_HOME": "/vol/hf/medscholar_15b", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(ROOT / "scripts/train_a2.py", "/app/scripts/train_a2.py")
    .add_local_file(ROOT / "src/agents/contracts.py", "/app/src/agents/contracts.py"))


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=10800, retries=0,
              volumes={"/vol": volume})
def run_training(packet: dict, epochs: int):
    import sys
    from datetime import datetime, timedelta, timezone
    sys.path.insert(0, "/app")
    from scripts.train_a2 import train
    name = "a2_medscholar_15b_" + datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S_%f")
    output = train(packet, Path("/vol/runs") / name, epochs=epochs, commit=volume.commit, model_id="medscholar_15b")
    print("Results:", output, flush=True)
    return name


@app.local_entrypoint()
def main(dataset_dir: str = str(ROOT / "data/processed/a2_drafting_v1"), epochs: int = 2):
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.train_a2 import load_data
    if epochs < 1:
        raise ValueError("epochs phải dương")
    packet = load_data(Path(dataset_dir))  # Validate before submitting the GPU job.
    print(packet["info"])
    name = run_training.remote(packet, epochs)
    print("Download with:")
    print(f"python -m modal volume get dental-a2-training runs/{name} outputs/a2_training/{name}")
