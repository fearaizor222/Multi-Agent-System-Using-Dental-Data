"""Model local dùng cho bước rerank và thử luồng tạo nháp/kiểm chứng."""
import json
from pathlib import Path

from src.rag.local_embeddings import file_sha256

MODEL_ROOT = Path(__file__).resolve().parents[2] / "data/models"
MODEL_SPECS = {
    "reranker": {"repo": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"},
    "draft_llm": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "entailment": {"repo": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"},
}


def model_directory(name):
    spec = MODEL_SPECS[name]
    return MODEL_ROOT / spec["repo"].replace("/", "--") / spec["revision"]


def checked_model(name):
    directory = model_directory(name)
    manifest_path = directory / "download_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Chưa có model {name}; chạy python -m scripts.prepare_models --model {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in MODEL_SPECS[name].items()):
        raise ValueError("Manifest model không khớp revision đã chốt")
    for filename, checksum in manifest["sha256"].items():
        if file_sha256(directory / filename) != checksum:
            raise ValueError("File model bị thay đổi: " + filename)
    return directory, {**MODEL_SPECS[name], "sha256": manifest["sha256"]}
