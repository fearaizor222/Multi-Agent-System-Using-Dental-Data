"""A separate official chat_template.jinja must be downloaded and verified."""
import json
from types import SimpleNamespace

from scripts import prepare_text_candidates as downloader
from src.training.a1_runtime import file_hash, write_json


def setup_download(monkeypatch, tmp_path, *, existing):
    spec = {"id": "candidate", "repo": "publisher/candidate", "revision": "a" * 40}
    catalog = tmp_path / "catalog.json"
    write_json(catalog, {"models": [spec]})
    directory = tmp_path / "model"
    directory.mkdir()
    files = {"config.json": b'{}', "model.safetensors": b'verified weights',
             "chat_template.jinja": b'{{ messages }}'}
    if existing:
        for name in ("config.json", "model.safetensors"):
            (directory / name).write_bytes(files[name])
        write_json(directory / "download_manifest.json", {"repo": spec["repo"], "revision": spec["revision"],
            "sha256": {name: file_hash(directory / name) for name in ("config.json", "model.safetensors")}})
    info = SimpleNamespace(sha=spec["revision"], gated=False,
        siblings=[SimpleNamespace(rfilename=name, lfs=None) for name in files])
    monkeypatch.setattr(downloader, "HfApi", lambda **kwargs: SimpleNamespace(model_info=lambda *args, **kw: info))
    monkeypatch.setattr(downloader, "candidate_directory", lambda _: directory)
    monkeypatch.setattr(downloader, "verify_candidate", lambda s: __import__(
        "src.training.model_candidates", fromlist=["verify_candidate"]).verify_candidate(s, directory))
    calls = []

    def download(repo, **kwargs):
        calls.append(kwargs)
        for name, content in files.items():
            if any(downloader.fnmatch.fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]):
                (directory / name).write_bytes(content)
        return str(directory)

    monkeypatch.setattr(downloader, "snapshot_download", download)
    output = tmp_path / "inventory.json"
    arguments = ["prepare", "--catalog", str(catalog), "--inventory", str(output)]
    if existing:
        arguments.append("--refresh-assets")
    monkeypatch.setattr("sys.argv", arguments)
    return directory, output, calls


def test_new_download_includes_separate_chat_template(monkeypatch, tmp_path):
    directory, output, calls = setup_download(monkeypatch, tmp_path, existing=False)
    downloader.main()
    assert "*.jinja" in calls[0]["allow_patterns"]
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"]["chat_template.jinja"] == file_hash(directory / "chat_template.jinja")
    assert json.loads(output.read_text(encoding="utf-8"))["models"][0]["action"] == "downloaded_verified"


def test_refresh_adds_template_without_redownloading_weights(monkeypatch, tmp_path):
    directory, output, calls = setup_download(monkeypatch, tmp_path, existing=True)
    original = file_hash(directory / "model.safetensors")
    downloader.main()
    assert calls[0]["allow_patterns"] == ["chat_template.jinja"]
    assert file_hash(directory / "model.safetensors") == original
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"]["chat_template.jinja"] == file_hash(directory / "chat_template.jinja")
    assert json.loads(output.read_text(encoding="utf-8"))["models"][0]["action"] == "verified_existing_added_assets"
