"""Kiểm tra lại bản xuất A1 trên đĩa, không gọi model và không sửa file."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.prepare_a1_dataset import SOURCE, build_records
from src.data.a1_dataset import A1Input, A1Decision, SYSTEM_PROMPT, validate_records


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_dataset(directory, source_path=None):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source_path = Path(source_path) if source_path is not None else Path(manifest.get("source", SOURCE))
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Nguồn scenario thay đổi so với bản xuất; cần tạo phiên bản mới")
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Checksum không khớp: " + name)
    if (directory / "system_prompt.txt").read_text(encoding="utf-8") != SYSTEM_PROMPT:
        raise ValueError("Policy prompt thay đổi so với bản xuất")
    for filename, model in (("input.schema.json", A1Input), ("output.schema.json", A1Decision)):
        if json.loads((directory / filename).read_text(encoding="utf-8")) != model.model_json_schema():
            raise ValueError("Schema thay đổi so với bản xuất: " + filename)
    records = []
    for split in ("train", "validation", "test"):
        metadata = read_lines(directory / f"{split}.metadata.jsonl")
        training = read_lines(directory / f"{split}.jsonl")
        if training != [{"messages": row["messages"]} for row in metadata]:
            raise ValueError("File SFT và metadata không khớp: " + split)
        if any(row["split"] != split for row in metadata):
            raise ValueError("Metadata bị đặt sai split")
        records.extend(metadata)
    summary = validate_records(records)
    expected = build_records(json.loads(source_path.read_text(encoding="utf-8")), manifest["split_seed"])
    if {row["id"]: row for row in records} != {row["id"]: row for row in expected}:
        raise ValueError("Bản xuất không thể tái tạo từ source và seed đã lưu")
    summary.update(checksums="passed", sft_metadata_alignment="passed", reproducibility="passed",
                   human_reviewed_records=manifest["human_reviewed_records"],
                   clinical_accuracy_evaluated=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_dataset(args.dataset_dir, args.source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
