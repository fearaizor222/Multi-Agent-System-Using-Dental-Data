"""Chuẩn hóa PDF Bộ Y tế; --pages dùng số trang PDF, không phải trang in."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.public_schema import write_documents
from src.data.sources.byt import parse_byt

ROOT = Path(__file__).resolve().parents[1]


def parse_pages(value):
    pages = []
    for item in value.split(","):
        bounds = item.strip().split("-")
        if len(bounds) == 1:
            pages.append(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[1]) >= int(bounds[0]):
            pages.extend(range(int(bounds[0]), int(bounds[1])+1))
        else:
            raise argparse.ArgumentTypeError("Trang có dạng 40-44 hoặc 40,42,43")
    return sorted(set(pages))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", type=parse_pages)
    parser.add_argument("--output", type=Path, default=ROOT/"data/processed/public_v1/corpus/byt_documents.jsonl")
    args = parser.parse_args()
    try:
        doc = parse_byt(args.pdf, ROOT/"data/raw/public_v1/byt", args.pages)
        write_documents(args.output, [doc])
        print(f"Đã lưu {len(doc.sections)} mục\n{args.output}")
        return 0
    except Exception as exc:
        print(f"Không xử lý được PDF: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
