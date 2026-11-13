"""Export report-ready A1 figures from a completed run; no weights/inference."""
import argparse
import json
from pathlib import Path

from src.training.a1_plotting import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoothing-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir, args.output_dir, args.smoothing_window, args.dpi), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
