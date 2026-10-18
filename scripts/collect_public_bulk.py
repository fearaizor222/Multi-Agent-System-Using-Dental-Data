"""Thu thập MSD/PMC theo phạm vi đã chốt; chạy lại để tiếp tục từ cache."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.bulk_public import OUTPUT,collect_msd,collect_pmc


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",choices=["msd","pmc"])
    parser.add_argument("--target",type=int,default=200)
    args=parser.parse_args()
    try:
        result=collect_msd() if args.source=="msd" else collect_pmc(args.target)
        print(json.dumps({key:value for key,value in result.items() if key not in {"rejected","errors","catalog_urls","pending_urls"}},ensure_ascii=False,indent=2))
        return 0 if not result.get("stopped_reason") and (args.source=="msd" and not result["errors"] or result.get("complete")) else 1
    except Exception as exc:
        print(f"Chưa hoàn tất: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
