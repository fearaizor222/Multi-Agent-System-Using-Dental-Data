"""Một request A1: stdin JSON, stdout JSON; tiến trình kết thúc giải phóng GPU."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from src.training.a1_protocol import PROTOCOL,input_sha256
from src.data.a1_dataset import A1Input


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--max-input-tokens",type=int,default=1536)
    args=parser.parse_args()
    fingerprint=None
    try:
        raw=sys.stdin.read(24001)
        if len(raw)>24000:
            raise ValueError("A1_INPUT_TOO_LONG: input vượt giới hạn ký tự")
        context=A1Input.model_validate_json(raw).model_dump()
        fingerprint=input_sha256(context)
        with redirect_stdout(sys.stderr):
            from scripts.predict_a1 import predict
            prediction=predict(args.run_dir,context,max_input_tokens=args.max_input_tokens)
        adapter=Path(prediction["adapter_dir"])
        manifest=json.loads((adapter.parent/"adapter_manifest.json").read_text(encoding="utf-8"))
        response={"protocol":PROTOCOL,"input_sha256":fingerprint,"status":"ok",
                  "raw_output":prediction["raw_output"],"identity":{"adapter_dir":str(adapter.resolve()),
                  "base_model":manifest["base_model"],"revision":manifest["revision"],
                  "adapter_sha256":manifest["files_sha256"]["adapter_model.safetensors"]}}
        code=0
    except Exception as exc:
        response={"protocol":PROTOCOL,"input_sha256":fingerprint,"status":"error",
                  "error":type(exc).__name__+": "+str(exc)}
        code=1
    print(json.dumps(response,ensure_ascii=False),flush=True)
    return code


if __name__=="__main__":
    raise SystemExit(main())
