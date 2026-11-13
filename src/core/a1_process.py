"""Cầu nối JSON tới A1 GPU trong venv riêng; không dùng shell để gọi model."""
import json
import os
from pathlib import Path
import subprocess
import time

from src.data.a1_dataset import A1Input
from src.training.a1_evaluation import strict_json
from src.training.a1_protocol import PROTOCOL,input_sha256

ROOT=Path(__file__).resolve().parents[2]
class A1ProcessError(RuntimeError):
    def __init__(self,message,diagnostics=""):
        super().__init__(message)
        self.diagnostics=diagnostics


def terminate_worker(process):
    # Windows venv python là launcher: kết thúc cả cây con của chính worker.
    if os.name=="nt":
        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW,check=False)
    if process.poll() is None:
        process.kill()


class A1ProcessClient:
    def __init__(self, python, run_dir, timeout_seconds=180, max_input_tokens=1536):
        self.python=Path(python).resolve()
        self.run_dir=Path(run_dir).resolve()
        if not self.python.is_file() or timeout_seconds<=0 or max_input_tokens<256:
            raise ValueError("Cấu hình worker A1 không hợp lệ")
        summary=json.loads((self.run_dir/"summary.json").read_text(encoding="utf-8"))
        if summary["status"]!="completed":
            raise ValueError("Cần một lượt A1 đã hoàn tất")
        self.adapter_dir=Path(summary["best_adapter_dir"]).resolve()
        self.timeout_seconds=timeout_seconds
        self.max_input_tokens=max_input_tokens

    def plan(self, context):
        context=A1Input.model_validate(context).model_dump()
        raw=json.dumps(context,ensure_ascii=False)
        if len(raw)>24000:
            raise A1ProcessError("A1_INPUT_TOO_LONG: input vượt giới hạn ký tự")
        command=[str(self.python),"-X","utf8","-m","scripts.a1_worker","--run-dir",str(self.run_dir),
                 "--max-input-tokens",str(self.max_input_tokens)]
        env={**os.environ,"PYTHONUTF8":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
        started=time.perf_counter()
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,encoding="utf-8",env=env,shell=False,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            stdout,stderr=process.communicate(raw,timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_worker(process)
            process.communicate()
            raise A1ProcessError("Worker A1 quá thời gian; không gọi A2")
        except KeyboardInterrupt:
            terminate_worker(process)
            process.communicate()
            raise
        try:
            if len(stdout)>64000:
                raise ValueError("Worker output quá dài")
            response=strict_json(stdout)
            if (not isinstance(response,dict) or response.get("protocol")!=PROTOCOL
                    or response.get("input_sha256")!=input_sha256(context)):
                raise ValueError("Protocol hoặc input fingerprint không khớp")
            if process.returncode!=0 or response.get("status")!="ok":
                raise ValueError(response.get("error","Worker thất bại"))
            if Path(response["identity"]["adapter_dir"]).resolve()!=self.adapter_dir:
                raise ValueError("Worker dùng adapter khác cấu hình")
            if not isinstance(response.get("raw_output"),str) or len(response["raw_output"])>16000:
                raise ValueError("Thiếu raw output hoặc output quá dài")
        except (ValueError,TypeError,KeyError) as exc:
            raise A1ProcessError(str(exc),stderr[-8000:]) from exc
        return {"raw_output":response["raw_output"],"identity":response["identity"],
                "milliseconds":(time.perf_counter()-started)*1000,"diagnostics":stderr[-8000:]}
