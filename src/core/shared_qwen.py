"""One persistent GPU worker: A1/A2 adapters, unmodified Qwen3 for A3/A4."""
import json
import atexit
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import threading
import time
from types import SimpleNamespace

from src.core.a1_process import ROOT, terminate_worker
from src.data.a1_dataset import A1Input, SYSTEM_PROMPT


class SharedQwenClient:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.full_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.config = self.full_config["text_models"]
        self.lock = threading.RLock()
        self.process = None
        self.responses = None
        self.identity = None
        self.sequence = 0
        self.log = None
        self.last_call = None
        atexit.register(self.close)

    def _read(self):
        try:
            line = self.responses.get(timeout=self.config.get("timeout_seconds", 300))
        except Empty:
            self.close()
            raise RuntimeError("Qwen3 quá thời gian; worker đã dừng. Có thể thử lại yêu cầu.")
        if not line:
            self.close()
            raise RuntimeError("Worker Qwen3 đã dừng; xem outputs/web_runtime/qwen_worker.log")
        value = json.loads(line)
        if value.get("status") != "ok":
            raise RuntimeError(value.get("error", "Worker Qwen3 không sẵn sàng"))
        return value

    def ensure_started(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.identity
            self.close()
            directory = ROOT / "outputs/web_runtime"
            directory.mkdir(parents=True, exist_ok=True)
            self.log = (directory / "qwen_worker.log").open("a", encoding="utf-8")
            self.responses = Queue()
            python = (ROOT / self.config["python"]).resolve()
            module = self.config.get("worker_module", "scripts.qwen_worker")
            self.process = subprocess.Popen([str(python), "-X", "utf8", "-u", "-m", module,
                "--config", str(self.config_path)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding="utf-8", shell=False,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            output, queue = self.process.stdout, self.responses
            def read_lines():
                try:
                    for line in output:
                        queue.put(line)
                finally:
                    queue.put(None)
            threading.Thread(target=read_lines, daemon=True).start()
            try:
                response = self._read()
                if response.get("event") != "ready":
                    raise RuntimeError("Worker Qwen3 khởi tạo không đúng giao thức")
                self.identity = response["identity"]
            except Exception:
                self.close()
                raise
            return self.identity

    def generate(self, role, messages=None, **extra):
        if role not in ("A1", "A2", "A3", "A4", "A5"):
            raise ValueError("Vai trò Qwen3 không hợp lệ")
        with self.lock:
            self.ensure_started()
            self.sequence += 1
            request = {**extra, "id": self.sequence, "role": role, "messages": messages}
            started = time.perf_counter()
            try:
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._read()
                if response.get("id") != self.sequence or response.get("role") != role:
                    raise RuntimeError("Worker trả kết quả khác yêu cầu")
                expected=self.identity["agents"][role]
                if (response["identity"].get("variant")!=expected["variant"]
                        or response["identity"].get("sha256")!=expected.get("sha256")):
                    raise RuntimeError("Worker dùng sai model hoặc adapter cho tác tử")
            except (OSError, ValueError):
                self.close()
                raise
            self.last_call = {**response, "milliseconds": (time.perf_counter() - started) * 1000}
            return self.last_call

    def plan(self, context):
        context = A1Input.model_validate(context).model_dump()
        return self.generate("A1", [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}])

    def close(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    terminate_worker(self.process)
                for stream in (self.process.stdin, self.process.stdout):
                    if stream:
                        stream.close()
                self.process = None
            if self.log is not None:
                self.log.close()
                self.log = None


class AgentChatModel:
    """Minimal invoke interface used by the existing structured-output agents."""
    def __init__(self, client, role):
        self.client, self.role = client, role

    @property
    def identity(self):
        spec = self.client.config
        if self.client.identity is not None:
            return {"repo": spec["base_model"], "revision": spec["revision"],
                    "role": self.role, **self.client.identity["agents"][self.role]}
        if self.role == "A4" and "synthesis" in self.client.full_config:
            spec = self.client.full_config["synthesis"]
            return {"role": self.role, "repo": spec["base_model"], "revision": spec["revision"],
                    "variant": "finetuned", "adapter": spec["adapter_dir"], "quantization": "NF4"}
        return {"repo": spec["base_model"], "revision": spec["revision"], "role": self.role,
                "variant": "finetuned" if self.role in ("A1", "A2") or spec.get(self.role.lower()+"_run_dir") else "base",
                "adapter_run": spec.get(self.role.lower() + "_run_dir"), "quantization": "NF4"}

    @property
    def _llm_type(self):
        return "shared-qwen3-" + self.role.lower()

    def invoke(self, messages):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        chat = [{"role": roles[m.type], "content": m.content} for m in messages]
        result = self.client.generate(self.role, chat)
        return SimpleNamespace(content=result["raw_output"])
