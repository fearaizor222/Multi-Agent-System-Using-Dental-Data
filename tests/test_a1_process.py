"""Kiểm tra biên tiến trình, fingerprint và timeout; không nạp GPU."""
import json
import subprocess

import pytest

from src.core import a1_process
from src.core.a1_process import A1ProcessClient,A1ProcessError,PROTOCOL,input_sha256
from src.core.a1_process import ROOT
from src.data.a1_dataset import A1Input


@pytest.fixture
def client(tmp_path):
    python=tmp_path/"python.exe";python.touch()
    run=tmp_path/"run";run.mkdir()
    adapter=run/"adapter";adapter.mkdir()
    (run/"summary.json").write_text(json.dumps({"status":"completed","best_adapter_dir":str(adapter)}),encoding="utf-8")
    return A1ProcessClient(python,run,timeout_seconds=1),adapter


def test_request_is_stdin_data_and_reply_is_bound_to_input(monkeypatch,client):
    bridge,adapter=client
    captured={}
    class Process:
        returncode=0
        def __init__(self,command,**kwargs):
            captured.update(command=command,kwargs=kwargs)
        def communicate(self,data,timeout):
            captured["input"]=data
            context=json.loads(data)
            return json.dumps({"protocol":PROTOCOL,"status":"ok","input_sha256":input_sha256(context),
                "raw_output":"{}","identity":{"adapter_dir":str(adapter)}}),"diagnostics"
    monkeypatch.setattr(a1_process.subprocess,"Popen",Process)
    question='Câu hỏi; $(khong-chay) "quoted"'
    reply=bridge.plan({"message":question})
    assert json.loads(captured["input"])["message"]==question
    assert question not in captured["command"] and captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"]=="1"
    assert reply["raw_output"]=="{}"


@pytest.mark.parametrize("mode",["fingerprint","adapter","extra_stdout","failure"])
def test_invalid_or_wrong_worker_responses_fail_closed(monkeypatch,client,mode):
    bridge,adapter=client
    class Process:
        returncode=0
        def __init__(self,*args,**kwargs): pass
        def communicate(self,data,timeout):
            response={"protocol":PROTOCOL,"status":"ok","input_sha256":input_sha256(json.loads(data)),
                      "raw_output":"{}","identity":{"adapter_dir":str(adapter)}}
            if mode=="fingerprint": response["input_sha256"]="wrong"
            if mode=="adapter": response["identity"]["adapter_dir"]=str(adapter.parent/"different")
            if mode=="failure": response.update(status="error",error="GPU failure")
            return ("extra\n" if mode=="extra_stdout" else "")+json.dumps(response),""
    monkeypatch.setattr(a1_process.subprocess,"Popen",Process)
    with pytest.raises(A1ProcessError): bridge.plan({"message":"Câu thử"})


def test_timeout_terminates_owned_worker_tree(monkeypatch,client):
    bridge,_=client
    terminated=[]
    class Process:
        pid=123
        def __init__(self,*args,**kwargs): self.calls=0
        def communicate(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1: raise subprocess.TimeoutExpired("worker",1)
            return "",""
    monkeypatch.setattr(a1_process.subprocess,"Popen",Process)
    monkeypatch.setattr(a1_process,"terminate_worker",lambda p:terminated.append(p.pid))
    with pytest.raises(A1ProcessError,match="quá thời gian"): bridge.plan({"message":"Câu thử"})
    assert terminated==[123]


def test_worker_import_in_gpu_environment_does_not_require_rag_packages():
    python=ROOT.parent/".venv-a1/Scripts/python.exe"
    if not python.is_file():
        pytest.skip("Môi trường A1 local không có trên máy test")
    completed=subprocess.run([str(python),"-X","utf8","-c",
        "import sys; import scripts.a1_worker; assert 'src.core' not in sys.modules; assert 'torch' not in sys.modules"],
        cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert completed.returncode==0,completed.stderr
