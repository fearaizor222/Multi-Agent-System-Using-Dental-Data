"""Hợp đồng IPC dùng chung, không import gói RAG/config của môi trường cũ."""
import hashlib
import json

PROTOCOL="a1-worker-v1"


def input_sha256(context):
    data=json.dumps(context,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
