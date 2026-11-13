"""Chạy pip với DNS dự phòng giới hạn trong tiến trình, không bỏ xác minh TLS."""
import contextlib
import json
import runpy
import socket
import sys
import urllib.parse
import urllib.request


@contextlib.contextmanager
def package_dns():
    original = socket.getaddrinfo
    cache = {}

    def resolve(host, port, *args, **kwargs):
        try:
            return original(host, port, *args, **kwargs)
        except socket.gaierror:
            allowed = isinstance(host, str) and (host == "pypi.org" or host.endswith((".pypi.org", ".pythonhosted.org", ".pytorch.org")))
            if not allowed:
                raise
            if host not in cache:
                request = urllib.request.Request("https://1.1.1.1/dns-query?" + urllib.parse.urlencode({"name": host, "type": "A"}),
                                                 headers={"Accept": "application/dns-json"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    payload = json.load(response)
                addresses = [item["data"] for item in payload.get("Answer", []) if item["type"] == 1]
                if not addresses:
                    raise socket.gaierror("DNS dự phòng không có bản ghi A: " + host)
                cache[host] = addresses[0]
            return original(cache[host], port, *args, **kwargs)

    socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original


if __name__ == "__main__":
    sys.argv = ["pip", *sys.argv[1:]]
    with package_dns():
        runpy.run_module("pip", run_name="__main__", alter_sys=True)
