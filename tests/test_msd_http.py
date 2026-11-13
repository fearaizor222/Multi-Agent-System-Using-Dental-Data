"""Kiểm tra phạm vi URL, robots và HTTP của bộ thu thập MSD."""
import pytest
import requests
from protego import Protego

from src.data.sources.msd_http import BASE_URL, SCOPE, CrawlStopped, get_html, normalize_url

URL = BASE_URL + SCOPE + "/chuong/bai-thu"
HTML = "<html lang=\"vi\"><body>Bài thử</body></html>"


class FakeSession:
    def __init__(self, status=200, headers=None):
        self.status, self.headers = status, headers or {"Content-Type": "text/html"}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = self.status
        response.headers.update(self.headers)
        response.url = url
        response._content = HTML.encode("utf-8")
        response.encoding = "utf-8"
        return response


def test_robots_wildcard_prevents_the_request():
    session = FakeSession()
    robots = Protego.parse("User-agent: *\nDisallow: */bai-thu\n")
    with pytest.raises(ValueError, match="robots.txt"):
        get_html(session, robots, URL, 0)
    assert session.calls == []


def test_redirect_outside_scope_is_not_followed():
    session = FakeSession(302, {"Location": "https://example.com/other"})
    robots = Protego.parse("User-agent: *\nAllow: /\n")
    with pytest.raises(ValueError, match="redirect"):
        get_html(session, robots, URL, 0)
    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.parametrize("status,headers", [(403, {}), (429, {"Retry-After": "120"})])
def test_denial_or_long_rate_limit_stops_crawl(status, headers):
    session = FakeSession(status, headers)
    robots = Protego.parse("User-agent: *\nAllow: /\n")
    with pytest.raises(CrawlStopped):
        get_html(session, robots, URL, 0)
    assert len(session.calls) == 1


def test_html_request_uses_delay_and_timeout(monkeypatch):
    pauses = []
    monkeypatch.setattr("src.data.sources.msd_http.time.sleep", pauses.append)
    session = FakeSession()
    robots = Protego.parse("User-agent: *\nAllow: /\n")
    html, final_url = get_html(session, robots, URL, 2)
    assert html == HTML and final_url == normalize_url(URL)
    assert pauses == [2]
    assert session.calls[0][1]["timeout"] == 30


def test_url_normalization_keeps_dental_scope_and_removes_tracking():
    assert normalize_url(URL + "#section") == normalize_url(URL + "?tracking=1")
    assert normalize_url(URL) == normalize_url(normalize_url(URL))
    assert normalize_url("https://example.com/a") is None
    assert normalize_url("http://www.msdmanuals.com" + SCOPE) is None
    assert normalize_url(BASE_URL + SCOPE + "/%2E%2E/tim-mach") is None
    assert normalize_url("/vi/home/a/b") is None
