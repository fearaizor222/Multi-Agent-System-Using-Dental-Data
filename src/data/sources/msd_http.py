"""Tải HTML MSD trong phạm vi nha khoa, kiểm tra robots và redirect."""
import time
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from protego import Protego

BASE_URL = "https://www.msdmanuals.com"
SCOPE = "/vi/professional/rối-loạn-nha-khoa"
START_URL = BASE_URL + SCOPE
USER_AGENT = "C20DentalResearch/0.1"


class CrawlStopped(RuntimeError):
    pass


def normalize_url(url, base=START_URL):
    """Chuẩn hóa URL và chỉ giữ các trang nằm trong mục nha khoa."""
    parsed = urlsplit(urljoin(base, url))
    if parsed.scheme != "https" or parsed.netloc.lower() != "www.msdmanuals.com":
        return None
    path = unicodedata.normalize("NFC", unquote(parsed.path)).rstrip("/")
    if path != SCOPE and not path.startswith(SCOPE + "/"):
        return None
    if any(part in [".", ".."] for part in path.split("/")):
        return None
    return urlunsplit(("https", "www.msdmanuals.com", quote(path, safe="/"), "", ""))


def is_article_url(url):
    url = normalize_url(url)
    return bool(url and len(urlsplit(url).path.strip("/").split("/")) >= 5)


def load_robots(session):
    response = session.get(BASE_URL + "/robots.txt", timeout=30, allow_redirects=False)
    response.raise_for_status()
    if response.is_redirect:
        raise CrawlStopped("robots.txt bị redirect; cần kiểm tra lại địa chỉ")
    return Protego.parse(response.text)


def get_html(session, robots, url, delay):
    url = normalize_url(url)
    if not url:
        raise ValueError("URL nằm ngoài phạm vi crawl")
    for _ in range(6):
        if not robots.can_fetch(url, USER_AGENT):
            raise ValueError("robots.txt không cho phép crawl URL này")
        time.sleep(delay)
        response = session.get(url, timeout=30, allow_redirects=False)
        if response.status_code in [401, 403, 429]:
            raise CrawlStopped(f"HTTP {response.status_code}; dừng tải, thử lại sau")
        if response.is_redirect:
            location = response.headers.get("Location")
            url = normalize_url(location, url) if location else None
            if not url:
                raise ValueError("Trang bị redirect ra ngoài phạm vi nha khoa")
            continue
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            raise ValueError("Nội dung trả về không phải HTML")
        return response.text, url
    raise ValueError("Quá nhiều redirect")
