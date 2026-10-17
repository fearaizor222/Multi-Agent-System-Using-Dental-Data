import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from protego import Protego


BASE_URL = "https://www.msdmanuals.com"
SCOPE = "/vi/professional/rối-loạn-nha-khoa"
START_URL = BASE_URL + SCOPE
USER_AGENT = "C20DentalResearch/0.1"

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "msdmanual"


class CrawlStopped(RuntimeError):
    pass


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def normalize_url(url, base=START_URL):
    """Chuẩn hóa URL và chỉ giữ các trang nằm trong mục nha khoa."""
    full_url = urljoin(base, url)
    parsed = urlsplit(full_url)

    if parsed.scheme != "https":
        return None
    if parsed.netloc.lower() != "www.msdmanuals.com":
        return None

    path = unicodedata.normalize("NFC", unquote(parsed.path)).rstrip("/")
    if path != SCOPE and not path.startswith(SCOPE + "/"):
        return None
    if any(part in [".", ".."] for part in path.split("/")):
        return None

    return urlunsplit((
        "https",
        "www.msdmanuals.com",
        quote(path, safe="/"),
        "",
        ""
    ))


def is_article_url(url):
    url = normalize_url(url)
    if not url:
        return False

    parts = urlsplit(url).path.strip("/").split("/")
    return len(parts) >= 5


def get_article_urls(html):
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.select("a[href]"):
        url = normalize_url(a.get("href", ""))
        if url and is_article_url(url) and url not in urls:
            urls.append(url)

    return urls


def load_robots(session):
    robots_url = BASE_URL + "/robots.txt"
    response = session.get(robots_url, timeout=30, allow_redirects=False)
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


def parse_article(html, source_url, fetched_at=None):
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one("#topicHeaderTitle")
    body = soup.select_one('[data-testid="topic-main-content"]')

    if title_tag is None or body is None:
        raise ValueError("Không tìm thấy tiêu đề hoặc nội dung bài viết")
    if soup.html and soup.html.get("lang", "").split("-")[0] != "vi":
        raise ValueError("Trang trả về không phải tiếng Việt")
    canonical = soup.select_one('link[rel="canonical"]')
    source_url = normalize_url(canonical.get("href", source_url) if canonical else source_url)
    if not source_url or not is_article_url(source_url):
        raise ValueError("URL không phải bài nha khoa tiếng Việt")
    header = soup.select_one('[class*="TopicHead_topic__authors__container"]')
    header_text = clean_text(header.get_text(" ", strip=True)) if header else ""
    reviewed = re.search(r"(?:Đánh giá toàn diện|Đã xem xét|Đã đánh giá)\s*:\s*(Thg\s*\d{1,2}\s+\d{4})", header_text)
    updated = re.search(r"(?:Cập nhật lần cuối|Sửa đổi nội dung lần cuối cùng)\s*:\s*(Thg\s*\d{1,2}\s+\d{4})", header_text)
    msd_id = re.search(r"\bv\d+_vi\b", header_text)

    for tag in body.select(
        'script, style, button, nav, [aria-hidden="true"], '
        '[class*="TopicImage_"], '
        '[class*="TopicVideo_"], '
        '[class*="TopicSubNavigation_"]'
    ):
        tag.decompose()

    sections = []
    current_heading = "Giới thiệu"
    current_text = []
    current_anchor = None
    current_tables = []
    definition = soup.select_one('[data-testid="topicDefinition"]')
    if definition:
        current_text.append(clean_text(definition.get_text(" ", strip=True)))

    for tag in body.select("h2, h3, h4, p, li, table"):
        if tag is definition:
            continue
        if tag.find_parent(["li", "table"]) is not None:
            continue

        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        if tag.name in ["h2", "h3", "h4"]:
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_text),
                    "tables": current_tables,
                    "source_url": source_url + ("#" + quote(current_anchor, safe="") if current_anchor else "")
                })

            current_heading = text
            current_text = []
            current_anchor = tag.get("id")
            current_tables = []
        elif tag.name == "table":
            rows = []
            for row in tag.select("tr"):
                cells = [clean_text(cell.get_text(" ", strip=True))
                         for cell in row.find_all(["th", "td"], recursive=False)]
                if any(cells):
                    rows.append(cells)
            if rows:
                current_tables.append(rows)
                current_text.append("\n".join(" | ".join(row) for row in rows))
        else:
            current_text.append(text)

    if current_text:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_text),
            "tables": current_tables,
            "source_url": source_url + ("#" + quote(current_anchor, safe="") if current_anchor else "")
        })

    if not sections:
        raise ValueError("Không lấy được nội dung bài viết")

    content_parts = []
    for section in sections:
        content_parts.append(section["heading"] + "\n" + section["content"])

    content = "\n\n".join(content_parts)
    article_id = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:12]
    stamp = fetched_at or datetime.now(timezone.utc).isoformat()

    return {
        "id": "PUB-MSD-" + msd_id.group(0) if msd_id else "MSD-" + article_id,
        "article_id": msd_id.group(0) if msd_id else None,
        "source_type": "public",
        "source_name": "Cẩm nang MSD",
        "source_url": source_url,
        "title": clean_text(title_tag.get_text(" ", strip=True)),
        "language": "vi",
        "edition": "professional",
        "source_last_reviewed": clean_text(reviewed.group(1)) if reviewed else None,
        "source_last_updated": clean_text(updated.group(1)) if updated else None,
        "source_checked_on": stamp[:10],
        "fetched_at": stamp,
        "content": content,
        "sections": sections,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "clinical_approved": False
    }


def load_existing_data(path):
    data = {}

    if not path.exists():
        return data

    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            url = normalize_url(item.get("source_url", ""))
            if not url or not is_article_url(url):
                raise ValueError(f"{path.name}, dòng {number}: source_url không hợp lệ")
            data[url] = item

    return data


def save_data(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for item in data.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_urls_file(path):
    urls = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    return urls


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Scrape bài nha khoa tiếng Việt từ MSD Manual"
    )
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--urls-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    if args.max_pages <= 0:
        parser.error("--max-pages phải lớn hơn 0")
    if args.delay < 0:
        parser.error("--delay không được âm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "articles.jsonl"
    manifest_file = args.output_dir / "manifest.json"

    try:
        saved_data = load_existing_data(output_file)
    except (ValueError, OSError) as e:
        print("Không đọc được dữ liệu đã lưu:", e)
        return 1
    saved = 0
    skipped = 0
    errors = []
    urls = []
    attempted = 0
    stopped_reason = None

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "vi"
    })

    started_at = datetime.now(timezone.utc).isoformat()

    try:
        robots = load_robots(session)
        args.delay = max(args.delay, robots.crawl_delay(USER_AGENT) or 0)

        requested_urls = list(args.url)
        if args.urls_file:
            requested_urls.extend(read_urls_file(args.urls_file))

        if requested_urls:
            for item in requested_urls:
                url = normalize_url(item)
                if url and is_article_url(url) and url not in urls:
                    urls.append(url)
                elif not url or not is_article_url(url):
                    raise ValueError(f"URL không phải bài nha khoa tiếng Việt: {item}")
        else:
            html, _ = get_html(session, robots, START_URL, args.delay)
            urls = get_article_urls(html)
        if not urls:
            raise ValueError("Không tìm được URL bài viết")

        print("Tìm được", len(urls), "URL")

        for url in urls:
            if attempted >= args.max_pages:
                break

            if url in saved_data and not args.refresh:
                skipped += 1
                continue

            attempted += 1

            try:
                html, final_url = get_html(session, robots, url, args.delay)
                article = parse_article(html, final_url)

                canonical_url = article["source_url"]
                if canonical_url in saved_data and not args.refresh:
                    skipped += 1
                    continue
                if url != canonical_url:
                    saved_data.pop(url, None)
                saved_data[canonical_url] = article
                save_data(output_file, saved_data)

                saved += 1
                print(f"[{attempted}] Đã lưu: {article['title']}")

            except CrawlStopped:
                raise
            except (requests.RequestException, ValueError) as e:
                errors.append({"url": url, "error": str(e)})
                print("Bỏ qua:", unquote(url), "-", e)

    except (CrawlStopped, requests.RequestException, ValueError, OSError) as e:
        stopped_reason = str(e)
        errors.append({"url": START_URL, "error": str(e)})
        print("Dừng crawl:", e)
    except KeyboardInterrupt:
        stopped_reason = "Đã dừng bằng Ctrl+C"
        print(stopped_reason)

    finally:
        session.close()

    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": START_URL,
        "discovered": len(urls),
        "attempted": attempted,
        "saved_this_run": saved,
        "skipped": skipped,
        "total_saved": len(saved_data),
        "errors": errors,
        "stopped_reason": stopped_reason
    }

    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Đã lưu", saved, "bài. Tổng cộng:", len(saved_data))
    print("Dữ liệu nằm ở:", output_file)
    return 1 if errors or stopped_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
