"""Download clinical dental images from MSD Manuals and Europe PMC into data/images/ and link them into metadata."""

import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT_DIR / "data" / "images"
MSD_IMAGES_DIR = IMAGES_DIR / "msdmanual"
EPMC_IMAGES_DIR = IMAGES_DIR / "europe_pmc"

MSD_RAW_PATH = ROOT_DIR / "data" / "raw" / "msdmanual" / "articles.jsonl"
EPMC_RAW_PATH = ROOT_DIR / "data" / "raw" / "europe_pmc" / "europe_pmc_articles.json"
OPEN_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "open_dental_knowledge_sections.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DentalResearch/1.0"}


def sanitize_filename(name: str) -> str:
    """Sanitize filename to be safe on Windows filesystem."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return clean[:80]


def is_valid_image(data: bytes) -> bool:
    """Validate image magic bytes to ensure file is genuine image and not HTML error page."""
    if not data or len(data) < 500:
        return False
    # Reject HTML / XML blocks immediately
    if data.strip().startswith((b"<!doctype", b"<!DOCTYPE", b"<html", b"<HTML", b"<?xml")):
        return False
    # JPEG magic bytes: \xff\xd8\xff
    if data.startswith(b"\xff\xd8\xff"):
        return True
    # PNG magic bytes: \x89PNG
    if data.startswith(b"\x89PNG"):
        return True
    # GIF magic bytes: GIF87a or GIF89a
    if data.startswith((b"GIF87a", b"GIF89a")):
        return True
    # WEBP magic bytes: RIFF....WEBP
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return True
    return False


def download_file(url: str, dest_path: Path, delay: float = 0.3) -> bool:
    """Download image file if not already exists with valid image content."""
    if dest_path.exists():
        try:
            content = dest_path.read_bytes()
            if is_valid_image(content):
                return True
            # Invalid/corrupted file (e.g. HTML captcha) -> delete and re-fetch
            dest_path.unlink(missing_ok=True)
        except Exception:
            pass

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(delay)

    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        if res.status_code == 200 and is_valid_image(res.content):
            dest_path.write_bytes(res.content)
            return True
    except Exception:
        pass
    return False


def process_and_download_msd_images(max_images_per_article: int = 3) -> Dict[str, List[Dict]]:
    """Download figures from MSD Manuals articles and return mapping of art_id -> figures."""
    if not MSD_RAW_PATH.exists():
        return {}

    msd_figures_map = {}
    total_downloaded = 0

    print("\n--- [1] Bắt đầu tải hình ảnh lâm sàng nha khoa từ Cẩm nang MSD ---")

    with open(MSD_RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            art = json.loads(line)
            art_id = art.get("id", "")
            source_url = art.get("source_url", "")
            if not source_url:
                continue

            try:
                time.sleep(0.5)
                res = requests.get(source_url, headers=HEADERS, timeout=20)
                if res.status_code != 200:
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                body = soup.select_one('[data-testid="topic-main-content"]')
                if not body:
                    continue

                topic_imgs = body.select('[class*="TopicImage_"]')
                art_figures = []

                for idx, ti in enumerate(topic_imgs[:max_images_per_article], 1):
                    # Find candidate img
                    img_url = ""
                    for img in ti.select("img"):
                        src = img.get("src", "")
                        if src.startswith("http") and "media/professional/images" in src:
                            img_url = src
                            break
                        srcset = img.get("srcset", "")
                        if "media/professional/images" in srcset:
                            # take first url
                            parts = srcset.split(",")
                            for p in parts:
                                u = p.strip().split(" ")[0]
                                if u.startswith("http"):
                                    img_url = u
                                    break
                            if img_url:
                                break

                    if not img_url:
                        continue

                    caption_elem = ti.select_one('[class*="caption"], [class*="Caption"]')
                    caption = caption_elem.get_text(strip=True) if caption_elem else f"Hình minh họa {idx}"

                    # Filename
                    url_clean = img_url.split("?")[0]
                    base_name = sanitize_filename(url_clean.split("/")[-1])
                    if not base_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        base_name += ".jpg"

                    filename = f"{art_id}_{base_name}"
                    dest_file = MSD_IMAGES_DIR / filename
                    rel_path = f"data/images/msdmanual/{filename}"

                    downloaded = download_file(img_url, dest_file)
                    if downloaded:
                        total_downloaded += 1
                        print(f"  [+] Đã tải ảnh MSD: {filename} ({dest_file.stat().st_size // 1024} KB)")

                    art_figures.append({
                        "label": f"Hình {idx}",
                        "caption": caption,
                        "image_url": img_url,
                        "local_path": rel_path,
                        "downloaded": downloaded,
                        "filesize_bytes": dest_file.stat().st_size if dest_file.exists() else 0
                    })

                if art_figures:
                    msd_figures_map[art_id] = art_figures

            except Exception:
                continue

    print(f"Hoàn tất tải ảnh MSD: Đã lưu {total_downloaded} ảnh vào {MSD_IMAGES_DIR.name}/")
    return msd_figures_map


def process_and_download_epmc_images(max_images_per_article: int = 4) -> Dict[str, List[Dict]]:
    """Download figures for Europe PMC articles using official EPMC supplementaryFiles ZIP and fullTextXML."""
    if not EPMC_RAW_PATH.exists():
        return {}

    with open(EPMC_RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    epmc_figures_map = {}
    total_downloaded = 0
    session = requests.Session()

    print("\n--- [2] Bắt đầu tải hình ảnh nghiên cứu từ Europe PMC (API Supplementary Files ZIP) ---")

    for art in articles:
        pmcid = art.get("pmcid")
        if not pmcid:
            continue

        time.sleep(0.8)

        # 1. Lấy FullText XML để trích xuất danh sách figure, label, caption và tên file graphic
        xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        try:
            r = session.get(xml_url, headers=HEADERS, timeout=25)
            if r.status_code != 200:
                continue

            root = ET.fromstring(r.content)
            figs = root.findall(".//fig")
            if not figs:
                continue

            # Kiểm tra xem các ảnh của bài này đã có sẵn trên đĩa và hợp lệ chưa
            needed_figs = []
            for fig in figs[:max_images_per_article]:
                label = fig.findtext("label") or "Figure"
                caption = fig.findtext(".//caption/p") or fig.findtext(".//caption/title") or ""
                graphic = fig.find(".//{http://www.w3.org/1999/xlink}graphic") or fig.find(".//graphic")
                if graphic is None:
                    continue
                href = graphic.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    continue
                clean_name = sanitize_filename(href)
                if not clean_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    clean_name += ".jpg"
                filename = f"{pmcid}_{clean_name}"
                dest_file = EPMC_IMAGES_DIR / filename
                rel_path = f"data/images/europe_pmc/{filename}"

                is_ok = False
                if dest_file.exists():
                    data = dest_file.read_bytes()
                    if is_valid_image(data):
                        is_ok = True
                    else:
                        dest_file.unlink(missing_ok=True)

                needed_figs.append({
                    "label": label,
                    "caption": caption.strip(),
                    "href": href,
                    "clean_name": clean_name,
                    "filename": filename,
                    "dest_file": dest_file,
                    "rel_path": rel_path,
                    "downloaded": is_ok,
                })

            # Nếu tất cả đã có sẵn ảnh hợp lệ, không cần tải lại ZIP
            all_downloaded = all(nf["downloaded"] for nf in needed_figs)

            supp_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
            zip_obj = None
            namelist_lower = {}

            if not all_downloaded:
                try:
                    r_zip = session.get(supp_url, headers=HEADERS, timeout=60, stream=True)
                    if r_zip.status_code == 200 and "zip" in r_zip.headers.get("Content-Type", "").lower():
                        buf = io.BytesIO()
                        for chunk in r_zip.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                buf.write(chunk)
                        buf.seek(0)
                        zip_obj = zipfile.ZipFile(buf)
                        namelist_lower = {name.lower(): name for name in zip_obj.namelist()}
                except Exception as ze:
                    print(f"  [-] Lưu ý {pmcid}: Lỗi tải zip ({ze})")

            art_figures = []
            for nf in needed_figs:
                label = nf["label"]
                caption = nf["caption"]
                href = nf["href"]
                clean_name = nf["clean_name"]
                filename = nf["filename"]
                dest_file = nf["dest_file"]
                rel_path = nf["rel_path"]
                downloaded = nf["downloaded"]
                source_ref = supp_url

                if downloaded:
                    # Đã có sẵn trên đĩa
                    total_downloaded += 1
                elif zip_obj:
                    matched_in_zip = None
                    candidates = [
                        href,
                        f"{href}.jpg",
                        f"{href}.jpeg",
                        f"{href}.png",
                        f"{href}.gif",
                        clean_name,
                    ]
                    for c in candidates:
                        if c.lower() in namelist_lower:
                            matched_in_zip = namelist_lower[c.lower()]
                            break

                    if not matched_in_zip:
                        href_stem = href.split("/")[-1].split(".")[0].lower()
                        for n_low, n_orig in namelist_lower.items():
                            if href_stem in n_low and n_low.endswith((".jpg", ".jpeg", ".png", ".gif")):
                                matched_in_zip = n_orig
                                break

                    if matched_in_zip:
                        try:
                            img_data = zip_obj.read(matched_in_zip)
                            if is_valid_image(img_data):
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                dest_file.write_bytes(img_data)
                                downloaded = True
                                source_ref = f"{supp_url}#{matched_in_zip}"
                        except Exception:
                            pass

                # Dự phòng: Tải từ endpoint bin chính thức nếu ZIP không có
                if not downloaded:
                    candidate_urls = [
                        f"https://europepmc.org/articles/{pmcid}/bin/{href}.jpg",
                        f"https://europepmc.org/articles/{pmcid}/bin/{href}",
                        f"https://europepmc.org/articles/{pmcid}/bin/{clean_name}",
                    ]
                    for u in candidate_urls:
                        if download_file(u, dest_file):
                            downloaded = True
                            source_ref = u
                            break

                if downloaded:
                    if not nf["downloaded"]:
                        total_downloaded += 1
                        print(f"  [+] Đã lưu ảnh PMC hợp lệ: {filename} ({dest_file.stat().st_size // 1024} KB) - {label}")
                    else:
                        print(f"  [=] Ảnh PMC đã có sẵn: {filename} ({dest_file.stat().st_size // 1024} KB) - {label}")

                art_figures.append({
                    "label": label,
                    "caption": caption,
                    "image_url": source_ref,
                    "local_path": rel_path,
                    "downloaded": downloaded,
                    "filesize_bytes": dest_file.stat().st_size if (dest_file.exists() and downloaded) else 0
                })

            epmc_figures_map[pmcid] = art_figures

        except Exception as e:
            print(f"  [-] Lỗi xử lý {pmcid}: {e}")
            continue

    print(f"Hoàn tất tải ảnh Europe PMC: Đã lưu {total_downloaded} ảnh hợp lệ vào {EPMC_IMAGES_DIR.name}/")
    return epmc_figures_map


def update_processed_json_with_images(
    epmc_figures_map: Dict[str, List[Dict]],
    msd_figures_map: Dict[str, List[Dict]]
):
    """Link local image paths and URLs directly into open_dental_knowledge_sections.json."""
    if not OPEN_PROCESSED_PATH.exists():
        print("[-] Không tìm thấy file open_dental_knowledge_sections.json")
        return

    with open(OPEN_PROCESSED_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    updated_count = 0
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        pmcid = meta.get("pmcid")
        chunk_id = chunk.get("id", "")

        # Check PMC
        if pmcid and pmcid in epmc_figures_map:
            figures_info = epmc_figures_map[pmcid]
            if figures_info:
                meta["figures_data"] = figures_info
                updated_count += 1

        # Check MSD
        for art_id, msd_figs in msd_figures_map.items():
            if chunk_id.startswith(art_id) and msd_figs:
                meta["figures_data"] = msd_figs
                updated_count += 1
                break

    with open(OPEN_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Đã cập nhật thông tin ảnh cục bộ vào {updated_count} chunks trong {OPEN_PROCESSED_PATH.name}")


def main():
    print("=" * 60)
    print("Bắt đầu quy trình tải ảnh lâm sàng và cập nhật đường dẫn vào JSON...")
    print("=" * 60)

    MSD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    EPMC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download MSD clinical images
    msd_map = process_and_download_msd_images(max_images_per_article=3)

    # 2. Download PMC scientific images
    epmc_map = process_and_download_epmc_images(max_images_per_article=3)

    # 3. Update processed JSON
    update_processed_json_with_images(epmc_map, msd_map)

    print("\n" + "=" * 60)
    print(f"Thư mục lưu ảnh: {IMAGES_DIR}")
    print("Hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    main()
