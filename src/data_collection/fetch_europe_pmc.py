"""Fetch open access dental scientific articles and full-text XML from Europe PMC REST API."""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "europe_pmc"

BASE_EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BASE_EPMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Curated high-impact clinical topics in Dentistry
DENTAL_TOPICS = [
    {
        "category": "Bệnh học Sâu răng & Tủy răng (Cariology & Endodontics)",
        "query": '(TITLE:"dental caries" OR TITLE:"pulpitis" OR TITLE:"endodontic") AND HAS_ABSTRACT:y AND OPEN_ACCESS:y',
    },
    {
        "category": "Bệnh lý Nha chu & Lợi (Periodontology)",
        "query": '(TITLE:"periodontitis" OR TITLE:"gingivitis" OR TITLE:"periodontal disease") AND HAS_ABSTRACT:y AND OPEN_ACCESS:y',
    },
    {
        "category": "Khớp thái dương hàm & Cắn khớp (TMJ & Occlusion)",
        "query": '(TITLE:"temporomandibular" OR TITLE:"malocclusion" OR TITLE:"occlusion" OR TITLE:"bruxism") AND HAS_ABSTRACT:y AND OPEN_ACCESS:y',
    },
    {
        "category": "Cấy ghép Nha khoa & Phục hình (Implantology & Prosthodontics)",
        "query": '(TITLE:"dental implant" OR TITLE:"prosthodontics" OR TITLE:"dental restoration") AND HAS_ABSTRACT:y AND OPEN_ACCESS:y',
    },
    {
        "category": "Phẫu thuật miệng & Nhổ răng (Oral Surgery & Third Molar)",
        "query": '(TITLE:"tooth extraction" OR TITLE:"third molar" OR TITLE:"wisdom tooth") AND HAS_ABSTRACT:y AND OPEN_ACCESS:y',
    },
]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    return " ".join(text.split())


def clean_html_abstract(text: str) -> str:
    """Strip HTML tags and normalize whitespace in abstract."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(" ", strip=True)
    return clean_text(cleaned)


def fetch_fulltext_sections(pmcid: str, delay: float = 0.4) -> List[Dict]:
    """Fetch structured full-text XML from Europe PMC and extract clinical sections with figures."""
    if not pmcid:
        return []

    url = BASE_EPMC_FULLTEXT_URL.format(pmcid=pmcid)
    time.sleep(delay)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        body = root.find(".//body")
        if body is None:
            return []

        sections = []
        skip_headings = [
            "author contribution", "acknowledgement", "conflict of interest",
            "competing interest", "funding", "reference", "consent for publication"
        ]

        for sec in body.findall(".//sec"):
            title = clean_text(sec.findtext("title") or "")
            title_lower = title.lower()

            if any(skip in title_lower for skip in skip_headings):
                continue

            # Extract paragraphs
            paragraphs = ["".join(p.itertext()).strip() for p in sec.findall(".//p")]
            text_body = "\n\n".join([p for p in paragraphs if p])

            # Extract figures and their captions
            figures = []
            for fig in sec.findall(".//fig"):
                fig_label = clean_text(fig.findtext("label") or "Hình")
                caption_p = fig.findtext(".//caption/p") or fig.findtext(".//caption/title") or ""
                fig_caption = clean_text(caption_p)
                fig_desc = f"[{fig_label}: {fig_caption}]" if fig_caption else f"[{fig_label}]"
                figures.append(fig_desc)

            # Extract tables and their captions
            tables = []
            for tbl in sec.findall(".//table-wrap"):
                tbl_label = clean_text(tbl.findtext("label") or "Bảng")
                caption_p = tbl.findtext(".//caption/p") or tbl.findtext(".//caption/title") or ""
                tbl_caption = clean_text(caption_p)
                tbl_desc = f"[{tbl_label}: {tbl_caption}]" if tbl_caption else f"[{tbl_label}]"
                tables.append(tbl_desc)

            # Combine figure & table captions into the section text so embeddings can find them
            extra_elements = []
            if figures:
                extra_elements.append("Hình minh họa:\n" + "\n".join(figures))
            if tables:
                extra_elements.append("Bảng biểu:\n" + "\n".join(tables))

            if extra_elements:
                text_body = text_body + "\n\n" + "\n\n".join(extra_elements)

            text_body = clean_text(text_body)
            if text_body and len(text_body) > 100:
                sections.append({
                    "heading": title if title else "Nội dung lâm sàng",
                    "content": text_body,
                    "figures": figures,
                    "tables": tables,
                })

        return sections

    except Exception as e:
        return []


def fetch_europe_pmc_topic(query: str, page_size: int = 5, delay: float = 0.5, fetch_fulltext: bool = True) -> List[Dict]:
    """Fetch articles for a query from Europe PMC REST API, optionally pulling full-text XML."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "core",
        "sort": "CITED desc",  # prioritize highly cited / clinically verified research
    }
    time.sleep(delay)
    response = requests.get(BASE_EPMC_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = data.get("resultList", {}).get("result", [])
    articles = []

    for item in results:
        abstract_raw = item.get("abstractText", "")
        abstract = clean_html_abstract(abstract_raw)
        if not abstract or len(abstract) < 80:
            continue

        item_id = item.get("id") or item.get("pmid")
        if not item_id:
            continue

        title = item.get("title", "").strip().rstrip(".")
        if not title:
            continue

        pmcid = item.get("pmcid")

        # Extract keywords & MeSH terms
        keywords = []
        kw_list = item.get("keywordList", {}).get("keyword", [])
        if isinstance(kw_list, list):
            keywords.extend(kw_list)
        elif isinstance(kw_list, str):
            keywords.append(kw_list)

        mesh_list = item.get("meshHeadingList", {}).get("meshHeading", [])
        if isinstance(mesh_list, list):
            for m in mesh_list:
                if isinstance(m, dict) and "descriptorName" in m:
                    keywords.append(m["descriptorName"])

        # Journal info
        journal_info = item.get("journalInfo", {})
        journal_title = journal_info.get("journal", {}).get("title") or "Europe PMC"
        pub_year = str(item.get("pubYear", ""))

        # External URL
        doi = item.get("doi", "")
        source_url = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{item_id}"

        # Fetch full-text XML sections if available
        sections = []
        has_fulltext = False
        if fetch_fulltext and pmcid:
            sections = fetch_fulltext_sections(pmcid, delay=delay)
            if sections:
                has_fulltext = True

        # Fallback to abstract if no sections
        if not sections:
            sections = [{"heading": "Tóm tắt (Abstract)", "content": abstract, "figures": [], "tables": []}]

        articles.append({
            "id": f"EPMC-{item_id}",
            "epmc_id": item_id,
            "pmid": item.get("pmid"),
            "pmcid": pmcid,
            "title": title,
            "content": abstract,  # keep abstract as main high-level summary
            "sections": sections, # full-text sections with figures & tables
            "has_fulltext": has_fulltext,
            "keywords": keywords[:15],
            "journal": journal_title,
            "pub_year": pub_year,
            "doi": doi,
            "source_url": source_url,
            "source": "Europe PMC",
            "language": "en",
            "cited_by_count": item.get("citedByCount", 0),
        })

    return articles


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch open access dental articles and full-text from Europe PMC")
    parser.add_argument("--max-per-topic", type=int, default=3, help="Số bài báo tối đa mỗi chủ đề")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=0.5, help="Độ trễ giữa các request")
    parser.add_argument("--no-fulltext", action="store_true", help="Chỉ lấy abstract, không kéo full-text XML")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "europe_pmc_articles.json"

    all_articles = []
    fetch_ft = not args.no_fulltext

    print(f"Bắt đầu thu thập dữ liệu nha khoa từ Europe PMC (Full-Text: {fetch_ft})...")

    new_collected = 0
    fulltext_count = 0

    for topic in DENTAL_TOPICS:
        category = topic["category"]
        query = topic["query"]
        print(f"\n[+] Đang truy vấn chuyên mục: {category}")

        try:
            articles = fetch_europe_pmc_topic(
                query, page_size=args.max_per_topic, delay=args.delay, fetch_fulltext=fetch_ft
            )

            for art in articles:
                art["category"] = category
                all_articles.append(art)
                new_collected += 1
                if art["has_fulltext"]:
                    fulltext_count += 1
                    ft_status = f"Đầy đủ toàn văn ({len(art['sections'])} sections)"
                else:
                    ft_status = "Chỉ có Abstract"
                print(f"    -> Đã nạp: [{art['id']}] {art['title'][:60]}... | {ft_status}")
        except Exception as e:
            print(f"    [!] Lỗi khi xử lý chủ đề '{category}': {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nHoàn tất Europe PMC! Tổng số bài: {len(all_articles)} (Có toàn văn Full-Text: {fulltext_count}/{len(all_articles)}).")
    print(f"Đường dẫn file: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
