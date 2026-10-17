"""Fetch high-quality dental and maxillofacial clinical articles from PubMed via NCBI E-utilities API."""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "pubmed"

# Standard dental query topics for RHM RAG
DENTAL_TOPICS = [
    {
        "category": "Bệnh học Sâu răng & Tủy răng (Cariology & Endodontics)",
        "query": '("dental caries"[Title/Abstract] OR "pulpitis"[Title/Abstract]) AND (review[pt] OR guideline[pt] OR "clinical trial"[pt]) AND hasabstract[text]',
    },
    {
        "category": "Bệnh lý Nha chu & Lợi (Periodontology)",
        "query": '("periodontitis"[Title/Abstract] OR "gingivitis"[Title/Abstract]) AND (review[pt] OR guideline[pt] OR "clinical trial"[pt]) AND hasabstract[text]',
    },
    {
        "category": "Khớp thái dương hàm & Cắn khớp (TMJ & Occlusion)",
        "query": '("temporomandibular joint"[Title/Abstract] OR "malocclusion"[Title/Abstract] OR "bruxism"[Title/Abstract]) AND (review[pt] OR guideline[pt]) AND hasabstract[text]',
    },
    {
        "category": "Cấy ghép Nha khoa & Phục hình (Implantology & Prosthodontics)",
        "query": '("dental implants"[Title/Abstract] OR "prosthodontics"[Title/Abstract]) AND (review[pt] OR guideline[pt]) AND hasabstract[text]',
    },
    {
        "category": "Phẫu thuật miệng & Nhổ răng (Oral Surgery & Third Molar)",
        "query": '("tooth extraction"[Title/Abstract] OR "third molar"[Title/Abstract]) AND (review[pt] OR guideline[pt] OR "clinical trial"[pt]) AND hasabstract[text]',
    },
]

BASE_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed_pmids(query: str, retmax: int = 10, delay: float = 0.5) -> List[str]:
    """Search PubMed E-utilities for IDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub_date",
    }
    time.sleep(delay)
    response = requests.get(BASE_ESEARCH_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pubmed_details(pmids: List[str], delay: float = 0.5) -> List[Dict]:
    """Fetch article XML and parse structured medical information."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    time.sleep(delay)
    response = requests.get(BASE_EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    articles = []
    root = ET.fromstring(response.text)

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            # 1. PMID
            pmid = article_elem.findtext(".//MedlineCitation/PMID")
            if not pmid:
                continue

            # 2. Title
            title = article_elem.findtext(".//Article/ArticleTitle") or ""
            title = title.strip().rstrip(".")

            # 3. Abstract
            abstract_parts = []
            for abstract_text in article_elem.findall(".//Article/Abstract/AbstractText"):
                label = abstract_text.get("Label")
                text = (abstract_text.text or "").strip()
                if not text:
                    continue
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n\n".join(abstract_parts).strip()

            if not abstract:
                continue

            # 4. MeSH terms / Keywords
            keywords = []
            for mesh in article_elem.findall(".//MedlineCitation/MeshHeadingList/MeshHeading"):
                desc = mesh.findtext("DescriptorName")
                if desc:
                    keywords.append(desc.strip())

            for kw in article_elem.findall(".//MedlineCitation/KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

            # 5. Journal & Pub Year
            journal = article_elem.findtext(".//Article/Journal/Title") or article_elem.findtext(".//Article/Journal/ISOAbbreviation") or "PubMed"
            pub_year = article_elem.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
            if not pub_year:
                pub_year = article_elem.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
                if pub_year:
                    pub_year = pub_year[:4]

            # 6. DOI
            doi = None
            for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            articles.append({
                "id": f"PUBMED-{pmid}",
                "pmid": pmid,
                "title": title,
                "content": abstract,
                "keywords": keywords[:15],
                "journal": journal,
                "pub_year": pub_year,
                "doi": doi,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "PubMed Central / NCBI",
                "language": "en"
            })
        except Exception as e:
            continue

    return articles


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch dental clinical articles from PubMed API")
    parser.add_argument("--max-per-topic", type=int, default=5, help="Số bài báo tối đa mỗi chuyên mục")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=0.5, help="Giãn cách gọi API tránh rate-limit")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "pubmed_dental_articles.json"

    all_articles = []
    existing_pmids = set()

    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                all_articles = saved
                existing_pmids = {a["pmid"] for a in saved if "pmid" in a}
                print(f"Đã có sẵn {len(all_articles)} bài trong file {output_file.name}")
        except Exception:
            pass

    print(f"Bắt đầu thu thập dữ liệu nha khoa từ PubMed API ({len(DENTAL_TOPICS)} chuyên mục)...")

    new_collected = 0
    for topic in DENTAL_TOPICS:
        category = topic["category"]
        query = topic["query"]
        print(f"\n[+] Đang tìm kiếm chuyên mục: {category}")

        try:
            pmids = search_pubmed_pmids(query, retmax=args.max_per_topic, delay=args.delay)
            unseen_pmids = [p for p in pmids if p not in existing_pmids]
            print(f"    Tìm thấy {len(pmids)} PMIDs, có {len(unseen_pmids)} bài mới cần tải...")

            if unseen_pmids:
                articles = fetch_pubmed_details(unseen_pmids, delay=args.delay)
                for art in articles:
                    art["category"] = category
                    all_articles.append(art)
                    existing_pmids.add(art["pmid"])
                    new_collected += 1
                    print(f"    -> Đã nạp: [{art['pmid']}] {art['title'][:70]}...")
        except Exception as e:
            print(f"    [!] Lỗi khi xử lý chuyên mục '{category}': {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\nHoàn tất! Thu thập mới {new_collected} bài. Tổng số bài lưu trữ: {len(all_articles)}.")
    print(f"Đường dẫn file: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
