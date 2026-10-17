"""Preprocess and unify open data sources (MSD Manuals, PubMed, Europe PMC) into standardized DentalDocument format."""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]

MSD_RAW_PATH = ROOT_DIR / "data" / "raw" / "msdmanual" / "articles.jsonl"
PUBMED_RAW_PATH = ROOT_DIR / "data" / "raw" / "pubmed" / "pubmed_dental_articles.json"
EPMC_RAW_PATH = ROOT_DIR / "data" / "raw" / "europe_pmc" / "europe_pmc_articles.json"

OUTPUT_PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "open_dental_knowledge_sections.json"


def clean_text(text: str) -> str:
    """Normalize text encoding and whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def extract_keywords_from_text(title: str, text: str, max_keywords: int = 10) -> List[str]:
    """Simple medical keyword extractor for search optimization."""
    words = re.findall(r"\b[A-Za-zÀ-ỹ]{3,}\b", (title + " " + text[:500]).lower())
    stop_words = {
        "trong", "những", "các", "được", "người", "không", "thường", "khi",
        "hoặc", "nhiều", "cũng", "theo", "sau", "cho", "với", "phần", "trên",
        "dưới", "ngoài", "bệnh", "này", "một", "hai", "của", "tại", "nếu",
        "và", "có", "là", "bị", "gây", "ra", "đến", "từ", "vào",
        "the", "and", "with", "for", "that", "this", "from", "were", "was", "are", "study"
    }
    freq = {}
    for w in words:
        if w not in stop_words and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_keywords]]


def process_msd_manuals() -> List[Dict[str, Any]]:
    """Process MSD Manuals articles, splitting into section-level chunks."""
    if not MSD_RAW_PATH.exists():
        print(f"[-] Không tìm thấy {MSD_RAW_PATH}")
        return []

    chunks = []
    with open(MSD_RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            art_id = article.get("id", "MSD-UNK")
            art_title = clean_text(article.get("title", ""))
            source_url = article.get("source_url", "")
            sections = article.get("sections", [])

            if not sections:
                content = clean_text(article.get("content", ""))
                if content:
                    keywords = extract_keywords_from_text(art_title, content)
                    chunks.append({
                        "id": art_id,
                        "title": art_title,
                        "content": content,
                        "category": "Cẩm nang Nha khoa MSD",
                        "keywords": keywords,
                        "metadata": {
                            "source": "Cẩm nang MSD (MSD Manuals)",
                            "source_url": source_url,
                            "language": "vi",
                            "edition": article.get("edition", "professional"),
                            "last_reviewed": article.get("source_last_reviewed", ""),
                        }
                    })
                continue

            for idx, sec in enumerate(sections, 1):
                heading = clean_text(sec.get("heading", ""))
                sec_content = clean_text(sec.get("content", ""))
                if not sec_content or len(sec_content) < 50:
                    continue

                full_title = f"{art_title} - {heading}" if heading and heading != art_title else art_title
                chunk_id = f"{art_id}_sec{idx}"
                keywords = extract_keywords_from_text(full_title, sec_content)

                chunks.append({
                    "id": chunk_id,
                    "title": full_title,
                    "content": sec_content,
                    "category": "Cẩm nang Nha khoa MSD",
                    "keywords": keywords,
                    "metadata": {
                        "source": "Cẩm nang MSD (MSD Manuals)",
                        "parent_title": art_title,
                        "section_heading": heading,
                        "source_url": sec.get("source_url", source_url),
                        "language": "vi",
                        "edition": article.get("edition", "professional"),
                    }
                })

    print(f"[+] Đã xử lý {len(chunks)} chunks từ MSD Manuals.")
    return chunks


def process_pubmed() -> List[Dict[str, Any]]:
    """Process PubMed clinical review articles into standardized chunks."""
    if not PUBMED_RAW_PATH.exists():
        print(f"[-] Không tìm thấy {PUBMED_RAW_PATH}")
        return []

    chunks = []
    with open(PUBMED_RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for art in articles:
        pmid = art.get("pmid", "")
        title = clean_text(art.get("title", ""))
        content = clean_text(art.get("content", ""))
        if not content:
            continue

        keywords = art.get("keywords", [])
        if not keywords:
            keywords = extract_keywords_from_text(title, content)

        chunks.append({
            "id": f"PUBMED-{pmid}",
            "title": f"[PubMed] {title}",
            "content": content,
            "category": art.get("category", "PubMed Clinical Research"),
            "keywords": keywords,
            "metadata": {
                "source": "PubMed (NCBI)",
                "pmid": pmid,
                "journal": art.get("journal", ""),
                "pub_year": art.get("pub_year", ""),
                "doi": art.get("doi", ""),
                "source_url": art.get("source_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                "language": "en",
            }
        })

    print(f"[+] Đã xử lý {len(chunks)} chunks từ PubMed.")
    return chunks


def process_europe_pmc() -> List[Dict[str, Any]]:
    """Process Europe PMC open access articles into standardized chunks, including full-text sections."""
    if not EPMC_RAW_PATH.exists():
        print(f"[-] Không tìm thấy {EPMC_RAW_PATH}")
        return []

    chunks = []
    with open(EPMC_RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for art in articles:
        epmc_id = art.get("epmc_id", art.get("id", ""))
        title = clean_text(art.get("title", ""))
        keywords = art.get("keywords", [])
        sections = art.get("sections", [])
        has_fulltext = art.get("has_fulltext", False)

        # Base metadata
        base_meta = {
            "source": "Europe PMC Open Access",
            "epmc_id": epmc_id,
            "pmcid": art.get("pmcid", ""),
            "journal": art.get("journal", ""),
            "pub_year": art.get("pub_year", ""),
            "doi": art.get("doi", ""),
            "source_url": art.get("source_url", ""),
            "cited_by_count": art.get("cited_by_count", 0),
            "language": "en",
        }

        if has_fulltext and sections:
            for idx, sec in enumerate(sections, 1):
                heading = clean_text(sec.get("heading", "Clinical Section"))
                sec_content = clean_text(sec.get("content", ""))
                if not sec_content or len(sec_content) < 80:
                    continue

                full_title = f"[Europe PMC] {title} - {heading}"
                chunk_id = f"EPMC-{epmc_id}_sec{idx}"
                sec_kw = keywords if keywords else extract_keywords_from_text(full_title, sec_content)

                chunk_meta = dict(base_meta)
                chunk_meta["section_heading"] = heading
                chunk_meta["figures"] = sec.get("figures", [])
                chunk_meta["tables"] = sec.get("tables", [])

                chunks.append({
                    "id": chunk_id,
                    "title": full_title,
                    "content": sec_content,
                    "category": art.get("category", "Europe PMC Research"),
                    "keywords": sec_kw,
                    "metadata": chunk_meta
                })
        else:
            # Fallback to abstract
            content = clean_text(art.get("content", ""))
            if not content:
                continue
            if not keywords:
                keywords = extract_keywords_from_text(title, content)

            chunks.append({
                "id": f"EPMC-{epmc_id}",
                "title": f"[Europe PMC] {title}",
                "content": content,
                "category": art.get("category", "Europe PMC Research"),
                "keywords": keywords,
                "metadata": base_meta
            })

    print(f"[+] Đã xử lý {len(chunks)} chunks từ Europe PMC (Có hỗ trợ Full-Text sections & Figures).")
    return chunks


def main():
    print("=" * 60)
    print("Bắt đầu tiền xử lý và đồng bộ dữ liệu mở nha khoa (MSD, PubMed, Europe PMC)...")
    print("=" * 60)

    all_chunks = []
    all_chunks.extend(process_msd_manuals())
    all_chunks.extend(process_pubmed())
    all_chunks.extend(process_europe_pmc())

    OUTPUT_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Tổng hợp thành công {len(all_chunks)} chunks kiến thức mở!")
    print(f"Đã lưu tại: {OUTPUT_PROCESSED_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
