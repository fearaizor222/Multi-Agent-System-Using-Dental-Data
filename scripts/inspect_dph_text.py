import pymupdf
import re
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

doc = pymupdf.open("data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf")
toc = doc.get_toc()

# Filter chapters 1-23 and glossary
raw_entries = []
for item in toc:
    level, title, page = item
    if level == 2 or (level == 1 and ("Unit" not in title and page >= 19)):
        raw_entries.append((level, title, page))

chapters = []
for i, entry in enumerate(raw_entries):
    level, title, start_page = entry
    if i < len(raw_entries) - 1:
        end_page = raw_entries[i+1][2] - 1
    else:
        end_page = len(doc)
    
    # only include chapters 1-23 and Glossary
    is_chapter = bool(re.match(r"^\d+\.", title))
    is_glossary = "Glossary" in title
    if is_chapter or is_glossary:
        chapters.append({
            "title": title,
            "start_page": start_page,
            "end_page": end_page,
            "is_glossary": is_glossary
        })

print(f"Total target chapters/sections: {len(chapters)}")
for ch in chapters:
    # check text length
    text_len = sum(len(doc[p-1].get_text()) for p in range(ch["start_page"], ch["end_page"]+1))
    print(f"[{ch['start_page']:3d} - {ch['end_page']:3d}] {ch['title'][:55]:55s} | chars={text_len:6d}")
