import pymupdf
import json

doc = pymupdf.open("data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf")
toc = doc.get_toc()

# Find chapter boundaries
# Chapters 1-23, plus Appendices, plus Glossary
entries = []
for item in toc:
    level, title, page = item
    if level == 2 or (level == 1 and ("Unit" not in title and page >= 19)):
        entries.append({"level": level, "title": title, "start_page": page})

# Calculate end_page for each entry
for i in range(len(entries)):
    if i < len(entries) - 1:
        entries[i]["end_page"] = entries[i+1]["start_page"] - 1
    else:
        entries[i]["end_page"] = len(doc)

print(f"Total entries found: {len(entries)}")
for e in entries:
    print(f"[{e['start_page']:3d} - {e['end_page']:3d}] {e['title']}")
