import pymupdf
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_page_text(text: str, page_num: int) -> str:
    # 1. Normalize spaces
    text = text.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ").replace("\t", " ")
    
    # 2. Remove running headers / footers
    lines = text.splitlines()
    cleaned_lines = []
    for l in lines:
        s = l.strip()
        # Skip lone page numbers
        if re.match(r"^\d+$", s) and abs(int(s) - page_num) <= 2:
            continue
        # Skip common running headers
        if re.match(r"^UNIT\s+[IVXLCDM]+\s*[:–-]", s, re.I):
            continue
        if re.match(r"^CHAPTER\s+\d+\s+.*", s, re.I):
            continue
        cleaned_lines.append(l)
        
    text = "\n".join(cleaned_lines)
    
    # 3. Fix hyphenation at line breaks: word- \n word -> word\n
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2\n", text)
    
    # 4. Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

doc = pymupdf.open("data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf")
p20 = doc[19].get_text()
cleaned = clean_page_text(p20, 20)

print("=== CLEANED PAGE 20 PREVIEW ===")
print(cleaned[:1000])
