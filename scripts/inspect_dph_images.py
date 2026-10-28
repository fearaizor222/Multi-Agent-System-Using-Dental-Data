import pymupdf
import re
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pdf_path = "data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf"
doc = pymupdf.open(pdf_path)

print("=== DETAILED IMAGE & CAPTION ANALYSIS ===")
seen_xrefs = set()
unique_images = []

for pno in range(18, 361):
    page = doc[pno]
    text = page.get_text()
    imgs = page.get_images(full=True)
    for img_info in imgs:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        base = doc.extract_image(xref)
        w, h, ext = base["width"], base["height"], base["ext"]
        if w >= 80 and h >= 80:
            caps = re.findall(r"(?:FIGURE|Figure)\s+\d+[-–]\d+[^\n\r]+", text)
            clean_caps = [c.strip()[:100] for c in caps]
            unique_images.append({
                "page": pno + 1,
                "xref": xref,
                "width": w,
                "height": h,
                "ext": ext,
                "size": len(base["image"]),
                "captions": clean_caps
            })

print(f"Total unique substantive images on chapter pages (19-361): {len(unique_images)}")
for item in unique_images:
    print(f"Page {item['page']}: xref={item['xref']}, {item['width']}x{item['height']}, size={item['size']} bytes | Captions: {item['captions']}")
