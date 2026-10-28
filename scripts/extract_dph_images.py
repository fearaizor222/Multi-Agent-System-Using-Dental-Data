import hashlib
import json
import re
import sys
from pathlib import Path
import pymupdf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data/raw/Dental_Public_Health_n_Research_-_Nathe_Christine.pdf"
IMAGES_DIR = ROOT / "data/images/dental_public_health"

def extract_all_figures():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF_PATH)
    
    # Pre-defined known clinical and key textbook figures mapping
    # to give them high quality labels, captions, and categories
    figure_metadata = {
        23: {
            2700: {
                "label": "Figure 1-3: Triumph of Death - Black Death",
                "caption": "Historical perspective of public health: Triumph of Death depicting the Black Death pandemic.",
                "category": "Historical Public Health",
                "filename": "fig_01_03_black_death.jpg"
            }
        },
        37: {
            3930: {
                "label": "Figure 2-1: Fones School of Dental Hygiene (Original)",
                "caption": "The original Fones School of Dental Hygiene in Bridgeport, Connecticut, the birthplace of dental hygiene.",
                "category": "Dental Hygiene History",
                "filename": "fig_02_01_fones_school_original.jpg"
            },
            3966: {
                "label": "Figure 2-2: Fones School of Dental Hygiene (Modern)",
                "caption": "Fones School of Dental Hygiene, Bridgeport, Connecticut (modern facility).",
                "category": "Dental Hygiene History",
                "filename": "fig_02_02_fones_school_modern.jpg"
            }
        },
        124: {
            2925: {
                "label": "Advocacy: Dental Hygienist in Public Health Setting",
                "caption": "Dental hygienist collaborating and advocating for dental care in community programs.",
                "category": "Advocacy & Community",
                "filename": "advocacy_community_hygienist.jpg"
            },
            2907: {
                "label": "Advocacy: Community Outreach and Screening",
                "caption": "Dental public health practitioner performing oral screening in community health center.",
                "category": "Advocacy & Community",
                "filename": "advocacy_community_screening.jpg"
            }
        },
        137: {
            4634: {
                "label": "Figure 9-1: Toothbrushing Drills & Oral Hygiene Instruction",
                "caption": "Classroom toothbrushing drills demonstrating oral health education methods.",
                "category": "Oral Health Education",
                "filename": "fig_09_01_toothbrushing_drills.jpg"
            }
        },
        230: {
            5680: {
                "label": "Figure 17: Normal Distribution Curve",
                "caption": "Biostatistics: Bell-shaped Gaussian normal distribution curve with standard deviations.",
                "category": "Biostatistics",
                "filename": "fig_17_normal_distribution.jpg"
            },
            5548: {
                "label": "Figure 17: Positively Skewed Distribution",
                "caption": "Biostatistics: Positively skewed distribution with tail extending toward the right.",
                "category": "Biostatistics",
                "filename": "fig_17_positive_skew.jpg"
            },
            5596: {
                "label": "Figure 17: Negatively Skewed Distribution",
                "caption": "Biostatistics: Negatively skewed distribution with tail extending toward the left.",
                "category": "Biostatistics",
                "filename": "fig_17_negative_skew.jpg"
            },
            5644: {
                "label": "Figure 17: Bimodal Distribution",
                "caption": "Biostatistics: Bimodal distribution with two distinct peaks.",
                "category": "Biostatistics",
                "filename": "fig_17_bimodal_distribution.jpg"
            }
        },
        282: {
            2239: {
                "label": "Figure 19-1: Gingivitis",
                "caption": "Clinical presentation of gingivitis: erythematous, edematous gingival margins and loss of stippling.",
                "category": "Clinical Pathology",
                "filename": "fig_19_01_gingivitis.jpg"
            }
        },
        285: {
            2611: {
                "label": "Figure 19-2: Early Childhood Caries (ECC)",
                "caption": "Clinical manifestation of severe Early Childhood Caries (baby bottle tooth decay) affecting maxillary anterior teeth.",
                "category": "Clinical Pathology",
                "filename": "fig_19_02_early_childhood_caries.jpg"
            }
        },
        292: {
            3127: {
                "label": "Figure 19-5: Oral Cancer - Squamous Cell Carcinoma",
                "caption": "Clinical presentation of oral cancer: persistent ulceration / lesion on the lateral border of the tongue.",
                "category": "Clinical Pathology",
                "filename": "fig_19_05_oral_cancer.jpg"
            }
        },
        293: {
            3211: {
                "label": "Figure 19-6: Herpetic Lesion (Oral Herpes)",
                "caption": "Clinical presentation of recurrent herpes labialis (cold sores) with clustered vesicles.",
                "category": "Clinical Pathology",
                "filename": "fig_19_06_herpetic_lesion.jpg"
            },
            3253: {
                "label": "Figure 19-7: Oral Candidiasis (Thrush)",
                "caption": "Clinical manifestation of oral candidiasis showing pseudomembranous white plaques on oral mucosa.",
                "category": "Clinical Pathology",
                "filename": "fig_19_07_oral_candidiasis.jpg"
            }
        },
        294: {
            3379: {
                "label": "Figure 19-8: Cleft Lip and Palate",
                "caption": "Congenital craniofacial malformation: infant with unilateral cleft lip and palate.",
                "category": "Craniofacial Anomalies",
                "filename": "fig_19_08_cleft_lip_palate.jpg"
            }
        },
        313: {
            5371: {
                "label": "Figure 21-1: HHS Organizational Chart",
                "caption": "United States Department of Health and Human Services (HHS) organizational structure.",
                "category": "Public Health Administration",
                "filename": "fig_21_01_hhs_org_chart.jpg"
            }
        },
        337: {
            7539: {
                "label": "Appendix A: Table Clinic Presentation Setup",
                "caption": "Visual display and table clinic layout for scientific presentation at dental conferences.",
                "category": "Scientific Presentation",
                "filename": "appendix_a_table_clinic.jpg"
            }
        }
    }
    
    extracted = []
    seen_xrefs = set()
    
    for pno in range(len(doc)):
        page_num = pno + 1
        page = doc[pno]
        images = page.get_images(full=True)
        
        for img_info in images:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            
            # Check if this image matches our designated textbook figures
            meta_match = figure_metadata.get(page_num, {}).get(xref)
            if not meta_match:
                # Check if it's cover (page 1)
                if page_num == 1 and xref == 270:
                    meta_match = {
                        "label": "Cover: Dental Public Health and Research (Nathe)",
                        "caption": "Front cover of Dental Public Health & Research: Contemporary Practice for the Dental Hygienist (4th Ed).",
                        "category": "Cover",
                        "filename": "cover_dental_public_health.jpg"
                    }
                else:
                    continue
            
            seen_xrefs.add(xref)
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = meta_match["filename"]
            out_file = IMAGES_DIR / filename
            out_file.write_bytes(image_bytes)
            
            sha = hashlib.sha256(image_bytes).hexdigest()
            record = {
                "figure_id": f"DPH-FIG-P{page_num:03d}-{xref}",
                "page": page_num,
                "xref": xref,
                "label": meta_match["label"],
                "caption": meta_match["caption"],
                "category": meta_match["category"],
                "filename": filename,
                "local_path": f"data/images/dental_public_health/{filename}",
                "dimensions": f"{base_image['width']}x{base_image['height']}",
                "width": base_image["width"],
                "height": base_image["height"],
                "filesize_bytes": len(image_bytes),
                "sha256": sha
            }
            extracted.append(record)
            print(f"Extracted [{record['page']}] {record['label']} -> {filename} ({record['dimensions']}, {record['filesize_bytes']} B)")

    catalog_path = IMAGES_DIR / "images_catalog.json"
    catalog_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved catalog: {catalog_path} with {len(extracted)} figures.")
    return extracted

if __name__ == "__main__":
    extract_all_figures()
