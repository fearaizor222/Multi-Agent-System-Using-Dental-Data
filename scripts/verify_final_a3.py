import sys
import json
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.verification_selection import SelectedAssessment
from src.agents.contracts import Evidence, EvidenceQuote, ClaimAssessment
from src.agents.evidence_review import audit_quote, quote_contexts

CORPUS_DIR = ROOT_DIR / "data/processed/public_v1/corpus"
DATA_DIR = ROOT_DIR / "data/processed/a3_verification_v1"

# Load doc titles
doc_titles = {}
for fname in ["byt_documents.jsonl", "msd_documents.jsonl", "pmc_documents.jsonl"]:
    path = CORPUS_DIR / fname
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                doc_titles[d["document_id"]] = d.get("title", "")

# Load chunks
chunks = {}
with open(CORPUS_DIR / "chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks[c.get("chunk_id")] = c


def verify_dataset():
    total_samples = 0
    schema_errors = 0
    quote_audit_errors = 0
    options_lt_2 = 0
    src_counter = Counter()

    for split in ["train", "val"]:
        json_file = DATA_DIR / f"{split}.jsonl"
        meta_file = DATA_DIR / f"{split}.metadata.jsonl"

        with open(json_file, "r", encoding="utf-8") as f_j, open(meta_file, "r", encoding="utf-8") as f_m:
            for line_idx, (l_j, l_m) in enumerate(zip(f_j, f_m), 1):
                total_samples += 1
                rec = json.loads(l_j)
                meta = json.loads(l_m)

                # 1. Check messages structure
                messages = rec.get("messages", [])
                if len(messages) != 3:
                    print(f"[{split} #{line_idx}] Invalid message count: {len(messages)}")
                    schema_errors += 1
                    continue

                user_str = messages[1]["content"]
                parts = user_str.split("\nJSON schema:\n")
                user_payload = json.loads(parts[0])
                opts = user_payload.get("public_quote_options", [])
                if len(opts) < 2:
                    options_lt_2 += 1

                catalog = {o["quote_id"]: o for o in opts}

                # 2. Check assistant output with SelectedAssessment
                asst_str = messages[2]["content"]
                asst_dict = json.loads(asst_str)
                try:
                    assessment = SelectedAssessment.model_validate(asst_dict)
                except Exception as e:
                    print(f"[{split} #{line_idx}] SelectedAssessment validation error: {e}")
                    schema_errors += 1
                    continue

                # 3. Check quote_ids exist in catalog
                for qid in assessment.quote_ids:
                    if qid not in catalog:
                        print(f"[{split} #{line_idx}] Quote ID {qid} not in options catalog!")
                        schema_errors += 1

                # 4. Check quote_contexts & audit_quote runtime compatibility
                if assessment.public_support != "insufficient":
                    main_c = chunks[meta["public_chunk_id"]]
                    dist_c = chunks[meta["distractor_chunk_id"]]
                    main_is_p1 = (meta["main_evidence_id"] == "P1")

                    c1 = main_c if main_is_p1 else dist_c
                    c2 = dist_c if main_is_p1 else main_c

                    ev1 = Evidence(
                        evidence_id="P1",
                        source_id=c1["document_id"],
                        chunk_id=c1["chunk_id"],
                        warehouse="public",
                        title=doc_titles.get(c1["document_id"], ""),
                        content=c1["content"],
                        metadata={"heading_path": c1.get("heading_path", []), "document_id": c1["document_id"]}
                    )
                    ev2 = Evidence(
                        evidence_id="P2",
                        source_id=c2["document_id"],
                        chunk_id=c2["chunk_id"],
                        warehouse="public",
                        title=doc_titles.get(c2["document_id"], ""),
                        content=c2["content"],
                        metadata={"heading_path": c2.get("heading_path", []), "document_id": c2["document_id"]}
                    )
                    public_bundle = [ev1, ev2]

                    # Test ClaimAssessment & quote_contexts
                    selected_items = [catalog[k] for k in assessment.quote_ids]
                    claim_ass = ClaimAssessment(
                        draft_support=assessment.draft_support,
                        public_support=assessment.public_support,
                        public_evidence_ids=list(dict.fromkeys(item["evidence_id"] for item in selected_items)),
                        public_quotes=[EvidenceQuote(evidence_id=item["evidence_id"], text=item["text"]) for item in selected_items],
                        reason=assessment.reason
                    )
                    try:
                        quote_contexts(claim_ass, public_bundle)
                    except Exception as e:
                        print(f"[{split} #{line_idx}] quote_contexts audit failure: {e}")
                        quote_audit_errors += 1

                # 5. Count sources
                doc_id = meta.get("public_document_id", "")
                if doc_id.startswith("BYT-"):
                    src = "BYT"
                elif doc_id.startswith("MSD-"):
                    src = "MSD"
                elif doc_id.startswith("PMC"):
                    src = "PMC"
                else:
                    src = "OTHER"
                src_counter[src] += 1

    print("\n" + "="*50)
    print("KẾT QUẢ KIỂM CHỨNG TỔNG THỂ (FINAL VERIFICATION):")
    print(f"1. Tổng số mẫu đã kiểm tra: {total_samples}")
    print(f"2. Lỗi SelectedAssessment schema: {schema_errors}")
    print(f"3. Lỗi Runtime audit_quote / quote_contexts: {quote_audit_errors}")
    print(f"4. Mẫu có ít hơn 2 options: {options_lt_2}")
    print(f"5. Phân bố nguồn:")
    for src, cnt in src_counter.items():
        print(f"   - {src}: {cnt} ({cnt / total_samples * 100:.1f}%)")
    print("="*50)

    if schema_errors == 0 and quote_audit_errors == 0 and options_lt_2 == 0:
        print("[SUCCESS] TẬP DỮ LIỆU ĐẠT TIÊU CHUẨN HOÀN TOÀN (100% PASS)!")
    else:
        print("[FAILURE] CÓ LỖI CẦN XỬ LÝ!")

if __name__ == "__main__":
    verify_dataset()
