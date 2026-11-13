import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def fix_remaining_anomalies():
    train_file = ROOT_DIR / "data/processed/a3_verification_v1/train.jsonl"
    val_file = ROOT_DIR / "data/processed/a3_verification_v1/val.jsonl"
    train_meta_file = ROOT_DIR / "data/processed/a3_verification_v1/train.metadata.jsonl"
    val_meta_file = ROOT_DIR / "data/processed/a3_verification_v1/val.metadata.jsonl"

    fixed_count = 0

    for j_path, m_path in [(train_file, train_meta_file), (val_file, val_meta_file)]:
        records = []
        metas = []
        with open(j_path, "r", encoding="utf-8") as f_j, open(m_path, "r", encoding="utf-8") as f_m:
            for l_j, l_m in zip(f_j, f_m):
                rec = json.loads(l_j)
                meta = json.loads(l_m)
                
                cid = meta.get("public_chunk_id")
                reason = meta.get("reason", "")
                
                modified = False
                
                # Case 1: Val 130 / Bennett movement (chunk-ca4e000b...)
                if "ca4e000b" in cid:
                    meta["draft_support"] = "supported"
                    meta["public_support"] = "insufficient"
                    meta["quote_ids"] = []
                    meta["reason"] = "Tài liệu nội bộ xác nhận toàn bộ nhận định về vận động Bennett; tài liệu công khai hoàn toàn không đề cập đến chủ đề này."
                    modified = True
                    
                # Case 2: Exosome research 2010 (chunk-24b4e816...)
                elif "24b4e816" in cid and meta.get("public_support") == "insufficient" and "xác nhận chính xác" in reason:
                    meta["public_support"] = "supported"
                    meta["quote_ids"] = ["P1.Q1"]
                    modified = True
                    
                # Case 3: Dhinsa et al 2021 (chunk-1a0ffd00...)
                elif "1a0ffd00" in cid and meta.get("public_support") == "insufficient" and "xác nhận chính xác" in reason:
                    meta["public_support"] = "supported"
                    meta["quote_ids"] = ["P1.Q1"]
                    modified = True
                    
                # Case 4: Trismus / khít hàm (chunk-6d184e18...)
                elif "6d184e18" in cid and meta.get("draft_support") == "contradicted" and "không đề cập" in reason:
                    meta["draft_support"] = "insufficient"
                    modified = True
                    
                # Case 5: Tooth wear abrasion 57.2% (chunk-cff00660...)
                elif "cff00660" in cid and meta.get("public_support") == "insufficient" and "xác nhận tỷ lệ" in reason:
                    meta["public_support"] = "supported"
                    meta["quote_ids"] = ["P1.Q1"]
                    modified = True
                    
                # Case 6: Gingivitis indices (chunk-cd55aa54...)
                elif "cd55aa54" in cid and meta.get("draft_support") == "contradicted" and "khớp với nội dung" in reason:
                    meta["draft_support"] = "supported"
                    modified = True

                # General check: If reason says public doesn't mention at all, but public_support was contradicted
                elif meta.get("public_support") == "contradicted" and ("không đề cập" in reason or "không chứa" in reason) and not ("mâu thuẫn" in reason or "trái ngược" in reason or "ngược lại" in reason):
                    meta["public_support"] = "insufficient"
                    meta["quote_ids"] = []
                    modified = True

                if modified:
                    fixed_count += 1
                    # Update assistant message
                    ans = json.loads(rec["messages"][2]["content"])
                    ans["draft_support"] = meta["draft_support"]
                    ans["public_support"] = meta["public_support"]
                    ans["quote_ids"] = meta["quote_ids"]
                    ans["reason"] = meta["reason"]
                    rec["messages"][2]["content"] = json.dumps(ans, ensure_ascii=False)

                records.append(rec)
                metas.append(meta)

        with open(j_path, "w", encoding="utf-8") as f_j, open(m_path, "w", encoding="utf-8") as f_m:
            for rec, meta in zip(records, metas):
                f_j.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_m.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Applied targeted fixes to {fixed_count} remaining anomalous samples.")

if __name__ == "__main__":
    fix_remaining_anomalies()
