"""Script rà soát và chuẩn hóa toàn bộ nhãn (Label Cleaning & Audit) cho tập dữ liệu A3.
Sử dụng Gemini API làm giám khảo y khoa độc lập (Medical Fact-Checking Auditor) với temperature=0.0.
"""
import os
import sys
import json
import time
import shutil
import random
import queue
import threading
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Tuple
import concurrent.futures
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.verification_selection import SelectedAssessment

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def is_suspicious_or_non_trivial(record: dict, meta: dict) -> Tuple[bool, str]:
    """Kiểm tra xem mẫu có cần được giám định lại bởi LLM Auditor hay không."""
    ds = meta.get("draft_support")
    ps = meta.get("public_support")
    reason = meta.get("reason", "").lower()
    qids = meta.get("quote_ids", [])
    scen = meta.get("scenario", "")

    # Mọi mẫu không phải supported thuần túy đều cần audit
    if ds != "supported":
        return True, f"draft_support={ds}"
    if ps != "supported":
        return True, f"public_support={ps}"

    # Kiểm tra bất thường giữa reason và nhãn supported
    has_contra = any(w in reason for w in ["mâu thuẫn", "trái ngược", "bác bỏ", "nói ngược", "sai lệch", "ngược lại"])
    has_insuff = any(w in reason for w in ["không đề cập", "chưa đủ", "không có thông tin", "không nói đến", "không chứa", "chỉ nói về", "chỉ cung cấp"])
    if has_contra or has_insuff:
        return True, "reason_contradicts_supported"

    if "draft_" in scen:
        return True, "augmented_draft_sample"

    return False, "clean"


class AuditKeyWorker:
    def __init__(self, worker_id: int, api_key: str, model_name: str, delay: float = 4.0):
        self.worker_id = worker_id
        self.api_key = api_key
        self.model_name = model_name
        self.delay = delay
        self.last_call_time = 0.0
        self.lock = threading.Lock()
        self.is_exhausted = False

        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def audit_sample(self, question: str, claim: str, internal_ev: list, public_opts: list) -> dict:
        with self.lock:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

            prompt = f"""Bạn là giám khảo y khoa nha khoa độc lập (Medical Fact-Checking Auditor).
Hãy đọc kỹ thông tin sau và đưa ra đánh giá khách quan, chuẩn xác tuyệt đối:

CÂU HỎI: {question}
NHẬN ĐỊNH CỦA AGENT 2 (claim):
{claim}

TÀI LIỆU NỘI BỘ (internal_evidence):
{json.dumps(internal_ev, ensure_ascii=False)}

CÁC ĐOẠN TRÍCH CÔNG KHAI (public_quote_options):
{json.dumps(public_opts, ensure_ascii=False)}

HÃY ĐÁNH GIÁ ĐỘC LẬP:
1. draft_support: 'supported' | 'contradicted' | 'insufficient'
   - supported: tài liệu nội bộ hỗ trợ / xác nhận đủ ý nhận định.
   - contradicted: tài liệu nội bộ nói ngược lại / mâu thuẫn nhận định.
   - insufficient: tài liệu nội bộ không nhắc tới / thiếu thông tin / nói chủ đề khác.
2. public_support: 'supported' | 'contradicted' | 'insufficient'
   - supported: tài liệu công khai xác nhận nhận định.
   - contradicted: tài liệu công khai nói ngược / phản bác nhận định.
   - insufficient: tài liệu công khai không đủ thông tin / không đề cập.
3. quote_ids: danh sách mã từ public_quote_options (BẮT BUỘC RỖNG [] nếu public_support là insufficient; chọn mã nếu supported hoặc contradicted).
4. reason: đúng MỘT câu giải thích ngắn gọn bằng tiếng Việt dưới 240 ký tự nêu rõ lý do đánh giá cả 2 nguồn.

Trả về duy nhất 1 JSON object:
{{
  "draft_support": "supported" | "contradicted" | "insufficient",
  "public_support": "supported" | "contradicted" | "insufficient",
  "quote_ids": [...],
  "reason": "..."
}}"""
            for attempt in range(1, 6):
                try:
                    config = self.types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    )
                    resp = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    self.last_call_time = time.time()
                    raw = resp.text.strip()
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    data = json.loads(raw.strip())
                    return data
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        sleep_time = min(30.0, 2 ** attempt + random.uniform(1, 3))
                        time.sleep(sleep_time)
                    else:
                        time.sleep(2.0)
            raise RuntimeError(f"Worker #{self.worker_id} thất bại khi audit sau 5 lần thử.")


def run_audit(
    dataset_dir: Path,
    model_name: str = "gemini-3.5-flash-lite",
    delay: float = 4.0,
    seed: int = 42,
    train_ratio: float = 0.85,
    max_audit_samples: Optional[int] = None,
):
    random.seed(seed)
    load_dotenv(ROOT_DIR / ".env")

    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    train_meta_file = dataset_dir / "train.metadata.jsonl"
    val_meta_file = dataset_dir / "val.metadata.jsonl"

    # 1. Backup dataset trước khi sửa
    backup_dir = dataset_dir.parent / f"{dataset_dir.name}_backup_pre_clean"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["train.jsonl", "val.jsonl", "train.metadata.jsonl", "val.metadata.jsonl"]:
            src = dataset_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)
        print(f"[Backup] Đã tạo bản sao lưu an toàn tại: {backup_dir}")

    # 2. Đọc toàn bộ mẫu
    all_samples = []
    for split_name, j_path, m_path in [("train", train_file, train_meta_file), ("val", val_file, val_meta_file)]:
        with open(j_path, "r", encoding="utf-8") as f_j, open(m_path, "r", encoding="utf-8") as f_m:
            for line_no, (l_j, l_m) in enumerate(zip(f_j, f_m), 1):
                rec = json.loads(l_j)
                meta = json.loads(l_m)
                all_samples.append({
                    "split": split_name,
                    "line_no": line_no,
                    "record": rec,
                    "meta": meta,
                })

    print(f"[Audit] Tổng số mẫu nạp vào: {len(all_samples)}")

    # 3. Thu thập danh sách cần Audit
    items_to_audit = []
    items_clean = []
    for item in all_samples:
        need_audit, reason_flag = is_suspicious_or_non_trivial(item["record"], item["meta"])
        item["audit_flag"] = reason_flag
        if need_audit:
            items_to_audit.append(item)
        else:
            items_clean.append(item)

    print(f"[Audit] Số mẫu thuần túy đã chuẩn: {len(items_clean)}")
    print(f"[Audit] Số mẫu cần giám khảo Gemini rà soát: {len(items_to_audit)}")

    if max_audit_samples and len(items_to_audit) > max_audit_samples:
        print(f"[Audit] Giới hạn xử lý {max_audit_samples} mẫu đầu tiên...")
        items_to_audit = items_to_audit[:max_audit_samples]

    # 4. Thiết lập hồ bơi workers
    keys = []
    for i in range(1, 11):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k and k.strip() and k.strip() not in keys:
            keys.append(k.strip())
    single = os.getenv("GEMINI_API_KEY")
    if single and single.strip() and single.strip() not in keys:
        keys.append(single.strip())

    if not keys:
        raise ValueError("Không tìm thấy GEMINI_API_KEY trong .env!")

    workers = [AuditKeyWorker(i, k, model_name, delay=delay) for i, k in enumerate(keys, 1)]
    worker_queue: queue.Queue[AuditKeyWorker] = queue.Queue()
    for w in workers:
        worker_queue.put(w)

    print(f"[Audit] Khởi tạo hồ bơi gồm {len(workers)} workers (Model: {model_name})")

    # Cache checkpoint cho audit
    cache_file = dataset_dir / ".cache_a3_audit.jsonl"
    audit_cache = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f_c:
            for line in f_c:
                if line.strip():
                    entry = json.loads(line.strip())
                    audit_cache[entry["chunk_id"]] = entry["audit_result"]
        print(f"[Cache] Nạp {len(audit_cache)} kết quả audit từ cache.")

    cache_append = open(cache_file, "a", encoding="utf-8")
    cache_lock = threading.Lock()

    # 5. Thực thi Audit song song
    corrections_ds = 0
    corrections_ps = 0

    pbar = tqdm(total=len(items_to_audit), desc="Auditing & Cleaning Labels", unit="sample")

    def process_item(item):
        nonlocal corrections_ds, corrections_ps
        rec = item["record"]
        meta = item["meta"]
        cid = meta["public_chunk_id"]

        user_payload = json.loads(rec["messages"][1]["content"].split("\nJSON schema:\n")[0])
        question = user_payload["question"]
        claim = user_payload["claim"]
        internal_ev = user_payload["internal_evidence"]
        public_opts = user_payload["public_quote_options"]
        allowed_qids = [opt["quote_id"] for opt in public_opts]

        if cid in audit_cache:
            audit_res = audit_cache[cid]
        else:
            w = worker_queue.get()
            try:
                audit_res = w.audit_sample(question, claim, internal_ev, public_opts)
                with cache_lock:
                    cache_append.write(json.dumps({"chunk_id": cid, "audit_result": audit_res}, ensure_ascii=False) + "\n")
                    cache_append.flush()
            finally:
                worker_queue.put(w)

        # Chuẩn hóa kết quả
        new_ds = audit_res.get("draft_support", "supported")
        new_ps = audit_res.get("public_support", "supported")
        new_qids = audit_res.get("quote_ids", [])
        new_reason = audit_res.get("reason", meta["reason"])

        if new_ps == "insufficient":
            new_qids = []
        else:
            valid = [q for q in new_qids if q in allowed_qids]
            new_qids = valid if valid else ([allowed_qids[0]] if allowed_qids else [])

        if len(new_reason) > 240:
            new_reason = new_reason[:237] + "..."

        # Validate SelectedAssessment
        validated = SelectedAssessment(
            draft_support=new_ds,
            public_support=new_ps,
            quote_ids=new_qids,
            reason=new_reason,
        )

        old_ds = meta["draft_support"]
        old_ps = meta["public_support"]

        if old_ds != validated.draft_support:
            corrections_ds += 1
        if old_ps != validated.public_support:
            corrections_ps += 1

        # Cập nhật metadata
        meta["draft_support"] = validated.draft_support
        meta["public_support"] = validated.public_support
        meta["quote_ids"] = validated.quote_ids
        meta["reason"] = validated.reason

        # Cập nhật assistant message
        rec["messages"][2]["content"] = json.dumps(validated.model_dump(), ensure_ascii=False)

        pbar.update(1)
        return item

    num_threads = min(len(workers), len(items_to_audit))
    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_item, it) for it in items_to_audit]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"\n[Error] Lỗi khi audit mẫu: {e}")
    else:
        for it in items_to_audit:
            process_item(it)

    pbar.close()
    cache_append.close()

    print(f"\n[Audit Results] Hoàn tất rà soát:")
    print(f"       + Số mẫu đã sửa `public_support`: {corrections_ps}")
    print(f"       + Số mẫu đã sửa `draft_support`:  {corrections_ds}")

    # 6. Gộp lại và phân tầng chia lại Train / Val
    all_cleaned = items_clean + items_to_audit
    print(f"[Merge] Tổng mẫu sau chuẩn hóa: {len(all_cleaned)}")

    strata = defaultdict(list)
    for it in all_cleaned:
        m = it["meta"]
        doc_id = m.get("public_document_id", "")
        if doc_id.startswith("BYT-"):
            src_grp = "BYT"
        elif doc_id.startswith("MSD-"):
            src_grp = "MSD"
        elif doc_id.startswith("PMC"):
            src_grp = "PMC"
        else:
            src_grp = "OTHER"
        key = (src_grp, m["draft_support"], m["public_support"])
        strata[key].append(it)

    new_train = []
    new_val = []
    for key, items in strata.items():
        random.shuffle(items)
        if len(items) == 1:
            new_train.append(items[0])
        else:
            n_v = max(1, round(len(items) * (1.0 - train_ratio)))
            new_val.extend(items[:n_v])
            new_train.extend(items[n_v:])

    random.shuffle(new_train)
    random.shuffle(new_val)

    # 7. Xuất file sau khi làm sạch
    with open(train_file, "w", encoding="utf-8") as f_j, open(train_meta_file, "w", encoding="utf-8") as f_m:
        for idx, it in enumerate(new_train, 1):
            m = it["meta"]
            m["id"] = f"a3_train_{idx:04d}"
            m["split"] = "train"
            f_j.write(json.dumps(it["record"], ensure_ascii=False) + "\n")
            f_m.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f_j, open(val_meta_file, "w", encoding="utf-8") as f_m:
        for idx, it in enumerate(new_val, 1):
            m = it["meta"]
            m["id"] = f"a3_val_{idx:04d}"
            m["split"] = "val"
            f_j.write(json.dumps(it["record"], ensure_ascii=False) + "\n")
            f_m.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\n[Export] Đã xuất thành công bộ dữ liệu làm sạch:")
    print(f"       + Train: {len(new_train)} mẫu -> {train_file}")
    print(f"       + Val:   {len(new_val)} mẫu -> {val_file}")

    # 8. In bảng thống kê chi tiết mới
    from scripts.generate_a3_dataset import print_detailed_summary
    print_detailed_summary(train_meta_file, val_meta_file)


def main():
    parser = argparse.ArgumentParser(description="Audit and clean A3 dataset labels.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT_DIR / "data/processed/a3_verification_v1")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audit-samples", type=int, default=None)

    args = parser.parse_args()
    run_audit(
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        delay=args.delay,
        seed=args.seed,
        max_audit_samples=args.max_audit_samples,
    )


if __name__ == "__main__":
    main()
