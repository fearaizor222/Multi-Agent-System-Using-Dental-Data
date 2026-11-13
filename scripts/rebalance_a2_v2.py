"""Rebalance A2 v2 dataset: replace toy partial questions with clinical missing details,
enrich partial_answer coverage, synchronize DRAFT_SYSTEM, and update manifest.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from src.agents.contracts import GeneratedDraft
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/processed/a2_drafting_v2'

CLINICAL_MISSING_QUESTIONS = [
    "Ngoài ra, tỷ lệ phần trăm người mắc tình trạng này theo thống kê dịch tễ học tại Việt Nam là bao nhiêu?",
    "Đồng thời cho biết liều lượng thuốc giảm đau chính xác bằng miligam được khuyến cáo là bao nhiêu?",
    "Thời gian theo dõi tái khám định kỳ được chỉ định cụ thể là bao nhiêu tháng?",
    "Tỷ lệ thành công sau 5 năm theo thống kê lâm sàng đạt bao nhiêu phần trăm?",
    "Ngưỡng thời gian chính xác tính bằng phút để triệu chứng tự thuyên giảm là bao lâu?",
    "Chi phí trung bình của một ca điều trị này là bao nhiêu theo bảng giá quy định?",
    "Thương hiệu vật liệu nha khoa cụ thể nào được khuyến nghị ưu tiên lựa chọn?",
    "Phác đồ dùng thuốc kháng sinh toàn thân phối hợp cụ thể trong trường hợp này là gì?",
    "Giá trị độ nhạy chẩn đoán của thiết bị đo huỳnh quang trong trường hợp này đạt bao nhiêu phần trăm?",
    "Thời gian trung bình phát hiện bệnh sau khi can thiệp điều trị là bao lâu?",
]

def update_split(name):
    jsonl_path = DATA_DIR / f"{name}.jsonl"
    meta_path = DATA_DIR / f"{name}.metadata.jsonl"
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        records = [json.loads(l) for l in f]
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = [json.loads(l) for l in f]
        
    assert len(records) == len(metas)
    
    enriched_records = []
    enriched_metas = []
    
    missing_q_idx = 0
    direct_converted = 0
    
    for i, (rec, meta) in enumerate(zip(records, metas)):
        payload_str = rec['messages'][1]['content'].split('\nJSON schema:\n')[0]
        payload = json.loads(payload_str)
        gold = json.loads(rec['messages'][2]['content'])
        
        # 1. Update system prompt
        expected_system = DRAFT_SYSTEM + (CONVERSATION_POLICY if 'conversation' in payload else '')
        rec['messages'][0]['content'] = expected_system
        
        # 2. Fix toy partial questions
        toy_str = " Ngoài ra, giá khám tại phòng khám hôm nay là bao nhiêu?"
        if toy_str in payload['question']:
            replacement = " " + CLINICAL_MISSING_QUESTIONS[missing_q_idx % len(CLINICAL_MISSING_QUESTIONS)]
            missing_q_idx += 1
            payload['question'] = payload['question'].replace(toy_str, replacement)
            meta['question'] = payload['question']
            
        # 3. Enrich train split: convert some direct (1-claim, no conversation) to partial_answer
        elif name == 'train' and meta.get('mode') == 'direct' and len(gold.get('claims', [])) == 1 and 'conversation' not in payload and direct_converted < 70:
            clinical_extra = " " + CLINICAL_MISSING_QUESTIONS[missing_q_idx % len(CLINICAL_MISSING_QUESTIONS)]
            missing_q_idx += 1
            payload['question'] = payload['question'] + clinical_extra
            meta['question'] = payload['question']
            meta['mode'] = 'partial_answer'
            meta['requested_mode'] = 'partial_answer'
            direct_converted += 1
            
        # Reserialize user content
        schema_part = '\nJSON schema:\n' + json.dumps(GeneratedDraft.model_json_schema(), ensure_ascii=False)
        rec['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False) + schema_part
        
        enriched_records.append(rec)
        enriched_metas.append(meta)
        
    print(f"[{name}] Total: {len(enriched_records)} | Converted direct to partial: {direct_converted}")
    
    # Write back
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in enriched_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(meta_path, 'w', encoding='utf-8') as f:
        for m in enriched_metas:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
            
    return enriched_records, enriched_metas

def main():
    all_data = {}
    for split in ('train', 'val', 'test'):
        all_data[split] = update_split(split)
        
    # Update manifest
    manifest_path = DATA_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    
    # Recount modes and labels
    for split in ('train', 'val', 'test'):
        recs, metas = all_data[split]
        manifest['labels'][split] = dict(Counter(m['status'] for m in metas))
        manifest['modes'][split] = dict(Counter(m['mode'] for m in metas))
        
    manifest['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob('*.jsonl')}
    manifest['runtime_system_sha256'] = hashlib.sha256(DRAFT_SYSTEM.encode('utf-8')).hexdigest()
    
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print("Manifest updated.")

if __name__ == '__main__':
    main()
