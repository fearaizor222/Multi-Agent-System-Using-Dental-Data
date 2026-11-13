"""Targeted A2 expansion: boost multi_evidence, distractor, and unused chunks.
Brings dataset to ~1,000-1,050 high-quality, balanced samples.
Guarantees 100% split isolation: only uses chunks from the 28 train documents.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import queue
import random
import re
import sys
import threading
import time
from typing import Literal
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.agents.contracts import GeneratedDraft
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY

DATA_DIR = ROOT / 'data/processed/a2_drafting_v2'
SOURCE = ROOT / 'data/raw/internal_imports/2026-11-02/chunks.jsonl'
CACHE_DIR = ROOT / 'outputs/a2_expansion_targeted'

GEN_MODEL = 'gemini-3.5-flash-lite'
AUDIT_MODEL = 'gemini-3.1-flash-lite'


def norm(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


class Proof(BaseModel):
    evidence_id: str
    quote: str


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=800)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)
    proofs: list[Proof] = Field(min_length=1, max_length=4)


class Turn(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class Candidate(BaseModel):
    id: str
    question: str = Field(min_length=5, max_length=600)
    history: list[Turn] = Field(max_length=4)
    status: Literal['draft', 'insufficient_evidence']
    claims: list[Claim] = Field(max_length=2)
    answerability_reason: str = Field(max_length=600)


class Batch(BaseModel):
    results: list[Candidate]


class Verdict(BaseModel):
    id: str
    status: Literal['draft', 'insufficient_evidence']
    clear: bool
    all_claims_supported: bool
    answers_current_question: bool
    supported_part_covered: bool
    citations_correct: bool
    actual_case: Literal['direct', 'missing_evidence', 'partial_answer', 'multi_source', 'history_dependent', 'topic_switch', 'injection', 'distractor']
    reason: str = Field(max_length=600)


class Reviews(BaseModel):
    results: list[Verdict]


GEN_POLICY = '''Bạn tạo dữ liệu A2 hỏi đáp nha khoa BẰNG TIẾNG VIỆT từ các đoạn tài liệu được cung cấp.
Chỉ dùng nguồn, không thêm kiến thức nhớ sẵn.
Mỗi item trả đúng id và một câu hỏi lâm sàng tự nhiên, tối đa 2 nhận định độc lập trả lời câu hỏi.
Mỗi claim có evidence_ids và proofs: trích NGUYÊN VĂN ngắn (>= 8 ký tự) từ chính evidence đó, đủ chứng minh MỌI chi tiết của claim.
Không thêm số/đơn vị/liều/thời gian ngoài proofs.
Nếu không có câu trả lời được nguồn hỗ trợ: status=insufficient_evidence, claims=[].

Ràng buộc cụ thể cho từng Mode:
1. multi_evidence: Đưa ra 2 đoạn I1 và I2. Câu hỏi BẮT BUỘC hỏi tổng hợp cả 2 ý lâm sàng (ví dụ: "Đoạn 1 nói gì về X và đoạn 2 nói gì về Y?"). Đáp án BẮT BUỘC có đúng 2 claims: claim 1 trích từ I1, claim 2 trích từ I2.
2. distractor: Đưa ra 2 đoạn I1 và I2, trong đó có 1 đoạn chính và 1 đoạn nhiễu. Câu hỏi CHỈ HỎI về nội dung của đoạn chính. Đáp án CHỈ TRÍCH DẪN đoạn chính (1 hoặc 2 claims), TUYỆT ĐỐI KHÔNG trích dẫn đoạn nhiễu.
3. direct: Hỏi câu hỏi lâm sàng trực tiếp về định nghĩa, cơ chế, triệu chứng hoặc phác đồ trong đoạn tài liệu chưa được khai thác.

history=[] ở các mode này. Dịch chuẩn xác tiếng Việt y khoa (root canal -> ống tủy, periapical -> quanh chóp, pulpitis -> viêm tủy). Không dịch root canal thành kẽ răng.'''

REVIEW_POLICY = '''Kiểm toán độc lập dữ liệu A2 hỏi đáp nha khoa.
Tự xác định status từ evidence trước, KHÔNG mặc định claims đề xuất là đúng.
draft khi có ít nhất phần câu hỏi trả lời được; insufficient_evidence chỉ khi không phần nào có căn cứ.
Đánh giá từng chi tiết claims: có được nguồn ĐƯỢC TRÍCH hỗ trợ trực tiếp? Số, đơn vị, điều kiện, phủ định phải đúng.
answers_current_question: câu trả lời đúng ý hỏi hiện tại.
supported_part_covered: đủ các ý có thể trả lời.
actual_case: phân loại tình huống thực tế (multi_source, distractor, partial_answer, missing_evidence, direct).
reason tiếng Việt ngắn gọn.'''


class ClientPool:
    def __init__(self):
        vals = dotenv_values(ROOT / '.env')
        keys = list(dict.fromkeys(v.strip() for k, v in vals.items()
                                  if re.fullmatch(r'GEMINI_API_KEY_\d+', k) and v.strip()))
        if not keys:
            raise ValueError("Không tìm thấy GEMINI_API_KEY_... trong .env")
        self.clients = [genai.Client(api_key=k) for k in keys]
        self.available = queue.Queue()
        for i in range(len(self.clients)):
            self.available.put(i)
        self.lock = threading.Lock()
        print(f"Khởi tạo ClientPool với {len(self.clients)} API keys.")

    def request(self, payload, model, schema, instruction):
        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=instruction,
            thinking_config=types.ThinkingConfig(thinking_level="LOW")
        )
        slot = self.available.get()
        try:
            for attempt in range(5):
                try:
                    resp = self.clients[slot].models.generate_content(
                        model=model,
                        contents=json.dumps(payload, ensure_ascii=False),
                        config=config
                    )
                    return json.loads(resp.text)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "503" in err_str:
                        time.sleep(2 ** attempt + random.random())
                    else:
                        raise e
            raise RuntimeError(f"Failed after 5 attempts on model {model}")
        finally:
            self.available.put(slot)


def validate_candidate(task, result):
    cand = Candidate.model_validate(result).model_dump()
    assert cand['id'] == task['id']
    assert (cand['status'] == 'draft') == bool(cand['claims'])
    sources = {e['evidence_id']: e['content'] for e in task['evidence']}
    for claim in cand['claims']:
        assert len(set(claim['evidence_ids'])) == len(claim['evidence_ids'])
        assert set(claim['evidence_ids']) == {p['evidence_id'] for p in claim['proofs']}
        for proof in claim['proofs']:
            assert proof['evidence_id'] in sources
            assert len(proof['quote'].strip()) >= 8
            assert norm(proof['quote']) in norm(sources[proof['evidence_id']])
        prooftext = ' '.join(p['quote'] for p in claim['proofs'])
        nums = lambda s: set(re.findall(r'\d+(?:[.,]\d+)?', s))
        assert nums(claim['text']) <= nums(prooftext)
    return cand


def plan_targeted_tasks():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    plan_file = CACHE_DIR / 'tasks_plan.jsonl'
    if plan_file.exists():
        return [json.loads(l) for l in plan_file.read_text(encoding='utf-8').splitlines() if l.strip()]

    all_chunks = [json.loads(l) for l in SOURCE.read_text(encoding='utf-8').splitlines() if l.strip()]
    
    val_meta = [json.loads(l) for l in (DATA_DIR / 'val.metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    test_meta = [json.loads(l) for l in (DATA_DIR / 'test.metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    train_meta = [json.loads(l) for l in (DATA_DIR / 'train.metadata.jsonl').read_text(encoding='utf-8').splitlines()]

    forbidden_docs = {m['document_id'] for m in val_meta} | {m['document_id'] for m in test_meta}
    forbidden_chunks = {c for m in val_meta for c in m['chunk_ids']} | {c for m in test_meta for c in m['chunk_ids']}
    
    train_docs = {m['document_id'] for m in train_meta} - forbidden_docs
    used_train_chunks = {c for m in train_meta for c in m['chunk_ids']}
    existing_ids = {m['id'] for m in train_meta}
    
    eligible = [c for c in all_chunks if c['document_id'] in train_docs 
                and c['chunk_id'] not in forbidden_chunks 
                and len(c['content'].split()) >= 50]
    
    by_doc = defaultdict(list)
    for c in eligible:
        by_doc[c['document_id']].append(c)

    unused_chunks = [c for c in eligible if c['chunk_id'] not in used_train_chunks]
    
    print(f"Train docs: {len(train_docs)} | Eligible chunks: {len(eligible)} | Unused chunks: {len(unused_chunks)}")

    rng = random.Random(20261111)
    tasks = []

    # 1. Multi-evidence tasks (target 130)
    for doc_id, doc_chunks in by_doc.items():
        if len(doc_chunks) >= 2:
            shuffled = list(doc_chunks)
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                c1, c2 = shuffled[i], shuffled[i+1]
                t_id = f"tgt_multi_{c1['chunk_id'][:8]}_{c2['chunk_id'][:8]}"
                if t_id in existing_ids: continue
                tasks.append({
                    'id': t_id,
                    'mode': 'multi_evidence',
                    'document_id': doc_id,
                    'chunk_ids': [c1['chunk_id'], c2['chunk_id']],
                    'evidence': [
                        {'evidence_id': 'I1', 'title': ' > '.join(c1.get('heading_path', ['Nội dung 1'])), 'content': c1['content']},
                        {'evidence_id': 'I2', 'title': ' > '.join(c2.get('heading_path', ['Nội dung 2'])), 'content': c2['content']}
                    ],
                    'suggested_question': 'Hỏi 2 ý lâm sàng: ý 1 từ I1 và ý 2 từ I2. Đáp án bắt buộc trả lời đủ cả 2 ý.'
                })
                if len([t for t in tasks if t['mode'] == 'multi_evidence']) >= 130:
                    break

    # 2. Distractor tasks (target 130)
    for doc_id, doc_chunks in by_doc.items():
        if len(doc_chunks) >= 3:
            shuffled = list(doc_chunks)
            rng.shuffle(shuffled)
            for i in range(len(shuffled) - 1):
                c_target = shuffled[i]
                c_distract = shuffled[i+1]
                t_id = f"tgt_dist_{c_target['chunk_id'][:8]}_{c_distract['chunk_id'][:8]}"
                if t_id in existing_ids: continue
                
                ev_target = {'title': ' > '.join(c_target.get('heading_path', ['Tài liệu chính'])), 'content': c_target['content']}
                ev_distract = {'title': ' > '.join(c_distract.get('heading_path', ['Tài liệu nhiễu'])), 'content': c_distract['content']}
                
                if rng.random() < 0.5:
                    ev = [ev_distract, ev_target]
                    c_ids = [c_distract['chunk_id'], c_target['chunk_id']]
                    main_eid = 'I2'
                else:
                    ev = [ev_target, ev_distract]
                    c_ids = [c_target['chunk_id'], c_distract['chunk_id']]
                    main_eid = 'I1'
                    
                for j, e in enumerate(ev, 1):
                    e['evidence_id'] = f'I{j}'
                    
                tasks.append({
                    'id': t_id,
                    'mode': 'distractor',
                    'document_id': doc_id,
                    'chunk_ids': c_ids,
                    'evidence': ev,
                    'suggested_question': f'Chỉ hỏi về nội dung trong đoạn {main_eid}, hoàn toàn bỏ qua đoạn còn lại.'
                })
                if len([t for t in tasks if t['mode'] == 'distractor']) >= 130:
                    break

    # 3. Direct tasks from unused chunks (target 100)
    rng.shuffle(unused_chunks)
    for c in unused_chunks[:100]:
        t_id = f"tgt_dir_{c['chunk_id'][:10]}"
        if t_id in existing_ids: continue
        tasks.append({
            'id': t_id,
            'mode': 'direct',
            'document_id': c['document_id'],
            'chunk_ids': [c['chunk_id']],
            'evidence': [{'evidence_id': 'I1', 'title': ' > '.join(c.get('heading_path', ['Tài liệu'])), 'content': c['content']}],
            'suggested_question': 'Hỏi trực tiếp về định nghĩa, triệu chứng hoặc phác đồ lâm sàng trong đoạn.'
        })

    with open(plan_file, 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
            
    print(f"Đã lập kế hoạch {len(tasks)} targeted tasks: {dict(Counter(t['mode'] for t in tasks))}")
    return tasks


def run_targeted_expansion(pool, tasks):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gen_cache_path = CACHE_DIR / 'generated.jsonl'
    rev_cache_path = CACHE_DIR / 'reviewed.jsonl'
    
    gen_cached = {r['id']: r for r in [json.loads(l) for l in gen_cache_path.read_text(encoding='utf-8').splitlines()] if 'result' in r} if gen_cache_path.exists() else {}
    
    # 1. Generation Stage
    todo_gen = [t for t in tasks if t['id'] not in gen_cached]
    print(f"[GEN] Cần sinh: {len(todo_gen)} / {len(tasks)} (Đã có: {len(gen_cached)})")
    
    batch_size = 4
    gen_batches = [todo_gen[i:i+batch_size] for i in range(0, len(todo_gen), batch_size)]
    
    def process_gen_batch(batch):
        payload = [{'id': t['id'], 'evidence': t['evidence'], 'mode': t['mode'], 'suggested_question': t['suggested_question']} for t in batch]
        raw_res = pool.request(payload, GEN_MODEL, Batch, GEN_POLICY)
        res_by_id = {r['id']: r for r in raw_res.get('results', [])}
        out_rows = []
        for t in batch:
            if t['id'] in res_by_id:
                try:
                    cand = validate_candidate(t, res_by_id[t['id']])
                    out_rows.append({'id': t['id'], 'task': t, 'result': cand})
                except Exception as e:
                    out_rows.append({'id': t['id'], 'task': t, 'rejected': str(e)})
            else:
                out_rows.append({'id': t['id'], 'task': t, 'rejected': 'missing_id_in_response'})
        return out_rows

    if todo_gen:
        with ThreadPoolExecutor(max_workers=len(pool.clients)) as executor:
            futures = [executor.submit(process_gen_batch, b) for b in gen_batches]
            completed_count = 0
            for f in as_completed(futures):
                rows = f.result()
                with gen_cache_path.open('a', encoding='utf-8') as stream:
                    for r in rows:
                        stream.write(json.dumps(r, ensure_ascii=False) + '\n')
                        if 'result' in r:
                            gen_cached[r['id']] = r
                completed_count += len(rows)
                if completed_count % 20 < 4 or completed_count >= len(todo_gen):
                    print(f"[GEN] Hoàn thành: {completed_count} / {len(todo_gen)}", flush=True)

    # 2. Review Stage
    valid_candidates = {t['id']: gen_cached[t['id']]['result'] for t in tasks if t['id'] in gen_cached and 'result' in gen_cached[t['id']]}
    rev_cached = {r['id']: r for r in [json.loads(l) for l in rev_cache_path.read_text(encoding='utf-8').splitlines()] if 'verdict' in r} if rev_cache_path.exists() else {}
    
    todo_rev = [t for t in tasks if t['id'] in valid_candidates and t['id'] not in rev_cached]
    print(f"[REV] Cần kiểm toán: {len(todo_rev)} / {len(valid_candidates)} (Đã kiểm toán: {len(rev_cached)})")
    
    rev_batch_size = 6
    rev_batches = [todo_rev[i:i+rev_batch_size] for i in range(0, len(todo_rev), rev_batch_size)]
    
    def process_rev_batch(batch):
        payload = [{'id': t['id'], 'evidence': t['evidence'], 'candidate': valid_candidates[t['id']]} for t in batch]
        raw_res = pool.request(payload, AUDIT_MODEL, Reviews, REVIEW_POLICY)
        res_by_id = {r['id']: r for r in raw_res.get('results', [])}
        out_rows = []
        for t in batch:
            if t['id'] in res_by_id:
                out_rows.append({'id': t['id'], 'task': t, 'verdict': res_by_id[t['id']]})
            else:
                out_rows.append({'id': t['id'], 'task': t, 'rejected': 'missing_verdict'})
        return out_rows

    if todo_rev:
        with ThreadPoolExecutor(max_workers=len(pool.clients)) as executor:
            futures = [executor.submit(process_rev_batch, b) for b in rev_batches]
            completed_rev = 0
            for f in as_completed(futures):
                rows = f.result()
                with rev_cache_path.open('a', encoding='utf-8') as stream:
                    for r in rows:
                        stream.write(json.dumps(r, ensure_ascii=False) + '\n')
                        if 'verdict' in r:
                            rev_cached[r['id']] = r
                completed_rev += len(rows)
                if completed_rev % 20 < 6 or completed_rev >= len(todo_rev):
                    print(f"[REV] Đã kiểm toán: {completed_rev} / {len(todo_rev)}", flush=True)

    # 3. Assemble Approved Additions
    approved_samples = []
    quarantine = []
    
    for t in tasks:
        if t['id'] not in valid_candidates or t['id'] not in rev_cached:
            continue
        c = valid_candidates[t['id']]
        v = rev_cached[t['id']]['verdict']
        
        flags = ('clear', 'all_claims_supported', 'answers_current_question', 'supported_part_covered', 'citations_correct')
        is_approved = (v['clear'] and v['actual_case'] == 'missing_evidence' and not c['claims']) if c['status'] == 'insufficient_evidence' else all(v.get(k) for k in flags)
        
        if not is_approved or v['status'] != c['status']:
            quarantine.append({'id': t['id'], 'reason': 'audit_rejected', 'verdict': v})
            continue
            
        payload = {
            'question': c['question'],
            'evidence': t['evidence'],
            'allowed_evidence_ids': [e['evidence_id'] for e in t['evidence']]
        }
        gold = {
            'status': c['status'],
            'claims': [{'text': x['text'], 'evidence_ids': x['evidence_ids']} for x in c['claims']]
        }
        GeneratedDraft.model_validate(gold)
        
        user_content = json.dumps(payload, ensure_ascii=False) + '\nJSON schema:\n' + json.dumps(GeneratedDraft.model_json_schema(), ensure_ascii=False)
        asst_content = json.dumps(gold, ensure_ascii=False)
        record = {
            'messages': [
                {'role': 'system', 'content': DRAFT_SYSTEM},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': asst_content}
            ]
        }
        metadata = {
            'id': t['id'],
            'split': 'train',
            'document_id': t['document_id'],
            'chunk_ids': t['chunk_ids'],
            'mode': v['actual_case'],
            'family_id': t['id'],
            'requested_mode': t['mode'],
            'question': c['question'],
            'status': c['status'],
            'claims_count': len(c['claims']),
            'evidence_ids': payload['allowed_evidence_ids'],
            'proofs': c['claims'],
            'review': v,
            'review_status': 'targeted_gemini_reviewed'
        }
        if c['status'] == 'insufficient_evidence':
            metadata['blind_answerability'] = {
                'id': t['id'],
                'status': 'insufficient_evidence',
                'decision_unambiguous': True,
                'answerable_quotes': [],
                'reason': v.get('reason', 'Nguồn không có thông tin.')
            }
        approved_samples.append((record, metadata))

    print(f"\nTổng mẫu targeted được duyệt: {len(approved_samples)} (Bị từ chối: {len(quarantine)})")
    return approved_samples


def main():
    pool = ClientPool()
    tasks = plan_targeted_tasks()
    approved = run_targeted_expansion(pool, tasks)
    
    if not approved:
        print("Không có mẫu nào được duyệt.")
        return

    # Load existing train
    train_jsonl = DATA_DIR / 'train.jsonl'
    train_meta = DATA_DIR / 'train.metadata.jsonl'
    
    existing_records = [json.loads(l) for l in train_jsonl.read_text(encoding='utf-8').splitlines() if l.strip()]
    existing_metas = [json.loads(l) for l in train_meta.read_text(encoding='utf-8').splitlines() if l.strip()]
    
    existing_ids = {m['id'] for m in existing_metas}
    new_records = []
    new_metas = []
    
    for rec, meta in approved:
        if meta['id'] not in existing_ids:
            new_records.append(rec)
            new_metas.append(meta)
            existing_ids.add(meta['id'])

    print(f"Thêm {len(new_records)} mẫu mới vào train (Từ {len(existing_records)} -> {len(existing_records) + len(new_records)})")
    
    with train_jsonl.open('a', encoding='utf-8') as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with train_meta.open('a', encoding='utf-8') as f:
        for m in new_metas:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')

    # Update manifest
    manifest_path = DATA_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    
    all_train_metas = existing_metas + new_metas
    manifest['counts']['train'] = len(all_train_metas)
    manifest['labels']['train'] = dict(Counter(m['status'] for m in all_train_metas))
    manifest['modes']['train'] = dict(Counter(m['mode'] for m in all_train_metas))
    manifest['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA_DIR.glob('*.jsonl')}
    
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print("Manifest đã được cập nhật thành công!")


if __name__ == '__main__':
    main()
