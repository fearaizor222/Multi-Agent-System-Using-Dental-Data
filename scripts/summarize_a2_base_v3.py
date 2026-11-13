"""Summarize frozen paired A2 inference and locked, blinded source-reading grades.

Runs locally, without model loading, training, retrieval, or external judge calls.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from scripts.train_a2 import parse_json
from src.agents.contracts import GeneratedDraft

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/a2_base_vs_v3_20261111'
DATA = ROOT / 'data/evaluation/a2_base_vs_v3_20261111'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    lock = read('blind_review_lock.json')
    assert hashlib.sha256((OUT / 'reviews_blind.json').read_bytes()).hexdigest() == lock['sha256']
    manifest = read('manifest.json')
    case_bytes = (DATA / 'cases.jsonl').read_bytes()
    assert hashlib.sha256(case_bytes).hexdigest() == manifest['cases_sha256']
    cases = [json.loads(x) for x in case_bytes.decode('utf-8').splitlines()]
    predictions = [json.loads(x) for x in (OUT / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert read('run_status.json')['status'] == 'completed'
    assert len(cases) == 52 and len(predictions) == 104
    reviews, key, identity = read('reviews_blind.json'), read('blinding_key.json'), read('identity.json')
    assert identity['selected_epoch'] == 2
    assert identity['adapter_sha256'] == '47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    assert not any(identity[k] for k in ('training_executed', 'other_agents_executed', 'rag_executed', 'nli_executed'))
    keyed = {(p['id'], p['variant']): p for p in predictions}
    assert len(keyed) == 104
    rows = []
    for i, case in enumerate(cases):
        assert set(key[case['id']].values()) == {'base', 'v3'}
        paired = [keyed[case['id'], v] for v in ('base', 'v3')]
        assert paired[0]['prompt_ids_sha256'] == paired[1]['prompt_ids_sha256']
        assert paired[0]['input_tokens'] == paired[1]['input_tokens']
        for label, variant in key[case['id']].items():
            p, review = keyed[case['id'], variant], reviews[str(i)][label]
            obj, schema_valid, error = None, False, None
            try:
                obj = parse_json(p['raw_output'])
                GeneratedDraft.model_validate(obj)
                schema_valid = True
            except (ValueError, TypeError) as exc:
                error = str(exc)
            claims = obj.get('claims', []) if isinstance(obj, dict) else []
            allowed = set(case['payload']['allowed_evidence_ids'])
            cited = [e for c in claims if isinstance(c, dict) for e in c.get('evidence_ids', [])]
            ids_valid = (schema_valid and len(claims) <= 2 and all(
                isinstance(c, dict) and c.get('evidence_ids') and set(c['evidence_ids']) <= allowed for c in claims))
            correct_status = isinstance(obj, dict) and obj.get('status') == case['expected']['status']
            rows.append({
                'index': i, 'id': case['id'], 'variant': variant, 'group': case['group'],
                'mode': case['mode'], 'source_language': '+'.join(case['source_languages']),
                'question': case['payload']['question'], 'expected_status': case['expected']['status'],
                'json_valid': obj is not None, 'schema_valid': schema_valid,
                'citation_ids_and_claim_limit_valid': bool(ids_valid), 'status_correct': correct_status,
                'has_claims': bool(claims), 'cited_ids': cited, 'parse_error': error,
                'manual_grade': review['grade'], 'review': review,
                'usable_pass': bool(schema_valid and ids_valid and correct_status and review['grade'] == 'pass'),
                'seconds': p['seconds'], 'output_tokens': p['output_tokens'],
                'input_tokens': p['input_tokens'], 'raw_output': p['raw_output'],
            })

    def aggregate(selected):
        counts = Counter(r['manual_grade'] for r in selected)
        n = len(selected)
        assessable = n - counts['unclear']
        return {
            'count': n, 'assessable': assessable, 'grades': dict(counts),
            'pass_rate_assessable': counts['pass'] / assessable if assessable else None,
            **{k: sum(bool(r[k]) for r in selected) for k in (
                'json_valid', 'schema_valid', 'citation_ids_and_claim_limit_valid', 'status_correct', 'usable_pass')},
            'correct_draft_decisions': sum(r['status_correct'] for r in selected if r['expected_status'] == 'draft'),
            'expected_draft_count': sum(r['expected_status'] == 'draft' for r in selected),
            'correct_insufficient_decisions': sum(r['status_correct'] for r in selected if r['expected_status'] == 'insufficient_evidence'),
            'expected_insufficient_count': sum(r['expected_status'] == 'insufficient_evidence' for r in selected),
            'median_seconds': statistics.median(r['seconds'] for r in selected),
            'mean_seconds': statistics.mean(r['seconds'] for r in selected),
            'mean_output_tokens': statistics.mean(r['output_tokens'] for r in selected),
            'outputs_reaching_token_cap': sum(r['output_tokens'] >= 512 for r in selected),
            'error_tags': dict(Counter(t for r in selected for t in r['review'].get('tags', []))),
        }

    variants = {}
    for v in ('base', 'v3'):
        selected = [r for r in rows if r['variant'] == v]
        variants[v] = {'all': aggregate(selected)}
        for dimension in ('group', 'source_language', 'mode'):
            variants[v][dimension] = {k: aggregate([r for r in selected if r[dimension] == k])
                                      for k in sorted({r[dimension] for r in selected})}
    paired_counts = Counter()
    for i in range(52):
        pair = {r['variant']: r for r in rows if r['index'] == i}
        b, a = pair['base']['manual_grade'], pair['v3']['manual_grade']
        if 'unclear' in (b, a): result = 'unclear'
        elif b == a == 'pass': result = 'both_pass'
        elif b == 'pass': result = 'base_only_pass'
        elif a == 'pass': result = 'v3_only_pass'
        else: result = 'neither_pass'
        paired_counts[result] += 1
    summary = {
        'scope': manifest['scope'], 'identity': identity, 'collection': read('collection.json'),
        'manual_grading': {'criteria': str(DATA / 'grading_criteria.json'), 'lock': lock,
            'reviewer': 'assistant source-reading, not a clinical expert or independent human panel',
            'unassessable_case_indices': sorted({r['index'] for r in rows if r['manual_grade'] == 'unclear'}),
            'citation_semantics_included': True,
            'note': 'Valid evidence IDs do not prove that the cited passage supports the claim. Content grades were assigned before model labels were revealed.'},
        'variants': variants, 'paired_pass_comparison': dict(paired_counts),
        'caveats': [
            '48 selected cases from an already-evaluated heldout set plus 4 known regressions; not a fresh independent clinical benchmark.',
            'Related family variants and repeated contexts occur; individual samples are not statistically independent.',
            'One quiz-source case lacks an answer key and is excluded from semantic pass-rate denominator for both variants.',
            'The source corpus includes OCR errors and older textbook material; grading checks supplied evidence, not current clinical correctness.',
            'Both variants use the same 4-bit NF4 base, not full precision. No prompt repair, retry, retrieval, or other agent is used.',
            'Cloud GPU timings do not predict laptop or end-to-end pipeline latency.',
        ],
    }
    (OUT / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'per_case_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

    b, a = (variants[v]['all'] for v in ('base', 'v3'))
    def fmt(n, d): return f'{n}/{d} ({100*n/d:.1f}%)' if d else '—'
    lines = [
        '# So sánh A2: Qwen3-1.7B gốc và adapter v3 epoch 2', '',
        '**Kết luận:** adapter v3 tăng số câu đạt yêu cầu nhưng chưa giải quyết tốt việc đọc nguồn tiếng Anh, dịch thuật và chọn đúng ý để trả lời. Không có căn cứ quy toàn bộ lỗi trước đây cho RAG.', '',
        '## Cách test', '',
        'Chỉ chạy A2, không chạy A1/A3/A4/ảnh, không truy xuất RAG, không gọi NLI. Hai bản nhận đúng cùng câu hỏi, đoạn nguồn nội bộ và prompt hiện tại; kiểm tra hash token đầu vào cho từng cặp. Model nền dùng cùng checkpoint lượng tử NF4 4-bit; tắt toàn bộ LoRA cho bản gốc và bật duy nhất A2 v3 epoch 2 cho bản adapter. Greedy, tắt thinking, tối đa 512 token, không sửa JSON hoặc thử lại.', '',
        '52 câu mỗi bản, gồm 48 mẫu chọn cố định từ tập test riêng đã đánh giá trước đây và 4 ca kiểm tra lỗi đã biết. Có câu trực tiếp, ghép nhiều đoạn, hỏi tiếp, câu chỉ trả lời được một phần và thiếu bằng chứng. Có các biến thể cùng chủ đề nên không xem là 52 ca hoàn toàn độc lập.', '',
        'Đọc đối chiếu đủ 104 đầu ra với nguồn; ẩn nhãn model theo từng câu trong lúc chấm rồi khóa kết quả trước khi mở nhãn. Đây là đánh giá đọc nguồn của trợ lý, chưa có chuyên gia nha khoa chấm độc lập. Điểm nội dung gồm đúng trọng tâm, đúng nghĩa nguồn, đủ ý chính và dẫn đúng đoạn; cách diễn đạt không cần giống đáp án mẫu. Điểm JSON/trạng thái được tính riêng.', '',
        'Một ca chỉ có phát biểu trắc nghiệm đúng/sai nhưng không có đáp án (index 19), không đủ căn cứ chấm nội dung; loại khỏi mẫu số của cả hai. Các lỗi OCR/đáp án mẫu khác cũng được đối chiếu với chính đoạn nguồn.', '',
        '## Kết quả', '',
        '| Tiêu chí | Qwen3 gốc | Qwen3 + A2 v3 |',
        '|---|---:|---:|',
        f'| Đạt nội dung và dẫn nguồn | {fmt(b["grades"].get("pass",0),b["assessable"])} | {fmt(a["grades"].get("pass",0),a["assessable"])} |',
        f'| Đúng một phần / sai / chưa chấm được | {b["grades"].get("partial",0)} / {b["grades"].get("fail",0)} / {b["grades"].get("unclear",0)} | {a["grades"].get("partial",0)} / {a["grades"].get("fail",0)} / {a["grades"].get("unclear",0)} |',
        f'| JSON hợp lệ | {fmt(b["json_valid"],52)} | {fmt(a["json_valid"],52)} |',
        f'| Hợp schema và nhất quán status/claims | {fmt(b["schema_valid"],52)} | {fmt(a["schema_valid"],52)} |',
        f'| Đúng trạng thái trả lời/thiếu bằng chứng theo nhãn | {fmt(b["status_correct"],52)} | {fmt(a["status_correct"],52)} |',
        f'| Có thể dùng: đạt nội dung + schema + trạng thái + mã dẫn nguồn | {fmt(b["usable_pass"],51)} | {fmt(a["usable_pass"],51)} |',
        f'| Thời gian sinh trung vị trên GPU {identity["gpu"]} | {b["median_seconds"]:.2f} giây | {a["median_seconds"]:.2f} giây |', '',
        'Tỉ lệ đúng trạng thái chỉ kiểm tra có trả lời hay từ chối; không chứng minh nội dung đúng. Mã I1/I2 hợp lệ cũng không có nghĩa câu được đoạn đó hỗ trợ.', '',
        '| Nhóm nguồn | Qwen3 gốc đạt nội dung | A2 v3 đạt nội dung |', '|---|---:|---:|',
    ]
    for lang, label in [('vi','Tiếng Việt'),('en','Tiếng Anh'),('en+vi','Anh + Việt')]:
        x, y = (variants[v]['source_language'][lang] for v in ('base','v3'))
        lines.append(f'| {label} | {fmt(x["grades"].get("pass",0),x["assessable"])} | {fmt(y["grades"].get("pass",0),y["assessable"])} |')
    lines.extend(['',
        f'Trong 51 cặp chấm được: cả hai đạt {paired_counts["both_pass"]}; chỉ bản gốc đạt {paired_counts["base_only_pass"]}; chỉ v3 đạt {paired_counts["v3_only_pass"]}; cả hai chưa đạt {paired_counts["neither_pass"]}. V3 tăng {100*(a["pass_rate_assessable"]-b["pass_rate_assessable"]):.1f} điểm phần trăm trong phép thử này.', '',
        '48 mẫu test cũ và 4 ca đã biết được lưu tách nhóm trong comparison.json; không gộp rồi gọi đây là bộ test mới.', '',
        '## Ví dụ thực tế', '',
    ])
    for case_id in ('caries_full','caries_focused','splint_full','a2v2_8a9f062d160f202a','a2v2_a23be770bbca145e','a2v2_cdb29c12781be1b8'):
        selected = [r for r in rows if r['id'] == case_id]
        if not selected: continue
        lines.extend([f'### {selected[0]["question"]}', ''])
        for r in sorted(selected,key=lambda r:r['variant']):
            obj = None
            try: obj = parse_json(r['raw_output'])
            except ValueError: pass
            answer = '; '.join(c['text'] for c in obj.get('claims',[])) if isinstance(obj,dict) else r['raw_output']
            if not answer: answer = 'insufficient_evidence (không đủ bằng chứng)'
            lines.extend([f'**{"Gốc" if r["variant"]=="base" else "A2 v3"}:** {answer}',
                          f'Chấm: {r["manual_grade"]}. {r["review"]["note"]}', ''])
    lines.extend([
        '## Diễn giải', '',
        'V3 giảm nhiều ca từ chối khi nguồn thực sự có đáp án và tốt hơn ở phần lớn câu tiếng Việt. Tuy nhiên, với đoạn tiếng Anh, v3 có các lỗi dịch sai thuật ngữ, trả lời chung chung hoặc chọn mức phổ biến thay vì định nghĩa. Bản gốc cũng không phải phương án thay thế tốt ngay: có nhiều ca từ chối không đúng và sai hoặc thiếu trích dẫn.', '',
        'Riêng câu “Sâu răng là gì?”: khi đưa nguyên chunk có nhiều thông tin, cả hai chưa đạt. Khi rút nguồn còn các đoạn trọng tâm, bản gốc trả được cơ chế nhưng dẫn thiếu đoạn hỗ trợ; v3 còn dịch sai “common cold” thành “ho gà”. Như vậy, ngay cả khi đã cấp bằng chứng đúng, A2 vẫn có lỗi đọc nguồn và sinh nội dung. Kết quả này tách lỗi đó khỏi bước truy xuất; không kết luận rằng mọi ca RAG đều đã đúng.', '',
        'Chưa nên gọi A2 v3 là đạt chất lượng trả lời tốt chỉ từ accuracy trạng thái cao. Phép thử này không xác định riêng mức tác động của dữ liệu huấn luyện, prompt, giới hạn model 1.7B hoặc lượng tử hóa; cần thí nghiệm đối chứng thêm nếu muốn phân biệt các nguyên nhân đó. Phiên này chỉ test, không huấn luyện hoặc đổi model tích hợp.', '',
        f'Thời gian lượt GPU (gồm nạp model): {read("run_status.json")["seconds"]:.1f} giây. Không dùng thời gian này để suy ra tốc độ trên laptop.', '',
        'Chi tiết: predictions.jsonl chứa đầu ra nguyên bản; per_case_results.jsonl chứa điểm/nhận xét từng câu; comparison.json chứa bảng tổng hợp, nhóm nguồn và nhóm tình huống; reviews_blind.json và blind_review_lock.json ghi dấu việc khóa chấm trước khi mở nhãn.',
    ])
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'variants': {v:variants[v]['all'] for v in variants}, 'paired':dict(paired_counts),
                      'report':str(OUT/'report.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
