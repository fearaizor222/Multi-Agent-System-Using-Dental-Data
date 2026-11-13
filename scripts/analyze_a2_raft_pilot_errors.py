"""Read existing pilot outputs and summarize source-reading judgments; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.train_a2 import parse_sample, score_answer, summarize_results

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/a2_training/a2_raft_qwen3_17b_20261112_122501_012229'
WORK = ROOT/'outputs/a2_raft_error_analysis_20261112'
PRIOR = ROOT/'outputs/a2_raft_training_20261112'
MODES = ('clean','noise_before','noise_middle','noise_after','without_evidence',
         'missing_support_same_topic','partial_answer')
TAGS = {'wrong_focus','noise_intrusion','meaning_changed','incomplete_answer',
        'unsupported_in_missing','false_refusal','non_vietnamese','quote_not_supporting'}
LABELS = {'wrong_focus':'Chọn nhầm ý so với câu hỏi',
    'noise_intrusion':'Dùng nội dung nguồn nhiễu khác chủ đề',
    'meaning_changed':'Đổi điều kiện, phủ định hoặc ý nghĩa nguồn',
    'incomplete_answer':'Thiếu ý hoặc điều kiện quan trọng',
    'unsupported_in_missing':'Vẫn trả lời khi ý được hỏi thiếu bằng chứng',
    'false_refusal':'Từ chối phần vẫn có bằng chứng',
    'non_vietnamese':'Nhận định chưa viết bằng tiếng Việt',
    'quote_not_supporting':'Quote không hỗ trợ đầy đủ nhận định'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare():
    WORK.mkdir(exist_ok=True)
    records=read_lines(RUN/'validation.jsonl')
    predictions=read_lines(RUN/'after/predictions.jsonl')
    pairs={r['id']:r for r in read_lines(RUN/'paired_probe/predictions.jsonl') if r['variant']=='pilot'}
    prior_packets=read_lines(PRIOR/'source_review.jsonl')
    previous_reviews=read_json(PRIOR/'manual_review.json')
    previous={r['id']:previous_reviews[str(r['index'])]['pilot'] for r in prior_packets}
    previous_cases={r['id']:r for r in read_json(ROOT/'outputs/a2_raft_prompt_probe_20261112/cases.json')}
    assert len(records)==len(predictions)==72 and len(pairs)==32
    packets=[]
    reusable={}
    rescored=[]
    for index,(record,pred) in enumerate(zip(records,predictions)):
        meta=record['metadata']
        assert meta['id']==pred['sample_id']
        if meta['id'] in previous_cases:
            assert record['messages']==previous_cases[meta['id']]['quote_prompt']['messages']
        payload,gold=parse_sample(record)
        assert gold==pred['expected']
        score=score_answer(record,pred['raw_output'])
        assert all(pred[k]==v for k,v in score.items())
        rescored.append(score)
        same=meta['id'] in pairs and pred['raw_output']==pairs[meta['id']]['raw_output']
        packets.append({'index':index,'id':meta['id'],'metadata':meta,'question':payload['question'],
            'sources':payload['evidence'],'gold':gold,'raw_output':pred['raw_output'],
            'automatic':score,'paired_probe_present':meta['id'] in pairs,
            'same_as_paired_probe':same,
            'paired_probe_raw':pairs.get(meta['id'],{}).get('raw_output')})
        if same:
            reusable[meta['id']]={**previous[meta['id']],
                'review_origin':'reused only for exactly identical paired output; tags require source reading'}
    assert summarize_results(rescored)==read_json(RUN/'metrics.json')['after']
    (WORK/'source_review.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in packets),encoding='utf-8')
    if not (WORK/'manual_review.json').exists():
        write_json(WORK/'manual_review.json',reusable)
    frozen=['validation.jsonl','after/predictions.jsonl','paired_probe/predictions.jsonl','metrics.json','adapter/adapter_model.safetensors']
    assert hashlib.sha256((RUN/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==read_json(PRIOR/'verification.json')['adapter_sha256']
    write_json(WORK/'input_verification.json',{'run':RUN.name,'rows':72,'automatic_metrics_reproduced':True,
        'paired_prompt_messages_identical':True,'adapter_matches_verified_checkpoint':True,
        'overlapping_paired_cases':32,'identical_overlapping_raw_outputs':len(reusable),
        'different_overlapping_raw_outputs':32-len(reusable),
        'input_sha256':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in frozen},
        'new_inference':False,'training_run':False,'independent_test_used':False})
    print(json.dumps({'packets':len(packets),'identical_paired_outputs':len(reusable),'work':str(WORK)}))


def summarize():
    packets=read_lines(WORK/'source_review.jsonl')
    reviews=read_json(WORK/'manual_review.json')
    verification=read_json(WORK/'input_verification.json')
    for name,sha in verification['input_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==sha
    assert set(reviews)=={p['id'] for p in packets} and len(packets)==72
    rows=[]
    for packet in packets:
        review=reviews[packet['id']]
        assert review['grade'] in ('pass','partial','fail')
        assert isinstance(review['vietnamese'],bool)
        assert review['quote_supports_claim'] is None or isinstance(review['quote_supports_claim'],bool)
        assert set(review['error_tags'])<=TAGS and len(set(review['error_tags']))==len(review['error_tags'])
        assert bool(review['note'].strip())
        output=json.loads(packet['raw_output'])
        # Tags for directly observable output-contract errors must be complete.
        assert ('non_vietnamese' in review['error_tags'])==(not review['vietnamese'])
        assert ('quote_not_supporting' in review['error_tags'])==(review['quote_supports_claim'] is False)
        assert ('false_refusal' in review['error_tags'])==(packet['gold']['status']=='draft' and output['status']=='insufficient_evidence')
        assert ('unsupported_in_missing' in review['error_tags'])==(packet['gold']['status']=='insufficient_evidence' and output['status']=='draft')
        if output['status']=='insufficient_evidence':
            assert review['quote_supports_claim'] is None
        else:
            assert review['quote_supports_claim'] is not None
        rows.append({**packet,'review':review})

    def group(items):
        return {'count':len(items),'grades':dict(Counter(r['review']['grade'] for r in items)),
            'content_and_vietnamese':sum(r['review']['grade']=='pass' and r['review']['vietnamese'] for r in items),
            'error_tags':dict(Counter(t for r in items for t in r['review']['error_tags']))}

    indexed={(r['metadata']['question_id'],r['metadata']['mode']):r for r in rows}
    noise_pairs=[]
    for r in rows:
        if not r['metadata']['mode'].startswith('noise_'):continue
        clean=indexed[r['metadata']['question_id'],'clean']
        noise_pairs.append({'question_id':r['metadata']['question_id'],'mode':r['metadata']['mode'],
            'clean_id':clean['id'],'noise_id':r['id'],'clean_grade':clean['review']['grade'],'noise_grade':r['review']['grade'],
            'clean_pass_noise_not_pass':clean['review']['grade']=='pass' and r['review']['grade']!='pass'})
    assert len(noise_pairs)==36
    full_context=[r for r in rows if r['metadata']['mode'] in ('clean','noise_before','noise_middle','noise_after')]
    changed=[]
    prior_packets={r['id']:r for r in read_lines(PRIOR/'source_review.jsonl')}
    prior_reviews=read_json(PRIOR/'manual_review.json')
    for r in rows:
        if r['paired_probe_present'] and not r['same_as_paired_probe']:
            old=prior_reviews[str(prior_packets[r['id']]['index'])]['pilot']
            changed.append({'id':r['id'],'after_grade':r['review']['grade'],'paired_grade':old['grade']})
    summary={'run':RUN.name,'scope':'72 correlated validation variants, 12 questions, 4 chapters; assistant source-reading, not independent clinical adjudication.',
        'all':group(rows),'by_mode':{m:group([r for r in rows if r['metadata']['mode']==m]) for m in MODES},
        'by_family':{f:group([r for r in rows if r['metadata']['family_id']==f]) for f in sorted({r['metadata']['family_id'] for r in rows})},
        'by_source_language':{lang:group([r for r in rows if r['metadata']['language']==lang]) for lang in ('en','vi')},
        'answerable':group([r for r in rows if r['gold']['status']=='draft']),
        'unanswerable':group([r for r in rows if r['gold']['status']=='insufficient_evidence']),
        'full_context_single_question':group([r for r in full_context if len(r['metadata']['question_fact_indices'])==1]),
        'full_context_combined_question':group([r for r in full_context if len(r['metadata']['question_fact_indices'])==2]),
        'noise_pairs':{'count':36,'clean_pass_noise_not_pass':sum(p['clean_pass_noise_not_pass'] for p in noise_pairs),
            'comparisons_with_clean_pass':sum(p['clean_grade']=='pass' for p in noise_pairs),
            'transitions':dict(Counter(p['clean_grade']+' -> '+p['noise_grade'] for p in noise_pairs))},
        'verbatim_quotes_in_predicted_drafts':{'valid':sum(r['automatic']['quotes_verbatim'] for r in rows if json.loads(r['raw_output'])['status']=='draft'),
            'total':sum(json.loads(r['raw_output'])['status']=='draft' for r in rows)},
        'literal_quote_but_unsupported_claim':sum(r['automatic']['quotes_verbatim'] and r['review']['quote_supports_claim'] is False for r in rows),
        'paired_output_differences':verification['different_overlapping_raw_outputs'],
        'different_outputs_review':changed}
    write_json(WORK/'summary.json',summary)
    write_json(WORK/'clean_noise_pairs.json',noise_pairs)
    (WORK/'reviewed_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    lines=['# Phân tích lỗi A2 RAFT pilot trên 72 validation','',
        '**Nhận định:** điểm yếu chính là chọn đúng ý theo câu hỏi và trả lời đủ các phần có nguồn. Lỗi đã xuất hiện khi context sạch, nên chỉ tăng nhiễu trong dữ liệu sẽ chưa giải quyết đủ. Giữ kết quả này làm chẩn đoán cho việc sửa tập pilot; chưa thay A2 v3.', '',
        'Chỉ đọc đầu ra đã có, không chạy thêm inference, GPU hoặc training. Bộ này gồm 12 câu hỏi thuộc 4 chương, tạo thành 72 biến thể context; không phải 72 câu hỏi độc lập và chưa phải bộ test 93 mẫu.', '',
        'Trợ lý đọc đối chiếu toàn bộ 72 câu trả lời với câu hỏi và nguồn. Chấm nội dung riêng với ngôn ngữ và supporting_quotes. Pass: đúng, đủ ý có bằng chứng; partial: đúng một phần, thiếu ý/điều kiện; fail: sai ý, thêm khẳng định không có nguồn hoặc từ chối phần vẫn đủ bằng chứng. Câu trả lời tiếng Anh vẫn có thể pass nội dung, nhưng không đạt yêu cầu tiếng Việt. Quote sửa OCR/dấu câu được tách khỏi sai nội dung. Chưa có chuyên gia chấm độc lập.', '',
        '## Kết quả theo loại tình huống','',
        '| Loại | Số mẫu | Đạt nội dung | Một phần | Không đạt |',
        '|---|---:|---:|---:|---:|']
    for mode,stats in summary['by_mode'].items():
        g=stats['grades'];lines.append(f'| {mode} | {stats["count"]} | {g.get("pass",0)} | {g.get("partial",0)} | {g.get("fail",0)} |')
    a=summary['answerable'];u=summary['unanswerable']
    total=summary['all']['grades']
    lines += ['',f'Tổng: {total.get("pass",0)}/72 đạt đầy đủ nội dung, {total.get("partial",0)}/72 đúng một phần, {total.get("fail",0)}/72 không đạt. Trong 52 ca có thể trả lời, chỉ {a["grades"].get("pass",0)}/52 đạt đầy đủ; trong 20 ca phải từ chối, {u["grades"].get("pass",0)}/20 từ chối đúng. Không diễn giải điểm đúng trạng thái 65/72 là điểm đúng nội dung.', '',
        f'Ở câu hỏi một ý có đủ nguồn: {summary["full_context_single_question"]["grades"].get("pass",0)}/32 đạt đầy đủ. Ở câu hỏi hai ý có đủ nguồn: {summary["full_context_combined_question"]["grades"].get("pass",0)}/16 đạt đầy đủ. Mẫu partial_answer chỉ đạt 1/4, gồm hai lần từ chối toàn bộ và một lần nhầm điều kiện.', '',
        '## Ma trận cùng câu hỏi, thay vị trí context','',
        '| Câu hỏi | Clean | Nhiễu trước | Nhiễu giữa | Nhiễu sau |','|---|---|---|---|---|']
    for qid in sorted({r['metadata']['question_id'] for r in rows}):
        grades=[indexed[qid,m]['review']['grade'] for m in MODES[:4]]
        lines.append('| '+qid+' | '+' | '.join(grades)+' |')
    lines+=['','## Nhóm lỗi','',
        'Một mẫu có thể có nhiều lỗi; tổng các dòng dưới đây không phải số mẫu sai.', '',
        '| Nhóm | Số mẫu |','|---|---:|']
    for tag,count in sorted(summary['all']['error_tags'].items(),key=lambda x:(-x[1],x[0])):
        lines.append(f'| {LABELS[tag]} ({tag}) | {count} |')
    lines+=['','## Tác động quan sát được khi thêm nhiễu','',
        f'Đối chiếu cùng 12 câu hỏi giữa clean và ba vị trí nhiễu tạo 36 cặp. Có {summary["noise_pairs"]["clean_pass_noise_not_pass"]} cặp từ pass ở clean thành partial/fail khi thêm nhiễu, trên {summary["noise_pairs"]["comparisons_with_clean_pass"]} cặp có clean đạt pass. Các cặp dùng chung câu hỏi nên không độc lập. Khi clean vốn đã sai, không quy toàn bộ lỗi ở bản nhiễu cho việc thêm nhiễu. Noise_intrusion chỉ ghi khi đầu ra dùng nội dung của nguồn nhiễu khác chủ đề; chọn nhầm ý trong nguồn chính được ghi wrong_focus.', '',
        '| Clean → Nhiễu | Số cặp |','|---|---:|']
    for transition,count in summary['noise_pairs']['transitions'].items():lines.append(f'| {transition} | {count} |')
    lines+=['','## Theo ngôn ngữ nguồn và nhóm nội dung','',
        '| Nhóm | Số mẫu | Đạt | Một phần | Không đạt |','|---|---:|---:|---:|---:|']
    for key,stats in {**summary['by_source_language'],**summary['by_family']}.items():
        g=stats['grades'];lines.append(f'| {key} | {stats["count"]} | {g.get("pass",0)} | {g.get("partial",0)} | {g.get("fail",0)} |')
    lines+=['','## Diễn giải và thứ tự sửa dữ liệu ở bước sau','',
        '1. Ưu tiên phân biệt các ý gần nhau trong cùng đoạn: đang hỏi điều kiện hay thực trạng, trục thật hay trục tự ý, học qua bắt chước hay khen thưởng/phạt. Có 28 ca chọn sai ý; 7/12 clean không đạt dù đã có nguồn đúng. Tạo thêm câu hỏi độc lập trên các đoạn thuộc train, không chỉ đảo vị trí của 24 câu hỏi cũ.', '',
        '2. Bổ sung bài tập trả lời hai ý và trả lời một phần có bằng chứng. Không có ca nào trong 16 biến thể câu hỏi hai ý với đủ nguồn đạt đầy đủ. Schema cho phép hai nhận định nên đây không phải giới hạn tối đa một câu trả lời. Hai ca partial bị từ chối dù prompt đã yêu cầu trả phần đủ nguồn.', '',
        f'3. Ràng buộc câu trích thực sự hỗ trợ nhận định, giữ điều kiện/phủ định. Có 30 draft có ít nhất một quote không hỗ trợ đầy đủ nhận định; {summary["literal_quote_but_unsupported_claim"]} ca trong số đó vẫn đạt phép kiểm tra quote nguyên văn. Cần phân biệt đúng chuỗi với đúng quan hệ ý nghĩa.', '',
        '4. Sau đó mở rộng nhiễu và vị trí nguồn. Có tác động của nhiễu ở một số câu, nhưng lỗi đọc ý đã tồn tại khi sạch; tăng riêng số mẫu nhiễu không có căn cứ để giải quyết toàn bộ vấn đề.', '',
        '5. Giữ nhận định tiếng Việt và thuật ngữ chính xác. Có 5 đầu ra dùng tiếng Anh cho nhận định; nhiều lỗi nhầm ý cũng xuất hiện trên nguồn tiếng Việt. Hai nhóm ngôn ngữ dùng các chủ đề khác nhau, không có cặp dịch tương đương, nên chưa thể kết luận mọi lỗi trên nguồn tiếng Anh là lỗi dịch hoặc tách riêng ảnh hưởng của ngôn ngữ.', '',
        'Prompt hiện đã yêu cầu chọn đúng thông tin, giữ phủ định/điều kiện, quote hỗ trợ đầy đủ và trả phần còn nguồn. Vì vậy không thể quy kết chỉ do thiếu các câu lệnh này trong prompt. Phép thử cung cấp context trực tiếp, không chạy retrieval: nó chứng minh A2 còn lỗi sau khi đã có đoạn đúng, không chứng minh RAG trong toàn hệ thống đã hết lỗi.', '',
        'Các ví dụ validation chỉ dùng để phân tích. Khi sửa dữ liệu, dùng các chương thuộc train và giữ nguyên ranh giới validation/test; không chép đáp án các ca này vào train. Bước phân tích hiện tại không sửa dữ liệu, prompt, script training hoặc model tích hợp.', '',
        '## Giới hạn đối chiếu với lượt so v3/pilot','',
        f'Trong 32 ca trùng với paired_probe, có {verification["identical_overlapping_raw_outputs"]} đầu ra pilot giống nguyên văn và {verification["different_overlapping_raw_outputs"]} đầu ra khác giữa đánh giá sau train và đánh giá sau nạp lại adapter. Báo cáo này chấm đúng raw của after/predictions.jsonl; không gán điểm cũ cho đầu ra khác. Hai lượt có cùng trọng số adapter và cùng nội dung prompt, nhưng đường nạp model khác; chưa xác định đầy đủ nguyên nhân sai khác số học. Không lấy số liệu 72 ca của pilot so với số liệu 32 ca của v3 như hai tập ngang nhau.', '',
        'Code đánh giá sau train dùng trực tiếp trainer.model, đã đi qua prepare_model_for_kbit_training; đường paired_probe nạp lại base BF16 rồi adapter. PEFT 0.18.1 có bước chuyển các tham số không lượng tử BF16/FP16 sang FP32 khi chuẩn bị train. Đây là một khác biệt cấu hình số học có cơ sở từ code, nhưng chưa có log dtype từng lớp hoặc phép thử cô lập để chốt nguyên nhân 6 đầu ra khác. Trước lần thử tiếp theo nên thống nhất đánh giá bằng đường nạp lại adapter. Không cần train lại để sửa cách đo này.', '',
        '| Ca có đầu ra khác | Chấm after | Chấm paired đã có |','|---|---|---|']
    for r in changed:lines.append(f'| {r["id"]} | {r["after_grade"]} | {r["paired_grade"]} |')
    lines += ['', '## Đầu ra và đáp án tham chiếu của từng ca','',
        'Nguồn đầy đủ, metadata và nhận xét từng ca nằm trong source_review.jsonl / reviewed_results.jsonl cùng thư mục.', '']
    for r in sorted(rows,key=lambda x:(x['metadata']['family_id'],x['metadata']['question_id'],MODES.index(x['metadata']['mode']))):
        review=r['review']
        lines += [f'### {r["id"]}', '',r['question'],'',
            f'Chấm: **{review["grade"]}**. {review["note"]}', '',
            f'Lỗi: {", ".join(review["error_tags"]) or "không"}.','',
            '```json',r['raw_output'],'```','',
            'Đáp án tham chiếu:', '```json',json.dumps(r['gold'],ensure_ascii=False),'```','']
    (WORK/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else summarize()
