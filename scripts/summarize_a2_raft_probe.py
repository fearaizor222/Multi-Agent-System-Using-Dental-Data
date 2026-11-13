"""Combine paired prompt outputs, quote checks and manual source-reading reviews."""
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts.train_a2 import score_answer, summarize_results

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/a2_raft_prompt_probe_20261112'


def main():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
    assert read('run_status.json')['status']=='completed'
    cases=read('cases.json')
    reviews=read('manual_review.json')
    predictions=[json.loads(x) for x in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    keyed={(r['id'],r['variant']):r for r in predictions}
    assert len(keyed)==len(predictions)==64 and len(cases)==32 and set(reviews)==set(map(str,range(32)))
    assert read('identity.json')['adapter_sha256']=='47f476ce9ee66b4568694e2c8b6b7e696a38aab0dc756d871d768476974a774d'
    actual={m['id']:r['messages'] for r,m in zip(
        [json.loads(x) for x in (ROOT/'data/processed/a2_raft_pilot_v1/val.jsonl').read_text(encoding='utf-8').splitlines()],
        [json.loads(x) for x in (ROOT/'data/processed/a2_raft_pilot_v1/val.metadata.jsonl').read_text(encoding='utf-8').splitlines()])}
    rows=[]
    for index,case in enumerate(cases):
        assert case['quote_prompt']['messages']==actual[case['id']]
        inputs=[json.loads(case[v]['messages'][1]['content'].split('\nJSON schema:\n')[0]) for v in ('old_prompt','quote_prompt')]
        assert inputs[0]==inputs[1]
        for variant in ('old_prompt','quote_prompt'):
            prediction=keyed[case['id'],variant]
            review=reviews[str(index)][variant]
            score=score_answer(case[variant],prediction['raw_output'])
            usable=(score['schema_valid'] and score['status_correct'] and score['citation_ids_valid']
                    and review['grade']=='pass' and review['vietnamese'])
            if variant=='quote_prompt':
                usable=usable and score['quotes_verbatim'] and review['quote_supports_claim'] is not False
            rows.append({**prediction,'index':index,'question':inputs[0]['question'],
                'language':case['metadata']['language'],'family':case['metadata']['family_id'],
                'mode':case['metadata']['mode'],'automatic':score,'source_reading':review,'usable':bool(usable)})
    summary={'count':32,'generations':64,'identity':read('identity.json'),
        'scope':'4 validation chapters/families, 12 questions and correlated context variants. Assistant source-reading, not blinded or independently expert-adjudicated. No training.',
        'variants':{}}
    for variant in ('old_prompt','quote_prompt'):
        selected=[r for r in rows if r['variant']==variant]
        aggregate=summarize_results([r['automatic'] for r in selected])
        aggregate.update(content_grades=dict(Counter(r['source_reading']['grade'] for r in selected)),
            content_pass_vietnamese_count=sum(r['source_reading']['grade']=='pass' and r['source_reading']['vietnamese'] for r in selected),
            usable_count=sum(r['usable'] for r in selected),
            non_vietnamese_count=sum(not r['source_reading']['vietnamese'] for r in selected),
            median_seconds=statistics.median(r['seconds'] for r in selected),
            mean_output_tokens=statistics.mean(r['output_tokens'] for r in selected),
            token_cap_count=sum(r['output_tokens']>=512 for r in selected),
            content_by_language={lang:dict(Counter(r['source_reading']['grade'] for r in selected if r['language']==lang)) for lang in ('vi','en')},
            content_by_mode={mode:dict(Counter(r['source_reading']['grade'] for r in selected if r['mode']==mode)) for mode in sorted({r['mode'] for r in selected})})
        summary['variants'][variant]=aggregate
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'reviewed_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    old,new=(summary['variants'][v] for v in ('old_prompt','quote_prompt'))
    percent=lambda f:f'{100*f:.1f}%'
    lines=['# Thử prompt chọn bằng chứng trên A2 v3','',
        'Đã chạy hai prompt trên cùng Qwen3-1.7B + adapter A2 v3 epoch 2, 32 ca mỗi prompt. Câu hỏi và các đoạn nguồn giống nhau. Prompt hiện tại yêu cầu claims/evidence_ids; prompt mới thêm supporting_quotes trong cùng lượt sinh. NF4, greedy, tắt thinking, tối đa 512 token, không sửa hoặc retry đầu ra.', '',
        '32 ca là biến thể của 12 câu hỏi trong 4 family/chương validation, gồm tiếng Anh và tiếng Việt. Không có huấn luyện trong lượt thử. Trợ lý đọc nguồn để chấm nội dung, biết loại prompt, chưa có chuyên gia chấm độc lập.', '',
        '| Tiêu chí | Prompt hiện tại | Prompt kèm câu trích |','|---|---:|---:|',
        f'| Đạt nội dung và dẫn đúng đoạn nguồn | {old["content_grades"].get("pass",0)}/32 | {new["content_grades"].get("pass",0)}/32 |',
        f'| Đúng một phần | {old["content_grades"].get("partial",0)}/32 | {new["content_grades"].get("partial",0)}/32 |',
        f'| Đạt nội dung và trả lời tiếng Việt | {old["content_pass_vietnamese_count"]}/32 | {new["content_pass_vietnamese_count"]}/32 |',
        f'| Đúng schema | {percent(old["schema_valid"])} | {percent(new["schema_valid"])} |',
        f'| Đúng quyết định trả lời/từ chối | {percent(old["status_correct"])} | {percent(new["status_correct"])} |',
        f'| Nhận định còn tiếng Anh | {old["non_vietnamese_count"]} | {new["non_vietnamese_count"]} |',
        f'| Đủ yêu cầu để sử dụng theo từng schema | {old["usable_count"]}/32 | {new["usable_count"]}/32 |',
        f'| Thời gian sinh trung vị trên {summary["identity"]["gpu"]} | {old["median_seconds"]:.2f}s | {new["median_seconds"]:.2f}s |','',
        'Điểm nội dung kiểm tra câu trả lời và đoạn được dẫn, tách khỏi câu quote bổ sung. Cột đủ yêu cầu sử dụng còn yêu cầu tiếng Việt, schema và status đúng; với prompt mới, quote phải nguyên văn và hỗ trợ nhận định. Hai cột có yêu cầu đầu ra khác nhau nên không dùng riêng tỷ lệ sử dụng để xếp hạng năng lực đọc hiểu.', '',
        f'Trong {new["quoted_draft_count"]} đầu ra draft đúng schema của prompt mới, tỷ lệ toàn bộ quote khớp nguyên văn là {percent(new["draft_quotes_verbatim_fraction"] or 0)}. Một số quote chỉ sửa OCR/dấu ngoặc nên sai kiểm tra nguyên văn dù đúng nghĩa. Các lỗi khác dịch quote sang tiếng Việt hoặc chọn câu không hỗ trợ nhận định.', '',
        'Tỷ lệ phủ câu trích tham chiếu là phép so chuỗi, không phải điểm hiểu nghĩa. Một câu trích nguyên văn vẫn có thể không hỗ trợ nhận định. Kết quả đọc nguồn theo từng ca nằm ở reviewed_results.jsonl.', '',
        '## Các ca cần chú ý','',
        'Ca learning_q0_missing_support_same_topic: model mới lấy hai câu có thật về modeling và nhãn fluoride, rồi khẳng định điều kiện hóa hoạt động dựa trên cả hai. Các quote khớp nguồn nhưng quan hệ được viết ra không có căn cứ.', '',
        'Ca hinge_axis_q1_clean: cả hai prompt đều nhầm trục bản lề tự ý với trục thật dù hai định nghĩa cùng có trong context.', '',
        'Ca bottle_label_q0_clean: prompt mới chọn đúng điều kiện liệt kê fluoride, nhưng nhận định trả bằng tiếng Anh. Một số ca khác model dịch phần quote sang tiếng Việt, làm mất tính nguyên văn.', '',
        '## Phần bộ đọc và đánh giá đã chuẩn bị','',
        'Script train_a2 đọc được tập pilot qua --dataset-dir, xác thực provenance và schema supporting_quotes. Hàm encode_sample đã kiểm tra đủ 144 train và 72 validation với tokenizer Qwen3 thật; chỉ token assistant được tính loss và không cắt mẫu. Các chỉ tiêu quote, mã nguồn và độ phủ proof được ghi riêng với chỉ tiêu status/schema. Các kiểm tra cũ tiếp tục chạy được.', '',
        'Các bước này chuẩn bị cho thử QLoRA tiếp theo. Lượt thử prompt chưa chứng minh dữ liệu RAFT cải thiện model, vì adapter đang dùng vẫn là A2 v3. Kết quả cho thấy cần huấn luyện và đánh giá trực tiếp việc chọn ý, sao chép câu trích, dịch nhận định và từ chối khi ý được hỏi vắng mặt.', '',
        'Thời gian GPU cloud chỉ dùng so hai prompt trong lượt này, không suy ra thời gian trên laptop hoặc cả hệ thống. Số mẫu nhỏ và có nhiều biến thể cùng nguồn.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'variants':summary['variants'],'report':str(OUT/'report.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
