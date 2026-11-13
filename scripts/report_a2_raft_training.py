"""Report the verified pilot using paired source-reading grades and automatic metrics."""
from collections import Counter
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'outputs/a2_raft_training_20261112'


def main():
    read=lambda name:json.loads((WORK/name).read_text(encoding='utf-8'))
    verification=read('verification.json')
    reviews=read('manual_review.json')
    packets=[json.loads(x) for x in (WORK/'source_review.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(packets)==32 and set(reviews)==set(map(str,range(32)))
    folder=Path(read('collection.json')['folder'])
    predictions=[json.loads(x) for x in (folder/'paired_probe/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    keyed={(r['id'],r['variant']):r for r in predictions}
    summary={'run':verification['run'],'selected_epoch':verification['selected_epoch'],
        'counts':verification['counts'],'gpu':json.loads((folder/'paired_probe/identity.json').read_text())['gpu'],
        'reviewer':'assistant source-reading, not blinded or independently expert-adjudicated',
        'scope':'32 correlated validation variants from 4 families and 12 questions; no independent test or end-to-end RAG evaluation.',
        'variants':{}}
    reviewed=[]
    for variant in ('v3','pilot'):
        selected=[]
        for packet in packets:
            review=reviews[str(packet['index'])][variant]
            assert review['grade'] in ('pass','partial','fail','unclear')
            row=keyed[packet['id'],variant]
            assert row['raw_output']==packet['outputs'][variant]
            usable=(review['grade']=='pass' and review['vietnamese'] and review['quote_supports_claim'] is not False
                and row['schema_valid'] and row['status_correct'] and row['citation_ids_valid'] and row['quotes_verbatim'])
            selected.append({**row,'review':review,'usable':bool(usable),'language':packet['metadata']['language'],
                'mode':packet['metadata']['mode'],'question':packet['question']})
        metrics=verification['paired_metrics'][variant]
        summary['variants'][variant]={'content_grades':dict(Counter(r['review']['grade'] for r in selected)),
            'content_and_vietnamese':sum(r['review']['grade']=='pass' and r['review']['vietnamese'] for r in selected),
            'usable_count':sum(r['usable'] for r in selected),
            'non_vietnamese_count':sum(not r['review']['vietnamese'] for r in selected),
            'correct_missing_evidence':sum(r['review']['grade']=='pass' for r in selected if r['mode']=='missing_support_same_topic'),
            'missing_evidence_count':sum(r['mode']=='missing_support_same_topic' for r in selected),
            'correct_answerable_content':sum(r['review']['grade']=='pass' for r in selected if r['mode']!='missing_support_same_topic'),
            'answerable_count':sum(r['mode']!='missing_support_same_topic' for r in selected),
            'usable_answerable_count':sum(r['usable'] for r in selected if r['mode']!='missing_support_same_topic'),
            'content_by_language':{lang:dict(Counter(r['review']['grade'] for r in selected if r['language']==lang)) for lang in ('en','vi')},
            'median_seconds':statistics.median(r['seconds'] for r in selected),
            'mean_output_tokens':statistics.mean(r['output_tokens'] for r in selected),
            'token_cap_count':sum(r['output_tokens']>=512 for r in selected),'automatic':metrics}
        reviewed.extend(selected)
    summary['paired_content_transitions']=dict(Counter(
        f'{reviews[str(p["index"])]["v3"]["grade"]} -> {reviews[str(p["index"])]["pilot"]["grade"]}'
        for p in packets))
    summary['recommendation']='Retain production A2 v3; pilot improves refusal but does not improve source-reading content on this validation probe.'
    (WORK/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (WORK/'reviewed_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed),encoding='utf-8')
    old,new=(summary['variants'][v] for v in ('v3','pilot'))
    percent=lambda value:f'{100*value:.1f}%'
    lines=['# Fine-tune thử A2 với dữ liệu RAFT pilot','',
        '**Kết quả: chưa chọn pilot thay A2 v3.** Pilot từ chối tốt hơn khi thiếu ý được hỏi, nhưng giảm số ca trả lời đầy đủ, đúng nội dung. Đây là kết quả của pilot nhỏ, chưa chứng minh kỹ thuật RAFT không phù hợp.', '',
        'Qwen3-1.7B được huấn luyện từ model nền với adapter QLoRA mới, 3 epoch, 144 mẫu train và 72 validation. NF4 double quantization, BF16, LoRA rank 16/alpha 32, learning rate 1e-4, batch hiệu dụng 8, seed 42, chỉ tính loss trên token assistant. Dữ liệu train/validation giữ riêng theo chương, gồm các câu trả lời có trích dẫn nguyên văn, nhiễu và thiếu bằng chứng.', '',
        '144 mẫu train là các biến thể của 24 câu hỏi thuộc 8 family/chương; 72 validation là các biến thể của 12 câu hỏi thuộc 4 family/chương khác. Số mẫu lớn hơn số câu hỏi độc lập. Đây là pilot theo hướng RAFT với trích dẫn và nhiễu, không huấn luyện chuỗi suy luận CoT.', '',
        f'Checkpoint được chọn theo validation loss là epoch {verification["selected_epoch"]:g}. File adapter xuất ra đã được kiểm tra trọng số trùng checkpoint được chọn. Thời gian Trainer chạy, gồm đánh giá loss và lưu checkpoint, là {verification["train_seconds"]:.1f} giây.', '',
        '| Epoch | Validation loss |','|---|---:|']
    for loss in verification['epoch_losses']:
        lines.append(f'| {loss["epoch"]:g} | {loss["eval_loss"]:.4f} |')
    lines.extend(['','## So trực tiếp với A2 v3','',
        f'Hai adapter được nạp trên cùng model nền NF4/BF16, cùng GPU {summary["gpu"]}, prompt yêu cầu supporting_quotes, cùng token đầu vào và greedy decoding tối đa 512 token. Đã đối chiếu hash token cho từng cặp. 32 ca là các biến thể của 12 câu hỏi trong 4 family validation; kết quả này chưa phải test độc lập.', '',
        'Trợ lý đã đọc đối chiếu câu trả lời và nguồn. Chấm nội dung tách khỏi tính nguyên văn của quote và ngôn ngữ đầu ra. Chưa có chuyên gia nha khoa chấm độc lập.', '',
        '| Tiêu chí | A2 v3 | RAFT pilot |','|---|---:|---:|',
        f'| Đạt nội dung và dẫn đúng nguồn | {old["content_grades"].get("pass",0)}/32 | {new["content_grades"].get("pass",0)}/32 |',
        f'| Đạt nội dung ở ca có thể trả lời từ nguồn | {old["correct_answerable_content"]}/24 | {new["correct_answerable_content"]}/24 |',
        f'| Đúng một phần | {old["content_grades"].get("partial",0)}/32 | {new["content_grades"].get("partial",0)}/32 |',
        f'| Đạt nội dung và trả lời tiếng Việt | {old["content_and_vietnamese"]}/32 | {new["content_and_vietnamese"]}/32 |',
        f'| Từ chối đúng khi thiếu ý được hỏi | {old["correct_missing_evidence"]}/8 | {new["correct_missing_evidence"]}/8 |',
        f'| Đúng schema | {percent(old["automatic"]["schema_valid"])} | {percent(new["automatic"]["schema_valid"])} |',
        f'| Quote nguyên văn trong các draft đúng schema | {percent(old["automatic"]["draft_quotes_verbatim_fraction"] or 0)} | {percent(new["automatic"]["draft_quotes_verbatim_fraction"] or 0)} |',
        f'| Đạt nội dung, tiếng Việt, status/schema và quote | {old["usable_count"]}/32 | {new["usable_count"]}/32 |',
        f'| Đạt đủ các yêu cầu trên trong ca có thể trả lời | {old["usable_answerable_count"]}/24 | {new["usable_answerable_count"]}/24 |',
        f'| Thời gian sinh trung vị riêng A2 | {old["median_seconds"]:.2f}s | {new["median_seconds"]:.2f}s |','',
        'Chấm pass khi trả lời đúng ý được hỏi, đủ phần có bằng chứng và không thêm khẳng định sai/không được nguồn hỗ trợ; partial khi có ý đúng nhưng thiếu phần hoặc diễn đạt chưa giữ rõ điều kiện; fail khi chọn sai ý, khẳng định sai, bịa khi thiếu nguồn hoặc từ chối phần vẫn trả lời được. Điểm nội dung dựa trên toàn đoạn mang evidence_id, còn supporting_quotes được chấm riêng. Ba ca pilot 13, 24 và 26 được chấm partial do chưa nêu rõ tiêu chí hoặc thuật ngữ trong nguồn; kể cả nâng cả ba lên pass thì pilot đạt 16/32, vẫn dưới 19/32 của v3.', '',
        'Lỗi đáng chú ý: pilot có lúc trả lời về fluoride cho câu hỏi điều kiện hóa hoạt động, nhầm trục thật với trục tự ý, và trả lời hành vi tích cực thay vì học qua bắt chước. Hai ca chỉ còn đủ bằng chứng cho một nửa câu hỏi bị từ chối toàn bộ. Các lỗi này cho thấy chọn đúng trạng thái chưa đồng nghĩa với trích đúng nội dung.', '',
        'Validation loss thấp nhất ở epoch 1 rồi tăng nhẹ, trong khi train loss tiếp tục giảm; đây là dấu hiệu học quá sát tập train nhỏ. Hai adapter khác nhau cả dữ liệu và quá trình huấn luyện, nên phép so này đánh giá hai bản ứng viên, không tách riêng tác động của kỹ thuật RAFT. V3 chưa được huấn luyện với yêu cầu supporting_quotes; cả hai nhận cùng prompt này trong phép đối chiếu.', '',
        'Quote nguyên văn là phép so chuỗi nghiêm ngặt, có thể không đạt vì model sửa lỗi OCR hoặc dấu ngoặc dù ý nghĩa còn đúng. Một quote có thật trong nguồn vẫn có thể không hỗ trợ nhận định; các ca đó được ghi riêng trong nhận xét đọc nguồn. Thời gian ở đây không bao gồm RAG/reranker và không suy ra độ trễ laptop.', '',
        '## Chỉ tiêu tự động trên đủ 72 validation','',
        '| Chỉ tiêu | Qwen3 nền trước train | Pilot sau train |','|---|---:|---:|'])
    for key,label in [('schema_valid','Đúng schema'),('status_correct','Đúng trạng thái'),('reference_proof_coverage','Phủ trích đoạn tham chiếu (so chuỗi)')]:
        lines.append(f'| {label} | {percent(verification["validation_metrics"]["before"][key])} | {percent(verification["validation_metrics"]["after"][key])} |')
    lines.extend(['','Các chỉ tiêu trên 72 mẫu được tính lại từ đầu ra raw để đối chiếu với log GPU. Chúng không thay thế phần chấm nội dung trên 32 ca. Tỷ lệ quote nguyên văn trong hình before_after có mẫu số khác nhau: model nền chỉ sinh 3 draft, pilot sinh 55 draft.','',
        '## Một số đầu ra để đối chiếu',''])
    for index in (0,9,17,23,28,30):
        packet=packets[index]
        lines.extend([f'**{packet["question"]}**',''])
        for variant in ('v3','pilot'):
            try:
                output=json.loads(packet['outputs'][variant])
                answer='; '.join(c['text'] for c in output.get('claims',[])) or 'insufficient_evidence'
            except (ValueError,KeyError):
                answer=packet['outputs'][variant]
            review=reviews[str(index)][variant]
            lines.extend([f'{variant}: {answer}',f'Chấm {review["grade"]}. {review["note"]}',''])
    lines.extend(['## Tệp kết quả','',
        f'Adapter và biểu đồ được tải về `{folder}`. Các hình loss.png, before_after.png, 04_confusion_matrix.png và 05_f1_by_decision.png là đầu ra của lượt huấn luyện. Hình before_after so model nền với pilot; bảng ở trên so v3 với pilot.', '',
        'verification.json chứa đối chiếu checksum, checkpoint và số liệu; reviewed_results.jsonl chứa raw output và nhận xét từng ca. Model tích hợp trong web vẫn là A2 v3. Chưa chạy bộ test riêng 93 mẫu và chưa đánh giá lại toàn luồng RAG.'])
    (WORK/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
