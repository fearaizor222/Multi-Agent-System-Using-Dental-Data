"""Summarize local timing spans without adding overlapping nested durations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    folder = args.directory.resolve()
    spans = json.loads((folder/'spans.json').read_text(encoding='utf-8'))
    turns = json.loads((folder/'turn_summary.json').read_text(encoding='utf-8'))
    warmup = next(s['seconds'] for s in spans if s['name']=='warmup')
    summary = {'warmup_seconds':warmup, 'turns':[],
        'scope':'Unchanged config and model weights; real local inference; three isolated questions, last two identical to measure warmed query cache. Small diagnostic sample, not a throughput benchmark.'}
    for turn in turns:
        selected = [s for s in spans if s['phase']==turn['turn']]
        # Only leaf work categories below are summed; retrieve/turn totals overlap.
        totals = defaultdict(float)
        calls = defaultdict(int)
        for s in selected:
            name = s['name']
            if name=='gpu.generate': name = 'gpu.'+s['detail']['role']
            if name not in ('reranker','nli','translation','embedding') and not name.startswith('gpu.A'): continue
            totals[name] += s['seconds']; calls[name] += 1
        assert sum(totals.values()) <= turn['seconds'] + 0.1
        totals['other'] = turn['seconds']-sum(totals.values())
        result = json.loads((folder/(turn['turn']+'.json')).read_text(encoding='utf-8'))
        summary['turns'].append({**turn,'breakdown_seconds':dict(totals),'calls':dict(calls),
            'rerank_pairs':sum(s['detail']['candidates'] for s in selected if s['name']=='reranker'),
            'rerank_fraction':totals.get('reranker',0)/turn['seconds'],
            'gpu_details':[s['detail'] for s in selected if s['name']=='gpu.generate'],
            'knowledge_status':result.get('knowledge',{}).get('status'),
            'synthesis_called':any(s['name']=='gpu.generate' and s['detail'].get('role')=='A4' for s in selected)})
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    names = [('gpu.A1','A1'),('translation','Dịch truy vấn'),('embedding','Embedding'),
             ('reranker','Reranker CPU'),('gpu.A2','A2 sinh nháp'),('nli','NLI CPU'),
             ('gpu.A3','A3 model'),('gpu.A4','Tổng hợp A4, gồm chuyển model'),('other','Phần khác')]
    lines = ['# Đo độ trễ trên máy hiện tại','',f'Khởi động/nạp model: {warmup:.2f} giây, tính riêng với thời gian câu hỏi.',
        '', 'Cấu hình, adapter, ngưỡng và số ứng viên không đổi. Không fine-tune. Các câu dùng lịch sử rỗng trên cùng một tiến trình. Câu máng nhai được hỏi lại để quan sát cache truy vấn; không dùng để ước lượng mọi câu hỏi mới.', '',
        '| Bước | Sâu răng | Máng nhai lần đầu | Máng nhai hỏi lại |','|---|---:|---:|---:|']
    for key,label in names:
        values = [t['breakdown_seconds'].get(key,0) for t in summary['turns']]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2f}s' for v in values)+' |')
    lines.append('| Tổng mỗi câu | '+' | '.join(f'{t["seconds"]:.2f}s' for t in turns)+' |')
    lines.extend(['','Số liệu các bước không cộng trùng thời gian truy xuất với reranker bên trong. 0 ở A3/A4 nghĩa là không được gọi trên nhánh đó, không có nghĩa agent chạy tức thì.', '', '## Đầu ra thực tế',''])
    for turn in summary['turns']:
        lines.extend([f'**{turn["question"]}** — {turn["seconds"]:.2f}s', '',str(turn['answer']),''])
    lines.extend(['## Kết luận từ lượt đo','',
        'Nút thắt lớn nhất trong ba lượt này là BGE reranker chạy CPU float32. Mỗi câu gọi reranker 5 lần, chấm 123–124 cặp truy vấn–đoạn văn (có thể cùng đoạn được chấm với truy vấn khác). A1/A2 sinh trên GPU chỉ chiếm một phần nhỏ hơn của tổng thời gian. Dịch truy vấn sau khi model đã nạp mất dưới một giây; cache bản dịch không loại bỏ chi phí rerank.', '',
        'Với câu máng nhai lần đầu, A2 mất khoảng 9.94 giây trong tổng 85.42 giây. Ngay cả giả định A2 sinh tức thì, phần còn lại vẫn khoảng 75.48 giây. Vì vậy đổi riêng A2 xuống model 0.5B không xử lý nút thắt chính ở những ca đo này.', '',
        'Ưu tiên tiếp theo là thử tối ưu cách thực thi reranker trên CPU và tránh phép chấm trùng hoàn toàn, rồi so sánh thứ hạng/chất lượng nguồn trước–sau. Giảm số ứng viên hoặc thay reranker cần kiểm tra hồi quy truy xuất; không tự tắt A3/NLI để giảm thời gian. Việc đổi model hoặc fine-tune tiếp chưa được thực hiện.', '',
        'Ảnh chụp bộ nhớ trong memory_during_rerank.json cho thấy RAM khả dụng chỉ khoảng 382 MB trong một thời điểm. Đây là dấu hiệu áp lực bộ nhớ trên máy 16 GB, chưa đủ để quy toàn bộ độ trễ cho paging.', ''])
    lines.extend(['## Giới hạn','',
        'Đây là ba lượt đo trên máy đang sử dụng, chịu ảnh hưởng của RAM trống, ứng dụng nền và cache. Nếu một câu kết thúc với thiếu bằng chứng, thời gian đó không đại diện cho một câu được trả lời thành công đủ tất cả agent.',
        'Không dùng kết quả này để khẳng định model mạnh hơn chắc chắn chậm gấp một hệ số nào đó. Không tắt NLI/kiểm chứng để làm đẹp thời gian.', '',
        'Nguồn số liệu: config.json, identity.json, events.jsonl, spans.json, turn_001.json–turn_003.json, summary.json.'])
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
