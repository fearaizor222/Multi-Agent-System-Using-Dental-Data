import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent

def scan_file(meta_path: Path):
    issues = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            d = json.loads(line)
            reason = d.get('reason', '').lower()
            ds = d.get('draft_support')
            ps = d.get('public_support')
            
            line_issues = []
            
            # 1. Public support vs reason
            has_confirm = any(w in reason for w in ['xác nhận', 'khẳng định đúng', 'phù hợp', 'trùng khớp', 'chính xác', 'đồng ý'])
            has_contradict = any(w in reason for w in ['mâu thuẫn', 'trái ngược', 'bác bỏ', 'nói ngược', 'sai lệch', 'ngược lại'])
            has_insufficient = any(w in reason for w in ['không đề cập', 'chưa đủ', 'không có thông tin', 'không nói đến', 'không chứa', 'chỉ nói về', 'chỉ cung cấp'])
            
            # Check public mismatches
            if ps == 'contradicted' and has_confirm and not has_contradict:
                line_issues.append(('public_support', 'contradicted', 'supported', 'Reason xác nhận nhưng nhãn là contradicted'))
            elif ps == 'contradicted' and has_insufficient and not has_contradict:
                line_issues.append(('public_support', 'contradicted', 'insufficient', 'Reason nói không đề cập nhưng nhãn là contradicted'))
            elif ps == 'insufficient' and has_confirm and not has_insufficient:
                line_issues.append(('public_support', 'insufficient', 'supported', 'Reason xác nhận nhưng nhãn là insufficient'))
            elif ps == 'supported' and has_contradict and not has_confirm:
                line_issues.append(('public_support', 'supported', 'contradicted', 'Reason mâu thuẫn nhưng nhãn là supported'))
            elif ps == 'supported' and has_insufficient and not has_confirm:
                line_issues.append(('public_support', 'supported', 'insufficient', 'Reason không đề cập nhưng nhãn là supported'))

            # 2. Draft support vs reason
            # Many reasons explain "Nguồn soạn nháp..." or "Nguồn nội bộ..."
            has_draft_insufficient = any(w in reason for w in ['nguồn soạn nháp không', 'nguồn nội bộ không', 'nội bộ không đề cập', 'soạn nháp chưa đủ'])
            has_draft_contradict = any(w in reason for w in ['nguồn soạn nháp mâu thuẫn', 'nguồn nội bộ mâu thuẫn', 'nội bộ nói ngược', 'soạn nháp nói ngược'])
            
            if ds == 'supported' and (has_draft_insufficient or has_draft_contradict):
                line_issues.append(('draft_support', 'supported', 'insufficient/contradicted', 'Reason nói nội bộ không có/mâu thuẫn nhưng draft_support=supported'))
            elif ds == 'insufficient' and not has_draft_insufficient and 'nguồn soạn nháp' in reason:
                line_issues.append(('draft_support', 'insufficient', 'check', 'Reason không rõ ràng về draft_insufficient'))
                
            if line_issues:
                issues.append({
                    'line': idx,
                    'id': d.get('id'),
                    'claim': d.get('claim'),
                    'reason': d.get('reason'),
                    'draft_support': ds,
                    'public_support': ps,
                    'quote_ids': d.get('quote_ids'),
                    'issues': line_issues,
                })
    return issues

def main():
    train_meta = ROOT_DIR / 'data/processed/a3_verification_v1/train.metadata.jsonl'
    val_meta = ROOT_DIR / 'data/processed/a3_verification_v1/val.metadata.jsonl'
    
    train_issues = scan_file(train_meta)
    val_issues = scan_file(val_meta)
    
    print(f'Train issues detected: {len(train_issues)}')
    print(f'Val issues detected: {len(val_issues)}')
    print(f'Total issues: {len(train_issues) + len(val_issues)}')
    print('=' * 80)
    
    print('Samples from Train issues:')
    for item in train_issues[:8]:
        print(f"Line {item['line']} ({item['id']}):")
        print(f"  Claim: {item['claim']}")
        print(f"  Reason: {item['reason']}")
        print(f"  Current: draft={item['draft_support']}, public={item['public_support']}, quotes={item['quote_ids']}")
        for iss in item['issues']:
            print(f"  -> Issue: field={iss[0]}, curr={iss[1]}, suggested={iss[2]}: {iss[3]}")
        print()

    print('Samples from Val issues:')
    for item in val_issues[:8]:
        print(f"Line {item['line']} ({item['id']}):")
        print(f"  Claim: {item['claim']}")
        print(f"  Reason: {item['reason']}")
        print(f"  Current: draft={item['draft_support']}, public={item['public_support']}, quotes={item['quote_ids']}")
        for iss in item['issues']:
            print(f"  -> Issue: field={iss[0]}, curr={iss[1]}, suggested={iss[2]}: {iss[3]}")
        print()

if __name__ == '__main__':
    main()
