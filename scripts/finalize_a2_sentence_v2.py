"""Export the source-read pilot with pinned review inputs; never trains a model."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from scripts.a2_sentence_review_policy import NOTES, allowed_keep
from src.training.a2_sentence_protocol import prepare_payload, resolve_selection

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/a2_sentence_selection_v2_review06'
OUTPUT=ROOT/'data/processed/a2_sentence_selection_v2_reviewed'
REVIEWED_CANDIDATE_SHA='3134af92cc8763d0aacb79689bce2978457770ff27ccdff5b03620919dd9ca60'
REVIEWED_POLICY_SHA='4ac86d57f4027e4fa6a9671b3a981bda3bcc266ac8a9650ab60411fb8988653a'

# Additional explicit source-reading notes for families without restrictive
# keep policies. This is an assistant adjudication, not an automatic NLI score.
REVIEW_NOTES={
 'sealants':'Hai cơ chế riêng; câu giảm nguy cơ chung không đủ trả lời cơ chế khi định nghĩa bị bỏ.',
 'leadership':'Hai câu riêng về cam kết phòng ngừa và năng lực lãnh đạo; nhiễu Board không nêu hai lợi ích này.',
 'splint_shape':'Tác động hở khớp của máng phẳng/lõm nằm ở hai câu riêng; hình chỉ nêu loại máng.',
 'splint_duration':'Giữ nói chung/cấp tính 6–8 tuần và tạm thời/định kỳ/mạn; nhiễu thuật ngữ không nêu thời gian.',
 'bennett':'Phân biệt yếu tố hướng vận động với thời điểm tức thì/tăng tiến; chú thích chỉ nêu tên hai loại.',
 'bottle_label':'Tách bảng, nghĩa vụ ghi chất và không bắt buộc ghi nồng độ; nhiễu learning không hỗ trợ ghi nhãn.',
 'reversals':'Mỗi hàng gồm định nghĩa và ví dụ đúng khái niệm, hai nhãn không còn cùng khối.',
 'hypotheses':'Chỉ dùng ví dụ Brand X; các giả thuyết về cạo vôi không trả lời được ví dụ này.',
 'payment_systems':'Tách hai/ba bên và phần văn bản thân bài sau bảng; giữ đúng bệnh nhân/bên bảo trợ trả tiền.',
 'socioeconomic_status':'Định nghĩa SES riêng với tần suất sâu răng; tương quan chung không chứng minh chiều tăng của sâu chưa trị.',
 'presentation_visuals':'Tách heading sau bullet một dòng; giữ lý do làm chậm đọc và giới hạn một dòng khi có thể.',
 'providers_research':'Nêu đủ loại hàm tháo lắp và phạm vi nghiên cứu oral health; nguồn McKinney không thay được hai định nghĩa.',
 'lesson_literature':'Hai mục glossary riêng; không dùng dòng licensure hoặc lobbying lân cận làm bằng chứng.',
 'qualitative_quantitative':'Giữ why/how đối lập numerical; không nhầm các mục measurement ở lân cận.',
 'survey_methods':'Mã hàng BRFSS, NHANES, BSS, NHIS tách biệt; NHANES chọn cả câu xác định đối tượng và phỏng vấn/khám.',
 'research_confidentiality':'Giữ never identified và may ask consent for photos; không suy nghĩa vụ mới từ quy định consent khác.',
 'splint_materials':'Nêu theo đoạn sách, không coi hiện nay trong sách là khuyến nghị cập nhật; giữ acrylic và ưu điểm kim loại riêng.',
 'occlusal_indications':'Hai mục đánh số giữ riêng: tiếp xúc sớm TQTT và quá mức LMTĐ; OCR dư được giữ nguyên quote.',
 'supporting_cusps':'Phân biệt cối lớn/cối nhỏ và trũng giữa/tam giác xa; không lấy bullet múi xa trong lân cận.',
 'splint_borders':'Giữ không chèn ép nướu và không bắt chéo gờ khẩu cái, hai vị trí khác nhau.',
 'functional_occlusion':'Giữ hài hòa hình thái/chức năng và không cản trở trượt; các tiêu chuẩn không đau khác không thay được.',
 'bonwill':'Giữ mốc 1855 và mô tả tam giác đều 10 cm/ba đỉnh; không lấy định nghĩa khớp cắn thăng bằng.',
 'functional_parafunctional':'Giữ mục đích chức năng đối lập không nhằm chức năng; ý thức/tiếp xúc là có hoặc không.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_candidate(row):
    if not allowed_keep(row['family_id'], row['kept_fact_indices']):
        raise ValueError('Unreviewed keep set')
    user=json.JSONDecoder().raw_decode(row['messages'][1]['content'])[0]
    payload=dict(question=user['question'],evidence=row['original_evidence'])
    if prepare_payload(payload)!=user:
        raise ValueError('Input/source mismatch')
    gold=json.loads(row['messages'][2]['content'])
    resolved=resolve_selection(gold,payload)
    if resolved.status!=row['proposed_status']:
        raise ValueError('Status mismatch')
    return payload,gold,resolved


def finalize(source=SOURCE, output=OUTPUT):
    if output.exists():raise ValueError('Refusing to overwrite reviewed snapshot')
    if sha(source/'candidates.jsonl')!=REVIEWED_CANDIDATE_SHA:
        raise ValueError('Candidate bytes changed; source review is no longer applicable')
    if sha(ROOT/'scripts/a2_sentence_review_policy.py')!=REVIEWED_POLICY_SHA:
        raise ValueError('Mask/review policy changed; re-review required')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol_sha256']!=sha(ROOT/'src/training/a2_sentence_protocol.py'):
        raise ValueError('Protocol changed after candidate generation')
    seed_path=ROOT/'data/processed/a2_raft_pilot_v2/seed_review.json'
    if manifest['source_sha256']!=sha(seed_path):raise ValueError('Source changed')
    seeds=json.loads(seed_path.read_text(encoding='utf-8'))
    notes={**REVIEW_NOTES,**NOTES}
    if {s['id'] for s in seeds}!=set(notes):raise ValueError('Missing family adjudication')
    rows=[json.loads(x) for x in (source/'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=858:raise ValueError('Review snapshot changed; re-review required')
    groups={}; splits=defaultdict(list); ids=set(); docs=defaultdict(set)
    for row in rows:
        if row['id'] in ids:raise ValueError('Duplicate ID')
        ids.add(row['id']);_,gold,resolved=validate_candidate(row)
        # Different rotations must express identical content and select the same
        # source texts, regardless of remapped evidence IDs.
        key=(row['family_id'],tuple(row['kept_fact_indices']),tuple(row['question_fact_indices']))
        signature=(gold['status'],tuple((c.text,tuple(q.quote for q in c.supporting_quotes)) for c in resolved.claims))
        if key in groups and groups[key]!=signature:raise ValueError('Rotation label conflict')
        groups[key]=signature
        seed=next(s for s in seeds if s['id']==row['family_id'])
        partner=next(s for s in seeds if s['id']==row['distractor']['owner'])
        if seed['split']!=row['split'] or partner['split']!=row['split']:raise ValueError('Cross-split context')
        docs[row['split']].update(s['document_id'] for s in seed['sources'].values())
        docs[row['split']].update(s['document_id'] for s in partner['sources'].values())
        splits[row['split']].append(row)
    if docs['train'] & docs['val']:raise ValueError('Train/val source document overlap')
    output.mkdir(parents=True)
    for split,items in splits.items():
        (output/(split+'.jsonl')).write_text(''.join(json.dumps({'messages':r['messages']},ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
        (output/(split+'.metadata.jsonl')).write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='messages'} | {
            'requires_source_review':False,'review_note':notes[r['family_id']],
            'review_method':'assistant source-reading by family/keep-set, rotations verified structurally; not clinical expert'},ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    reviews=[dict(family_id=s['id'],note=notes[s['id']],
        source_sha256={k:hashlib.sha256(v['content'].encode()).hexdigest() for k,v in s['sources'].items()}) for s in seeds]
    (output/'source_review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(version='a2_sentence_selection_v2_source_reviewed',contract='a2_sentence_selection_v1',
        counts={k:len(v) for k,v in splits.items()},statuses={k:dict(Counter(r['proposed_status'] for r in v)) for k,v in splits.items()},
        source_review_complete=True,reviewer='assistant, not clinical expert',
        review_scope='44 source families and allowed keep-sets; 858 rotations/questions structurally verified, not 858 independent manual judgments',
        data_ready_for_pilot=True,runner_compatible=False,training_ready=False,
        candidate_sha256=sha(source/'candidates.jsonl'),policy_sha256=sha(ROOT/'scripts/a2_sentence_review_policy.py'),
        protocol_sha256=manifest['protocol_sha256'],seed_sha256=manifest['source_sha256'],
        validation='84 new development variants from four existing families; NOT the old 72 validation or independent test',
        limitations=['Same 80 train facts, 120 formulations; no new independent clinical coverage.',
            'Synthetic unit-removal contexts; not a substitute for actual failed RAG retrievals.',
            'Some negatives use other-topic full chunks; same-topic coverage remains limited.',
            'No clinical expert validation; semantic judgments may be revised.'],
        training_executed=False,test_examples_used=False,
        files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(finalize(),ensure_ascii=False,indent=2))
