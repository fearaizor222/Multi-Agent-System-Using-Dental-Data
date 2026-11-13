"""Apply source-read corrections from the November 12 A2 audit. No model calls."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from scripts.train_a2 import parse_sample
from scripts.a2_dataset_v2 import load_v2, canonical_sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/a2_drafting_v3'
OUTPUT = ROOT / 'data/processed/a2_drafting_v4'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def main():
    load_v2(SOURCE, parse_sample)
    OUTPUT.mkdir(exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError('Output directory must be empty')
    data = read_rows(SOURCE/'train.jsonl')
    metadata = read_rows(SOURCE/'train.metadata.jsonl')
    lookup = {meta['id']:(record, meta) for record, meta in zip(data, metadata)}
    queued = {r['id'] for r in read_rows(ROOT/'outputs/a2_data_audit_20261112/revision_queue.jsonl')}
    hold_families = {
        'a2v2_7df091a52e91b189': 'Trắc nghiệm thiếu khóa đáp án.',
        'a2v2_ae35fed38a20c73f': 'Trắc nghiệm thiếu khóa đáp án.',
        'a2v2_c6473602677b8f2f': 'Câu OCR tiếng Anh không rõ chủ thể.',
        'a2v2_532d6a2de0a325bc': 'Trắc nghiệm thiếu khóa đáp án.',
        'a2v2_88328ecc883e24db': 'Khóa đáp án được cấp không bao gồm câu hỏi 44.',
        'a2v2_bb1797cedf613ac0': 'Trắc nghiệm thiếu khóa đáp án.',
        'exp_multi_chunk-03_chunk-60': 'Một nhận định lấy lựa chọn trắc nghiệm chưa có đáp án.',
        'exp_dir_chunk-d221': 'Câu chứa thống kê bị cắt trước phần cần diễn giải.',
        'tgt_multi_chunk-87_chunk-3b': 'Các lựa chọn về cơ nâng/hạ hàm thiếu khóa đáp án.',
        'tgt_dir_chunk-87b2': 'Lựa chọn cơ nâng hàm chưa có khóa đáp án.',
        'tgt_dir_chunk-587b': 'Bảng thuật ngữ OCR thiếu cặp đối chiếu rõ.',
        'a2v2_e059ef64b38c5da0': 'Diễn giải xác suất p-value trong nguồn cần duyệt thống kê trước khi dùng làm đáp án huấn luyện.',
    }
    kept, metas, changes, quarantine = [], [], [], []
    for original, old_meta in zip(data, metadata):
        row, meta = deepcopy(original), deepcopy(old_meta)
        ident = meta['id']
        before = {'messages_sha256':canonical_sha(original['messages']), 'metadata_sha256':canonical_sha(old_meta)}
        if meta['family_id'] in hold_families:
            assert ident in queued
            reason = hold_families[meta['family_id']]
            quarantine.append({'id':ident, 'family_id':meta['family_id'], 'reason':reason,
                               'record':original, 'metadata':old_meta})
            changes.append({'id':ident, 'action':'quarantine', 'reason':reason, **before})
            continue
        if ident not in queued:
            kept.append(row); metas.append(meta)
            continue
        payload, gold = parse_sample(row)
        payload, gold = deepcopy(payload), deepcopy(gold)
        if ident == 'a2v2_39022361e359ccee':
            key_row, key_meta = lookup['a2v2_88328ecc883e24db']
            key_payload, _ = parse_sample(key_row)
            assert key_meta['document_id'] == meta['document_id']
            key_source = deepcopy(key_payload['evidence'][0])
            key_source['evidence_id'] = 'I3'
            payload['evidence'].append(key_source)
            payload['allowed_evidence_ids'].append('I3')
            meta['chunk_ids'].append(key_meta['chunk_ids'][0])
        sources = {e['evidence_id']:' '.join(e['content'].split()) for e in payload['evidence']}

        def proof(eid, start, end=None):
            text = sources[eid]
            first = text.index(start)
            last = text.index(end, first)+len(end) if end else first+len(start)
            return {'evidence_id':eid, 'quote':text[first:last]}

        def set_claims(items):
            meta['proofs'] = []
            for text, quotes in items:
                meta['proofs'].append({'text':text, 'evidence_ids':list(dict.fromkeys(q['evidence_id'] for q in quotes)), 'proofs':quotes})

        reason = 'Chuẩn hóa đáp án tiếng Việt và đối chiếu lại câu hỗ trợ.'
        if ident == 'a2v2_d2d6a8ae009c302d':
            payload['question'] = 'Theo tên nghiên cứu được trích trong tài liệu, dữ liệu về tật hở môi và vòm miệng tại British Columbia thuộc giai đoạn nào?'
            set_claims([('Theo tên nghiên cứu được trích, giai đoạn được đề cập là từ năm 1952 đến năm 1986.', [proof('I1','Birth prevalence of cleft','Stability of rates.')])])
        elif meta['family_id'] == 'a2v2_4dee6a577c0ad715':
            items = [('Ở tư thế ăn tỉnh thức, hàm dưới hơi nghiêng về phía trước so với tư thế nghỉ khi đầu thẳng đứng.', [proof('I2','Ở tư thế này hàm dưới','đầu thẳng đứng.')])]
            if ident.endswith('_partial_answer'):
                items.append(('Tư thế ăn tỉnh thức có liên quan có ý nghĩa với những tương quan chức năng của răng.', [proof('I2','Tư thế ăn tỉnh thức (alert feeding position) có liên quan','chức năng của răng.')]))
            set_claims(items)
            meta['mode'] = 'direct'
        elif ident == 'a2v2_f56c0ac73cb2cb11':
            payload['question'] = 'Theo tài liệu, hội chứng Sjögren được nêu là bệnh mạn tính có biểu hiện chủ yếu ở vùng nào?'
            set_claims([('Hội chứng Sjögren được nêu là một bệnh mạn tính có biểu hiện chủ yếu ở vùng miệng.', [proof('I2','Some chronic diseases,','primary oral symptoms.')])])
            meta['mode'] = 'direct'
            reason = 'Câu hỏi đã thu hẹp theo đúng thông tin nguồn, tránh đòi triệu chứng cụ thể không được cung cấp.'
        elif ident == 'a2v2_39022361e359ccee':
            payload['question'] = 'Theo đáp án câu hỏi 40 trong tài liệu, công ty sản xuất găng tay nên chọn hành động nào trước tỷ lệ nhạy cảm latex cao và chi phí găng tay latex thấp?'
            set_claims([('Đáp án được chọn là tiếp tục sản xuất găng tay latex và cung cấp thêm một loại găng tay thay thế.', [proof('I2','40. What is the best action','c. Continue making latex gloves and offer an alternative type of glove'), proof('I3','40. (c)')])])
            reason = 'Loại phương án b sai; bổ sung chunk có khóa đáp án 40.(c) từ cùng chương trong train làm I3. Context ban đầu không chứa khóa đáp án câu 40.'
        elif ident == 'a2v2_48e53006d27bf0e0':
            set_claims([('Theo đuổi nghề nghiệp trong y tế công cộng giúp tăng năng lực lãnh đạo và khả năng thay đổi nghề nghiệp theo hướng có lợi hơn cho cộng đồng.', [proof('I2','Pursuing a career in public health','beneficial for the public.')])])
        elif ident == 'exp_multi_chunk-16_chunk-a0':
            payload['question'] = 'Vai trò ban đầu của chuyên viên vệ sinh răng miệng và các yếu tố góp phần giảm tỷ lệ bệnh răng miệng được mô tả như thế nào?'
            set_claims([
                ('Vai trò của chuyên viên vệ sinh răng miệng được hình thành để phát hiện và điều trị nhiễm trùng răng miệng trước khi quá nặng, chuyển bệnh nhân đến nhân viên y tế phù hợp và phòng bệnh răng miệng trong dân số.', [proof('I1','The role of the dental hygienist was created','dental disease in all populations.')]),
                ('Theo tài liệu, chăm sóc vệ sinh răng miệng chuyên nghiệp, fluoride và chất trám bít hố rãnh là những yếu tố góp phần nhiều nhất vào việc giảm tỷ lệ bệnh răng miệng.', [proof('I2','The introduction of professional','reduction in oral disease rates.')]),
            ])
        elif ident == 'exp_multi_chunk-17_chunk-70':
            set_claims([
                ('Dịch tễ học nghiên cứu số lượng, sự phân bố, các yếu tố quyết định và việc kiểm soát bệnh cùng các tình trạng sức khỏe trong một dân số nhất định.', [proof('I1','Epidemiology is the study','among a given population.')]),
                ('Bác sĩ Sara Baker xây dựng các chương trình giúp những người nghèo trong thành phố giữ cho trẻ sơ sinh của họ khỏe mạnh.', [proof('I2','Baker established many programs','their infants healthy.')]),
            ])
        elif ident == 'exp_multi_chunk-40_chunk-25':
            set_claims([
                ('Tại Viện Y tế Quốc gia, cô McKinney điều trị cho bệnh nhân có tình trạng sức khỏe phức tạp hoặc bệnh cấp tính và giáo dục sức khỏe răng miệng cho nhân viên y tế.', [proof('I1','She treated medically compromised','health care providers.')]),
                ('Các kỹ năng thiết yếu được nêu gồm hiểu biết về thống kê, nghiên cứu, ngân sách và quản lý chương trình.', [proof('I2','Familiarity with statistics,','public health sector.')]),
            ])
        elif ident == 'exp_multi_chunk-a4_chunk-6a':
            payload['question'] = 'Theo tài liệu, chuyên viên vệ sinh răng miệng có thể thay đổi hành vi nguy cơ và tham gia phát hiện ung thư miệng như thế nào?'
            set_claims([
                ('Chuyên viên vệ sinh răng miệng có thể hỗ trợ thay đổi hành vi nguy cơ bằng tư vấn ngừng hút thuốc, ngừng uống rượu và khuyến khích dinh dưỡng lành mạnh.', [proof('I1','Dental hygienists may apply','promoting healthy nutrition.')]),
                ('Ở một số quốc gia, chuyên viên vệ sinh răng miệng được phép thực hiện sinh thiết bằng bàn chải ít xâm lấn trên tổn thương miệng nghi ngờ, kết hợp phân tích hàm lượng DNA nhân bằng máy tính để hỗ trợ phát hiện ung thư miệng.', [proof('I1','In addition, in some countries,','suspicious oral lesions.')]),
            ])
            reason = 'Dịch tiếng Việt, giữ điều kiện ở một số quốc gia và kỹ thuật kết hợp; thu hẹp câu hỏi vào hai ý có bằng chứng rõ.'
        elif ident == 'exp_multi_chunk-78_chunk-73':
            set_claims([
                ('Chuyên viên vệ sinh răng miệng có thể đảm nhận vai trò tác nhân thay đổi và vận động cho việc chăm sóc sức khỏe răng miệng hiệu quả ở các nhóm dân cư chưa được phục vụ đầy đủ.', [proof('I1','Identify four roles','agent of change'), proof('I1','Advocate for effective oral health care for underserved populations')]),
                ('Thay đổi y tế công cộng nha khoa cần là mối quan tâm của chuyên viên vệ sinh răng miệng; họ phải vận động cho sức khỏe bệnh nhân thông qua thay đổi hành vi, quy trình phòng khám, sản phẩm và các hệ thống pháp lý, quản lý.', [proof('I2','Changing dental public health','legislative and regulatory systems.')]),
            ])
        elif ident == 'exp_multi_chunk-91_chunk-2b':
            set_claims([
                ('Đề xuất tài trợ tốt cần có bố cục rõ, vấn đề được nghiên cứu và ghi nhận đầy đủ, chiến lược sáng tạo, mục tiêu khả thi và đo lường được cùng kế hoạch đánh giá phù hợp.', [proof('I1','Features of a strong proposal','Inclusion of a sound evaluation plan')]),
                ('Các thành phần tiêu chuẩn gồm thư giới thiệu, trang tiêu đề và tóm tắt, trình bày nhu cầu hoặc vấn đề, mô tả dự án, kế hoạch đánh giá, dự toán và giải trình ngân sách, cùng năng lực của đơn vị đề xuất.', [proof('I2','Table 7–2 Standard Components','requesting organization’s project.')]),
            ])
        elif ident == 'exp_dir_chunk-a724':
            payload['question'] = 'Theo tài liệu, thời lượng chương trình đào tạo chuyên viên vệ sinh răng miệng ở Hà Lan đã thay đổi theo trình tự nào?'
            set_claims([('Chương trình trước đó đã đổi từ hai năm lên ba năm, rồi tiếp tục chuyển thành khóa học bốn năm.', [proof('I1','The program (which had been changed','four-year course of study.')])])
            reason = 'Sửa trình tự hai lên ba rồi bốn năm; loại cách diễn đạt hai hoặc ba năm.'
        elif meta['family_id'] == 'exp_dir_chunk-5588':
            payload['question'] = 'Theo đoạn tài liệu được cung cấp, Medicare thường có chi trả chăm sóc nha khoa không và có ngoại lệ nào được nêu?'
            set_claims([('Theo tài liệu, Medicare nhìn chung không tài trợ chăm sóc nha khoa; trong một số trường hợp cụ thể, chương trình có thể hoàn trả chi phí thủ thuật nha khoa thực hiện tại bệnh viện cho người cao tuổi hoặc người khuyết tật.', [proof('I1','In specific cases, Medicare','does not fund dental care.')])])
        elif ident == 'exp_dir_chunk-8ccc':
            set_claims([('Y tế công cộng nha khoa tập trung vào chăm sóc và giáo dục sức khỏe răng miệng cho một dân số, nhấn mạnh việc vận dụng các khoa học về vệ sinh răng miệng.', [proof('I1','Dental public health focuses','dental hygiene sciences.')])])
        elif ident == 'tgt_dir_chunk-15b7':
            payload['question'] = 'Theo đáp án câu hỏi 2 trong tài liệu, cần nhận được sự hỗ trợ từ ai để tránh các phản ứng tiêu cực và xung đột?'
            set_claims([('Cần có sự hỗ trợ của y tá trường học và giáo viên.', [proof('I1','2. Possible negative reactions','c. Support of the school nurse and teachers'), proof('I1','Answers: 1. (d) 2. (c)')])])
            reason = 'Bổ sung khóa đáp án để xác lập phương án c, không chỉ trích lựa chọn.'
        else:
            raise ValueError('Queued sample not handled: '+ident)

        if 'conversation' in payload:
            payload['conversation']['routing_hint'] = payload['question']
        gold = {'status':'draft', 'claims':[{k:c[k] for k in ('text','evidence_ids')} for c in meta['proofs']]}
        if meta['mode'] not in ('distractor', 'partial_answer', 'injection'):
            cited_ids = {i for c in gold['claims'] for i in c['evidence_ids']}
            meta['mode'] = 'multi_source' if len(cited_ids) > 1 else 'direct'
        row['messages'][1]['content'] = json.dumps(payload, ensure_ascii=False)+'\nJSON schema:\n'+original['messages'][1]['content'].split('\nJSON schema:\n')[1]
        row['messages'][2]['content'] = json.dumps(gold, ensure_ascii=False)
        for key in ('review','review_record','review_status','binding_sha256','blind_record','blind_answerability','natural_question_review','context_profile','mode_basis'):
            meta.pop(key, None)
        meta.update(question=payload['question'], status='draft', claims_count=len(gold['claims']),
                    evidence_ids=list(dict.fromkeys(i for c in gold['claims'] for i in c['evidence_ids'])),
                    manual_correction=True, review_status='assistant_source_reading_20261112',
                    binding_sha256=canonical_sha(row['messages']))
        parse_sample(row)
        for c in meta['proofs']:
            for q in c['proofs']:
                assert q['quote'] in sources[q['evidence_id']]
        change = {'id':ident,'action':'repair','reason':reason, **before,
                  'after_messages_sha256':canonical_sha(row['messages']), 'after_metadata_sha256':canonical_sha(meta),
                  'before_record':original,'before_metadata':old_meta,'after_record':row,'after_metadata':meta,
                  'reviewer':'assistant source-reading; no fresh provider or clinical expert review'}
        changes.append(change); kept.append(row); metas.append(meta)
    assert {c['id'] for c in changes} == queued
    write_rows(OUTPUT/'train.jsonl', kept); write_rows(OUTPUT/'train.metadata.jsonl', metas)
    for split in ('val','test'):
        for suffix in ('.jsonl','.metadata.jsonl'):
            shutil.copy2(SOURCE/(split+suffix), OUTPUT/(split+suffix))
    write_rows(OUTPUT/'changes.jsonl',changes)
    write_rows(OUTPUT/'quarantine.jsonl',quarantine)
    manifest = {'version':'a2-v4-audited-20261112-r1','parent_directory':'../a2_drafting_v3',
        'parent_files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.iterdir() if f.is_file()},
        'counts':{'train':len(kept),'val':107,'test':93}, 'actions':dict(Counter(c['action'] for c in changes)),
        'scope':'Targeted corrections of 32 flagged train rows. Unchanged rows retain v3 review; patched rows reviewed by assistant against sources, not clinical gold. RAFT expansion not performed.',
        'files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUTPUT.iterdir() if f.is_file()}}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUTPUT),'counts':manifest['counts'],'actions':manifest['actions']},ensure_ascii=False))


if __name__=='__main__':main()
