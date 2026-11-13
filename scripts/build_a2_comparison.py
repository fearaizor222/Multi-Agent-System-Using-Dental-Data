"""Author a frozen 72-case paired benchmark from source-read, non-A2-training documents."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from src.agents.drafting import DRAFT_SYSTEM, CONVERSATION_POLICY
from src.agents.contracts import GeneratedDraft

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/evaluation/a2_paired_20261110'


def family(name, ids, qa, qb, a, b, proofa, proofb, missing, source_b=0):
    return dict(name=name, ids=ids, qa=qa,qb=qb,a=a,b=b,proofa=proofa,proofb=proofb,missing=missing,source_b=source_b)


FAMILIES=[
family('wisdom',['chunk-51ff3dc6f4b2dab8a21ff56f8aadc7c1fffdb97694373e33dd84d1aa2b31a7ec'],
 'Răng khôn mọc lệch được mô tả là bất thường về những yếu tố nào?', 'Tình trạng này ảnh hưởng thế nào đến chức năng ăn nhai?',
 'Răng khôn mọc lệch là tình trạng mọc bất thường về trục, hướng và vị trí.',
 'Răng khôn mọc lệch không có chức năng ăn nhai và có thể gây biến chứng.',
 'Là tình trạng mọc bất thường về trục, hướng và vị trí của răng khôn',
 'làm cho răng không có chức năng ăn nhai và có thể gây biến chứng.',
 'Tỷ lệ phần trăm người có răng khôn mọc lệch trong dân số Việt Nam là bao nhiêu?'),
family('caries_ph',['chunk-8d214db4f74ede71689ffb8dab5620c633e195a20d12e3a4d3caef149064ccba'],
 'Theo đoạn được cấp, khi pH ở mức nào thì xuất hiện tổn thương hủy khoáng gây sâu răng sữa?',
 'Vi khuẩn chuyển hóa thức ăn có nguồn gốc gluxit thành chất gì?',
 'Đoạn nguồn nêu pH dưới 5 gây tổn thương hủy khoáng làm mất mô cứng của răng.',
 'Men của vi khuẩn ở mảng bám chuyển hóa thức ăn có nguồn gốc gluxit thành axit.',
 'Khi môi trường có pH< 5 thì gây ra tổn thương hủy khoáng làm mất mô cứng của răng',
 'Do các men của vi khuẩn ở mảng bám răng tác động lên các thức ăn có nguồn gốc Gluxit còn dính lại ở bề mặt răng, chuyển hóa thành axit.',
 'Phải duy trì pH dưới ngưỡng đó chính xác bao nhiêu phút thì bắt đầu hủy khoáng?'),
family('early_caries',['chunk-b225f59c27313fd4ca47557cdb19b4e0b50643be68d5164d566473bf2c9ac184','chunk-638c86e87147b3dc617221f7a9830482aac6e2a5216ad66f0bb29ac96f23a4f1'],
 'Tổn thương sâu răng sữa sớm có màu gì và bề mặt men còn nguyên vẹn không?',
 'Giá trị đo bằng thiết bị Laser huỳnh quang được đoạn nguồn nêu trong khoảng nào?',
 'Tổn thương sớm có vết trắng đục hoặc nâu vàng, bề mặt men còn nguyên vẹn.',
 'Đoạn nguồn nêu giá trị đo Laser huỳnh quang từ 10 đến 20.',
 'Vùng tổn thương sâu răng sớm xuất hiện các vết trắng đục hoặc nâu vàng, bề mặt men răng còn nguyên vẹn.',
 'Vùng tổn thương biểu hiện mức độ mất khoáng tương ứng với giá trị từ 10 đến 20 khi đo bằng thiết bị Laser huỳnh quang',
 'Thiết bị Laser huỳnh quang trong đoạn có độ nhạy chẩn đoán chính xác bao nhiêu phần trăm?',1),
family('implant_table',['chunk-ea0efa45ea6169fe81d92df413d99df865cb2750b5604ca8a01644c0f8dbca0b'],
 'Bảng trong nguồn đánh dấu tiêu xương trên X-quang ở viêm quanh Implant như thế nào?',
 'Còn viêm lợi quanh Implant được đánh dấu như thế nào?',
 'Bảng đánh dấu có tiêu xương quanh Implant trên X-quang ở viêm quanh Implant.',
 'Bảng đánh dấu không có tiêu xương quanh Implant trên X-quang ở viêm lợi quanh Implant.',
 'Triệu chứng | Viêm quanh Implant | Viêm lợi quanh Implant\nX quang: Tiêu xương quanh Implant | + | -',
 'Triệu chứng | Viêm quanh Implant | Viêm lợi quanh Implant\nX quang: Tiêu xương quanh Implant | + | -',
 'Tiêu xương trên X-quang xuất hiện sau chính xác bao nhiêu tháng kể từ khi đặt Implant?'),
family('dry_mouth',['chunk-a34d188419b21c254dbe1cbeb914357ee64bb5124f7a27c2c8027a461431588e'],
 'Đoạn nguồn ước tính khô miệng ảnh hưởng khoảng bao nhiêu phần trăm người cao tuổi?',
 'Khô miệng kéo dài có thể dẫn đến hai bệnh lý nào được nêu trong đoạn?',
 'Khô miệng ảnh hưởng khoảng 20% người cao tuổi theo đoạn nguồn.',
 'Khô miệng kéo dài có thể dẫn đến sâu răng và nấm candida miệng nặng.',
 'ảnh hưởng đến khoảng 20% người cao tuổi.',
 'Khô miệng kéo dài có thể dẫn đến sâu răng và nấm candida miệng nặng.',
 'Người bị khô miệng cần dùng liều thuốc kích thích nước bọt bao nhiêu miligam mỗi ngày?'),
family('caries_definition',['chunk-d9c193df7879668f57f95b08d28e22db2a69dada2bda1276b4f5ff643ae0f5dd'],
 'Sâu răng được định nghĩa như thế nào trong tài liệu?',
 'Các triệu chứng đau và nhạy cảm thường xuất hiện sớm hay muộn?',
 'Sâu răng là hiện tượng răng bị phá hủy, thường được gọi là lỗ sâu.',
 'Các triệu chứng đau và nhạy cảm thường xuất hiện muộn.',
 'Sâu răng là hiện tượng răng bị phá hủy, thường được gọi là lỗ sâu.',
 'Các triệu chứng như đau, nhạy cảm thường xuất hiện muộn.',
 'Tài liệu nêu tỷ lệ sâu răng ở tỉnh Hà Nam năm 2025 là bao nhiêu phần trăm?'),
family('pulpitis',['chunk-1f897838a4e6b89127e40429d0b4948a9a318c546f12eccf0a945f909268fe85'],
 'Viêm tủy được định nghĩa là gì?', 'Hãy nêu hai nguyên nhân là sâu răng và chấn thương được nguồn đề cập.',
 'Viêm tủy là phản ứng viêm của tủy răng.',
 'Sâu răng không được điều trị và chấn thương là hai nguyên nhân được nguồn nêu.',
 'Viêm tủy là phản ứng viêm của tủy răng',
 'hậu quả của sâu răng không được điều trị, chấn thương',
 'Thời gian đau bao nhiêu giây là ngưỡng chính xác để phân biệt viêm tủy hồi phục và không hồi phục?'),
family('gingivitis',['chunk-a93a8f48b43dc9420852db6687da1d8169add956540c5f4a4eba47251e2ef3f6'],
 'Viêm lợi là tình trạng viêm xảy ra ở mô nào?', 'Nguồn có nêu chảy máu và đỏ lợi trong các biểu hiện không?',
 'Viêm lợi là tình trạng viêm ở lợi.',
 'Chảy máu và đỏ lợi là các biểu hiện được nguồn nêu.',
 'Bệnh viêm lợi là viêm ở lợi',
 'gây ra chảy máu, sung, đỏ, chảy dịch',
 'Mỗi người bị viêm lợi phải dùng chính xác bao nhiêu miligam kháng sinh mỗi ngày?'),
family('english_prevention',['chunk-394843a09ccd9588aa88d7b439ddc834dc4f682521aef4506aa5d8d9ae82b844'],
 'Đoạn tiếng Anh nêu hai thói quen vệ sinh nào giúp loại bỏ mảng bám?',
 'Axit gây khử khoáng men răng được tạo ra từ quá trình nào trong miệng?',
 'Đánh răng thường xuyên bằng kem có fluor và dùng chỉ nha khoa giúp loại bỏ mảng bám.',
 'Axit được tạo ra do vi khuẩn lên men carbohydrate trong khoang miệng.',
 'Regular toothbrushing with fluoridated toothpaste, and dental floss helps remove plaque',
 'acid produced by bacterial fermentation of carbohydrates in the oral cavity',
 'Nghiên cứu trong đoạn xác định chải răng chính xác bao nhiêu giây mỗi lần là tối ưu?'),
family('english_risk',['chunk-58bffef1062773e504bb52a49d049662592c31fab947da6486ed765c0005cc90'],
 'Theo đoạn tiếng Anh, vệ sinh răng miệng kém và thường xuyên dùng đường tự do có liên quan thế nào đến sâu răng?',
 'Thiếu tiếp xúc fluor và hạn chế tiếp cận dịch vụ dự phòng có được xem là yếu tố nguy cơ không?',
 'Vệ sinh răng miệng kém và thường xuyên tiêu thụ đường tự do là các yếu tố nguy cơ liên quan đến tiến triển sâu răng.',
 'Thiếu tiếp xúc fluor và hạn chế tiếp cận dịch vụ nha khoa dự phòng cũng là các yếu tố nguy cơ.',
 'Poor oral hygiene, frequent consumption of free sugars, inadequate fluoride exposure, and limited access to preventive dental services are among the most important risk factors associated with disease progression',
 'inadequate fluoride exposure, and limited access to preventive dental services are among the most important risk factors associated with disease progression',
 'Đoạn nguồn xác định mức tăng nguy cơ sâu răng do đường tự do là chính xác bao nhiêu phần trăm?'),
family('english_aligners',['chunk-33ca25c1a4f8fe207cadc5f08d85f5ee5c506dcdc570663c052389293e8438f1'],
 'Theo đoạn tiếng Anh, khí cụ chỉnh nha cố định có thể làm tăng nguy cơ sâu răng vì những yếu tố nào?',
 'Máng chỉnh nha trong suốt có đặc điểm gì có thể thuận lợi cho vệ sinh răng miệng?',
 'Khí cụ chỉnh nha cố định có thể làm tăng nguy cơ sâu răng do lưu giữ mảng bám và thay đổi hệ vi sinh miệng.',
 'Máng chỉnh nha trong suốt là lựa chọn tháo lắp, có thể tạo thuận lợi cho vệ sinh răng miệng.',
 'Orthodontic treatments, particularly fixed appliances, may increase caries risk due to plaque retention and changes in the oral microbiota.',
 'Clear aligners, such as Invisalign®, represent a removable alternative that may facilitate oral hygiene and oral health.',
 'Đoạn nguồn chứng minh máng trong suốt giảm sâu răng chính xác bao nhiêu phần trăm so với khí cụ cố định?'),
family('english_apical',['chunk-e96f8dcdd4afdf530fe54a2ae081f5250e09818163dd81ca2bd585a884c6a88f'],
 'Viêm quanh chóp thường liên quan đến nhiễm trùng ở vị trí nào theo đoạn tiếng Anh?',
 'Tình trạng này có thể gây những tổn thương mô và xương nào?',
 'Viêm quanh chóp thường do nhiễm khuẩn dai dẳng trong hệ thống ống tủy.',
 'Tình trạng này đặc trưng bởi phá hủy mô quanh chóp và tiêu xương.',
 'typically resulting from persistent microbial infection within the root canal system',
 'Periapical tissue destruction and bone resorption characterize this condition',
 'Đoạn nguồn xác định tỷ lệ thành công sau điều trị viêm quanh chóp ở mốc 5 năm là bao nhiêu?'),
]


def norm(text):return ' '.join(text.casefold().split())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    catalog={r['chunk_id']:r for r in [json.loads(l) for l in (ROOT/'data/processed/public_v1/corpus/chunks.jsonl').read_text(encoding='utf-8').splitlines()]}
    seen_docs=set(); seen_chunks=set(); train_hashes={}; trained_texts=set()
    for path in [ROOT/'outputs/a2_comparison/20261107_completed/qwen3/dataset_info.json', ROOT/'outputs/a2_training/a2_qwen3_17b_20261109_213712_836549/dataset_info.json']:
        info=json.loads(path.read_text(encoding='utf-8')); parent=path.parent
        for split in ('train','validation'):
            rows=[json.loads(l) for l in (parent/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            # Old run may have metadata saved separately.
            meta_path=parent/(split+'.metadata.jsonl')
            metas=[json.loads(l) for l in meta_path.read_text(encoding='utf-8').splitlines()] if meta_path.exists() else []
            if not metas and split=='train' and not rows[0].get('metadata'):
                # Documents/chunks can be identified against immutable v1 source metadata.
                for s in ('train','val'):
                    metas.extend([json.loads(l) for l in (ROOT/'data/processed/a2_drafting_v1'/(s+'.metadata.jsonl')).read_text(encoding='utf-8').splitlines()])
            for r in rows:
                payload=json.loads(r['messages'][1]['content'].split('\nJSON schema:\n')[0])
                trained_texts.update(norm(e['content']) for e in payload['evidence'])
                if r.get('metadata'): metas.append(r['metadata'])
            for m in metas: seen_docs.add(m['document_id']);seen_chunks.update(m['chunk_ids'])
            train_hashes[str(parent.name)+'/'+split]=sha(parent/(split+'.jsonl'))
    rows=[]
    for i,f in enumerate(FAMILIES):
        sources=[catalog[c] for c in f['ids']]
        for c in sources:
            assert c['document_id'] not in seen_docs and c['chunk_id'] not in seen_chunks
            assert norm(c['content']) not in trained_texts
        assert norm(f['proofa']) in norm(sources[0]['content'])
        assert norm(f['proofb']) in norm(sources[f['source_b']]['content'])
        base=[{'evidence_id':f'I{j+1}','title':' > '.join(s['heading_path']), 'content':s['content']} for j,s in enumerate(sources)]
        for mode in ('direct','combine','distractor','missing','partial','history'):
            ev=json.loads(json.dumps(base)); history=[]; qa=f['qa']; refs=[(f['a'],'I1',f['proofa'])]
            if mode=='combine': qa+=' '+f['qb'];refs.append((f['b'],f'I{f["source_b"]+1}',f['proofb']))
            if mode=='partial': qa+=' '+f['missing']
            if mode=='missing':qa=f['missing'];refs=[]
            chunk_ids=list(f['ids'])
            if mode=='distractor':
                other=catalog[FAMILIES[(i+5)%len(FAMILIES)]['ids'][0]]
                ev.insert(0,{'evidence_id':'I0','title':' > '.join(other['heading_path']),'content':other['content']})
                chunk_ids.insert(0,other['chunk_id'])
                for j,e in enumerate(ev,1):e['evidence_id']=f'I{j}'
                refs=[(text,f'I{int(id[1:])+1}',proof) for text,id,proof in refs]
            if mode=='history':
                history=[{'role':'user','content':qa},{'role':'assistant','content':'Mình sẽ dựa trên tài liệu để trả lời câu hỏi của bạn.'}]
                qa='Bạn trả lời câu hỏi vừa rồi giúp tôi nhé.'
            payload={'question':qa,'evidence':ev,'allowed_evidence_ids':[e['evidence_id'] for e in ev]}
            if history:payload['conversation']={'history':history,'routing_hint':f['qa']}
            gold={'status':'draft' if refs else 'insufficient_evidence','claims':[{'text':text,'evidence_ids':[id]} for text,id,_ in refs]}
            GeneratedDraft.model_validate(gold)
            record={'id':f'{f["name"]}_{mode}','family':f['name'],'case':mode,'language':sources[0]['language'],
                'payload':payload,'gold':gold,'proofs':[{'text':text,'evidence_id':id,'quote':proof} for text,id,proof in refs],
                'chunk_ids':chunk_ids,'source_documents':[catalog[c]['document_id'] for c in chunk_ids],
                'source_urls':[catalog[c]['source_url'] for c in chunk_ids],
                'system':DRAFT_SYSTEM+(CONVERSATION_POLICY if history else '')}
            rows.append(record)
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'cases.jsonl'
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    if target.exists() and target.read_text(encoding='utf-8')!=content:
        raise ValueError('Refuse to change existing frozen benchmark')
    target.write_text(content,encoding='utf-8')
    manifest={'count':len(rows),'families':len(FAMILIES),'case_counts':dict(Counter(r['case'] for r in rows)),
        'languages':dict(Counter(r['language'] for r in rows)),'benchmark_sha256':sha(target),'compared_training_snapshots':train_hashes,
        'source_doc_chunk_exact_content_overlap':0,'authoring':'source-read fixed questions and expected facts, frozen before model outputs',
        'scope':'Controlled external-source transfer benchmark, not retrieval/end-to-end or expert clinical validation. Six related cases per family, not72 independent clinical observations.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
