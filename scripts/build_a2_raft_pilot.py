"""Build a small, source-read RAFT-inspired pilot. Data preparation only."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random

from src.training.a2_raft_protocol import GroundedDraft, SYSTEM
from scripts.train_a2 import load_data, parse_sample

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/processed/a2_drafting_v4'
OUTPUT = ROOT/'data/processed/a2_raft_pilot_v1'



def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def make_seeds():
    """Questions and source spans selected and read by the assistant before building variants."""
    load_data(SOURCE)
    parents = {}
    for split in ('train','val'):
        for row, meta in zip(read_rows(SOURCE/(split+'.jsonl')), read_rows(SOURCE/(split+'.metadata.jsonl'))):
            payload, _ = parse_sample(row)
            parents[meta['id']] = {'split':split, 'payload':payload, 'metadata':meta}
    seeds = []

    def seed(name, parent_id, language, partner, questions, combined, facts):
        parent = parents[parent_id]
        source_map = {}
        for evidence, chunk_id in zip(parent['payload']['evidence'],parent['metadata']['chunk_ids']):
            source_map[evidence['evidence_id']] = {
                'chunk_id':chunk_id,'document_id':parent['metadata']['document_id'],
                'title':evidence['title'],'content':' '.join(evidence['content'].split())}
        claims = []
        for text, eid, first, last in facts:
            source = source_map[eid]
            start = source['content'].index(first)
            end = source['content'].index(last, start)+len(last)
            claims.append({'text':text,'source_key':eid,'quote':source['content'][start:end],
                           'quote_start':start,'quote_end':end})
        used = {claim['source_key'] for claim in claims}
        seeds.append({'id':name,'parent_id':parent_id,'parent_family':parent['metadata']['family_id'],
            'split':parent['split'],'language':language,'partner':partner,
            'questions':questions,'combined_question':combined,'claims':claims,
            'sources':{eid:source for eid,source in source_map.items() if eid in used},
            'review':'assistant read against supplied source; not clinical expert review'})

    seed('sealants', 'a2v2_62507906173ad9c4','en','mckinney',[
        'Theo tài liệu, chất trám bít hố rãnh bảo vệ răng bằng cơ chế nào?',
        'Theo tài liệu, fluoride có tác dụng gì đối với khử khoáng và tái khoáng men răng?'],
        'Theo tài liệu, chất trám bít hố rãnh bảo vệ răng bằng cách nào và fluoride có tác dụng gì đối với khử khoáng, tái khoáng?',[
        ('Chất trám bít hố rãnh tạo hàng rào bảo vệ các hố và rãnh khỏi sự khử khoáng.','I2','dental \u00adsealants create','demineralization.'),
        ('Fluoride bảo vệ răng khỏi khử khoáng và có thể hỗ trợ tái khoáng hóa bề mặt răng đã bị khử khoáng.','I2','Fluoride protects teeth','tooth surfaces.')])
    seed('mckinney','exp_multi_chunk-40_chunk-25','en','sealants',[
        'Cô McKinney đã thực hiện những hoạt động chăm sóc và giáo dục nào tại Viện Y tế Quốc gia?',
        'Theo lời cô McKinney, những kỹ năng nào là thiết yếu khi làm việc trong y tế công cộng?'],
        'Cô McKinney làm những việc chăm sóc, giáo dục nào tại Viện Y tế Quốc gia và nêu những kỹ năng thiết yếu nào cho công việc y tế công cộng?',[
        ('Cô điều trị cho bệnh nhân có tình trạng sức khỏe phức tạp hoặc bệnh cấp tính và giáo dục sức khỏe răng miệng cho nhân viên y tế.','I1','Ms. McKinney previously worked','health care providers.'),
        ('Các kỹ năng thiết yếu gồm hiểu biết về thống kê, nghiên cứu, ngân sách và quản lý chương trình.','I2','Familiarity with statistics,','public health sector.')])
    seed('epidemiology','exp_multi_chunk-17_chunk-70','en','leadership',[
        'Dịch tễ học được định nghĩa như thế nào trong tài liệu?',
        'Các chương trình của bác sĩ Sara Baker được nêu có mục đích gì đối với trẻ sơ sinh của người nghèo?'],
        'Tài liệu định nghĩa dịch tễ học như thế nào và các chương trình của bác sĩ Sara Baker giúp trẻ sơ sinh của người nghèo với mục đích gì?',[
        ('Dịch tễ học nghiên cứu số lượng, phân bố, yếu tố quyết định và việc kiểm soát bệnh cùng các tình trạng sức khỏe trong một dân số nhất định.','I1','Epidemiology is the study','among a given population.'),
        ('Bà xây dựng nhiều chương trình giúp những người nghèo trong thành phố giữ cho trẻ sơ sinh của họ khỏe mạnh.','I2','Baker established many programs','their infants healthy.')])
    seed('leadership','a2v2_48e53006d27bf0e0','en','epidemiology',[
        'Làm việc trong y tế công cộng nha khoa giúp chuyên viên nhìn rõ điều gì về phòng ngừa theo đoạn nguồn?',
        'Theo đoạn nguồn, theo đuổi nghề nghiệp trong y tế công cộng phát triển những năng lực nào?'],
        'Làm việc trong y tế công cộng nha khoa giúp nhìn rõ điều gì về phòng ngừa và phát triển những năng lực nghề nghiệp nào?',[
        ('Công việc giúp nhìn rõ bản chất của việc phòng ngừa mà chuyên viên vệ sinh răng miệng đã cam kết khi tuyên thệ.','I2','The reality check','pledged their oath.'),
        ('Theo đuổi nghề nghiệp này giúp tăng năng lực lãnh đạo và khả năng thay đổi nghề nghiệp theo hướng có lợi hơn cho cộng đồng.','I2','Pursuing a career in public health','beneficial for the public.')])
    seed('splint_shape','a2v2_0c8b7988a4d3cfb5','vi','splint_duration',[
        'Theo tài liệu, sự hở khớp vùng cửa như thế nào khi máng nhai phẳng và thẳng theo chiều ngoài-trong?',
        'Làm lõm rãnh nhai trên máng có tác động gì đến sự hở khớp vùng cửa?'],
        'Sự hở khớp vùng cửa được mô tả ra sao với máng nhai phẳng, và làm lõm rãnh nhai có tác động gì?',[
        ('Sự hở khớp vùng cửa sẽ lớn khi máng nhai phẳng và thẳng theo chiều ngoài-trong.','I1','Sự hở khớp vùng cửa sẽ lớn','chiều ngoài-trong.'),
        ('Máng nhai có làm lõm ở rãnh nhai làm giảm sự hở khớp vùng cửa.','I1','Máng nhai có làm lõm','hở khớp vùng cửa.')])
    seed('splint_duration','a2v2_8faf395f73ed8b72','vi','splint_shape',[
        'Theo tài liệu, thời gian đeo máng nói chung trong các trường hợp rối loạn chức năng cấp tính là bao lâu?',
        'Theo tài liệu, máng có thể được sử dụng theo cách nào trong viêm khớp thái dương hàm mạn?'],
        'Theo tài liệu, thời gian đeo máng trong rối loạn chức năng cấp tính và cách sử dụng máng trong viêm khớp thái dương hàm mạn khác nhau thế nào?',[
        ('Nói chung, trong các trường hợp rối loạn chức năng cấp tính, thời gian đeo máng ngắn, khoảng 6–8 tuần.','I1','Nói chung, trong các trường hợp','máng ngắn (6-8 tuần)'),
        ('Máng có thể được đeo tạm thời và định kỳ lặp lại trong viêm khớp thái dương hàm mạn như một phương tiện kiểm soát.','I2','Máng có thể đeo tạm thời','phương tiện để kiểm soát.')])
    seed('feeding_position','a2v2_4dee6a577c0ad715_partial_answer','vi','bennett',[
        'Ở tư thế ăn tỉnh thức, hàm dưới ở đâu so với tư thế nghỉ khi đầu thẳng đứng?',
        'Theo tài liệu, tư thế ăn tỉnh thức liên hệ như thế nào với các tương quan chức năng của răng?'],
        'Ở tư thế ăn tỉnh thức, hàm dưới ở đâu so với tư thế nghỉ khi đầu thẳng đứng và tư thế này liên hệ thế nào với các tương quan chức năng của răng?',[
        ('Hàm dưới hơi nghiêng về phía trước so với tư thế nghỉ khi đầu thẳng đứng.','I2','Ở tư thế này hàm dưới','đầu thẳng đứng.'),
        ('Tư thế ăn tỉnh thức có liên quan có ý nghĩa với những tương quan chức năng của răng.','I2','Tư thế ăn tỉnh thức (alert feeding position) có liên quan','chức năng của răng.')])
    seed('bennett','tgt_dir_chunk-8877_partial_v3','vi','feeding_position',[
        'Hướng của vận động Bennett phụ thuộc vào những yếu tố nào được nêu trong tài liệu?',
        'Vận động sang bên diễn ra sớm hoặc diễn ra dần dần được gọi tương ứng là gì?'],
        'Hướng vận động Bennett phụ thuộc vào yếu tố nào, và vận động sang bên diễn ra sớm hoặc dần dần được gọi tương ứng là gì?',[
        ('Hướng vận động Bennett phụ thuộc vào cấu trúc khớp thái dương hàm, hướng dẫn của răng và sự co của cơ chân bướm ngoài.','I1','Hướng của vận động Bennett','cơ chân bướm ngoài.'),
        ('Vận động sang bên diễn ra sớm được gọi là vận động sang bên lập tức; diễn ra dần dần được gọi là vận động sang bên tăng tiến.','I1','nếu vận động sang bên diễn ra sớm','sang bên tăng tiến')])
    seed('bottle_label','a2v2_cd6c1142e9fa2527','en','learning',[
        'Theo đoạn tài liệu, trong điều kiện nào fluoride phải được liệt kê trên nhãn nước đóng chai?',
        'Theo đoạn tài liệu, nhà đóng chai có bắt buộc ghi nồng độ fluoride trên nhãn không?'],
        'Theo đoạn tài liệu, khi nào phải liệt kê fluoride trên nhãn nước đóng chai và có bắt buộc ghi nồng độ fluoride không?',[
        ('Theo đoạn tài liệu, chỉ khi nhà đóng chai thêm fluoride trong quá trình xử lý thì mới bắt buộc liệt kê chất này trên nhãn.','I1','FDA regulations require','during processing'),
        ('Theo đoạn tài liệu, nhà đóng chai không bắt buộc ghi nồng độ fluoride trên nhãn.','I1','the bottler is not required','fluoride concentration')])
    seed('learning','a2v2_d766cba4dbd0b56d','en','bottle_label',[
        'Theo tài liệu, điều kiện hóa hoạt động dựa trên những khái niệm nào?',
        'Theo tài liệu, học theo hành vi mẫu có thể hỗ trợ học tập thông qua cơ chế nào?'],
        'Theo tài liệu, điều kiện hóa hoạt động dựa trên khái niệm nào và học theo hành vi mẫu hỗ trợ học tập bằng cơ chế gì?',[
        ('Điều kiện hóa hoạt động dựa trên việc thưởng cho hành vi tốt và phạt hành vi xấu.','I1','Operant conditioning,','punishing bad behavior.'),
        ('Học theo hành vi mẫu có thể hỗ trợ học tập thông qua sự bắt chước.','I1','Modeling behavior can facilitate','through imitation.')])
    seed('hinge_axis','a2v2_bbfca06572dfd355','vi','swallowing',[
        'Theo tài liệu, trục bản lề là trục thật khi dùng loại cung mặt hoặc dụng cụ nào?',
        'Theo tài liệu, khi nào trục bản lề được xem là trục tự ý?'],
        'Theo tài liệu, trục bản lề thật và trục bản lề tự ý tương ứng với những loại cung mặt hoặc dụng cụ nào?',[
        ('Trục bản lề là trục thật khi sử dụng cung mặt động hoặc dụng cụ định vị trục bản lề.','I2','Trục bản lễ là một trục “thật”','dụng cụ định vị trục bản lề.'),
        ('Trục bản lề là trục tự ý khi sử dụng cung mặt đơn giản.','I2','Trục bản lễ là trục “tự ý”','cung mặt đơn giản.')])
    seed('swallowing','a2v2_385c96fce3cbb614','vi','hinge_axis',[
        'Trong động tác nuốt trống điển hình, các răng ở vị trí nào theo đoạn tài liệu?',
        'Trong động tác nuốt trống không điển hình, các răng có tiếp xúc nhau không?'],
        'Theo tài liệu, các răng ở vị trí nào khi nuốt trống điển hình và có tiếp xúc nhau khi nuốt trống không điển hình không?',[
        ('Trong nuốt trống điển hình, các răng ở vị trí lồng múi hoặc tiếp xúc lui sau.','I2','Trong động tác nuốt trống điển hình,','như đã nói trên.'),
        ('Trong nuốt trống không điển hình, các răng không tiếp xúc nhau.','I2','Trong động tác nuốt trống không điển hình,','các răng không tiếp xúc nhau')])
    return seeds


def full_context(seed):
    return [dict(source_key=eid, owner=seed['id'], content=source['content'],
                 excerpt_start=0, excerpt_end=len(source['content'])) for eid,source in seed['sources'].items()]


def excerpt_context(seed, claim_index):
    claim = seed['claims'][claim_index]
    return {'source_key':claim['source_key'],'owner':seed['id'],'content':claim['quote'],
            'excerpt_start':claim['quote_start'],'excerpt_end':claim['quote_end']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    output = parser.parse_args().output_dir
    seeds = make_seeds()
    by_id = {seed['id']:seed for seed in seeds}
    output.mkdir(exist_ok=False)
    records, metadata = {'train':[], 'val':[]}, {'train':[], 'val':[]}
    for seed in seeds:
        partner = by_id[seed['partner']]
        assert partner['split'] == seed['split'] and partner['language'] == seed['language']
        oracle = full_context(seed)
        # Keep at most three supplied passages. Every noise passage is a real source excerpt.
        noise = [excerpt_context(partner, i) for i in range(2 if len(oracle)==1 else 1)]
        variants = {'clean':oracle, 'noise_before':noise+oracle,
                    'noise_middle':([noise[0],*oracle,*noise[1:]] if len(oracle)==1 else [oracle[0],*noise,*oracle[1:]]),
                    'noise_after':oracle+noise,
                    'without_evidence':[excerpt_context(partner,0),excerpt_context(partner,1)]}
        questions = seed['questions']+[seed['combined_question']]

        def add(q_index, mode, contexts, claim_indices):
            question = questions[q_index]
            evidence, origins = [], []
            for i, context in enumerate(contexts,1):
                eid = f'I{i}'
                owner = by_id[context['owner']]
                source = owner['sources'][context['source_key']]
                evidence.append({'evidence_id':eid,'title':source['title'],'content':context['content']})
                origins.append({'evidence_id':eid,'owner_family':owner['id'],'document_id':source['document_id'],
                    'chunk_id':source['chunk_id'],'source_key':context['source_key'],
                    'excerpt_start':context['excerpt_start'],'excerpt_end':context['excerpt_end']})
            claims = []
            for index in claim_indices:
                fact = seed['claims'][index]
                matches = [f'I{i}' for i,context in enumerate(contexts,1)
                    if context['owner']==seed['id'] and context['source_key']==fact['source_key'] and fact['quote'] in context['content']]
                assert len(matches)==1
                claims.append({'text':fact['text'],'evidence_ids':matches,
                    'supporting_quotes':[{'evidence_id':matches[0],'quote':fact['quote']}]})
            gold = {'status':'draft' if claims else 'insufficient_evidence','claims':claims}
            GroundedDraft.model_validate(gold)
            payload = {'question':question,'evidence':evidence,'allowed_evidence_ids':[e['evidence_id'] for e in evidence]}
            record = {'messages':[{'role':'system','content':SYSTEM},
                {'role':'user','content':json.dumps(payload,ensure_ascii=False)+'\nJSON schema:\n'+json.dumps(GroundedDraft.model_json_schema(),ensure_ascii=False)},
                {'role':'assistant','content':json.dumps(gold,ensure_ascii=False)}]}
            ident = f'raft_{seed["id"]}_q{q_index}_{mode}'
            meta = {'id':ident,'split':seed['split'],'family_id':seed['id'],'parent_id':seed['parent_id'],
                'parent_family':seed['parent_family'],'language':seed['language'],'question_id':f'{seed["id"]}_q{q_index}',
                'mode':mode,'status':gold['status'],'supported_fact_indices':claim_indices,
                'question_fact_indices':[q_index] if q_index<2 else [0,1],
                'source_origins':origins,'messages_sha256':fingerprint(record['messages']),
                'review_origin':'assistant source-read seed facts and paired distractor excerpts; deterministic variants'}
            records[seed['split']].append(record); metadata[seed['split']].append(meta)

        for q_index in range(3):
            facts = [q_index] if q_index<2 else [0,1]
            seen_contexts = set()
            for mode, contexts in variants.items():
                # Avoid duplicate examples when two oracle passages make middle/before identical.
                signature = fingerprint(contexts)
                if signature in seen_contexts:
                    continue
                seen_contexts.add(signature)
                add(q_index, mode, contexts, [] if mode=='without_evidence' else facts)
            if q_index < 2:
                # Retain a different fact from the same topic, not the requested fact.
                remaining = [excerpt_context(seed,1-q_index),excerpt_context(partner,0)]
                if q_index == 1:
                    remaining.reverse()
                add(q_index,'missing_support_same_topic',remaining,[])
        add(2,'partial_answer',[excerpt_context(seed,1),excerpt_context(partner,0),excerpt_context(partner,1)],[1])
    for split in records:
        paired = list(zip(records[split],metadata[split]))
        random.Random(20261112).shuffle(paired)
        write_rows(output/(split+'.jsonl'),[p[0] for p in paired])
        write_rows(output/(split+'.metadata.jsonl'),[p[1] for p in paired])
    write_json(output/'schema.json', GroundedDraft.model_json_schema())
    write_json(output/'seed_review.json',seeds)
    manifest = {'version':'a2-raft-pilot-v1-20261112','source_dataset':'a2_drafting_v4',
        'source_files_sha256':{name:sha(SOURCE/name) for name in ('manifest.json','train.jsonl','train.metadata.jsonl','val.jsonl','val.metadata.jsonl')},
        'counts':{split:len(rows) for split,rows in records.items()},
        'families':{split:len({m['family_id'] for m in metas}) for split,metas in metadata.items()},
        'modes':{split:dict(Counter(m['mode'] for m in metas)) for split,metas in metadata.items()},
        'scope':'Small RAFT-inspired source-selection pilot with explicit quotes, abstention and partial answers. Manual source-reading by assistant, no external clinical adjudication. Derived variants are correlated, not independent QA cases.',
        'split_policy':'All oracle and distractor sources retain the v4 parent train/validation chapter split. No test examples used.',
        'training_executed':False,'inference_executed':False,
        'files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
