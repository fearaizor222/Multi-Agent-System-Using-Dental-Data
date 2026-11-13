"""Record source-reading adjudication after inspecting all144 raw answers.

P=meets fixed rubric; F=known failure; U=ambiguous wording, needs expert reading.
This is assistant source reading assisted by AI, never expert clinical validation.
Does not change benchmark, predictions, adapters or original automated judgments.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
from scripts.train_a2 import parse_json
from src.agents.contracts import GeneratedDraft

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'outputs/a2_paired_20261110/paired_20261110_075253_523581'
DATA=ROOT/'data/evaluation/a2_paired_20261110'
MODES=['direct','combine','distractor','missing','partial','history']
VERDICTS={
 'wisdom':{'old':'PPPPFP','new':'PPPPFP'},
 'caries_ph':{'old':'PFFPFP','new':'PPPPFP'},
 'early_caries':{'old':'PFPPFF','new':'PFPPFP'},
 'implant_table':{'old':'PPPPFP','new':'PPPPFP'},
 'dry_mouth':{'old':'PFPPFP','new':'PFPPPP'},
 'caries_definition':{'old':'PFPPPP','new':'PPPPPP'},
 'pulpitis':{'old':'PPFPFP','new':'PPPPFP'},
 'gingivitis':{'old':'PPPPFP','new':'PPPPFP'},
 'english_prevention':{'old':'FFFPFP','new':'FFUPFF'},
 'english_risk':{'old':'PPFPFP','new':'PPPPPP'},
 'english_aligners':{'old':'PFPPPP','new':'PFUPFP'},
 'english_apical':{'old':'FFFPFF','new':'FFFPFF'},
}

REASONS={
 'wisdom_partial':{'old':'Claims có evidence_ids rỗng: không đúng schema.', 'new':'Claims có evidence_ids rỗng: không đúng schema.'},
 'caries_ph_combine':{'old':'Thiếu phần vi khuẩn chuyển hóa gluxit thành axit.'},
 'caries_ph_distractor':{'old':'Đổi quan hệ: gán điều kiện pH<5 cho chuyển hóa gluxit thay vì hủy khoáng; không giải thích đúng tổn thương được hỏi.'},
 'caries_ph_partial':{'old':'Tự thêm mốc15phút không có trong nguồn.', 'new':'Từ chối phần thiếu đúng về ý nhưng tạo claim có evidence_ids rỗng, không đúng schema.'},
 'early_caries_combine':{'old':'Thiếu ý bề mặt men còn nguyên vẹn.', 'new':'Chỉ trả giá trị Laser, thiếu màu tổn thương và tình trạng bề mặt men.'},
 'early_caries_partial':{'old':'Thiếu bề mặt men còn nguyên vẹn; thêm giá trị mất khoáng thay vì trả độ nhạy.',
                          'new':'Gán giá trị mất khoáng10–20 thành độ nhạy chẩn đoán, sai loại đại lượng.'},
 'early_caries_history':{'old':'Lặp lại màu tổn thương hai lần, thiếu ý men còn nguyên vẹn.'},
 'implant_table_partial':{'old':'Tự thêm52tháng, nhầm số thứ tự chương thành thời gian.',
                          'new':'Nhận định đầu đúng; nhận định sau nói về bệnh khác, không trực tiếp trả phần thời gian được hỏi. Không bịa thời gian.'},
 'dry_mouth_combine':{'old':'Thiếu sâu răng và candida miệng.', 'new':'Thiếu sâu răng và candida miệng.'},
 'dry_mouth_partial':{'old':'Tự thêm liều10–20mg/ngày không có trong nguồn, đồng thời dùng evidence_ids rỗng.'},
 'caries_definition_combine':{'old':'Chỉ nói thời điểm triệu chứng, thiếu định nghĩa sâu răng.'},
 'pulpitis_distractor':{'old':'Nhận định thứ hai không được nguồn trích I2 hỗ trợ; trộn bệnh căn/tổn thương quanh chóp vào viêm tủy.'},
 'pulpitis_partial':{'old':'Bịa ngưỡng10giây; JSON có dấu chấm ngoài và claim có citation rỗng.',
                    'new':'Định nghĩa đúng; nhận định triệu chứng đau có nguồn nhưng không trả câu hỏi ngưỡng thời gian. Không bịa ngưỡng.'},
 'gingivitis_partial':{'old':'Từ chối toàn bộ dù định nghĩa vẫn trả lời được.', 'new':'Từ chối toàn bộ dù định nghĩa vẫn trả lời được.'},
 'english_prevention_direct':{'old':'Dịch toothpaste thành bột và floss thành dây chích.',
                             'new':'Pha tiếng Anh và chuỗi dịch đường kính/đường phân hủy carbohydrate không rõ nghĩa.'},
 'english_prevention_combine':{'old':'Dịch sai toothpaste và thiếu cơ chế lên men carbohydrate tạo axit.',
                              'new':'Trả đúng hai thói quen nhưng thiếu cơ chế tạo axit.'},
 'english_prevention_distractor':{'old':'Gán fluor cho bàn chải; nói giảm sản xuất đường thay vì giảm carbohydrate sẵn có cho tạo axit.',
                                 'new':'Hai thói quen đúng ý; cụm các đường kính có thể bị phân hủy không rõ là cách diễn đạt đường lên men hay lỗi dịch. Giữ U, không tự tính đúng.'},
 'english_prevention_partial':{'old':'Đổi fluoride thành chất tẩy trắng và carbohydrate thành đường huyết.',
                              'new':'Thiếu chỉ nha khoa và dịch sai thành đường huyết.'},
 'english_prevention_history':{'new':'Đổi chải răng thành súc miệng bằng kem đánh răng; thiếu chỉ nha khoa.'},
 'english_risk_distractor':{'old':'Thêm nhận định ngoài ý hỏi và dịch general anaesthesia thành gây tê.'},
 'english_risk_partial':{'old':'Tự thêm mức tăng100% không có trong nguồn.'},
 'english_aligners_combine':{'old':'Thêm khẳng định không gây xáo trộn răng miệng không có trong nguồn.',
                            'new':'Bỏ phần máng tháo lắp có thể thuận lợi cho vệ sinh.'},
 'english_aligners_distractor':{'new':'Thay lưu giữ mảng bám bằng tích tụ vi khuẩn. Gist gần nhau nhưng không hoàn toàn tương đương; giữ U để tránh tự quyết một lỗi thuật ngữ.'},
 'english_aligners_partial':{'new':'Từ chối cả câu dù nguyên nhân nguy cơ vẫn có trong nguồn.'},
 'english_apical_direct':{'old':'Không trả vị trí nhiễm trùng trong hệ thống ống tủy.', 'new':'Dịch root canal system thành hệ thống kẽ răng, sai vị trí.'},
 'english_apical_combine':{'old':'Thiếu vị trí ống tủy và tổn thương phá hủy mô/tiêu xương.', 'new':'Thiếu vị trí ống tủy và tổn thương phá hủy mô/tiêu xương.'},
 'english_apical_distractor':{'old':'Không trả vị trí ống tủy.', 'new':'Dịch sai ống tủy thành kẽ răng.'},
 'english_apical_partial':{'old':'Từ chối cả câu dù vị trí nhiễm trùng có nguồn.', 'new':'Từ chối cả câu dù vị trí nhiễm trùng có nguồn.'},
 'english_apical_history':{'old':'Không trả vị trí ống tủy.', 'new':'Dịch sai ống tủy thành kẽ răng.'},
}


def main():
    cases=[json.loads(l) for l in (DATA/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    predictions=[json.loads(l) for l in (FOLDER/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(predictions)==144 and len({(p['id'],p['adapter']) for p in predictions})==144
    final=[]
    for c in cases:
        for adapter in ('old','new'):
            p=next(p for p in predictions if p['id']==c['id'] and p['adapter']==adapter)
            verdict=VERDICTS[c['family']][adapter][MODES.index(c['case'])]
            schema=True;status=False;citations=False
            try:
                answer=GeneratedDraft.model_validate(parse_json(p['raw_output'])).model_dump()
                assert len(answer['claims'])<=2
                status=answer['status']==c['gold']['status']
                citations=all(set(claim['evidence_ids'])<=set(c['payload']['allowed_evidence_ids']) for claim in answer['claims'])
            except (ValueError,TypeError,AssertionError):schema=False
            if verdict=='P':assert schema and status and citations
            reason=REASONS.get(c['id'],{}).get(adapter)
            if verdict!='P':assert reason, (c['id'],adapter)
            final.append({**p,'family':c['family'],'case':c['case'],'language':c['language'],'source_reading_verdict':verdict,
                'schema_valid':schema,'status_correct':status,'citation_ids_valid':citations,
                'reason':reason or 'Đúng trạng thái, ý trả lời được nguồn trích hỗ trợ, đủ các ý được hỏi, không thêm chi tiết sai.',
                'source_reading_by':'assistant_not_clinical_expert'})
    report={'benchmark_sha256':hashlib.sha256((DATA/'cases.jsonl').read_bytes()).hexdigest(),
            'prediction_sha256':hashlib.sha256((FOLDER/'predictions.jsonl').read_bytes()).hexdigest(),
            'grading':'All144 answers source-read by assistant after two blinded AI screenings; no clinician adjudication. U remains unresolved.',
            'rubric':'Correct schema/status/citations; source-supported, relevant, covers requested supported facts. Unsupported extra statements, omitted requested facts, or separate unrequested claims fail. Equivalent wording accepted.',
            'scope':'72 controlled cases from12 source families; not72 independent observations or end-to-end clinical/RAG validation.',
            'models':{},'paired':{},'ambiguous':[]}
    for adapter in ('old','new'):
        subset=[r for r in final if r['adapter']==adapter]
        counts=Counter(r['source_reading_verdict'] for r in subset)
        report['models'][adapter]={'n':72,'pass':counts['P'],'fail':counts['F'],'unresolved':counts['U'],
            'confirmed_pass_rate':counts['P']/72,'max_pass_rate_if_all_unresolved_pass':(counts['P']+counts['U'])/72,
            'schema_valid':sum(r['schema_valid'] for r in subset),'status_correct':sum(r['status_correct'] for r in subset),
            'citation_ids_valid':sum(r['citation_ids_valid'] for r in subset),
            'by_case':{mode:dict(Counter(r['source_reading_verdict'] for r in subset if r['case']==mode)) for mode in MODES},
            'by_language':{lang:dict(Counter(r['source_reading_verdict'] for r in subset if r['language']==lang)) for lang in ('vi','en')}}
    for c in cases:
        a,b=[next(r['source_reading_verdict'] for r in final if r['id']==c['id'] and r['adapter']==adapter) for adapter in ('old','new')]
        category='unresolved' if 'U' in (a,b) else 'both_pass' if a==b=='P' else 'both_fail' if a==b=='F' else 'new_only' if b=='P' else 'old_only'
        report['paired'][category]=report['paired'].get(category,0)+1
    report['ambiguous']=[{'id':r['id'],'adapter':r['adapter'],'reason':r['reason']} for r in final if r['source_reading_verdict']=='U']
    report['unsupported_quantities']={'old':['caries_ph_partial','implant_table_partial','dry_mouth_partial','pulpitis_partial','english_risk_partial'],
        'new':['early_caries_partial'],'note':'Counts include invalid-schema outputs and wrong numeric meaning, not just new numeric tokens.'}
    (FOLDER/'source_reading.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    (FOLDER/'final_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
