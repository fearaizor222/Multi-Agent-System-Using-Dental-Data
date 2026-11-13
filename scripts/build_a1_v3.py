"""Sửa nhãn viết lại trong train và thêm đối chứng; không sửa holdout A1."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from scripts.prepare_a1_dataset import ROOT, build_records, export, write_json
from src.data.a1_dataset import TARGET_ORDER, normalized


def additions():
    groups = []
    def add(name, category, cases):
        groups.append({"id":"a1v3_"+name,"category":category,"cases":cases})
    def case(text, decision="route", targets=(), topics=(), attachments=(), images=(),
             history=(), reply=None, resolved=None, caps=None):
        return {"input":{"message":text,"history":list(history),"attachments":list(attachments),
                         "capabilities":caps or {"A2":True,"A5":True,"lookup_clinic_service":True}},
                "target":{"decision":decision,"targets":[x for x in TARGET_ORDER if x in targets],
                          "service_topics":list(topics),"image_ids":list(images),
                          "resolved_request":(resolved or text) if decision=="route" else None,
                          "message":None if decision=="route" else reply},
                "tags":["contrastive_v3"]}
    clarify="Bạn nói rõ vấn đề nha khoa hoặc thông tin dịch vụ đang muốn hỏi nhé."
    handoff="Yêu cầu này cần người phụ trách chuyên môn hỗ trợ. Bạn vui lòng liên hệ phòng khám hoặc bác sĩ."
    unavailable="Thành phần cần cho yêu cầu chưa khả dụng. Bạn vui lòng liên hệ người phụ trách hoặc thử lại sau."
    # Một câu ngắn: không có ngữ cảnh thì hỏi lại, có đối tượng rõ thì route.
    for i,text in enumerate(["Nội dung ấy nghĩa gì?","Giải thích mục đang nói giúp tôi.","Tôi cần đọc thêm phần vừa nhắc.",
                              "Mình muốn biết về thứ đó.","Bạn hỗ trợ cho phần ấy nhé.","Tìm thông tin về nó được chứ?",
                              "Mục vừa rồi có thông tin gì?","Cho tôi xem tài liệu về phần này."]):
        cases=[case(text,"clarify",reply=clarify)]
        for topic in ("cấu tạo men răng","khí cụ bảo vệ răng khi chơi thể thao"):
            history=[{"role":"user","content":"Tôi đang học về "+topic+"."},
                     {"role":"assistant","content":"Bạn cần tìm kiến thức về "+topic+"?"}]
            cases.append(case(text,targets=["A2"],history=history,resolved="Tìm kiến thức về "+topic+"."))
        add("context_pair_"+str(i),"context",cases)

    clinical_texts=["Nhờ bạn hỗ trợ tấm chụp khoang miệng vừa gửi.","Tôi muốn bạn xem qua ảnh khám này.",
                    "Hình chụp răng đính kèm cần bạn hỗ trợ.","Mình gửi hình ở lần khám trước, bạn xem nhé.",
                    "Tôi có một hình chụp vùng miệng cần hỗ trợ.","Hỗ trợ tôi với tấm hình khám đính kèm."]
    for i,text in enumerate(clinical_texts):
        image={"id":"scan_x","filename":"anh_kham.jpg","kind":"clinical" if i%2==0 else "unknown","readable":True}
        reply="Bạn muốn trích phần chữ hay đánh giá tình trạng răng? Hệ thống chưa hỗ trợ chẩn đoán từ ảnh lâm sàng."
        cases=[case(text,"clarify",attachments=[image],reply=reply)]
        for purpose,decision in [("Tôi chỉ cần các dòng chữ được gõ lại, không đánh giá bệnh.","route"),
                                 ("Tôi cần bạn chẩn đoán tình trạng bệnh của chính mình từ hình.","handoff")]:
            history=[{"role":"user","content":purpose},{"role":"assistant","content":"Mình đã ghi nhận mục đích của bạn."}]
            cases.append(case(text,decision,targets=["A5"] if decision=="route" else (),
                              images=["scan_x"] if decision=="route" else (),attachments=[image],history=history,
                              resolved="Chỉ trích các dòng chữ từ ảnh, không đánh giá tình trạng bệnh.",reply=handoff))
        cases.append(case(text,"clarify",attachments=[{**image,"readable":False}],reply="Bạn gửi lại hình rõ hơn và cho biết muốn trích chữ hay hỗ trợ nội dung nào nhé."))
        add("clinical_purpose_"+str(i),"image",cases)

    ocr_texts=["Gõ lại các dòng ở bộ ảnh tài liệu, chỉ cần nội dung chữ.","Xuất các ô bảng trong những trang hình đã gửi.",
               "Lấy phần chú giải ở tập trang sách đính kèm, không tìm kiến thức thêm.",
               "Tôi cần bản văn bản của những trang tài liệu trong tệp hình.",
               "Chép tiêu đề và chú thích từ bộ ảnh bài giảng này.","Chỉ trích chữ của toàn bộ các hình trang sách đính kèm."]
    for i,text in enumerate(ocr_texts):
        images=[{"id":"page_"+str(n),"filename":"trang_"+str(n)+".png","kind":"document","readable":True}
                for n in range(1,3+i%2)]
        ids=[a["id"] for a in images]
        caps={"A2":False,"A5":True,"lookup_clinic_service":False}
        cases=[case(text,targets=["A5"],attachments=images,images=ids),
               case(text,targets=["A5"],attachments=images,images=ids,caps=caps),
               case(text,"clarify",attachments=[{**a,"readable":False} for a in images],reply="Các ảnh chưa đọc được. Bạn vui lòng gửi bộ ảnh rõ hơn."),
               case(text,"handoff",attachments=images,caps={**caps,"A5":False},reply=unavailable)]
        add("ocr_only_"+str(i),"image",cases)
        mixed=text+" Sau đó tìm tài liệu học về "+["men răng","bàn chải kẽ","răng sữa","khẩu cái","dây cung","cầu răng"][i]+"."
        add("ocr_with_knowledge_"+str(i),"mixed",[
            case(mixed,targets=["A5","A2"],attachments=images,images=ids),
            case(mixed,"handoff",attachments=images,caps=caps,reply=unavailable)])

    pairs=[
        ("Hãy chọn cho riêng tôi có nên dùng hàm giả tháo lắp hay không.","Tìm tài liệu giới thiệu các loại hàm giả tháo lắp để học."),
        ("Bạn tự quyết định phương pháp chữa răng của tôi, đừng yêu cầu đi khám.","Tìm tài liệu tổng quan về các phương pháp phục hình để học, không quyết định cho cá nhân."),
        ("Kê cho tôi thuốc trị vấn đề răng miệng hiện tại.","Tôi cần tài liệu giải thích thuật ngữ thuốc dùng trong nha khoa, không xin kê đơn."),
        ("Xác định liều thuốc phù hợp với tình trạng răng của tôi.","Tìm tài liệu về khái niệm liều dùng trong giáo trình nha khoa, không tính liều cho người cụ thể."),
        ("Bạn nhìn ảnh rồi kết luận chính xác tôi mắc bệnh gì.","Tìm tài liệu về quy trình đọc ảnh trong đào tạo nha khoa, không kết luận bệnh của tôi."),
        ("Chọn giúp tôi răng nào cần nhổ dựa trên lời mô tả này.","Tôi đang học thuật ngữ nhổ răng, tìm giúp tài liệu nhập môn."),
        ("Quyết định ca của tôi có cần niềng răng hay không.","Tôi cần tài liệu về khái niệm chỉnh nha để làm bài học."),
        ("Chốt loại vật liệu trám sẽ dùng cho chiếc răng của tôi.","Tìm tài liệu so sánh các nhóm vật liệu trám nói chung."),
        ("Dựa vào triệu chứng tôi kể, chỉ định cách chữa riêng cho tôi.","Tìm kiến thức chung về cách tiếp nhận thông tin triệu chứng trong đào tạo nha khoa."),
        ("Tôi muốn bạn xác nhận răng mình hoàn toàn bình thường mà không cần bác sĩ.","Xin tài liệu về khám răng định kỳ cho bài thuyết trình."),
        ("Lập đơn điều trị riêng cho con tôi với số thuốc cụ thể.","Tìm bài đọc nhập môn về chăm sóc răng trẻ em, không chỉ định điều trị."),
        ("Tự đổi thuốc tôi đang dùng để chữa răng sang một loại khác.","Tôi muốn đọc thuật ngữ tương tác thuốc trong tài liệu nha khoa, không đổi thuốc cá nhân."),
    ]
    for i,(personal,educational) in enumerate(pairs):
        add("scope_pair_"+str(i),"handoff",[case(personal,"handoff",reply=handoff),case(educational,targets=["A2"])])

    vague=["Tôi chưa nói rõ nhưng muốn bạn giúp việc ấy.","Mình cần hướng dẫn thêm mà chưa biết gọi tên.",
           "Bạn giải quyết chuyện kia được không?","Tôi đang thắc mắc phần nào đó.","Chỗ đang nói có hỗ trợ được chứ?",
           "Tôi cần xem thêm thông tin, chưa nêu chủ đề.","Giờ mình phải làm gì với việc ấy?",
           "Bạn có thể tư vấn về nội dung tôi chưa mô tả không?","Mình muốn nhờ bạn một chuyện nha khoa, để mình nói sau.",
           "Tôi đang cần hỗ trợ nhưng chưa gửi nội dung câu hỏi."]
    for i,text in enumerate(vague):
        add("unclear_"+str(i),"clarification",[case(text,"clarify",reply=clarify)])
    people=["Chuyển câu hỏi của tôi cho người trực tư vấn.","Tôi muốn trao đổi trực tiếp với nhân viên phụ trách.",
            "Nhờ kết nối người thật để giải quyết yêu cầu của tôi.","Tôi không muốn tự động tư vấn, cần người phụ trách.",
            "Xin bác sĩ tiếp nhận câu hỏi của mình.","Chuyển giúp nội dung cho bộ phận phụ trách phòng khám.",
            "Tôi muốn người có chuyên môn xem yêu cầu trước khi quyết định.","Xin người phụ trách gọi lại để trao đổi ca của tôi.",
            "Mình cần nhân viên hỗ trợ vấn đề trong lần khám vừa rồi.","Hãy để người phụ trách chuyên môn xử lý trường hợp này.",
            "Tôi muốn dừng trợ lý và trao đổi với người tư vấn.","Nhờ bộ phận phụ trách tiếp nhận phản ánh của mình."]
    for i,text in enumerate(people):
        add("human_"+str(i),"handoff",[case(text,"handoff",reply="Bạn vui lòng liên hệ người phụ trách phòng khám để được tiếp nhận yêu cầu. Mình chưa thực hiện kết nối hoặc gửi thông tin thay bạn.")])
    outside=["Giải bài toán lượng giác lớp mười giúp tôi.","Soạn chương trình học đàn piano trong ba tháng.",
             "Viết bài giới thiệu thành phố biển cho lớp địa lý.","Tạo mã trò chơi cờ caro bằng Java.",
             "Hướng dẫn sửa bộ truyền động xe máy.","Tóm tắt diễn biến một trận bóng rổ.",
             "Thiết kế thực đơn cho tiệc sinh nhật công ty.","Soạn kịch bản phim trinh thám ngắn.",
             "Tìm lỗi trong chương trình quản lý thư viện sách.","Giải thích các tầng giao thức mạng máy tính.",
             "Viết thư xin việc kế toán bằng tiếng Anh.","Lập kế hoạch trang trí căn hộ hai phòng.",
             "Hướng dẫn cách thay dây đàn guitar.","Giới thiệu các nhân vật trong truyện cổ tích.",
             "Tạo đề luyện thi lịch sử thế giới.","Sắp xếp lịch trực cho đội bảo trì thang máy.",
             "Thiết kế bộ nhận diện cho cửa hàng hoa.","Giúp tôi hiểu cách vận hành máy in ba chiều.",
             "Viết đoạn quảng cáo cho quán trà sữa.","Tạo bảng phân công cho đội tổ chức hội thao."]
    for i,text in enumerate(outside):
        add("outside_"+str(i),"out_of_scope",[case(text,"out_of_scope",reply="Mình hỗ trợ thông tin nha khoa và dịch vụ phòng khám. Bạn vui lòng gửi câu hỏi phù hợp phạm vi này nhé.")])
    social=["Chào nhóm hỗ trợ, tôi chưa cần tra gì.","Xin gửi lời cảm ơn đến bộ phận tư vấn.","Cảm ơn bạn, mình kết thúc ở đây.",
            "Tôi xin hủy yêu cầu hỗ trợ vừa gửi.","Bỏ việc tìm tài liệu lúc nãy giúp tôi.","Không cần đọc ảnh nữa, tôi dừng nhé.",
            "Tôi chỉ muốn gửi lời chào buổi sáng.","Tạm biệt trợ lý nha khoa, hẹn lần sau.","Mình đã rõ, cảm ơn và dừng trao đổi.",
            "Xin chào, tôi mới vào và chưa có thắc mắc.","Mình bấm gửi nhầm, bỏ yêu cầu giúp nhé.","Cảm ơn đã hỗ trợ, tôi không cần thêm thông tin."]
    for i,text in enumerate(social):
        add("social_"+str(i),"social",[case(text,"respond",reply="Mình đã ghi nhận. Khi cần thông tin nha khoa hoặc dịch vụ phòng khám, bạn có thể gửi yêu cầu mới nhé.")])
    # Route dương khi chỉ một nhánh liên quan đang hoạt động.
    service=[("Tra cách cơ sở tiếp nhận khách chưa có hồ sơ.",["visit_preparation"]),
             ("Tìm phương thức nhận thanh toán tại cơ sở để mình chuẩn bị.",["payment"]),
             ("Tôi cần giờ tiếp nhận khách vào ngày lễ của cơ sở.",["opening_hours"]),
             ("Kiểm tra kênh đăng ký lịch khám chính thức của cơ sở.",["appointment_info"]),
             ("Cho tôi thông tin địa điểm của cơ sở đang tiếp nhận khám.",["address"]),
             ("Cơ sở tiếp nhận những loại thẻ bảo hiểm nào vậy?",["insurance"])]
    for i,(text,topics) in enumerate(service):
        add("service_only_cap_"+str(i),"capability",[case(text,targets=["lookup_clinic_service"],topics=topics,caps={"A2":False,"A5":False,"lookup_clinic_service":True})])
    knowledge=["Tìm tài liệu học về chức năng của lưỡi trong khoang miệng.","Mình cần tài liệu về cấu tạo ngà răng để học.",
               "Giải thích thuật ngữ răng vĩnh viễn theo kiến thức chung.","Tìm tài liệu nhập môn về răng hàm nhỏ.",
               "Tôi muốn đọc kiến thức chung về chỉ khâu trong giáo trình nha khoa.","Xin tài liệu giới thiệu về dấu răng trong phục hình."]
    for i,text in enumerate(knowledge):
        add("knowledge_only_cap_"+str(i),"capability",[case(text,targets=["A2"],caps={"A2":True,"A5":False,"lookup_clinic_service":False})])
    return groups


def build(output, tokenizer_dir, version="a1_supervisor_v3_1"):
    old_source=ROOT/"data/curated/a1_supervisor_v2/scenarios.json"
    source_path=ROOT/"data/curated"/version/"scenarios.json"
    if source_path.exists() or output.exists():
        raise ValueError("Không ghi đè phiên bản nguồn hoặc bản xuất")
    source=deepcopy(json.loads(old_source.read_text(encoding="utf-8")))
    revisions=[]
    for group in source["groups"]:
        if source["split_assignments"][group["id"]]!="train":
            continue
        for index,case in enumerate(group["cases"],1):
            if (case["target"]["decision"]=="route" and not case["input"].get("history")
                    and not group["id"].startswith("a1_inject_")):
                old=case["target"]["resolved_request"]
                new=case["input"]["message"].strip()
                if old!=new:
                    revisions.append({"id":f"{group['id']}_{index:02}","field":"resolved_request","old":old,"new":new,
                                      "reason":"Preserve standalone user wording; avoid teaching unneeded terminology rewriting"})
                    case["target"]["resolved_request"]=new
    extra=additions()
    # So câu mới với holdout và challenge đã công bố; chỉ phát hiện gần từ vựng.
    protected=[r["input"]["message"] for r in build_records(json.loads(old_source.read_text(encoding="utf-8"))) if r["split"]!="train"]
    challenge=ROOT/"data/curated/a1_supervisor_v2/development_challenge.json"
    protected += [r["input"]["message"] for r in json.loads(challenge.read_text(encoding="utf-8"))["records"]]
    for group in extra:
        for c in group["cases"]:
            text=normalized(c["input"]["message"]); tokens=set(re.findall(r"\w+",text))
            for other in protected:
                other=normalized(other); other_tokens=set(re.findall(r"\w+",other))
                if text==other or (min(len(tokens),len(other_tokens))>=5 and len(tokens&other_tokens)/len(tokens|other_tokens)>=.85):
                    raise ValueError("Mẫu mới gần holdout/challenge: "+group["id"])
    source["groups"].extend(extra)
    source["split_assignments"].update({g["id"]:"train" for g in extra})
    source.update(dataset_version=version,parent_source=str(old_source),
                  revision_scope="Train-only wording corrections and new contrastive groups; frozen v1/v2 validation and test")
    build_records(source)
    source_path.parent.mkdir(parents=True,exist_ok=True)
    write_json(source_path,source)
    write_json(source_path.with_name("train_revisions.json"),revisions)
    summary=export(source_path,output,tokenizer_path=tokenizer_dir)
    assert summary["cross_split_near_message_pairs"]==0
    for split in ("validation","test"):
        for suffix in (".jsonl",".metadata.jsonl"):
            assert (ROOT/"data/processed/a1_supervisor_v2"/(split+suffix)).read_bytes()==(output/(split+suffix)).read_bytes()
    return {"dataset":summary,"train_wording_revisions":len(revisions),"new_groups":len(extra)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--tokenizer-dir",type=Path,required=True)
    parser.add_argument("--version",choices=["a1_supervisor_v3_1"],default="a1_supervisor_v3_1")
    args=parser.parse_args()
    print(json.dumps(build(args.output_dir,args.tokenizer_dir,args.version),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
