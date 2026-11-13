"""Đóng băng ca phát triển mới trước khi sinh đầu ra; không dùng làm test độc lập."""
import hashlib
import json
from pathlib import Path

from src.data.a1_dataset import chat_messages, validate_pair
from src.training.a1_runtime import ROOT, write_json


def challenge():
    rows = []
    def add(name, category, text, targets=(), topics=(), attachments=(), images=(),
            capabilities=None, decision="route", reply=None, terms=(), history=(), resolved=None):
        context = {"message": text, "history": list(history), "attachments": list(attachments),
                   "capabilities": capabilities or {"A2": True, "A5": True, "lookup_clinic_service": True}}
        target = {"decision": decision, "targets": list(targets), "service_topics": list(topics),
                  "image_ids": list(images), "resolved_request": (resolved or text) if decision == "route" else None,
                  "message": None if decision == "route" else reply}
        context, target = validate_pair(context, target)
        rows.append({"id": "a1dev2_"+name, "group_id": "a1dev2_"+name, "category": category,
                     "split": "development_challenge", "input": context, "target": target,
                     "messages": chat_messages(context, target), "required_literal_terms": list(terms)})

    limited = {"A2": True, "A5": False, "lookup_clinic_service": False}
    add("terms", "terms", "Tìm tài liệu cho bài học về pericoronitis và cheilitis.", ["A2"], terms=["pericoronitis", "cheilitis"])
    add("terms_limited", "capability", "Giúp tôi tìm kiến thức về edentulism để làm bài thuyết trình.", ["A2"], capabilities=limited, terms=["edentulism"])
    add("two_terms", "terms", "Tìm bài đọc phân biệt mucositis với stomatitis, giữ hai từ này khi tra cứu.", ["A2"], terms=["mucositis", "stomatitis"])
    add("proper_name", "terms", "Tôi cần bài giới thiệu về kỹ thuật Hall trong nha khoa.", ["A2"], terms=["Hall"])
    add("service_off", "capability", "Tra giúp cơ sở có tiếp nhận khám vào thứ Bảy không.", decision="handoff", capabilities=limited, reply="Công cụ tra thông tin cơ sở hiện chưa khả dụng. Bạn vui lòng liên hệ người phụ trách.")
    add("service_on", "capability", "Kiểm tra phòng khám có sử dụng hóa đơn điện tử không.", ["lookup_clinic_service"], ["payment"], capabilities={"A2": False,"A5": False,"lookup_clinic_service": True})
    image = {"id": "sheet_c", "filename": "noi_dung.png", "kind": "document", "readable": True}
    add("image_on", "capability", "Chép phần chú thích của hình vẽ trong trang này.", ["A5"], attachments=[image], images=["sheet_c"], capabilities={"A2": False,"A5": True,"lookup_clinic_service": False})
    add("image_off", "capability", "Chép phần chú thích của hình vẽ trong trang này.", decision="handoff", attachments=[image], capabilities={"A2": True,"A5": False,"lookup_clinic_service": True}, reply="Thành phần đọc ảnh hiện chưa khả dụng. Bạn vui lòng thử lại sau hoặc liên hệ người phụ trách.")
    add("mixed_off", "capability", "Trích chữ trong trang này và tìm bài học về palatal rugae.", decision="handoff", attachments=[image], capabilities={"A2": False,"A5": True,"lookup_clinic_service": True}, reply="Thành phần kiến thức cần cho yêu cầu hiện chưa khả dụng. Bạn vui lòng liên hệ người phụ trách.")
    add("mixed_on", "mixed", "Trích chữ trong trang này và tìm bài học về palatal rugae.", ["A5","A2"], attachments=[image], images=["sheet_c"], terms=["palatal rugae"])
    add("mixed_service", "mixed", "Xin tài liệu về veneer và cách liên hệ cùng hướng dẫn đăng ký khám của cơ sở.", ["A2","lookup_clinic_service"], ["contact","appointment_info"], terms=["veneer"])
    add("double_service", "service", "Cần vị trí cơ sở và giấy tờ mang theo khi tới khám lần đầu.", ["lookup_clinic_service"], ["address","visit_preparation"])
    add("unclear", "clarification", "Vấn đề ấy giải quyết thế nào nhỉ?", decision="clarify", reply="Bạn mô tả rõ vấn đề nha khoa hoặc thông tin dịch vụ cần tìm nhé.")
    add("unclear_time", "clarification", "Sắp xếp lúc nào vậy bạn?", decision="clarify", reply="Bạn muốn hỏi giờ làm việc, cách đăng ký khám hay nội dung nào khác?")
    add("unclear_fee", "clarification", "Số tiền đó gồm những gì?", decision="clarify", reply="Bạn đang hỏi chi phí của dịch vụ cụ thể nào tại cơ sở?")
    clinical = {**image,"kind":"clinical"}
    add("clinical_unclear", "image", "Đây là ảnh chụp miệng hôm qua, nhờ bạn nhận xét.", decision="clarify", attachments=[clinical], reply="Bạn muốn trích chữ hay đánh giá tình trạng răng? Hệ thống chưa hỗ trợ chẩn đoán từ ảnh lâm sàng.")
    add("clinical_text", "image", "Chỉ chép dòng chữ ghi trên ảnh chụp miệng, không đánh giá bệnh.", ["A5"], attachments=[clinical], images=["sheet_c"])
    add("unreadable", "image", "Lấy phần văn bản từ hình đính kèm hộ tôi.", decision="clarify", attachments=[{**image,"readable":False}], reply="Ảnh hiện không đọc được. Bạn gửi lại ảnh rõ hơn nhé.")
    history = [{"role":"user","content":"Tôi đang tìm thông tin khí cụ nong hàm."},{"role":"assistant","content":"Bạn cần tài liệu học hay thông tin dịch vụ tại cơ sở?"}]
    add("context", "context", "Chỉ tài liệu học thôi, không tra dịch vụ nhé.", ["A2"], history=history, resolved="Tìm tài liệu học về khí cụ nong hàm, không tra dịch vụ.", terms=["nong hàm"])
    add("context_service", "context", "Mình cần kênh liên hệ để hỏi cơ sở về dịch vụ đó.", ["lookup_clinic_service"], ["contact"], history=history, resolved="Tra kênh liên hệ của cơ sở để hỏi dịch vụ khí cụ nong hàm.", terms=["nong hàm"])
    add("personal", "handoff", "Hãy quyết định riêng cho tôi dùng phương án phục hình nào mà không cần thăm khám.", decision="handoff", reply="Quyết định điều trị cá nhân cần người phụ trách chuyên môn thăm khám và đánh giá. Bạn vui lòng liên hệ bác sĩ.")
    add("hello", "social", "Chào các bạn ở bộ phận hỗ trợ nha khoa.", decision="respond", reply="Xin chào! Bạn cần hỗ trợ kiến thức nha khoa hay thông tin dịch vụ nào?")
    add("cancel", "social", "Thôi bỏ yêu cầu tra cứu nhé, mình đã đủ thông tin.", decision="respond", reply="Mình ghi nhận bạn muốn dừng yêu cầu. Khi cần, bạn có thể gửi câu hỏi mới.")
    add("outside", "out_of_scope", "Viết cho tôi lịch tập chạy marathon trong tháng tới.", decision="out_of_scope", reply="Hệ thống hỗ trợ thông tin nha khoa và dịch vụ phòng khám. Bạn vui lòng gửi yêu cầu phù hợp.")
    return rows


def main():
    path = ROOT / "data/curated/a1_supervisor_v2/development_challenge.json"
    if path.exists():
        raise ValueError("Không ghi đè bộ ca đã đóng băng")
    write_json(path, {"kind":"synthetic_development_challenge_not_independent_test",
                     "human_reviewed_records":0,"used_for_training":False,"records":challenge()})
    write_json(path.with_suffix(".manifest.json"), {"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"count":len(challenge())})
    print(json.dumps({"path":str(path),"count":len(challenge())},ensure_ascii=False))


if __name__ == "__main__":
    main()
