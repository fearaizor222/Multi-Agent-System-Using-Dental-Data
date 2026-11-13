"""Biên soạn phần bổ sung train; đóng băng mọi mẫu và split của A1 v1."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import re

from scripts.prepare_a1_dataset import SOURCE, ROOT, build_records, export, write_json
from src.data.a1_dataset import TARGET_ORDER, normalized


def additions():
    groups = []

    def group(name, category, cases):
        groups.append({"id": "a1v2_" + name, "category": category, "cases": cases})

    def case(text, decision="route", targets=(), topics=(), images=(), reply=None,
             history=(), attachments=(), capabilities=None, resolved=None, tags=()):
        return {"input": {"message": text, "history": list(history), "attachments": list(attachments),
                          "capabilities": capabilities or {"A2": True, "A5": True, "lookup_clinic_service": True}},
                "target": {"decision": decision, "targets": [t for t in TARGET_ORDER if t in targets],
                           "service_topics": list(topics), "image_ids": list(images),
                           "resolved_request": (resolved or text) if decision == "route" else None,
                           "message": None if decision == "route" else reply}, "tags": list(tags)}

    handoff = "Thành phần cần cho yêu cầu hiện chưa khả dụng. Bạn vui lòng liên hệ người phụ trách hoặc thử lại sau."
    image = {"id": "page_a", "filename": "trang_tai_lieu.jpg", "kind": "document", "readable": True}
    # Cố ý đủ 8 tổ hợp: nhánh không liên quan tắt không được làm hỏng route.
    capability_tasks = [
        ("cementum", "Tìm tài liệu nhập môn về cementum giúp nhóm học của mình.", ["A2"], [], []),
        ("drymouth", "Mình muốn đọc kiến thức chung về thuật ngữ xerostomia.", ["A2"], [], []),
        ("wire", "Xin giải thích thuật ngữ dây cung trong tài liệu chỉnh nha.", ["A2"], [], []),
        ("wallet", "Cơ sở có cho khách thanh toán bằng ví điện tử không?", ["lookup_clinic_service"], ["payment"], []),
        ("parking", "Tôi cần thông tin chuẩn bị giấy tờ cho lần khám đầu tại cơ sở.", ["lookup_clinic_service"], ["visit_preparation"], []),
        ("insurance", "Tra giúp danh sách bảo hiểm cơ sở đang tiếp nhận.", ["lookup_clinic_service"], ["insurance"], []),
        ("ocr", "Chép lại tiêu đề trên trang tài liệu đã gửi, chưa cần giải thích.", ["A5"], [], [image]),
        ("diagram", "Đọc các nhãn chữ trong sơ đồ ở trang đính kèm.", ["A5"], [], [image]),
        ("occlusion_page", "Đọc trang này và tìm kiến thức về occlusion.", ["A5", "A2"], [], [image]),
        ("enamel_service", "Giải thích enamel và tra cơ sở có cung cấp dịch vụ fluor không.", ["A2", "lookup_clinic_service"], ["availability"], []),
        ("page_contact", "Đọc chữ trong tệp ảnh rồi tìm kênh liên hệ của cơ sở.", ["A5", "lookup_clinic_service"], ["contact"], [image]),
        ("all", "Chép nội dung trang đính kèm, tìm kiến thức về dentin và cách đăng ký khám ở cơ sở.", ["A5", "A2", "lookup_clinic_service"], ["appointment_info"], [image]),
    ]
    for name, text, targets, topics, attachments in capability_tasks:
        cases = []
        for mask in itertools.product((False, True), repeat=3):
            caps = dict(zip(("A2", "A5", "lookup_clinic_service"), mask))
            enabled = all(caps[t] for t in targets)
            cases.append(case(text, targets=targets if enabled else (), topics=topics if enabled else (),
                              images=[a["id"] for a in attachments] if enabled else (),
                              decision="route" if enabled else "handoff", reply=handoff,
                              capabilities=caps, attachments=attachments, tags=["capability_counterfactual"]))
        group("caps_" + name, "capability", cases)

    ambiguous = [
        ("it", ["Cái đó phải làm sao vậy bạn?", "Mình muốn biết việc này.", "Bạn giải thích phần kia được chứ?"], "Bạn muốn tìm hiểu vấn đề hoặc thuật ngữ nha khoa nào?"),
        ("fee", ["Khoản này tính như nào nhỉ?", "Cho tôi hỏi mức phí đó.", "Tính tiền phần ấy giúp mình."], "Bạn muốn tra chi phí cho dịch vụ cụ thể nào tại phòng khám?"),
        ("time", ["Khi nào thì được vậy?", "Lúc nào tiện nhỉ?", "Mình hỏi thời gian ấy bạn."], "Bạn muốn hỏi thời gian mở cửa, cách đăng ký khám hay nội dung nào khác?"),
        ("place", ["Đến chỗ nào thế?", "Vị trí của cái này đâu?", "Mình tìm nơi đó mãi."], "Bạn đang cần địa chỉ phòng khám hay muốn hỏi vị trí nào khác?"),
        ("procedure", ["Các bước thế nào ạ?", "Hướng dẫn quy trình đó đi.", "Mình phải bắt đầu từ đâu nhỉ?"], "Bạn muốn hỏi cách đăng ký khám hay tìm hiểu quy trình nha khoa cụ thể nào?"),
        ("unfinished", ["Mình muốn hỏi thêm về...", "Chuyện răng ấy, mình chưa biết hỏi từ đâu.", "Bên nha khoa có vấn đề cần nhờ bạn."], "Bạn mô tả rõ điều cần tìm hiểu để mình hỗ trợ nhé."),
    ]
    for name, texts, reply in ambiguous:
        group("unclear_" + name, "clarification", [case(t, "clarify", reply=reply) for t in texts])

    clinical_reply = "Bạn muốn trích chữ trên ảnh hay đánh giá tình trạng răng? Hệ thống chưa hỗ trợ chẩn đoán từ ảnh lâm sàng."
    for kind in ("clinical", "unknown"):
        attachment = {"id": "scan_b", "filename": "tep_dinh_kem.png", "kind": kind, "readable": True}
        texts = (["Nhờ bạn nhận xét tấm chụp này.", "Ảnh mình gửi trông ra sao?", "Đây là ảnh khám của mình, bạn hỗ trợ nhé.",
                  "Em gửi ảnh răng để bạn xem qua."] if kind == "clinical" else
                 ["Nhờ kiểm tra tệp hình vừa đính kèm.", "Bạn hỗ trợ nội dung trong tấm hình nhé.",
                  "Mình chưa rõ cần làm gì với hình này, xem giúp.", "Bạn nhìn tấm này rồi nói mình biết với."])
        group("ambiguous_image_" + kind, "image", [case(t, "clarify", reply=clinical_reply, attachments=[attachment]) for t in texts])
        group("explicit_ocr_" + kind, "image", [case(t, targets=["A5"], images=["scan_b"], attachments=[attachment]) for t in
              ["Chỉ chép phần chữ ở ảnh, không đánh giá tình trạng răng.", "Trích các dòng văn bản từ hình đính kèm, bỏ qua việc chẩn đoán."]])

    for name, attachment, texts, reply in [
        ("missing", None, ["Chuyển chữ từ tấm ảnh tôi nói tới sang văn bản.", "Đọc hộ bảng trong hình nhé, tôi sẽ gửi sau."], "Bạn gửi ảnh tài liệu cần trích nội dung nhé."),
        ("blur", {**image, "readable": False}, ["Chép văn bản trong trang ảnh này.", "Trích bảng ở trang tài liệu vừa đính kèm."], "Ảnh hiện không đọc được. Bạn vui lòng gửi bản rõ hơn."),
    ]:
        group("image_" + name, "image", [case(t, "clarify", reply=reply, attachments=[attachment] if attachment else []) for t in texts])

    # Giữ nguyên thuật ngữ/yêu cầu độc lập; không dạy A1 tự dịch hay giải thích y khoa.
    terms = [
        "Tôi cần tài liệu định nghĩa enamel và dentin để đọc thêm.",
        "Xin tìm bài nhập môn phân biệt cementum với dentin.",
        "Mình đang học từ khóa occlusion, tìm giúp kiến thức nền.",
        "Tìm tài liệu về xerostomia, giữ đúng từ khóa khi tra nhé.",
        "Giải thích thuật ngữ fluorosis trong tài liệu, chỉ kiến thức chung.",
        "Tôi đọc thấy apicoectomy, giúp tìm tài liệu về thuật ngữ đó.",
        "Xin tài liệu nói về sự khác biệt giữa attrition và abrasion.",
        "Tìm bài học về erosion và abfraction trong nha khoa.",
        "Tra kiến thức chung về alveolar bone trong tài liệu nha khoa.",
        "Mình cần tài liệu về thuật ngữ dental plaque và calculus.",
        "Giúp tôi đọc thêm về periodontal ligament, giữ nguyên thuật ngữ này.",
        "Tìm tài liệu giới thiệu về amelogenesis imperfecta để học.",
        "Dentinogenesis imperfecta là từ khóa tôi cần tài liệu nhập môn.",
        "Xin kiến thức chung về khẩu cái cứng và khẩu cái mềm.",
        "Tôi muốn tìm tài liệu về trám bít hố rãnh, không hỏi giá.",
        "Giúp mình tra thuật ngữ khô ổ răng cho bài học, không tư vấn cho cá nhân.",
    ]
    group("term_preservation", "knowledge", [case(t, targets=["A2"], tags=["preserve_request_verbatim"]) for t in terms])

    mixed = [
        ("Tìm kiến thức về khí cụ duy trì rồi hỏi cơ sở có cung cấp loại dịch vụ này không.", ["availability"]),
        ("Mình cần bài đọc về trám bít hố rãnh và cách liên hệ cơ sở để hỏi thêm.", ["contact"]),
        ("Tìm tài liệu về răng giả tháo lắp, đồng thời tra phương thức thanh toán của cơ sở.", ["payment"]),
        ("Giải thích từ khóa dental sealant và tìm hướng dẫn đăng ký khám tại cơ sở.", ["appointment_info"]),
        ("Cho mình đọc về men răng và kiểm tra thủ tục giấy tờ khi đến khám.", ["visit_preparation"]),
        ("Xin kiến thức chung về chỉ nha khoa và danh sách bảo hiểm cơ sở chấp nhận.", ["insurance"]),
        ("Tôi cần kiến thức về phục hình tháo lắp kèm số liên hệ và bảng giá dịch vụ ở cơ sở.", ["contact", "pricing"]),
        ("Cho mình tài liệu nhập môn về cầu răng, tra nơi khám và kênh đăng ký nhé.", ["address", "appointment_info"]),
        ("Tìm bài học về phục hình răng rồi tra giờ làm việc và phương thức thanh toán của cơ sở.", ["opening_hours", "payment"]),
        ("Tôi muốn đọc kiến thức về bàn chải kẽ và hỏi cơ sở có dịch vụ hướng dẫn vệ sinh không.", ["availability"]),
        ("Xin kiến thức về đánh bóng răng, đồng thời kiểm tra chi phí và bảo hiểm tại cơ sở.", ["pricing", "insurance"]),
        ("Mình cần tài liệu về hàm phủ, rồi tìm số điện thoại và địa chỉ cơ sở.", ["address", "contact"]),
    ]
    group("multiple_needs", "mixed", [case(t, targets=["A2", "lookup_clinic_service"], topics=topics) for t, topics in mixed])

    services = [
        ("Tôi cần số tổng đài hoặc email mà cơ sở đang dùng.", ["contact"]),
        ("Tra hộ mình cách khách mới đăng ký buổi khám ở cơ sở.", ["appointment_info"]),
        ("Cơ sở yêu cầu mang loại giấy tờ nào khi tiếp nhận khách mới?", ["visit_preparation"]),
        ("Tôi muốn kiểm tra cơ sở có tiếp nhận thanh toán bằng thẻ quốc tế không.", ["payment"]),
        ("Bảo hiểm tư nhân nào được cơ sở tiếp nhận hiện nay?", ["insurance"]),
        ("Cơ sở có nhận làm phục hình tháo lắp không nhỉ?", ["availability"]),
        ("Tôi cần địa điểm và số liên hệ chính thức của cơ sở.", ["address", "contact"]),
        ("Tra bảng phí làm khí cụ duy trì và cách thanh toán tại cơ sở.", ["pricing", "payment"]),
        ("Kiểm tra giờ tiếp nhận khách và hướng dẫn đăng ký buổi khám.", ["opening_hours", "appointment_info"]),
        ("Cơ sở có khám vào Chủ nhật không, tra thông tin chính thức giúp nhé.", ["opening_hours"]),
        ("Tôi cần thông tin chuẩn bị cho lần đầu đến cơ sở, bao gồm giấy tờ bảo hiểm.", ["insurance", "visit_preparation"]),
        ("Hỏi giúp cơ sở có phục vụ trẻ em không và liên hệ qua đâu.", ["availability", "contact"]),
    ]
    group("service_topics", "service", [case(t, targets=["lookup_clinic_service"], topics=topics) for t, topics in services])

    contexts = [
        ("Giới thiệu các loại bàn chải kẽ giúp mình.", "Bạn cần kiến thức chung hay hỏi cơ sở có bán không?", "Chỉ kiến thức, khỏi tra dịch vụ.", ["A2"], [], "Tìm kiến thức chung về các loại bàn chải kẽ."),
        ("Tìm hiểu phục hình tháo lắp.", "Bạn muốn đọc tài liệu hay hỏi chi phí tại cơ sở?", "Mình chọn chi phí nhé, chưa cần tài liệu.", ["lookup_clinic_service"], ["pricing"], "Tra chi phí phục hình tháo lắp tại cơ sở."),
        ("Cho tôi thông tin cơ sở.", "Bạn cần số liên hệ, phương thức thanh toán hay thủ tục tiếp nhận?", "Số liên hệ trước nhé.", ["lookup_clinic_service"], ["contact"], "Tra số liên hệ của cơ sở."),
        ("Tôi đang quan tâm khí cụ duy trì.", "Bạn muốn kiến thức chung hay kiểm tra dịch vụ tại cơ sở?", "Kiểm tra ở cơ sở có làm loại đó không.", ["lookup_clinic_service"], ["availability"], "Tra cơ sở có cung cấp dịch vụ khí cụ duy trì không."),
        ("Tôi muốn đến lần đầu.", "Bạn cần cách đăng ký hay giấy tờ mang theo?", "Chọn giấy tờ cần chuẩn bị.", ["lookup_clinic_service"], ["visit_preparation"], "Tra giấy tờ cần chuẩn bị khi đến khám lần đầu."),
        ("Mình đọc thấy thuật ngữ enamel.", "Bạn cần tìm tài liệu về enamel chứ?", "Đúng, tìm thêm giúp mình.", ["A2"], [], "Tìm tài liệu về enamel."),
        ("Mình tìm hiểu cầu răng.", "Bạn muốn hỏi giờ mở cửa của cơ sở nữa không?", "Không, bỏ giờ mở cửa và chỉ tìm bài đọc về chủ đề cũ.", ["A2"], [], "Tìm tài liệu kiến thức chung về cầu răng, không tra giờ mở cửa."),
        ("Tìm kiến thức về hàm phủ và cách đăng ký khám.", "Mình có thể hỗ trợ hai nhu cầu đó.", "Giờ bỏ phần học, chỉ cần cách đăng ký thôi.", ["lookup_clinic_service"], ["appointment_info"], "Tra cách đăng ký khám tại cơ sở."),
        ("Tôi đang hỏi cách thanh toán tại cơ sở.", "Bạn muốn kiểm tra hình thức thanh toán?", "Đúng, kiểm tra chuyển khoản và thẻ giúp nhé.", ["lookup_clinic_service"], ["payment"], "Tra cơ sở có chấp nhận chuyển khoản và thẻ không."),
        ("Tìm tài liệu về răng giả tháo lắp.", "Bạn muốn đọc tài liệu để học phải không?", "Thêm cả cách liên hệ để mình hỏi cơ sở.", ["A2", "lookup_clinic_service"], ["contact"], "Tìm tài liệu về răng giả tháo lắp và kênh liên hệ của cơ sở."),
    ]
    group("context_resolution", "context", [case(text, targets=targets, topics=topics,
          history=[{"role": "user", "content": first}, {"role": "assistant", "content": assistant}], resolved=resolved)
          for first, assistant, text, targets, topics, resolved in contexts])

    social = [
        ("Chúc cơ sở một ngày vui vẻ.", "Cảm ơn bạn! Mình sẵn sàng hỗ trợ thông tin nha khoa khi bạn cần."),
        ("Mình chỉ ghé chào, chưa có câu hỏi đâu.", "Chào bạn! Khi cần thông tin nha khoa hoặc dịch vụ phòng khám, bạn cứ gửi câu hỏi nhé."),
        ("Ổn rồi, mình dừng trao đổi ở đây.", "Được nhé. Khi cần hỗ trợ thêm, bạn có thể gửi câu hỏi mới."),
        ("Cảm ơn, không cần tra cứu thêm nữa.", "Rất vui được hỗ trợ bạn. Mình dừng yêu cầu tra cứu nhé."),
        ("Chào đội ngũ tư vấn nha khoa!", "Xin chào! Bạn muốn tìm hiểu kiến thức nha khoa hay thông tin dịch vụ nào?"),
        ("Xin dừng yêu cầu vừa rồi giúp mình.", "Mình ghi nhận bạn muốn dừng yêu cầu. Khi cần, bạn có thể gửi câu hỏi khác."),
    ]
    group("social", "social", [case(t, "respond", reply=reply) for t, reply in social])
    return groups


def build(output, tokenizer_dir):
    parent = ROOT / "data/processed/a1_supervisor_v1"
    source_dir = ROOT / "data/curated/a1_supervisor_v2"
    source_path = source_dir / "scenarios.json"
    if output.exists() or source_path.exists():
        raise ValueError("Không ghi đè source hoặc bản xuất đã có")
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    old_rows = build_records(original)
    extension = additions()
    # Kiểm tra các câu mới với holdout bằng từ vựng, không dùng nhãn holdout để biên soạn.
    collisions = []
    holdout = [r for r in old_rows if r["split"] != "train"]
    for g in extension:
        for c in g["cases"]:
            message = normalized(c["input"]["message"])
            tokens = set(re.findall(r"\w+", message))
            for old in holdout:
                old_message = normalized(old["input"]["message"])
                old_tokens = set(re.findall(r"\w+", old_message))
                if (message == old_message or
                    (min(len(tokens), len(old_tokens)) >= 5 and len(tokens & old_tokens) / len(tokens | old_tokens) >= .85)):
                    collisions.append((g["id"], old["id"]))
    if collisions:
        raise ValueError("Câu bổ sung gần holdout; cần thay tình huống mới: " + str(collisions))
    source = deepcopy(original)
    source.update(dataset_version="a1_supervisor_v2", parent_source=str(SOURCE),
                  expansion_scope="New synthetic train groups only; v1 validation/test and original train preserved byte-for-byte")
    source["groups"].extend(extension)
    source["split_assignments"] = {r["group_id"]: r["split"] for r in old_rows}
    source["split_assignments"].update({g["id"]: "train" for g in extension})
    # Kiểm tra toàn bộ trước khi ghi source và export.
    new_rows = build_records(source)
    by_id = {r["id"]: r for r in new_rows}
    assert all(by_id[r["id"]] == r for r in old_rows)
    source_dir.mkdir(parents=True, exist_ok=True)
    write_json(source_path, source)
    summary = export(source_path, output, tokenizer_path=tokenizer_dir)
    assert summary["cross_split_near_message_pairs"] == 0
    for split in ("validation", "test"):
        for suffix in (".jsonl", ".metadata.jsonl"):
            assert (parent / (split + suffix)).read_bytes() == (output / (split + suffix)).read_bytes()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.tokenizer_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
