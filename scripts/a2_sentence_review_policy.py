"""Assistant source-reading decisions, not a clinical expert certification.

Reviewed whole seed sources and the deterministic partner sources on 2026-11-12.
Keep-set restrictions intentionally discard ambiguous negative/partial variants.
"""

# Missing this fact can leave alternative support, a pronoun without its
# antecedent, or a caption/example containing the same answer.
REQUIRE_PRESENT = {
    'mckinney': {0, 1}, 'epidemiology': {1}, 'feeding_position': {0},
    'hinge_axis': {1}, 'dental_board': {0}, 'global_care': {0},
    'regression_example': {1}, 'program_baseline': {0},
    'central_relation': {0, 1}, 'border_contact_movements': {0},
    'closing_arc_limits': {0, 1}, 'articular_disc': {0, 1},
    'quick_master': {0}, 'opening_movements': {0},
    'clinical_research_skills': {0}, 'zayan_education': {0},
    'passive_movement': {0}, 'soft_hard_end_feel': {0},
}

NOTES = {
 'mckinney':'Còn phối hợp chăm sóc bệnh viện và báo cáo thống kê/quản lý ngân sách ở đoạn khác; chỉ giữ context đầy đủ.',
 'epidemiology':'Hướng dẫn mẹ chăm con và giảm tử vong trẻ vẫn hỗ trợ một phần mục đích chương trình Baker.',
 'feeding_position':'Tiếp xúc răng phía trước và thay đổi đường đóng còn có thể hỗ trợ suy luận hướng hàm.',
 'hinge_axis':'Phần sau vẫn liên hệ trục tự ý với cung mặt đơn giản và các phương pháp xác định.',
 'dental_board':'Các câu hỏi về lợi ích/bảo vệ công chúng tiếp tục nhắc lại nhiệm vụ Board.',
 'global_care':'Nghiên cứu cai thuốc lá còn lại hỗ trợ một phần câu hỏi hoạt động WHO.',
 'regression_example':'Câu sau vẫn mô tả suy luận từ hồi quy ra dân số.',
 'program_baseline':'Các câu kế tiếp còn nói đo mức đạt mục tiêu và hiệu quả chương trình.',
 'central_relation':'Định nghĩa hàm-sọ lặp lại; phần cơ duỗi cũng gây nhập nhằng với câu hỏi cơ căng.',
 'border_contact_movements':'Phần mô tả hình bao và giới hạn vận động tiếp tục hỗ trợ khái niệm vận động biên.',
 'closing_arc_limits':'Chú thích hình lặp lại khác biệt trục/cung; phần phục hồi sai tiếp tục hỗ trợ hạn chế.',
 'articular_disc':'Nguồn I2 nhắc lại mô sợi; hình ở I1 còn ghi buồng trên/dưới nên có hỗ trợ một phần ý thứ hai.',
 'quick_master':'Chú thích hình lặp lại mặt phẳng Frankfort.',
 'opening_movements':'Chú thích hình lặp lại hướng dịch chuyển của lồi cầu.',
 'learning':'Khi ẩn một fact phải bỏ cả ví dụ trực tiếp tương ứng; không chỉ bỏ câu định nghĩa.',
 'swallowing':'Khi ẩn nuốt không điển hình phải bỏ cả mô tả lưỡi nằm giữa răng phía sau.',
 'flexors_extensors':'Chỉ giữ vế duỗi sẽ mất tiền ngữ hai xương; loại riêng keep=[1].',
 'clinical_research_skills':'Phần lời khuyên cuối còn nêu lãnh đạo, làm việc nhóm; tránh nhãn âm cho câu hỏi kỹ năng rộng.',
 'zayan_education':'Phần chọn ngành còn liên hệ chăm sóc/giáo dục; không gán thiếu toàn bộ ý trải nghiệm.',
 'passive_movement':'Nguồn nhiễu end feel cũng mô tả há dưới hướng dẫn thầy thuốc, hỗ trợ một phần ý đầu.',
 'soft_hard_end_feel':'Phần trước còn mô tả mức há có hướng dẫn lớn hơn tự há trong giới hạn do cơ; giữ ý đầu để tránh âm mơ hồ.',
}


def allowed_keep(family, kept):
    return REQUIRE_PRESENT.get(family, set()) <= set(kept) and not (family=='flexors_extensors' and kept==[1])


def additional_removal(family, kept, text):
    if family == 'learning':
        return (0 not in kept and text.startswith('This theory suggests')) or (
            1 not in kept and text.startswith('For example, if a child is present'))
    if family == 'swallowing' and 1 not in kept:
        return text.startswith('Đây là kiểu nuốt liên quan') or text.startswith('Động tác nuốt trống không điển hình cũng')
    return False
