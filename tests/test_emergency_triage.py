"""Bộ kiểm thử phân loại Triage Cấp cứu Nha khoa cho Agent 1 (Supervisor Agent)."""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pytest
from src.agents.supervisor import SupervisorAgent, clinical_rule_triage


def test_user_emergency_prompt_swelling_bleeding_severe_pain():
    """Kiểm tra ca bệnh thực tế người dùng phản ánh: 'mặt tôi sưng, chảy máu và rất nhức răng...'"""
    prompt = "mặt tôi sưng, chảy máu và rất nhức răng..."
    agent = SupervisorAgent()
    decision = agent.evaluate(prompt)

    assert decision.is_emergency is True, "Phải bật cờ is_emergency = True"
    assert decision.emergency_level == "critical", "Mức độ phải là critical vì có cả sưng mặt, chảy máu và đau dữ dội"
    assert decision.intent == "emergency_triage"
    assert "sưng" in decision.triage_reason.lower() or "viêm mô tế bào" in decision.triage_reason.lower()
    print("\n[PASSED] User emergency case:", decision.model_dump())


def test_airway_compromise_emergency():
    """Dấu hiệu nguy kịch đe dọa đường thở: Sưng to lan hốc mắt kèm sốt cao khó thở."""
    prompt = "Má bên phải tôi sưng to húp cả mắt, sốt 39 độ và nuốt rất đau khó thở"
    decision = clinical_rule_triage(prompt)

    assert decision is not None
    assert decision.is_emergency is True
    assert decision.emergency_level == "critical"
    assert decision.intent == "emergency_triage"
    print("\n[PASSED] Airway compromise case:", decision.model_dump())


def test_dental_trauma_avulsion():
    """Chấn thương rơi răng ra ngoài ổ răng (Avulsion)."""
    prompt = "Tôi vừa bị ngã xe đập mặt xuống đường, răng cửa văng ra ngoài còn nguyên chân"
    decision = clinical_rule_triage(prompt)

    assert decision is not None
    assert decision.is_emergency is True
    assert decision.emergency_level == "urgent"
    assert "rơi" in decision.triage_reason.lower() or "avulsion" in decision.triage_reason.lower()
    print("\n[PASSED] Avulsion trauma case:", decision.model_dump())


def test_emergency_with_image_attachment():
    """Khi người bệnh gửi kèm hình ảnh nhưng triệu chứng là cấp cứu thì KHÔNG ĐƯỢC tắt cờ cấp cứu."""
    prompt = "mặt tôi sưng to một bên và nhức răng buốt óc"
    agent = SupervisorAgent()
    decision = agent.evaluate(prompt, image_path="data/uploads/sample.jpg")

    assert decision.is_emergency is True, "Gửi kèm ảnh vẫn phải giữ cờ cấp cứu nếu triệu chứng nguy hiểm"
    assert decision.intent == "emergency_triage"
    print("\n[PASSED] Emergency with image attachment case:", decision.model_dump())


def test_routine_dental_inquiry():
    """Bệnh lý nha khoa thông thường: Sâu răng ngà nông/sâu không cấp cứu."""
    prompt = "Răng hàm của tôi bị đau buốt khi ăn đồ ngọt hoặc uống nước đá lạnh, soi thấy lỗ sâu đen"
    decision = clinical_rule_triage(prompt)

    # Không nằm trong nhóm cấp cứu
    assert decision is None, "Câu hỏi sâu răng thông thường không được bật cấp cứu trong Rule Engine"
    print("\n[PASSED] Routine inquiry case passed.")


def test_service_info_inquiry():
    """Câu hỏi thông tin giờ làm việc hoặc chi phí dịch vụ."""
    prompt = "Phòng khám có mở cửa ngày chủ nhật không và giá tẩy trắng răng là bao nhiêu tiền?"
    decision = clinical_rule_triage(prompt)

    assert decision is not None
    assert decision.is_emergency is False
    assert decision.intent == "service_info"
    print("\n[PASSED] Service info inquiry case:", decision.model_dump())


if __name__ == "__main__":
    test_user_emergency_prompt_swelling_bleeding_severe_pain()
    test_airway_compromise_emergency()
    test_dental_trauma_avulsion()
    test_emergency_with_image_attachment()
    test_routine_dental_inquiry()
    test_service_info_inquiry()
    print("\n🎉 TOÀN BỘ CÁC TEST CASES TRIAGE CẤP CỨU ĐÃ VƯỢT QUA 100%!")
