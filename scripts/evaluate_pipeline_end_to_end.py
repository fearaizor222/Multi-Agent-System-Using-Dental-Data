import sys
import re
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.graph.full_dental_graph import create_full_5agent_pipeline

def evaluate_and_score(question: str, final_response: str) -> dict:
    score = 10
    penalties = []
    highlights = []

    # 1. Kiểm tra ngôn ngữ và ảo giác
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", final_response))
    if has_chinese:
        score -= 4
        penalties.append("Chứa chữ Hán / tiếng Trung (-4)")

    for hallucination in ["kem chống nắng", "tiêu chảy", "suy dinh dưỡng", "thiếu oxy", "mất âm đạo", "sa sút trí tuệ"]:
        if hallucination in final_response.lower():
            score -= 3
            penalties.append(f"Ảo giác phi y khoa: '{hallucination}' (-3)")

    # 2. Kiểm tra chuẩn xác y khoa theo câu hỏi
    q_low = question.lower()
    res_low = final_response.lower()

    if "sâu" in q_low:
        # Giai đoạn sớm
        if any(w in res_low for w in ["fluor", "tái khoáng"]):
            highlights.append("Đã nêu giải pháp tái khoáng hóa / Fluor cho sâu răng sớm (+)")
        else:
            score -= 2
            penalties.append("Chưa nêu giải pháp tái khoáng hóa / Fluor cho sâu sớm (-2)")

        # Giai đoạn lỗ sâu
        if any(w in res_low for w in ["hàn", "trám", "lỗ sâu", "phục hồi", "mô cứng"]):
            highlights.append("Đã nêu giải pháp hàn trám phục hồi mô cứng cho lỗ sâu (+)")
        else:
            score -= 2
            penalties.append("Chưa nêu giải pháp hàn trám phục hồi mô cứng (-2)")

        # Khuyến cáo nha sĩ
        if any(w in res_low for w in ["nha sĩ", "bác sĩ", "khám", "cơ sở", "phòng khám"]):
            highlights.append("Có khuyến cáo khám nha sĩ chuyên môn (+)")

    elif "thái dương hàm" in q_low:
        if any(w in res_low for w in ["máng nhai", "khớp cắn", "vật lý trị liệu", "thăm khám", "tập há"]):
            highlights.append("Đã nêu can thiệp khớp cắn / máng nhai / cơ khớp (+)")
        else:
            score -= 2
            penalties.append("Chưa nêu can thiệp máng nhai / khớp cắn (-2)")

        if "cắt toàn bộ u" in res_low:
            score -= 4
            penalties.append("Nhầm lẫn sang phẫu thuật u tuyến (-4)")

    score = max(0, min(10, score))
    return {
        "score": score,
        "penalties": penalties,
        "highlights": highlights
    }

print("=" * 80)
print("🚀 KHỞI ĐỘNG VÀ ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG 5 AGENT (OFFLINE LOCAL CPU)")
print("=" * 80)

pipeline = create_full_5agent_pipeline(offline_mode=True)

test_cases = [
    {
        "name": "Ca lâm sàng 1: Bệnh lý Sâu Răng",
        "question": "các phương án xử lý 1 cái răng sâu",
        "image": None
    },
    {
        "name": "Ca lâm sàng 2: Rối loạn Khớp Thái Dương Hàm",
        "question": "tôi bị đau khớp thái dương hàm và há miệng kêu lục cục thì chữa thế nào",
        "image": None
    }
]

overall_scores = []

for case in test_cases:
    print("\n" + "#" * 80)
    print(f"📋 {case['name']}")
    print(f"👉 Câu hỏi: \"{case['question']}\"")
    print("#" * 80)

    start_time = time.perf_counter()
    state = pipeline.invoke({
        "question": case["question"],
        "image_path": case["image"],
        "trace": []
    })
    elapsed = time.perf_counter() - start_time

    print("\n⏱️ Tiến trình 5 Agent (Trace):")
    for step in state.get("trace", []):
        print(f"  • {step['step']}: {step['status']} ({step['duration_ms']:.1f}ms)")

    print(f"\n📚 Bằng chứng Agent 2 (Kho đóng nội bộ): {len(state.get('internal_evidence', []))} đoạn")
    for d in state.get("internal_evidence", []):
        print(f"    - [{d.get('evidence_id')}] {d.get('title')}")

    print(f"\n🌐 Bằng chứng Agent 3 (Kho mở Bộ Y Tế): {len(state.get('public_evidence', []))} đoạn")
    for d in state.get("public_evidence", []):
        print(f"    - [{d.get('evidence_id')}] {d.get('title')}")

    print("\n🩺 KẾT QUẢ TƯ VẤN CUỐI CÙNG (AGENT 5 DRAFTING):")
    print("-" * 75)
    final_resp = state.get("final_response", "")
    print(final_resp)
    print("-" * 75)

    eval_res = evaluate_and_score(case["question"], final_resp)
    overall_scores.append(eval_res["score"])

    print(f"\n🎯 ĐIỂM TỰ ĐÁNH GIÁ CHUẨN XÁC: {eval_res['score']}/10")
    if eval_res["highlights"]:
        print(f"  ✅ Điểm mạnh: {', '.join(eval_res['highlights'])}")
    if eval_res["penalties"]:
        print(f"  ⚠️ Điểm trừ: {', '.join(eval_res['penalties'])}")
    print(f"⏱️ Tổng thời gian thực thi ca bệnh: {elapsed:.2f}s")

print("\n" + "=" * 80)
print(f"🏁 TỔNG KẾT THỬ NGHIỆM: ĐIỂM TRUNG BÌNH HỆ THỐNG = {sum(overall_scores)/len(overall_scores):.1f}/10")
print("=" * 80)
