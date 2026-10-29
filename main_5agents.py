"""CLI thực thi Hệ thống 5 Agent Nha khoa Chuẩn C20 (Hỗ trợ Interactive Chat & Offline Mode)
1. Agent 1: Supervisor (Điều phối ý định & Triage cấp cứu)
2. Agent 2: Closed-data RAG (Dữ liệu đóng: Giáo trình Cắn khớp học RHM)
3. Agent 3: Open-data RAG (Dữ liệu mở: Hướng dẫn điều trị RHM Bộ Y Tế)
4. Agent 4: Dental Vision (Xử lý ảnh lâm sàng & X-quang)
5. Agent 5: Drafting Response (Soạn thảo bản nháp nhận định & Trả lời tư vấn)
"""
import argparse
import os
import sys
import time

# Cấu hình UTF-8 toàn diện cho console Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.graph.full_dental_graph import create_full_5agent_pipeline


def print_banner():
    banner = """
================================================================================
 🦷 HỆ THỐNG 5 AGENT NHA KHOA ĐA PHƯƠNG THỨC — NHÓM C20
 [A1] Supervisor ➔ [A4] Vision ➔ [A2] Closed RAG ➔ [A3] Open RAG ➔ [A5] Drafting
================================================================================
 1. Agent 1: Supervisor (Điều phối ý định & Triage cấp cứu)
 2. Agent 2: Closed-data RAG (Dữ liệu đóng: Giáo trình Cắn khớp học RHM)
 3. Agent 3: Open-data RAG (Dữ liệu mở: Hướng dẫn điều trị RHM Bộ Y Tế)
 4. Agent 4: Dental Vision Agent (Phân tích ảnh lâm sàng khoang miệng & X-quang)
 5. Agent 5: Drafting Response Agent (Soạn thảo bản nháp nhận định & Trả lời)
================================================================================
"""
    print(banner)


def process_query(pipeline, question: str, image_path: str = None):
    """Thực thi một câu hỏi qua pipeline 5 Agent và in kết quả chi tiết."""
    print(f"\n👉 Câu hỏi người bệnh: \"{question}\"")
    if image_path:
        print(f"🖼️ Hình ảnh đính kèm: {image_path}")

    initial_state = {
        "question": question,
        "image_path": image_path,
        "trace": []
    }

    print("\n🚀 Bắt đầu điều phối luồng xử lý qua 5 Agent...\n" + "-" * 75)
    final_state = None
    for state in pipeline.stream(initial_state, stream_mode="values"):
        final_state = state

    print("-" * 75)
    print("\n📊 BÁO CÁO TIẾN TRÌNH THỰC THI (TRACE):")
    for step in final_state.get("trace", []):
        print(f"  • [{step['step']}]: {step['status'].upper()} (thời gian: {step['duration_ms']:.1f}ms)")

    # 1. Agent 1: Supervisor
    print("\n" + "=" * 75)
    print("👔 [AGENT 1: SUPERVISOR & TRIAGE]")
    print(f"  - Phân loại ý định: {final_state.get('intent')}")
    print(f"  - Cờ đỏ cấp cứu: {'🚨 CÓ NGUY CƠ KHẨN CẤP' if final_state.get('is_emergency') else '✅ Ổn định / Thông thường'}")
    print(f"  - Đánh giá: {final_state.get('triage_reason')}")

    # 2. Agent 4: Vision (nếu có ảnh)
    if final_state.get("vision_findings"):
        vf = final_state["vision_findings"]
        print("\n" + "=" * 75)
        print("👁️ [AGENT 4: DENTAL VISION VLM]")
        print(f"  - Loại ảnh: {vf.get('image_type')}")
        print(f"  - Vùng giải phẫu nhận diện: {', '.join(vf.get('anatomical_regions', []))}")
        print(f"  - Nhận định thị giác: {vf.get('preliminary_impression')}")
        for p in vf.get("detected_pathologies", []):
            print(f"    * {p.get('name')} tại {p.get('location')} ({p.get('severity')}): {p.get('visual_evidence')}")

    # 3. Agent 2: Closed-data RAG (Kho nội bộ)
    if final_state.get("internal_evidence"):
        print("\n" + "=" * 75)
        print(f"📚 [AGENT 2: CLOSED-DATA RAG (KHO NỘI BỘ - GIÁO TRÌNH RHM)]")
        print(f"  - Đã truy xuất {len(final_state['internal_evidence'])} đoạn chứng cứ:")
        for doc in final_state["internal_evidence"]:
            print(f"    * [{doc.get('evidence_id')}] {doc.get('title')}")

    # 4. Agent 3: Open-data RAG (Kho công khai)
    if final_state.get("public_evidence"):
        print("\n" + "=" * 75)
        print(f"🌐 [AGENT 3: OPEN-DATA RAG (KHO CÔNG KHAI - BỘ Y TẾ)]")
        print(f"  - Đã đối chiếu {len(final_state['public_evidence'])} đoạn chứng cứ:")
        for doc in final_state["public_evidence"]:
            print(f"    * [{doc.get('evidence_id')}] {doc.get('title')}")

    # 5. Agent 5: Drafting Response
    if final_state.get("key_claims"):
        print("\n" + "=" * 75)
        print("✍️ [AGENT 5: DRAFTING RESPONSE (CÁC NHẬN ĐỊNH CỐT LÕI)]")
        for c in final_state["key_claims"]:
            cits = ", ".join(c.get("citations", []))
            print(f"    * [{c.get('claim_id')}]: \"{c.get('text')}\" [Trích dẫn: {cits}]")

    # Kết quả tổng hợp cuối cùng
    print("\n" + "=" * 75)
    print("🩺 [KẾT QUẢ TƯ VẤN CUỐI CÙNG TỪ AGENT 5 CHO NGƯỜI BỆNH]")
    print("=" * 75)
    print(final_state.get("final_response", "Không có câu trả lời."))
    print("=" * 75 + "\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Chạy luồng 5 Agent Nha khoa C20")
    parser.add_argument("question", nargs="?", default=None,
                        help="Câu hỏi hoặc triệu chứng của bệnh nhân")
    parser.add_argument("--image", type=str, default=None,
                        help="Đường dẫn file ảnh lâm sàng hoặc X-quang nha khoa")
    parser.add_argument("--offline", action="store_true",
                        help="Chạy chế độ Offline 100%% trên Local CPU (Qwen2.5-0.5B + Local ONNX RAG, không cần API Key)")

    parser.add_argument("--emergency-demo", action="store_true",
                        help="Chạy thử nghiệm ca cấp cứu nha khoa khẩn cấp")
    parser.add_argument("--service-demo", action="store_true",
                        help="Chạy thử nghiệm câu hỏi hành chính/dịch vụ")
    args = parser.parse_args()

    try:
        if args.emergency_demo:
            initial_question = "Mặt tôi bị sưng to một bên hàm, sốt cao 39.5 độ và rất khó thở, nuốt đau nghẹn!"
            initial_image = None
        elif args.service_demo:
            initial_question = "Phòng khám có làm việc vào ngày Chủ nhật không và giá lấy cao răng là bao nhiêu?"
            initial_image = None
        else:
            initial_question = args.question
            initial_image = args.image

        if args.offline:
            print("⚡ CHẾ ĐỘ: 100% OFFLINE LOCAL CPU (Qwen2.5-0.5B + Local ONNX RAG + BM25)")
        else:
            print("🌐 CHẾ ĐỘ: HYBRID (Cloud VLM / LLM + Local ONNX RAG + BM25)")

        print("⏳ Đang khởi tạo hệ sinh thái 5 Agent và nạp tri thức y khoa...")
        start_init = time.perf_counter()
        pipeline = create_full_5agent_pipeline(offline_mode=args.offline)
        init_duration = time.perf_counter() - start_init
        print(f"✅ Hệ thống 5 Agent đã sẵn sàng! (Khởi tạo mất {init_duration:.2f}s)\n")

        # 1. Nếu có câu hỏi khởi tạo từ tham số dòng lệnh thì chạy trước
        if initial_question:
            process_query(pipeline, initial_question, image_path=initial_image)

        # 2. Vòng lặp tương tác (Interactive Loop) cho phép người dùng hỏi liên tục mà không bị tắt cửa sổ
        while True:
            print("\n" + "=" * 75)
            print("💡 Bạn có thể tiếp tục đặt câu hỏi nha khoa (hoặc gõ 'exit'/'q' để thoát):")
            try:
                user_input = input("👉 Câu hỏi của bạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã nhận tín hiệu dừng.")
                break

            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                print("Cảm ơn bạn đã sử dụng Hệ thống Trợ lý Nha khoa C20!")
                break

            # Kiểm tra xem người dùng có muốn đính kèm ảnh không
            img_input = None
            try:
                img_prompt = input("🖼️ Đường dẫn ảnh đính kèm (Nhấn Enter nếu không có ảnh): ").strip()
                if img_prompt and os.path.exists(img_prompt):
                    img_input = img_prompt
                elif img_prompt:
                    print(f"⚠️ Không tìm thấy file ảnh: {img_prompt}. Sẽ xử lý dưới dạng câu hỏi thuần văn bản.")
            except (EOFError, KeyboardInterrupt):
                pass

            process_query(pipeline, user_input, image_path=img_input)

    except Exception as err:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình thực thi: {err}")
        import traceback
        traceback.print_exc()

    finally:
        # Giữ cửa sổ console luôn mở để người dùng đọc kết quả
        try:
            input("\n" + "=" * 75 + "\n👋 Nhấn phím Enter để đóng cửa sổ...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
