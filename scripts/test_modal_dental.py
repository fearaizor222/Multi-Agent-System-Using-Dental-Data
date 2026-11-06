"""Kiểm thử kết nối và suy luận đa Agent Nha khoa trên Modal Cloud GPU.

Cách sử dụng:
1. Khi chưa deploy:
   python scripts/test_modal_dental.py --check-auth
2. Khi đã có URL endpoint Modal:
   python scripts/test_modal_dental.py --url https://<your-subdomain>.modal.run
   hoặc để trống nếu đã cấu hình MODAL_API_URL trong file .env
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import httpx
except ImportError:
    print("Vui lòng cài đặt httpx: pip install httpx")
    sys.exit(1)

# Danh sách test cases lâm sàng thực tế cho từng Agent
DENTAL_TEST_CASES = [
    {
        "agent_id": "agent1_supervisor",
        "title": "Agent 1: Supervisor & Triage Cấp cứu",
        "prompt": "Bệnh nhân sốt 39.5 độ C, sưng phồng lớn vùng sàn miệng và góc hàm dưới lan xuống cổ, nuốt đau và khó thở, há miệng hạn chế dưới 1.5 cm. Đánh giá cấp độ triage và cờ đỏ.",
    },
    {
        "agent_id": "agent2_internal_rag",
        "title": "Agent 2: Bác sĩ Chuyên khoa Cắn khớp & TMJ",
        "prompt": "Bệnh nhân có tiếng lục cục rõ ở khớp thái dương hàm bên phải khi há to, cơ cắn mỏi nhiều sau khi ngủ dậy. Cơ chế và hướng xử trí ban đầu là gì?",
    },
    {
        "agent_id": "agent3_public_rag",
        "title": "Agent 3: Phác đồ Kỹ thuật Điều trị Bộ Y Tế",
        "prompt": "Phác đồ điều trị sâu ngà nông và quy trình trám bít theo hướng dẫn chuyên môn Bộ Y Tế.",
    },
    {
        "agent_id": "agent5_drafting",
        "title": "Agent 5: Trưởng khoa Lâm sàng Soạn thảo Tư vấn",
        "prompt": "Tổng hợp kết quả tư vấn ngắn gọn, ân cần cho bệnh nhân bị viêm nướu chảy máu kèm ê buốt cổ răng chân răng.",
    },
]


def check_modal_auth():
    print("=" * 65)
    print("🔍 KIỂM TRA TRẠNG THÁI MODAL CLI TRÊN MÁY TÍNH")
    print("=" * 65)
    import subprocess
    try:
        res = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True, check=True)
        profile_name = res.stdout.strip()
        print(f"✅ Modal CLI đã sẵn sàng! Profile hiện tại: '{profile_name}'")
        return True
    except Exception as e:
        print("⚠️ Chưa tìm thấy xác thực Modal!")
        print("👉 Vui lòng mở terminal và gõ lệnh sau để đăng nhập (chỉ mất 5 giây):")
        print("   modal setup")
        return False


def test_endpoint(base_url: str):
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU KIỂM THỬ MODAL CLOUD GPU ENDPOINT: {base_url}")
    print("=" * 65)

    base_url = base_url.rstrip("/")
    # Nếu URL đã là endpoint generate chuyên biệt của Modal
    if "-generate." in base_url:
        generate_url = base_url
        health_url = base_url.replace("-generate.", "-health.")
    else:
        generate_url = base_url
        health_url = f"{base_url}/health"

    # 1. Healthcheck
    print(f"\n1️⃣ Đang kiểm tra Healthcheck GPU ({health_url})...")
    try:
        t0 = time.perf_counter()
        resp = httpx.get(health_url, timeout=30.0)
        t_health = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            print(f"   ✅ Server Online! ({t_health:.1f}ms)")
            print(f"   ℹ️ Thông tin GPU & Model: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"   ⚠️ Healthcheck trả về mã {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"   ⚠️ Không thể gọi healthcheck ({e}). Vẫn tiếp tục thử nghiệm endpoint generate...")

    # 2. Chạy test các agent
    print("\n2️⃣ Bắt đầu kiểm thử suy luận từng Agent trên GPU NVIDIA T4...")
    for i, test in enumerate(DENTAL_TEST_CASES, 1):
        print("-" * 65)
        print(f"🧪 Test {i}/4: {test['title']} (ID: {test['agent_id']})")
        print(f"📝 Câu hỏi: {test['prompt']}")

        payload = {
            "agent_id": test["agent_id"],
            "prompt": test["prompt"],
            "max_new_tokens": 256,
            "temperature": 0.2
        }

        try:
            t0 = time.perf_counter()
            resp = httpx.post(generate_url, json=payload, timeout=90.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                print(f"⏱️ Tổng thời gian phản hồi: {elapsed_ms:.1f}ms (Thời gian GPU suy luận: {data.get('latency_ms', 0)}ms)")
                print(f"⚡ Tốc độ sinh token: {data.get('tokens_per_sec', 0)} tokens/giây ({data.get('tokens_generated', 0)} tokens)")
                print(f"🤖 Mô hình: {data.get('model_name')}")
                print(f"💬 Phản hồi y khoa:")
                print(data.get("response", "").strip())
            else:
                print(f"❌ Lỗi HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ Lỗi kết nối: {e}")

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT KIỂM THỬ HỆ THỐNG TRÊN MODAL GPU!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Modal Dental Multi-Agent Server")
    parser.add_argument("--url", type=str, default=None, help="URL của Modal endpoint")
    parser.add_argument("--check-auth", action="store_true", help="Kiểm tra đăng nhập Modal")
    args = parser.parse_args()

    if args.check_auth:
        check_modal_auth()
        sys.exit(0)

    url = args.url
    if not url:
        # 1. Đọc trực tiếp từ file .env
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MODAL_API_URL="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                url = val
                                break
            except Exception:
                pass

    if not url:
        # 2. Thử đọc từ src.core.config
        try:
            from src.core.config import settings
            url = getattr(settings, "MODAL_API_URL", None)
        except Exception:
            pass

    # Tự động sửa nếu bị thiếu đuôi .run do copy
    if url and url.endswith(".modal"):
        url = url + ".run"

    if not url:
        print("⚠️ Chưa có URL Modal!")
        print("Bạn có thể cung cấp qua: python scripts/test_modal_dental.py --url <URL_CUA_BAN>")
        print("Hoặc thêm vào file .env: MODAL_API_URL=<URL_CUA_BAN>")
        print("\nKiểm tra trạng thái đăng nhập Modal...")
        check_modal_auth()
    else:
        test_endpoint(url)

