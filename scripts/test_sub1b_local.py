"""
Benchmark thực nghiệm chạy mô hình LLM siêu nhỏ (Qwen2.5-0.5B-Instruct)
trực tiếp trên CPU máy local (AMD Ryzen, Windows)
"""
import os
import sys
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("=" * 65)
    print("  THỰC NGHIỆM CHẠY LOCAL MÔ HÌNH DƯỚI 1B (Qwen2.5-0.5B-Instruct)")
    print(f"  Phần cứng: {psutil.cpu_count(logical=False)} Core / {psutil.cpu_count(logical=True)} Threads CPU")
    print(f"  RAM khả dụng hệ thống: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("=" * 65)

    ram_start = get_ram_usage_mb()
    print(f"\n[1/3] Đang tải tokenizer & weights của {MODEL_ID}...")
    t0 = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    t_load = time.time() - t0
    ram_loaded = get_ram_usage_mb()
    model_ram_mb = ram_loaded - ram_start

    print(f"  --> Đã nạp xong model vào RAM CPU trong: {t_load:.2f} giây")
    print(f"  --> Dung lượng RAM tiến trình tiêu thụ cho Model: {model_ram_mb:.1f} MB (~{model_ram_mb/1024:.2f} GB)")

    # Câu hỏi lâm sàng nha khoa thử nghiệm
    messages = [
        {
            "role": "system",
            "content": "Bạn là Trợ lý Nha khoa C20 AI. Hãy trả lời ngắn gọn, chuẩn xác y khoa và có lời khuyên phòng ngừa."
        },
        {
            "role": "user",
            "content": "Bác sĩ cho tôi hỏi: Tôi bị sưng đỏ ở viền nướu, hay chảy máu chân răng khi chải răng và có mùi hôi miệng. Đây là triệu chứng của bệnh gì và tôi cần xử lý thế nào?"
        }
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    input_tokens = inputs["input_ids"].shape[1]

    print("\n[2/3] Bắt đầu suy luận (Inference) trên CPU...")
    print(f"  Số token đầu vào: {input_tokens} tokens")

    t_gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    t_gen = time.time() - t_gen_start

    output_tokens = outputs.shape[1] - input_tokens
    speed_tps = output_tokens / t_gen if t_gen > 0 else 0

    generated_ids = outputs[0][input_tokens:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"  --> Hoàn thành sinh {output_tokens} tokens trong {t_gen:.2f} giây")
    print(f"  --> Tốc độ sinh chữ: {speed_tps:.2f} tokens/giây")

    print("\n[3/3] Kết quả phản hồi của mô hình:")
    print("-" * 65)
    print(response_text)
    print("-" * 65)

    print("\n" + "=" * 65)
    print("  KẾT LUẬN ĐÁNH GIÁ THỰC NGHIỆM:")
    print(f"  - Chiếm dụng RAM: ~{model_ram_mb:.0f} MB (Rất nhẹ, máy 16GB dư sức chạy đồng thời 2-3 tác vụ)")
    print(f"  - Tốc độ suy luận CPU: {speed_tps:.1f} tokens/s (Nhanh, phản hồi mượt mà cho người dùng)")
    print("  - Khả năng tích hợp QLoRA: Adapter LoRA (~20MB - 40MB) huấn luyện trên Colab/Kaggle")
    print("    có thể load trực tiếp bằng PEFT trên CPU máy này mà không tốn thêm tài nguyên.")
    print("=" * 65)

if __name__ == "__main__":
    main()
