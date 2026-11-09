"""Modal Cloud GPU Pipeline: Fine-tuning Qwen2.5-1.5B-Instruct (QLoRA 4-bit) cho Agent 5 (Drafting & Synthesis).

Pipeline bao gồm:
1. Đọc tập dữ liệu SFT (550 mẫu) từ data/training/agent5_drafting_sft.jsonl.
2. Cấu hình QLoRA (4-bit BitsAndBytes, LoRA rank=16, alpha=32) cho Qwen/Qwen2.5-1.5B-Instruct trên GPU A10G.
3. Huấn luyện SFT 2 Epochs bằng PyTorch Native loop (Target Token Masking) và lưu LoRA Adapter vào Modal Volume.
4. Kiểm thử suy luận (Inference test) tức thì sau khi huấn luyện xong trên ca bệnh nha khoa thực tế.
"""

import json
import os
import sys
import time
from pathlib import Path
import modal

app = modal.App("dental-agent5-drafting-finetune")
agent5_volume = modal.Volume.from_name("dental-agent5-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.46.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0"
    )
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "/vol/agent5/adapter"


@app.function(
    gpu="A10G",
    image=image,
    timeout=1800,
    volumes={"/vol/agent5": agent5_volume},
    secrets=[modal.Secret.from_dotenv()]
)
def train_agent5_drafting_on_modal(training_samples: list, num_epochs: int = 2, learning_rate: float = 2e-4):
    import torch
    import math
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )

    print("\n" + "=" * 70)
    print(f"🚀 [Modal GPU A10G (24GB)] BẮT ĐẦU FINE-TUNE AGENT 5 (DRAFTING & SYNTHESIS)")
    print(f"📦 Mô hình nền: {MODEL_ID}")
    print(f"📊 Số lượng mẫu SFT: {len(training_samples)}")
    print(f"🔄 Số Epochs: {num_epochs} | Learning Rate: {learning_rate}")
    print(f"💾 Nơi lưu Adapter: {ADAPTER_DIR}")
    print("=" * 70 + "\n")

    t_start = time.perf_counter()

    # 1. Cấu hình lượng tử hóa 4-bit (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("⏬ [1/4] Đang nạp Tokenizer & Mô hình nền 4-bit Quantization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 2. Cấu hình PEFT / LoRA
    print("🔧 [2/4] Cấu hình LoRA Adapter (rank=16, alpha=32)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Huấn luyện bằng Native PyTorch Loop với Target Masking
    print("🔥 [3/4] Bắt đầu vòng lặp huấn luyện SFT chuyên biệt...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    gradient_accumulation_steps = 4
    model.train()

    total_steps = len(training_samples) * num_epochs
    step = 0
    total_loss = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"\n--- [Epoch {epoch + 1}/{num_epochs}] ---")

        for idx, item in enumerate(training_samples):
            step += 1
            messages = item["messages"]

            # Prompt: system + user
            prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            # Full: system + user + assistant
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            encoded_prompt = tokenizer(prompt_text, add_special_tokens=False)
            encoded_full = tokenizer(full_text, max_length=2048, truncation=True, return_tensors="pt")

            input_ids = encoded_full["input_ids"].to(model.device)
            attention_mask = encoded_full["attention_mask"].to(model.device)

            # Target label masking: Chỉ tính loss trên phần trả lời của Assistant
            labels = input_ids.clone()
            prompt_len = len(encoded_prompt["input_ids"])
            if prompt_len < labels.shape[1]:
                labels[:, :prompt_len] = -100
            else:
                labels[:, :-1] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            current_loss = float(outputs.loss.item())
            epoch_loss += current_loss
            total_loss += current_loss

            if step % gradient_accumulation_steps == 0 or (idx + 1) == len(training_samples):
                # Cosine learning rate decay
                progress = step / total_steps
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate * max(lr_mult, 0.1)

                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(training_samples):
                avg_so_far = epoch_loss / (idx + 1)
                print(f"  Step [{idx + 1:3d}/{len(training_samples)}] | Batch Loss: {current_loss:.4f} | Avg Loss: {avg_so_far:.4f}")

        print(f"📊 Kết thúc Epoch {epoch + 1}: Loss trung bình = {epoch_loss / len(training_samples):.4f}")

    total_time = time.perf_counter() - t_start
    final_avg_loss = total_loss / total_steps
    print(f"\n🎉 Huấn luyện hoàn tất! Loss cuối cùng: {final_avg_loss:.4f} (Thời gian: {total_time:.2f}s)")

    # 4. Lưu LoRA Adapter vào Volume
    print(f"💾 [4/4] Đang lưu LoRA Adapter vào {ADAPTER_DIR}...")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    agent5_volume.commit()
    print("✅ LoRA Adapter đã được đồng bộ an toàn vào Modal Persistent Volume!")

    return {
        "status": "success",
        "training_loss": final_avg_loss,
        "elapsed_seconds": total_time,
        "adapter_dir": ADAPTER_DIR
    }


@app.function(
    gpu="A10G",
    image=image,
    timeout=600,
    volumes={"/vol/agent5": agent5_volume}
)
def evaluate_agent5_inference():
    """Kiểm thử suy luận của mô hình vừa fine-tune trên một ca lâm sàng cắn khớp & viêm nướu thực tế."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("\n" + "=" * 70)
    print("🧪 [KIỂM THỬ LÂM SÀNG] ĐÁNH GIÁ MÔ HÌNH AGENT 5 ĐÃ FINE-TUNE VỚI LORA ADAPTER")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    test_system = """Bạn là Trợ lý Thông tin Nha khoa C20 AI (Agent 5: Drafting & Consultation Synthesis Agent).
Nhiệm vụ: Căn cứ vào các ý kiến hội chẩn chuyên khoa và tài liệu y khoa được cung cấp, hãy trình bày bài hướng dẫn và tư vấn thông tin y tế chi tiết, chuẩn xác gửi tới người hỏi.
Chỉ trả về DUY NHẤT một JSON hợp lệ tuân thủ schema:
{
  "key_claims": [
    {"claim_id": "c1", "text": "...", "citations": ["I1", "P1"]}
  ],
  "draft_response": "Nội dung bài tư vấn hoàn chỉnh..."
}"""

    test_user = """Ý kiến hội chẩn của các Bác sĩ chuyên khoa:
👨‍⚕️ [Ý kiến Chuyên khoa Cắn khớp & TMJ (Agent 2)]:
Phát hiện dấu hiệu mòn ngót cổ răng vùng răng hàm nhỏ do lực uốn nhai quá tải kết hợp thói quen chải răng ngang.

👨‍⚕️ [Ý kiến Chuyên khoa Phác đồ Bộ Y Tế (Agent 3)]:
Áp dụng quy trình kỹ thuật trám phục hồi Composite theo phác đồ Bộ Y Tế để bảo vệ tủy răng và giảm ê buốt.

Tài liệu y khoa tham khảo:
- [I1] (Nội bộ: Tổn thương Mòn Cổ Răng do Lực Uốn): Ứng suất uốn thân răng phá vỡ men cổ răng làm lộ ngà nhạy cảm.
- [P1] (Bộ Y Tế: Phác đồ Trám Khuyết Cổ Răng): Làm sạch rãnh khuyết, xoi mòn và trám phục hồi kín bằng Composite đàn hồi.

Câu hỏi của người bệnh: "Răng số 14 của tôi bị khuyết một rãnh sâu sát nướu, hít gió lạnh vào là buốt nhói. Bác sĩ tư vấn cách chữa giúp tôi."

Bác sĩ hãy tổng hợp các tài liệu và kết quả thị giác trên để viết bài tư vấn nha khoa hoàn chỉnh gửi người bệnh.
Chỉ trả về DUY NHẤT một JSON hợp lệ tuân thủ schema:
{{
  "key_claims": [
    {{"claim_id": "c1", "text": "...", "citations": ["I1", "P1"]}}
  ],
  "draft_response": "Nội dung bài tư vấn hoàn chỉnh..."
}}"""

    messages = [
        {"role": "system", "content": test_system},
        {"role": "user", "content": test_user}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("🧠 Đang sinh câu trả lời...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.perf_counter() - t0

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n⏱️ Thời gian suy luận: {gen_time:.2f}s")
    print(f"📄 Kết quả thô từ mô hình:\n{output_text}\n")

    # Kiểm tra tính hợp lệ của JSON
    is_valid_json = False
    try:
        clean_text = output_text
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        parsed = json.loads(clean_text)
        is_valid_json = True
        print("✅ KIỂM TRA ĐỊNH DẠNG: JSON HỢP LỆ 100%!")
        print(f"- Số nhận định (key_claims): {len(parsed.get('key_claims', []))}")
        for c in parsed.get('key_claims', []):
            print(f"  + [{c.get('claim_id')}] {c.get('text')} (Trích dẫn: {c.get('citations')})")
        print(f"- Độ dài bài tư vấn: {len(parsed.get('draft_response', ''))} ký tự")
    except Exception as e:
        print(f"❌ Cảnh báo định dạng JSON: {e}")

    return {
        "valid_json": is_valid_json,
        "inference_seconds": gen_time,
        "output_text": output_text
    }


@app.local_entrypoint()
def main(epochs: int = 2, lr: float = 2e-4):
    data_path = Path("data/training/agent5_drafting_sft.jsonl")
    if not data_path.exists():
        print(f"❌ Không tìm thấy file dữ liệu SFT: {data_path}")
        print("Vui lòng chạy: python scripts/generate_agent5_drafting_dataset.py trước!")
        return

    print(f"📖 Đang đọc tập dữ liệu từ {data_path}...")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"📦 Đã tải {len(samples)} mẫu huấn luyện. Đang gửi lên Modal Cloud GPU...")
    res = train_agent5_drafting_on_modal.remote(samples, num_epochs=epochs, learning_rate=lr)
    print("\n🎯 KẾT QUẢ HUẤN LUYỆN TRÊN MODAL:")
    print(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n🔬 BẮT ĐẦU CHẠY THỬ NGHIỆM ĐÁNH GIÁ SUY LUẬN TRÊN MODAL...")
    eval_res = evaluate_agent5_inference.remote()
    print("\n🎯 KẾT QUẢ ĐÁNH GIÁ SUY LUẬN:")
    print(f"- JSON hợp lệ: {eval_res['valid_json']}")
    print(f"- Tốc độ suy luận: {eval_res['inference_seconds']:.2f}s")
    print("=" * 70)
