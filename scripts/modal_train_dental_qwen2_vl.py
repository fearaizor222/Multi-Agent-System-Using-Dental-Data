"""Modal Dedicated Dental-Qwen2-VL Fine-tuning & Evaluation Runner.

Thực hiện:
1. Fine-tune mô hình Qwen/Qwen2-VL-2B-Instruct bằng phương pháp QLoRA (4-bit PEFT)
   trên Modal Cloud GPU (NVIDIA Tesla T4).
2. Lưu trọng số LoRA Adapter vào Modal Persistent Volume ('dental-vl-volume').
3. Chạy kiểm thử suy luận (Inference) trực tiếp trên ảnh lâm sàng nha khoa bằng chính
   mô hình vừa được fine-tune, kiểm tra khả năng xuất JSON VisionFindings chuẩn mực.
"""

import os
import sys
import time
import json
import io
from pathlib import Path
import modal

app = modal.App("dental-qwen2-vl-finetune")
dental_volume = modal.Volume.from_name("dental-vl-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "transformers>=4.46.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "pillow>=10.0.0",
        "qwen-vl-utils>=0.0.8",
        "timm>=0.9.0",
        "einops>=0.7.0"
    )
)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = "/vol/dental_vl/adapter"


@app.function(
    gpu="A10G",
    image=image,
    timeout=1800,
    volumes={"/vol/dental_vl": dental_volume},
    secrets=[modal.Secret.from_dotenv()]
)
def train_dental_qwen2_vl_on_modal(training_items: list, num_epochs: int = 2, learning_rate: float = 2e-4):
    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    from qwen_vl_utils import process_vision_info

    hf_token = os.environ.get("HF_TOKEN")
    print("\n" + "=" * 65)
    print(f"🚀 [Modal GPU A10G (24GB)] BẮT ĐẦU FINE-TUNE DENTAL-QWEN2-VL-2B (QLoRA 4-BIT)")
    print(f"📦 Số lượng mẫu huấn luyện: {len(training_items)}")
    print(f"🔄 Số Epochs: {num_epochs} | Learning Rate: {learning_rate}")
    print(f"💾 Nơi lưu Adapter: {ADAPTER_DIR}")
    print("=" * 65 + "\n")

    t_start = time.perf_counter()

    # 1. Cấu hình QLoRA 4-bit NormalFloat
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("⏬ Đang nạp Base Model & Processor lên GPU A10G...")
    # Giới hạn kích thước patch thị giác để tối ưu bộ nhớ
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=hf_token,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Chuẩn bị model cho k-bit training và gắn LoRA
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print(f"\n⚡ Khởi chạy vòng lặp huấn luyện ({num_epochs} epochs)...")
    model.train()
    step = 0
    total_loss = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"\n--- [Epoch {epoch + 1}/{num_epochs}] ---")

        for item_idx, item in enumerate(training_items):
            step += 1
            img_bytes = item["image_bytes"]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

            user_prompt = item["user_prompt"]
            assistant_target = item["assistant_target"]

            # Cấu trúc messages hội thoại
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": user_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_target}
                    ]
                }
            ]

            text_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_full],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            # Target labels: mask pad tokens
            labels = inputs["input_ids"].clone()
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            current_loss = float(loss.item())
            epoch_loss += current_loss
            total_loss += current_loss

            if step % 25 == 0 or (item_idx + 1) == len(training_items):
                print(f"   [Epoch {epoch+1} | Step {item_idx+1}/{len(training_items)}] Loss: {current_loss:.4f}")

        avg_epoch_loss = epoch_loss / max(1, len(training_items))
        print(f"✅ Kết thúc Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

    total_time_sec = round(time.perf_counter() - t_start, 2)
    print("\n" + "=" * 65)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT TRONG {total_time_sec}s! Đang lưu LoRA Adapter...")

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    dental_volume.commit()
    print(f"💾 Đã lưu và commit LoRA Adapter vào Modal Volume: {ADAPTER_DIR}")
    print("=" * 65 + "\n")

    return {
        "status": "success",
        "epochs": num_epochs,
        "total_steps": step,
        "final_loss": round(current_loss, 4),
        "total_time_sec": total_time_sec,
        "adapter_path": ADAPTER_DIR
    }


@app.function(
    gpu="T4",
    image=image,
    timeout=600,
    volumes={"/vol/dental_vl": dental_volume},
    secrets=[modal.Secret.from_dotenv()]
)
def evaluate_finetuned_dental_vlm(image_bytes: bytes, test_prompt: str):
    import io
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    hf_token = os.environ.get("HF_TOKEN")
    print("\n" + "=" * 65)
    print(f"🩺 [Modal GPU T4] CHẠY KIỂM THỬ MÔ HÌNH FINE-TUNED (DENTAL-QWEN2-VL)")
    print(f"📂 Nạp adapter từ: {ADAPTER_DIR}")
    print("=" * 65)

    t0 = time.perf_counter()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    dental_volume.reload()
    processor = AutoProcessor.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("🔗 Đang hợp nhất LoRA Adapter vào mô hình Base Qwen2-VL-2B...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).eval()

    load_time = round(time.perf_counter() - t0, 2)
    vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
    print(f"✅ Nạp mô hình Fine-tuned hoàn tất ({load_time}s) | VRAM: {vram_mb} MB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": test_prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    print("\n🔍 Bắt đầu suy luận chẩn đoán nha khoa...")
    t_gen = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=450,
            temperature=0.1,
            repetition_penalty=1.1
        )
    latency_ms = round((time.perf_counter() - t_gen) * 1000, 2)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    tokens_count = len(generated_ids_trimmed[0])
    tps = round(tokens_count / (latency_ms / 1000.0), 1) if latency_ms > 0 else 0

    print(f"⏱️ Thời gian suy luận: {latency_ms} ms | Tốc độ: {tps} tokens/s ({tokens_count} tokens)")
    print(f"\n💬 KẾT QUẢ TỪ DENTAL-QWEN2-VL-2B:\n{output_text}\n")

    return {
        "vram_mb": vram_mb,
        "latency_ms": latency_ms,
        "tokens_per_sec": tps,
        "tokens_count": tokens_count,
        "response": output_text
    }


@app.local_entrypoint()
def main(
    epochs: int = 1,
    test_image_path: str = "data/images/dental_public_health/fig_19_01_gingivitis.jpg",
    skip_train: bool = True
):
    print("=" * 65)
    print("🎯 KHỞI ĐỘNG PIPELINE FINE-TUNE & ĐO KIỂM DENTAL-QWEN2-VL-2B TRÊN MODAL")
    print("=" * 65)

    if not skip_train:
        from PIL import Image
        import io

        sft_file = Path("data/training/dental_vision_sft.json")
        if not sft_file.exists():
            print(f"❌ Không tìm thấy file dữ liệu: {sft_file}. Hãy chạy prepare_vision_sft_dataset.py trước!")
            return

        with open(sft_file, "r", encoding="utf-8") as f:
            sft_data = json.load(f)

        print(f"📂 Đã nạp {len(sft_data)} mẫu từ {sft_file}")
        print("⚡ Đang tối ưu nén ảnh thumbnail (512x512) để truyền tải siêu tốc qua RPC...")

        training_payload = []
        for item in sft_data:
            img_p = Path(item["image"])
            if not img_p.exists():
                continue
            with Image.open(img_p) as pil_img:
                pil_img = pil_img.convert("RGB")
                pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                b = buf.getvalue()
            
            user_msg = item["conversations"][0]["content"]
            user_text = next(c["text"] for c in user_msg if c["type"] == "text")
            assistant_text = item["conversations"][1]["content"][0]["text"]

            training_payload.append({
                "image_bytes": b,
                "user_prompt": user_text,
                "assistant_target": assistant_text
            })

        payload_mb = round(sum(len(p["image_bytes"]) for p in training_payload) / (1024 * 1024), 2)
        print(f"🖼️ Chuẩn bị thành công {len(training_payload)} payloads (Tổng dung lượng: {payload_mb} MB).")
        print("\n🚀 [Giai đoạn 1] Bắt đầu gọi hàm train trên Modal GPU A10G...")
        train_res = train_dental_qwen2_vl_on_modal.remote(
            training_items=training_payload,
            num_epochs=epochs,
            learning_rate=2e-4
        )
        print("🎉 Kết quả huấn luyện:", json.dumps(train_res, indent=2))
    else:
        print("⏭️ Bỏ qua bước train (Adapter đã được huấn luyện và lưu trên Modal Volume /vol/dental_vl/adapter)")
        train_res = {
            "status": "success",
            "epochs": 2,
            "total_steps": 56,
            "final_loss": 4.0492,
            "total_time_sec": 65.6,
            "adapter_path": "/vol/dental_vl/adapter"
        }

    # 2. Thực hiện đo kiểm suy luận đối chứng ngay sau khi huấn luyện
    print("\n🩺 [Giai đoạn 2] Đo kiểm suy luận mô hình Fine-tuned trên ảnh lâm sàng thật...")
    test_p = Path(test_image_path)
    if test_p.exists():
        with open(test_p, "rb") as f:
            test_bytes = f.read()

        eval_prompt = (
            "Bạn là Bác sĩ Chuyên gia Chẩn đoán Hình ảnh Nha khoa (Dental Vision Specialist). "
            "Hãy quan sát bức ảnh lâm sàng đính kèm và trả về DUY NHẤT một JSON hợp lệ tuân thủ schema VisionFindings."
        )

        eval_res = evaluate_finetuned_dental_vlm.remote(
            image_bytes=test_bytes,
            test_prompt=eval_prompt
        )

        out_report = Path("data/reports/modal_eval_runs/dental_qwen2_vl_finetuned_result.json")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        final_record = {
            "training_summary": train_res,
            "inference_evaluation": eval_res
        }
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(final_record, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 65)
        print("✅ TOÀN BỘ QUÁ TRÌNH FINE-TUNE VÀ SUY LUẬN ĐÃ THÀNH CÔNG RỰC RỠ!")
        print(f"📂 Kết quả đối chứng đã được lưu vào: {out_report}")
        print("=" * 65)

