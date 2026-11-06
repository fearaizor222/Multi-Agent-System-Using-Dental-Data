"""Script Huấn Luyện Tinh Chỉnh (Fine-tuning) Agent 1 (Supervisor & Clinical Triage).

Sử dụng LoRA (PEFT) và HuggingFace Transformers để fine-tune mô hình
Qwen/Qwen2.5-0.5B-Instruct trên bộ dữ liệu cấp cứu nha khoa lâm sàng (>1300 mẫu).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model nền cần fine-tune (mặc định: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT_DIR / "data/training/agent1_triage_dataset.jsonl",
        help="Đường dẫn file dữ liệu JSONL huấn luyện",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data/models/agent1_supervisor_finetuned",
        help="Thư mục xuất adapter / checkpoint đã fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Độ dài chuỗi tối đa")
    parser.add_argument("--execute-train", action="store_true", help="Thực thi vòng lặp huấn luyện thực sự")
    return parser.parse_args()


def load_triage_samples(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu: {path}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def main():
    args = parse_args()
    print("=" * 70)
    print("   TIẾN TRÌNH FINE-TUNING AGENT 1: SUPERVISOR & CLINICAL TRIAGE")
    print("=" * 70)
    print(f"[*] Base Model   : {args.model_name}")
    print(f"[*] Dataset Path : {args.dataset_path}")
    print(f"[*] Output Dir   : {args.output_dir}")
    print(f"[*] Hyperparams  : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")

    samples = load_triage_samples(args.dataset_path)
    print(f"[✓] Đã nạp thành công {len(samples)} mẫu huấn luyện phân loại cấp cứu nha khoa.")

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"\n[!] Thiếu thư viện phụ thuộc ({e}). Cài đặt bằng lệnh:")
        print("    pip install peft accelerate datasets")
        print("\n[*] Dữ liệu huấn luyện (>1300 mẫu) đã sẵn sàng tại:", args.dataset_path)
        return

    print("\n[*] Đang tải Tokenizer & Cấu trúc Mô hình...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị phát hiện: {device.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    print("[*] Thông số tham số huấn luyện LoRA:")
    model.print_trainable_parameters()

    # Chuẩn bị dữ liệu định dạng Chat Template
    formatted_texts = []
    for s in samples:
        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append(text)

    print(f"[✓] Đã format {len(formatted_texts)} chuỗi Chat Template chuẩn Qwen2.5.")
    print("\n--- MẪU HUẤN LUYỆN ĐẦU TIÊN (PREVIEW) ---")
    print(formatted_texts[0][:280] + "...\n----------------------------------------")

    if not args.execute_train:
        print("\n[ℹ️] Chế độ thẩm định hoàn tất (Dry Run).")
        print("     Để tiến hành huấn luyện thực tế và lưu checkpoint, hãy chạy:")
        print(f"     python scripts/finetune_agent1_supervisor.py --execute-train --epochs {args.epochs}")
        return

    print("\n[🚀] BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN LORA...")
    from datasets import Dataset
    raw_dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_func(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_dataset = raw_dataset.map(tokenize_func, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=(device == "cuda"),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    trainer.train()
    print(f"\n[✓] Huấn luyện thành công! Đang lưu LoRA weights vào {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
