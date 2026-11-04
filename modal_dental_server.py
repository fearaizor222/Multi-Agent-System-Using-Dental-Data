"""Modal Cloud GPU Server cho Hệ sinh thái 5 Specialized Small Language Models (SLMs) Nha khoa.

Chạy trên serverless GPU NVIDIA T4 (16GB VRAM) của Modal:
- Agent 1 (Supervisor & Triage): Qwen/Qwen2.5-0.5B-Instruct
- Agent 2 (Occlusion & TMJ): HuggingFaceTB/SmolLM2-360M-Instruct
- Agent 3 (BYT Clinical Protocols): google/flan-t5-small
- Agent 5 (Clinical Synthesis & Drafting): Qwen/Qwen2.5-0.5B-Instruct

Tối ưu hóa:
- Scale down to 0 sau 120s không hoạt động (tiết kiệm tối đa credit $30 free/tháng).
- Pre-download và cache model trong image container để khởi động siêu nhanh.
- Phục vụ đồng thời qua HTTP POST endpoint bảo mật (HTTPS) và CLI Direct RPC.
"""

import time
from typing import Any, Dict, List, Optional
import modal
from pydantic import BaseModel, Field

# Khởi tạo Modal App
app = modal.App("dental-multi-agent")

# Định nghĩa hàm cache model vào Image layer trong quá trình build container
def download_model_weights():
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    print("⏬ [Cache Image] Đang tải mô hình Qwen2.5-0.5B-Instruct...")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    print("⏬ [Cache Image] Đang tải mô hình SmolLM2-360M-Instruct...")
    AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

    print("⏬ [Cache Image] Đang tải mô hình flan-t5-small...")
    AutoTokenizer.from_pretrained("google/flan-t5-small")
    AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    print("✅ [Cache Image] Toàn bộ weights mô hình đã được nướng sẵn vào Image!")


# Xây dựng Docker Container Image tối ưu cho PyTorch & CUDA
dental_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "fastapi[standard]>=0.110.0",
        "pydantic>=2.0.0",
    )
    .run_function(download_model_weights)
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="system, user, hoặc assistant")
    content: str = Field(..., description="Nội dung tin nhắn")


class GenerateRequest(BaseModel):
    agent_id: str = Field(
        default="agent1_supervisor",
        description="ID Agent: agent1_supervisor, agent2_internal_rag, agent3_public_rag, agent5_drafting"
    )
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Danh sách tin nhắn theo định dạng ChatML"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Chuỗi prompt trực tiếp nếu không gửi dạng messages"
    )
    max_new_tokens: int = Field(
        default=384,
        description="Số lượng token sinh ra tối đa"
    )
    temperature: float = Field(
        default=0.2,
        description="Độ ngẫu nhiên (0.0: suy luận chính xác, >0.0: sinh phong phú hơn)"
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Hệ số phạt chống lặp từ"
    )


@app.cls(
    gpu="T4",                     # NVIDIA T4 (16GB VRAM, ~0.59$/giờ)
    image=dental_image,
    scaledown_window=120,         # Tự động ngủ đông sau 2 phút idle (0$ khi không dùng)
    timeout=300
)
class DentalAgentServer:
    @modal.enter()
    def setup_gpu_models(self):
        """Khởi tạo và nạp toàn bộ 3 nhóm mô hình vào GPU VRAM (tổng VRAM ~2.5 GB / 16 GB)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"🚀 [Modal GPU] Đang nạp các chuyên gia SLM lên {self.device} ({self.dtype})...")

        # 1. Agent 1 (Supervisor & Triage Router) & Agent 5 (Drafting Synthesizer)
        print("  🔹 Nạp Qwen2.5-0.5B-Instruct...")
        self.qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()

        # 2. Agent 2 (Bác sĩ Chuyên khoa Cắn khớp & TMJ)
        print("  🔹 Nạp SmolLM2-360M-Instruct...")
        self.smollm_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        self.smollm_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()

        # 3. Agent 3 (Phác đồ Kỹ thuật Điều trị RHM Bộ Y Tế)
        print("  🔹 Nạp flan-t5-small...")
        self.flant5_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.flant5_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small",
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()

        print("✅ [Modal GPU] Sẵn sàng phục vụ toàn bộ 5 Agent Nha khoa trên Cloud GPU!")

    def _execute_inference(self, agent_id: str, chat: List[Dict[str, str]], max_new_tokens: int, temperature: float, repetition_penalty: float) -> Dict[str, Any]:
        """Hàm logic nội bộ thực thi sinh văn bản trên GPU."""
        import torch
        start_time = time.perf_counter()

        # Định tuyến theo từng Agent
        if agent_id == "agent2_internal_rag":
            # SmolLM2-360M
            tok = self.smollm_tok
            mod = self.smollm_model
            model_name = "SmolLM2-360M-Instruct (TMJ Specialist)"
            is_seq2seq = False
        elif agent_id == "agent3_public_rag":
            # Flan-T5-Small
            tok = self.flant5_tok
            mod = self.flant5_model
            model_name = "Flan-T5-Small (BYT Protocol)"
            is_seq2seq = True
        else:
            # agent1_supervisor, agent5_drafting hoặc mặc định: Qwen2.5-0.5B
            tok = self.qwen_tok
            mod = self.qwen_model
            model_name = "Qwen2.5-0.5B-Instruct (Supervisor / Synthesis)"
            is_seq2seq = False

        if is_seq2seq:
            # Xử lý Flan-T5 (Encoder-Decoder)
            prompt_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nANSWER: "
            inputs = tok(prompt_str, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = mod.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 256)
                )
            output_tokens_count = out.shape[-1]
            response_text = tok.decode(out[0], skip_special_tokens=True).strip()
        else:
            # Xử lý Causal LM (Qwen2.5, SmolLM2)
            if hasattr(tok, "apply_chat_template") and tok.chat_template:
                try:
                    formatted_prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                except Exception:
                    formatted_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nASSISTANT: "
            else:
                formatted_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nASSISTANT: "

            inputs = tok(formatted_prompt, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[1]

            do_sample = (temperature > 0.05)
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": repetition_penalty,
                    "pad_token_id": tok.eos_token_id or tok.pad_token_id,
                }
                if do_sample:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9
                else:
                    gen_kwargs["do_sample"] = False

                out = mod.generate(**inputs, **gen_kwargs)

            new_tokens = out[0][input_length:]
            output_tokens_count = len(new_tokens)
            response_text = tok.decode(new_tokens, skip_special_tokens=True).strip()

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        tokens_per_sec = round(output_tokens_count / (latency_ms / 1000.0), 1) if latency_ms > 0 else 0

        return {
            "status": "success",
            "agent_id": agent_id,
            "model_name": model_name,
            "response": response_text,
            "tokens_generated": output_tokens_count,
            "latency_ms": latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "device": self.device,
        }

    @modal.fastapi_endpoint(method="POST")
    def generate(self, req: GenerateRequest) -> Dict[str, Any]:
        """Endpoint HTTP POST cho Laptop local và Frontend kết nối qua HTTPS."""
        chat = []
        if req.messages:
            for m in req.messages:
                chat.append({"role": m.role, "content": m.content})
        elif req.prompt:
            chat.append({"role": "user", "content": req.prompt})
        else:
            chat.append({"role": "user", "content": "Xin chào bác sĩ."})

        return self._execute_inference(
            agent_id=req.agent_id or "agent1_supervisor",
            chat=chat,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty
        )

    @modal.method()
    def infer(self, agent_id: str, prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
        """Phương thức Direct RPC cho phép gọi nhanh qua CLI: modal run modal_dental_server.py."""
        chat = [{"role": "user", "content": prompt}]
        return self._execute_inference(
            agent_id=agent_id,
            chat=chat,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            repetition_penalty=1.1
        )

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> Dict[str, Any]:
        """Health check endpoint kiểm tra tình trạng GPU và các mô hình đã nạp."""
        import torch

        return {
            "status": "online",
            "service": "Dental Multi-Agent Cloud GPU Inference",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if torch.cuda.is_available() else 0,
            "models": [
                {"agent": "agent1_supervisor", "model": "Qwen/Qwen2.5-0.5B-Instruct"},
                {"agent": "agent2_internal_rag", "model": "HuggingFaceTB/SmolLM2-360M-Instruct"},
                {"agent": "agent3_public_rag", "model": "google/flan-t5-small"},
                {"agent": "agent5_drafting", "model": "Qwen/Qwen2.5-0.5B-Instruct"},
            ]
        }


@app.local_entrypoint()
def main():
    """Hàm chạy thử nghiệm nhanh từ CLI: modal run modal_dental_server.py."""
    print("=" * 65)
    print("🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM INFERENCE TRÊN GPU NVIDIA T4 TỪ MODAL...")
    print("=" * 65)

    server = DentalAgentServer()

    prompt = (
        "Bệnh nhân nam 28 tuổi, đau răng dữ dội góc hàm dưới phải, sưng to mặt lan xuống cổ, "
        "sốt 39 độ, há miệng hạn chế dưới 1.5cm. Hãy đánh giá cấp độ triage và cờ đỏ khẩn cấp."
    )
    print(f"\n📝 Câu hỏi lâm sàng gửi tới Agent 1 (Supervisor & Triage):\n   '{prompt}'\n")

    res = server.infer.remote(agent_id="agent1_supervisor", prompt=prompt, max_new_tokens=256)

    print("🎉 KẾT QUẢ SUY LUẬN TRỰC TIẾP:")
    print(f"🤖 Model: {res.get('model_name')} (Chạy trên: {res.get('device')})")
    print(f"⏱️ Thời gian suy luận: {res.get('latency_ms')} ms")
    print(f"⚡ Tốc độ: {res.get('tokens_per_sec')} tokens/s ({res.get('tokens_generated')} tokens)")
    print(f"\n💬 Phản hồi của Agent 1:\n{res.get('response')}")
    print("\n" + "=" * 65)
    print("💡 Để deploy thành endpoint HTTPS vĩnh viễn dùng cho Web UI:")
    print("   modal deploy modal_dental_server.py")
    print("=" * 65)
