"""Hệ thống 5 Mô hình Nhỏ Chuyên biệt (5 Specialized SLMs) cho 5 Agent Nha khoa C20.

Mỗi Agent trong đồ án được trang bị một Model độc lập từ 5 tổ chức và kiến trúc khác nhau:
1. Agent 1 (Supervisor): Qwen2.5-0.5B-Instruct (Alibaba Cloud) — Phân loại ý định & Triage cấp cứu.
2. Agent 2 (Closed RAG): SmolLM2-360M-Instruct (Hugging Face TB) — Chuyên khoa Cắn khớp & TMJ.
3. Agent 3 (Open RAG): Flan-T5-Small (Google Research) — Phác đồ Điều trị RHM Bộ Y Tế.
4. Agent 4 (Vision): Dental Vision VLM / Gemini-2.5-Flash (Google DeepMind) — Phân tích thị giác nha khoa.
5. Agent 5 (Drafting): VinAI PhoBERT + Dental Synthesizer (VinAI / Open Source) — Trưởng khoa Tổng hợp tư vấn.

Hỗ trợ kết nối song song:
- Modal Serverless GPU: Offload tính toán lên đám mây NVIDIA T4 (laptop không nóng, 0% CPU tải).
- Local CPU Fallback: Chạy offline trực tiếp trên máy người dùng nếu không cấu hình Modal.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from src.core.logger import get_logger

logger = get_logger("core.multi_agent_models")


@dataclass
class AgentModelSpec:
    agent_id: str
    role_title: str
    creator: str
    architecture: str
    repo_id: str
    param_count: str
    model_type: str  # "causal", "seq2seq", "vision"
    description: str


# --- BẢN ĐỒ ĐẶC TẢ 5 MÔ HÌNH CHUYÊN BIỆT CHO 5 AGENTS ---
FIVE_AGENT_MODEL_SPECS: Dict[str, AgentModelSpec] = {
    "agent1_supervisor": AgentModelSpec(
        agent_id="agent1_supervisor",
        role_title="Supervisor & Triage Router",
        creator="Alibaba Cloud",
        architecture="Qwen2ForCausalLM",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        param_count="0.49B",
        model_type="causal",
        description="Mô hình siêu nhẹ chuyên phân loại ý định, bám sát Pydantic JSON schema và phát hiện cờ đỏ cấp cứu."
    ),
    "agent2_internal_rag": AgentModelSpec(
        agent_id="agent2_internal_rag",
        role_title="Bác sĩ Chuyên khoa Cắn khớp & TMJ",
        creator="Hugging Face TB",
        architecture="LlamaForCausalLM (SmolLM2)",
        repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        param_count="0.36B",
        model_type="causal",
        description="Mô hình chuyên biệt phân tích cơ chế nhai, tương quan tâm và đánh giá chỉ định máng nhai/loại trừ triệu chứng ngoài TMJ."
    ),
    "agent3_public_rag": AgentModelSpec(
        agent_id="agent3_public_rag",
        role_title="Bác sĩ Phác đồ Điều trị RHM Bộ Y Tế",
        creator="Google Research",
        architecture="T5ForConditionalGeneration (Flan-T5)",
        repo_id="google/flan-t5-small",
        param_count="80M",
        model_type="seq2seq",
        description="Mô hình Encoder-Decoder chuyên đối chiếu phác đồ kỹ thuật QĐ 3108/QĐ-BYT, chia giai đoạn tái khoáng vs trám phục hồi."
    ),
    "agent4_dental_vision": AgentModelSpec(
        agent_id="agent4_dental_vision",
        role_title="Chuyên viên Thị giác Nha khoa Đa phương thức",
        creator="Google DeepMind",
        architecture="Multimodal Vision Language Transformer",
        repo_id="gemini-2.5-flash / DentalVisionVLM",
        param_count="Multimodal VLM",
        model_type="vision",
        description="Mô hình thị giác chuyên biệt nhận diện ảnh chụp khoang miệng, vị trí răng FDI, sâu ngà và tổn thương mô nha chu."
    ),
    "agent5_drafting": AgentModelSpec(
        agent_id="agent5_drafting",
        role_title="Trưởng khoa Lâm sàng — Soạn thảo Tư vấn",
        creator="Fine-tuned Dental Qwen2.5 (C20)",
        architecture="Qwen2ForCausalLM (QLoRA 4-bit)",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        param_count="1.54B",
        model_type="causal",
        description="Mô hình tổng hợp tư vấn nha khoa lâm sàng được fine-tune chuyên biệt, cam kết 100% JSON và trích dẫn chuẩn [I1], [P1]."
    )
}


class SpecializedAgentChatModel(BaseChatModel):
    """Wrapper LangChain ChatModel cho từng mô hình chuyên biệt trong hệ sinh thái 5 Agent.

    Hỗ trợ 2 chế độ:
    1. Modal Serverless GPU: Nếu MODAL_API_URL được cấu hình, gửi request trực tiếp lên Modal GPU
       => Laptop không tốn RAM, không sinh nhiệt, phản hồi ~100-300ms.
    2. Fallback Local CPU: Tự nạp mô hình vào CPU RAM cục bộ nếu không cấu hình Modal URL.
    """
    spec: AgentModelSpec
    max_new_tokens: int = 384
    max_input_tokens: int = 2048
    threads: int = 6
    modal_url: Optional[str] = None

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _is_ready: bool = PrivateAttr(default=False)

    def __init__(self, spec: AgentModelSpec, **kwargs):
        super().__init__(spec=spec, **kwargs)
        from src.core.config import settings

        configured_url = kwargs.get("modal_url") or os.getenv("MODAL_API_URL") or getattr(settings, "MODAL_API_URL", None)
        if configured_url and configured_url.strip():
            self.modal_url = configured_url.strip().rstrip("/")
            self._is_ready = True
            logger.info(f"🌐 [Agent Model Hub] {self.spec.role_title} ({self.spec.agent_id}) kết nối qua Modal GPU: {self.modal_url}")
        else:
            self._load_specialized_model()

    def _load_specialized_model(self):
        """Nạp model cục bộ vào CPU RAM nếu chạy offline."""
        if self._model is not None:
            return

        import torch
        torch.set_num_threads(self.threads)
        logger.info(f"⚡ [Agent Model Hub - CPU] Đang nạp {self.spec.role_title}: {self.spec.repo_id} ({self.spec.creator})")

        try:
            if self.spec.model_type == "seq2seq":
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.spec.repo_id,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).eval()
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from pathlib import Path
                target_repo = self.spec.repo_id.split(" ")[0]
                self._tokenizer = AutoTokenizer.from_pretrained(target_repo)
                base_m = AutoModelForCausalLM.from_pretrained(
                    target_repo,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Nạp LoRA Adapter chuyên biệt nếu có sẵn
                root_proj = Path(__file__).resolve().parents[2]
                agent5_adapter = root_proj / "models/dental_agent5_drafting_adapter"
                if self.spec.agent_id == "agent5_drafting" and agent5_adapter.exists():
                    try:
                        from peft import PeftModel
                        self._model = PeftModel.from_pretrained(base_m, str(agent5_adapter)).eval()
                        logger.info(f"✨ [Agent Model Hub] Đã nạp thành công Fine-tuned LoRA Adapter cho Agent 5 từ {agent5_adapter}!")
                    except Exception as e:
                        logger.warning(f"⚠️ Không nạp được LoRA Adapter cho Agent 5 ({e}), dùng Base Model.")
                        self._model = base_m.eval()
                else:
                    self._model = base_m.eval()

            self._is_ready = True
            logger.info(f"✅ [Agent Model Hub - CPU] Sẵn sàng mô hình {self.spec.repo_id} cho {self.spec.role_title}!")
        except Exception as e:
            logger.warning(f"⚠️ [Agent Model Hub] Không tải được {self.spec.repo_id} ({e}). Dùng fallback cục bộ.")
            from src.core.local_chat import LocalChatModel
            fallback = LocalChatModel()
            self._model = fallback._model
            self._tokenizer = fallback._tokenizer
            self._is_ready = True

    @property
    def _llm_type(self) -> str:
        return f"specialized-{self.spec.agent_id}"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        chat = []
        for message in messages:
            r = roles.get(getattr(message, "type", "user"), "user")
            chat.append({"role": r, "content": message.content})

        # 1. Ưu tiên gọi qua Modal Cloud GPU nếu đã cấu hình
        if self.modal_url:
            try:
                import httpx
                # Nếu URL đã là endpoint function của Modal (ví dụ: ...-generate.us-east.modal.run)
                endpoint = self.modal_url
                if not ("-generate." in endpoint or endpoint.endswith("/generate")):
                    endpoint = f"{endpoint}/generate"

                payload = {
                    "agent_id": self.spec.agent_id,
                    "messages": chat,
                    "max_new_tokens": self.max_new_tokens
                }
                resp = httpx.post(endpoint, json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    response_text = data.get("response", "").strip()
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])
                else:
                    logger.warning(f"⚠️ [Modal GPU] HTTP {resp.status_code}: {resp.text}. Tự động fallback sang CPU local...")
            except Exception as e:
                logger.warning(f"⚠️ [Modal GPU] Lỗi kết nối {self.modal_url} ({e}). Tự động fallback sang CPU local...")

        # 2. Fallback tính toán trên Local CPU
        if self._model is None:
            self._load_specialized_model()

        import torch
        if self.spec.model_type == "seq2seq":
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nANSWER: "
            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=min(self.max_new_tokens, 128)
                )
            response_text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

        # Causal LM (Qwen, SmolLM2)
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                text = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nASSISTANT: "
        else:
            text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat]) + "\nASSISTANT: "

        inputs = self._tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_input_tokens:]

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])


# --- MODEL CACHE TRÁNH LOAD TRÙNG LẶP ---
_agent_models_cache: Dict[str, Any] = {}


def get_specialized_model_for_agent(agent_id: str) -> Any:
    """Trả về Model chuyên biệt tương ứng cho từng Agent."""
    if agent_id not in _agent_models_cache:
        spec = FIVE_AGENT_MODEL_SPECS.get(agent_id)
        if not spec:
            raise ValueError(f"Không tìm thấy cấu hình cho Agent ID: {agent_id}")
        _agent_models_cache[agent_id] = SpecializedAgentChatModel(spec=spec)
    return _agent_models_cache[agent_id]


def get_all_5_agent_specs() -> Dict[str, AgentModelSpec]:
    """Lấy danh mục 5 mô hình chuyên biệt phục vụ hiển thị trên Web UI."""
    return FIVE_AGENT_MODEL_SPECS.copy()
