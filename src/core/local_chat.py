"""LLM siêu nhỏ chạy trên CPU local (Qwen2.5-0.5B-Instruct) phục vụ chế độ Offline."""
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from src.core.logger import get_logger

logger = get_logger("core.local_chat")


class LocalChatModel(BaseChatModel):
    """Mô hình LLM local chạy 100% trên CPU (mặc định: Qwen/Qwen2.5-0.5B-Instruct)."""
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Optional[str] = None
    max_new_tokens: int = 384
    max_input_tokens: int = 2048
    threads: int = 6
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _identity: dict = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self.threads)
        model_target = self.model_name_or_path

        # Ưu tiên kiểm tra thư mục đã tải cố định trong data/models
        try:
            from src.core.model_registry import checked_model
            directory, identity = checked_model("draft_llm")
            model_target = str(directory)
            self._identity = identity
        except Exception:
            self._identity = {"repo": self.model_name_or_path, "local": True}

        logger.info(f"[LocalChatModel] Đang nạp mô hình local từ: {model_target}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_target,
            trust_remote_code=False
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_target,
            dtype=torch.float32,
            trust_remote_code=False,
            low_cpu_mem_usage=True
        ).eval()


        # Nạp LoRA adapter nếu có
        candidate_adapter = self.adapter_path or "data/adapters/dental_qwen_lora"
        if Path(candidate_adapter).exists():
            try:
                from peft import PeftModel
                logger.info(f"[LocalChatModel] Phát hiện và nạp LoRA adapter từ: {candidate_adapter}")
                base_model = PeftModel.from_pretrained(base_model, candidate_adapter)
            except Exception as e:
                logger.warning(f"[LocalChatModel] Không nạp được LoRA adapter ({e}), dùng model gốc.")

        self._model = base_model
        logger.info("[LocalChatModel] Mô hình CPU local đã sẵn sàng!")

    @property
    def identity(self):
        return self._identity.copy()

    @property
    def _llm_type(self):
        return "local-qwen2.5-cpu"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import torch
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        chat = []
        for message in messages:
            r = roles.get(getattr(message, "type", "user"), "user")
            chat.append({"role": r, "content": message.content})

        text = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt")

        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_input_tokens:]

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id
            )


        generated = output[0, inputs["input_ids"].shape[1]:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not content:
            content = "Hệ thống đã phân tích các chứng cứ y khoa nhưng chưa thể tạo câu trả lời cụ thể."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

