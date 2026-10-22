"""LLM nhỏ chạy CPU để thử pipeline; tải model bằng prepare_models trước."""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from src.core.model_registry import checked_model


class LocalChatModel(BaseChatModel):
    max_new_tokens: int = 384
    max_input_tokens: int = 4096
    threads: int = 6
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _identity: dict = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(self.threads)
        directory, self._identity = checked_model("draft_llm")
        self._tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self._model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float32).eval()

    @property
    def identity(self):
        return self._identity.copy()

    @property
    def _llm_type(self):
        return "local-qwen2.5"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import torch
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        chat = [{"role": roles[message.type], "content": message.content} for message in messages]
        text = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError("Prompt vượt giới hạn; giảm số đoạn ngữ cảnh")
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self._tokenizer.eos_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:]
        content = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not content:
            raise RuntimeError("Model trả nội dung rỗng")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
