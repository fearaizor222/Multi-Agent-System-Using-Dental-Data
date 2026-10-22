"""Chọn LLM và embedding theo cấu hình."""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("core.llm")


class MockChatModel(BaseChatModel):
    """Mock Chat Model for local testing without active API keys."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_msg = messages[-1].content if messages else ""
        content = (
            f"[MOCK LLM RESPONSE] Đã nhận thông tin: '{last_msg[:50]}...'. "
            "Đây là phản hồi thử nghiệm từ hệ thống Dental Multi-Agent. "
            "Vui lòng cấu hình GOOGLE_API_KEY trong .env để dùng mô hình Google Gemini thực tế."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


class MockEmbeddings(Embeddings):
    """Simple deterministic mock embeddings for testing vector operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10) / 10.0] * 64 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10) / 10.0] * 64


def get_llm(
    temperature: float = 0.2,
    streaming: bool = True,
    override_provider: Optional[str] = None
) -> BaseChatModel:
    """Khởi tạo LLM đã chọn; báo lỗi nếu thiếu model hoặc cấu hình."""
    provider = override_provider or settings.LLM_PROVIDER.lower()

    if provider == "huggingface":
        from src.core.local_chat import LocalChatModel
        return LocalChatModel()

    if provider == "google":
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GOOGLE_API_KEY; cấu hình API hoặc dùng provider huggingface")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL_NAME,
                temperature=temperature,
                google_api_key=api_key,
                streaming=streaming,
            )
        except Exception as e:
            raise RuntimeError("Không khởi tạo được Google LLM") from e

    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=temperature,
            )
        except Exception as e:
            raise RuntimeError("Không khởi tạo được Ollama LLM") from e

    if provider == "mock":
        return MockChatModel()
    raise ValueError("LLM provider chưa được hỗ trợ: " + provider)


def get_embeddings() -> Embeddings:
    """Chọn embedding thật theo cấu hình; mock chỉ được dùng khi chọn rõ."""
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "huggingface":
        from src.rag.local_embeddings import MODEL, LocalE5Embeddings
        if settings.HF_EMBEDDING_MODEL != MODEL:
            raise ValueError(f"Kho hiện dùng {MODEL}; cần đổi đồng thời model và tạo lại index")
        return LocalE5Embeddings()
    if provider == "mock":
        return MockEmbeddings()

    if provider == "google":
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model=settings.GOOGLE_EMBEDDING_MODEL,
                    google_api_key=api_key,
                )
            except Exception as e:
                raise RuntimeError("Không khởi tạo được Google embedding") from e
        else:
            raise ValueError("Chưa có GOOGLE_API_KEY cho Google embedding")

    raise ValueError(f"Chưa hỗ trợ embedding provider: {provider}")
