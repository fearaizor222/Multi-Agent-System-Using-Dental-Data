"""LLM and Embeddings factory module configured for Google Gemini."""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
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
        return {"generations": [{"text": content, "message": AIMessage(content=content)}]}

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
    """Factory to instantiate the configured Chat LLM (Google Gemini default)."""
    provider = override_provider or settings.LLM_PROVIDER.lower()

    if provider == "google":
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY chưa được cấu hình. Tự động chuyển sang Mock LLM.")
            return MockChatModel()
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL_NAME,
                temperature=temperature,
                google_api_key=api_key,
                streaming=streaming,
            )
        except Exception as e:
            logger.error(f"Lỗi khởi tạo ChatGoogleGenerativeAI: {e}")
            return MockChatModel()

    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Lỗi khởi tạo ChatOllama: {e}")
            return MockChatModel()

    logger.info("Sử dụng Mock Chat Model mặc định.")
    return MockChatModel()


def get_embeddings() -> Embeddings:
    """Factory to instantiate the configured Embedding model (Google Gemini default)."""
    provider = settings.EMBEDDING_PROVIDER.lower()

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
                logger.error(f"Lỗi khởi tạo GoogleGenerativeAIEmbeddings: {e}")
        else:
            logger.warning("GOOGLE_API_KEY chưa được đặt cho Embeddings, sử dụng MockEmbeddings.")

    return MockEmbeddings()
