"""
LLM and embedding providers used across the pipeline.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings

DEFAULT_CHAT_MODEL = "gpt-5-nano"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class LLMProvider:
    """Factory for creating chat and embedding models configured from settings."""

    def __init__(self, settings: Settings, chat_model: str = DEFAULT_CHAT_MODEL, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
        """Validate OpenAI credentials and store default model identifiers."""
        if not settings.has_openai:
            raise ValueError("OPENAI_API_KEY is required for LLM operations")
        self.settings = settings
        self.chat_model_name = chat_model
        self.embed_model_name = embed_model

    def get_chat_model(self, temperature: float = 0.0) -> BaseChatModel:
        """Create a ChatOpenAI instance with the configured defaults."""
        return ChatOpenAI(
            model=self.chat_model_name,
            temperature=temperature,
            api_key=self.settings.openai_api_key,
        )

    def get_embeddings(self) -> Embeddings:
        """Return an embedding client tied to the configured OpenAI model."""
        return OpenAIEmbeddings(
            model=self.embed_model_name,
            api_key=self.settings.openai_api_key,
        )


__all__ = ["LLMProvider", "DEFAULT_CHAT_MODEL", "DEFAULT_EMBED_MODEL"]
