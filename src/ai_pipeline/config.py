
"""
Runtime configuration helpers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openweather_api_key: str
    openai_api_key: Optional[str]
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    qdrant_url: str
    qdrant_api_key: Optional[str]
    qdrant_collection: str
    document_path: Path

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_langsmith(self) -> bool:
        return bool(self.langsmith_api_key)

    def configure_langsmith(self) -> None:
        """Propagate LangSmith environment variables when credentials are present."""
        if not self.has_langsmith:
            return
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ.setdefault("LANGCHAIN_API_KEY", self.langsmith_api_key or "")
        os.environ.setdefault("LANGSMITH_API_KEY", self.langsmith_api_key or "")
        if self.langsmith_project:
            os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)
            os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith_project)

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir_env = os.getenv("BASE_DIR")
        base_dir = Path(base_dir_env) if base_dir_env else Path.cwd()
        document_path_env = os.getenv("DOCUMENT_PATH")
        document_path = Path(document_path_env) if document_path_env else base_dir / "data" / "docs" / "reference.pdf"
        return cls(
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT"),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "langgraph_demo"),
            document_path=document_path,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.configure_langsmith()
    return settings
