"""
RAG ingestion and retrieval utilities backed by Qdrant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import Settings
from .logger import get_logger

LOGGER = get_logger(__name__)


class VectorStoreManager:
    """Manages ingestion and retrieval against a Qdrant collection."""

    def __init__(self, settings: Settings, embeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        if settings.qdrant_url.startswith("http"):
            self._client_kwargs = {
                "url": settings.qdrant_url,
                "prefer_grpc": False,
            }
            if settings.qdrant_api_key:
                self._client_kwargs["api_key"] = settings.qdrant_api_key
        else:
            self._client_kwargs = {"path": settings.qdrant_url}
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(**self._client_kwargs)
        return self._client

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            LOGGER.info("Connecting to Qdrant collection %s", self.settings.qdrant_collection)
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.settings.qdrant_collection,
                embedding=self.embeddings,
            )
        return self._vector_store

    def _reset_client(self) -> None:
        """Close and drop the cached client so a fresh connection can be created."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            finally:
                self._client = None

    def ingest_documents(self, documents: Iterable[Document]) -> None:
        """Store documents in the collection, creating it first when necessary."""
        docs: List[Document] = list(documents)
        if not docs:
            LOGGER.warning("No documents supplied for ingestion")
            return
        if self._vector_store is None:
            LOGGER.info("Creating collection %s", self.settings.qdrant_collection)
            if "path" in self._client_kwargs:
                self._reset_client()
            self._vector_store = QdrantVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name=self.settings.qdrant_collection,
                **self._client_kwargs,
            )
            self._client = self._vector_store.client
        else:
            self._vector_store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return the top-k most similar documents for the given query."""
        return self.vector_store.similarity_search(query, k=k)


class PDFIngestor:
    """Loads and chunks PDF documents."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load(self, path: Path) -> List[Document]:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {path}")
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return self.splitter.split_documents(docs)


class RAGService:
    """High-level RAG interface combining PDF ingestion and vector search."""

    def __init__(self, settings: Settings, vector_manager: VectorStoreManager, ingestor: Optional[PDFIngestor] = None) -> None:
        self.settings = settings
        self.vector_manager = vector_manager
        self.ingestor = ingestor or PDFIngestor()

    def ensure_ingested(self) -> None:
        """Loads the configured PDF and ingests it if the collection is empty or missing."""
        try:
            count = self.vector_manager.client.count(
                collection_name=self.settings.qdrant_collection,
                exact=False,
            ).count
            if count and count > 0:
                return
        except Exception:
            # Collection missing, will create on ingestion
            pass
        LOGGER.info("Ingesting PDF from %s", self.settings.document_path)
        documents = self.ingestor.load(self.settings.document_path)
        self.vector_manager.ingest_documents(documents)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search after ensuring the knowledge base exists."""
        return self.vector_manager.similarity_search(query, k=k)


__all__ = ["VectorStoreManager", "PDFIngestor", "RAGService"]
