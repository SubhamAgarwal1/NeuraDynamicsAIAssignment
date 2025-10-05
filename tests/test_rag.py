from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_core.embeddings import Embeddings

from ai_pipeline.config import Settings
from ai_pipeline.rag import RAGService, VectorStoreManager
from ai_pipeline.testing_utils import describe_test


class SimpleEmbeddings(Embeddings):
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(i) for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(0)

    def _vector(self, seed: int) -> List[float]:
        return [float(seed + i) for i in range(self.dimension)]


def build_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        openweather_api_key="dummy",
        openai_api_key="dummy-openai",
        langsmith_api_key=None,
        langsmith_project=None,
        qdrant_url=str(tmp_path / "qdrant.db"),
        qdrant_api_key=None,
        qdrant_collection="unit_test_collection",
        document_path=tmp_path / "reference.pdf",
    )


@describe_test(
    purpose="Validates that VectorStoreManager ingests documents and surfaces the most relevant match for a query.",
    targets=[VectorStoreManager, VectorStoreManager.ingest_documents, VectorStoreManager.similarity_search],
)
def test_vector_store_similarity_search(tmp_path: Path):
    """Ingest two documents into a temporary Qdrant store and confirm similarity search returns the LangGraph snippet."""
    settings = build_test_settings(tmp_path)
    embeddings = SimpleEmbeddings()
    vector_manager = VectorStoreManager(settings=settings, embeddings=embeddings)

    docs = [
        Document(page_content="LangGraph orchestrates LangChain components."),
        Document(page_content="The weather in Berlin is usually mild."),
    ]

    vector_manager.ingest_documents(docs)
    results = vector_manager.similarity_search("How do I build with LangGraph?", k=1)

    assert len(results) == 1
    assert "LangGraph" in results[0].page_content


@describe_test(
    purpose="Ensures RAGService detects an empty collection and triggers PDF ingestion before serving queries.",
    targets=[RAGService, RAGService.ensure_ingested, VectorStoreManager.ingest_documents],
    notes="Monkeypatches the Qdrant client to simulate a missing collection so ingestion is forcefully executed.",
)
def test_rag_service_ensure_ingested_triggers_loader(monkeypatch, tmp_path: Path):
    """Fake a missing Qdrant collection and check that ensure_ingested loads and adds documents."""
    settings = build_test_settings(tmp_path)
    embeddings = SimpleEmbeddings()
    vector_manager = VectorStoreManager(settings=settings, embeddings=embeddings)

    loaded_docs = [Document(page_content="Sample reference content.")]

    class DummyIngestor:
        def load(self, _path: Path):
            return loaded_docs

    ingestor = DummyIngestor()
    rag_service = RAGService(settings=settings, vector_manager=vector_manager, ingestor=ingestor)

    # Force collection to appear empty by raising from count
    def fake_count(*_args, **_kwargs):
        raise RuntimeError("collection missing")

    monkeypatch.setattr(vector_manager.client, "count", fake_count)

    rag_service.ensure_ingested()

    results = vector_manager.similarity_search("reference", k=1)
    assert results
