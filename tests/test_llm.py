import pytest

from ai_pipeline.config import get_settings
from ai_pipeline.llm import LLMProvider
from ai_pipeline.testing_utils import describe_test


@describe_test(
    purpose="Exercises LLMProvider to fetch a deterministic ChatOpenAI model and validates a minimal completion round-trip.",
    targets=[LLMProvider, LLMProvider.get_chat_model],
    notes="Skipped automatically when OPENAI credentials are absent to keep CI offline-friendly.",
)
def test_llm_provider_returns_real_response():
    """Call the live OpenAI chat model and require the canonical OK response."""
    settings = get_settings()
    if not settings.has_openai:
        pytest.skip("OPENAI_API_KEY must be set to run real LLM tests")

    provider = LLMProvider(settings)
    model = provider.get_chat_model(temperature=0.0)

    response = model.invoke(
        "Reply with the two uppercase letters OK and nothing else."
    )
    content = response.content.strip().upper()

    assert content == "OK"
