import pytest

from ai_pipeline.config import get_settings
from ai_pipeline.llm import LLMProvider


def test_llm_provider_returns_real_response():
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
