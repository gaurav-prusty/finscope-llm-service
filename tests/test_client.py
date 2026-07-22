"""Tests for app/llm/client.py.

The one meaningful test here needs a real Anthropic API call -- there's
nothing to validate offline about "does the SDK correctly call the API."
It's skipped automatically when ANTHROPIC_API_KEY isn't set (e.g. in an
environment that hasn't configured secrets yet), so `pytest -q` stays green
without a key. This test intentionally uses its own tiny schema, not
FilingAnalysis -- it's exercising the client mechanism, not the filing
business logic (that's Part 9's job, against the real schema).
"""

import pytest
from pydantic import BaseModel

from app.config import get_settings
from app.llm.client import AnthropicClient

_HAS_API_KEY = bool(get_settings().anthropic_api_key)


class _Greeting(BaseModel):
    language: str
    greeting: str


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_generate_structured_returns_validated_model_and_usage() -> None:
    client = AnthropicClient()

    result = client.generate_structured(
        system="Respond in the requested structured format only.",
        user="Give me a friendly greeting in French.",
        response_model=_Greeting,
    )

    assert isinstance(result.parsed, _Greeting)
    assert result.parsed.language.lower().startswith("fr")
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.model  # the serving model id came back non-empty
