"""Tests for GeminiClient (app/llm/client.py).

Mirrors test_client.py's structure for AnthropicClient -- same retry/cost/
streaming behaviors are expected of both, since services/ treats them as
interchangeable LLMClient implementations. Offline tests monkeypatch
client._client.models.generate_content / generate_content_stream (both
stable attributes on the SDK's Client). The one live-gated test skips
automatically without GEMINI_API_KEY.
"""

import logging

import pytest
from google.genai import errors as genai_errors
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm.client import GeminiClient

_HAS_API_KEY = bool(get_settings().gemini_api_key)


class _Greeting(BaseModel):
    language: str
    greeting: str


def _status_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code=code, response_json={"message": "error", "status": "ERROR"})


def _validation_error() -> ValidationError:
    try:
        _Greeting(language="French")  # missing required 'greeting'
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


class _ScriptedCall:
    """Stands in for client._client.models.generate_content: raises/returns
    each scripted move in order, one per call."""

    def __init__(self, moves: list) -> None:
        self._moves = list(moves)
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        move = self._moves.pop(0)
        if isinstance(move, BaseException):
            raise move
        return move


class _FakeUsageMetadata:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.prompt_token_count = input_tokens
        self.candidates_token_count = output_tokens


class _FakeGeminiResponse:
    """.text must be valid JSON -- GeminiClient validates it itself rather
    than trusting response.parsed (see client.py's GeminiClient docstring)."""

    def __init__(self) -> None:
        self.text = _Greeting(language="French", greeting="Bonjour").model_dump_json()
        self.usage_metadata = _FakeUsageMetadata()


class _FakeStreamChunk:
    def __init__(self, text: str, usage_metadata: _FakeUsageMetadata | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata


def test_generate_structured_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = GeminiClient(api_key="test-key")
    fake_call = _ScriptedCall([_status_error(503), _status_error(429), _FakeGeminiResponse()])
    monkeypatch.setattr(client._client.models, "generate_content", fake_call)

    result = client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 3
    assert result.parsed.greeting == "Bonjour"


def test_generate_structured_gives_up_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = GeminiClient(api_key="test-key")
    # settings.llm_max_retries defaults to 3 -- script exactly that many failures
    fake_call = _ScriptedCall([_status_error(503), _status_error(503), _status_error(503)])
    monkeypatch.setattr(client._client.models, "generate_content", fake_call)

    with pytest.raises(genai_errors.APIError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 3  # bounded -- no unbounded retry


def test_generate_structured_does_not_retry_client_errors(monkeypatch) -> None:
    client = GeminiClient(api_key="test-key")
    fake_call = _ScriptedCall([_status_error(400)])
    monkeypatch.setattr(client._client.models, "generate_content", fake_call)

    with pytest.raises(genai_errors.APIError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 1  # a 400 fails identically every time -- retrying is pointless


def test_generate_structured_does_not_retry_validation_errors(monkeypatch) -> None:
    """Same Part 5/Part 6 boundary as AnthropicClient: a well-formed
    response that fails schema validation is never retried here."""
    client = GeminiClient(api_key="test-key")
    fake_response = _FakeGeminiResponse()
    fake_response.text = '{"language": "French"}'  # missing required 'greeting'
    fake_call = _ScriptedCall([fake_response])
    monkeypatch.setattr(client._client.models, "generate_content", fake_call)

    with pytest.raises(ValidationError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 1


def test_generate_structured_logs_cost_telemetry(monkeypatch, caplog) -> None:
    client = GeminiClient(api_key="test-key")
    fake_call = _ScriptedCall([_FakeGeminiResponse()])
    monkeypatch.setattr(client._client.models, "generate_content", fake_call)

    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert any("llm_usage" in record.getMessage() for record in caplog.records)
    assert any("estimated_cost_usd=0.000000" in record.getMessage() for record in caplog.records)


def test_stream_structured_yields_deltas_and_final_result(monkeypatch) -> None:
    full_json = _Greeting(language="French", greeting="Bonjour").model_dump_json()
    half = len(full_json) // 2
    chunks = [
        _FakeStreamChunk(full_json[:half]),
        _FakeStreamChunk(full_json[half:], usage_metadata=_FakeUsageMetadata()),
    ]
    client = GeminiClient(api_key="test-key")
    monkeypatch.setattr(client._client.models, "generate_content_stream", lambda **kwargs: iter(chunks))

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        deltas = list(stream.text_stream)
        result = stream.get_final_result()

    assert "".join(deltas) == full_json
    assert result.parsed.greeting == "Bonjour"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_stream_structured_get_final_result_propagates_validation_error(monkeypatch) -> None:
    chunks = [_FakeStreamChunk('{"language": "French"}')]  # missing 'greeting', valid-looking JSON
    client = GeminiClient(api_key="test-key")
    monkeypatch.setattr(client._client.models, "generate_content_stream", lambda **kwargs: iter(chunks))

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        deltas = list(stream.text_stream)  # exhausting text_stream does NOT raise
        with pytest.raises(ValidationError):
            stream.get_final_result()

    assert deltas == ['{"language": "French"}']


def test_stream_structured_missing_usage_metadata_defaults_to_zero(monkeypatch) -> None:
    """Defends against the reported flakiness of Gemini's streamed
    usage_metadata (not every chunk carries it) -- should degrade to 0,
    not crash."""
    full_json = _Greeting(language="French", greeting="Bonjour").model_dump_json()
    chunks = [_FakeStreamChunk(full_json)]  # no usage_metadata on any chunk
    client = GeminiClient(api_key="test-key")
    monkeypatch.setattr(client._client.models, "generate_content_stream", lambda **kwargs: iter(chunks))

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        list(stream.text_stream)
        result = stream.get_final_result()

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires GEMINI_API_KEY")
def test_generate_structured_returns_validated_model_and_usage() -> None:
    client = GeminiClient()

    result = client.generate_structured(
        system="Respond in the requested structured format only.",
        user="Give me a friendly greeting in French.",
        response_model=_Greeting,
    )

    assert isinstance(result.parsed, _Greeting)
    assert result.parsed.language.lower().startswith("fr")
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.model
