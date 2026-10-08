"""Tests for GroqClient (app/llm/client.py).

Mirrors test_client.py's/test_gemini_client.py's structure -- same
retry/cost behaviors are expected of all three LLMClient implementations.
Offline tests monkeypatch client._client.chat.completions.create (a stable
attribute on the SDK's Client, Stainless-generated same as Anthropic's own
SDK). GroqClient does NOT support streaming at all (Groq's API can't combine
response_format with streaming) -- see the supports_streaming/
stream_structured tests below instead of a streaming-success test. The one
live-gated test skips automatically without GROQ_API_KEY.
"""

import copy
import logging

import groq
import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm.client import GroqClient, _to_groq_strict_schema
from app.llm.schemas import FilingAnalysis

_HAS_API_KEY = bool(get_settings().groq_api_key)


class _Greeting(BaseModel):
    language: str
    greeting: str


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _connection_error() -> groq.APIConnectionError:
    return groq.APIConnectionError(request=_request())


def _status_error(status_code: int) -> groq.APIStatusError:
    return groq.APIStatusError("error", response=httpx.Response(status_code, request=_request()), body=None)


def _validation_error() -> ValidationError:
    try:
        _Greeting(language="French")  # missing required 'greeting'
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


class _ScriptedCall:
    """Stands in for client._client.chat.completions.create: raises/returns
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


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeGroqResponse:
    def __init__(self, content: str | None = None) -> None:
        content = content or _Greeting(language="French", greeting="Bonjour").model_dump_json()
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def test_generate_structured_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = GroqClient(api_key="test-key")
    fake_call = _ScriptedCall([_status_error(503), _connection_error(), _FakeGroqResponse()])
    monkeypatch.setattr(client._client.chat.completions, "create", fake_call)

    result = client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 3
    assert result.parsed.greeting == "Bonjour"


def test_generate_structured_gives_up_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = GroqClient(api_key="test-key")
    # settings.llm_max_retries defaults to 3 -- script exactly that many failures
    fake_call = _ScriptedCall([_status_error(503), _status_error(503), _status_error(503)])
    monkeypatch.setattr(client._client.chat.completions, "create", fake_call)

    with pytest.raises(groq.APIStatusError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 3  # bounded -- no unbounded retry


def test_generate_structured_does_not_retry_client_errors(monkeypatch) -> None:
    client = GroqClient(api_key="test-key")
    fake_call = _ScriptedCall([_status_error(400)])
    monkeypatch.setattr(client._client.chat.completions, "create", fake_call)

    with pytest.raises(groq.APIStatusError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 1  # a 400 fails identically every time -- retrying is pointless


def test_generate_structured_does_not_retry_validation_errors(monkeypatch) -> None:
    """Same Part 5/Part 6 boundary as the other clients: a well-formed
    response that fails schema validation is never retried here -- even
    though strict mode is supposed to guarantee this can't happen, we don't
    trust that blindly (see GroqClient's docstring)."""
    client = GroqClient(api_key="test-key")
    fake_call = _ScriptedCall([_FakeGroqResponse(content='{"language": "French"}')])  # missing 'greeting'
    monkeypatch.setattr(client._client.chat.completions, "create", fake_call)

    with pytest.raises(ValidationError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_call.call_count == 1


def test_generate_structured_logs_cost_telemetry(monkeypatch, caplog) -> None:
    client = GroqClient(api_key="test-key")
    fake_call = _ScriptedCall([_FakeGroqResponse()])
    monkeypatch.setattr(client._client.chat.completions, "create", fake_call)

    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert any("llm_usage" in record.getMessage() for record in caplog.records)
    assert any("estimated_cost_usd=0.000000" in record.getMessage() for record in caplog.records)


def _object_nodes(node: object) -> list[dict]:
    """Every object-with-properties node in a JSON schema, at any depth
    (including pydantic's $defs, where nested models live)."""
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_object_nodes(item))
    return found


def test_strict_schema_satisfies_groq_rules_for_filing_analysis() -> None:
    """The offline guard that would have caught both live 400s: Groq requires
    additionalProperties: false on every object AND every property listed in
    `required` -- FilingAnalysis.caveats (has a pydantic default) is the real
    case that broke, and its nested models live under $defs."""
    raw = FilingAnalysis.model_json_schema()
    assert "caveats" not in raw["required"]  # the pydantic default that Groq rejects

    strict = _to_groq_strict_schema(raw)

    nodes = _object_nodes(strict)
    assert len(nodes) == 3  # FilingAnalysis + FinancialHighlight + RiskFactor
    for node in nodes:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])
    assert "caveats" in strict["required"]


def test_strict_schema_does_not_mutate_its_input() -> None:
    raw = _Greeting.model_json_schema()
    before = copy.deepcopy(raw)

    _to_groq_strict_schema(raw)

    assert raw == before


def test_generate_structured_sends_strict_json_schema_response_format(monkeypatch) -> None:
    """Guards the exact response_format shape Groq's API requires (confirmed
    via console.groq.com/docs/structured-outputs) -- a regression here would
    otherwise only surface as a live 400, not an offline test failure. (This
    test originally asserted the RAW model_json_schema() was sent -- which
    locked in exactly the shape Groq rejected live.)"""
    client = GroqClient(api_key="test-key")
    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeGroqResponse()

    monkeypatch.setattr(client._client.chat.completions, "create", _fake_create)

    client.generate_structured(system="s", user="u", response_model=_Greeting)

    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    sent_schema = response_format["json_schema"]["schema"]
    assert sent_schema == _to_groq_strict_schema(_Greeting.model_json_schema())
    assert sent_schema["additionalProperties"] is False


def test_supports_streaming_is_false() -> None:
    assert GroqClient.supports_streaming is False


def test_stream_structured_raises_not_implemented() -> None:
    client = GroqClient(api_key="test-key")

    with pytest.raises(NotImplementedError):
        client.stream_structured(system="s", user="u", response_model=_Greeting)


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires GROQ_API_KEY")
def test_generate_structured_returns_validated_model_and_usage() -> None:
    client = GroqClient()

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
