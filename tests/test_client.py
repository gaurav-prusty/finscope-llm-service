"""Tests for app/llm/client.py.

The retry tests below are offline -- they monkeypatch the underlying
Anthropic SDK call (client._client.messages, a stable @cached_property) to
script a sequence of failures/successes, rather than depending on the real
API happening to misbehave. The one test that needs a real API call is
skipped automatically when ANTHROPIC_API_KEY isn't set, so `pytest -q` stays
green without a key.
"""

import logging

import anthropic
import httpx
import pytest
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm.client import AnthropicClient

_HAS_API_KEY = bool(get_settings().anthropic_api_key)


class _Greeting(BaseModel):
    language: str
    greeting: str


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=_request())


def _status_error(status_code: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError("error", response=httpx.Response(status_code, request=_request()), body=None)


def _validation_error() -> ValidationError:
    try:
        _Greeting(language="French")  # missing required 'greeting'
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


class _ScriptedParse:
    """Stands in for client._client.messages.parse: raises/returns each
    scripted move in order, one per call."""

    def __init__(self, moves: list) -> None:
        self._moves = list(moves)
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        move = self._moves.pop(0)
        if isinstance(move, BaseException):
            raise move
        return move


class _FakeResponse:
    def __init__(self) -> None:
        self.parsed_output = _Greeting(language="French", greeting="Bonjour")
        self.usage = _FakeUsage()
        self.model = "claude-sonnet-5"


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


def test_generate_structured_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = AnthropicClient(api_key="sk-ant-test-key")
    fake_parse = _ScriptedParse([_connection_error(), _status_error(503), _FakeResponse()])
    monkeypatch.setattr(client._client.messages, "parse", fake_parse)

    result = client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_parse.call_count == 3
    assert result.model == "claude-sonnet-5"


def test_generate_structured_gives_up_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    client = AnthropicClient(api_key="sk-ant-test-key")
    # settings.llm_max_retries defaults to 3 -- script exactly that many failures
    fake_parse = _ScriptedParse([_status_error(503), _status_error(503), _status_error(503)])
    monkeypatch.setattr(client._client.messages, "parse", fake_parse)

    with pytest.raises(anthropic.APIStatusError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_parse.call_count == 3  # bounded -- no unbounded retry


def test_generate_structured_does_not_retry_client_errors(monkeypatch) -> None:
    client = AnthropicClient(api_key="sk-ant-test-key")
    fake_parse = _ScriptedParse([_status_error(400)])
    monkeypatch.setattr(client._client.messages, "parse", fake_parse)

    with pytest.raises(anthropic.APIStatusError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_parse.call_count == 1  # a 400 fails identically every time -- retrying is pointless


def test_generate_structured_does_not_retry_validation_errors(monkeypatch) -> None:
    """The critical boundary between Part 5 and Part 6: a well-formed response
    that fails schema validation is never retried here -- that's the repair-
    then-fail loop's job (services/summarize.py), a different failure mode
    from 'the call itself didn't complete'."""
    client = AnthropicClient(api_key="sk-ant-test-key")
    fake_parse = _ScriptedParse([_validation_error()])
    monkeypatch.setattr(client._client.messages, "parse", fake_parse)

    with pytest.raises(ValidationError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert fake_parse.call_count == 1


def test_generate_structured_logs_cost_telemetry(monkeypatch, caplog) -> None:
    """Confirms the telemetry hook (Part 7) actually fires on a real call
    path, not just that log_usage() works in isolation (tests/test_cost.py)."""
    client = AnthropicClient(api_key="sk-ant-test-key")
    fake_parse = _ScriptedParse([_FakeResponse()])
    monkeypatch.setattr(client._client.messages, "parse", fake_parse)

    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert any("llm_usage" in record.getMessage() for record in caplog.records)


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
