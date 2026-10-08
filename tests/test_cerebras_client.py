"""Tests for CerebrasClient (app/llm/client.py).

Mirrors test_groq_client.py's coverage -- same retry/cost/validation behaviors
are expected of every LLMClient implementation. CerebrasClient talks plain
httpx (no vendor SDK), so instead of monkeypatching an SDK method these tests
swap the client's httpx.Client for one backed by httpx.MockTransport: the real
request-building and response-parsing code runs, only the network is faked.
The one live-gated test skips automatically without CEREBRAS_API_KEY.
"""

import json
import logging

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import get_settings
from app.llm.client import CerebrasClient, LLMProviderError, _to_groq_strict_schema

_HAS_API_KEY = bool(get_settings().cerebras_api_key)


class _Greeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    greeting: str


def _ok_response(content: str | None = None) -> httpx.Response:
    content = content or _Greeting(language="French", greeting="Bonjour").model_dump_json()
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


class _Script:
    """Plays back one scripted move per request: an httpx.Response is returned,
    an exception is raised. Records every request it sees."""

    def __init__(self, moves: list) -> None:
        self._moves = list(moves)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        move = self._moves.pop(0)
        if isinstance(move, BaseException):
            raise move
        return move


def _client(script: _Script) -> CerebrasClient:
    client = CerebrasClient(api_key="test-key")
    client._http = httpx.Client(
        base_url="https://api.cerebras.ai/v1",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(script),
    )
    return client


def _connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("boom", request=httpx.Request("POST", "https://api.cerebras.ai/v1/x"))


def test_generate_structured_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    script = _Script([httpx.Response(503, text="overloaded"), _connect_error(), _ok_response()])
    client = _client(script)

    result = client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert len(script.requests) == 3
    assert result.parsed.greeting == "Bonjour"


def test_generate_structured_retries_429(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    script = _Script([httpx.Response(429, text="rate limited"), _ok_response()])
    client = _client(script)

    client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert len(script.requests) == 2


def test_generate_structured_gives_up_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    # settings.llm_max_retries defaults to 3 -- script exactly that many failures
    script = _Script([httpx.Response(503, text="x")] * 3)
    client = _client(script)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert exc_info.value.status_code == 503
    assert len(script.requests) == 3  # bounded -- no unbounded retry


def test_transport_failure_becomes_llm_provider_error_with_no_status(monkeypatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda seconds: None)
    script = _Script([_connect_error()] * 3)
    client = _client(script)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert exc_info.value.status_code is None


def test_generate_structured_does_not_retry_client_errors() -> None:
    script = _Script([httpx.Response(400, text="bad request")])
    client = _client(script)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert exc_info.value.status_code == 400
    assert len(script.requests) == 1  # a 400 fails identically every time


def test_generate_structured_does_not_retry_validation_errors() -> None:
    """Same Part 5/Part 6 boundary as every other client: a well-formed
    response that fails schema validation is never retried here."""
    script = _Script([_ok_response(content='{"language": "French"}')])  # missing 'greeting'
    client = _client(script)

    with pytest.raises(ValidationError):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert len(script.requests) == 1


def test_generate_structured_logs_cost_telemetry(caplog) -> None:
    client = _client(_Script([_ok_response()]))

    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        client.generate_structured(system="s", user="u", response_model=_Greeting)

    assert any("llm_usage" in record.getMessage() for record in caplog.records)
    assert any("estimated_cost_usd=0.000000" in record.getMessage() for record in caplog.records)


def test_generate_structured_sends_expected_request_shape() -> None:
    """Guards the request Cerebras's API requires (auth header, model, strict
    json_schema in the Groq-adapted shape) -- a regression here would otherwise
    only surface as a live 400/401."""
    script = _Script([_ok_response()])
    client = _client(script)

    client.generate_structured(system="sys", user="usr", response_model=_Greeting)

    request = script.requests[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == _to_groq_strict_schema(
        _Greeting.model_json_schema()
    )


def _sse_response(deltas: list[str], usage: dict | None = None, done: bool = True) -> httpx.Response:
    """A fake Cerebras SSE body: one chunk per delta, then a usage-only chunk
    (empty delta), then the optional [DONE] sentinel."""
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": d}}]}) for d in deltas
    ]
    lines.append(
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
        if usage
        else "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    )
    if done:
        lines.append("data: [DONE]")
    return httpx.Response(
        200, content=("\n\n".join(lines) + "\n\n").encode(), headers={"content-type": "text/event-stream"}
    )


def test_supports_streaming_is_true() -> None:
    assert CerebrasClient.supports_streaming is True


def test_stream_structured_yields_deltas_and_final_result(caplog) -> None:
    full = _Greeting(language="French", greeting="Bonjour").model_dump_json()
    script = _Script([_sse_response([full[:10], full[10:]], usage={"prompt_tokens": 12, "completion_tokens": 7})])
    client = _client(script)

    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
            deltas = list(stream.text_stream)
            result = stream.get_final_result()

    assert "".join(deltas) == full
    assert len(deltas) == 2
    assert result.parsed.greeting == "Bonjour"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 7)
    assert any("llm_usage" in r.getMessage() for r in caplog.records)
    body = json.loads(script.requests[0].content)
    assert body["stream"] is True
    assert body["response_format"]["json_schema"]["strict"] is True


def test_stream_structured_works_without_done_sentinel() -> None:
    full = _Greeting(language="French", greeting="Bonjour").model_dump_json()
    client = _client(_Script([_sse_response([full], usage={"prompt_tokens": 1, "completion_tokens": 1}, done=False)]))

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        list(stream.text_stream)
        assert stream.get_final_result().parsed.language == "French"


def test_stream_structured_missing_usage_defaults_to_zero() -> None:
    full = _Greeting(language="French", greeting="Bonjour").model_dump_json()
    client = _client(_Script([_sse_response([full])]))  # no usage on any chunk

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        list(stream.text_stream)
        result = stream.get_final_result()

    assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)


def test_stream_structured_get_final_result_propagates_validation_error() -> None:
    client = _client(_Script([_sse_response(['{"language": "French"}'])]))  # missing 'greeting'

    with client.stream_structured(system="s", user="u", response_model=_Greeting) as stream:
        list(stream.text_stream)
        with pytest.raises(ValidationError):
            stream.get_final_result()


def test_stream_structured_http_error_raises_provider_error() -> None:
    client = _client(_Script([httpx.Response(429, text="Tokens per minute limit exceeded")]))

    with pytest.raises(LLMProviderError) as exc_info:
        with client.stream_structured(system="s", user="u", response_model=_Greeting):
            pass

    assert exc_info.value.status_code == 429


def test_stream_structured_connection_failure_raises_provider_error() -> None:
    client = _client(_Script([_connect_error()]))

    with pytest.raises(LLMProviderError) as exc_info:
        with client.stream_structured(system="s", user="u", response_model=_Greeting):
            pass

    assert exc_info.value.status_code is None


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires CEREBRAS_API_KEY")
def test_generate_structured_returns_validated_model_and_usage() -> None:
    client = CerebrasClient()

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
