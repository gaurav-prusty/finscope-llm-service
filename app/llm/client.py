"""Provider-agnostic LLM client.

LLMClient is the seam between "call an LLM" and "what we do with the answer"
(services/summarize.py, Part 5). It knows nothing about filings or
FilingAnalysis -- it takes a system prompt, a user prompt, and a target
pydantic model, and returns a validated instance of that model. This keeps
every provider-specific type confined to its own client class
(AnthropicClient, GeminiClient, GroqClient, CerebrasClient); services/ only ever sees
LLMClient, LLMResult, and LLMStream. get_llm_client() at the bottom resolves
config.py's llm_provider setting to a concrete client -- the one place that
knows all three implementations exist.

Retry policy (Part 6): AnthropicClient retries transient call failures
(network errors, 429, 5xx) internally, with tenacity as the ONE authoritative
retry layer -- the SDK's own silent retry is disabled (max_retries=0 below)
so every attempt is deliberate and logged, instead of two invisible retry
layers compounding their backoffs. This is a *client* concern (any caller
wants a reliable call), unlike Part 5's repair-then-fail, which is
filing-summarization business policy and stays in services/summarize.py.
The two must never merge: pydantic.ValidationError (a well-formed response
that fails our schema) is never retried here -- only failures where the call
itself didn't complete are.

Cost telemetry (Part 7): every successful call logs its token counts and
estimated $ via telemetry/cost.py, for the same reason retry lives here --
"how much did this call cost" is a property of the call itself, not
filing-summarization policy, so every caller gets it for free.

Streaming (Part 8): stream_structured() exists because Anthropic's SDK only
validates a streamed structured response against our schema once the content
block completes -- near the END of the stream, not per-token (confirmed by
reading anthropic/lib/streaming/_messages.py: parse_text() runs on
content_block_stop, after nearly all text has already been yielded). That
means a caller streaming this to an HTTP client has already sent most of the
content before a validation failure could even be detected -- there is no
clean way to "take it back" mid-stream. So unlike generate_structured(),
stream_structured() gets NO repair-then-fail retry of its own; that
guarantee only exists on the non-streaming path. Callers needing a
guaranteed-valid result should use generate_structured(), not this.

Provider capability (Post-Phase-1): not every provider CAN stream structured
output at all -- Groq's API flatly rejects combining response_format
(schema-constrained JSON) with streaming (confirmed via
console.groq.com/docs/structured-outputs: "Streaming with Structured Outputs"
is explicitly listed as unsupported). LLMClient.supports_streaming is a class
flag callers (app/main.py) check BEFORE ever calling stream_structured(), so
an unsupported provider fails with a clean HTTP error before any response has
started -- not mid-stream, where headers are already committed and there's no
way to take it back (see StreamingNotSupportedError below).
"""

import copy
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
import groq
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel
from tenacity import Retrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.telemetry.cost import log_usage

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    parsed: T
    usage: LLMUsage
    model: str  # the model that actually served the request


@dataclass(frozen=True)
class LLMStream(Generic[T]):
    text_stream: Iterator[str]
    get_final_result: Callable[[], LLMResult[T]]


class StreamingNotSupportedError(Exception):
    """Raised when /summarize/stream is requested against a provider whose API
    can't combine streaming with schema-constrained structured output (see
    LLMClient.supports_streaming). Callers must check supports_streaming and
    raise this BEFORE starting a StreamingResponse -- once streaming begins,
    HTTP headers are already committed and there's no clean way to turn an
    in-progress stream into an error response."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"The active LLM provider ({provider!r}) does not support streaming structured "
            "output. Use POST /summarize (non-streaming) instead, or switch LLM_PROVIDER to "
            "a provider that supports streaming."
        )


class LLMClient(ABC):
    """Two methods: prompt in, validated pydantic object out -- either all
    at once, or as a live text stream with the validated object available
    only once the stream is exhausted.

    Contract (generate_structured): raises pydantic.ValidationError if the
    model's response doesn't satisfy response_model -- there is no "invalid
    result" return value, only a valid LLMResult or an exception. From the
    caller's perspective this is one logical call (repair-then-fail policy,
    catching that ValidationError, is the caller's job -- see
    services/summarize.py, Part 5); an implementation MAY retry transparently
    underneath for transient failures that aren't ValidationError, as
    AnthropicClient does.

    Contract (stream_structured): a context manager yielding an LLMStream.
    Iterate text_stream for live text, then call get_final_result() (which
    may raise pydantic.ValidationError, possibly after most of the content
    has already been streamed -- see this module's docstring). No retry of
    any kind here, transient or repair. Only call this if supports_streaming
    is True -- see StreamingNotSupportedError.
    """

    supports_streaming: bool = True

    @abstractmethod
    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        ...

    @abstractmethod
    def stream_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> AbstractContextManager[LLMStream[T]]:
        ...


def _is_transient_anthropic_error(exc: BaseException) -> bool:
    """Safe to retry: the call didn't complete, so nothing was double-done.
    Excludes 4xx client errors (bad request, auth, permissions, not found) --
    those fail identically on every retry, so retrying is pure wasted latency."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(
            api_key=api_key or settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # tenacity below is the one retry layer, not the SDK's own
        )
        self._model = model or settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._max_attempts = settings.llm_max_retries

    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        retryer = Retrying(
            retry=retry_if_exception(_is_transient_anthropic_error),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        response = retryer(
            self._client.messages.parse,
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=response_model,
        )
        log_usage(response.model, response.usage.input_tokens, response.usage.output_tokens)
        return LLMResult(
            parsed=response.parsed_output,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            model=response.model,
        )

    @contextmanager
    def stream_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> Iterator[LLMStream[T]]:
        # Anthropic's own stream() is itself a context manager -- nested here
        # so ITS __exit__ (closing the HTTP stream) runs when OURS does,
        # tied to the caller's `with` block (e.g. the SSE generator's
        # lifetime), not left dangling if get_final_result() is never called.
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=response_model,
        ) as raw_stream:

            def get_final_result() -> LLMResult[T]:
                message = raw_stream.get_final_message()
                log_usage(message.model, message.usage.input_tokens, message.usage.output_tokens)
                return LLMResult(
                    parsed=message.parsed_output,
                    usage=LLMUsage(
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens,
                    ),
                    model=message.model,
                )

            yield LLMStream(text_stream=raw_stream.text_stream, get_final_result=get_final_result)


def _is_transient_gemini_error(exc: BaseException) -> bool:
    """Same intent as _is_transient_anthropic_error, adapted to Gemini's
    flatter exception model: one APIError class carrying an HTTP status
    code, rather than a set of subclasses to isinstance-check."""
    if not isinstance(exc, genai_errors.APIError):
        return False
    return exc.code == 429 or exc.code >= 500


class GeminiClient(LLMClient):
    """The second LLMClient implementation -- confirms the Part 3 bet that
    keeping this interface task-agnostic and provider-agnostic would let a
    new provider slot in without touching services/, main.py, or the prompts.
    Chosen as the default provider (config.py) when Anthropic credits ran
    low; genuinely free-tier at the pinned model (gemini-3.6-flash -- see
    config.py's comment for why the originally-pinned gemini-2.5-flash had to
    be swapped out).

    Does not trust response.parsed's failure semantics (undocumented whether
    it raises pydantic.ValidationError the same deterministic way Anthropic's
    .parse() does) -- validates the raw JSON text itself instead, so the
    contract callers see (raises ValidationError, or returns a valid
    LLMResult, nothing in between) is identical across both providers
    regardless of what each SDK does internally.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = genai.Client(
            api_key=api_key or settings.gemini_api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000)  # SDK wants milliseconds
            ),
            # No retryOptions -- unlike Anthropic's client (which defaults to
            # retrying and has to be told max_retries=0), google-genai's
            # client does not auto-retry unless configured to, so tenacity
            # below is already the only retry layer with no override needed.
        )
        self._model = model or settings.gemini_model
        self._max_tokens = settings.llm_max_tokens
        self._max_attempts = settings.llm_max_retries

    def _config(self, system: str, response_model: type[T]) -> genai_types.GenerateContentConfig:
        # response_json_schema, not response_schema: response_schema is the SDK's
        # own restricted Schema type (a subset of OpenAPI 3.0) and does NOT
        # support additionalProperties -- passing response_model directly there
        # 400s with "Unknown name additional_properties", since Part 2 deliberately
        # sets additionalProperties: false on every nested schema (required for
        # Anthropic's structured outputs). response_json_schema takes a raw JSON
        # Schema dict and explicitly documents additionalProperties support
        # (confirmed by reading its field description on the installed SDK), so
        # model_json_schema() can be passed through as-is.
        return genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self._max_tokens,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
        )

    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        retryer = Retrying(
            retry=retry_if_exception(_is_transient_gemini_error),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        response = retryer(
            self._client.models.generate_content,
            model=self._model,
            contents=user,
            config=self._config(system, response_model),
        )
        parsed = response_model.model_validate_json(response.text)
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        log_usage(self._model, input_tokens, output_tokens)
        return LLMResult(
            parsed=parsed,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=self._model,
        )

    @contextmanager
    def stream_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> Iterator[LLMStream[T]]:
        # Unlike Anthropic's messages.stream(), this SDK gives a plain
        # iterator of response chunks, not a context manager with its own
        # get_final_message(). We accumulate chunks ourselves and validate
        # once exhausted. Known limitation (matches our one actual call
        # site, services/summarize.py, which always fully iterates
        # text_stream before calling get_final_result -- but unlike
        # Anthropic's version, this one does NOT force-drain the stream if
        # get_final_result() is called early): calling get_final_result()
        # before text_stream is exhausted will validate incomplete text.
        chunks: list[genai_types.GenerateContentResponse] = []

        def _text_stream() -> Iterator[str]:
            for chunk in self._client.models.generate_content_stream(
                model=self._model,
                contents=user,
                config=self._config(system, response_model),
            ):
                chunks.append(chunk)
                if chunk.text:
                    yield chunk.text

        def get_final_result() -> LLMResult[T]:
            full_text = "".join(c.text for c in chunks if c.text)
            parsed = response_model.model_validate_json(full_text)
            # Streamed usage_metadata is a known rough edge for this SDK
            # (not every chunk carries it) -- take it from whichever chunk
            # has it, defaulting to 0 rather than crashing if none do.
            usage_chunk = next((c for c in reversed(chunks) if c.usage_metadata), None)
            input_tokens = usage_chunk.usage_metadata.prompt_token_count if usage_chunk else 0
            output_tokens = usage_chunk.usage_metadata.candidates_token_count if usage_chunk else 0
            log_usage(self._model, input_tokens, output_tokens)
            return LLMResult(
                parsed=parsed,
                usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                model=self._model,
            )

        yield LLMStream(text_stream=_text_stream(), get_final_result=get_final_result)


def _is_transient_groq_error(exc: BaseException) -> bool:
    """Same shape as _is_transient_anthropic_error -- Groq's SDK is
    Stainless-generated (same toolchain as Anthropic's and OpenAI's), so it
    has the identical APIConnectionError/RateLimitError/APIStatusError
    hierarchy, just under the groq module instead of anthropic."""
    if isinstance(exc, (groq.APIConnectionError, groq.RateLimitError)):
        return True
    return isinstance(exc, groq.APIStatusError) and exc.status_code >= 500


def _to_groq_strict_schema(schema: dict) -> dict:
    """Groq's strict mode is stricter than Anthropic's or Gemini's about schema
    SHAPE (both rejections hit live, as 400s, the first time a real schema was
    sent): every object must set additionalProperties: false, and every
    property must be listed in `required` -- a field with a pydantic default
    (FilingAnalysis.caveats) is left out of `required` by model_json_schema(),
    which Groq rejects outright. Adapting a copy here keeps that quirk inside
    the Groq client instead of reshaping the provider-neutral contract in
    schemas.py (owner's call, post-Phase-1) -- the response is still validated
    against the original response_model, so the default stays meaningful
    there; Groq just always sends the field (possibly as an empty list).
    CerebrasClient reuses this unchanged -- same gpt-oss model family, same
    two strict-mode rules (confirmed in Cerebras's structured-outputs docs)."""
    strict = copy.deepcopy(schema)

    def _visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(strict)
    return strict


class GroqClient(LLMClient):
    """The third LLMClient implementation -- added when Gemini's free tier
    turned out to be only 20 requests/day/model/project (discovered live, see
    CLAUDE.md's Post-Phase-1 entry), too tight for real dev use. Groq's free
    tier is 1,000 requests/day on the models that support structured outputs
    (openai/gpt-oss-20b, openai/gpt-oss-120b -- NOT llama-3.3-70b-versatile,
    despite that being the model usually cited for Groq's rate limits;
    confirmed via console.groq.com/docs/structured-outputs).

    supports_streaming = False: Groq's API cannot combine response_format
    (schema-constrained JSON) with streaming at all -- not a client-side
    choice like GeminiClient's "don't trust .parsed", a hard API limitation.
    stream_structured() raises if called directly as a defensive backstop,
    but the primary guard is app/main.py checking supports_streaming before
    ever starting a StreamingResponse (see StreamingNotSupportedError).

    Validates response.choices[0].message.content itself via
    model_validate_json() rather than trusting strict-mode's guarantee
    blindly -- same defense-in-depth reasoning as GeminiClient, costs
    nothing, keeps the raise-or-return contract identical across all three
    providers regardless of what each one promises internally.

    The schema sent is _to_groq_strict_schema(response_model.model_json_schema()),
    not the raw pydantic schema -- see that function for the two shape rules
    Groq enforces that the other providers don't.
    """

    supports_streaming = False

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = groq.Groq(
            api_key=api_key or settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # tenacity below is the one retry layer, not the SDK's own
        )
        self._model = model or settings.groq_model
        self._max_tokens = settings.llm_max_tokens
        self._max_attempts = settings.llm_max_retries

    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        retryer = Retrying(
            retry=retry_if_exception(_is_transient_groq_error),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        response = retryer(
            self._client.chat.completions.create,
            model=self._model,
            max_completion_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _to_groq_strict_schema(response_model.model_json_schema()),
                },
            },
        )
        parsed = response_model.model_validate_json(response.choices[0].message.content)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        log_usage(self._model, input_tokens, output_tokens)
        return LLMResult(
            parsed=parsed,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=self._model,
        )

    def stream_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> AbstractContextManager[LLMStream[T]]:
        # Deliberately not @contextmanager -- raises immediately on the call
        # itself (before any `with` block runs), rather than on __enter__, so
        # this fails as fast as possible for any caller that skips the
        # supports_streaming check.
        raise NotImplementedError(
            "GroqClient does not support stream_structured(): Groq's API cannot combine "
            "response_format (schema-constrained JSON output) with streaming. Check "
            "supports_streaming before calling this -- app/main.py already does."
        )


class LLMProviderError(Exception):
    """A provider failure raised by clients that talk plain HTTP (CerebrasClient)
    instead of through a vendor SDK with its own exception hierarchy -- the one
    provider-neutral exception type, so retry classification and app/main.py's
    502 mapping work without importing anything provider-specific.
    status_code is None for transport-level failures (DNS, connect, timeout)
    where no HTTP response ever arrived."""

    def __init__(self, provider: str, status_code: int | None, detail: str) -> None:
        self.provider = provider
        self.status_code = status_code
        where = f"HTTP {status_code}" if status_code is not None else "no response"
        super().__init__(f"{provider} API error ({where}): {detail}")


def _is_transient_cerebras_error(exc: BaseException) -> bool:
    """Same intent as the other clients' classifiers: retry only when the call
    didn't complete (network failure, 429, 5xx), never on other 4xx."""
    if not isinstance(exc, LLMProviderError):
        return False
    return exc.status_code is None or exc.status_code == 429 or exc.status_code >= 500


class CerebrasClient(LLMClient):
    """The fourth LLMClient implementation -- added when Groq's free tier (8K
    tokens/min) and Gemini's (20 requests/day/model) both blocked a same-day
    deploy. Cerebras free tier: gpt-oss-120b, 65K context, 30K uncached
    tokens/min, 1M tokens/day, but only 5 requests/min (docs:
    inference-docs.cerebras.ai/support/rate-limits).

    Talks to the OpenAI-compatible /v1/chat/completions endpoint directly via
    httpx (already a dependency) rather than a vendor SDK -- no new install,
    no lockfile/Docker dependency change. Same strict json_schema response_format
    as GroqClient (same model family, same two shape rules: see
    _to_groq_strict_schema), and the same defense-in-depth: the returned text is
    re-validated with model_validate_json().

    Streaming: Cerebras's docs flag only the legacy json_object mode as
    incompatible with stream=true, not json_schema, and say token usage rides
    on the final chunk -- so stream_structured() parses the OpenAI-style SSE
    ("data: {...}" lines, delta.content) by hand. Like every streaming path
    here, it gets no repair-then-fail (see this module's docstring).
    """

    supports_streaming = True
    _BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._http = httpx.Client(
            base_url=self._BASE_URL,
            headers={"Authorization": f"Bearer {api_key or settings.cerebras_api_key}"},
            timeout=settings.llm_timeout_seconds,
        )
        self._model = model or settings.cerebras_model
        self._max_tokens = settings.llm_max_tokens
        self._max_attempts = settings.llm_max_retries

    def _post_chat(self, payload: dict) -> dict:
        try:
            response = self._http.post("/chat/completions", json=payload)
        except httpx.TransportError as exc:
            raise LLMProviderError("cerebras", None, f"{type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMProviderError("cerebras", response.status_code, response.text)
        return response.json()

    def _payload(self, system: str, user: str, response_model: type[T]) -> dict:
        return {
            "model": self._model,
            "max_completion_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _to_groq_strict_schema(response_model.model_json_schema()),
                },
            },
        }

    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        retryer = Retrying(
            retry=retry_if_exception(_is_transient_cerebras_error),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        data = retryer(self._post_chat, self._payload(system, user, response_model))
        parsed = response_model.model_validate_json(data["choices"][0]["message"]["content"])
        input_tokens = data["usage"]["prompt_tokens"]
        output_tokens = data["usage"]["completion_tokens"]
        log_usage(self._model, input_tokens, output_tokens)
        return LLMResult(
            parsed=parsed,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=self._model,
        )

    @contextmanager
    def stream_structured(
        self, *, system: str, user: str, response_model: type[T]
    ) -> Iterator[LLMStream[T]]:
        payload = {**self._payload(system, user, response_model), "stream": True}
        parts: list[str] = []
        usage: dict = {}

        # ExitStack so the HTTP stream is closed when the caller's `with` ends,
        # and so a connection failure at open time becomes an LLMProviderError
        # (the same one generate_structured raises) without also swallowing
        # exceptions the caller's own block throws into this generator.
        with ExitStack() as stack:
            try:
                response = stack.enter_context(self._http.stream("POST", "/chat/completions", json=payload))
            except httpx.TransportError as exc:
                raise LLMProviderError("cerebras", None, f"{type(exc).__name__}: {exc}") from exc
            if response.status_code >= 400:
                response.read()
                raise LLMProviderError("cerebras", response.status_code, response.text)

            def _text_stream() -> Iterator[str]:
                # OpenAI-style SSE: "data: {json}" lines, optional "data: [DONE]".
                # Token usage arrives on a late chunk (Cerebras docs) -- collected
                # whenever it appears; text comes from choices[0].delta.content
                # (gpt-oss reasoning, if streamed, uses a separate field we ignore).
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    if chunk.get("usage"):
                        usage.update(chunk["usage"])
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta", {}).get("content") if choices else None
                    if delta:
                        parts.append(delta)
                        yield delta

            def get_final_result() -> LLMResult[T]:
                parsed = response_model.model_validate_json("".join(parts))
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                log_usage(self._model, input_tokens, output_tokens)
                return LLMResult(
                    parsed=parsed,
                    usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    model=self._model,
                )

            yield LLMStream(text_stream=_text_stream(), get_final_result=get_final_result)


_PROVIDER_CLASSES: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
    "cerebras": CerebrasClient,
}


def get_llm_client() -> LLMClient:
    """The provider toggle (config.py's llm_provider) resolved to a concrete
    client -- the one place that knows all four implementations exist. Every
    other call site (services/summarize.py) asks for "the" LLMClient and
    never imports a concrete client directly."""
    settings = get_settings()
    try:
        return _PROVIDER_CLASSES[settings.llm_provider]()
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Expected one of "
            f"{sorted(_PROVIDER_CLASSES)}."
        ) from None


def provider_supports_streaming(provider: str) -> bool:
    """A class-level capability check -- deliberately does NOT construct a
    client (which would require a real API key and would fail eagerly for
    some SDKs, e.g. Gemini's __init__ validates the key immediately). Used by
    app/main.py to reject an unsupported provider's /summarize/stream request
    BEFORE calling get_llm_client(), so mocked HTTP-wiring tests that never
    touch a real client still work regardless of which provider is
    configured."""
    cls = _PROVIDER_CLASSES.get(provider)
    return cls.supports_streaming if cls is not None else True
