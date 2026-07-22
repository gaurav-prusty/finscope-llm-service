"""Provider-agnostic LLM client.

LLMClient is the seam between "call an LLM" and "what we do with the answer"
(services/summarize.py, Part 5). It knows nothing about filings or
FilingAnalysis -- it takes a system prompt, a user prompt, and a target
pydantic model, and returns a validated instance of that model. This keeps
every Anthropic-specific type (the SDK client, its response shape) confined
to AnthropicClient; swapping providers later means writing one new class,
not touching services/.

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
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
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
    any kind here, transient or repair.
    """

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
