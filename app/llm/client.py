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
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
from pydantic import BaseModel
from tenacity import Retrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

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


class LLMClient(ABC):
    """One method: prompt in, validated pydantic object out.

    Contract: raises pydantic.ValidationError if the model's response doesn't
    satisfy response_model -- there is no "invalid result" return value, only
    a valid LLMResult or an exception. From the caller's perspective this is
    one logical call (repair-then-fail policy, catching that ValidationError,
    is the caller's job -- see services/summarize.py, Part 5); an
    implementation MAY retry transparently underneath for transient failures
    that aren't ValidationError, as AnthropicClient does.
    """

    @abstractmethod
    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
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
        return LLMResult(
            parsed=response.parsed_output,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            model=response.model,
        )
