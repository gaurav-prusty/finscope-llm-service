"""Provider-agnostic LLM client.

LLMClient is the seam between "call an LLM" and "what we do with the answer"
(services/summarize.py, Part 5). It knows nothing about filings or
FilingAnalysis -- it takes a system prompt, a user prompt, and a target
pydantic model, and returns a validated instance of that model. This keeps
every Anthropic-specific type (the SDK client, its response shape) confined
to AnthropicClient; swapping providers later means writing one new class,
not touching services/.

No retry/backoff logic here on top of the SDK's own defaults (2 retries on
429/5xx) -- that's Part 6's job, layered on top of this client, not inside it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import anthropic
from pydantic import BaseModel

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)


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
    a valid LLMResult or an exception. This one call is intentionally
    single-shot with no retry of its own; repair-then-fail policy (catching
    that ValidationError and retrying with a repair prompt) is the caller's
    job -- see services/summarize.py (Part 5).
    """

    @abstractmethod
    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(
            api_key=api_key or settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        self._model = model or settings.llm_model
        self._max_tokens = settings.llm_max_tokens

    def generate_structured(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        response = self._client.messages.parse(
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
