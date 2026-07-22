"""Core summarization service: build prompt -> call LLM -> validate -> repair-then-fail.

Policy (locked in at plan time): one repair attempt on a validation failure,
then fail loudly. This module owns that policy -- AnthropicClient stays a
thin, single-shot "call once, validate once" wrapper (see app/llm/client.py);
repair-then-fail is filing-summarization business logic, not a generic
client capability.
"""

from pydantic import ValidationError

from app.llm.client import AnthropicClient, LLMClient
from app.llm.prompts import DEFAULT_VERSION, get_prompt_module
from app.llm.schemas import FilingAnalysis, FilingSummary
from app.services.edgar import FilingMeta


class SummarizationFailedError(Exception):
    """Raised when the repair attempt also fails validation -- fail loudly,
    not silently. Carries both errors so the caller can see what the model
    got wrong on each attempt."""

    def __init__(self, ticker: str, original_error: ValidationError, repair_error: ValidationError) -> None:
        self.ticker = ticker
        self.original_error = original_error
        self.repair_error = repair_error
        super().__init__(f"Summarization failed for {ticker!r} after a repair attempt: {repair_error}")


def summarize_filing(
    meta: FilingMeta,
    section_text: str,
    *,
    client: LLMClient | None = None,
    prompt_version: str = DEFAULT_VERSION,
) -> FilingSummary:
    """Summarize one filing excerpt into a validated FilingSummary.

    client is injectable so tests can exercise the repair path with a fake
    that deterministically fails validation, instead of depending on the
    real model happening to misbehave (see tests/test_summarize.py).
    """
    llm = client or AnthropicClient()
    prompt = get_prompt_module(prompt_version)
    user_prompt = prompt.build_user_prompt(meta, section_text)

    try:
        result = llm.generate_structured(
            system=prompt.SYSTEM_PROMPT,
            user=user_prompt,
            response_model=FilingAnalysis,
        )
    except ValidationError as original_error:
        repair_prompt = prompt.build_repair_user_prompt(user_prompt, original_error)
        try:
            result = llm.generate_structured(
                system=prompt.SYSTEM_PROMPT,
                user=repair_prompt,
                response_model=FilingAnalysis,
            )
        except ValidationError as repair_error:
            raise SummarizationFailedError(meta.ticker, original_error, repair_error) from repair_error

    return FilingSummary(meta=meta, analysis=result.parsed)
