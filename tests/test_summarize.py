"""Tests for app/services/summarize.py -- the repair-then-fail policy.

The repair path is exercised with a fake LLMClient that deterministically
raises pydantic.ValidationError, rather than depending on the real model
happening to misbehave -- that's exactly why LLMClient is an ABC (Part 3):
this is the moment the abstraction earns its keep. The one live-gated test
at the bottom is the real end-to-end check against the actual model,
including whether Part 4's "may lack financial figures" concern is real.
"""

from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.llm.client import LLMClient, LLMResult, LLMStream, LLMUsage
from app.llm.schemas import FilingAnalysis, FilingSummary
from app.services.summarize import (
    SummarizationFailedError,
    StreamDelta,
    StreamDone,
    StreamError,
    stream_summarize_filing,
    summarize_filing,
)

_HAS_API_KEY = bool(get_settings().anthropic_api_key)


def _valid_analysis() -> FilingAnalysis:
    return FilingAnalysis(
        financial_highlights=[{"metric": "Total net sales", "value": "$416.2 billion", "period": "FY2025"}],
        risk_factors=[{"category": "regulatory", "summary": "Antitrust scrutiny of App Store practices."}],
        sentiment="confident",
        sentiment_rationale="Management emphasizes continued growth despite headwinds.",
    )


def _make_validation_error() -> ValidationError:
    try:
        FilingAnalysis(
            financial_highlights=[],  # violates min_length=1
            risk_factors=[{"category": "regulatory", "summary": "x"}],
            sentiment="confident",
            sentiment_rationale="x",
        )
    except ValidationError as e:
        return e
    raise AssertionError("expected FilingAnalysis(...) to raise ValidationError")


class _FakeLLMClient(LLMClient):
    """Plays back a scripted sequence of results/errors, one per
    generate_structured() call, plus a separate fixed script for
    stream_structured() (deltas + a final result-or-error)."""

    def __init__(self, moves: list, stream_deltas: list[str] | None = None, stream_final=None) -> None:
        self._moves = list(moves)
        self.calls: list[dict] = []
        self._stream_deltas = stream_deltas or []
        self._stream_final = stream_final
        self.stream_calls: list[dict] = []

    def generate_structured(self, *, system, user, response_model):
        self.calls.append({"system": system, "user": user, "response_model": response_model})
        move = self._moves.pop(0)
        if isinstance(move, Exception):
            raise move
        return LLMResult(parsed=move, usage=LLMUsage(input_tokens=1, output_tokens=1), model="fake-model")

    @contextmanager
    def stream_structured(self, *, system, user, response_model):
        self.stream_calls.append({"system": system, "user": user, "response_model": response_model})

        def get_final_result():
            if isinstance(self._stream_final, Exception):
                raise self._stream_final
            return LLMResult(
                parsed=self._stream_final, usage=LLMUsage(input_tokens=1, output_tokens=1), model="fake-model"
            )

        yield LLMStream(text_stream=iter(self._stream_deltas), get_final_result=get_final_result)


def test_summarize_filing_returns_valid_summary_on_first_attempt(aapl_filing) -> None:
    meta, section_text = aapl_filing
    fake = _FakeLLMClient([_valid_analysis()])

    summary = summarize_filing(meta, section_text, client=fake)

    assert isinstance(summary, FilingSummary)
    assert summary.meta.ticker == "AAPL"
    assert len(fake.calls) == 1


def test_summarize_filing_repairs_after_first_validation_failure(aapl_filing) -> None:
    meta, section_text = aapl_filing
    fake = _FakeLLMClient([_make_validation_error(), _valid_analysis()])

    summary = summarize_filing(meta, section_text, client=fake)

    assert isinstance(summary, FilingSummary)
    assert len(fake.calls) == 2
    repair_user_prompt = fake.calls[1]["user"]
    assert "did not satisfy the required output format" in repair_user_prompt
    assert section_text in repair_user_prompt  # original filing content is re-sent, not dropped


def test_summarize_filing_fails_loudly_after_repair_also_fails(aapl_filing) -> None:
    meta, section_text = aapl_filing
    fake = _FakeLLMClient([_make_validation_error(), _make_validation_error()])

    with pytest.raises(SummarizationFailedError) as exc_info:
        summarize_filing(meta, section_text, client=fake)

    assert exc_info.value.ticker == "AAPL"
    assert len(fake.calls) == 2  # exactly one repair attempt -- no unbounded retry


def test_stream_summarize_filing_yields_deltas_then_done(aapl_filing) -> None:
    meta, section_text = aapl_filing
    fake = _FakeLLMClient([], stream_deltas=["Hello", " world"], stream_final=_valid_analysis())

    events = list(stream_summarize_filing(meta, section_text, client=fake))

    assert [e.text for e in events if isinstance(e, StreamDelta)] == ["Hello", " world"]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].summary.meta.ticker == "AAPL"


def test_stream_summarize_filing_yields_error_on_validation_failure_no_repair(aapl_filing) -> None:
    """No repair attempt for the streaming path (see app/llm/client.py's
    docstring for why) -- confirmed here by asserting generate_structured
    was never called."""
    meta, section_text = aapl_filing
    fake = _FakeLLMClient([], stream_deltas=["partial content"], stream_final=_make_validation_error())

    events = list(stream_summarize_filing(meta, section_text, client=fake))

    assert events[0] == StreamDelta(text="partial content")
    assert isinstance(events[-1], StreamError)
    assert "failed validation" in events[-1].detail
    assert fake.calls == []  # generate_structured (the repair path) was never touched


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_summarize_filing_end_to_end_on_real_fixture(aapl_filing) -> None:
    meta, section_text = aapl_filing

    summary = summarize_filing(meta, section_text)

    assert isinstance(summary, FilingSummary)
    assert summary.meta.ticker == "AAPL"
    assert len(summary.analysis.financial_highlights) >= 1
    assert len(summary.analysis.risk_factors) >= 1


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_stream_summarize_filing_end_to_end_on_real_fixture(aapl_filing) -> None:
    meta, section_text = aapl_filing

    events = list(stream_summarize_filing(meta, section_text))

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    assert len(deltas) > 0  # real streaming actually produced incremental chunks
    assert isinstance(events[-1], StreamDone)
    assert events[-1].summary.meta.ticker == "AAPL"
