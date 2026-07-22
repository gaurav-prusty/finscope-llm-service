"""Tests for the output contract in app/llm/schemas.py.

These matter more than most tests in this repo: FilingAnalysis IS the
contract every LLM response has to satisfy. A gap here is a gap in what
Part 5's repair-then-fail loop can actually catch.
"""

import pytest
from pydantic import ValidationError

from app.llm.schemas import (
    FilingAnalysis,
    FilingSummary,
    FinancialHighlight,
    RiskCategory,
    RiskFactor,
    Sentiment,
)
from app.services.edgar import FilingMeta


def _valid_analysis_kwargs() -> dict:
    return {
        "financial_highlights": [
            {"metric": "Total net sales", "value": "$416.2 billion", "period": "FY2025"},
        ],
        "risk_factors": [
            {"category": "regulatory", "summary": "Antitrust scrutiny of App Store practices."},
        ],
        "sentiment": "confident",
        "sentiment_rationale": "Management emphasizes continued growth despite headwinds.",
    }


def _valid_meta_kwargs() -> dict:
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "accession_number": "0000320193-25-000079",
        "form": "10-K",
        "filing_date": "2025-10-31",
        "report_date": "2025-09-27",
        "primary_document": "aapl-20250927.htm",
    }


# --- happy path ---


def test_valid_analysis_parses() -> None:
    analysis = FilingAnalysis(**_valid_analysis_kwargs())
    assert analysis.sentiment == Sentiment.CONFIDENT
    assert analysis.risk_factors[0].category == RiskCategory.REGULATORY
    assert analysis.caveats == []  # optional, defaults to empty


def test_caveats_accepts_explicit_list() -> None:
    analysis = FilingAnalysis(**_valid_analysis_kwargs(), caveats=["Segment breakdown not in this excerpt."])
    assert analysis.caveats == ["Segment breakdown not in this excerpt."]


def test_filing_summary_composes_meta_and_analysis() -> None:
    summary = FilingSummary(
        meta=FilingMeta(**_valid_meta_kwargs()),
        analysis=FilingAnalysis(**_valid_analysis_kwargs()),
    )
    assert summary.meta.ticker == "AAPL"
    assert summary.analysis.sentiment == Sentiment.CONFIDENT


# --- rejection: missing / wrong-type fields ---


def test_missing_required_field_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    del kwargs["sentiment"]
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


def test_invalid_sentiment_value_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    kwargs["sentiment"] = "ecstatic"  # not in the enum
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


def test_invalid_risk_category_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    kwargs["risk_factors"] = [{"category": "aliens", "summary": "..."}]
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


# --- rejection: the constraints structured output can't enforce server-side,
# but pydantic still enforces client-side (this IS the repair-loop's hook) ---


def test_empty_financial_highlights_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    kwargs["financial_highlights"] = []
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


def test_empty_risk_factors_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    kwargs["risk_factors"] = []
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


def test_blank_string_field_rejected() -> None:
    kwargs = _valid_analysis_kwargs()
    kwargs["sentiment_rationale"] = ""
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


def test_unknown_field_rejected() -> None:
    """extra='forbid' -> additionalProperties: false. If the model ever
    returns a field we didn't ask for, that's a signal something's off --
    we want validation to fail loudly, not silently swallow it."""
    kwargs = _valid_analysis_kwargs()
    kwargs["extra_unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        FilingAnalysis(**kwargs)


# --- structural guard: every nested object schema must be strict-output-safe ---


def _assert_no_open_objects(schema: dict) -> None:
    """Recursively assert every object schema sets additionalProperties: false.

    Guards against a future regression: someone adds a nested model to
    FilingAnalysis without ConfigDict(extra="forbid"), which would silently
    violate Anthropic's structured-output constraints (every object needs
    additionalProperties: false) -- see app/llm/schemas.py's module docstring.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False, f"open object schema: {schema}"
        for value in schema.values():
            _assert_no_open_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            _assert_no_open_objects(item)


def test_analysis_schema_has_no_open_objects() -> None:
    schema = FilingAnalysis.model_json_schema()
    _assert_no_open_objects(schema)


def test_financial_highlight_and_risk_factor_schemas_have_no_open_objects() -> None:
    _assert_no_open_objects(FinancialHighlight.model_json_schema())
    _assert_no_open_objects(RiskFactor.model_json_schema())
