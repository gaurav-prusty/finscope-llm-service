"""Tests for the /summarize and /summarize/stream endpoints in app/main.py.

Endpoint tests monkeypatch app.main's own imported references to
fetch_filing_section / summarize_filing / stream_summarize_filing -- these
are already unit-tested in isolation (test_edgar.py, test_summarize.py);
here we're only testing HTTP wiring: request parsing, response shape,
exception -> status code mapping, and SSE formatting. The two live-gated
tests at the bottom exercise the real, unmocked pipeline end to end.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.llm.schemas import FilingAnalysis, FilingSummary
from app.main import app
from app.services.edgar import FilingMeta
from app.services.summarize import StreamDelta, StreamDone, StreamError, SummarizationFailedError

_HAS_API_KEY = bool(get_settings().anthropic_api_key)

client = TestClient(app)


def _fake_meta() -> FilingMeta:
    return FilingMeta(
        ticker="AAPL",
        company_name="Apple Inc.",
        cik="0000320193",
        accession_number="0000320193-25-000079",
        form="10-K",
        filing_date="2025-10-31",
        report_date="2025-09-27",
        primary_document="aapl-20250927.htm",
    )


def _fake_analysis() -> FilingAnalysis:
    return FilingAnalysis(
        financial_highlights=[{"metric": "Total net sales", "value": "$416.2 billion", "period": "FY2025"}],
        risk_factors=[{"category": "regulatory", "summary": "Antitrust scrutiny of App Store practices."}],
        sentiment="confident",
        sentiment_rationale="Management emphasizes continued growth despite headwinds.",
    )


def test_summarize_endpoint_returns_filing_summary(monkeypatch) -> None:
    meta = _fake_meta()
    monkeypatch.setattr("app.main.fetch_filing_section", lambda *a, **kw: (meta, "some risk factors text"))
    monkeypatch.setattr("app.main.summarize_filing", lambda *a, **kw: FilingSummary(meta=meta, analysis=_fake_analysis()))

    response = client.post("/summarize", json={"ticker": "AAPL"})

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["ticker"] == "AAPL"
    assert body["analysis"]["sentiment"] == "confident"


def test_summarize_endpoint_unknown_ticker_returns_404(monkeypatch) -> None:
    def _raise_unknown_ticker(*a, **kw):
        raise ValueError("No CIK found for ticker 'BOGUS'")

    monkeypatch.setattr("app.main.fetch_filing_section", _raise_unknown_ticker)

    response = client.post("/summarize", json={"ticker": "BOGUS"})

    assert response.status_code == 404
    assert "BOGUS" in response.json()["detail"]


def test_summarize_endpoint_summarization_failure_returns_502(monkeypatch) -> None:
    meta = _fake_meta()
    monkeypatch.setattr("app.main.fetch_filing_section", lambda *a, **kw: (meta, "some risk factors text"))

    def _raise_summarization_failed(*a, **kw):
        raise SummarizationFailedError("AAPL", ValueError("first"), ValueError("second"))  # type: ignore[arg-type]

    monkeypatch.setattr("app.main.summarize_filing", _raise_summarization_failed)

    response = client.post("/summarize", json={"ticker": "AAPL"})

    assert response.status_code == 502


def test_summarize_stream_endpoint_emits_sse_events(monkeypatch) -> None:
    meta = _fake_meta()
    monkeypatch.setattr("app.main.fetch_filing_section", lambda *a, **kw: (meta, "some risk factors text"))

    def _fake_stream(*a, **kw):
        yield StreamDelta(text="Hello")
        yield StreamDelta(text=" world")
        yield StreamDone(summary=FilingSummary(meta=meta, analysis=_fake_analysis()))

    monkeypatch.setattr("app.main.stream_summarize_filing", _fake_stream)

    response = client.post("/summarize/stream", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert 'event: delta\ndata: {"text": "Hello"}' in body
    assert 'event: delta\ndata: {"text": " world"}' in body
    assert "event: done" in body
    assert '"ticker": "AAPL"' in body


def test_summarize_stream_endpoint_emits_error_event_on_validation_failure(monkeypatch) -> None:
    meta = _fake_meta()
    monkeypatch.setattr("app.main.fetch_filing_section", lambda *a, **kw: (meta, "some risk factors text"))

    def _fake_stream(*a, **kw):
        yield StreamDelta(text="partial")
        yield StreamError(detail="Streamed response failed validation: some reason")

    monkeypatch.setattr("app.main.stream_summarize_filing", _fake_stream)

    response = client.post("/summarize/stream", json={"ticker": "AAPL"})

    assert response.status_code == 200  # the stream itself starts fine -- the failure is an SSE event, not an HTTP error
    assert "event: error" in response.text
    assert "failed validation" in response.text


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_summarize_endpoint_end_to_end_real_pipeline() -> None:
    response = client.post("/summarize", json={"ticker": "AAPL"})

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["ticker"] == "AAPL"
    assert len(body["analysis"]["risk_factors"]) >= 1


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_summarize_stream_endpoint_end_to_end_real_pipeline() -> None:
    response = client.post("/summarize/stream", json={"ticker": "AAPL"})

    assert response.status_code == 200
    assert "event: delta" in response.text
    assert "event: done" in response.text
