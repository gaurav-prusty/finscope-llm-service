"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload

This is the Spring Boot @SpringBootApplication equivalent: it builds the app
object, and uvicorn (the ASGI server - think embedded Tomcat) serves it.
Routes are added with decorators instead of @RestController/@GetMapping, but
the shape is the same: a function per endpoint, a return value FastAPI
serializes to JSON.

Exception -> HTTP status mapping is centralized here via
@app.exception_handler, the FastAPI equivalent of Spring's
@ControllerAdvice/@ExceptionHandler - route bodies stay free of try/except
for errors that map the same way everywhere:
  - ValueError (edgar.py: unknown ticker / no matching filing)   -> 404
  - SummarizationFailedError (repair attempt also failed)        -> 502
  - anthropic.APIError (retries exhausted, or a non-retryable
    upstream failure - see llm/client.py's retry policy)         -> 502
Both endpoints are Item-1A-only for now (RISK_FACTORS_ITEM_*_RE in
services/edgar.py) - the only section this pipeline has been fixture-tested
against.
"""

import json
from collections.abc import Iterator

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.llm.schemas import FilingSummary
from app.middleware.ratelimit import RateLimitMiddleware, TokenBucket
from app.services.edgar import (
    RISK_FACTORS_ITEM_END_RES,
    RISK_FACTORS_ITEM_START_RE,
    FilingMeta,
    fetch_filing_section,
)
from app.services.summarize import (
    StreamDelta,
    StreamDone,
    StreamError,
    SummarizationFailedError,
    stream_summarize_filing,
    summarize_filing,
)

app = FastAPI(
    title="FinScope LLM Service",
    description="Turns SEC filings into validated, structured JSON summaries.",
    version="0.1.0",
)

_settings = get_settings()
app.add_middleware(
    RateLimitMiddleware,
    bucket=TokenBucket(
        capacity=_settings.rate_limit_capacity,
        refill_per_second=_settings.rate_limit_refill_per_second,
    ),
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(SummarizationFailedError)
async def summarization_failed_handler(request: Request, exc: SummarizationFailedError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(anthropic.APIError)
async def anthropic_api_error_handler(request: Request, exc: anthropic.APIError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": f"Upstream LLM provider error: {exc}"})


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. No LLM call here - this must stay fast and free."""
    settings = get_settings()
    return {"status": "ok", "env": settings.app_env}


class SummarizeRequest(BaseModel):
    ticker: str
    form_type: str = "10-K"


@app.post("/summarize")
def summarize(request: SummarizeRequest) -> FilingSummary:
    meta, section_text = fetch_filing_section(
        request.ticker,
        RISK_FACTORS_ITEM_START_RE,
        RISK_FACTORS_ITEM_END_RES,
        form_type=request.form_type,
    )
    return summarize_filing(meta, section_text)


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_events(meta: FilingMeta, section_text: str) -> Iterator[str]:
    for event in stream_summarize_filing(meta, section_text):
        if isinstance(event, StreamDelta):
            yield _format_sse("delta", {"text": event.text})
        elif isinstance(event, StreamDone):
            yield _format_sse("done", event.summary.model_dump(mode="json"))
        elif isinstance(event, StreamError):
            yield _format_sse("error", {"detail": event.detail})


@app.post("/summarize/stream")
def summarize_stream(request: SummarizeRequest) -> StreamingResponse:
    # The EDGAR fetch runs before the response starts (it's fast once cached;
    # SSE is about the LLM generation phase, not this step) -- a bad ticker
    # surfaces as a normal 404 via the ValueError handler above, before any
    # streaming begins.
    meta, section_text = fetch_filing_section(
        request.ticker,
        RISK_FACTORS_ITEM_START_RE,
        RISK_FACTORS_ITEM_END_RES,
        form_type=request.form_type,
    )
    return StreamingResponse(_sse_events(meta, section_text), media_type="text/event-stream")
