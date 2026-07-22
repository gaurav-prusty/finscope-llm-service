"""Polite, cached access to SEC EDGAR.

Two hosts are involved, with different jobs:
  - data.sec.gov          JSON metadata API (company filing lists, XBRL facts)
  - www.sec.gov/Archives  static file server for the actual filing documents

Both require a descriptive User-Agent per SEC's fair-access policy
(https://www.sec.gov/os/webmaster-faq#developers) -- treat it like an API key,
except it's just an honest "who is calling" string (see app/config.py).

Pipeline: ticker -> CIK -> filing list -> document HTML -> plain text -> section.
Everything here is synchronous (plain httpx.get, no asyncio) -- FastAPI runs
sync `def` route handlers in a thread pool automatically, so this stays simple
to read and test without adding async complexity this early.
"""

import json
import re
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx
from pydantic import BaseModel

from app.config import get_settings

CACHE_DIR = Path(".cache/edgar")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC's stated ceiling is 10 req/s; we stay well under it.
_MIN_REQUEST_INTERVAL_SECONDS = 0.25


class _RateLimiter:
    """Enforces a minimum gap between outbound requests.

    The hand-rolled equivalent of a Resilience4j RateLimiter bean: one lock,
    one "last call" timestamp. A cache hit in _cached_get skips this entirely.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = _RateLimiter(_MIN_REQUEST_INTERVAL_SECONDS)


def _cached_get(url: str, cache_key: str) -> bytes:
    """GET url with SEC's required headers; rate-limited; cached to disk.

    A cache hit returns immediately -- no network call, no rate-limit wait.
    This is what keeps repeated dev runs (and, later, tests that reuse the
    cache) fast and independent of SEC's servers being reachable.
    """
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return cache_path.read_bytes()

    settings = get_settings()
    _rate_limiter.wait()
    response = httpx.get(
        url,
        headers={"User-Agent": settings.sec_user_agent},
        timeout=15.0,
        follow_redirects=True,
    )
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return response.content


def _format_cik(cik: int) -> str:
    """SEC CIKs are plain ints in JSON but zero-padded to 10 digits in URLs."""
    return f"{cik:010d}"


def _find_ticker_entry(ticker: str) -> dict:
    """Look up a ticker's raw entry ({cik_str, ticker, title}) in SEC's
    published ticker->CIK mapping. Shared by get_company_cik and
    get_company_name so both resolve against the same cached lookup."""
    raw = _cached_get(TICKERS_URL, "company_tickers.json")
    tickers = json.loads(raw)  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    ticker_upper = ticker.upper()
    for entry in tickers.values():
        if entry["ticker"] == ticker_upper:
            return entry
    raise ValueError(f"No CIK found for ticker {ticker!r}")


def get_company_cik(ticker: str) -> str:
    """Resolve a ticker (e.g. 'AAPL') to its 10-digit zero-padded CIK."""
    return _format_cik(_find_ticker_entry(ticker)["cik_str"])


def get_company_name(ticker: str) -> str:
    """Resolve a ticker to the company's registered name, e.g. 'Apple Inc.'."""
    return _find_ticker_entry(ticker)["title"]


class FilingMeta(BaseModel):
    """One filing's metadata -- deterministic, sourced entirely from our own
    EDGAR fetch. This never passes through the LLM: see app/llm/schemas.py
    for why (we don't ask a model to re-derive facts we already have)."""

    ticker: str
    company_name: str
    cik: str
    accession_number: str
    form: str
    filing_date: str  # when the filing was submitted to SEC
    report_date: str  # the fiscal period this filing actually covers
    primary_document: str


def get_recent_filings(
    cik: str,
    ticker: str,
    company_name: str,
    form_type: str | None = None,
    limit: int = 5,
) -> list[FilingMeta]:
    """List a company's most recent filings, optionally filtered by form type.

    The submissions endpoint returns a "struct of arrays" (parallel lists,
    indexed together) rather than the more usual "array of structs" -- we
    zip them back into one FilingMeta per filing here.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    raw = _cached_get(url, f"submissions_{cik}.json")
    recent = json.loads(raw)["filings"]["recent"]

    filings = [
        FilingMeta(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            accession_number=accession,
            form=form,
            filing_date=filing_date,
            report_date=report_date,
            primary_document=primary_document,
        )
        for accession, form, filing_date, report_date, primary_document in zip(
            recent["accessionNumber"],
            recent["form"],
            recent["filingDate"],
            recent["reportDate"],
            recent["primaryDocument"],
        )
        if form_type is None or form == form_type
    ]
    return filings[:limit]


def get_filing_document_html(filing: FilingMeta) -> str:
    """Fetch the raw HTML of a filing's primary document."""
    accession_no_dashes = filing.accession_number.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(filing.cik)}/{accession_no_dashes}/{filing.primary_document}"
    )
    cache_key = f"{filing.cik}_{accession_no_dashes}_{filing.primary_document}"
    raw = _cached_get(url, cache_key)
    return raw.decode("utf-8", errors="replace")


_BLOCK_TAGS = {"p", "div", "tr", "br", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text: drops <script>/<style> content, turns common
    block tags into line breaks. Deliberately not a full renderer -- SEC
    filings are table/div-heavy, not a document we need pixel-perfect."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def strip_html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def extract_item_section(text: str, start_re: str, end_res: list[str]) -> str:
    """Slice out one Item's text from a full filing's plain text.

    Heuristic (documented caveat, not bulletproof): the heading also appears
    in the filing's Table of Contents near the top, so we take the LAST
    match of start_re as the real section start rather than the first. The
    section ends at the earliest of any end_res match found after that
    point; pass multiple candidates (e.g. Item 1B, Item 1C, Item 2) since
    which Item follows varies by filer and filing year.

    This is a v1 best-effort extractor -- real-world filings are messy.
    Validate on an actual filing before trusting it (see Part 1 fixture).
    """
    start_matches = list(re.finditer(start_re, text, re.IGNORECASE))
    if not start_matches:
        raise ValueError(f"section start pattern not found: {start_re!r}")
    start = start_matches[-1].start()

    end = len(text)
    for end_re in end_res:
        end_matches = [m for m in re.finditer(end_re, text, re.IGNORECASE) if m.start() > start]
        if end_matches:
            end = min(end, end_matches[0].start())

    return text[start:end].strip()


# Item 1A is the only section the pipeline has been built and fixture-tested
# against (see tests/fixtures/aapl_10k_risk_factors.*) -- centralized here as
# named constants so app/main.py's endpoints (Part 8) don't hand-roll regex,
# and so widening to other sections later (e.g. Item 7 MD&A, see Part 4's
# resolved note in CLAUDE.md) means adding a constant, not touching call sites.
RISK_FACTORS_ITEM_START_RE = r"item\s+1a\.?\s*risk\s+factors"
RISK_FACTORS_ITEM_END_RES = [r"item\s+1b", r"item\s+2"]


def fetch_filing_section(
    ticker: str,
    item_start: str,
    item_end_candidates: list[str],
    form_type: str = "10-K",
) -> tuple[FilingMeta, str]:
    """End-to-end: ticker -> most recent matching filing -> one plain-text section."""
    cik = get_company_cik(ticker)
    company_name = get_company_name(ticker)
    filings = get_recent_filings(cik, ticker=ticker, company_name=company_name, form_type=form_type, limit=1)
    if not filings:
        raise ValueError(f"No {form_type} filings found for {ticker}")
    filing = filings[0]

    html = get_filing_document_html(filing)
    text = strip_html_to_text(html)
    section = extract_item_section(text, item_start, item_end_candidates)
    return filing, section
