"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.llm.client import provider_supports_streaming
from app.services.edgar import FilingMeta

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def has_active_provider_key() -> bool:
    """Whether the currently configured LLM_PROVIDER (config.py) has a key
    set -- the correct skip condition for any live-gated test that calls the
    real pipeline via get_llm_client() (app/llm/client.py), since that
    function resolves to whichever provider is active, not always Anthropic."""
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return bool(settings.gemini_api_key)
    if settings.llm_provider == "groq":
        return bool(settings.groq_api_key)
    if settings.llm_provider == "cerebras":
        return bool(settings.cerebras_api_key)
    return bool(settings.anthropic_api_key)


def active_provider_supports_streaming() -> bool:
    """Whether the currently configured LLM_PROVIDER can stream structured
    output (see app/llm/client.py's provider_supports_streaming) -- live-gated
    streaming tests need to skip under an unsupported provider (e.g. Groq),
    not fail, since a 501 there is correct behavior, not a bug."""
    return provider_supports_streaming(get_settings().llm_provider)


def _load_filing_fixture(stem: str) -> tuple[FilingMeta, str]:
    meta_json = json.loads((FIXTURE_DIR / f"{stem}.meta.json").read_text(encoding="utf-8"))
    section_text = (FIXTURE_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    return FilingMeta(**meta_json), section_text


@pytest.fixture
def aapl_filing() -> tuple[FilingMeta, str]:
    return _load_filing_fixture("aapl_10k_risk_factors")


@pytest.fixture
def msft_filing() -> tuple[FilingMeta, str]:
    return _load_filing_fixture("msft_10k_risk_factors")
