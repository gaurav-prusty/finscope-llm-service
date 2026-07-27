"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest

from app.services.edgar import FilingMeta

FIXTURE_DIR = Path(__file__).parent / "fixtures"


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
