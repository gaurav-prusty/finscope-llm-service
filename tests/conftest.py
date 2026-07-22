"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest

from app.services.edgar import FilingMeta

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_filing() -> tuple[FilingMeta, str]:
    meta_json = json.loads((FIXTURE_DIR / "aapl_10k_risk_factors.meta.json").read_text(encoding="utf-8"))
    section_text = (FIXTURE_DIR / "aapl_10k_risk_factors.txt").read_text(encoding="utf-8")
    return FilingMeta(**meta_json), section_text
