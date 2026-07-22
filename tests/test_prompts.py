"""Tests for app/llm/prompts -- template rendering only, no live LLM calls.

Verifies the version registry and that v1's templates render correctly
against a real fixture. Whether the resulting prompt makes the model produce
a good FilingAnalysis is Part 9's job (regression tests); this file only
checks that prompt construction itself is correct.
"""

import json
from pathlib import Path

import pytest

from app.llm.prompts import DEFAULT_VERSION, get_prompt_module, v1
from app.services.edgar import FilingMeta

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_aapl_fixture() -> tuple[FilingMeta, str]:
    meta_json = json.loads((FIXTURE_DIR / "aapl_10k_risk_factors.meta.json").read_text(encoding="utf-8"))
    section_text = (FIXTURE_DIR / "aapl_10k_risk_factors.txt").read_text(encoding="utf-8")
    return FilingMeta(**meta_json), section_text


def test_default_version_is_v1() -> None:
    assert DEFAULT_VERSION == "v1"
    assert get_prompt_module() is v1


def test_get_prompt_module_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_prompt_module("v99")


def test_system_prompt_states_grounding_constraint() -> None:
    assert "prior knowledge" in v1.SYSTEM_PROMPT.lower()


def test_user_prompt_includes_deterministic_meta_fields() -> None:
    meta, section_text = _load_aapl_fixture()
    prompt = v1.build_user_prompt(meta, section_text)

    assert meta.company_name in prompt
    assert meta.ticker in prompt
    assert meta.form in prompt
    assert meta.report_date in prompt


def test_user_prompt_includes_full_section_text() -> None:
    meta, section_text = _load_aapl_fixture()
    prompt = v1.build_user_prompt(meta, section_text)

    assert section_text in prompt
    assert prompt.startswith("Company: Apple Inc. (AAPL)")
