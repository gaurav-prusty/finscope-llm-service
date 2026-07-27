"""Tests for app/llm/prompts.

Two kinds of test live here:
  - Rendering tests (offline, no LLM calls): verify the version registry and
    that v1's templates render correctly against real fixtures.
  - Regression tests (Part 9, live-gated on ANTHROPIC_API_KEY): one real
    call per held-out filing (AAPL and MSFT), asserting on the shape/quality
    of what comes back. These do NOT re-test schema validity -- pydantic
    already guarantees that on every real call (Part 2/5); a
    schema-invalid response never reaches these assertions at all. What
    they catch instead is a prompt wording change that stays schema-valid
    but quietly gets worse -- e.g. under-extracting risks, dumping
    everything into "other", or losing the Part 4/5 fallback behavior that
    lets financial_highlights be satisfied from a Risk-Factors-only
    excerpt without hallucinating revenue figures. Thresholds below were
    set by running both fixtures live and checking the actual numbers, not
    guessed -- see CLAUDE.md's Part 9 entry. Per SKILLS.md's own rule,
    changing v1's wording means re-running these and updating them if the
    expected shape genuinely, deliberately changes.
"""

import pytest

from app.config import get_settings
from app.llm.prompts import DEFAULT_VERSION, get_prompt_module, v1
from app.llm.schemas import FilingAnalysis
from app.services.summarize import summarize_filing

_HAS_API_KEY = bool(get_settings().anthropic_api_key)


def test_default_version_is_v1() -> None:
    assert DEFAULT_VERSION == "v1"
    assert get_prompt_module() is v1


def test_get_prompt_module_unknown_version_raises() -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_prompt_module("v99")


def test_system_prompt_states_grounding_constraint() -> None:
    assert "prior knowledge" in v1.SYSTEM_PROMPT.lower()


def test_user_prompt_includes_deterministic_meta_fields(aapl_filing) -> None:
    meta, section_text = aapl_filing
    prompt = v1.build_user_prompt(meta, section_text)

    assert meta.company_name in prompt
    assert meta.ticker in prompt
    assert meta.form in prompt
    assert meta.report_date in prompt


def test_user_prompt_includes_full_section_text(aapl_filing) -> None:
    meta, section_text = aapl_filing
    prompt = v1.build_user_prompt(meta, section_text)

    assert section_text in prompt
    assert prompt.startswith("Company: Apple Inc. (AAPL)")


def _assert_reasonable_risk_factors_analysis(analysis: FilingAnalysis) -> None:
    # Under-extraction guard: a real 10-K risk factors section discusses
    # many distinct risks -- a regression that makes the model skim would
    # show up as a suspiciously short list. Live baseline: 8-10.
    assert len(analysis.risk_factors) >= 3

    # Categorization guard: everything landing in one bucket (especially
    # "other") suggests the category guidance broke. Live baseline: 7.
    categories_used = {r.category for r in analysis.risk_factors}
    assert len(categories_used) >= 2

    # The Part 4/5 fallback-instruction guard: this excerpt has no headline
    # financial figures, so financial_highlights >= 1 only holds up if the
    # model is still finding SOME concrete figure in the text.
    assert len(analysis.financial_highlights) >= 1

    # A pure risk-disclosure section should never read as "confident" -- if
    # it does, the model likely misread the section's purpose.
    assert analysis.sentiment.value != "confident"

    # The model should be flagging the missing-financial-data gap somehow;
    # an empty caveats list here is a red flag the caveats guidance broke.
    assert len(analysis.caveats) >= 1


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_v1_regression_aapl_risk_factors(aapl_filing) -> None:
    meta, section_text = aapl_filing
    summary = summarize_filing(meta, section_text)
    _assert_reasonable_risk_factors_analysis(summary.analysis)


@pytest.mark.skipif(not _HAS_API_KEY, reason="requires ANTHROPIC_API_KEY")
def test_v1_regression_msft_risk_factors(msft_filing) -> None:
    meta, section_text = msft_filing
    summary = summarize_filing(meta, section_text)
    _assert_reasonable_risk_factors_analysis(summary.analysis)
