"""The output contract: every LLM response must validate against FilingAnalysis.

Two models, deliberately kept separate -- this split is the core architectural
decision of Part 2:

  - FilingMeta (app/services/edgar.py): deterministic, sourced from our own
    EDGAR fetch. NEVER passed to the LLM. Company name, ticker, CIK, and
    filing/period dates are facts we already have -- asking an LLM to
    re-derive them from prose only invites hallucination on data we could
    just look up. (Same instinct as never trusting a computed field over the
    source-of-truth row when you already have the row.)

  - FilingAnalysis (below): the ONLY model passed to output_config.format.
    This is the actual product value -- everything a human couldn't get for
    free from the filing's own metadata.

FilingSummary composes the two into the full API response.

Design decisions (each is a real judgment call, not a default):
  - financial_highlights values stay strings, not floats: filings report
    units/formats inconsistently ("$416.2 billion" vs "416,199" thousands),
    and forcing a numeric type risks either mangled values or a parse
    failure on something a human would read as a number just fine.
  - risk_factors are categorized, not one prose blob: matches Item 1A's own
    itemized structure, and stays queryable/comparable across filings later
    (the actual "financial-research assistant" value), instead of shipping
    text a later phase has to regex back apart -- the exact anti-pattern
    SKILLS.md calls out.
  - sentiment is a tone enum + a grounding rationale, not a numeric score.
    LLM-generated continuous scores are known to be poorly calibrated and
    inconsistent run-to-run; an enum + a one-sentence rationale is honest
    about the precision we can actually claim, and gives us something
    concrete to spot-check in regression tests (Part 9).
  - caveats replace a self-reported "confidence" number, for the same
    calibration reason -- "what couldn't be determined" is a claim we can
    verify; "0.82 confidence" is not.

Structured-output constraints this schema is designed around (Anthropic
`output_config.format`): no recursive schemas, no numeric/string length
bounds enforced server-side, `additionalProperties: false` required on every
object (set via `model_config = ConfigDict(extra="forbid")` below). Field
constraints like `min_length` below ARE still enforced -- just client-side,
by pydantic, after the response comes back. That's not a consolation prize:
it's exactly the check our repair-then-fail loop (Part 5) will hook into.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.services.edgar import FilingMeta


class RiskCategory(str, Enum):
    REGULATORY = "regulatory"
    COMPETITION = "competition"
    SUPPLY_CHAIN = "supply_chain"
    MACROECONOMIC = "macroeconomic"
    CYBERSECURITY = "cybersecurity"
    LITIGATION = "litigation"
    OTHER = "other"


class Sentiment(str, Enum):
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    CONCERNED = "concerned"
    NEUTRAL = "neutral"


class FinancialHighlight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, description="Name of the metric, e.g. 'Total net sales'")
    value: str = Field(min_length=1, description="Reported value as stated in the filing, e.g. '$416.2 billion'")
    period: str = Field(min_length=1, description="The reporting period this value covers, e.g. 'FY2025'")


class RiskFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: RiskCategory
    summary: str = Field(min_length=1, description="One to two sentence plain-English summary of this risk")


class FilingAnalysis(BaseModel):
    """The only model passed to output_config.format -- what the LLM produces."""

    model_config = ConfigDict(extra="forbid")

    financial_highlights: list[FinancialHighlight] = Field(
        min_length=1,
        description="Key financial metrics reported in this filing",
    )
    risk_factors: list[RiskFactor] = Field(
        min_length=1,
        description="Top risk factors disclosed in this filing, categorized",
    )
    sentiment: Sentiment = Field(description="Overall tone of management's forward-looking language")
    sentiment_rationale: str = Field(
        min_length=1,
        description="One sentence grounding the sentiment label in specific filing content",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Anything this summary could not determine, or ambiguous/truncated source content",
    )


class FilingSummary(BaseModel):
    """The full API response: deterministic filing metadata + LLM-generated analysis."""

    model_config = ConfigDict(extra="forbid")

    meta: FilingMeta
    analysis: FilingAnalysis
