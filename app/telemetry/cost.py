"""Per-request token + cost telemetry (Part 7).

log_usage() is the one hook point every LLM call goes through -- wired into
every LLMClient implementation's generate_structured()/stream_structured()
(app/llm/client.py), the same
placement logic as Part 6's retry layer: "how much did this call cost" is a
client concern (any caller wants this), not filing-summarization business
policy that belongs in services/.

Pricing is $/million tokens, standard (non-promotional) rates. Some models
(e.g. Sonnet 5) have a time-limited "intro" discount; we deliberately price
at the stable standard rate rather than branching on today's date -- this is
an estimate for cost awareness, not a billing system, and a discount that
silently expires is a worse failure mode than a slightly conservative
estimate.
"""

import logging

logger = logging.getLogger(__name__)

# (input $/MTok, output $/MTok) -- from the Anthropic pricing table.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Explicitly (0.0, 0.0), not omitted -- this IS the free tier's real price,
    # not an unpriced/unknown model (see the None-vs-0.0 distinction below).
    # If usage ever exceeds the free-tier quota, Gemini rate-limits (429) at
    # the API-key tier rather than silently billing, so 0.0 stays accurate.
    "gemini-3.6-flash": (0.0, 0.0),
    # Same reasoning: Groq's Free tier is a separate, non-billing tier from
    # its opt-in "Developer" plan (console.groq.com/settings/billing/plans) --
    # exceeding the free quota 429s, it doesn't charge, unless you explicitly
    # upgrade. If you do upgrade, the real standard rate is $0.15/$0.60 per
    # MTok in/out (groq.com/pricing) -- update this entry if that happens.
    "openai/gpt-oss-120b": (0.0, 0.0),
    # Cerebras's free tier (no "openai/" prefix on its model id). Free-tier
    # overage 429s rather than bills, same reasoning as Groq above.
    "gpt-oss-120b": (0.0, 0.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """None (not 0.0) for an unpriced model -- zero would silently
    understate cost; None makes "we don't know" explicit to the caller."""
    pricing = PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def log_usage(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Logs the usage line and returns the estimated cost (or None if the
    model has no pricing entry) -- returned, not just logged, so callers can
    aggregate it later without re-parsing log lines."""
    cost = estimate_cost_usd(model, input_tokens, output_tokens)
    if cost is None:
        logger.warning(
            "llm_usage model=%s input_tokens=%d output_tokens=%d estimated_cost_usd=unknown (no pricing entry)",
            model,
            input_tokens,
            output_tokens,
        )
        return None

    logger.info(
        "llm_usage model=%s input_tokens=%d output_tokens=%d estimated_cost_usd=%.6f",
        model,
        input_tokens,
        output_tokens,
        cost,
    )
    return cost
