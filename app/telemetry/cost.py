"""Per-request token + cost telemetry (Part 7).

log_usage() is the one hook point every LLM call goes through -- wired into
AnthropicClient.generate_structured() (app/llm/client.py), the same
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
