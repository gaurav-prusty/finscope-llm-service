"""Tests for app/telemetry/cost.py."""

import logging

from app.telemetry.cost import estimate_cost_usd, log_usage


def test_estimate_cost_usd_known_model_computes_expected_value() -> None:
    cost = estimate_cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 3.00 + 15.00


def test_estimate_cost_usd_scales_with_token_count() -> None:
    cost = estimate_cost_usd("claude-haiku-4-5", input_tokens=500_000, output_tokens=0)
    assert cost == 0.50  # half of $1.00/MTok input price


def test_estimate_cost_usd_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("some-future-model", input_tokens=100, output_tokens=100) is None


def test_log_usage_known_model_logs_info_and_returns_cost(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="app.telemetry.cost"):
        cost = log_usage("claude-sonnet-5", input_tokens=1000, output_tokens=500)

    assert cost == (1000 / 1_000_000) * 3.00 + (500 / 1_000_000) * 15.00
    assert any("llm_usage" in record.getMessage() for record in caplog.records)
    assert any(record.levelno == logging.INFO for record in caplog.records)


def test_log_usage_unknown_model_logs_warning_and_returns_none(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="app.telemetry.cost"):
        cost = log_usage("some-future-model", input_tokens=1000, output_tokens=500)

    assert cost is None
    assert any(record.levelno == logging.WARNING for record in caplog.records)
