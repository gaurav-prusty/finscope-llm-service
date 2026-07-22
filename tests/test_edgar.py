"""Offline tests for app/services/edgar.py.

These only exercise the pure text-processing logic (HTML stripping, section
extraction, CIK formatting) -- no network calls. The live network path
(ticker -> CIK -> filing -> document) is exercised manually to produce the
committed fixture in tests/fixtures/; see the Part 1 verification notes.
"""

import pytest

from app.services.edgar import (
    RISK_FACTORS_ITEM_END_RES,
    RISK_FACTORS_ITEM_START_RE,
    _format_cik,
    extract_item_section,
    strip_html_to_text,
)


def test_format_cik_pads_to_ten_digits() -> None:
    assert _format_cik(320193) == "0000320193"


def test_strip_html_to_text_skips_script_and_style() -> None:
    html = """
    <html><body>
      <style>.x { color: red; }</style>
      <p>Hello world.</p>
      <script>alert('nope');</script>
      <div>Second paragraph.</div>
    </body></html>
    """
    text = strip_html_to_text(html)
    assert "Hello world." in text
    assert "Second paragraph." in text
    assert "alert" not in text
    assert "color: red" not in text


def test_extract_item_section_skips_table_of_contents_occurrence() -> None:
    text = (
        "TABLE OF CONTENTS\n"
        "Item 1A. Risk Factors ... 12\n"
        "Item 1B. Unresolved Staff Comments ... 30\n"
        "\n"
        "Item 1A. Risk Factors\n"
        "Our business faces significant risks.\n"
        "\n"
        "Item 1B. Unresolved Staff Comments\n"
        "None.\n"
    )
    section = extract_item_section(
        text,
        start_re=RISK_FACTORS_ITEM_START_RE,
        end_res=RISK_FACTORS_ITEM_END_RES,
    )
    assert section.startswith("Item 1A. Risk Factors")
    assert "significant risks" in section
    assert "Unresolved Staff Comments" not in section


def test_extract_item_section_raises_when_start_missing() -> None:
    with pytest.raises(ValueError):
        extract_item_section("nothing relevant here", r"item\s+1a", [r"item\s+1b"])
