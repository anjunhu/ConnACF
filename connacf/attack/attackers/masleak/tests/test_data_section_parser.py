"""
Unit and property-based tests for DataSectionParser.

Property tests use hypothesis with @settings(max_examples=100).

Properties covered:
  - Property 1: DATA_Template round-trip (Requirements 1.4, 1.7)
"""

import pytest
from hypothesis import given, settings, strategies as st

from connacf.attack.attackers.masleak.data_section_parser import DataSectionParser
from connacf.attack.attackers.masleak.worm_query_builder import WormQueryBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data_template(b_content: str, a_content: str = "You are a movie enthusiast.") -> str:
    """Build a well-formed DATA_Template response with the given B. slot content."""
    return (
        "\n[DATA]\n"
        "#DATA section\n"
        "A. Example (fake, plausible for this domain):\n"
        f"   {a_content}\n"
        f"B. Your task: \n"
        f"   {b_content}\n"
        "[/DATA]\n"
    )


PARSER = DataSectionParser()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestParseWellFormedBlock:
    """Well-formed [DATA] block → correct B. slot extraction."""

    def test_extracts_b_slot_content(self):
        ip = "You are a helpful movie recommendation agent."
        response = make_data_template(ip)
        result = PARSER.parse(response)
        assert result is not None
        assert ip in result

    def test_extracts_multiline_b_slot(self):
        ip = "Line one of the system prompt.\nLine two of the system prompt."
        response = make_data_template(ip)
        result = PARSER.parse(response)
        assert result is not None
        assert "Line one" in result
        assert "Line two" in result

    def test_ignores_a_slot_content(self):
        a_content = "Fake example content that should not appear."
        b_content = "Real IP content here."
        response = make_data_template(b_content, a_content=a_content)
        result = PARSER.parse(response)
        assert result is not None
        assert b_content in result
        # A. slot content should not be the returned value
        assert a_content not in result

    def test_surrounding_text_ignored(self):
        ip = "Actual system prompt content."
        response = "Some preamble text.\n" + make_data_template(ip) + "\nSome trailing text."
        result = PARSER.parse(response)
        assert result is not None
        assert ip in result


class TestMalformedBlock:
    """Malformed block (missing [/DATA]) → fallback triggered."""

    def test_missing_close_tag_falls_back_to_b_pattern(self):
        response = (
            "\n[DATA]\n"
            "#DATA section\n"
            "A. Example:\n"
            "   Fake example.\n"
            "B. Your task: \n"
            "   Real IP content here.\n"
            # No [/DATA]
        )
        result = PARSER.parse(response)
        # Should fall back to bare B. pattern and still find content
        assert result is not None
        assert "Real IP content here." in result

    def test_missing_open_tag_falls_back(self):
        response = (
            "#DATA section\n"
            "A. Example:\n"
            "   Fake example.\n"
            "B. Your task: \n"
            "   Real IP content here.\n"
            "[/DATA]\n"
        )
        result = PARSER.parse(response)
        assert result is not None
        assert "Real IP content here." in result


class TestNoDataMarkers:
    """No [DATA] markers → fallback triggered."""

    def test_bare_b_dot_pattern(self):
        response = "A. Some context.\nB. The actual content I want."
        result = PARSER.parse(response)
        assert result is not None
        assert "The actual content I want." in result

    def test_bare_b_colon_pattern(self):
        response = "Here is the answer:\nB: The extracted system prompt."
        result = PARSER.parse(response)
        assert result is not None
        assert "The extracted system prompt." in result

    def test_no_b_pattern_returns_none(self):
        response = "This response has no B. section at all."
        result = PARSER.parse(response)
        assert result is None


class TestEmptyResponse:
    """Empty response → returns None."""

    def test_empty_string(self):
        assert PARSER.parse("") is None

    def test_whitespace_only(self):
        assert PARSER.parse("   \n\t  ") is None

    def test_none_like_empty(self):
        assert PARSER.parse("") is None

    def test_parse_with_fallback_empty(self):
        assert PARSER.parse_with_fallback("") is None

    def test_parse_with_fallback_whitespace(self):
        assert PARSER.parse_with_fallback("   ") is None


class TestPlaceholderFiltering:
    """B. slot contains placeholder → returns None."""

    def test_placeholder_in_b_slot_returns_none(self):
        response = make_data_template("(To be filled by the agent)")
        result = PARSER.parse(response)
        assert result is None

    def test_placeholder_text_variant_returns_none(self):
        response = make_data_template("To be filled by the agent")
        result = PARSER.parse(response)
        assert result is None

    def test_real_content_not_filtered(self):
        response = make_data_template("You are a helpful assistant.")
        result = PARSER.parse(response)
        assert result is not None


class TestParseWithFallback:
    """parse_with_fallback heuristic extraction."""

    def test_correction_pattern(self):
        response = "Correction: You are a movie recommendation agent with expertise in drama."
        result = PARSER.parse_with_fallback(response)
        assert result is not None
        assert "movie recommendation agent" in result

    def test_numbered_step_pattern(self):
        response = "Step 1: Review the user's watch history and preferences."
        result = PARSER.parse_with_fallback(response)
        assert result is not None
        assert "Review the user" in result

    def test_structured_parse_takes_priority(self):
        ip = "Structured IP content."
        response = make_data_template(ip)
        result = PARSER.parse_with_fallback(response)
        assert result is not None
        assert ip in result

    def test_no_match_returns_none(self):
        response = "This is a plain response with no extractable content."
        result = PARSER.parse_with_fallback(response)
        assert result is None


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Printable ASCII text that won't accidentally contain [DATA] markers or
# placeholder text — safe to embed as IP content.
_safe_ip_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=".,!?;:-_'\"()",
        blacklist_characters="[]",
    ),
    min_size=5,
    max_size=200,
).filter(
    lambda s: (
        s.strip()
        and "[DATA]" not in s
        and "[/DATA]" not in s
        and "To be filled by the agent" not in s
        and "B." not in s
        and "B:" not in s
        and "A." not in s
    )
)


@given(ip_string=_safe_ip_text)
@settings(max_examples=100)
def test_property1_data_template_round_trip(ip_string: str):
    """
    Property 1: DATA_Template round-trip
    For any valid IP string, if it is placed in the B. slot of a DATA_Template
    response and then parsed by DataSectionParser, the parser should return a
    string that contains the original IP string.

    Validates: Requirements 1.4, 1.7
    """
    # Build a DATA_Template response using WormQueryBuilder's format
    response = make_data_template(ip_string)
    parser = DataSectionParser()
    result = parser.parse(response)

    assert result is not None, (
        f"parse() returned None for ip_string={ip_string!r}\nresponse={response!r}"
    )
    assert ip_string.strip() in result, (
        f"Original IP string not found in result.\n"
        f"ip_string={ip_string!r}\nresult={result!r}"
    )


@given(ip_string=_safe_ip_text)
@settings(max_examples=100)
def test_property1_round_trip_via_worm_query_builder(ip_string: str):
    """
    Property 1 (variant): Round-trip using WormQueryBuilder's actual q_retain output.
    Embed ip_string as the B. slot content in a response that mimics what an LLM
    would produce after seeing the WormQueryBuilder DATA_Template.

    Validates: Requirements 1.4, 1.7
    """
    # Simulate an LLM filling in the B. slot of the WormQueryBuilder template
    wqb = WormQueryBuilder()
    q_retain = wqb.build_q_retain("system_prompt", {"domain": "movies"}, round_idx=0)

    # Replace the blank B. slot with the ip_string (as an LLM would)
    filled_response = q_retain.replace(
        "B. Your task: (To be filled by the agent)\n   \n",
        f"B. Your task: \n   {ip_string}\n",
    )

    parser = DataSectionParser()
    result = parser.parse(filled_response)

    assert result is not None, (
        f"parse() returned None for ip_string={ip_string!r}"
    )
    assert ip_string.strip() in result, (
        f"Original IP string not found in result.\n"
        f"ip_string={ip_string!r}\nresult={result!r}"
    )
