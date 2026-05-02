"""
Unit and property-based tests for ExtractionAggregator.

Property tests use hypothesis with @settings(max_examples=100).

Properties covered:
  - Property 4: find_matched_content symmetry (Requirement 2.3)
  - Property 5: find_matched_content is a common substring (Requirement 2.3)
  - Property 6: Consensus subsumes candidates (Requirements 2.4, 2.5)
"""

import pytest
from hypothesis import given, settings, strategies as st

from connacf.attack.attackers.masleak.extraction_aggregator import ExtractionAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agg(min_match_length: int = 20) -> ExtractionAggregator:
    return ExtractionAggregator(min_match_length=min_match_length)


# ---------------------------------------------------------------------------
# Unit tests — find_matched_content
# ---------------------------------------------------------------------------

class TestFindMatchedContent:
    """Unit tests for find_matched_content."""

    def test_symmetry(self):
        agg = make_agg()
        a = "You are a movie recommendation agent with expertise in drama."
        b = "You are a movie recommendation system focused on drama films."
        assert agg.find_matched_content(a, b) == agg.find_matched_content(b, a)

    def test_empty_first_arg(self):
        agg = make_agg()
        assert agg.find_matched_content("", "some content here") == ""

    def test_empty_second_arg(self):
        agg = make_agg()
        assert agg.find_matched_content("some content here", "") == ""

    def test_both_empty(self):
        agg = make_agg()
        assert agg.find_matched_content("", "") == ""

    def test_identical_strings_above_threshold(self):
        agg = make_agg(min_match_length=20)
        s = "You are a helpful movie recommendation agent."
        result = agg.find_matched_content(s, s)
        assert result == s

    def test_disjoint_strings_returns_empty(self):
        agg = make_agg(min_match_length=20)
        a = "abcdefghijklmnopqrstuvwxyz"
        b = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        assert agg.find_matched_content(a, b) == ""

    def test_known_common_substring(self):
        agg = make_agg(min_match_length=10)
        common = "recommendation agent"
        a = "You are a recommendation agent for movies."
        b = "This recommendation agent specializes in drama."
        result = agg.find_matched_content(a, b)
        assert result != ""
        assert result in a
        assert result in b

    def test_common_substring_below_threshold_returns_empty(self):
        agg = make_agg(min_match_length=20)
        # Only 5 chars in common
        a = "hello world"
        b = "hello there"
        assert agg.find_matched_content(a, b) == ""

    def test_result_is_substring_of_both(self):
        agg = make_agg(min_match_length=5)
        a = "The quick brown fox jumps over the lazy dog"
        b = "A quick brown fox ran past the sleeping cat"
        result = agg.find_matched_content(a, b)
        if result:
            assert result in a
            assert result in b

    def test_result_length_at_least_min_match_length(self):
        agg = make_agg(min_match_length=15)
        a = "You are a movie recommendation agent with expertise."
        b = "You are a movie recommendation system for users."
        result = agg.find_matched_content(a, b)
        if result:
            assert len(result) >= 15


# ---------------------------------------------------------------------------
# Unit tests — get_consensus edge cases
# ---------------------------------------------------------------------------

class TestGetConsensusEdgeCases:
    """Unit tests for get_consensus with 0, 1, 2, and 3+ candidates."""

    def test_zero_candidates(self):
        agg = make_agg()
        text, conf = agg.get_consensus("system_prompt", agent_id=0)
        assert text == ""
        assert conf == 0.0

    def test_one_candidate(self):
        agg = make_agg()
        agg.add_candidate("system_prompt", 0, "You are a movie agent.")
        text, conf = agg.get_consensus("system_prompt", agent_id=0)
        assert text == "You are a movie agent."
        assert conf == 0.0

    def test_two_candidates_with_match(self):
        agg = make_agg(min_match_length=10)
        agg.add_candidate("sp", 1, "You are a movie recommendation agent for drama.")
        agg.add_candidate("sp", 1, "You are a movie recommendation agent for comedy.")
        text, conf = agg.get_consensus("sp", agent_id=1)
        assert text != ""
        # Both candidates agreed, so confidence should be 1.0
        assert conf == 1.0

    def test_two_candidates_no_match_returns_longest(self):
        agg = make_agg(min_match_length=20)
        short = "abc"
        long_cand = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        agg.add_candidate("sp", 2, short)
        agg.add_candidate("sp", 2, long_cand)
        text, conf = agg.get_consensus("sp", agent_id=2)
        assert text == long_cand
        assert conf == 0.0

    def test_three_candidates_selects_best(self):
        agg = make_agg(min_match_length=10)
        # c1 and c2 share a long common substring; c3 is disjoint
        common = "You are a movie recommendation agent"
        c1 = common + " specializing in drama."
        c2 = common + " focused on comedy."
        c3 = "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
        agg.add_candidate("sp", 3, c1)
        agg.add_candidate("sp", 3, c2)
        agg.add_candidate("sp", 3, c3)
        text, conf = agg.get_consensus("sp", agent_id=3)
        # c1 or c2 should win (they share the common substring)
        assert text in (c1, c2)
        # c1 and c2 agreed, c3 did not → 2/3 agreeing
        assert abs(conf - 2 / 3) < 1e-9

    def test_all_candidates_agree(self):
        agg = make_agg(min_match_length=10)
        base = "You are a helpful movie recommendation agent."
        c1 = base + " Version 1."
        c2 = base + " Version 2."
        c3 = base + " Version 3."
        agg.add_candidate("sp", 4, c1)
        agg.add_candidate("sp", 4, c2)
        agg.add_candidate("sp", 4, c3)
        text, conf = agg.get_consensus("sp", agent_id=4)
        assert text in (c1, c2, c3)
        assert conf == 1.0

    def test_no_candidates_agree_returns_longest(self):
        agg = make_agg(min_match_length=100)  # very high threshold
        c1 = "short"
        c2 = "also short"
        c3 = "this is the longest candidate string here"
        agg.add_candidate("sp", 5, c1)
        agg.add_candidate("sp", 5, c2)
        agg.add_candidate("sp", 5, c3)
        text, conf = agg.get_consensus("sp", agent_id=5)
        assert text == c3
        assert conf == 0.0

    def test_different_targets_are_independent(self):
        agg = make_agg()
        agg.add_candidate("system_prompt", 0, "Candidate for system_prompt.")
        agg.add_candidate("task_instructions", 0, "Candidate for task_instructions.")
        sp_text, _ = agg.get_consensus("system_prompt", 0)
        ti_text, _ = agg.get_consensus("task_instructions", 0)
        assert sp_text == "Candidate for system_prompt."
        assert ti_text == "Candidate for task_instructions."

    def test_different_agent_ids_are_independent(self):
        agg = make_agg()
        agg.add_candidate("sp", 0, "Agent 0 candidate.")
        agg.add_candidate("sp", 1, "Agent 1 candidate.")
        t0, _ = agg.get_consensus("sp", 0)
        t1, _ = agg.get_consensus("sp", 1)
        assert t0 == "Agent 0 candidate."
        assert t1 == "Agent 1 candidate."


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Strategy: printable text strings of reasonable length
_text_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=".,!?;:-_()'\"",
    ),
    min_size=0,
    max_size=200,
)

_nonempty_text_st = _text_st.filter(lambda s: len(s) > 0)

_min_match_st = st.integers(min_value=1, max_value=50)


@given(a=_text_st, b=_text_st)
@settings(max_examples=100)
def test_property4_find_matched_content_symmetry(a: str, b: str):
    """
    Property 4: find_matched_content symmetry
    For any two strings c_i and c_j, find_matched_content(c_i, c_j) should
    equal find_matched_content(c_j, c_i).

    Validates: Requirements 2.3
    """
    agg = ExtractionAggregator(min_match_length=5)
    assert agg.find_matched_content(a, b) == agg.find_matched_content(b, a), (
        f"Symmetry violated: f({a!r}, {b!r}) != f({b!r}, {a!r})"
    )


@given(a=_text_st, b=_text_st, min_len=_min_match_st)
@settings(max_examples=100)
def test_property5_find_matched_content_is_common_substring(a: str, b: str, min_len: int):
    """
    Property 5: find_matched_content is a common substring
    The result of find_matched_content(c_i, c_j) should be a substring of both
    c_i and c_j, and its length should be at most min(len(c_i), len(c_j)).

    Validates: Requirements 2.3
    """
    agg = ExtractionAggregator(min_match_length=min_len)
    result = agg.find_matched_content(a, b)

    if result:
        assert result in a, f"Result {result!r} not a substring of a={a!r}"
        assert result in b, f"Result {result!r} not a substring of b={b!r}"
        assert len(result) <= min(len(a), len(b)), (
            f"Result length {len(result)} > min({len(a)}, {len(b)})"
        )
        assert len(result) >= min_len, (
            f"Result length {len(result)} < min_match_length={min_len}"
        )


@given(
    candidates=st.lists(
        _nonempty_text_st,
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_property6_consensus_subsumes_candidates(candidates: list):
    """
    Property 6: Consensus subsumes candidates
    For any non-empty list of extraction candidates, the consensus text returned
    by get_consensus() should be a substring of at least one of the input
    candidates.

    Validates: Requirements 2.4, 2.5
    """
    agg = ExtractionAggregator(min_match_length=5)
    for i, text in enumerate(candidates):
        agg.add_candidate("target", agent_id=0, text=text)

    consensus, _ = agg.get_consensus("target", agent_id=0)

    # consensus must be a substring of at least one candidate
    assert any(consensus in c for c in candidates), (
        f"Consensus {consensus!r} is not a substring of any candidate.\n"
        f"Candidates: {candidates}"
    )
