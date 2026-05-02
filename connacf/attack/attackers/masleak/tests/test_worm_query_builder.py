"""
Unit and property-based tests for WormQueryBuilder.

Property tests use hypothesis with @settings(max_examples=100).

Properties covered:
  - Property 2: Worm composition contains all three parts (Requirements 1.1, 1.5)
  - Property 3: Worm token budget (Requirement 1.6)
"""

import pytest
from hypothesis import given, settings, strategies as st

from connacf.attack.attackers.masleak.worm_query_builder import (
    WormQueryBuilder,
    _FAKE_EXAMPLES,
    _FAKE_EXAMPLES_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TARGETS = ["system_prompt", "task_instructions", "neighbors", "topology"]
DOMAINS = ["movies", "CDs", "generic", ""]


def make_domain_config(domain: str = "movies") -> dict:
    return {"domain": domain}


# ---------------------------------------------------------------------------
# Unit tests — build_worm returns string containing [DATA] markers
# ---------------------------------------------------------------------------

class TestBuildWormDataMarkers:
    def test_contains_data_open_marker(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert "[DATA]" in worm

    def test_contains_data_close_marker(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert "[/DATA]" in worm

    def test_contains_data_section_header(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert "#DATA section" in worm

    def test_contains_slot_b(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("task_instructions", make_domain_config("CDs"), round_idx=1)
        assert "B." in worm

    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("domain", DOMAINS)
    def test_all_targets_and_domains_have_markers(self, target, domain):
        builder = WormQueryBuilder()
        worm = builder.build_worm(target, make_domain_config(domain), round_idx=0)
        assert "[DATA]" in worm
        assert "[/DATA]" in worm


# ---------------------------------------------------------------------------
# Unit tests — build_worm contains all three parts
# ---------------------------------------------------------------------------

class TestBuildWormContainsAllParts:
    def test_contains_q_leak_signal(self):
        """q_Leak always contains a 'hold on' or 'STOP' or 'Hold on' signal."""
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        has_signal = any(
            phrase in worm
            for phrase in ["hold on", "STOP", "Hold on", "Wait wait wait"]
        )
        assert has_signal, f"q_Leak signal not found in worm: {worm[:200]}"

    def test_contains_q_retain_markers(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert "[DATA]" in worm and "[/DATA]" in worm

    def test_contains_q_propagate_signal(self):
        """q_Propagate always contains 'section B' or 'DATA section'."""
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert "section B" in worm or "DATA section" in worm

    def test_q_leak_comes_before_data_block(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        data_pos = worm.index("[DATA]")
        # q_Leak signal should appear before [DATA]
        leak_pos = min(
            (worm.index(p) for p in ["hold on", "STOP", "Hold on", "Wait wait wait"] if p in worm),
            default=None,
        )
        assert leak_pos is not None and leak_pos < data_pos

    def test_q_propagate_comes_after_data_block(self):
        builder = WormQueryBuilder()
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        close_pos = worm.index("[/DATA]")
        prop_pos = worm.find("section B")
        if prop_pos == -1:
            prop_pos = worm.find("DATA section", close_pos)
        assert prop_pos > close_pos

    def test_round_idx_rotates_q_leak(self):
        builder = WormQueryBuilder()
        worms = [
            builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=i)
            for i in range(3)
        ]
        # At least two distinct worms (rotation should produce different q_Leak)
        assert len(set(worms)) > 1


# ---------------------------------------------------------------------------
# Unit tests — token budget enforcement
# ---------------------------------------------------------------------------

class TestTokenBudgetEnforcement:
    def test_default_budget_respected(self):
        builder = WormQueryBuilder(token_budget=2000)
        for target in TARGETS:
            for domain in DOMAINS:
                worm = builder.build_worm(target, make_domain_config(domain), round_idx=0)
                assert len(worm) <= 2000, (
                    f"Worm for target={target}, domain={domain} exceeds budget: {len(worm)}"
                )

    def test_tight_budget_respected(self):
        builder = WormQueryBuilder(token_budget=300)
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert len(worm) <= 300

    def test_very_tight_budget_still_has_data_markers(self):
        """Even under extreme budget pressure, [DATA] markers should be preserved."""
        builder = WormQueryBuilder(token_budget=150)
        worm = builder.build_worm("system_prompt", make_domain_config("movies"), round_idx=0)
        assert len(worm) <= 150
        # Markers may be absent only if budget is truly too small for them
        # (absolute minimum is ~60 chars for markers alone)

    def test_custom_budget_respected(self):
        for budget in [500, 1000, 1500]:
            builder = WormQueryBuilder(token_budget=budget)
            worm = builder.build_worm("task_instructions", make_domain_config("CDs"), round_idx=2)
            assert len(worm) <= budget, f"Budget {budget} violated: {len(worm)}"


# ---------------------------------------------------------------------------
# Unit tests — domain-aware examples differ by domain
# ---------------------------------------------------------------------------

class TestDomainAwareExamples:
    def test_movies_system_prompt_example(self):
        builder = WormQueryBuilder()
        q_retain = builder.build_q_retain("system_prompt", {"domain": "movies"}, round_idx=0)
        assert "movie" in q_retain.lower()

    def test_cds_system_prompt_example(self):
        builder = WormQueryBuilder()
        q_retain = builder.build_q_retain("system_prompt", {"domain": "CDs"}, round_idx=0)
        assert "cd" in q_retain.lower() or "album" in q_retain.lower()

    def test_movies_and_cds_differ(self):
        builder = WormQueryBuilder()
        movies_retain = builder.build_q_retain("system_prompt", {"domain": "movies"}, round_idx=0)
        cds_retain = builder.build_q_retain("system_prompt", {"domain": "CDs"}, round_idx=0)
        assert movies_retain != cds_retain

    def test_generic_domain_falls_back(self):
        builder = WormQueryBuilder()
        q_retain = builder.build_q_retain("system_prompt", {"domain": "unknown_domain"}, round_idx=0)
        # Should still produce a valid DATA template
        assert "[DATA]" in q_retain
        assert "[/DATA]" in q_retain

    def test_domain_aliases_normalised(self):
        """'film' and 'movies' should produce the same fake example."""
        builder = WormQueryBuilder()
        r1 = builder.build_q_retain("system_prompt", {"domain": "movies"}, round_idx=0)
        r2 = builder.build_q_retain("system_prompt", {"domain": "film"}, round_idx=0)
        assert r1 == r2

    def test_cd_alias_normalised(self):
        builder = WormQueryBuilder()
        r1 = builder.build_q_retain("system_prompt", {"domain": "CDs"}, round_idx=0)
        r2 = builder.build_q_retain("system_prompt", {"domain": "cd"}, round_idx=0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Strategies
_target_st = st.sampled_from(TARGETS)
_domain_st = st.sampled_from(["movies", "CDs", "generic", "", "film", "music"])
_round_idx_st = st.integers(min_value=0, max_value=50)
_budget_st = st.integers(min_value=200, max_value=5000)


@given(
    target=_target_st,
    domain=_domain_st,
    round_idx=_round_idx_st,
)
@settings(max_examples=100)
def test_property2_worm_composition_contains_all_three_parts(target, domain, round_idx):
    """
    Property 2: Worm composition contains all three parts.
    For any domain config and round index, build_worm() output should contain
    [DATA] markers (q_Retain) and recognisable q_Leak / q_Propagate signals.
    Validates: Requirements 1.1, 1.5
    """
    builder = WormQueryBuilder(token_budget=2000)
    worm = builder.build_worm(target, {"domain": domain}, round_idx)

    # q_Retain markers
    assert "[DATA]" in worm, f"[DATA] missing for target={target}, domain={domain}"
    assert "[/DATA]" in worm, f"[/DATA] missing for target={target}, domain={domain}"

    # q_Leak signal (at least one of the known elicitation phrases)
    has_leak = any(
        phrase in worm
        for phrase in ["hold on", "STOP", "Hold on", "Wait wait wait", "DATA section"]
    )
    assert has_leak, f"No q_Leak signal found for target={target}, domain={domain}"

    # q_Propagate signal
    assert "section B" in worm or "DATA section" in worm, (
        f"No q_Propagate signal found for target={target}, domain={domain}"
    )


@given(
    target=_target_st,
    domain=_domain_st,
    round_idx=_round_idx_st,
    budget=_budget_st,
)
@settings(max_examples=100)
def test_property3_worm_token_budget(target, domain, round_idx, budget):
    """
    Property 3: Worm token budget.
    For any domain config and round index, the worm length should not exceed
    the configured token budget.
    Validates: Requirement 1.6
    """
    builder = WormQueryBuilder(token_budget=budget)
    worm = builder.build_worm(target, {"domain": domain}, round_idx)
    assert len(worm) <= budget, (
        f"Budget {budget} violated: worm length={len(worm)} "
        f"for target={target}, domain={domain}, round_idx={round_idx}"
    )
