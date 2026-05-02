"""
Unit and property-based tests for NeighborExtractor.

Property tests use hypothesis with @settings(max_examples=100).

Properties covered:
  - Property 7: Candidate filtering and history-signal override (Requirements 4.2, 4.3, 4.4)
  - Property 8: Mention frequency tracking and high-confidence filtering (Requirements 4.5, 4.6)
"""

import pytest
from hypothesis import given, settings, assume, strategies as st

from connacf.attack.attackers.masleak.worm_queries import NeighborExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_extractor() -> NeighborExtractor:
    return NeighborExtractor()


def make_catalog(item_ids: list[int], names: list[str]) -> dict[int, str]:
    """Build an item_catalog from parallel lists of ids and names."""
    return dict(zip(item_ids, names))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestExtractNeighborsBasic:
    """Unit tests for extract_neighbors core behaviour."""

    def test_empty_catalog_returns_empty(self):
        ne = make_extractor()
        result = ne.extract_neighbors("some response", user_id=1, turn=0, item_catalog={})
        assert result == {}

    def test_none_catalog_returns_empty(self):
        ne = make_extractor()
        result = ne.extract_neighbors("some response", user_id=1, turn=0, item_catalog=None)
        assert result == {}

    def test_item_not_in_response_not_extracted(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors("I like action movies.", user_id=1, turn=0,
                                      item_catalog=catalog)
        assert 1 not in result.get('neighbor_ids', [])

    def test_item_in_response_no_candidates_extracted(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors("I really enjoyed Inception last week.", user_id=1,
                                      turn=0, item_catalog=catalog)
        assert 1 in result['neighbor_ids']

    def test_candidate_item_without_history_signal_excluded(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors("Would you recommend Inception?", user_id=1, turn=0,
                                      item_catalog=catalog, current_candidates=[1])
        assert 1 not in result.get('neighbor_ids', [])

    def test_candidate_item_with_history_signal_included(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors(
            "Previously you liked Inception, so I recommend it again.",
            user_id=1, turn=1, item_catalog=catalog, current_candidates=[1]
        )
        assert 1 in result['neighbor_ids']

    def test_history_context_detected_flag(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors(
            "Based on your history, Inception seems like a good fit.",
            user_id=1, turn=0, item_catalog=catalog
        )
        assert result['history_context_detected'] is True

    def test_no_history_context_flag(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        result = ne.extract_neighbors(
            "Would you like to watch Inception tonight?",
            user_id=1, turn=0, item_catalog=catalog
        )
        assert result['history_context_detected'] is False

    def test_non_candidate_item_always_included(self):
        ne = make_extractor()
        catalog = {1: "Inception", 2: "The Matrix"}
        # Only item 1 is a candidate; item 2 should always be included
        result = ne.extract_neighbors(
            "I think Inception and The Matrix are both great.",
            user_id=1, turn=0, item_catalog=catalog, current_candidates=[1]
        )
        assert 2 in result['neighbor_ids']

    def test_mention_counts_updated(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        ne.extract_neighbors("I enjoyed Inception.", user_id=1, turn=0, item_catalog=catalog)
        ne.extract_neighbors("Inception is a classic.", user_id=1, turn=1, item_catalog=catalog)
        assert ne.mention_counts[1][1] == 2

    def test_mention_counts_updated_even_for_filtered_candidates(self):
        """mention_counts should be updated even when item is filtered from output."""
        ne = make_extractor()
        catalog = {1: "Inception"}
        # Item 1 is a candidate with no history signal — filtered from output but count updated
        ne.extract_neighbors("Would you recommend Inception?", user_id=1, turn=0,
                              item_catalog=catalog, current_candidates=[1])
        assert ne.mention_counts[1][1] == 1

    def test_high_confidence_neighbors_empty_user(self):
        ne = make_extractor()
        assert ne.high_confidence_neighbors(user_id=99) == []

    def test_high_confidence_neighbors_below_threshold(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        ne.extract_neighbors("I enjoyed Inception.", user_id=1, turn=0, item_catalog=catalog)
        # Only 1 mention, min_mentions=2 → should not appear
        assert 1 not in ne.high_confidence_neighbors(user_id=1, min_mentions=2)

    def test_high_confidence_neighbors_at_threshold(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        ne.extract_neighbors("I enjoyed Inception.", user_id=1, turn=0, item_catalog=catalog)
        ne.extract_neighbors("Inception is great.", user_id=1, turn=1, item_catalog=catalog)
        assert 1 in ne.high_confidence_neighbors(user_id=1, min_mentions=2)

    def test_get_high_confidence_neighbors_alias(self):
        """Backward-compat alias should behave identically."""
        ne = make_extractor()
        catalog = {1: "Inception"}
        ne.extract_neighbors("I enjoyed Inception.", user_id=1, turn=0, item_catalog=catalog)
        ne.extract_neighbors("Inception is great.", user_id=1, turn=1, item_catalog=catalog)
        assert ne.get_high_confidence_neighbors(1, 2) == ne.high_confidence_neighbors(1, 2)

    def test_history_signals_class_attribute_is_list(self):
        assert isinstance(NeighborExtractor.HISTORY_SIGNALS, list)
        assert len(NeighborExtractor.HISTORY_SIGNALS) > 0

    def test_all_history_signals_trigger_detection(self):
        ne = make_extractor()
        catalog = {1: "Inception"}
        for signal in NeighborExtractor.HISTORY_SIGNALS:
            result = ne.extract_neighbors(
                f"{signal} Inception was mentioned.",
                user_id=1, turn=0, item_catalog=catalog
            )
            assert result['history_context_detected'] is True, (
                f"Signal '{signal}' did not trigger history_context_detected"
            )


# ---------------------------------------------------------------------------
# Strategies for property-based tests
# ---------------------------------------------------------------------------

# Item names: at least 4 chars, no year suffixes, no PAD
_item_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Zs")),
    min_size=4,
    max_size=30,
).filter(lambda s: s.strip() and s.strip() != '[PAD]' and '(' not in s)

# Small catalogs: 1–5 items
_catalog_st = st.lists(
    st.tuples(st.integers(min_value=1, max_value=100), _item_name_st),
    min_size=1,
    max_size=5,
).map(dict)

# A history signal phrase drawn from the class attribute
_history_signal_st = st.sampled_from(NeighborExtractor.HISTORY_SIGNALS)

# User IDs
_user_id_st = st.integers(min_value=0, max_value=50)

# Turn numbers
_turn_st = st.integers(min_value=0, max_value=20)


def build_response_with_items(catalog: dict[int, str], item_ids: list[int]) -> str:
    """Build a response string that mentions the given item names."""
    parts = []
    for iid in item_ids:
        if iid in catalog:
            parts.append(catalog[iid])
    return " ".join(parts) if parts else "nothing relevant"


# ---------------------------------------------------------------------------
# Property 7: Candidate filtering and history-signal override
# Validates: Requirements 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------

@given(
    catalog=_catalog_st,
    user_id=_user_id_st,
    turn=_turn_st,
)
@settings(max_examples=100)
def test_property7a_candidates_only_no_history_signal_gives_empty(
    catalog: dict, user_id: int, turn: int
):
    """
    Property 7 (part A): Candidate filtering and history-signal override

    For any response that mentions ONLY items from current_candidates with no
    history-context signal, extract_neighbors() should return an empty neighbor set.

    Validates: Requirements 4.2, 4.3, 4.4
    """
    ne = make_extractor()
    candidate_ids = list(catalog.keys())

    # Build a response that mentions only candidate item names, no history signal
    response = build_response_with_items(catalog, candidate_ids)

    # Ensure no history signal is accidentally present
    assume(not any(sig in response.lower() for sig in NeighborExtractor.HISTORY_SIGNALS))

    result = ne.extract_neighbors(
        response=response,
        user_id=user_id,
        turn=turn,
        item_catalog=catalog,
        current_candidates=candidate_ids,
    )

    assert result.get('neighbor_ids', []) == [], (
        f"Expected empty neighbor_ids but got {result['neighbor_ids']}.\n"
        f"Response: {response!r}\n"
        f"Candidates: {candidate_ids}"
    )


@given(
    catalog=_catalog_st,
    user_id=_user_id_st,
    turn=_turn_st,
    history_signal=_history_signal_st,
)
@settings(max_examples=100)
def test_property7b_history_signal_overrides_candidate_filter(
    catalog: dict, user_id: int, turn: int, history_signal: str
):
    """
    Property 7 (part B): Candidate filtering and history-signal override

    For any response that contains a history-context signal AND mentions items
    from current_candidates, the extractor should include those items in the
    extracted neighbor set.

    Validates: Requirements 4.2, 4.3, 4.4
    """
    ne = make_extractor()
    candidate_ids = list(catalog.keys())

    # Build a response with a history signal and candidate item names
    item_mentions = build_response_with_items(catalog, candidate_ids)
    response = f"{history_signal} {item_mentions}"

    result = ne.extract_neighbors(
        response=response,
        user_id=user_id,
        turn=turn,
        item_catalog=catalog,
        current_candidates=candidate_ids,
    )

    assert len(result.get('neighbor_ids', [])) > 0, (
        f"Expected non-empty neighbor_ids with history signal '{history_signal}' "
        f"but got empty.\nResponse: {response!r}\nCatalog: {catalog}"
    )


# ---------------------------------------------------------------------------
# Property 8: Mention frequency tracking and high-confidence filtering
# Validates: Requirements 4.5, 4.6
# ---------------------------------------------------------------------------

@given(
    catalog=_catalog_st,
    user_id=_user_id_st,
    num_mentions=st.integers(min_value=1, max_value=5),
    min_mentions=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100)
def test_property8_mention_frequency_and_high_confidence_filtering(
    catalog: dict, user_id: int, num_mentions: int, min_mentions: int
):
    """
    Property 8: Mention frequency tracking and high-confidence filtering

    For any user and item, if the item is mentioned in fewer than min_mentions
    rounds, high_confidence_neighbors(user_id, min_mentions) should NOT include
    that item. If mentioned in min_mentions or more rounds, it SHOULD be included.

    Validates: Requirements 4.5, 4.6
    """
    ne = make_extractor()

    # Pick one item from the catalog to track
    item_id = next(iter(catalog))
    item_name = catalog[item_id]

    # Build a response that mentions this item (not as a candidate, so it's always included)
    response = f"I really enjoyed {item_name} and would recommend it."

    # Ensure no history signal accidentally present
    assume(not any(sig in response.lower() for sig in NeighborExtractor.HISTORY_SIGNALS))

    # Call extract_neighbors num_mentions times
    for t in range(num_mentions):
        ne.extract_neighbors(
            response=response,
            user_id=user_id,
            turn=t,
            item_catalog=catalog,
            current_candidates=[],  # not a candidate, so always counted
        )

    hcn = ne.high_confidence_neighbors(user_id=user_id, min_mentions=min_mentions)

    if num_mentions >= min_mentions:
        assert item_id in hcn, (
            f"Item {item_id} mentioned {num_mentions} times (>= min_mentions={min_mentions}) "
            f"but not in high_confidence_neighbors: {hcn}"
        )
    else:
        assert item_id not in hcn, (
            f"Item {item_id} mentioned {num_mentions} times (< min_mentions={min_mentions}) "
            f"but found in high_confidence_neighbors: {hcn}"
        )
