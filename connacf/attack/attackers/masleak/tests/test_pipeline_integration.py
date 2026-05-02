"""
Integration test: verify collect_response() passes current_candidates to NeighborExtractor.

Validates: Requirement 4.8
"""

from unittest.mock import patch, MagicMock
from connacf.attack.attackers.masleak.masleak_attacker import MASLeakAttacker


def test_collect_response_passes_current_candidates_to_neighbor_extractor():
    """
    Verify that collect_response() passes current_candidates to
    NeighborExtractor.extract_neighbors() (Requirement 4.8).
    """
    attacker = MASLeakAttacker(config={'use_v4': True})

    item_catalog = {1: 'Toy Story', 2: 'Fargo', 3: 'GoodFellas'}
    attacker.set_item_catalog(item_catalog)

    current_candidates = [1, 2]
    response = "I previously enjoyed Toy Story and Fargo."

    captured_calls = []

    original_extract = attacker._neighbor_extractor.extract_neighbors

    def capturing_extract(resp, uid, turn, catalog, current_candidates=None):
        captured_calls.append(current_candidates)
        return original_extract(resp, uid, turn, catalog, current_candidates=current_candidates)

    attacker._neighbor_extractor.extract_neighbors = capturing_extract

    attacker.collect_response(
        response=response,
        source_agent_id=42,
        agent_type='user',
        turn=1,
        user_id=10,
        current_candidates=current_candidates,
    )

    assert len(captured_calls) == 1, "extract_neighbors should have been called once"
    assert captured_calls[0] == current_candidates, (
        f"Expected current_candidates={current_candidates}, got {captured_calls[0]}"
    )


def test_collect_response_current_candidates_none_by_default():
    """
    Verify backward compatibility: when current_candidates is not passed,
    NeighborExtractor receives None (Requirement 4.8).
    """
    attacker = MASLeakAttacker(config={'use_v4': True})

    item_catalog = {1: 'Toy Story'}
    attacker.set_item_catalog(item_catalog)

    captured_calls = []

    original_extract = attacker._neighbor_extractor.extract_neighbors

    def capturing_extract(resp, uid, turn, catalog, current_candidates=None):
        captured_calls.append(current_candidates)
        return original_extract(resp, uid, turn, catalog, current_candidates=current_candidates)

    attacker._neighbor_extractor.extract_neighbors = capturing_extract

    attacker.collect_response(
        response="I liked Toy Story.",
        source_agent_id=1,
        agent_type='user',
        turn=0,
        user_id=5,
        # current_candidates not passed — should default to None
    )

    assert len(captured_calls) == 1
    assert captured_calls[0] is None, (
        f"Expected None when current_candidates not passed, got {captured_calls[0]}"
    )


def test_collect_response_filters_candidates_without_history_signal():
    """
    End-to-end: items in current_candidates without a history signal
    should NOT appear in the topology extractor (Requirement 4.2).
    """
    attacker = MASLeakAttacker(config={'use_v4': True})

    item_catalog = {1: 'Toy Story', 2: 'Fargo'}
    attacker.set_item_catalog(item_catalog)

    # Response mentions Toy Story but it's a current candidate and no history signal
    attacker.collect_response(
        response="You should watch Toy Story.",
        source_agent_id=1,
        agent_type='user',
        turn=0,
        user_id=7,
        current_candidates=[1],  # Toy Story is a current candidate
    )

    edges = attacker._topology_extractor.get_extracted_edges()
    # Toy Story (id=1) should be filtered out — no history signal
    assert (7, 1) not in edges, "Candidate item without history signal should be filtered"
