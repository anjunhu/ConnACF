"""
ExtractionAggregator — implements Algorithm 2 from the MASLeak paper.

Collects multiple extraction attempts per (target, agent_id) and applies
pairwise find_matched_content() comparison to select the most consistent
candidate as the consensus result.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Tuple


class ExtractionAggregator:
    """
    Implements Algorithm 2: pairwise comparison across extraction candidates.

    Usage::

        agg = ExtractionAggregator(min_match_length=20)
        agg.add_candidate("system_prompt", agent_id=0, text="You are a movie agent...")
        agg.add_candidate("system_prompt", agent_id=0, text="You are a movie recommendation agent...")
        agg.add_candidate("system_prompt", agent_id=0, text="You are a movie enthusiast agent...")
        consensus, confidence = agg.get_consensus("system_prompt", agent_id=0)
    """

    def __init__(self, min_match_length: int = 20) -> None:
        self.min_match_length = min_match_length
        # key: (target, agent_id) -> list of extraction attempt strings
        self.candidates: Dict[Tuple[str, int], List[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_candidate(self, target: str, agent_id: int, text: str) -> None:
        """Record one extraction attempt for (target, agent_id)."""
        key = (target, agent_id)
        if key not in self.candidates:
            self.candidates[key] = []
        self.candidates[key].append(text)

    def find_matched_content(self, c_i: str, c_j: str) -> str:
        """
        Return the longest common substring of c_i and c_j that exceeds
        self.min_match_length. Returns '' if no such substring exists.
        """
        if not c_i or not c_j:
            return ""

        matcher = SequenceMatcher(None, c_i, c_j, autojunk=False)
        best_match = matcher.find_longest_match(0, len(c_i), 0, len(c_j))

        if best_match.size >= self.min_match_length:
            return c_i[best_match.a : best_match.a + best_match.size]
        return ""

    def get_consensus(self, target: str, agent_id: int) -> Tuple[str, float]:
        """
        Apply Algorithm 2 pairwise comparison across all candidates for
        (target, agent_id).

        Returns:
            (consensus_text, confidence) where confidence is the fraction of
            candidates whose score > 0 (i.e., agreed with at least one other).

        Edge cases:
            0 candidates → ('', 0.0)
            1 candidate  → (candidate, 0.0)
            2+ candidates → run Algorithm 2
        """
        key = (target, agent_id)
        cands = self.candidates.get(key, [])

        if len(cands) == 0:
            return ("", 0.0)

        if len(cands) == 1:
            return (cands[0], 0.0)

        # Algorithm 2
        scores: Dict[int, int] = {i: 0 for i in range(len(cands))}

        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                match = self.find_matched_content(cands[i], cands[j])
                if match:
                    scores[i] += len(match)
                    scores[j] += len(match)

        # Select candidate with highest total score
        best_idx = max(scores, key=lambda idx: scores[idx])

        # If no pair agreed at all, fall back to the longest candidate
        if scores[best_idx] == 0:
            best_idx = max(range(len(cands)), key=lambda idx: len(cands[idx]))
            return (cands[best_idx], 0.0)

        consensus_text = cands[best_idx]

        # Confidence = fraction of candidates that agreed with at least one other
        num_agreeing = sum(1 for s in scores.values() if s > 0)
        confidence = num_agreeing / len(cands)

        return (consensus_text, confidence)
