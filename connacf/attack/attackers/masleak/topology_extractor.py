"""
TopologyExtractor: builds the extracted Bipartite U-I Graph from per-user response collections.

Implements Requirements 3.1, 3.2, 3.5, 3.6.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class BipartiteGraph:
    """Represents a bipartite user-item interaction graph."""

    user_nodes: Set[int]
    item_nodes: Set[int]
    edges: Set[Tuple[int, int]]  # (user_id, item_id)

    def precision_recall_vs(self, ground_truth: "BipartiteGraph") -> Dict[str, float]:
        """Compute precision, recall, and GS (F1) against a ground-truth graph."""
        tp = len(self.edges & ground_truth.edges)
        fp = len(self.edges - ground_truth.edges)
        fn = len(ground_truth.edges - self.edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        gs = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"gs": gs, "precision": precision, "recall": recall}


class TopologyExtractor:
    """
    Extracts the bipartite U-I graph G = (U ∪ I, E) from agent responses.

    Edges (user_id, item_id) are recorded whenever a user agent mentions an item.
    Turn information is stored for debugging but does not affect the edge set.
    """

    def __init__(self) -> None:
        # edges: set of (user_id, item_id) pairs
        self._edges: Set[Tuple[int, int]] = set()
        # adjacency: user_id -> set of item_ids
        self._adjacency: Dict[int, Set[int]] = defaultdict(set)
        # turn log: (user_id, item_id) -> list of turns (for debugging)
        self._turn_log: Dict[Tuple[int, int], list] = defaultdict(list)

    def record_mention(self, user_id: int, item_id: int, turn: int) -> None:
        """Record that user_id mentioned item_id in their response at the given turn."""
        edge = (user_id, item_id)
        self._edges.add(edge)
        self._adjacency[user_id].add(item_id)
        self._turn_log[edge].append(turn)

    def get_extracted_edges(self) -> Set[Tuple[int, int]]:
        """Return all (user_id, item_id) pairs recorded."""
        return set(self._edges)

    def get_bipartite_adjacency(self) -> Dict[int, Set[int]]:
        """Return {user_id: {item_id, ...}} adjacency dict."""
        return {uid: set(items) for uid, items in self._adjacency.items()}

    def to_bipartite_graph(self) -> BipartiteGraph:
        """Return a BipartiteGraph built from all recorded mentions."""
        edges = self.get_extracted_edges()
        user_nodes: Set[int] = set()
        item_nodes: Set[int] = set()
        for user_id, item_id in edges:
            user_nodes.add(user_id)
            item_nodes.add(item_id)
        return BipartiteGraph(
            user_nodes=user_nodes,
            item_nodes=item_nodes,
            edges=edges,
        )
