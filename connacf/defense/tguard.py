"""T-Guard defense for TOMA attacks in ConnaCF.

Adapts T-Guard (TOMA paper §6) to ConnaCF's bipartite User↔Item graph.
The visual cross-modal validator is not applicable here; we implement:
  - Topology Trust Evaluator: taint propagation model (Algorithm 3 in paper)
  - Access Control Manager: restrict low-trust agents from memory updates

Usage in attack config:
    defense:
      enable_defense: true
      defense_method: "t-guard"
      t_guard:
        decay_factor: 0.05
        max_iterations: 100
        taint_threshold_quarantine: 0.8
        taint_threshold_restrict: 0.5
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TaintPropagationModel:
    """Taint propagation model from T-Guard (Algorithm 3 in TOMA paper).

    Computes taint values T[v] ∈ [0,1] for each node, then derives
    trust R[v] = 1 - T[v].

    Args:
        decay_factor: Per-hop attenuation (default 0.05, paper uses 0.05).
        max_iterations: Max propagation steps (default 100).
        epsilon: Convergence threshold (default 1e-4).
    """

    def __init__(
        self,
        decay_factor: float = 0.05,
        max_iterations: int = 100,
        epsilon: float = 1e-4,
    ):
        self.decay_factor = decay_factor
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def propagate(
        self,
        node_ids: List[int],
        edges: List[Tuple[int, int]],
        seed_nodes: Set[int],
    ) -> Dict[int, float]:
        """Run taint propagation and return trust scores.

        Args:
            node_ids: All node IDs in the graph.
            edges: List of (src, dst) directed edges.
            seed_nodes: Initially compromised nodes (T=1.0).

        Returns:
            trust: Mapping node_id -> trust score R[v] = 1 - T[v].
        """
        # Build adjacency: neighbors[v] = list of nodes pointing TO v
        neighbors: Dict[int, List[int]] = {n: [] for n in node_ids}
        for src, dst in edges:
            if dst in neighbors:
                neighbors[dst].append(src)

        # Initialize taint
        T = {n: (1.0 if n in seed_nodes else 0.0) for n in node_ids}

        for _ in range(self.max_iterations):
            T_prev = T.copy()
            for v in node_ids:
                if not neighbors[v]:
                    continue
                avg = sum(T_prev[u] for u in neighbors[v]) / len(neighbors[v])
                update = (1 - T_prev[v]) * avg * self.decay_factor
                T[v] = min(1.0, T_prev[v] + update)

            # Check convergence
            if max(abs(T[v] - T_prev[v]) for v in node_ids) < self.epsilon:
                break

        return {v: 1.0 - T[v] for v in node_ids}  # trust = 1 - taint


class TGuardDefense:
    """T-Guard defense adapted for ConnaCF.

    Computes trust scores via taint propagation, then restricts
    low-trust agents from participating in memory updates.

    Taint thresholds (from paper Table X):
      T > 0.8  → QUARANTINE (block all operations)
      0.5 < T ≤ 0.8 → RESTRICT (read-only / no backward update)
      T ≤ 0.5  → LOG (allow but monitor)

    Args:
        decay_factor: Taint propagation decay.
        taint_threshold_quarantine: Taint above this → quarantine.
        taint_threshold_restrict: Taint above this → restrict.
    """

    def __init__(
        self,
        decay_factor: float = 0.05,
        taint_threshold_quarantine: float = 0.8,
        taint_threshold_restrict: float = 0.5,
        max_iterations: int = 100,
    ):
        self.model = TaintPropagationModel(decay_factor, max_iterations)
        self.tq = taint_threshold_quarantine
        self.tr = taint_threshold_restrict
        self._trust_scores: Dict[int, float] = {}
        self._quarantined: Set[int] = set()
        self._restricted: Set[int] = set()

    def update(
        self,
        node_ids: List[int],
        edges: List[Tuple[int, int]],
        suspected_attackers: Set[int],
    ) -> None:
        """Recompute trust scores given current suspected attackers.

        Args:
            node_ids: All agent node IDs.
            edges: Interaction edges (src, dst).
            suspected_attackers: Known or suspected compromised nodes.
        """
        self._trust_scores = self.model.propagate(node_ids, edges, suspected_attackers)

        self._quarantined = {
            v for v, trust in self._trust_scores.items()
            if (1 - trust) > self.tq  # taint = 1 - trust
        }
        self._restricted = {
            v for v, trust in self._trust_scores.items()
            if self.tr < (1 - trust) <= self.tq
        } - self._quarantined

        logger.info(
            "[T-Guard] Trust update: %d quarantined, %d restricted out of %d agents",
            len(self._quarantined), len(self._restricted), len(node_ids),
        )

    def is_quarantined(self, node_id: int) -> bool:
        return node_id in self._quarantined

    def is_restricted(self, node_id: int) -> bool:
        return node_id in self._restricted

    def should_block_backward(self, node_id: int) -> bool:
        """Return True if this agent should be blocked from backward memory update."""
        return node_id in self._quarantined or node_id in self._restricted

    def get_trust(self, node_id: int) -> float:
        return self._trust_scores.get(node_id, 1.0)

    def get_stats(self) -> Dict:
        return {
            "quarantined": sorted(self._quarantined),
            "restricted": sorted(self._restricted),
            "num_quarantined": len(self._quarantined),
            "num_restricted": len(self._restricted),
        }
