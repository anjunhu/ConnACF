"""Training data generator for G-safeguard / BlindGuard defense.

Collects labeled graph snapshots from ConnaCF attack runs and serializes
them as a pickle file compatible with :class:`AgentGraphDataset`.

Each graph instance is stored as a dict with keys:
    - ``features``: ``np.ndarray`` of shape ``(num_nodes, embedding_dim)``
    - ``labels``: ``np.ndarray`` of shape ``(num_nodes,)`` — 0=benign, 1=attacker
    - ``edge_index``: ``np.ndarray`` of shape ``(2, num_edges)``
    - ``edge_attr``: ``np.ndarray`` of shape ``(num_edges, num_temporal_windows, embedding_dim)``
    - ``adj_matrix``: ``np.ndarray`` of shape ``(num_nodes, num_nodes)``
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Set

import numpy as np

from connacf.defense.graph_constructor import GraphConstructor

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Collects labeled graph data from ConnaCF attack runs.

    Args:
        graph_constructor: A :class:`GraphConstructor` instance used to
            build PyTorch Geometric ``Data`` objects from interaction batches.
    """

    def __init__(self, graph_constructor: GraphConstructor) -> None:
        self.graph_constructor = graph_constructor
        self._collected: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_collected(self) -> int:
        """Number of graph snapshots collected so far."""
        return len(self._collected)

    def collect_graph(
        self,
        batch_user: List[int],
        batch_items: List[int],
        attacker_indices: Set[int],
        round_idx: int,
    ) -> None:
        """Collect one graph snapshot with attacker labels.

        Uses :meth:`GraphConstructor.build_graph` to create the graph, then
        extracts numpy arrays in the format expected by
        :class:`AgentGraphDataset`.

        Args:
            batch_user: List of user agent indices for the current batch.
            batch_items: List of item agent indices for the current batch
                (same length as *batch_user*).
            attacker_indices: Set of *global* agent indices that are known
                attackers.  User agents occupy indices ``0..n_users-1`` and
                item agents occupy ``n_users..n_users+n_items-1``.
            round_idx: Current training round (used for logging).
        """
        n_total = (
            self.graph_constructor.n_users + self.graph_constructor.n_items
        )

        # Build attacker_labels dict expected by GraphConstructor.
        attacker_labels: Dict[int, int] = {}
        for idx in range(n_total):
            attacker_labels[idx] = 1 if idx in attacker_indices else 0

        data = self.graph_constructor.build_graph(
            batch_user, batch_items, attacker_labels=attacker_labels
        )

        # Convert PyG Data tensors to numpy arrays.
        features = data.x.numpy()  # (num_nodes, embedding_dim)
        labels = data.y.numpy()  # (num_nodes,)
        edge_index = data.edge_index.numpy()  # (2, num_edges)
        edge_attr = data.edge_attr.numpy()  # (num_edges, tw, embedding_dim)

        # Build adjacency matrix from edge_index.
        adj_matrix = self._build_adj_matrix(edge_index, n_total)

        graph_dict: Dict[str, np.ndarray] = {
            "features": features,
            "labels": labels,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "adj_matrix": adj_matrix,
        }

        self._collected.append(graph_dict)
        logger.info(
            "Collected graph snapshot %d at round %d: %d nodes, %d edges.",
            self.num_collected,
            round_idx,
            features.shape[0],
            edge_index.shape[1],
        )

    def save_dataset(self, output_path: str) -> None:
        """Serialize collected graphs as a pickle file.

        The output file is a list of dicts, each containing ``features``,
        ``labels``, ``edge_index``, ``edge_attr``, and ``adj_matrix`` as
        numpy arrays — the format consumed by :class:`AgentGraphDataset`.

        Args:
            output_path: File path for the output pickle file.

        Raises:
            ValueError: If no graphs have been collected.
        """
        if not self._collected:
            raise ValueError(
                "No graphs collected. Call collect_graph() before saving."
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self._collected, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Saved %d graph snapshots to %s.", self.num_collected, output_path
        )

    def reset(self) -> None:
        """Clear all collected graph snapshots."""
        self._collected.clear()
        logger.debug("Training data generator reset.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_adj_matrix(
        edge_index: np.ndarray, n_total: int
    ) -> np.ndarray:
        """Build a dense adjacency matrix from an edge index array.

        Args:
            edge_index: ``(2, num_edges)`` array of source/target indices.
            n_total: Total number of nodes.

        Returns:
            ``(n_total, n_total)`` binary adjacency matrix (float32).
        """
        adj = np.zeros((n_total, n_total), dtype=np.float32)
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # undirected
        return adj
