"""Graph constructor for G-safeguard / BlindGuard defense.

Builds PyTorch Geometric ``Data`` objects from ConnaCF user-item interactions.
The resulting bipartite graph has user nodes (indices ``0..n_users-1``) and
item nodes (indices ``n_users..n_users+n_items-1``), with edges representing
training-batch interactions and edge attributes drawn from the temporal
embedding buffer maintained by :class:`EmbeddingExtractor`.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from connacf.defense.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Builds PyTorch Geometric Data objects from ConnaCF interactions.

    Args:
        embedding_extractor: An :class:`EmbeddingExtractor` instance that
            holds the temporal embedding buffer for all agents.
        n_users: Total number of user agents in the system.
        n_items: Total number of item agents in the system.
    """

    def __init__(
        self,
        embedding_extractor: EmbeddingExtractor,
        n_users: int,
        n_items: int,
    ):
        self.embedding_extractor = embedding_extractor
        self.n_users = n_users
        self.n_items = n_items

    def build_graph(
        self,
        batch_user: List[int],
        batch_items: List[int],
        attacker_labels: Optional[Dict[int, int]] = None,
    ) -> Data:
        """Construct an interaction graph for the current batch.

        Nodes comprise all user and item agents in the system (bipartite
        layout). Edges connect each user in ``batch_user`` to the
        corresponding item in ``batch_items``.

        Args:
            batch_user: List of user agent indices (one per interaction).
            batch_items: List of item agent indices (one per interaction,
                same length as ``batch_user``).
            attacker_labels: Optional mapping from *global* agent index to
                label (1 = attacker, 0 = benign). When provided, the ``y``
                field of the returned ``Data`` object is populated.

        Returns:
            A PyTorch Geometric ``Data`` object with fields ``x``,
            ``edge_index``, ``edge_attr``, and optionally ``y``.
        """
        n_total = self.n_users + self.n_items

        # --- edges --------------------------------------------------------
        src_indices: List[int] = []
        dst_indices: List[int] = []
        for uid, iid in zip(batch_user, batch_items):
            src_indices.append(uid)
            # Item nodes are offset by n_users in the bipartite layout.
            dst_indices.append(self.n_users + iid)

        edge_index = torch.tensor(
            [src_indices + dst_indices, dst_indices + src_indices], dtype=torch.long
        )  # (2, 2*num_edges) — undirected so both user and item nodes receive messages

        # --- edge attributes (duplicated for reverse edges) ---------------
        edge_attr = self._build_edge_attr(batch_user, batch_items)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # --- node features via scatter_mean --------------------------------
        x = self._compute_node_features(edge_index, edge_attr, n_total)

        # --- labels --------------------------------------------------------
        y = self._build_labels(n_total, attacker_labels)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if y is not None:
            data.y = y
        return data

    def build_full_graph(
        self,
        all_interactions: List[Tuple[int, int]],
        attacker_labels: Optional[Dict[int, int]] = None,
    ) -> Data:
        """Build a graph from *all* recorded interactions.

        This is used by the training-data generator to create a single
        graph snapshot covering the full interaction history.

        Args:
            all_interactions: List of ``(user_id, item_id)`` tuples.
            attacker_labels: Optional mapping from global agent index to
                label (1 = attacker, 0 = benign).

        Returns:
            A PyTorch Geometric ``Data`` object.
        """
        users = [u for u, _ in all_interactions]
        items = [i for _, i in all_interactions]
        return self.build_graph(users, items, attacker_labels)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_edge_attr(
        self, batch_user: List[int], batch_items: List[int]
    ) -> torch.Tensor:
        """Assemble temporal edge attributes from the embedding extractor.

        For each (user, item) interaction edge, the edge attribute is the
        temporal embedding of the *user* (source) agent. Using the source
        embedding preserves the full attacker signal rather than diluting it
        by averaging with the item embedding.

        Returns:
            Tensor of shape ``(num_edges, max_temporal_windows, embedding_dim)``.
        """
        tw = self.embedding_extractor.max_temporal_windows
        dim = self.embedding_extractor.embedding_dim
        num_edges = len(batch_user)

        edge_attr_np = np.zeros((num_edges, tw, dim), dtype=np.float32)
        for idx, (uid, iid) in enumerate(zip(batch_user, batch_items)):
            # Use the source (user) embedding as the edge attribute so that
            # attacker signal is not diluted by the item embedding.
            edge_attr_np[idx] = self.embedding_extractor.get_temporal_embeddings(uid)

        return torch.tensor(edge_attr_np, dtype=torch.float32)

    def _compute_node_features(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        n_total: int,
    ) -> torch.Tensor:
        """Compute node features as scatter_mean of incoming edge attributes.

        Following the reference code pattern, node features are computed by
        taking the first temporal window (index 0) of each incoming edge's
        attribute and aggregating via ``scatter_mean`` over the target node
        dimension.

        Args:
            edge_index: ``(2, num_edges)`` tensor.
            edge_attr: ``(num_edges, max_temporal_windows, embedding_dim)`` tensor.
            n_total: Total number of nodes in the graph.

        Returns:
            Node feature tensor of shape ``(n_total, embedding_dim)``.
        """
        if edge_attr.shape[0] == 0:
            dim = self.embedding_extractor.embedding_dim
            return torch.zeros((n_total, dim), dtype=torch.float32)

        # First temporal window of each edge attribute.
        first_window = edge_attr[:, 0, :]  # (num_edges, embedding_dim)
        target_nodes = edge_index[1]  # (num_edges,)

        x = scatter(
            first_window, target_nodes, dim=0, dim_size=n_total, reduce="mean"
        )
        return x

    def _build_labels(
        self,
        n_total: int,
        attacker_labels: Optional[Dict[int, int]],
    ) -> Optional[torch.Tensor]:
        """Build a label tensor from the attacker_labels mapping.

        Args:
            n_total: Total number of nodes.
            attacker_labels: Mapping from global agent index to label.
                User agents use indices ``0..n_users-1``; item agents use
                indices ``n_users..n_users+n_items-1``.

        Returns:
            A ``(n_total,)`` float tensor, or ``None`` if no labels given.
        """
        if attacker_labels is None:
            return None

        y = torch.zeros(n_total, dtype=torch.float32)
        for agent_idx, label in attacker_labels.items():
            if 0 <= agent_idx < n_total:
                y[agent_idx] = float(label)
            else:
                logger.warning(
                    "Attacker label index %d out of range [0, %d); skipping.",
                    agent_idx,
                    n_total,
                )
        return y
