"""BlindGuard unsupervised anomaly detection model.

Implements the BlindGuard architecture from arXiv:2508.08127:
  - Hierarchical Agent Encoder: self + neighbor + global features
  - Corruption-Guided Detector: directional noise + contrastive learning
  - Inference: negative average cosine similarity (no threshold needed)

Unlike G-Safeguard (supervised GAT), BlindGuard trains only on normal data
and generalizes to unseen attack types.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter

logger = logging.getLogger(__name__)


class HierarchicalAgentEncoder(nn.Module):
    """Three-level hierarchical encoder for agent representations.

    Fuses self, neighbor, and global features via MLP (Eq. 3-4 in paper).

    Args:
        input_dim: Dimension of SentenceBERT embeddings.
        hidden_dim: Hidden dimension of the MLP.
        output_dim: Output representation dimension.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        # MLP g_theta: concat(h_self, h_neigh, h_graph) -> z
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Compute hierarchical representations.

        Args:
            x: Node features [N, input_dim].
            edge_index: Edge indices [2, E] (source -> target).
            num_nodes: Total number of nodes N.

        Returns:
            z: Agent representations [N, output_dim].
        """
        # Self-level features
        h_self = x  # [N, D]

        # Neighbor-level: normalized adjacency aggregation
        # For each node, aggregate features from its neighbors.
        # Use both directions: col->row and row->col so that user nodes
        # (sources) and item nodes (targets) both receive neighbor signal.
        row, col = edge_index
        all_src = torch.cat([row, col])
        all_dst = torch.cat([col, row])
        deg = scatter(torch.ones(all_src.size(0), device=x.device), all_dst, dim=0,
                      dim_size=num_nodes, reduce='sum').clamp(min=1)
        h_neigh = scatter(x[all_src], all_dst, dim=0, dim_size=num_nodes, reduce='sum')
        h_neigh = h_neigh / deg.unsqueeze(-1)  # [N, D]

        # Global-level: mean of all node features
        h_graph = x.mean(dim=0, keepdim=True).expand(num_nodes, -1)  # [N, D]

        # Concatenate and transform
        h_cat = torch.cat([h_self, h_neigh, h_graph], dim=-1)  # [N, 3D]
        z = self.mlp(h_cat)  # [N, output_dim]
        return z


class BlindGuardModel(nn.Module):
    """BlindGuard unsupervised anomaly detection model.

    Combines hierarchical encoding with corruption-guided contrastive training.
    At inference, anomaly score = negative average cosine similarity to peers.

    Args:
        input_dim: SentenceBERT embedding dimension (default 384).
        hidden_dim: MLP hidden dimension (default 512).
        output_dim: Representation dimension (default 256).
        corruption_alpha: Noise scaling factor for synthetic anomalies (default 1.0).
        temperature: Contrastive loss temperature (default 0.07).
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 256,
        corruption_alpha: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = HierarchicalAgentEncoder(input_dim, hidden_dim, output_dim)
        self.corruption_alpha = corruption_alpha
        self.temperature = temperature

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        """Encode agents into representation space.

        Args:
            x: Node features [N, input_dim].
            edge_index: Edge indices [2, E].
            num_nodes: Total nodes (inferred from x if None).

        Returns:
            z: L2-normalized representations [N, output_dim].
        """
        if num_nodes is None:
            num_nodes = x.size(0)
        z = self.encoder(x, edge_index, num_nodes)
        return F.normalize(z, p=2, dim=-1)

    def corrupt(self, x: Tensor) -> Tensor:
        """Apply magnitude-scaled directional noise (Eq. 5 in paper).

        Args:
            x: Node features [N, D].

        Returns:
            x_tilde: Corrupted features [N, D].
        """
        eps = torch.randn_like(x)
        eps = F.normalize(eps, p=2, dim=-1)  # unit direction
        magnitude = self.corruption_alpha * x.norm(p=2, dim=-1, keepdim=True)
        return x + magnitude * eps

    def contrastive_loss(
        self, z: Tensor, labels: Tensor
    ) -> Tensor:
        """Supervised contrastive loss (Eq. 6 in paper).

        Args:
            z: L2-normalized representations [N, output_dim].
            labels: Binary labels [N] (0=normal, 1=corrupted).

        Returns:
            Scalar loss.
        """
        N = z.size(0)
        # Cosine similarity matrix [N, N]
        sim = torch.mm(z, z.t()) / self.temperature

        loss = torch.tensor(0.0, device=z.device)
        count = 0
        for i in range(N):
            pos_mask = (labels == labels[i]) & (torch.arange(N, device=z.device) != i)
            neg_mask = labels != labels[i]
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            # log(exp(s_ij) / (exp(s_ij) + sum_neg exp(s_ik)))
            pos_sims = sim[i][pos_mask]
            neg_sims = sim[i][neg_mask]
            for s_pos in pos_sims:
                denom = torch.exp(s_pos) + torch.exp(neg_sims).sum()
                loss = loss - torch.log(torch.exp(s_pos) / denom.clamp(min=1e-8))
                count += 1

        return loss / max(count, 1)

    def anomaly_scores(self, z: Tensor) -> Tensor:
        """Compute anomaly scores as negative average cosine similarity (Eq. 7).

        Args:
            z: L2-normalized representations [N, output_dim].

        Returns:
            scores: Anomaly scores [N] (higher = more anomalous).
        """
        N = z.size(0)
        # Cosine similarity matrix [N, N]
        sim_matrix = torch.mm(z, z.t())  # already normalized
        # Negative average similarity to all agents (including self)
        scores = -sim_matrix.mean(dim=1)
        return scores


def train_blindguard(
    model: BlindGuardModel,
    normal_graphs: list,
    n_epochs: int = 50,
    lr: float = 1e-3,
    corruption_fraction: float = 0.3,
    device: str = "cpu",
) -> BlindGuardModel:
    """Train BlindGuard on normal (unattacked) interaction graphs.

    Args:
        model: BlindGuardModel to train.
        normal_graphs: List of PyG Data objects from normal interactions.
        n_epochs: Training epochs.
        lr: Learning rate.
        corruption_fraction: Fraction of nodes to corrupt per graph.
        device: Torch device.

    Returns:
        Trained model.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for graph in normal_graphs:
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            N = x.size(0)

            # Corrupt a random subset
            n_corrupt = max(1, int(N * corruption_fraction))
            corrupt_idx = torch.randperm(N, device=device)[:n_corrupt]
            labels = torch.zeros(N, dtype=torch.long, device=device)
            labels[corrupt_idx] = 1

            # Build augmented features
            x_aug = x.clone()
            x_aug[corrupt_idx] = model.corrupt(x[corrupt_idx])

            # Forward pass
            z = model(x_aug, edge_index, N)

            # Contrastive loss
            loss = model.contrastive_loss(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.debug(f"[BlindGuard] Epoch {epoch+1}/{n_epochs}, loss={total_loss/max(len(normal_graphs),1):.4f}")

    model.eval()
    return model
