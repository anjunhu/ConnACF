"""Defense detector for G-safeguard and BlindGuard anomaly detection.

Runs GNN inference on interaction graphs and classifies agents as attackers.

G-safeguard: sigmoid(scores) >= threshold -> attacker (supervised GAT)
BlindGuard: top-k by negative avg cosine similarity -> attackers (unsupervised hierarchical encoder)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import torch
from torch_geometric.data import Data

from connacf.defense.gnn_model import MyGAT
from connacf.defense.blindguard_model import BlindGuardModel

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of a single detection round.

    Attributes:
        detected_attacker_indices: Indices of agents classified as attackers.
        all_scores: Mapping of agent index to anomaly score.
        defense_mode: Detection strategy used ('g-safeguard' or 'blindguard').
        round_idx: Training round when detection was performed.
        timestamp: ISO-format timestamp of detection.
        precision: Detection precision if ground truth is available.
        recall: Detection recall if ground truth is available.
        f1: Detection F1 score if ground truth is available.
    """

    detected_attacker_indices: List[int]
    all_scores: Dict[int, float]
    defense_mode: str
    round_idx: int
    timestamp: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


class DefenseDetector:
    """Runs GNN inference and classifies agents as attackers.

    Supports two detection modes:
    - 'g-safeguard': Supervised GAT + threshold-based classification.
    - 'blindguard': Unsupervised hierarchical encoder + top-k by negative
      average cosine similarity (no threshold, no attack labels needed).

    Args:
        gnn_model: Trained model — MyGAT for g-safeguard, BlindGuardModel for blindguard.
        defense_mode: Detection strategy, 'g-safeguard' or 'blindguard'.
        threshold: Classification threshold for g-safeguard mode.
        top_k: Number of top agents to flag in blindguard mode.
        device: Torch device for inference ('cpu' or 'cuda:X').
    """

    VALID_MODES = ("g-safeguard", "blindguard")

    def __init__(
        self,
        gnn_model: Union[MyGAT, BlindGuardModel],
        defense_mode: str = "g-safeguard",
        threshold: float = 0.5,
        top_k: int = 3,
        device: str = "cpu",
    ):
        if defense_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid defense_mode '{defense_mode}'. "
                f"Must be one of {self.VALID_MODES}."
            )
        self.gnn_model = gnn_model
        self.defense_mode = defense_mode
        self.threshold = threshold
        self.top_k = top_k
        self.device = device

        self.gnn_model.to(self.device)
        self.gnn_model.eval()

    def detect(self, graph_data: Data, round_idx: int = 0) -> DetectionResult:
        """Run model on graph and return detection results.

        Args:
            graph_data: PyTorch Geometric Data object with x, edge_index,
                and optionally edge_attr fields.
            round_idx: Current training round index.

        Returns:
            DetectionResult with detected attacker indices and scores.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if graph_data.edge_index is None or graph_data.edge_index.numel() == 0:
            logger.warning("Empty graph at round %d. Skipping detection.", round_idx)
            return DetectionResult(
                detected_attacker_indices=[],
                all_scores={},
                defense_mode=self.defense_mode,
                round_idx=round_idx,
                timestamp=timestamp,
            )

        graph_data = graph_data.to(self.device)
        num_nodes = graph_data.x.size(0)

        with torch.no_grad():
            if self.defense_mode == "blindguard":
                scores = self._score_blindguard(graph_data)
            else:
                scores = self._score_gsafeguard(graph_data)

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.error("NaN/Inf in scores at round %d. Skipping.", round_idx)
            return DetectionResult(
                detected_attacker_indices=[],
                all_scores={i: float("nan") for i in range(num_nodes)},
                defense_mode=self.defense_mode,
                round_idx=round_idx,
                timestamp=timestamp,
            )

        all_scores = {i: scores[i].item() for i in range(num_nodes)}

        if self.defense_mode == "g-safeguard":
            detected = [i for i, s in all_scores.items() if s >= self.threshold]
        else:
            # BlindGuard: top-k by anomaly score (higher = more anomalous)
            k = min(self.top_k, num_nodes)
            _, top_idx = torch.topk(scores, k)
            detected = top_idx.tolist()

        logger.info(
            "Detection round %d (%s): flagged %d/%d agents.",
            round_idx, self.defense_mode, len(detected), num_nodes,
        )

        return DetectionResult(
            detected_attacker_indices=detected,
            all_scores=all_scores,
            defense_mode=self.defense_mode,
            round_idx=round_idx,
            timestamp=timestamp,
        )

    def _score_gsafeguard(self, graph_data: Data) -> torch.Tensor:
        """G-Safeguard: sigmoid of supervised GAT logits."""
        raw_logits = self.gnn_model(
            graph_data.x, graph_data.edge_index, graph_data.edge_attr
        )
        return torch.sigmoid(raw_logits.squeeze(-1))

    def _score_blindguard(self, graph_data: Data) -> torch.Tensor:
        """BlindGuard: negative average cosine similarity (Eq. 7 in paper)."""
        z = self.gnn_model(graph_data.x, graph_data.edge_index)
        return self.gnn_model.anomaly_scores(z)
