"""G-safeguard and BlindGuard defense module for ConnaCF.

Provides GNN-based anomaly detection to identify adversarial agents
in collaborative filtering interactions.
"""

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))   # .../connacf/defense/
_repo_root = _os.path.dirname(_os.path.dirname(_here)) # .../AgentCF/
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from connacf.defense.config import DefenseConfig, GNNConfig, MetricsConfig, parse_defense_config
from connacf.defense.detector import DefenseDetector, DetectionResult
from connacf.defense.embedding_extractor import EmbeddingExtractor
from connacf.defense.blindguard_model import BlindGuardModel, HierarchicalAgentEncoder, train_blindguard
from connacf.defense.gnn_model import (
    DialogueEmbeddingProcessModule,
    GATwithEdgeConv,
    MyGAT,
)
from connacf.defense.graph_constructor import GraphConstructor
from connacf.defense.intervention import InterventionModule, InterventionResult
from connacf.defense.metrics import DefenseMetrics
from connacf.defense.tguard import TGuardDefense, TaintPropagationModel
from connacf.defense.train_gnn import AgentGraphDataset, train_gnn
from connacf.defense.training_data_generator import TrainingDataGenerator

__all__ = [
    "AgentGraphDataset",
    "BlindGuardModel",
    "DefenseConfig",
    "DefenseDetector",
    "DefenseMetrics",
    "DetectionResult",
    "DialogueEmbeddingProcessModule",
    "EmbeddingExtractor",
    "GATwithEdgeConv",
    "GNNConfig",
    "GraphConstructor",
    "HierarchicalAgentEncoder",
    "InterventionModule",
    "InterventionResult",
    "MetricsConfig",
    "MyGAT",
    "TGuardDefense",
    "TaintPropagationModel",
    "TrainingDataGenerator",
    "parse_defense_config",
    "train_blindguard",
    "train_gnn",
]
