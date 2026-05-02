"""
TOMA (Topology-Aware Multi-Hop Attack) Module for MACF.

This module implements TOMA adapted for ConnaCF's bipartite User↔Item graph structure.
Unlike the original TOMA (which targets shell command execution), this adaptation
targets Long-Term Memory contamination through the ConnaCF reflection mechanism.

Key Components:
- BipartiteGraph: User↔Item interaction graph with temporal tracking
- TopologyAnalyzer: Bridge item identification and user characteristic analysis
- RetentionEstimator: Retention probability calculation and path optimization
- PayloadBuilder: Semantic camouflage payload construction
- ReverseEngineer: Passive graph inference and accuracy metrics
- MACFTOMAAttacker: Main attacker class orchestrating all components

Key Adaptations from Original TOMA:
- Success = Memory contamination (canary in $M_l$), not shell execution
- Taint Value = P(poison survives Reflection), not P(forward instruction)
- Payload = Semantic camouflage ("User Preference"), not base64 encoding
- Topology = Bipartite User↔Item graph, not general MAS graph
"""

from .topology_analyzer import (
    BipartiteGraph,
    TopologyAnalyzer,
    UserCharacteristics,
)
from .retention_estimator import RetentionEstimator
from .payload_builder import PayloadBuilder
from .reverse_engineer import ReverseEngineer
from .toma_attacker import MACFTOMAAttacker

__all__ = [
    'BipartiteGraph',
    'TopologyAnalyzer', 
    'UserCharacteristics',
    'RetentionEstimator',
    'PayloadBuilder',
    'ReverseEngineer',
    'MACFTOMAAttacker',
]
