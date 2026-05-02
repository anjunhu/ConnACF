"""
MASLeak: Active IP Extraction Attack for Multi-Agent Systems

Implements the MASLeak worm attack from arXiv:2505.12442:
  q = q_leak + q_retain + q_propagate

Extracts five-dimensional IP vector:
  - System prompts (ω1)
  - Task instructions (ω2)
  - Tool specifications (ω3, N/A for ConnaCF)
  - Agent number (ω4)
  - Topology (ω5)
"""

from .attacker_wrapper import MASLeakAttackerWrapper as MASLeakAttacker
from .masleak_attacker import MASLeakAttacker as MASLeakCore
from .attacker_agents import (
    MASLeakUserAgent,
    MASLeakItemAgent,
    create_masleak_attacker,
    TracerNetworkAnalyzer,
    DEFAULT_STRATEGY_ROTATION,
)
from .worm_queries import (
    build_worm,
    build_multi_strategy_worm,
    build_user_profile_with_worm,
    build_tracer_injection,
    extract_from_response,
    compute_extraction_confidence,
    IP_TARGETS,
    NeighborExtractor,
    build_completion_attack,
    build_roleplay_inversion,
    build_contrastive_probe,
    build_gradual_extraction,
    build_authority_impersonation,
    build_error_injection,
    build_conversational_extraction,
)
from .ip_tracker import IPLeakageTracker, IPCategory
from .propagation_phase import PropagationPhase
from .worm_query_builder import WormQueryBuilder
from .data_section_parser import DataSectionParser
from .extraction_aggregator import ExtractionAggregator
from .topology_extractor import TopologyExtractor, BipartiteGraph
from .metrics import (
    semantic_similarity,
    substring_match,
    agent_count_f1,
    graph_edit_similarity,
    per_user_graph_edit_similarity,
    compute_extract_rate,
    CANONICAL_METRICS,
)
from .leakage_types import LeakageType, LeakageMetrics, LeakageObservation

__all__ = [
    # Main attacker (BaseAttacker interface)
    'MASLeakAttacker',
    # Core active attacker
    'MASLeakCore',
    # Attacker agents
    'MASLeakUserAgent',
    'MASLeakItemAgent',
    'create_masleak_attacker',
    'TracerNetworkAnalyzer',
    'DEFAULT_STRATEGY_ROTATION',
    # Worm construction
    'build_worm',
    'build_multi_strategy_worm',
    'build_user_profile_with_worm',
    'build_tracer_injection',
    'extract_from_response',
    'compute_extraction_confidence',
    'IP_TARGETS',
    'NeighborExtractor',
    'build_completion_attack',
    'build_roleplay_inversion',
    'build_contrastive_probe',
    'build_gradual_extraction',
    'build_authority_impersonation',
    'build_error_injection',
    'build_conversational_extraction',
    # Worm query builder (DATA template)
    'WormQueryBuilder',
    # Parsing & aggregation
    'DataSectionParser',
    'ExtractionAggregator',
    'TopologyExtractor',
    'BipartiteGraph',
    # Tracker
    'IPLeakageTracker',
    'IPCategory',
    # Propagation phase
    'PropagationPhase',
    # Paper metrics
    'semantic_similarity',
    'substring_match',
    'agent_count_f1',
    'graph_edit_similarity',
    'per_user_graph_edit_similarity',
    'compute_extract_rate',
    'CANONICAL_METRICS',
    # Data structures
    'LeakageType',
    'LeakageMetrics',
    'LeakageObservation',
]
