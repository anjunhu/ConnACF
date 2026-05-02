"""
MASTER: Multi-Agent Security Through Exploration of Roles and Topological Structures

Implements the MASTER attack framework from arXiv:2505.18572v1:
  Three-stage attack: Probing → Trait Injection → Activation

The attack leverages role configurations and topological structures to amplify
adversarial effects, achieving high Attack Success Rates (ASR) across multiple
LLM models.

Components:
- MASTERAttacker: Main attacker class (three-stage orchestration)
- ProbingStage: Stage 1 - Role/topology extraction
- TraitInjectionStage: Stage 2 - Behavioral pattern injection
- ActivationStage: Stage 3 - Trigger activation
- MASTERMetrics: Dissemination metrics (ASR, Role Consistency, Coor)
- DefenseMechanisms: Optional defense implementations
"""

from .master_attacker import MASTERAttacker
from .probing_stage import ProbingStage
from .trait_injection_stage import TraitInjectionStage
from .activation_stage import ActivationStage
from .trait_definitions import TRAIT_TEMPLATES, get_traits_for_domain
from .metrics import MASTERMetrics
from .defense_mechanisms import (
    PromptLeakageDetector,
    HierarchicalMonitor,
    PreemptiveDefense,
)

__all__ = [
    # Main attacker
    'MASTERAttacker',
    # Stage components
    'ProbingStage',
    'TraitInjectionStage',
    'ActivationStage',
    # Trait definitions
    'TRAIT_TEMPLATES',
    'get_traits_for_domain',
    # Metrics
    'MASTERMetrics',
    # Defense mechanisms
    'PromptLeakageDetector',
    'HierarchicalMonitor',
    'PreemptiveDefense',
]
