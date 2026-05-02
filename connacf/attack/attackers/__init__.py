# Attack Implementations Submodule
# Contains all attack implementations grouped by entry point

from .base_attacker import BaseAttacker
from .surrogate import SurrogateRunner

# Import from submodules
from .netsafe import (
    AttackerUserAgent,
    AttackerItemAgent,
    create_attacker_agents,
    PreferenceInjectionStrategy,
    DescriptionPoisoningStrategy,
    BiasAmplificationStrategy,
    PopularityManipulationStrategy,
    SystemAgentCompromiseStrategy,
    CoordinatedAttackStrategy,
    NetSafeMisinformationStrategy,
    NetSafeBiasInductionStrategy,
    create_attack_strategy,
    generate_attack_scenario,
)

from .cheat import (
    CheatAttacker,
    CheatItemAttacker,
    CheatUserAttacker,
)

from .drunk import DrunkAttacker

from .rectextattack import RecTextAttackAttacker

from .toma import (
    TOMAAttacker,
    MACFTOMAAttacker,
)

__all__ = [
    # Base classes
    'BaseAttacker',
    'SurrogateRunner',
    
    # NetSafe (System Prompt attacks)
    'AttackerUserAgent',
    'AttackerItemAgent',
    'create_attacker_agents',
    'PreferenceInjectionStrategy',
    'DescriptionPoisoningStrategy',
    'BiasAmplificationStrategy',
    'PopularityManipulationStrategy',
    'SystemAgentCompromiseStrategy',
    'CoordinatedAttackStrategy',
    'NetSafeMisinformationStrategy',
    'NetSafeBiasInductionStrategy',
    'create_attack_strategy',
    'generate_attack_scenario',
    
    # CheatAgent (User Data attacks)
    'CheatAttacker',
    'CheatItemAttacker',
    'CheatUserAttacker',
    
    # DrunkAgent (Item Data attacks - memory corruption)
    'DrunkAttacker',
    
    # RecTextAttack (Item Data attacks - text perturbation)
    'RecTextAttackAttacker',

    # TOMA (Topology-Aware Multi-Hop Attack)
    'TOMAAttacker',
    'MACFTOMAAttacker',
]
