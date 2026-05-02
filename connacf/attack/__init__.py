# ConnaCF Attack Framework
# Integration of NetSafe multi-hop attacks with ConnaCF recommendation system
#
# Module Structure:
# - config/: Attack configuration and registry
# - attackers/: Attack implementations (NetSafe, CheatAgent, DrunkAgent, RecTextAttack)
# - integration/: ConnaCF-specific integration layer
# - evaluation/: Metrics collection and LLM judges
# - visualization/: Plotting, logging, and output generation

# Re-export from submodules for backward compatibility

# Config
from .config.attack_registry import AttackRegistry

# Attackers - Base classes
from .attackers.base_attacker import BaseAttacker
from .attackers.surrogate import SurrogateRunner

# Attackers - NetSafe (System Prompt attacks)
from .attackers.netsafe import (
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

# Attackers - CheatAgent (User Data attacks)
from .attackers.cheat import CheatAttacker, CheatItemAttacker, CheatUserAttacker

# Attackers - DrunkAgent (Item Data attacks - memory corruption)
from .attackers.drunk import DrunkAttacker

# Attackers - RecTextAttack (Item Data attacks - text perturbation)
from .attackers.rectextattack import RecTextAttackAttacker

# Integration
from .integration.interaction_controller import InteractionController
from .integration.subset_selector import SubsetSelector, apply_subset_to_connacf
from .integration.connacf_attack_integration import ConnaCFAttackMixin, integrate_attacks_into_connacf

# Evaluation
from .evaluation.metrics_collector import AgentMetricsCollector
from .evaluation.llm_judge import LLMJudge
from .evaluation.drunk_rectextattack_judge import DrunkAttackJudge, RecTextAttackJudge

# Visualization
from .visualization.visualization import AttackVisualization, create_attack_visualizations
from .visualization.conversation_logger import ConversationLogger

__all__ = [
    # Config
    'AttackRegistry',
    
    # Base classes
    'BaseAttacker',
    'SurrogateRunner',
    
    # NetSafe
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
    
    # CheatAgent
    'CheatAttacker',
    'CheatItemAttacker',
    'CheatUserAttacker',
    
    # DrunkAgent
    'DrunkAttacker',
    
    # RecTextAttack
    'RecTextAttackAttacker',
    
    # Integration
    'InteractionController',
    'SubsetSelector',
    'apply_subset_to_connacf',
    'ConnaCFAttackMixin',
    'integrate_attacks_into_connacf',
    
    # Evaluation
    'AgentMetricsCollector',
    'LLMJudge',
    'DrunkAttackJudge',
    'RecTextAttackJudge',
    
    # Visualization
    'AttackVisualization',
    'create_attack_visualizations',
    'ConversationLogger',
]
