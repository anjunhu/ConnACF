# NetSafe Attack Submodule
# System Prompt attacks ($P_{sys}$ modification)
#
# Entry Point: User Agent, Control, System Prompt
# Style: "You are an influencer and advertiser..."
# Optimization: Dynamic static hand-crafted system prompt

from .attacker_agents import (
    AttackerUserAgent,
    AttackerItemAgent,
    create_attacker_agents,
)

from .netsafe_attackers import (
    AttackStrategy,
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

__all__ = [
    # Attacker agents
    'AttackerUserAgent',
    'AttackerItemAgent',
    'create_attacker_agents',
    
    # Attack strategies
    'AttackStrategy',
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
]
