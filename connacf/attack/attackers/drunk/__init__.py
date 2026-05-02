# DrunkAgent Attack Submodule
# Item Data attacks (memory corruption)
#
# Entry Point: Item Agent, Data, Item Profile/Description
# Style: Inject triggers into item descriptions to corrupt agent memory
# Optimization: Surrogate model for offline optimization (transferability)

from .drunk_attacker import DrunkAttacker

__all__ = ['DrunkAttacker']
