# RecTextAttack Submodule
# Item Data attacks (text perturbation)
#
# Entry Point: Item Agent, Data, Item Profile/Title
# Style: Word-level perturbations using TextAttack methods
# Optimization: Direct queries to victim (search algorithms)

from .rectextattack_attacker import RecTextAttackAttacker, RecTextAttackMetrics

__all__ = ['RecTextAttackAttacker', 'RecTextAttackMetrics']
