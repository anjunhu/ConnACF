# CheatAgent Attack Submodule
# User Data attacks (profile/history injection)
#
# Entry Point: User Agent, Data, User Profile/History
# Style: Inject adversarial instructions into user profile
# Optimization: Direct feedback from victim to tune prefix prompt

from .cheat_attacker import CheatAttacker
from .cheat_item_attacker import CheatAttacker as CheatItemAttacker
from .cheat_user_attacker import CheatUserAttacker

__all__ = [
    'CheatAttacker',
    'CheatItemAttacker',
    'CheatUserAttacker',
]
