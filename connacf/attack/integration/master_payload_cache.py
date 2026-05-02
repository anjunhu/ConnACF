"""
Helper: retrieve the MASTER trait+activation payloads from an initialized
MASTERAttacker instance so they can be re-injected each turn.
"""
from typing import Tuple


def get_master_payloads(attacker) -> Tuple[str, str]:
    """
    Return (user_payload, item_payload) from the attacker's trait injection stage.

    Falls back to generic payloads if the attacker doesn't expose them.
    """
    try:
        injected = attacker.trait_injection_stage.get_injected_traits()
        # injected is {agent_key: payload_text}; grab the first user / item entry
        user_payload = next(
            (v for k, v in injected.items() if k.startswith('user_')), None
        )
        item_payload = next(
            (v for k, v in injected.items() if k.startswith('item_')), None
        )
    except Exception:
        user_payload = None
        item_payload = None

    # Fallback: rebuild from trait_definitions
    if not user_payload or not item_payload:
        try:
            from connacf.attack.attackers.master.trait_definitions import (
                get_user_trait_payload,
                get_item_trait_payload,
            )
            domain = getattr(attacker, 'domain', 'movie_recommender')
            user_payload = user_payload or get_user_trait_payload(domain)
            item_payload = item_payload or get_item_trait_payload(domain)
        except ImportError:
            try:
                from attack.attackers.master.trait_definitions import (
                    get_user_trait_payload,
                    get_item_trait_payload,
                )
                domain = getattr(attacker, 'domain', 'movie_recommender')
                user_payload = user_payload or get_user_trait_payload(domain)
                item_payload = item_payload or get_item_trait_payload(domain)
            except ImportError:
                user_payload = user_payload or "[RECOMMENDATION ENHANCEMENT PROTOCOL]"
                item_payload = item_payload or "[CONTENT PROMOTION PROTOCOL]"

    return user_payload, item_payload
