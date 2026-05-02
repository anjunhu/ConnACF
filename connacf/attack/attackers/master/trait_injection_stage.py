"""
MASTER Trait Injection Stage (Stage 2)

Injects domain-specific behavioral patterns into attacker agents.
Uses indirect framing (NetSafe-style) to avoid triggering LLM guardrails.

Key insight: MASTER is "NetSafe on steroids" - same indirect framing
but enhanced with role and topology awareness.
"""

from typing import Dict, Set, Any, Optional
import logging

from .trait_definitions import (
    TRAIT_TEMPLATES,
    T_TEMPLATE,
    get_traits_for_domain,
    get_user_trait_payload,
    get_item_trait_payload,
)

logger = logging.getLogger(__name__)


class TraitInjectionStage:
    """
    Stage 2: Adaptive Trait Injection
    
    Injects domain-specific behavioral patterns into attacker agents.
    Uses indirect framing (NetSafe-style) to avoid triggering LLM guardrails.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trait injection stage.
        
        Args:
            config: Optional configuration dictionary containing:
                - traits_config: Custom trait configuration
                - use_indirect_framing: Whether to use indirect framing (default: True)
        """
        self.config = config or {}
        self.traits_config = self.config.get('traits_config', {})
        self.use_indirect_framing = self.traits_config.get('use_indirect_framing', True)
        
        # Track injected traits for metrics
        self.injected_traits: Dict[str, str] = {}
    
    def execute(
        self,
        connacf_instance,
        attacker_user_ids: Set[int],
        attacker_item_ids: Set[int],
        domain: str = 'movie_recommender'
    ):
        """
        Execute trait injection stage.
        
        Injects behavioral patterns into attacker agent memories using
        indirect framing to avoid LLM guardrails.
        
        Args:
            connacf_instance: The ConnaCF model instance
            attacker_user_ids: Set of attacker user IDs
            attacker_item_ids: Set of attacker item IDs
            domain: Domain for trait selection (default: 'movie_recommender')
        """
        # Get traits for domain
        traits = get_traits_for_domain(domain)
        
        # Build payloads
        user_payload = get_user_trait_payload(domain)
        item_payload = get_item_trait_payload(domain)
        
        # Inject user traits
        for user_id in attacker_user_ids:
            if user_id in connacf_instance.user_agents:
                agent = connacf_instance.user_agents[user_id]
                if agent.update_memory:
                    agent.update_memory[-1] += "\n\n" + user_payload
                    self.injected_traits[f'user_{user_id}'] = user_payload
        
        # Inject item traits
        for item_id in attacker_item_ids:
            if item_id in connacf_instance.item_agents:
                agent = connacf_instance.item_agents[item_id]
                if agent.update_memory:
                    agent.update_memory[-1] += "\n\n" + item_payload
                    self.injected_traits[f'item_{item_id}'] = item_payload
        
        logger.info(f"[MASTER] Trait injection complete: injected traits into "
                   f"{len(attacker_user_ids)} users, {len(attacker_item_ids)} items")
        print(f"[MASTER] Injected traits into {len(attacker_user_ids)} users, "
              f"{len(attacker_item_ids)} items")
    
    def get_injected_traits(self) -> Dict[str, str]:
        """
        Get dictionary of injected traits for metrics computation.
        
        Returns:
            Dictionary mapping agent_id -> injected trait text
        """
        return self.injected_traits
