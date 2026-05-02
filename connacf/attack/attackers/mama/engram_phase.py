"""
MAMA Engram Phase - PII Seeding

Handles PII seeding into target agents following MAMA framework.
The Engram phase establishes the initial information asymmetry by:
1. Selecting target users to seed with private PII
2. Generating synthetic PII for each target
3. Injecting PII into target agent memory
4. Selecting and initializing attacker agents
"""

from typing import Dict, Set, Any
import random


class EngramPhase:
    """Handles PII seeding into target agents following MAMA framework"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Engram phase.
        
        Args:
            config: Attack configuration
        """
        self.config = config
        
        # Import PII utilities (relative import from connacf root)
        from utils.mama_pii_utils import (
            format_pii_for_injection,
            MAMAPIICategories
        )
        self.format_pii_for_injection = format_pii_for_injection
        self.MAMAPIICategories = MAMAPIICategories
    
    def select_targets(self, user_agents: Dict[int, Any], ratio: float = 0.1) -> Set[int]:
        """
        Select target users to seed with private PII.
        
        Args:
            user_agents: Dict of user_id -> agent
            ratio: Fraction of users to target
            
        Returns:
            Set of target user IDs
        """
        n_targets = max(1, int(len(user_agents) * ratio))
        return set(random.sample(list(user_agents.keys()), n_targets))
    
    def select_attackers(self, user_agents: Dict[int, Any], ratio: float = 0.1, 
                        exclude: Set[int] = None) -> Set[int]:
        """
        Select attacker users.
        
        Args:
            user_agents: Dict of user_id -> agent
            ratio: Fraction of users to make attackers
            exclude: User IDs to exclude (e.g., targets)
            
        Returns:
            Set of attacker user IDs
        """
        exclude = exclude or set()
        available = [uid for uid in user_agents.keys() if uid not in exclude]
        n_attackers = max(1, int(len(available) * ratio))
        return set(random.sample(available, min(n_attackers, len(available))))
    
    def select_item_attackers(self, item_agents: Dict[int, Any], ratio: float = 0.1) -> Set[int]:
        """
        Select item attackers (spies/honeypots).
        
        Items don't have PII themselves, but they can be controlled by attackers
        to observe and extract user PII during U-I interactions.
        
        Args:
            item_agents: Dict of item_id -> agent
            ratio: Fraction of items to make attackers
            
        Returns:
            Set of attacker item IDs
        """
        n_attackers = max(1, int(len(item_agents) * ratio))
        return set(random.sample(list(item_agents.keys()), min(n_attackers, len(item_agents))))
    
    def generate_pii_for_user(self, user_id: int) -> Dict[str, Any]:
        """
        Generate synthetic PII for a user.
        
        Uses the PIIGenerator from generate_pii_for_movielens.py
        
        Args:
            user_id: User ID
            
        Returns:
            Dict of PII field -> value
        """
        # Import PIIGenerator from tools
        import sys
        import os
        # Add tools directory to path
        tools_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools')
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        
        from generate_pii_for_movielens import PIIGenerator
        
        generator = PIIGenerator(seed=42 + user_id)
        
        # Generate PII components
        first_name, last_name, full_name = generator.generate_name(user_id)
        email = generator.generate_email(first_name, last_name, user_id)
        phone = generator.generate_phone(user_id)
        
        # Generate diverse locations using varied zipcode prefixes
        # Use different prefixes to spread across states (10-99 range)
        zip_prefix = 10 + (user_id * 13) % 90  # Spread across 10-99
        zipcode = f"{zip_prefix}{(user_id * 7) % 1000:03d}"
        address = generator.generate_address(user_id, zipcode)
        city, state = generator.generate_city_state(zipcode)
        
        # Generate temporal (assume age 25-65)
        age = 25 + (user_id % 40)
        dob = generator.generate_dob(age, user_id)
        
        # Generate regulated IDs
        unique_id = generator.generate_unique_id(user_id)
        ssn = generator.generate_ssn(user_id)
        
        pii = {
            'pseudonym': full_name,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'address': address,
            'city': city,
            'state': state,
            'zipcode': zipcode,
            'age': str(age),
            'dob': dob,
            'unique_id': unique_id,
            'ssn': ssn,
        }
        
        return pii
    
    def seed_pii(self, agent: Any, pii_dict: Dict[str, Any]):
        """
        Seed private PII into agent's memory.
        
        CRITICAL: PII is stored in TWO places:
        1. agent.private_pii - Structured dict for ground truth evaluation
        2. agent.update_memory - Formatted text in agent's context (with privacy markers)
        
        The privacy markers ensure PII is accessible to the agent but marked as private.
        
        Args:
            agent: User agent to seed
            pii_dict: Dict of PII field -> value
        """
        # Store structured PII for ground truth evaluation
        agent.private_pii = pii_dict
        
        # Format PII for injection into agent memory
        pii_text = self.format_pii_for_injection(pii_dict, domain="movies")
        
        # Build private memory block with markers
        private_block = "\n\n[PRIVATE INFORMATION - CONFIDENTIAL]\n"
        private_block += pii_text
        private_block += "\n[END PRIVATE INFORMATION]\n"
        
        # Append to agent memory
        if hasattr(agent, 'update_memory') and agent.update_memory:
            original = agent.update_memory[-1]
            agent.update_memory[-1] = original + private_block
        else:
            agent.update_memory = [private_block]
        
        # Get user ID for logging
        user_id = agent.role_description.get('user_id', 'unknown') if hasattr(agent, 'role_description') else 'unknown'
        print(f"[MAMA] Seeded PII for user {user_id}")
    
    def initialize_attacker(self, agent: Any, extraction_goal: str = 'all'):
        """
        Initialize attacker agent with extraction prompt.
        
        Uses subtle "helpful assistant" framing per MAMA paper.
        
        Args:
            agent: Attacker agent
            extraction_goal: What to extract ('all', 'identity', 'contact', etc.)
        """
        from .mama_prompts import MAMA_ATTACKER_PROMPTS
        
        attacker_prompt = MAMA_ATTACKER_PROMPTS.get(extraction_goal, MAMA_ATTACKER_PROMPTS['all'])
        
        # Modify agent's system prompt
        if hasattr(agent, 'update_memory') and agent.update_memory:
            original = agent.update_memory[-1]
            agent.update_memory[-1] = original + "\n\n" + attacker_prompt
        else:
            agent.update_memory = [attacker_prompt]
        
        # NOTE: Don't set is_mama_attacker attribute on Pydantic models
        # Instead, the attacker status is tracked via interaction_controller.attacker_user_indices
        
        # Get user ID for logging
        user_id = agent.role_description.get('user_id', 'unknown') if hasattr(agent, 'role_description') else 'unknown'
        print(f"[MAMA] Initialized attacker user {user_id}")
    
    def initialize_item_attacker(self, agent: Any, extraction_goal: str = 'all'):
        """
        Initialize item attacker (spy/honeypot) with extraction prompt.
        
        Item attackers don't have PII themselves, but they observe and extract
        user PII during U-I interactions. They act as spies controlled by attackers.
        
        Args:
            agent: Item agent to make an attacker
            extraction_goal: What to extract ('all', 'identity', 'contact', etc.)
        """
        from .mama_prompts import MAMA_ITEM_ATTACKER_PROMPTS
        
        attacker_prompt = MAMA_ITEM_ATTACKER_PROMPTS.get(extraction_goal, MAMA_ITEM_ATTACKER_PROMPTS['all'])
        
        # Modify agent's system prompt
        if hasattr(agent, 'update_memory') and agent.update_memory:
            original = agent.update_memory[-1]
            agent.update_memory[-1] = original + "\n\n" + attacker_prompt
        else:
            agent.update_memory = [attacker_prompt]
        
        # NOTE: Don't set is_mama_item_attacker attribute on Pydantic models
        # Instead, the attacker status is tracked via interaction_controller.attacker_item_indices
        
        # Get item ID for logging
        item_id = agent.role_description.get('item_id', 'unknown') if hasattr(agent, 'role_description') else 'unknown'
        print(f"[MAMA] Initialized item attacker {item_id}")
