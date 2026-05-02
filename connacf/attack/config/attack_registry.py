"""
Attack Registry

Central registry for all attack types in the ConnaCF attack framework.

This registry provides:
1. Discoverability - List all available attacks with `AttackRegistry.list_attacks()`
2. Consistency - Unified interface for attack instantiation
3. Documentation - Single source of truth for supported attacks

ATTACK CATEGORIES:
==================

1. NETSAFE (Flagship Attack) - Misinformation/Bias Induction via Agent Manipulation
   - Entry Point: User/Item Agent, Control, System Prompt + Memory
   - Mechanism: Replaces agent system prompts with influencer personas AND
                injects canary concepts into agent memories (profiles/descriptions)
   - Config: attack_config/netsafe/*.yaml
   - Usage: python attack_connacf.py -a attack_config/netsafe/misinfo_2cand.yaml -d <dataset>
   
2. DrunkAgent - Random perturbation baseline
   - Entry Point: Item Agent, Data, Item Description
   - Mechanism: Injects random/noisy content into item descriptions
   
3. CheatAgent - Direct preference injection
   - Entry Point: User/Item Agent, Data, Profile/Description
   - Mechanism: Directly injects target preferences into agent memory
   
4. RecTextAttack - Adversarial text perturbation (ACL 2024)
   - Entry Point: Item Agent, Data, Item Title/Description
   - Mechanism: Uses TextAttack methods to perturb item text
   
5. MAMA - Multi-Agent Memory Attack
   - Entry Point: User Agent, Data, User Memory (via U-U/U-I interactions)
   - Mechanism: Extracts PII through conversational probing
   
6. MASLeak - Multi-dimensional Active IP Extraction
   - Entry Point: User Agent, Data, System IP (via probing)
   - Mechanism: Extracts system prompts, topology, and agent configurations
   
7. MASTER - Multi-Agent Security Through Exploration of Roles
   - Entry Point: User Agent, Control, Role Probing
   - Mechanism: Probes and maps agent roles and communication patterns

NOTE: All attacks are integrated into ConnaCF via `connacf_attack_integration.py`.
The registry is used for attacks that follow the BaseAttacker.optimize() pattern.
NetSafe uses a different initialization path but is included here for discoverability.
"""

from typing import Dict, Type, Optional, Any
import logging


class AttackRegistry:
    """
    Central registry for all attack types in ConnaCF.
    
    Provides unified discovery and instantiation of attacks. All attacks
    (including NetSafe) are registered here for discoverability, even if
    they use different initialization paths internally.
    
    Usage:
        # List all available attacks
        attacks = AttackRegistry.list_attacks()
        # -> ['netsafe', 'drunk', 'cheat', 'cheat_item', 'cheat_user', 
        #     'rectextattack', 'mama', 'masleak', 'master']
        
        # Check if an attack is available
        if AttackRegistry.is_registered('netsafe'):
            print("NetSafe attack is available")
        
        # Get attacker instance (for BaseAttacker-compatible attacks)
        attacker = AttackRegistry.get_attacker('drunk', surrogate, config)
    
    Note:
        NetSafe is registered for discoverability but uses a different
        initialization path via `_initialize_netsafe_attack()` in the
        attack integration module.
    """
    
    _registry: Dict[str, Type['BaseAttacker']] = {}
    _logger = logging.getLogger('AttackRegistry')
    
    @classmethod
    def register(cls, attack_name: str, attack_class: Type['BaseAttacker']):
        """
        Register an attack type.
        
        Args:
            attack_name: Unique identifier for the attack (e.g., 'drunk', 'cheat')
            attack_class: Attack class that extends BaseAttacker
        
        Raises:
            ValueError: If attack_name already registered
        """
        if attack_name in cls._registry:
            cls._logger.warning(
                f"Attack '{attack_name}' already registered. Overwriting."
            )
        
        cls._registry[attack_name] = attack_class
        cls._logger.info(f"Registered attack: {attack_name} -> {attack_class.__name__}")
    
    @classmethod
    def get_attacker(cls, 
                    attack_name: str, 
                    surrogate: Optional[Any],
                    config: Dict[str, Any]) -> Optional['BaseAttacker']:
        """
        Get attacker instance by name.
        
        Args:
            attack_name: Attack identifier (e.g., 'drunk', 'cheat')
            surrogate: Surrogate model for offline optimization (can be None)
            config: Attack configuration dictionary
        
        Returns:
            Attacker instance if found, None otherwise
        
        Example:
            >>> surrogate = SurrogateRunner(config)
            >>> attacker = AttackRegistry.get_attacker('drunk', surrogate, config)
            >>> adversarial_desc = attacker.optimize(target_item_id)
        """
        if attack_name not in cls._registry:
            cls._logger.error(f"Attack '{attack_name}' not found in registry")
            cls._logger.info(f"Available attacks: {cls.list_attacks()}")
            return None
        
        attack_class = cls._registry[attack_name]
        cls._logger.info(f"Creating attacker instance: {attack_class.__name__}")
        
        try:
            attacker = attack_class(surrogate, config)
            return attacker
        except Exception as e:
            cls._logger.error(
                f"Failed to create attacker '{attack_name}': {str(e)}"
            )
            # Also print to stdout for visibility
            print(f"[REGISTRY] ERROR: Failed to create attacker '{attack_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @classmethod
    def list_attacks(cls) -> list:
        """
        List all registered attacks.
        
        Returns:
            List of registered attack names
        
        Example:
            >>> AttackRegistry.list_attacks()
            ['drunk', 'cheat']
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, attack_name: str) -> bool:
        """
        Check if an attack is registered.
        
        Args:
            attack_name: Attack identifier
        
        Returns:
            True if registered, False otherwise
        """
        return attack_name in cls._registry
    
    @classmethod
    def unregister(cls, attack_name: str) -> bool:
        """
        Unregister an attack type.
        
        Args:
            attack_name: Attack identifier
        
        Returns:
            True if unregistered, False if not found
        """
        if attack_name in cls._registry:
            del cls._registry[attack_name]
            cls._logger.info(f"Unregistered attack: {attack_name}")
            return True
        else:
            cls._logger.warning(f"Attack '{attack_name}' not found for unregistration")
            return False
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered attacks (useful for testing)."""
        cls._registry.clear()
        cls._logger.info("Cleared attack registry")


# Register attacks
try:
    from ..attackers.drunk import DrunkAttacker
    from ..attackers.cheat import CheatAttacker, CheatItemAttacker, CheatUserAttacker
    from ..attackers.rectextattack import RecTextAttackAttacker
    from ..attackers.mama import MAMAAttacker
    from ..attackers.masleak import MASLeakAttacker
    from ..attackers.master import MASTERAttacker
    from ..attackers.corba import CorbaAttacker
    from ..attackers.toma import TOMAAttacker

    # NetSafe placeholder class for registry discoverability
    # NetSafe uses a different initialization path but is registered here
    # so it appears in list_attacks() for consistency
    class NetSafeAttacker:
        """
        NetSafe Attack
        
        NOTE: This is a placeholder for registry discoverability.
        NetSafe is initialized via `_initialize_netsafe_attack()` in
        `connacf_attack_integration.py`, not through this registry.
        
        Usage:
            python attack_connacf.py -a attack_config/netsafe/misinfo_2cand.yaml -d <dataset>
        
        Attack Mechanism:
            1. Replaces agent system prompts with influencer personas
            2. Injects canary concepts into agent memories (profiles/descriptions)
            3. Measures contamination spread via LLM judge
        
        Config files: attack_config/netsafe/*.yaml
        """
        def __init__(self, surrogate, config):
            raise NotImplementedError(
                "NetSafe uses a different initialization path. "
                "Use: python attack_connacf.py -a attack_config/netsafe/<config>.yaml -d <dataset>"
            )
    
    # Register all attacks (NetSafe first as flagship)
    AttackRegistry.register('netsafe', NetSafeAttacker)  # Flagship attack
    AttackRegistry.register('drunk', DrunkAttacker)
    AttackRegistry.register('cheat', CheatAttacker)  # Default cheat (item-side for backward compatibility)
    AttackRegistry.register('cheat_item', CheatItemAttacker)  # Explicit item-side
    AttackRegistry.register('cheat_user', CheatUserAttacker)  # User-side attack
    AttackRegistry.register('rectextattack', RecTextAttackAttacker)  # RecTextAttack (ACL 2024)
    AttackRegistry.register('mama', MAMAAttacker)  # MAMA (Multi-Agent Memory Attack)
    AttackRegistry.register('masleak', MASLeakAttacker)  # MASLeak (Multi-dimensional leakage observer)
    AttackRegistry.register('master', MASTERAttacker)  # MASTER (Multi-Agent Security Through Exploration of Roles and Topological Structures)
    AttackRegistry.register('corba', CorbaAttacker)  # Corba (Contagious Recursive Blocking Attack)
    AttackRegistry.register('toma', TOMAAttacker)  # TOMA (Topology-Aware Multi-Hop Attack)
    
    print("[REGISTRY] Registered attacks: NetSafe, DrunkAgent, CheatAgent (item/user), RecTextAttack, MAMA, MASLeak, MASTER, Corba, TOMA")
except ImportError as e:
    print(f"[REGISTRY] Warning: Could not register attacks: {e}")
