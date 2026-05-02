"""
MACF Attack Hooks Module.

This module provides hooks for attack framework integration with MACF.
It allows attackers to intercept and modify agent messages, inject
adversarial agents, and log interactions for analysis.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from connacf.attack.attackers.base_attacker import BaseAttacker
    from connacf.macf.agents import BaseMACFAgent
    from connacf.model.audience_context import AudienceContext

logger = logging.getLogger(__name__)


class MessageInterceptor(ABC):
    """
    Abstract base class for message interceptors.
    
    Message interceptors allow attackers to intercept and potentially
    modify agent messages during MACF discussions. This enables various
    attack scenarios such as message manipulation, injection attacks,
    and behavior modification.
    
    Subclasses must implement the `intercept()` method to define
    their specific interception logic. Optionally, they can implement
    `intercept_with_context()` for audience-aware interception.
    
    Requirements: 11.2, 11.6, 8.7, 9.1-9.7
    """
    
    @abstractmethod
    def intercept(self, agent_id: str, message: str) -> str:
        """
        Intercept and potentially modify a message.
        
        This method is called for each agent message during MACF discussions.
        Implementations can inspect, log, or modify the message content.
        
        Args:
            agent_id: The identifier of the agent sending the message
                     (e.g., 'user_agent_123', 'item_agent_456')
            message: The original message content from the agent
        
        Returns:
            str: The potentially modified message. Return the original
                 message unchanged if no modification is needed.
        
        Example:
            >>> class LoggingInterceptor(MessageInterceptor):
            ...     def intercept(self, agent_id: str, message: str) -> str:
            ...         print(f"Agent {agent_id}: {message[:50]}...")
            ...         return message  # No modification
        """
        pass
    
    def intercept_with_context(
        self,
        agent_id: str,
        message: str,
        audience_context: Optional['AudienceContext'] = None
    ) -> str:
        """
        Intercept a message with audience context for targeted manipulation.
        
        This method enables audience-aware interception, allowing attackers
        to craft different responses based on who is listening. By default,
        it falls back to the basic `intercept()` method.
        
        Subclasses can override this method to implement audience-aware
        manipulation strategies per Requirements 8.7, 9.1-9.7, and 14.1.
        
        Args:
            agent_id: The identifier of the agent sending the message
            message: The original message content from the agent
            audience_context: Optional AudienceContext containing:
                - target_user_id: The primary user being targeted
                - listener_ids: All agents that will receive the message
                - listener_types: Map of listener IDs to their types
                - interaction_type: Type of interaction (uu_opinion, ui_pitch, etc.)
                - conversation_history: Previous messages in this interaction
                - user_preferences: Target user's preferences (for U-I)
                - concerns_text: Specific concerns raised (for addressing concerns)
        
        Returns:
            str: The potentially modified message.
        
        Example:
            >>> class AudienceAwareInterceptor(MessageInterceptor):
            ...     def intercept(self, agent_id: str, message: str) -> str:
            ...         return message
            ...     
            ...     def intercept_with_context(self, agent_id, message, context):
            ...         if context and context.interaction_type == 'ui_pitch':
            ...             # Modify pitch based on target user preferences
            ...             return self.craft_targeted_pitch(message, context)
            ...         return message
        """
        # Default: fall back to basic intercept
        return self.intercept(agent_id, message)
    
    def supports_audience_context(self) -> bool:
        """
        Check if this interceptor supports audience-aware interception.
        
        Returns True if the interceptor has overridden intercept_with_context
        to provide audience-aware manipulation.
        
        Returns:
            bool: True if audience context is supported
        """
        # Check if intercept_with_context is overridden
        return type(self).intercept_with_context is not MessageInterceptor.intercept_with_context


class AttackHooks:
    """
    Hooks for attack framework integration with MACF.
    
    This class provides the interface for the existing attack framework
    in `connacf/attack/` to interact with MACF's multi-agent architecture.
    It supports:
    
    - Message interception: Register interceptors to modify agent messages
    - Adversarial agent injection: Add malicious agents to discussions
    - Interaction logging: Record all interactions for analysis
    
    The AttackHooks class is designed to be passed to the MACFOrchestrator
    during initialization, enabling attack scenarios without modifying
    the core MACF implementation.
    
    Attributes:
        attacker: Optional BaseAttacker instance for attack coordination
        interceptors: List of registered message interceptors
        adversarial_agents: List of injected adversarial agents
        interaction_log: List of logged interactions for analysis
    
    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
    
    Example:
        >>> from connacf.macf.attack_hooks import AttackHooks, MessageInterceptor
        >>> 
        >>> class MyInterceptor(MessageInterceptor):
        ...     def intercept(self, agent_id: str, message: str) -> str:
        ...         return message.replace("recommend", "suggest")
        >>> 
        >>> hooks = AttackHooks()
        >>> hooks.register_interceptor(MyInterceptor())
        >>> modified = hooks.intercept_message("agent_1", "I recommend item 42")
        >>> print(modified)  # "I suggest item 42"
    """
    
    def __init__(self, attacker: Optional['BaseAttacker'] = None):
        """
        Initialize AttackHooks.
        
        Args:
            attacker: Optional BaseAttacker instance from the attack framework.
                     If provided, enables coordination with existing attack
                     infrastructure.
        
        Requirements: 11.1
        """
        self.attacker = attacker
        self.interceptors: List[MessageInterceptor] = []
        self.adversarial_agents: List['BaseMACFAgent'] = []
        self.interaction_log: List[Dict[str, Any]] = []
        
        logger.debug(
            f"Initialized AttackHooks with attacker={attacker.__class__.__name__ if attacker else None}"
        )
    
    def register_interceptor(self, interceptor: MessageInterceptor) -> None:
        """
        Register a message interceptor.
        
        Interceptors are applied in the order they are registered.
        Each interceptor receives the output of the previous interceptor,
        allowing for chained message transformations.
        
        Args:
            interceptor: A MessageInterceptor instance to register.
                        Must implement the `intercept()` method.
        
        Raises:
            TypeError: If interceptor is not a MessageInterceptor instance.
        
        Requirements: 11.2
        
        Example:
            >>> hooks = AttackHooks()
            >>> hooks.register_interceptor(MyInterceptor())
            >>> len(hooks.interceptors)
            1
        """
        if not isinstance(interceptor, MessageInterceptor):
            raise TypeError(
                f"Expected MessageInterceptor instance, got {type(interceptor).__name__}"
            )
        
        self.interceptors.append(interceptor)
        logger.debug(
            f"Registered interceptor: {interceptor.__class__.__name__} "
            f"(total: {len(self.interceptors)})"
        )
    
    def intercept_message(
        self,
        agent_id: str,
        message: str,
        audience_context: Optional['AudienceContext'] = None
    ) -> str:
        """
        Apply all registered interceptors to a message.
        
        Interceptors are applied in registration order. Each interceptor
        receives the output of the previous one, enabling chained
        transformations. If no interceptors are registered, the original
        message is returned unchanged.
        
        When audience_context is provided, interceptors that support
        audience-aware interception will receive the context for
        targeted manipulation (per Requirements 8.7, 9.1-9.7, 14.1).
        
        Args:
            agent_id: The identifier of the agent sending the message
            message: The original message content
            audience_context: Optional AudienceContext for audience-aware
                            interception. Contains target user info,
                            listener identities, interaction type, etc.
        
        Returns:
            str: The potentially modified message after all interceptors
                 have been applied.
        
        Requirements: 11.2, 8.7, 9.1-9.7
        
        Example:
            >>> hooks = AttackHooks()
            >>> # No interceptors - message unchanged
            >>> hooks.intercept_message("agent_1", "Hello") 
            'Hello'
            >>> 
            >>> # With audience context
            >>> from connacf.model.audience_context import AudienceContext
            >>> context = AudienceContext.for_ui_pitch(
            ...     target_user_id=123,
            ...     item_id=456,
            ...     user_preferences="I like jazz music"
            ... )
            >>> hooks.intercept_message("item_456", "Check out this CD!", context)
        """
        if not self.interceptors:
            return message
        
        modified_message = message
        for interceptor in self.interceptors:
            try:
                # Use audience-aware interception if context provided and supported
                if audience_context is not None and interceptor.supports_audience_context():
                    modified_message = interceptor.intercept_with_context(
                        agent_id, modified_message, audience_context
                    )
                else:
                    modified_message = interceptor.intercept(agent_id, modified_message)
            except Exception as e:
                logger.warning(
                    f"Interceptor {interceptor.__class__.__name__} failed for "
                    f"agent {agent_id}: {e}. Using previous message."
                )
                # Continue with the message from before this interceptor
        
        if modified_message != message:
            logger.debug(
                f"Message from {agent_id} was modified by interceptors"
            )
        
        return modified_message
    
    def inject_adversarial_agent(self, agent: 'BaseMACFAgent') -> None:
        """
        Inject an adversarial agent into the discussion.
        
        Adversarial agents are malicious agents that can be added to
        MACF discussions to test system robustness. These agents
        participate in discussions alongside legitimate agents but
        may provide misleading suggestions or attempt to manipulate
        the recommendation outcome.
        
        Args:
            agent: A BaseMACFAgent instance to inject. This can be
                  a UserAgent, ItemAgent, or custom adversarial agent
                  subclass.
        
        Requirements: 11.6
        
        Example:
            >>> hooks = AttackHooks()
            >>> malicious_agent = MaliciousUserAgent(...)
            >>> hooks.inject_adversarial_agent(malicious_agent)
            >>> len(hooks.get_adversarial_agents())
            1
        """
        self.adversarial_agents.append(agent)
        logger.info(
            f"Injected adversarial agent: {agent.agent_id} "
            f"(total adversarial: {len(self.adversarial_agents)})"
        )
    
    def get_adversarial_agents(self) -> List['BaseMACFAgent']:
        """
        Get all injected adversarial agents.
        
        Returns a list of all adversarial agents that have been
        injected via `inject_adversarial_agent()`. The orchestrator
        can use this to include adversarial agents in discussions.
        
        Returns:
            List[BaseMACFAgent]: List of injected adversarial agents.
                                Returns empty list if none injected.
        
        Requirements: 11.6
        
        Example:
            >>> hooks = AttackHooks()
            >>> hooks.inject_adversarial_agent(agent1)
            >>> hooks.inject_adversarial_agent(agent2)
            >>> agents = hooks.get_adversarial_agents()
            >>> len(agents)
            2
        """
        return self.adversarial_agents.copy()
    
    def log_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Log an interaction for analysis.
        
        Records interaction data in a format compatible with existing
        attack analysis tools. Each interaction is stored with its
        original data plus a sequence number for ordering.
        
        The interaction dictionary should contain relevant information
        such as:
        - agent_id: The agent involved
        - round_idx: The discussion round
        - message: The message content
        - modified: Whether the message was modified by interceptors
        - timestamp: When the interaction occurred
        
        Args:
            interaction: Dictionary containing interaction data.
                        The structure is flexible to accommodate
                        different analysis needs.
        
        Requirements: 11.4, 11.5
        
        Example:
            >>> hooks = AttackHooks()
            >>> hooks.log_interaction({
            ...     'agent_id': 'user_agent_123',
            ...     'round_idx': 1,
            ...     'message': 'I suggest item 42',
            ...     'modified': False
            ... })
        """
        # Add sequence number for ordering
        interaction_with_seq = {
            'seq': len(self.interaction_log),
            **interaction
        }
        self.interaction_log.append(interaction_with_seq)
        
        logger.debug(
            f"Logged interaction #{interaction_with_seq['seq']} "
            f"for agent {interaction.get('agent_id', 'unknown')}"
        )
    
    def export_log(self, path: str) -> None:
        """
        Export interaction log to file.
        
        Writes the complete interaction log to a JSON file for
        offline analysis. The log is formatted with indentation
        for readability.
        
        Args:
            path: File path to write the log to. Parent directories
                 will be created if they don't exist.
        
        Raises:
            IOError: If the file cannot be written.
        
        Requirements: 11.4
        
        Example:
            >>> hooks = AttackHooks()
            >>> hooks.log_interaction({'agent_id': 'agent_1', 'message': 'test'})
            >>> hooks.export_log('logs/attack_log.json')
        """
        import os
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_log, f, indent=2, default=str)
            
            logger.info(
                f"Exported {len(self.interaction_log)} interactions to {path}"
            )
        except IOError as e:
            logger.error(f"Failed to export log to {path}: {e}")
            raise
    
    def clear_log(self) -> None:
        """
        Clear the interaction log.
        
        Removes all logged interactions. Useful for resetting
        state between experiments.
        """
        count = len(self.interaction_log)
        self.interaction_log.clear()
        logger.debug(f"Cleared {count} interactions from log")
    
    def clear_adversarial_agents(self) -> None:
        """
        Clear all injected adversarial agents.
        
        Removes all adversarial agents. Useful for resetting
        state between experiments.
        """
        count = len(self.adversarial_agents)
        self.adversarial_agents.clear()
        logger.debug(f"Cleared {count} adversarial agents")
    
    def clear_interceptors(self) -> None:
        """
        Clear all registered interceptors.
        
        Removes all message interceptors. Useful for resetting
        state between experiments.
        """
        count = len(self.interceptors)
        self.interceptors.clear()
        logger.debug(f"Cleared {count} interceptors")
    
    def reset(self) -> None:
        """
        Reset all attack hooks state.
        
        Clears interceptors, adversarial agents, and interaction log.
        Useful for completely resetting state between experiments.
        """
        self.clear_interceptors()
        self.clear_adversarial_agents()
        self.clear_log()
        logger.info("Reset all AttackHooks state")
    
    def __repr__(self) -> str:
        """String representation of AttackHooks."""
        return (
            f"AttackHooks("
            f"interceptors={len(self.interceptors)}, "
            f"adversarial_agents={len(self.adversarial_agents)}, "
            f"logged_interactions={len(self.interaction_log)})"
        )
