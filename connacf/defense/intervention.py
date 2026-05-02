"""Intervention module for suppressing detected attackers' influence.

Provides two strategies for handling detected adversarial agents:
- 'filter': Remove attacker messages entirely from interactions.
- 'flag': Mark attacker messages with reduced weight, allowing partial influence.

The module tracks which agents are currently suppressed and supports
clearing suppressions before re-detection rounds.

Note: Intervention only affects an attacker's outgoing influence on other
agents. It does not modify the attacker agent's own internal state.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Set

logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("filter", "flag")


@dataclass
class InterventionResult:
    """Result of applying intervention to detected attackers.

    Attributes:
        suppressed_agents: List of agent IDs that were suppressed.
        strategy: Intervention strategy used ('filter' or 'flag').
        round_idx: Training round when intervention was applied.
    """

    suppressed_agents: List[int]
    strategy: str
    round_idx: int


class InterventionModule:
    """Suppresses detected attackers' influence on other agents.

    Supports two intervention strategies:
    - 'filter': Completely remove attacker messages from interactions,
      preventing any outgoing influence.
    - 'flag': Mark attacker messages with reduced weight, allowing
      partial influence while signaling untrustworthiness.

    The module only affects outgoing influence; it does not modify
    the attacker agent's own internal state (profile, memory, embeddings).

    Args:
        strategy: Intervention strategy, 'filter' or 'flag'.

    Raises:
        ValueError: If strategy is not 'filter' or 'flag'.
    """

    def __init__(self, strategy: str = "filter"):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid intervention strategy '{strategy}'. "
                f"Must be one of {VALID_STRATEGIES}."
            )
        self.strategy = strategy
        self._suppressed: Set[int] = set()

    def apply_intervention(
        self,
        detected_attackers: List[int],
        agent_type: str = "user",
        round_idx: int = 0,
    ) -> InterventionResult:
        """Apply suppression to detected attackers.

        Adds all detected attacker IDs to the suppressed set and logs
        the intervention action.

        Args:
            detected_attackers: List of agent IDs detected as attackers.
            agent_type: Type of agents being suppressed ('user' or 'item').
            round_idx: Current training round index.

        Returns:
            InterventionResult with suppressed agent IDs and strategy.
        """
        for agent_id in detected_attackers:
            self._suppressed.add(agent_id)

        logger.info(
            "Intervention round %d: suppressed %d %s agent(s) using '%s' strategy. "
            "Agents: %s",
            round_idx,
            len(detected_attackers),
            agent_type,
            self.strategy,
            detected_attackers,
        )

        return InterventionResult(
            suppressed_agents=list(detected_attackers),
            strategy=self.strategy,
            round_idx=round_idx,
        )

    def is_suppressed(self, agent_id: int) -> bool:
        """Check if an agent is currently suppressed.

        Args:
            agent_id: The agent ID to check.

        Returns:
            True if the agent is suppressed, False otherwise.
        """
        return agent_id in self._suppressed

    def clear_suppressions(self) -> None:
        """Reset all suppressions.

        Typically called before a new detection round so that only
        freshly detected attackers are suppressed.
        """
        count = len(self._suppressed)
        self._suppressed.clear()
        logger.debug("Cleared %d suppression(s).", count)
