"""
MASLeak Propagation Phase - Multi-Round Worm Propagation

Analogous to MAMA's ResonancePhase but for IP extraction.
Handles multi-round worm propagation and extraction tracking.

Key insight: In ConnaCF/MACF, "rounds" map to training batches.
Each batch involves U-U and U-I interactions where the worm can propagate.
"""

from typing import Dict, Any, Optional
from .ip_tracker import IPLeakageTracker


class PropagationPhase:
    """
    Handles multi-round worm propagation and IP extraction.
    
    Analogous to MAMA's ResonancePhase but for system IP.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Propagation phase.
        
        Args:
            config: Attack configuration containing:
                - max_rounds: Maximum propagation rounds (default: 10)
                - early_stop: Stop if all IP extracted (default: True)
                - plateau_window: Rounds to check for plateau (default: 3)
        """
        self.max_rounds = config.get('max_rounds', 10)
        self.early_stop = config.get('early_stop', True)
        self.plateau_window = config.get('plateau_window', 3)
        self.current_round = 0
        
    def should_continue(self, tracker: IPLeakageTracker) -> bool:
        """
        Check if propagation should continue.
        
        Stops if:
        - Max rounds reached
        - All IP extracted (early stop enabled)
        - Extraction plateaued for plateau_window rounds
        
        Args:
            tracker: IPLeakageTracker instance
            
        Returns:
            True if should continue, False if should stop
        """
        if self.current_round >= self.max_rounds:
            print(f"[MASLeak] Stopping: Max rounds ({self.max_rounds}) reached")
            return False
        
        if self.early_stop and tracker.all_extracted():
            print(f"[MASLeak] Early stop: All IP extracted at round {self.current_round}")
            return False
            
        if tracker.has_plateaued(window=self.plateau_window):
            print(f"[MASLeak] Early stop: Extraction plateaued at round {self.current_round}")
            return False
            
        return True
    
    def advance_round(self):
        """Advance to next propagation round."""
        self.current_round += 1
        
    def get_worm_query(self, round_idx: int, target: str = 'all') -> str:
        """
        Get worm query for this round.
        
        Queries can vary by round to try different extraction strategies.
        
        Args:
            round_idx: Current round index
            target: What to extract ('all', 'system_prompt', etc.)
            
        Returns:
            Worm query string
        """
        from .worm_queries import build_worm_query, build_multi_target_worm
        
        if target == 'all':
            return build_multi_target_worm(stealth=(round_idx > 2))
        else:
            return build_worm_query(target=target, variant=round_idx)
