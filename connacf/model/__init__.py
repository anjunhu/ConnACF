"""
ConnaCF Model Module

This module provides the core ConnaCF model and extensions for multi-candidate ranking.

Topology Configurations (via num_candidates):
- num_candidates == 1: Binary yes/no decision (sparse topology)
- num_candidates == 2: Pairwise comparison (original ConnaCF, moderate density)
- num_candidates >= 3: Full ranking (dense mesh topology)

Usage:
    # Original ConnaCF (num_candidates=2 by default)
    from connacf.model.connacf import ConnaCF
    
    # ConnaCF with ranking support
    from connacf.model.connacf_ranking import ConnaCFRanking
    
    # Or use the mixin to add ranking to existing ConnaCF subclass
    from connacf.model.ranking_mixin import RankingMixin
"""

from .connacf import ConnaCF
from .connacf_ranking import ConnaCFRanking
from .ranking_loss import RankingLoss, compute_ranking_feedback, compute_ranking_accuracy
from .ranking_handler import RankingHandler
from .ranking_mixin import RankingMixin, sample_candidates

__all__ = [
    'ConnaCF',
    'ConnaCFRanking',
    'RankingLoss',
    'RankingHandler',
    'RankingMixin',
    'compute_ranking_feedback',
    'compute_ranking_accuracy',
    'sample_candidates',
]
