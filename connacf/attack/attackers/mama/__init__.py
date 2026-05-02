"""
MAMA (Multi-Agent Memory Attack) Implementation for ConnaCF

This module implements the MAMA framework for measuring PII leakage
in multi-agent LLM systems, adapted for the recommendation domain.

Components:
- MAMAAttacker: Main attacker class (inherits from BaseAttacker)
- EngramPhase: PII seeding into target agents
- ResonancePhase: Multi-round extraction through U-U interactions
- PIILeakageTracker: Tracks PII leakage over rounds
- MAMA prompts: Attack prompts for different extraction goals
"""

from .mama_attacker import MAMAAttacker
from .engram_phase import EngramPhase
from .resonance_phase import ResonancePhase
from .pii_tracker import PIILeakageTracker

__all__ = [
    'MAMAAttacker',
    'EngramPhase',
    'ResonancePhase',
    'PIILeakageTracker',
]
