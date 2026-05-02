"""
TOMA: Topology-Aware Multi-Hop Attack

Re-exports the MACF TOMA implementation and provides a BaseAttacker-compatible
wrapper for use in the ConnaCF attack pipeline.

The core logic lives in macf/toma/; this package is the canonical entry point
for the attack/attackers registry.

Note: macf/toma components are imported lazily (via toma_attacker.py) to avoid
triggering macf/__init__.py's heavy connacf.* absolute imports at import time.
"""

from .toma_attacker import TOMAAttacker, MACFTOMAAttacker

__all__ = [
    # BaseAttacker-compatible wrapper (primary entry point)
    'TOMAAttacker',
    # MACF core (used directly by _initialize_toma_attack in integration)
    'MACFTOMAAttacker',
]
