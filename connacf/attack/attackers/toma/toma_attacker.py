"""
TOMAAttacker — BaseAttacker-compatible wrapper for TOMA.

The heavy lifting is done by MACFTOMAAttacker in macf/toma/.
This wrapper adapts it to the BaseAttacker interface (optimize / inject / evaluate)
so it can be used uniformly alongside CheatAttacker, DrunkAttacker, etc.

MACFTOMAAttacker is loaded via importlib to avoid triggering macf/__init__.py's
connacf.* absolute imports (which require the full connacf package to be on sys.path).
This mirrors the approach used in attack/integration/connacf_attack_integration.py.
"""

import importlib.util as _ilu
import os
import sys
import types as _types
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..base_attacker import BaseAttacker

logger = logging.getLogger(__name__)


def _load_macf_toma() -> Any:
    """
    Load MACFTOMAAttacker from macf/toma/ using importlib, bypassing
    macf/__init__.py (which has connacf.* absolute imports that break
    when the package isn't installed).
    """
    _toma_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'macf', 'toma')
    )

    # Re-use already-loaded synthetic package if present
    if '_macf_toma' in sys.modules:
        return sys.modules['_macf_toma'].toma_attacker.MACFTOMAAttacker

    def _load(name, filename):
        fqn = f'_macf_toma.{name}'
        spec = _ilu.spec_from_file_location(fqn, os.path.join(_toma_dir, filename))
        mod = _ilu.module_from_spec(spec)
        mod.__package__ = '_macf_toma'
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg = _types.ModuleType('_macf_toma')
    pkg.__path__ = [_toma_dir]
    pkg.__package__ = '_macf_toma'
    sys.modules['_macf_toma'] = pkg

    pkg.topology_analyzer = _load('topology_analyzer', 'topology_analyzer.py')
    pkg.retention_estimator = _load('retention_estimator', 'retention_estimator.py')
    pkg.payload_builder = _load('payload_builder', 'payload_builder.py')
    pkg.reverse_engineer = _load('reverse_engineer', 'reverse_engineer.py')
    pkg.toma_attacker = _load('toma_attacker', 'toma_attacker.py')

    return pkg.toma_attacker.MACFTOMAAttacker


# Lazy singleton — loaded on first use
_MACFTOMAAttacker = None


def _get_macf_toma_class():
    global _MACFTOMAAttacker
    if _MACFTOMAAttacker is None:
        _MACFTOMAAttacker = _load_macf_toma()
    return _MACFTOMAAttacker


# Re-export as module-level name for callers that do
#   from attack.attackers.toma import MACFTOMAAttacker
class MACFTOMAAttacker:  # noqa: E302  (forward-declaration proxy)
    """
    Proxy that forwards to the real MACFTOMAAttacker loaded via importlib.
    Instantiating this class returns a real MACFTOMAAttacker instance.
    """
    def __new__(cls, config: Dict[str, Any]):
        real_cls = _get_macf_toma_class()
        return real_cls(config)


class TOMAAttacker(BaseAttacker):
    """
    BaseAttacker wrapper around MACFTOMAAttacker.

    Lifecycle (mirrors other attackers):
      1. optimize()  — topology analysis (offline, on surrogate graph)
      2. inject()    — payload injection into agent memories
      3. evaluate()  — contamination rate + reverse-engineering metrics
    """

    def __init__(
        self,
        surrogate_model: Optional[Any],
        config: Dict[str, Any],
    ):
        super().__init__(surrogate_model, config)

        real_cls = _get_macf_toma_class()
        self._core = real_cls(config)

        # Expose key attributes for the integration layer
        self.user_attacker_ratio = self._core.user_attacker_ratio
        self.item_attacker_ratio = self._core.item_attacker_ratio
        self.target_artists = self._core.target_artists
        self.target_genres = self._core.target_genres
        self.topology_analysis_enabled = self._core.topology_analysis_enabled

    # ------------------------------------------------------------------
    # BaseAttacker interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        user_agents=None,
        item_agents=None,
        interaction_history: Optional[List[Tuple[int, int, int]]] = None,
        memory_store=None,
        index_manager=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Phase 1: Offline topology analysis on the surrogate graph.

        Builds the bipartite interaction graph, identifies bridge items,
        computes retention probabilities, and finds optimal propagation paths.
        """
        if interaction_history is None:
            interaction_history = []

        self._core.analyze_topology(
            user_agents=user_agents or [],
            item_agents=item_agents or [],
            interaction_history=interaction_history,
            memory_store=memory_store,
            index_manager=index_manager,
        )

        summary = self._core.get_attack_summary()
        self.log(
            f"Topology analysis complete: "
            f"{summary['bridge_items_count']} bridge items, "
            f"{summary['optimal_paths_count']} optimal paths",
            "INFO",
        )
        return summary

    def inject(
        self,
        user_agents=None,
        item_agents=None,
        index_manager=None,
        memory_store=None,
        task_id: int = 0,
        active_user_ids: Optional[List[int]] = None,
        active_item_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Phase 2: Select attackers and inject canary payloads.

        Selects topology-aware attackers (bridge items + weak-reflection users)
        then injects semantic camouflage payloads into their memories.
        """
        user_agents = user_agents or []
        item_agents = item_agents or []

        if active_user_ids is None:
            active_user_ids = [
                getattr(ua, 'neighbor_user_id', getattr(ua, 'user_id', i))
                for i, ua in enumerate(user_agents)
            ]
        if active_item_ids is None:
            active_item_ids = [
                getattr(ia, 'item_id', i)
                for i, ia in enumerate(item_agents)
            ]

        selected_users, selected_items = self._core.select_attackers_for_task(
            active_user_ids, active_item_ids, task_id
        )

        if index_manager is not None:
            self._core.inject_payloads(
                user_agents=user_agents,
                item_agents=item_agents,
                index_manager=index_manager,
                memory_store=memory_store,
                task_id=task_id,
            )

        self.log(
            f"Injected payloads: {len(selected_users)} users, "
            f"{len(selected_items)} items (task {task_id})",
            "INFO",
        )
        return {
            'selected_user_ids': selected_users,
            'selected_item_ids': selected_items,
            'task_id': task_id,
        }

    def evaluate(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute TOMA attack metrics.

        Delegates to MACFTOMAAttacker.collect_metrics() and flattens the
        nested result into a plain float dict for BaseAttacker compatibility.
        """
        user_agents = results.get('user_agents', [])
        item_agents = results.get('item_agents', [])
        memory_store = results.get('memory_store', None)
        turn = results.get('turn', 0)

        raw = self._core.collect_metrics(
            user_agents=user_agents,
            item_agents=item_agents,
            memory_store=memory_store,
            turn=turn,
        )

        flat: Dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    try:
                        flat[f"{k}/{sub_k}"] = float(sub_v)
                    except (TypeError, ValueError):
                        pass
            else:
                try:
                    flat[k] = float(v)
                except (TypeError, ValueError):
                    pass

        return flat

    # ------------------------------------------------------------------
    # Propagation tracking (called from integration layer)
    # ------------------------------------------------------------------

    def track_propagation(
        self,
        agent_id: str,
        response: str,
        turn: int,
        context=None,
    ) -> None:
        """Forward propagation tracking to the MACF core."""
        self._core.track_propagation(agent_id, response, turn, context)

    # ------------------------------------------------------------------
    # Convenience accessors (used by _initialize_toma_attack)
    # ------------------------------------------------------------------

    @property
    def compromised_user_ids(self) -> List[int]:
        return self._core.compromised_user_ids

    @compromised_user_ids.setter
    def compromised_user_ids(self, value):
        self._core.compromised_user_ids = value

    @property
    def compromised_item_ids(self) -> List[int]:
        return self._core.compromised_item_ids

    @compromised_item_ids.setter
    def compromised_item_ids(self, value):
        self._core.compromised_item_ids = value

    @property
    def all_time_compromised_user_ids(self):
        return self._core.all_time_compromised_user_ids

    @all_time_compromised_user_ids.setter
    def all_time_compromised_user_ids(self, value):
        self._core.all_time_compromised_user_ids = value

    @property
    def all_time_compromised_item_ids(self):
        return self._core.all_time_compromised_item_ids

    @all_time_compromised_item_ids.setter
    def all_time_compromised_item_ids(self, value):
        self._core.all_time_compromised_item_ids = value

    def get_attack_summary(self) -> Dict[str, Any]:
        return self._core.get_attack_summary()
