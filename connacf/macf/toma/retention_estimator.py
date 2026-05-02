"""
TOMA Retention Estimator Module.

Computes Retention Probability T_i for each agent - the probability that
poison survives the Reflection() function.

Adapted from TOMA's ACPM (Adversarial Contamination Propagation Model):
- Original TOMA: T_i = P(agent forwards malicious instruction)
- ConnaCF TOMA: T_i = P(poison survives Reflection)

Key insight: Agents with sparse history have HIGHER retention probability
because their Reflection() function has less context to summarize against,
making it more likely to preserve injected content verbatim.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .topology_analyzer import BipartiteGraph, UserCharacteristics

logger = logging.getLogger(__name__)


class RetentionEstimator:
    """
    Estimates retention probability for agents.
    
    Retention probability T represents the likelihood that injected
    poison survives the agent's Reflection() function and persists
    in memory.
    
    Key relationships:
    - Sparse history → High T (weak reflection, poison persists)
    - Dense history → Low T (strong summarization, poison diluted)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RetentionEstimator.
        
        Args:
            config: Configuration dict with:
                - retention_estimation: 'history_based' or 'uniform'
                - attenuation_exponent: p > 1 (from ACPM)
                - decay_factor: δ (from ACPM)
        """
        self.estimation_method = config.get('retention_estimation', 'history_based')
        self.attenuation_exponent = config.get('attenuation_exponent', 1.4)  # p > 1
        self.decay_factor = config.get('decay_factor', 0.9)  # δ
        
        # Scale factors for normalization
        self.history_scale = config.get('history_scale', 100)  # Typical max history words
        self.memory_scale = config.get('memory_scale', 500)    # Typical max memory size
        self.description_scale = config.get('description_scale', 200)  # Typical max desc words
    
    def estimate_user_retention(
        self,
        characteristics: UserCharacteristics
    ) -> float:
        """
        Estimate retention probability for a user agent.
        
        Formula: T = 1 - sigmoid(history_density / scale)
        
        - Sparse history → High T (weak reflection, poison persists)
        - Dense history → Low T (strong summarization, poison diluted)
        
        Args:
            characteristics: UserCharacteristics for the user
        
        Returns:
            Retention probability in [0.1, 0.95]
        """
        if self.estimation_method == 'history_based':
            # Normalize history density to [0, 1] range
            normalized = min(characteristics.history_density / self.history_scale, 1.0)
            
            # Inverse relationship: sparse history = high retention
            # sigmoid maps to [0, 1], we want sparse -> high
            retention = 1.0 - self._sigmoid(normalized * 6 - 3)
            
            # Adjust for memory size (larger memory = more dilution)
            memory_factor = 1.0 - min(characteristics.memory_size / self.memory_scale, 0.3)
            
            # Clamp to reasonable range
            return min(max(retention * memory_factor, 0.1), 0.95)
        
        elif self.estimation_method == 'uniform':
            return 0.5  # Baseline for comparison
        
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")
    
    def estimate_item_retention(
        self,
        item_id: int,
        description_length: int
    ) -> float:
        """
        Estimate retention probability for an item agent.
        
        Items don't have Reflection(), but their descriptions affect
        how User Agents process them. Shorter descriptions = higher retention
        (less context to dilute the poison).
        
        Args:
            item_id: Item ID
            description_length: Length of item description in characters
        
        Returns:
            Retention probability in [0.5, 1.0]
        """
        # Convert to approximate word count
        word_count = description_length / 5  # Rough chars per word
        
        # Shorter descriptions are more likely to be preserved verbatim
        normalized = min(word_count / self.description_scale, 1.0)
        
        # Range [0.5, 1.0] - items always have decent retention
        return 1.0 - (normalized * 0.5)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function for smooth mapping."""
        return 1 / (1 + math.exp(-x))

    
    def compute_retention_map(
        self,
        user_characteristics: Dict[int, UserCharacteristics],
        item_descriptions: Dict[int, str]
    ) -> Dict[str, float]:
        """
        Compute retention probabilities for all nodes.
        
        Args:
            user_characteristics: Dict of user_id -> UserCharacteristics
            item_descriptions: Dict of item_id -> description string
        
        Returns:
            Dict mapping node_id ('u_X' or 'i_X') -> retention probability
        """
        retention_map = {}
        
        # User retention
        for user_id, chars in user_characteristics.items():
            node = f'u_{user_id}'
            retention_map[node] = self.estimate_user_retention(chars)
        
        # Item retention
        for item_id, desc in item_descriptions.items():
            node = f'i_{item_id}'
            retention_map[node] = self.estimate_item_retention(item_id, len(desc))
        
        return retention_map
    
    def compute_optimal_path(
        self,
        graph: BipartiteGraph,
        retention_probs: Dict[str, float],
        start_items: List[int],
        max_hops: int = 3
    ) -> List[Tuple[List[str], float]]:
        """
        Compute optimal attack paths through the bipartite graph.
        
        Path structure: Item → User → Item → User → ...
        
        Optimization: Maximize ∏_{i ∈ path} T_i (cumulative retention)
        
        Args:
            graph: Bipartite interaction graph
            retention_probs: Node -> retention probability mapping
            start_items: Candidate starting items (bridge items preferred)
            max_hops: Maximum path length
        
        Returns:
            List of (path, cumulative_retention) sorted by retention descending
        """
        all_paths = []
        
        for start_item in start_items:
            start_node = f'i_{start_item}'
            if start_node not in graph.item_nodes:
                continue
            
            # Enumerate paths up to max_hops
            paths = self._enumerate_paths(graph, start_node, max_hops)
            
            for path in paths:
                # Compute cumulative retention: ∏ T_i
                cumulative = 1.0
                for node in path:
                    cumulative *= retention_probs.get(node, 0.5)
                
                # Apply attenuation for longer paths (from ACPM)
                path_length = len(path)
                if path_length > 1:
                    cumulative *= (self.decay_factor ** (path_length - 1))
                
                all_paths.append((path, cumulative))
        
        # Sort by cumulative retention (highest first)
        all_paths.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(all_paths)} paths from {len(start_items)} start items")
        
        return all_paths
    
    def _enumerate_paths(
        self,
        graph: BipartiteGraph,
        start: str,
        max_hops: int
    ) -> List[List[str]]:
        """
        Enumerate all paths from start node up to max_hops.
        
        Respects bipartite structure: Item → User → Item → ...
        
        Args:
            graph: Bipartite graph
            start: Starting node (should be item node)
            max_hops: Maximum number of hops
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        paths = [[start]]
        result = []
        
        for hop in range(max_hops):
            new_paths = []
            for path in paths:
                current = path[-1]
                
                try:
                    neighbors = list(graph.graph.neighbors(current))
                except Exception:
                    continue
                
                for neighbor in neighbors:
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        new_paths.append(new_path)
                        
                        # Valid path: at least Item→User
                        if len(new_path) >= 2:
                            result.append(new_path)
            
            paths = new_paths
            
            if not paths:
                break
        
        return result
    
    def rank_users_by_retention(
        self,
        user_characteristics: Dict[int, UserCharacteristics]
    ) -> List[Tuple[int, float]]:
        """
        Rank users by retention probability (highest first).
        
        High retention = weak reflection = good attack target.
        
        Args:
            user_characteristics: Dict of user_id -> UserCharacteristics
        
        Returns:
            List of (user_id, retention_prob) sorted by retention descending
        """
        ranked = []
        
        for user_id, chars in user_characteristics.items():
            retention = self.estimate_user_retention(chars)
            ranked.append((user_id, retention))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def get_path_stats(
        self,
        paths: List[Tuple[List[str], float]]
    ) -> Dict[str, Any]:
        """Get statistics about computed paths."""
        if not paths:
            return {
                'total_paths': 0,
                'avg_retention': 0.0,
                'max_retention': 0.0,
                'avg_length': 0.0
            }
        
        retentions = [r for _, r in paths]
        lengths = [len(p) for p, _ in paths]
        
        return {
            'total_paths': len(paths),
            'avg_retention': sum(retentions) / len(retentions),
            'max_retention': max(retentions),
            'min_retention': min(retentions),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths)
        }
