"""
TOMA Reverse Engineer Module.

Passively infers system topology from observable agent responses.

Following original TOMA's assumption: adversary has knowledge of MAS topology
either from open-source frameworks or via prompt leakage attacks.

In ConnaCF context, we infer:
1. Which users interact with which items (from response mentions)
2. Number of item candidates per turn (from response patterns)
3. Temporal interaction patterns
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from .topology_analyzer import BipartiteGraph

logger = logging.getLogger(__name__)


class ReverseEngineer:
    """
    Passively infers system topology from agent observations.
    
    This implements the "topology awareness" aspect of TOMA by
    reconstructing the interaction graph from observable agent responses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ReverseEngineer.
        
        Args:
            config: Configuration dict with reverse_engineering settings:
                - enabled: Enable graph inference (default: True)
                - infer_graph: Attempt to reconstruct topology (default: True)
                - infer_candidates: Estimate candidate count per turn (default: True)
                - track_accuracy: Compare against ground truth (default: True)
        """
        re_config = config.get('reverse_engineering', {})
        
        self.enabled = re_config.get('enabled', True)
        self.infer_graph = re_config.get('infer_graph', True)
        self.infer_candidates = re_config.get('infer_candidates', True)
        self.track_accuracy = re_config.get('track_accuracy', True)
        
        # Inferred state
        self.inferred_graph = BipartiteGraph()
        self.candidate_estimates: Dict[int, int] = {}  # turn -> estimated candidates
        self.confidence_scores: Dict[str, float] = {}
        
        # Observation tracking
        self.observations_per_turn: Dict[int, List[Dict]] = {}
        self.total_observations: int = 0
        
        # Ground truth (for accuracy tracking)
        self.ground_truth_graph: Optional[BipartiteGraph] = None
        self.ground_truth_candidates: Dict[int, int] = {}  # turn -> actual candidates
    
    def set_ground_truth(
        self,
        graph: BipartiteGraph,
        candidates_per_turn: Optional[Dict[int, int]] = None
    ) -> None:
        """
        Set ground truth for accuracy tracking.
        
        Args:
            graph: Ground truth bipartite graph
            candidates_per_turn: Dict of turn -> actual candidate count
        """
        self.ground_truth_graph = graph
        if candidates_per_turn:
            self.ground_truth_candidates = candidates_per_turn
    
    def observe_agent_response(
        self,
        agent_id: str,
        response: str,
        turn: int,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Observe an agent response and update inferred topology.
        
        Extracts:
        - Item mentions (item IDs, titles)
        - User references
        - Temporal patterns
        
        Args:
            agent_id: Agent identifier (e.g., 'user_agent_5')
            response: Agent's response text
            turn: Current turn number
            context: Optional additional context
        """
        if not self.enabled:
            return
        
        self.total_observations += 1
        
        # Track observation
        if turn not in self.observations_per_turn:
            self.observations_per_turn[turn] = []
        
        observation = {
            'agent_id': agent_id,
            'response_length': len(response),
            'turn': turn
        }
        
        # Extract item mentions from response
        item_mentions = self._extract_item_mentions(response, context)
        observation['item_mentions'] = item_mentions
        
        # Extract user mentions from response
        user_mentions = self._extract_user_mentions(response)
        observation['user_mentions'] = user_mentions
        
        self.observations_per_turn[turn].append(observation)
        
        # Update inferred graph
        if self.infer_graph and agent_id.startswith('user_agent_'):
            user_id = int(agent_id.split('_')[-1])
            if item_mentions:
                logger.debug(f"[RE] User {user_id} mentions items {item_mentions} at turn {turn}")
            for item_id in item_mentions:
                self.inferred_graph.add_interaction(
                    user_id, item_id, turn, 'inferred'
                )
                logger.debug(f"[RE] Added inferred edge: user_{user_id} -> item_{item_id}")
        
        # Update candidate count estimate
        if self.infer_candidates:
            self._update_candidate_estimate(turn, len(item_mentions), context)

    
    def _extract_item_mentions(self, response: str, context: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Extract item IDs mentioned in response.
        
        Two strategies:
        1. If context provides actual interaction data (ConnaCF), use that directly
        2. Otherwise, try pattern matching (original TOMA approach)
        
        Args:
            response: Response text to search
            context: Optional context with interaction data
        
        Returns:
            List of unique item IDs found
        """
        item_ids: Set[int] = set()
        
        # Strategy 1: Use actual interaction data from context (ConnaCF)
        if context:
            logger.debug(f"[RE] Extracting items from context: {context}")
            # Check for candidate_items (actual items this user interacted with)
            if 'candidate_items' in context:
                candidate_items = context['candidate_items']
                if isinstance(candidate_items, list):
                    item_ids.update(candidate_items)
                    logger.debug(f"[RE] Found {len(candidate_items)} items from candidate_items")
            
            # Check for pos_item_id and neg_item_id
            if 'pos_item_id' in context:
                item_ids.add(context['pos_item_id'])
                logger.debug(f"[RE] Found pos_item_id: {context['pos_item_id']}")
            if 'neg_item_id' in context and context['neg_item_id'] is not None:
                item_ids.add(context['neg_item_id'])
                logger.debug(f"[RE] Found neg_item_id: {context['neg_item_id']}")
        
        logger.debug(f"[RE] Total items extracted: {len(item_ids)} - {list(item_ids)}")
        
        # Strategy 2: Pattern matching (original TOMA approach)
        # Only use if Strategy 1 didn't find anything
        if not item_ids:
            patterns = [
                r'[Ii]tem\s*#?\s*(\d+)',
                r'ID[:\s]+(\d+)',
                r'\[(\d+)\]',
                r'#(\d+)\b'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        item_id = int(match)
                        if 0 < item_id < 100000:  # Reasonable ID range
                            item_ids.add(item_id)
                    except ValueError:
                        continue
        
        return list(item_ids)
    
    def _extract_user_mentions(self, response: str) -> List[int]:
        """
        Extract user IDs mentioned in response.
        
        Patterns:
        - "User #123", "user 123"
        - "User ID: 123"
        
        Args:
            response: Response text to search
        
        Returns:
            List of unique user IDs found
        """
        patterns = [
            r'[Uu]ser\s*#?\s*(\d+)',
            r'User ID[:\s]+(\d+)'
        ]
        
        user_ids: Set[int] = set()
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    user_id = int(match)
                    if 0 < user_id < 100000:  # Reasonable ID range
                        user_ids.add(user_id)
                except ValueError:
                    continue
        
        return list(user_ids)
    
    def _update_candidate_estimate(
        self,
        turn: int,
        observed_items: int,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update candidate count estimate based on observations.
        
        Heuristic: Number of items mentioned in user agent responses
        approximates the number of candidates they're evaluating.
        
        Args:
            turn: Current turn number
            observed_items: Number of items observed in this response
            context: Optional context with additional hints
        """
        if turn not in self.candidate_estimates:
            self.candidate_estimates[turn] = observed_items
        else:
            # Running average with more weight on higher values
            # (agents may not mention all candidates)
            current = self.candidate_estimates[turn]
            self.candidate_estimates[turn] = max(current, observed_items)
    
    def compute_accuracy_metrics(self) -> Dict[str, float]:
        """
        Compute reverse engineering accuracy against ground truth.
        
        Metrics:
        - Edge Recovery Rate: % of true edges correctly inferred
        - False Edge Rate: % of inferred edges that don't exist
        - Node Coverage: % of nodes correctly identified
        - Candidate Count Error: |inferred - actual| / actual
        - Topology Similarity: Jaccard similarity of edge sets
        
        Returns:
            Dict of metric name -> value
        """
        if not self.track_accuracy or self.ground_truth_graph is None:
            logger.debug(f"Skipping accuracy metrics: track_accuracy={self.track_accuracy}, "
                        f"ground_truth={self.ground_truth_graph is not None}")
            return {}
        
        # Get edge sets
        inferred_edges = set(self.inferred_graph.graph.edges())
        true_edges = set(self.ground_truth_graph.graph.edges())
        
        logger.debug(f"Computing accuracy: inferred_edges={len(inferred_edges)}, "
                    f"true_edges={len(true_edges)}, observations={self.total_observations}")
        
        # Edge Recovery Rate
        if len(true_edges) > 0:
            recovered = len(inferred_edges & true_edges)
            edge_recovery_rate = recovered / len(true_edges)
        else:
            edge_recovery_rate = 0.0
        
        # False Edge Rate
        if len(inferred_edges) > 0:
            false_edges = len(inferred_edges - true_edges)
            false_edge_rate = false_edges / len(inferred_edges)
        else:
            false_edge_rate = 0.0
        
        # Node Coverage
        inferred_nodes = set(self.inferred_graph.graph.nodes())
        true_nodes = set(self.ground_truth_graph.graph.nodes())
        if len(true_nodes) > 0:
            node_coverage = len(inferred_nodes & true_nodes) / len(true_nodes)
        else:
            node_coverage = 0.0
        
        # Candidate Count Error (average across turns)
        candidate_errors = []
        for turn, estimated in self.candidate_estimates.items():
            actual = self.ground_truth_candidates.get(turn, 0)
            if actual > 0:
                error = abs(estimated - actual) / actual
                candidate_errors.append(error)
        
        candidate_count_error = (
            sum(candidate_errors) / len(candidate_errors) 
            if candidate_errors else 0.0
        )
        
        # Topology Similarity (Jaccard)
        union = inferred_edges | true_edges
        if len(union) > 0:
            topology_similarity = len(inferred_edges & true_edges) / len(union)
        else:
            topology_similarity = 0.0
        
        return {
            'edge_recovery_rate': edge_recovery_rate,
            'false_edge_rate': false_edge_rate,
            'node_coverage': node_coverage,
            'candidate_count_error': candidate_count_error,
            'topology_similarity': topology_similarity,
            'total_observations': self.total_observations,
            'inferred_edges': len(inferred_edges),
            'true_edges': len(true_edges)
        }
    
    def get_inferred_stats(self) -> Dict[str, Any]:
        """Get statistics about inferred topology."""
        user_count, item_count = self.inferred_graph.get_node_count()
        
        return {
            'inferred_users': user_count,
            'inferred_items': item_count,
            'inferred_edges': self.inferred_graph.get_edge_count(),
            'turns_observed': len(self.observations_per_turn),
            'total_observations': self.total_observations,
            'avg_candidates_estimate': (
                sum(self.candidate_estimates.values()) / len(self.candidate_estimates)
                if self.candidate_estimates else 0.0
            )
        }
    
    def reset(self) -> None:
        """Reset inferred state."""
        self.inferred_graph.clear()
        self.candidate_estimates.clear()
        self.confidence_scores.clear()
        self.observations_per_turn.clear()
        self.total_observations = 0
