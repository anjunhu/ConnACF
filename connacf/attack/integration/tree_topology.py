"""
TREE Topology for ConnaCF

A hub-and-spoke architecture where:
1. A stateful TreeManager learns from all interactions
2. Agents have profiles but NO direct edges between them
3. Recruitment uses FROZEN history-based similarity (pre-attack snapshot)
4. Attack propagation is slow because:
   - Attackers only influence users with genuinely similar history
   - No cross-agent propagation in backward pass
   - Manager shares patterns only with similar-history users

This provides a controlled comparison against standard ConnaCF topologies
(sparse/pairwise/dense) while keeping all other variables constant.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TreeManagerMemory:
    """
    Stateful memory for the TreeManager.
    
    Stores learned patterns segmented by user history clusters.
    Patterns are only shared with users who have similar history.
    """
    # Interaction outcomes: (user_id, item_id, was_correct, turn)
    interaction_history: List[Tuple[int, int, bool, int]] = field(default_factory=list)
    
    # Learned patterns per history cluster
    # cluster_id -> list of (pattern_text, confidence, source_users)
    cluster_patterns: Dict[int, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    
    # Item performance tracking: item_id -> {shown: int, correct: int}
    item_performance: Dict[int, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'shown': 0, 'correct': 0}))
    
    # User preference signals: user_id -> list of (item_id, liked: bool)
    user_signals: Dict[int, List[Tuple[int, bool]]] = field(default_factory=lambda: defaultdict(list))


class TreeTopologyManager:
    """
    Stateful manager for TREE topology in ConnaCF.
    
    Key responsibilities:
    1. Maintain FROZEN similarity matrix (computed once from initial history)
    2. Recruit similar users/items based on frozen similarity
    3. Learn patterns from interactions (stateful)
    4. Share patterns only with similar-history users
    
    Attack defense properties:
    - Attackers can't manipulate recruitment (frozen embeddings)
    - Attackers only reach users with genuinely similar history
    - No direct agent-to-agent propagation
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_history: Dict[int, List[int]],  # user_id -> list of item_ids
        neighbor_count: int = 3,
        history_item_count: int = 3,
        n_clusters: int = 10
    ):
        """
        Initialize TreeTopologyManager.
        
        Args:
            n_users: Total number of users
            n_items: Total number of items
            user_history: Pre-attack interaction history (FROZEN)
            neighbor_count: Number of similar users to recruit
            history_item_count: Number of relevant items to recruit
            n_clusters: Number of history clusters for pattern segmentation
        """
        self.n_users = n_users
        self.n_items = n_items
        self.neighbor_count = neighbor_count
        self.history_item_count = history_item_count
        self.n_clusters = n_clusters
        
        # FROZEN history snapshot (never updated)
        self.frozen_user_history = {k: list(v) for k, v in user_history.items()}
        
        # Compute FROZEN similarity matrices
        print("[TREE] Computing frozen similarity matrices...")
        self.frozen_user_similarity = self._compute_user_similarity(user_history)
        self.frozen_item_cooccurrence = self._compute_item_cooccurrence(user_history)
        
        # Assign users to history clusters (FROZEN)
        self.user_clusters = self._cluster_users_by_history(user_history)
        print(f"[TREE] Assigned {n_users} users to {len(set(self.user_clusters.values()))} clusters")
        
        # Stateful manager memory
        self.memory = TreeManagerMemory()
        
        # Turn counter
        self.current_turn = 0
        
        print(f"[TREE] TreeTopologyManager initialized: {n_users} users, {n_items} items")
    
    def _compute_user_similarity(self, user_history: Dict[int, List[int]]) -> np.ndarray:
        """
        Compute user-user similarity based on history overlap (Jaccard).
        
        This is FROZEN at initialization - attackers cannot manipulate it.
        """
        similarity = np.zeros((self.n_users, self.n_users))
        
        # Convert histories to sets for efficient intersection
        history_sets = {u: set(items) for u, items in user_history.items()}
        
        for u1 in range(self.n_users):
            set1 = history_sets.get(u1, set())
            if not set1:
                continue
            for u2 in range(u1 + 1, self.n_users):
                set2 = history_sets.get(u2, set())
                if not set2:
                    continue
                
                # Jaccard similarity
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                if union > 0:
                    sim = intersection / union
                    similarity[u1, u2] = sim
                    similarity[u2, u1] = sim
        
        return similarity
    
    def _compute_item_cooccurrence(self, user_history: Dict[int, List[int]]) -> np.ndarray:
        """
        Compute item-item co-occurrence matrix.
        
        Items that appear together in user histories are considered similar.
        """
        cooccurrence = np.zeros((self.n_items, self.n_items))
        
        for user_id, items in user_history.items():
            item_set = set(items)
            for i1 in item_set:
                if i1 >= self.n_items:
                    continue
                for i2 in item_set:
                    if i2 >= self.n_items or i1 == i2:
                        continue
                    cooccurrence[i1, i2] += 1
        
        # Normalize by max
        max_val = cooccurrence.max()
        if max_val > 0:
            cooccurrence /= max_val
        
        return cooccurrence
    
    def _cluster_users_by_history(self, user_history: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Assign users to clusters based on their history patterns.
        
        Simple approach: cluster by most common items in history.
        """
        user_clusters = {}
        
        # Count item frequencies across all users
        item_counts = defaultdict(int)
        for items in user_history.values():
            for item in items:
                item_counts[item] += 1
        
        # Get top items as cluster anchors
        top_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)[:self.n_clusters]
        
        for user_id in range(self.n_users):
            history = set(user_history.get(user_id, []))
            if not history:
                user_clusters[user_id] = 0  # Default cluster
                continue
            
            # Assign to cluster based on which top item appears most in history
            best_cluster = 0
            best_overlap = 0
            for cluster_id, anchor_item in enumerate(top_items):
                if anchor_item in history:
                    # Count how many items from this user's history co-occur with anchor
                    overlap = sum(1 for item in history if self.frozen_item_cooccurrence[anchor_item, item] > 0)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cluster = cluster_id
            
            user_clusters[user_id] = best_cluster
        
        return user_clusters
    
    def get_similar_users(self, target_user_id: int, exclude: Set[int] = None) -> List[int]:
        """
        Get users similar to target based on FROZEN history similarity.
        
        Attackers cannot manipulate this - it's based on pre-attack history.
        
        Args:
            target_user_id: The user to find neighbors for
            exclude: Set of user IDs to exclude (e.g., attackers to skip)
            
        Returns:
            List of similar user IDs (up to neighbor_count)
        """
        exclude = exclude or set()
        
        # Get similarity scores from FROZEN matrix
        similarities = self.frozen_user_similarity[target_user_id].copy()
        
        # Exclude self and specified users
        similarities[target_user_id] = -1
        for uid in exclude:
            if uid < len(similarities):
                similarities[uid] = -1
        
        # Get top-k similar users
        top_indices = np.argsort(similarities)[::-1][:self.neighbor_count * 2]  # Get extra in case of filtering
        
        similar_users = []
        for idx in top_indices:
            if similarities[idx] > 0 and idx not in exclude:
                similar_users.append(int(idx))
                if len(similar_users) >= self.neighbor_count:
                    break
        
        return similar_users
    
    def get_relevant_items(self, target_user_id: int, candidate_items: List[int]) -> List[int]:
        """
        Get items from user's history relevant to candidates.
        
        Based on FROZEN co-occurrence matrix.
        """
        user_history = self.frozen_user_history.get(target_user_id, [])
        if not user_history:
            return []
        
        # Score history items by co-occurrence with candidates
        item_scores = {}
        for hist_item in user_history:
            if hist_item >= self.n_items:
                continue
            score = sum(
                self.frozen_item_cooccurrence[hist_item, cand] 
                for cand in candidate_items 
                if cand < self.n_items
            )
            item_scores[hist_item] = score
        
        # Return top items
        sorted_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)
        return sorted_items[:self.history_item_count]
    
    def get_user_cluster(self, user_id: int) -> int:
        """Get the history cluster for a user."""
        return self.user_clusters.get(user_id, 0)
    
    def record_interaction(
        self,
        user_id: int,
        item_id: int,
        was_correct: bool,
        user_profile: str = None,
        item_profile: str = None
    ):
        """
        Record an interaction outcome in manager memory.
        
        This is how the manager learns - but patterns are only shared
        with users in the same history cluster.
        """
        self.memory.interaction_history.append((user_id, item_id, was_correct, self.current_turn))
        
        # Update item performance
        self.memory.item_performance[item_id]['shown'] += 1
        if was_correct:
            self.memory.item_performance[item_id]['correct'] += 1
        
        # Update user signals
        self.memory.user_signals[user_id].append((item_id, was_correct))
        
        # Learn pattern for user's cluster
        if user_profile and was_correct:
            cluster = self.get_user_cluster(user_id)
            pattern = {
                'text': f"Users like {user_profile[:100]} tend to enjoy item {item_id}",
                'confidence': 0.5,
                'source_users': {user_id},
                'turn': self.current_turn
            }
            self.memory.cluster_patterns[cluster].append(pattern)
    
    def get_patterns_for_user(self, user_id: int, max_patterns: int = 5) -> List[str]:
        """
        Get learned patterns relevant to this user.
        
        Only returns patterns from the user's history cluster.
        """
        cluster = self.get_user_cluster(user_id)
        patterns = self.memory.cluster_patterns.get(cluster, [])
        
        # Sort by confidence and recency
        sorted_patterns = sorted(
            patterns,
            key=lambda p: (p['confidence'], p['turn']),
            reverse=True
        )
        
        return [p['text'] for p in sorted_patterns[:max_patterns]]
    
    def get_item_performance(self, item_id: int) -> float:
        """Get success rate for an item."""
        perf = self.memory.item_performance.get(item_id, {'shown': 0, 'correct': 0})
        if perf['shown'] == 0:
            return 0.5  # Default
        return perf['correct'] / perf['shown']
    
    def advance_turn(self):
        """Advance to next turn."""
        self.current_turn += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'total_interactions': len(self.memory.interaction_history),
            'patterns_per_cluster': {
                c: len(patterns) for c, patterns in self.memory.cluster_patterns.items()
            },
            'items_tracked': len(self.memory.item_performance),
            'current_turn': self.current_turn
        }



class TreeTopologyIntegration:
    """
    Integration layer for TREE topology with ConnaCF attack framework.
    
    Wraps the existing ConnaCFAttackIntegration to add TREE-specific behavior:
    1. Use TreeTopologyManager for recruitment
    2. Disable cross-agent propagation in backward pass
    3. Route learning through manager instead of direct edges
    """
    
    def __init__(
        self,
        base_integration,  # ConnaCFAttackIntegration instance
        neighbor_count: int = 3,
        history_item_count: int = 3
    ):
        """
        Initialize TREE topology integration.
        
        Args:
            base_integration: The underlying ConnaCFAttackIntegration
            neighbor_count: Number of similar users to recruit
            history_item_count: Number of relevant items to recruit
        """
        self.base = base_integration
        self.neighbor_count = neighbor_count
        self.history_item_count = history_item_count
        
        # Extract user history from training data
        user_history = self._extract_user_history()
        
        # Initialize TreeTopologyManager
        self.tree_manager = TreeTopologyManager(
            n_users=len(self.base.user_agents),
            n_items=len(self.base.item_agents),
            user_history=user_history,
            neighbor_count=neighbor_count,
            history_item_count=history_item_count
        )
        
        print(f"[TREE] TreeTopologyIntegration initialized")
    
    def _extract_user_history(self) -> Dict[int, List[int]]:
        """Extract user interaction history from training data."""
        user_history = defaultdict(list)
        
        # Try to get from dataset
        if hasattr(self.base, 'train_data') and self.base.train_data is not None:
            # RecBole dataset format
            try:
                inter = self.base.train_data.inter_feat
                user_ids = inter['user_id'].numpy() if hasattr(inter['user_id'], 'numpy') else inter['user_id']
                item_ids = inter['item_id'].numpy() if hasattr(inter['item_id'], 'numpy') else inter['item_id']
                
                for uid, iid in zip(user_ids, item_ids):
                    user_history[int(uid)].append(int(iid))
            except Exception as e:
                print(f"[TREE] Warning: Could not extract history from train_data: {e}")
        
        # Fallback: use agent historical_interactions if available
        if not user_history:
            for uid, agent in self.base.user_agents.items():
                if hasattr(agent, 'historical_interactions'):
                    user_history[uid] = list(agent.historical_interactions.keys())
        
        print(f"[TREE] Extracted history for {len(user_history)} users")
        return dict(user_history)
    
    def recruit_for_batch(
        self,
        batch_user: List[int],
        candidate_items: List[List[int]],
        attacker_user_indices: Set[int] = None
    ) -> Dict[int, Dict[str, List[int]]]:
        """
        Recruit similar users and items for each user in batch.
        
        Uses FROZEN similarity - attackers cannot manipulate recruitment.
        
        Args:
            batch_user: List of target user IDs
            candidate_items: List of candidate item lists per user
            attacker_user_indices: Set of attacker user IDs to potentially exclude
            
        Returns:
            Dict mapping user_id -> {'similar_users': [...], 'relevant_items': [...]}
        """
        attacker_user_indices = attacker_user_indices or set()
        recruitment = {}
        
        for i, user_id in enumerate(batch_user):
            # Get similar users (excluding attackers from recruitment pool)
            # Note: We DON'T exclude attackers - they can be recruited if genuinely similar
            # This is the key insight: attackers only influence similar-history users
            similar_users = self.tree_manager.get_similar_users(
                target_user_id=user_id,
                exclude={user_id}  # Only exclude self
            )
            
            # Get relevant items from user's history
            items = candidate_items[i] if i < len(candidate_items) else []
            relevant_items = self.tree_manager.get_relevant_items(
                target_user_id=user_id,
                candidate_items=items
            )
            
            recruitment[user_id] = {
                'similar_users': similar_users,
                'relevant_items': relevant_items,
                'cluster': self.tree_manager.get_user_cluster(user_id)
            }
        
        return recruitment
    
    def aggregate_opinions(
        self,
        target_user_id: int,
        similar_users: List[int],
        candidate_items: List[int]
    ) -> str:
        """
        Aggregate opinions from recruited users.
        
        This is where attacker influence is naturally limited:
        - Attacker is just one voice among similar users
        - If attacker's opinion is outlier, it gets diluted
        """
        opinions = []
        
        for uid in similar_users:
            agent = self.base.user_agents.get(uid)
            if agent:
                profile = agent.update_memory[-1] if agent.update_memory else ""
                is_attacker = uid in self.base.interaction_controller.attacker_user_indices
                opinions.append({
                    'user_id': uid,
                    'profile': profile[:200],  # Truncate for context
                    'is_attacker': is_attacker
                })
        
        # Also get manager's learned patterns for this user's cluster
        patterns = self.tree_manager.get_patterns_for_user(target_user_id)
        
        # Format aggregated context
        context_parts = []
        for op in opinions:
            marker = "[ATK]" if op['is_attacker'] else ""
            context_parts.append(f"User {op['user_id']}{marker}: {op['profile']}")
        
        if patterns:
            context_parts.append(f"Learned patterns: {'; '.join(patterns[:3])}")
        
        return "\n".join(context_parts)
    
    def record_batch_outcomes(
        self,
        batch_user: List[int],
        candidate_items: List[List[int]],
        accuracy: List[int]
    ):
        """
        Record batch outcomes in manager memory.
        
        Manager learns from all interactions, but patterns are segmented by cluster.
        """
        for i, user_id in enumerate(batch_user):
            items = candidate_items[i] if i < len(candidate_items) else []
            was_correct = accuracy[i] == 1 if i < len(accuracy) else False
            
            for item_id in items[:1]:  # Record for primary item
                user_profile = ""
                if user_id in self.base.user_agents:
                    agent = self.base.user_agents[user_id]
                    user_profile = agent.update_memory[-1] if agent.update_memory else ""
                
                self.tree_manager.record_interaction(
                    user_id=user_id,
                    item_id=item_id,
                    was_correct=was_correct,
                    user_profile=user_profile
                )
        
        self.tree_manager.advance_turn()
    
    def backward_pass_tree(
        self,
        batch_user: List[int],
        accuracy: List[int],
        system_reasons: List[str]
    ):
        """
        TREE-specific backward pass.
        
        Key difference from standard ConnaCF:
        - Agents update their OWN profiles based on feedback
        - NO cross-agent propagation (no U-U or I-I edges)
        - Learning happens through manager memory instead
        """
        for i, user_id in enumerate(batch_user):
            if user_id in self.base.interaction_controller.attacker_user_indices:
                continue  # Attackers don't update
            
            was_correct = accuracy[i] == 1 if i < len(accuracy) else False
            reason = system_reasons[i] if i < len(system_reasons) else ""
            
            # Agent updates its own profile (standard backward)
            # But we skip the similarity-based propagation to neighbors
            agent = self.base.user_agents.get(user_id)
            if agent and hasattr(agent, 'update_memory'):
                # The actual profile update happens in the base backward pass
                # We just ensure no cross-propagation by not calling propagate methods
                pass
        
        # Record outcomes in manager (this is how TREE learns globally)
        # Patterns will be shared with similar-history users in future forward passes


def create_tree_topology_config() -> Dict[str, Any]:
    """Create default configuration for TREE topology."""
    return {
        'topology': 'tree',
        'neighbor_count': 3,
        'history_item_count': 3,
        'cross_agent_propagation': False,  # Key difference from other topologies
        'manager_memory_enabled': True,
        'frozen_recruitment': True,  # Use pre-attack similarity
    }
