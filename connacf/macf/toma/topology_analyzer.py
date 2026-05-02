"""
TOMA Topology Analyzer Module.

Constructs and analyzes the bipartite User↔Item interaction graph for
topology-aware attack targeting.

Key Components:
- BipartiteGraph: Graph representation with temporal edge tracking
- UserCharacteristics: Data class for user analysis
- TopologyAnalyzer: Bridge item identification and user characteristic analysis
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

try:
    import networkx as nx
except ImportError:
    nx = None  # Will raise error if used without networkx

if TYPE_CHECKING:
    from connacf.macf.agents import UserAgent
    from connacf.macf.memory_store import MACFMemoryStore

logger = logging.getLogger(__name__)


@dataclass
class UserCharacteristics:
    """
    Characteristics used to estimate retention probability.
    
    Users with sparse history are flagged as "weak reflection" targets
    (more likely to preserve injected content verbatim).
    """
    user_id: int
    history_density: int = 0       # Number of words in history summary
    memory_size: int = 0           # Profile length + preference count
    interaction_recency: int = 0   # Turns since last activity
    preference_count: int = 0      # Number of learned preferences
    profile_length: int = 0        # Length of profile text


class BipartiteGraph:
    """
    Bipartite graph representation of User↔Item interactions.
    
    Nodes:
    - User nodes: 'u_{user_id}'
    - Item nodes: 'i_{item_id}'
    
    Edges:
    - (user_node, item_node) with attributes:
      - turn: Turn number when interaction occurred
      - interaction_type: 'positive', 'negative', 'candidate'
    """
    
    def __init__(self):
        if nx is None:
            raise ImportError("networkx is required for TOMA topology analysis")
        
        self.graph = nx.Graph()
        self.user_nodes: Set[str] = set()
        self.item_nodes: Set[str] = set()
        self.turn_edges: Dict[int, List[Tuple[str, str]]] = {}  # turn -> edges
        self.node_first_seen: Dict[str, int] = {}  # node -> first turn seen
    
    def add_interaction(
        self, 
        user_id: int, 
        item_id: int, 
        turn: int,
        interaction_type: str = 'candidate'
    ) -> None:
        """
        Add user-item interaction edge with temporal info.
        
        Args:
            user_id: User ID
            item_id: Item ID
            turn: Turn number when interaction occurred
            interaction_type: Type of interaction ('positive', 'negative', 'candidate')
        """
        u_node = f'u_{user_id}'
        i_node = f'i_{item_id}'
        
        # Track node sets
        self.user_nodes.add(u_node)
        self.item_nodes.add(i_node)
        
        # Track first seen turn
        if u_node not in self.node_first_seen:
            self.node_first_seen[u_node] = turn
        if i_node not in self.node_first_seen:
            self.node_first_seen[i_node] = turn
        
        # Add or update edge
        if self.graph.has_edge(u_node, i_node):
            # Update with latest turn
            self.graph[u_node][i_node]['turns'].append(turn)
            self.graph[u_node][i_node]['interaction_types'].append(interaction_type)
        else:
            self.graph.add_edge(
                u_node, i_node,
                turns=[turn],
                interaction_types=[interaction_type],
                first_turn=turn
            )
        
        # Track edges by turn
        if turn not in self.turn_edges:
            self.turn_edges[turn] = []
        self.turn_edges[turn].append((u_node, i_node))
    
    def identify_bridge_items(self) -> List[Tuple[int, int]]:
        """
        Identify bridge items (item-centric hubs).
        
        Bridge items are items that connect multiple users across turns.
        They serve as propagation hubs in the bipartite graph.
        
        Returns:
            List of (item_id, user_degree) sorted by degree descending.
            Items with degree >= 2 are considered bridges.
        """
        bridge_items = []
        
        for item_node in self.item_nodes:
            # Count distinct users connected to this item
            neighbors = list(self.graph.neighbors(item_node))
            user_neighbors = [n for n in neighbors if n.startswith('u_')]
            user_degree = len(user_neighbors)
            
            if user_degree >= 2:  # Bridge threshold
                item_id = int(item_node.split('_')[1])
                bridge_items.append((item_id, user_degree))
        
        # Sort by degree (highest connectivity first)
        return sorted(bridge_items, key=lambda x: x[1], reverse=True)
    
    def get_user_degree(self, user_id: int) -> int:
        """Get number of items a user has interacted with."""
        u_node = f'u_{user_id}'
        if u_node not in self.graph:
            return 0
        return len([n for n in self.graph.neighbors(u_node) if n.startswith('i_')])
    
    def get_item_degree(self, item_id: int) -> int:
        """Get number of users that have interacted with an item."""
        i_node = f'i_{item_id}'
        if i_node not in self.graph:
            return 0
        return len([n for n in self.graph.neighbors(i_node) if n.startswith('u_')])
    
    def get_users_for_item(self, item_id: int) -> List[int]:
        """Get all users that have interacted with an item."""
        i_node = f'i_{item_id}'
        if i_node not in self.graph:
            return []
        neighbors = self.graph.neighbors(i_node)
        return [int(n.split('_')[1]) for n in neighbors if n.startswith('u_')]
    
    def get_items_for_user(self, user_id: int) -> List[int]:
        """Get all items a user has interacted with."""
        u_node = f'u_{user_id}'
        if u_node not in self.graph:
            return []
        neighbors = self.graph.neighbors(u_node)
        return [int(n.split('_')[1]) for n in neighbors if n.startswith('i_')]
    
    def get_node_count(self) -> Tuple[int, int]:
        """Get (user_count, item_count)."""
        return len(self.user_nodes), len(self.item_nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()
    
    def clear(self) -> None:
        """Clear the graph."""
        self.graph.clear()
        self.user_nodes.clear()
        self.item_nodes.clear()
        self.turn_edges.clear()
        self.node_first_seen.clear()


class TopologyAnalyzer:
    """
    Analyzes the bipartite User↔Item topology for attack targeting.
    
    Responsibilities:
    - Build and maintain the bipartite interaction graph
    - Identify bridge items (item-centric hubs)
    - Analyze user characteristics for retention estimation
    """
    
    def __init__(self):
        self.graph = BipartiteGraph()
        self.bridge_items: List[Tuple[int, int]] = []
        self.user_characteristics: Dict[int, UserCharacteristics] = {}
        self.current_turn: int = 0
    
    def build_graph_from_history(
        self,
        interaction_history: List[Tuple[int, int, int]]
    ) -> None:
        """
        Build bipartite graph from interaction history.
        
        Args:
            interaction_history: List of (user_id, item_id, turn) tuples
        """
        for user_id, item_id, turn in interaction_history:
            self.graph.add_interaction(user_id, item_id, turn)
            self.current_turn = max(self.current_turn, turn)
        
        user_count, item_count = self.graph.get_node_count()
        edge_count = self.graph.get_edge_count()
        logger.info(f"Built bipartite graph: {user_count} users, {item_count} items, "
                   f"{edge_count} edges")
    
    def add_interaction(
        self,
        user_id: int,
        item_id: int,
        turn: int,
        interaction_type: str = 'candidate'
    ) -> None:
        """Add a single interaction to the graph."""
        self.graph.add_interaction(user_id, item_id, turn, interaction_type)
        self.current_turn = max(self.current_turn, turn)
    
    def identify_bridge_items(self) -> List[Tuple[int, int]]:
        """
        Identify bridge items and cache the result.
        
        Returns:
            List of (item_id, user_degree) sorted by degree descending.
        """
        self.bridge_items = self.graph.identify_bridge_items()
        logger.info(f"Identified {len(self.bridge_items)} bridge items "
                   f"(total items: {len(self.graph.item_nodes)}, "
                   f"total users: {len(self.graph.user_nodes)}, "
                   f"edges: {self.graph.get_edge_count()})")
        if self.bridge_items:
            top_bridges = self.bridge_items[:5]
            logger.info(f"Top 5 bridge items: {top_bridges}")
        return self.bridge_items
    
    def analyze_user_characteristics(
        self,
        user_agents: List['UserAgent'],
        memory_store: Optional['MACFMemoryStore'] = None
    ) -> Dict[int, UserCharacteristics]:
        """
        Extract characteristics for all users to estimate retention probability.
        
        Users with sparse history are flagged as "weak reflection" targets
        (more likely to preserve injected content verbatim).
        
        Args:
            user_agents: List of UserAgent instances
            memory_store: Optional MACFMemoryStore for additional memory info
        
        Returns:
            Dict mapping user_id -> UserCharacteristics
        """
        self.user_characteristics = {}
        
        for user_agent in user_agents:
            # Handle both neighbor_user_id (ConnaCF) and user_id (fallback)
            user_id = getattr(user_agent, 'neighbor_user_id', getattr(user_agent, 'user_id', None))
            if user_id is None:
                logger.warning(f"User agent has no user_id or neighbor_user_id, skipping")
                continue
            
            # Extract history density (words in history summary)
            history_summary = getattr(user_agent, 'neighbor_history_summary', '') or ''
            history_density = len(history_summary.split())
            
            # Extract memory info
            profile_length = 0
            preference_count = 0
            
            if hasattr(user_agent, 'memory') and user_agent.memory:
                profile = getattr(user_agent.memory, 'profile', '') or ''
                profile_length = len(profile)
                
                preferences = getattr(user_agent.memory, 'preferences', []) or []
                preference_count = len(preferences)
            
            # Try memory store as fallback
            if memory_store and (profile_length == 0 or preference_count == 0):
                try:
                    memory = memory_store.get_user_memory(user_id)
                    if profile_length == 0:
                        profile_length = len(memory.profile or '')
                    if preference_count == 0:
                        preference_count = len(memory.preferences or [])
                except Exception:
                    pass
            
            # Calculate interaction recency
            u_node = f'u_{user_id}'
            first_seen = self.graph.node_first_seen.get(u_node, self.current_turn)
            interaction_recency = self.current_turn - first_seen
            
            self.user_characteristics[user_id] = UserCharacteristics(
                user_id=user_id,
                history_density=history_density,
                memory_size=profile_length + preference_count,
                interaction_recency=interaction_recency,
                preference_count=preference_count,
                profile_length=profile_length
            )
        
        logger.info(f"Analyzed characteristics for {len(self.user_characteristics)} users")
        return self.user_characteristics
    
    def get_weak_reflection_users(
        self,
        threshold_history_density: int = 50
    ) -> List[int]:
        """
        Get users with weak reflection (sparse history).
        
        Args:
            threshold_history_density: Users below this are considered weak
        
        Returns:
            List of user IDs with weak reflection, sorted by weakness
        """
        weak_users = [
            (user_id, chars.history_density)
            for user_id, chars in self.user_characteristics.items()
            if chars.history_density < threshold_history_density
        ]
        
        # Sort by history density (lowest = weakest reflection)
        weak_users.sort(key=lambda x: x[1])
        
        return [user_id for user_id, _ in weak_users]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph."""
        user_count, item_count = self.graph.get_node_count()
        
        return {
            'user_count': user_count,
            'item_count': item_count,
            'edge_count': self.graph.get_edge_count(),
            'bridge_item_count': len(self.bridge_items),
            'current_turn': self.current_turn,
            'avg_user_degree': sum(
                self.graph.get_user_degree(int(u.split('_')[1])) 
                for u in self.graph.user_nodes
            ) / max(user_count, 1),
            'avg_item_degree': sum(
                self.graph.get_item_degree(int(i.split('_')[1]))
                for i in self.graph.item_nodes
            ) / max(item_count, 1)
        }
