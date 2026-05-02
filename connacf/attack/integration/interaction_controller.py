"""
Interaction Controller for ConnaCF Attack Framework

Controls agent interaction patterns within ConnaCF's existing structure.
Instead of arbitrary graph topologies, works with natural interaction patterns:
- User similarity-based interactions
- Item co-occurrence patterns  
- System agent centralized influence
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class InteractionController:
    """Controls agent interaction patterns for attack propagation"""
    
    def __init__(self, config: Dict, dataset, n_users: int, n_items: int):
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        
        # Attack configuration
        self.interaction_density = config.get('interaction_density', 0.3)  # 0.0 to 1.0
        self.interaction_pattern = config.get('interaction_pattern', 'similarity')  # similarity, random, clustered
        self.max_similar_users = config.get('max_similar_users', 5)
        self.max_similar_items = config.get('max_similar_items', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.5)
        
        # Attacker configuration
        self.attacker_user_indices = set(config.get('attacker_user_indices', []))
        self.attacker_item_indices = set(config.get('attacker_item_indices', []))
        self.system_agent_compromised = config.get('system_agent_compromised', False)
        
        # Interaction masks (computed lazily)
        self._user_similarity_cache = {}
        self._item_similarity_cache = {}
        self._interaction_history = defaultdict(list)
        
    def should_enable_user_influence(self, user_id: int) -> bool:
        """Determine if user should be influenced by other users"""
        if user_id in self.attacker_user_indices:
            return False  # Attackers don't get influenced
        
        # Control interaction density
        return np.random.random() < self.interaction_density
    
    def should_enable_item_influence(self, item_id: int) -> bool:
        """Determine if item should be influenced by other items"""
        if item_id in self.attacker_item_indices:
            return False  # Attacker items don't get influenced
            
        return np.random.random() < self.interaction_density
    
    def find_similar_users(self, user_id: int, user_embeddings: torch.Tensor) -> List[int]:
        """Find similar users based on embedding similarity"""
        if user_id in self._user_similarity_cache:
            return self._user_similarity_cache[user_id]
        
        if self.interaction_pattern == 'similarity':
            similar_users = self._find_similar_users_by_embedding(user_id, user_embeddings)
        elif self.interaction_pattern == 'random':
            similar_users = self._find_random_users(user_id)
        elif self.interaction_pattern == 'clustered':
            similar_users = self._find_clustered_users(user_id)
        else:
            similar_users = []
        
        self._user_similarity_cache[user_id] = similar_users
        return similar_users
    
    def find_similar_items(self, item_id: int, item_embeddings: torch.Tensor) -> List[int]:
        """Find similar items based on embedding similarity"""
        if item_id in self._item_similarity_cache:
            return self._item_similarity_cache[item_id]
        
        if self.interaction_pattern == 'similarity':
            similar_items = self._find_similar_items_by_embedding(item_id, item_embeddings)
        elif self.interaction_pattern == 'random':
            similar_items = self._find_random_items(item_id)
        elif self.interaction_pattern == 'clustered':
            similar_items = self._find_clustered_items(item_id)
        else:
            similar_items = []
        
        self._item_similarity_cache[item_id] = similar_items
        return similar_items
    
    def _find_similar_users_by_embedding(self, user_id: int, user_embeddings: torch.Tensor) -> List[int]:
        """Find users with similar embeddings"""
        user_embedding = user_embeddings[user_id].unsqueeze(0)
        similarities = torch.cosine_similarity(user_embedding, user_embeddings)
        
        # Filter by threshold and exclude self
        valid_indices = torch.where(
            (similarities > self.similarity_threshold) & 
            (torch.arange(len(similarities), device=similarities.device) != user_id)
        )[0]
        
        # Get top-k most similar
        if len(valid_indices) > self.max_similar_users:
            top_similarities = similarities[valid_indices]
            top_k_indices = torch.topk(top_similarities, k=self.max_similar_users).indices
            similar_users = valid_indices[top_k_indices].tolist()
        else:
            similar_users = valid_indices.tolist()
        
        return similar_users
    
    def _find_similar_items_by_embedding(self, item_id: int, item_embeddings: torch.Tensor) -> List[int]:
        """Find items with similar embeddings"""
        item_embedding = item_embeddings[item_id].unsqueeze(0)
        similarities = torch.cosine_similarity(item_embedding, item_embeddings)
        
        # Filter by threshold and exclude self
        valid_indices = torch.where(
            (similarities > self.similarity_threshold) & 
            (torch.arange(len(similarities), device=similarities.device) != item_id)
        )[0]
        
        # Get top-k most similar
        if len(valid_indices) > self.max_similar_items:
            top_similarities = similarities[valid_indices]
            top_k_indices = torch.topk(top_similarities, k=self.max_similar_items).indices
            similar_items = valid_indices[top_k_indices].tolist()
        else:
            similar_items = valid_indices.tolist()
        
        return similar_items
    
    def _find_random_users(self, user_id: int) -> List[int]:
        """Find random users for interaction"""
        all_users = list(range(self.n_users))
        all_users.remove(user_id)  # Exclude self
        
        num_to_select = min(self.max_similar_users, len(all_users))
        return np.random.choice(all_users, size=num_to_select, replace=False).tolist()
    
    def _find_random_items(self, item_id: int) -> List[int]:
        """Find random items for interaction"""
        all_items = list(range(self.n_items))
        all_items.remove(item_id)  # Exclude self
        
        num_to_select = min(self.max_similar_items, len(all_items))
        return np.random.choice(all_items, size=num_to_select, replace=False).tolist()
    
    def _find_clustered_users(self, user_id: int) -> List[int]:
        """Find users in same cluster (simplified clustering)"""
        # Simple clustering based on user_id ranges
        cluster_size = max(1, self.n_users // 10)  # 10 clusters
        cluster_id = user_id // cluster_size
        
        cluster_start = cluster_id * cluster_size
        cluster_end = min((cluster_id + 1) * cluster_size, self.n_users)
        
        cluster_users = list(range(cluster_start, cluster_end))
        cluster_users.remove(user_id)  # Exclude self
        
        num_to_select = min(self.max_similar_users, len(cluster_users))
        if num_to_select > 0:
            return np.random.choice(cluster_users, size=num_to_select, replace=False).tolist()
        return []
    
    def _find_clustered_items(self, item_id: int) -> List[int]:
        """Find items in same cluster (simplified clustering)"""
        # Simple clustering based on item_id ranges
        cluster_size = max(1, self.n_items // 10)  # 10 clusters
        cluster_id = item_id // cluster_size
        
        cluster_start = cluster_id * cluster_size
        cluster_end = min((cluster_id + 1) * cluster_size, self.n_items)
        
        cluster_items = list(range(cluster_start, cluster_end))
        cluster_items.remove(item_id)  # Exclude self
        
        num_to_select = min(self.max_similar_items, len(cluster_items))
        if num_to_select > 0:
            return np.random.choice(cluster_items, size=num_to_select, replace=False).tolist()
        return []
    
    def record_interaction(self, agent1_id: int, agent2_id: int, agent_type: str, round_num: int):
        """Record agent interaction for analysis.
        
        Args:
            agent1_id: First agent ID (user for user-item, user1 for user-user, item1 for item-item)
            agent2_id: Second agent ID (item for user-item, user2 for user-user, item2 for item-item)
            agent_type: Type of interaction - 'user' (user-user), 'item' (item-item), or 'user_item' (user-item)
            round_num: The round/turn number
        """
        # Determine attacker status based on interaction type
        if agent_type == 'user_item':
            # User-item interaction: agent1 is user, agent2 is item
            agent1_is_attacker = agent1_id in self.attacker_user_indices
            agent2_is_attacker = agent2_id in self.attacker_item_indices
        elif agent_type == 'user':
            # User-user interaction
            agent1_is_attacker = agent1_id in self.attacker_user_indices
            agent2_is_attacker = agent2_id in self.attacker_user_indices
        else:  # 'item'
            # Item-item interaction
            agent1_is_attacker = agent1_id in self.attacker_item_indices
            agent2_is_attacker = agent2_id in self.attacker_item_indices
        
        interaction = {
            'round': round_num,
            'agent1': agent1_id,
            'agent2': agent2_id,
            'type': agent_type,
            'agent1_is_attacker': agent1_is_attacker,
            'agent2_is_attacker': agent2_is_attacker
        }
        self._interaction_history[round_num].append(interaction)
    
    def get_interaction_statistics(self) -> Dict:
        """Get statistics about agent interactions"""
        total_interactions = sum(len(interactions) for interactions in self._interaction_history.values())
        attacker_interactions = 0
        
        for interactions in self._interaction_history.values():
            for interaction in interactions:
                if interaction['agent1_is_attacker'] or interaction['agent2_is_attacker']:
                    attacker_interactions += 1
        
        return {
            'total_interactions': total_interactions,
            'attacker_interactions': attacker_interactions,
            'attacker_interaction_ratio': attacker_interactions / max(1, total_interactions),
            'rounds_recorded': len(self._interaction_history),
            'avg_interactions_per_round': total_interactions / max(1, len(self._interaction_history))
        }
    
    def record_detailed_interaction(self, agent1_id: int, agent2_id: int, 
                                  interaction_details: Dict, round_num: int):
        """Record detailed interaction with contamination transfer info"""
        interaction = {
            'round': round_num,
            'agent1_id': agent1_id,
            'agent2_id': agent2_id,
            'interaction_type': interaction_details.get('type', 'unknown'),
            'contamination_transfer': interaction_details.get('contamination_transfer', 0.0),
            'influence_strength': interaction_details.get('influence_strength', 0.0),
            'timestamp': interaction_details.get('timestamp', ''),
            'agent1_is_attacker': (
                agent1_id in self.attacker_user_indices if interaction_details.get('type') == 'user'
                else agent1_id in self.attacker_item_indices
            ),
            'agent2_is_attacker': (
                agent2_id in self.attacker_user_indices if interaction_details.get('type') == 'user'
                else agent2_id in self.attacker_item_indices
            ),
            'details': interaction_details
        }
        self._interaction_history[round_num].append(interaction)
    
    def get_round_interaction_graph(self, round_num: int) -> Dict:
        """Get interaction graph data for specific round"""
        interactions = self._interaction_history.get(round_num, [])
        
        # Build graph structure
        nodes = set()
        edges = []
        
        for interaction in interactions:
            agent1 = interaction['agent1']
            agent2 = interaction['agent2']
            
            nodes.add(agent1)
            nodes.add(agent2)
            
            edges.append({
                'source': agent1,
                'target': agent2,
                'weight': interaction.get('contamination_transfer', 0.0),
                'type': interaction.get('type', 'unknown')
            })
        
        return {
            'round': round_num,
            'nodes': list(nodes),
            'edges': edges,
            'total_interactions': len(interactions),
            'attacker_interactions': len([i for i in interactions 
                                        if i['agent1_is_attacker'] or i['agent2_is_attacker']])
        }
    
    def calculate_contamination_flow(self, round_num: int) -> Dict:
        """Calculate how contamination flows between agents in round"""
        interactions = self._interaction_history.get(round_num, [])
        
        contamination_flow = {
            'total_flow': 0.0,
            'attacker_to_clean': 0.0,
            'clean_to_clean': 0.0,
            'flow_by_type': defaultdict(float),
            'top_spreaders': defaultdict(float),
            'top_receivers': defaultdict(float)
        }
        
        for interaction in interactions:
            transfer = interaction.get('contamination_transfer', 0.0)
            agent1_attacker = interaction.get('agent1_is_attacker', False)
            agent2_attacker = interaction.get('agent2_is_attacker', False)
            interaction_type = interaction.get('type', 'unknown')
            
            contamination_flow['total_flow'] += transfer
            contamination_flow['flow_by_type'][interaction_type] += transfer
            
            # Track spreader and receiver
            contamination_flow['top_spreaders'][interaction['agent1']] += transfer
            contamination_flow['top_receivers'][interaction['agent2']] += transfer
            
            # Categorize flow
            if agent1_attacker and not agent2_attacker:
                contamination_flow['attacker_to_clean'] += transfer
            elif not agent1_attacker and not agent2_attacker:
                contamination_flow['clean_to_clean'] += transfer
        
        # Convert defaultdicts to regular dicts and sort
        contamination_flow['flow_by_type'] = dict(contamination_flow['flow_by_type'])
        contamination_flow['top_spreaders'] = dict(sorted(
            contamination_flow['top_spreaders'].items(), 
            key=lambda x: x[1], reverse=True
        )[:10])
        contamination_flow['top_receivers'] = dict(sorted(
            contamination_flow['top_receivers'].items(), 
            key=lambda x: x[1], reverse=True
        )[:10])
        
        return contamination_flow
    
    def clear_cache(self):
        """Clear similarity caches (call when embeddings change significantly)"""
        self._user_similarity_cache.clear()
        self._item_similarity_cache.clear()