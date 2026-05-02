#!/usr/bin/env python3
"""
Compute NetSafe-adapted metrics for ConnaCF datasets.

Metrics computed:
1. CPP (Collaborative Propagation Potential) - adapted from Network Efficiency
2. SC (Susceptibility Centrality) - adapted from Eigenvector Centrality  
3. PER (Probabilistic Exposure Risk) - adapted from Attack Path Vulnerability

For each dataset × topology configuration:
- tree: Hub-and-spoke, neighbor_count=3
- cand_1: Binary classification (k=1)
- cand_2: Pairwise comparison (k=2)
- cand_3: 3-candidate ranking (k=3)
- cand_4: Full ranking (k=4)
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import shortest_path

DATASET_DIR = Path(__file__).parent.parent / "dataset"


class DatasetMetrics:
    """Compute NetSafe-adapted metrics for a dataset."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = DATASET_DIR / dataset_name
        
        # Load data
        self.user_items = {}  # user_id -> list of item_ids (positive history)
        self.item_users = defaultdict(set)  # item_id -> set of user_ids (positive)
        self.all_users = set()
        self.all_items = set()
        
        # Negative sample data per topology
        # item_id -> count of times it appears as a negative sample
        self.neg_item_counts = {}  # {topo_name: {item_id: count}}
        self.missing_neg_files = set()  # Track which candidate files are missing
        
        # Training rows count (for 2_cand popularity estimation)
        self.train_row_count = 0
        self.train_target_counts = defaultdict(int)  # item_id -> count as target in .train.inter
        
        self._load_data()
        self._load_negative_samples()
    
    def _load_data(self):
        """Load interaction data from train/valid/test files."""
        # Find the train file
        train_file = None
        for f in self.dataset_path.iterdir():
            if f.name.endswith('.train.inter'):
                train_file = f
                break
        
        if not train_file:
            raise FileNotFoundError(f"No train.inter file in {self.dataset_path}")
        
        # Parse interactions
        user_items_temp = defaultdict(list)
        
        with open(train_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    item_list = [int(x) for x in parts[1].split()]
                    target_item = int(parts[2])
                    
                    self.train_row_count += 1
                    self.train_target_counts[target_item] += 1
                    
                    # Store the longest sequence for each user
                    full_seq = item_list + [target_item]
                    if len(full_seq) > len(user_items_temp[user_id]):
                        user_items_temp[user_id] = full_seq
        
        # Also load valid and test
        for suffix in ['.valid.inter', '.test.inter']:
            for f in self.dataset_path.iterdir():
                if f.name.endswith(suffix):
                    with open(f, 'r') as file:
                        next(file)
                        for line in file:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                user_id = int(parts[0])
                                target_item = int(parts[2])
                                if target_item not in user_items_temp[user_id]:
                                    user_items_temp[user_id].append(target_item)
        
        self.user_items = dict(user_items_temp)
        self.all_users = set(self.user_items.keys())
        
        for user_id, items in self.user_items.items():
            for item_id in items:
                self.all_items.add(item_id)
                self.item_users[item_id].add(user_id)
        
        self.item_users = dict(self.item_users)
    
    def _load_negative_samples(self):
        """Load negative sample counts from pre-generated JSON files."""
        self.neg_item_counts = {}
        
        # --- cand_1: binary_train_1cand.json ---
        binary_path = self.dataset_path / "binary_train_1cand.json"
        if binary_path.exists():
            with open(binary_path, 'r') as f:
                binary_data = json.load(f)
            counts = defaultdict(int)
            for user_samples in binary_data['data'].values():
                for sample in user_samples:
                    item_id = int(sample[0])
                    is_positive = sample[1]
                    if not is_positive:
                        counts[item_id] += 1
            self.neg_item_counts['cand_1'] = dict(counts)
        else:
            self.missing_neg_files.add('cand_1')
        
        # --- cand_3: ranking_train_3cand.json ---
        ranking3_path = self.dataset_path / "ranking_train_3cand.json"
        if ranking3_path.exists():
            with open(ranking3_path, 'r') as f:
                ranking3_data = json.load(f)
            counts = defaultdict(int)
            num_pos = 3 // 2  # 1 positive, 2 negatives
            for user_samples in ranking3_data['data'].values():
                for sample in user_samples:
                    # Format: [pos1, neg1, neg2] — last 2 are negatives
                    for neg_token in sample[num_pos:]:
                        counts[int(neg_token)] += 1
            self.neg_item_counts['cand_3'] = dict(counts)
        else:
            self.missing_neg_files.add('cand_3')

        # --- cand_4: ranking_train_4cand.json ---
        ranking_path = self.dataset_path / "ranking_train_4cand.json"
        if ranking_path.exists():
            with open(ranking_path, 'r') as f:
                ranking_data = json.load(f)
            counts = defaultdict(int)
            for user_samples in ranking_data['data'].values():
                for sample in user_samples:
                    # Format: [pos1, pos2, neg1, neg2] — last 2 are negatives
                    for neg_token in sample[2:]:
                        counts[int(neg_token)] += 1
            self.neg_item_counts['cand_4'] = dict(counts)
        else:
            self.missing_neg_files.add('cand_4')
        
        # --- cand_2: estimated from popularity distribution ---
        # RecBole samples 1 negative per training row, weighted by item popularity.
        # We estimate: E[neg_count(item)] = train_rows * (pos_degree(item) / total_pos)
        total_pos = sum(self.train_target_counts.values())
        if total_pos > 0:
            counts = {}
            for item_id in self.all_items:
                pop = self.train_target_counts.get(item_id, 0)
                # Expected negative samples for this item (popularity-proportional)
                counts[item_id] = self.train_row_count * (pop / total_pos)
            self.neg_item_counts['cand_2'] = counts
    
    def get_item_degree_stats(self, topo_name: str) -> Dict:
        """
        Compute positive, negative, and overall item degree stats for a topology.
        
        Args:
            topo_name: 'cand_1', 'cand_2', or 'cand_4'
        
        Returns:
            Dict with pos/neg/overall degree statistics
        """
        neg_counts = self.neg_item_counts.get(topo_name, {})
        
        pos_degrees = []
        neg_degrees = []
        overall_degrees = []
        
        for item_id in self.all_items:
            pos_deg = len(self.item_users.get(item_id, set()))
            neg_deg = neg_counts.get(item_id, 0)
            
            pos_degrees.append(pos_deg)
            neg_degrees.append(neg_deg)
            overall_degrees.append(pos_deg + neg_deg)
        
        if not pos_degrees:
            return {
                'pos_degree_mean': 0, 'pos_degree_median': 0,
                'neg_degree_mean': 0, 'neg_degree_median': 0,
                'overall_degree_mean': 0, 'overall_degree_median': 0,
            }
        
        return {
            'pos_degree_mean': float(np.mean(pos_degrees)),
            'pos_degree_median': float(np.median(pos_degrees)),
            'pos_degree_max': float(np.max(pos_degrees)),
            'pos_degree_min': float(np.min(pos_degrees)),
            'neg_degree_mean': float(np.mean(neg_degrees)),
            'neg_degree_median': float(np.median(neg_degrees)),
            'neg_degree_max': float(np.max(neg_degrees)),
            'neg_degree_min': float(np.min(neg_degrees)),
            'overall_degree_mean': float(np.mean(overall_degrees)),
            'overall_degree_median': float(np.median(overall_degrees)),
            'overall_degree_max': float(np.max(overall_degrees)),
            'overall_degree_min': float(np.min(overall_degrees)),
        }

    
    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics."""
        n_users = len(self.all_users)
        n_items = len(self.all_items)
        
        user_degrees = [len(items) for items in self.user_items.values()]
        item_degrees = [len(users) for users in self.item_users.values()]
        
        total_interactions = sum(user_degrees)
        
        # Density = actual edges / possible edges in bipartite graph
        density = total_interactions / (n_users * n_items) if n_users * n_items > 0 else 0
        
        # User-user overlap (average number of shared items between user pairs)
        user_list = list(self.all_users)
        total_overlap = 0
        pairs = 0
        for i in range(min(100, len(user_list))):  # Sample for efficiency
            for j in range(i+1, min(100, len(user_list))):
                u1, u2 = user_list[i], user_list[j]
                overlap = len(set(self.user_items[u1]) & set(self.user_items[u2]))
                total_overlap += overlap
                pairs += 1
        avg_user_overlap = total_overlap / pairs if pairs > 0 else 0
        
        # Average number of users who have interacted with each item (item popularity)
        # This represents how many users see an item in their history
        avg_users_per_item = np.mean(item_degrees)
        
        return {
            'n_users': n_users,
            'n_items': n_items,
            'total_interactions': total_interactions,
            'density': density,
            'avg_user_degree': np.mean(user_degrees),
            'min_user_degree': min(user_degrees),
            'max_user_degree': max(user_degrees),
            'avg_item_degree': np.mean(item_degrees),
            'avg_users_per_item': avg_users_per_item,  # How many users have seen each item
            'avg_user_overlap': avg_user_overlap,
        }
    
    def compute_effective_weight(self, user_id: int, item_id: int, k: int) -> float:
        """
        Compute effective weight A_tilde(u,i) = I((u,i) in E) * min(1, k/d_u)
        
        Args:
            user_id: User node
            item_id: Item node
            k: Inference density (num_candidates)
        """
        if item_id not in self.user_items.get(user_id, []):
            return 0.0
        
        d_u = len(self.user_items[user_id])
        return min(1.0, k / d_u)
    
    def compute_users_per_item_per_turn(self, k: int) -> Dict[str, float]:
        """
        Compute how many users see an item on average in one turn.
        
        In each turn, each user samples k items from their history.
        This computes the expected number of users who will sample each item.
        
        Args:
            k: Number of candidates per turn (1, 2, or 4)
        
        Returns:
            Dictionary with mean, max, min statistics
        """
        users_per_item_counts = []
        
        for item_id in self.all_items:
            # Users who have this item in their history
            users_with_item = self.item_users.get(item_id, set())
            
            # Expected number of users who will sample this item in one turn
            expected_users = 0.0
            for user_id in users_with_item:
                d_u = len(self.user_items[user_id])
                # Probability this user samples this item = min(1, k/d_u)
                prob_sample = min(1.0, k / d_u)
                expected_users += prob_sample
            
            users_per_item_counts.append(expected_users)
        
        if not users_per_item_counts:
            return {
                'mean_users_per_item_per_turn': 0.0,
                'max_users_per_item_per_turn': 0.0,
                'min_users_per_item_per_turn': 0.0,
                'std_users_per_item_per_turn': 0.0,
            }
        
        return {
            'mean_users_per_item_per_turn': float(np.mean(users_per_item_counts)),
            'max_users_per_item_per_turn': float(np.max(users_per_item_counts)),
            'min_users_per_item_per_turn': float(np.min(users_per_item_counts)),
            'std_users_per_item_per_turn': float(np.std(users_per_item_counts)),
        }

    
    def compute_cpp(self, k: int) -> float:
        """
        Compute Collaborative Propagation Potential (CPP).
        
        CPP = (1 / N(N-1)) * sum_{u in U, j in I} 1/d_tilde(u,j)
        
        Where d_tilde is shortest path using weights 1/A_tilde.
        Higher CPP = faster signal propagation.
        """
        users = list(self.all_users)
        items = list(self.all_items)
        n_users = len(users)
        n_items = len(items)
        
        if n_users == 0 or n_items == 0:
            return 0.0
        
        # Build user and item index mappings
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {it: i + n_users for i, it in enumerate(items)}
        
        n_total = n_users + n_items
        
        # Build weighted adjacency matrix (using resistance = 1/weight)
        # For shortest path, we want distance = 1/A_tilde
        rows, cols, data = [], [], []
        
        for user_id, items_list in self.user_items.items():
            u_idx = user_idx[user_id]
            d_u = len(items_list)
            weight = min(1.0, k / d_u)
            
            if weight > 0:
                distance = 1.0 / weight  # Resistance distance
                for item_id in items_list:
                    i_idx = item_idx[item_id]
                    # Bidirectional edges
                    rows.extend([u_idx, i_idx])
                    cols.extend([i_idx, u_idx])
                    data.extend([distance, distance])
        
        if not rows:
            return 0.0
        
        # Create sparse matrix
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))
        
        # Compute shortest paths (sample for efficiency)
        sample_size = min(50, n_users)
        sampled_users = np.random.choice(users, sample_size, replace=False)
        
        total_efficiency = 0.0
        count = 0
        
        for user_id in sampled_users:
            u_idx = user_idx[user_id]
            distances = shortest_path(adj, indices=u_idx, directed=False)
            
            for item_id in items:
                i_idx = item_idx[item_id]
                d = distances[i_idx]
                if d > 0 and d < np.inf:
                    total_efficiency += 1.0 / d
                    count += 1
        
        cpp = total_efficiency / count if count > 0 else 0.0
        return cpp

    
    def compute_sc(self, k: int) -> Dict[str, float]:
        """
        Compute Susceptibility Centrality (SC) for users.
        
        SC(u) = (1/lambda) * sum_{i in N(u)} A_tilde(u,i) * SC(i)
        
        Returns mean, max, and distribution stats.
        """
        users = list(self.all_users)
        items = list(self.all_items)
        n_users = len(users)
        n_items = len(items)
        
        if n_users == 0 or n_items == 0:
            return {'mean_sc': 0.0, 'max_sc': 0.0, 'min_sc': 0.0}
        
        # Build user and item index mappings
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {it: i + n_users for i, it in enumerate(items)}
        
        n_total = n_users + n_items
        
        # Build weighted adjacency matrix with effective weights
        rows, cols, data = [], [], []
        
        for user_id, items_list in self.user_items.items():
            u_idx = user_idx[user_id]
            d_u = len(items_list)
            weight = min(1.0, k / d_u)
            
            for item_id in items_list:
                i_idx = item_idx[item_id]
                rows.extend([u_idx, i_idx])
                cols.extend([i_idx, u_idx])
                data.extend([weight, weight])
        
        if not rows:
            return {'mean_sc': 0.0, 'max_sc': 0.0, 'min_sc': 0.0}
        
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))
        
        try:
            # Compute principal eigenvector
            eigenvalues, eigenvectors = eigsh(adj.astype(float), k=1, which='LM')
            principal_eigenvector = np.abs(eigenvectors[:, 0])
            
            # Normalize
            max_val = np.max(principal_eigenvector)
            if max_val > 0:
                principal_eigenvector /= max_val
            
            # Extract user centralities
            user_centralities = [principal_eigenvector[user_idx[u]] for u in users]
            
            return {
                'mean_sc': float(np.mean(user_centralities)),
                'max_sc': float(np.max(user_centralities)),
                'min_sc': float(np.min(user_centralities)),
                'std_sc': float(np.std(user_centralities)),
            }
        except Exception as e:
            print(f"  Warning: Eigenvector computation failed: {e}")
            return {'mean_sc': 0.0, 'max_sc': 0.0, 'min_sc': 0.0, 'std_sc': 0.0}

    
    def compute_per(self, k: int, attacker_ratio: float = 0.25) -> Dict[str, float]:
        """
        Compute Probabilistic Exposure Risk (PER).
        
        PER(u, v_atk) = 1 - prod_{path} (1 - Pi_path)
        
        Simulates attacker placement and computes exposure probability.
        """
        users = list(self.all_users)
        items = list(self.all_items)
        n_users = len(users)
        n_items = len(items)
        
        if n_users == 0 or n_items == 0:
            return {'mean_per': 0.0, 'max_per': 0.0}
        
        # Simulate attacker items (most connected items are more dangerous)
        item_degrees = [(it, len(self.item_users.get(it, []))) for it in items]
        item_degrees.sort(key=lambda x: -x[1])
        
        n_attacker_items = max(1, int(n_items * attacker_ratio))
        attacker_items = set([it for it, _ in item_degrees[:n_attacker_items]])
        
        # Compute PER for each user
        user_pers = []
        
        for user_id in users:
            user_items_set = set(self.user_items.get(user_id, []))
            d_u = len(user_items_set)
            
            if d_u == 0:
                user_pers.append(0.0)
                continue
            
            # Direct exposure: probability of sampling an attacker item
            direct_attacker_items = user_items_set & attacker_items
            n_direct = len(direct_attacker_items)
            
            # Probability of NOT sampling any attacker item in k draws
            # P(no attacker) = C(d_u - n_direct, k) / C(d_u, k)
            if n_direct == 0:
                direct_per = 0.0
            elif k >= d_u:
                direct_per = 1.0 if n_direct > 0 else 0.0
            else:
                # Approximate: P(at least one attacker) = 1 - (1 - n_direct/d_u)^k
                p_single = n_direct / d_u
                direct_per = 1.0 - (1.0 - p_single) ** k
            
            # 2-hop exposure: through shared items with other users who have attacker items
            indirect_per = 0.0
            for item_id in user_items_set:
                if item_id in attacker_items:
                    continue
                # Other users who share this item
                other_users = self.item_users.get(item_id, set()) - {user_id}
                for other_user in other_users:
                    other_items = set(self.user_items.get(other_user, []))
                    other_attacker = other_items & attacker_items
                    if other_attacker:
                        # Probability chain: this user samples item_id AND other user samples attacker
                        d_other = len(other_items)
                        p_this = min(1.0, k / d_u)
                        p_other = min(1.0, k / d_other) * (len(other_attacker) / d_other)
                        indirect_per = max(indirect_per, p_this * p_other * 0.5)  # Discount for indirection
            
            total_per = min(1.0, direct_per + indirect_per * (1 - direct_per))
            user_pers.append(total_per)
        
        return {
            'mean_per': float(np.mean(user_pers)),
            'max_per': float(np.max(user_pers)),
            'min_per': float(np.min(user_pers)),
            'std_per': float(np.std(user_pers)),
        }

    
    def compute_tree_metrics(self, neighbor_count: int = 3) -> Dict[str, float]:
        """
        Compute metrics for TREE topology.
        
        In tree topology:
        - No direct agent-to-agent edges
        - Manager recruits neighbor_count similar users per forward pass
        - Similarity is based on frozen (pre-attack) history overlap
        """
        users = list(self.all_users)
        n_users = len(users)
        
        if n_users == 0:
            return {'tree_cpp': 0.0, 'tree_sc': 0.0, 'tree_per': 0.0}
        
        # Compute user-user similarity matrix (Jaccard)
        user_similarities = {}
        for i, u1 in enumerate(users):
            items1 = set(self.user_items.get(u1, []))
            for j, u2 in enumerate(users[i+1:], i+1):
                items2 = set(self.user_items.get(u2, []))
                intersection = len(items1 & items2)
                union = len(items1 | items2)
                sim = intersection / union if union > 0 else 0.0
                user_similarities[(u1, u2)] = sim
                user_similarities[(u2, u1)] = sim
        
        # For each user, find top-k similar neighbors
        user_neighbors = {}
        for u in users:
            sims = [(u2, user_similarities.get((u, u2), 0.0)) for u2 in users if u2 != u]
            sims.sort(key=lambda x: -x[1])
            user_neighbors[u] = [u2 for u2, _ in sims[:neighbor_count]]
        
        # Tree CPP: Information flows through manager hub
        # Effective distance is 2 (user -> manager -> user)
        # But only to neighbors, so CPP is limited
        avg_neighbor_sim = np.mean([
            user_similarities.get((u, n), 0.0) 
            for u in users for n in user_neighbors[u]
        ])
        tree_cpp = avg_neighbor_sim * 0.5  # Discounted for hub-spoke
        
        # Tree SC: All users have similar centrality (star topology)
        # Manager is central, users are peripheral
        tree_sc = 1.0 / (neighbor_count + 1)  # Approximate
        
        # Tree PER: Attackers can only reach similar-history users
        # Much lower than dense topologies
        tree_per = avg_neighbor_sim * 0.25  # Heavily discounted
        
        return {
            'tree_cpp': float(tree_cpp),
            'tree_sc': float(tree_sc),
            'tree_per': float(tree_per),
            'avg_neighbor_similarity': float(avg_neighbor_sim),
        }


    def _build_matrix_data(self, max_items: Optional[int] = None) -> dict:
        """Build the interaction matrix and marginals for plotting."""
        users = sorted(self.all_users, key=lambda u: len(self.user_items.get(u, [])))
        items_sorted = sorted(
            self.all_items,
            key=lambda i: len(self.item_users.get(i, set())),
            reverse=True,
        )
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {it: i for i, it in enumerate(items_sorted)}

        n_u = len(users)
        n_i_actual = len(items_sorted)
        n_i_display = max(n_i_actual, max_items) if max_items else n_i_actual

        mat = np.zeros((n_u, n_i_display), dtype=np.float32)
        for u in users:
            for it in self.user_items.get(u, []):
                if it in item_idx:
                    mat[user_idx[u], item_idx[it]] = 1.0

        user_degrees = mat.sum(axis=1)
        item_degrees = mat.sum(axis=0)
        density = mat.sum() / (n_u * n_i_actual) if n_u * n_i_actual > 0 else 0
        nnz = int(mat.sum())

        return dict(
            mat=mat, user_degrees=user_degrees, item_degrees=item_degrees,
            n_u=n_u, n_i_actual=n_i_actual, n_i_display=n_i_display,
            density=density, nnz=nnz,
        )

    def plot_interaction_matrix(self, output_path: Optional[str] = None,
                               max_items: Optional[int] = None) -> Optional[str]:
        """Single-dataset interaction matrix (fallback for unpaired datasets)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import matplotlib.gridspec as gridspec

        d = self._build_matrix_data(max_items)
        mat, n_u = d['mat'], d['n_u']
        n_i_actual, n_i_display = d['n_i_actual'], d['n_i_display']

        fig = plt.figure(figsize=(8, 4.5), dpi=150)
        gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1], height_ratios=[1, 5],
                               wspace=0.04, hspace=0.04)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_corner = fig.add_subplot(gs[0, 1])

        cmap = ListedColormap(['#f7f7f7', '#2166ac'])
        ax_main.imshow(mat, aspect='auto', cmap=cmap, interpolation='nearest')
        ax_main.set_xlabel('Items (sorted by popularity →)', fontsize=8)
        ax_main.set_ylabel('Users (sorted by degree →)', fontsize=8)
        ax_main.tick_params(labelsize=6)
        ax_main.set_xticks(np.linspace(0, n_i_display - 1, min(6, n_i_display)).astype(int))
        ax_main.set_yticks(np.linspace(0, n_u - 1, min(6, n_u)).astype(int))

        if max_items and n_i_actual < n_i_display:
            ax_main.axvspan(n_i_actual - 0.5, n_i_display - 0.5,
                            color='#fff3e0', alpha=0.5, zorder=0)
            ax_main.axvline(x=n_i_actual - 0.5, color='#e65100', lw=1.2,
                            ls='--', alpha=0.7)

        ax_top.bar(range(n_i_display), d['item_degrees'], width=1.0,
                   color='#2166ac', alpha=0.7)
        ax_top.set_xlim(-0.5, n_i_display - 0.5)
        ax_top.set_ylabel('Pop', fontsize=7)
        ax_top.tick_params(labelsize=5, labelbottom=False)

        ax_right.barh(range(n_u), d['user_degrees'], height=1.0,
                      color='#b2182b', alpha=0.7)
        ax_right.set_ylim(-0.5, n_u - 0.5)
        ax_right.set_xlabel('Deg', fontsize=7)
        ax_right.tick_params(labelsize=5, labelleft=False)
        ax_right.invert_yaxis()

        ax_corner.axis('off')
        il = f"{n_i_actual}i" if n_i_actual == n_i_display else f"{n_i_actual}/{n_i_display}i"
        ax_corner.text(0.5, 0.5, f"{n_u}u × {il}\nρ={d['density']:.4f}\nnnz={d['nnz']}",
                       ha='center', va='center', fontsize=7, family='monospace',
                       bbox=dict(boxstyle='round,pad=0.3', fc='#ffffcc', alpha=0.8))

        fig.suptitle(f'{self.dataset_name}  —  Interaction Matrix', fontsize=10, y=0.98)
        if output_path is None:
            output_path = str(Path(__file__).parent / f'interaction_matrix_{self.dataset_name}.png')
        fig.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Interaction matrix saved to {output_path}")
        return output_path


def plot_paired_interaction_matrix(
    dm_dense: 'DatasetMetrics',
    dm_sparse: 'DatasetMetrics',
    output_path: Optional[str] = None,
) -> str:
    """
    Side-by-side interaction matrix for a dense/sparse pair.

    Both panels share the same item-axis width (= sparse item count).
    The dense panel's unused catalog region is shaded to make the
    smaller catalog visually obvious.  A column-density curve and
    large density annotations reinforce the contrast.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec

    max_items = len(dm_sparse.all_items)
    dd = dm_dense._build_matrix_data(max_items)
    ds = dm_sparse._build_matrix_data(max_items)

    cmap = LinearSegmentedColormap.from_list(
        'interact', ['#f5f5f5', '#1565c0', '#0d47a1'], N=3)

    fig = plt.figure(figsize=(16, 5), dpi=150)
    outer = gridspec.GridSpec(1, 2, wspace=0.12, figure=fig)

    pair_label = dm_dense.dataset_name.rsplit('-', 1)[0]

    for col_idx, (dm, d, label) in enumerate([
        (dm_dense, dd, 'Dense'),
        (dm_sparse, ds, 'Sparse'),
    ]):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[col_idx],
            width_ratios=[6, 1], height_ratios=[1.2, 5],
            wspace=0.04, hspace=0.06,
        )
        ax_top   = fig.add_subplot(inner[0, 0])
        ax_main  = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        ax_stat  = fig.add_subplot(inner[0, 1])

        mat = d['mat']
        n_u, n_i_actual, n_i_display = d['n_u'], d['n_i_actual'], d['n_i_display']

        # main heatmap
        ax_main.imshow(mat, aspect='auto', cmap=cmap, interpolation='nearest',
                       vmin=0, vmax=1)

        # shade unused catalog region
        if n_i_actual < n_i_display:
            ax_main.axvspan(n_i_actual - 0.5, n_i_display - 0.5,
                            color='#fff3e0', alpha=0.55, zorder=0)
            ax_main.axvline(x=n_i_actual - 0.5, color='#e65100',
                            lw=1.5, ls='--', alpha=0.8, zorder=3)
            mid_x = (n_i_actual + n_i_display) / 2 - 0.5
            ax_main.text(mid_x, n_u / 2, 'no items',
                         ha='center', va='center', fontsize=7,
                         color='#bf360c', alpha=0.7, style='italic',
                         rotation=90 if (n_i_display - n_i_actual) < n_u else 0)

        ax_main.set_xlabel(f'Items  (catalog = {n_i_actual})', fontsize=8)
        if col_idx == 0:
            ax_main.set_ylabel('Users (sorted by degree →)', fontsize=8)
        ax_main.tick_params(labelsize=6)
        ax_main.set_xticks(np.linspace(0, n_i_display - 1, min(6, n_i_display)).astype(int))
        ax_main.set_yticks(np.linspace(0, n_u - 1, min(6, n_u)).astype(int))

        # column-density curve overlay
        window = max(1, n_i_display // 15)
        col_density = np.convolve(
            mat.mean(axis=0), np.ones(window) / window, mode='same')
        ax_density = ax_main.twinx()
        ax_density.fill_between(range(n_i_display), col_density,
                                color='#ff6f00', alpha=0.18, zorder=2)
        ax_density.plot(range(n_i_display), col_density,
                        color='#e65100', lw=0.8, alpha=0.6, zorder=2)
        ax_density.set_ylim(0, max(col_density.max() * 2.5, 0.01))
        ax_density.set_yticks([])
        ax_density.set_xlim(-0.5, n_i_display - 0.5)

        # top bar: item popularity
        ax_top.bar(range(n_i_display), d['item_degrees'], width=1.0,
                   color='#1565c0', alpha=0.7)
        if n_i_actual < n_i_display:
            ax_top.axvspan(n_i_actual - 0.5, n_i_display - 0.5,
                           color='#fff3e0', alpha=0.55, zorder=0)
            ax_top.axvline(x=n_i_actual - 0.5, color='#e65100',
                           lw=1.5, ls='--', alpha=0.8, zorder=3)
        ax_top.set_xlim(-0.5, n_i_display - 0.5)
        ax_top.set_ylabel('Pop', fontsize=7)
        ax_top.tick_params(labelsize=5, labelbottom=False)
        ax_top.set_title(f'{label}', fontsize=11, fontweight='bold', pad=4)

        # right bar: user degree
        ax_right.barh(range(n_u), d['user_degrees'], height=1.0,
                      color='#b71c1c', alpha=0.65)
        ax_right.set_ylim(-0.5, n_u - 0.5)
        ax_right.set_xlabel('Deg', fontsize=7)
        ax_right.tick_params(labelsize=5, labelleft=False)
        ax_right.invert_yaxis()

        # stat box
        ax_stat.axis('off')
        rho_str = f"ρ = {d['density']:.4f}"
        box_color = '#c8e6c9' if label == 'Dense' else '#ffcdd2'
        ax_stat.text(
            0.5, 0.5,
            f"{n_u}u × {n_i_actual}i\n{rho_str}\nnnz={d['nnz']}",
            ha='center', va='center', fontsize=8, family='monospace',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc=box_color, ec='#666666',
                      lw=0.8, alpha=0.9),
        )

    fig.suptitle(f'{pair_label}  —  Dense vs Sparse Interaction Matrix',
                 fontsize=12, fontweight='bold', y=1.01)

    if output_path is None:
        output_path = str(
            Path(__file__).parent / f'interaction_matrix_pair_{pair_label}.png')
    fig.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Paired interaction matrix saved to {output_path}")
    return output_path


def compute_all_metrics(dataset_name: str) -> Tuple[Optional[Dict], Optional['DatasetMetrics']]:
    """Compute all metrics for a dataset across all topology configurations.
    
    Returns:
        Tuple of (results dict, DatasetMetrics instance) — either may be None on error.
    """
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        dm = DatasetMetrics(dataset_name)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return None, None
    
    # Basic stats
    stats = dm.get_basic_stats()
    print(f"  Users: {stats['n_users']}, Items: {stats['n_items']} (total item catalog)")
    print(f"  Total interactions: {stats['total_interactions']}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Avg user degree (history length): {stats['avg_user_degree']:.1f}")
    print(f"  Avg users per item (pos only): {stats['avg_users_per_item']:.1f}")
    print(f"  Avg user overlap: {stats['avg_user_overlap']:.1f}")
    
    results = {
        'dataset': dataset_name,
        'basic_stats': stats,
        'topologies': {},
        'item_degrees': {},
        'missing_neg_files': list(dm.missing_neg_files),
    }
    
    # Topology configurations
    topologies = [
        ('tree', {'type': 'tree', 'neighbor_count': 3}),
        ('cand_1', {'type': 'candidates', 'k': 1}),
        ('cand_2', {'type': 'candidates', 'k': 2}),
        ('cand_3', {'type': 'candidates', 'k': 3}),
        ('cand_4', {'type': 'candidates', 'k': 4}),
    ]
    
    for topo_name, topo_config in topologies:
        print(f"\n  Topology: {topo_name}")
        
        # Check if neg sample file is missing for this topology
        if topo_name in dm.missing_neg_files:
            print(f"    ⚠ ranking file missing for {topo_name} (neg sample stats unavailable)")
        
        if topo_config['type'] == 'tree':
            metrics = dm.compute_tree_metrics(topo_config['neighbor_count'])
            results['topologies'][topo_name] = {
                'cpp': metrics['tree_cpp'],
                'sc': metrics['tree_sc'],
                'per': metrics['tree_per'],
                'extra': {'avg_neighbor_similarity': metrics['avg_neighbor_similarity']}
            }
        else:
            k = topo_config['k']
            cpp = dm.compute_cpp(k)
            sc = dm.compute_sc(k)
            per = dm.compute_per(k)
            users_per_item = dm.compute_users_per_item_per_turn(k)
            
            # Item degree stats (pos / neg / overall)
            deg_stats = dm.get_item_degree_stats(topo_name)
            results['item_degrees'][topo_name] = deg_stats
            
            results['topologies'][topo_name] = {
                'cpp': cpp,
                'sc': sc['mean_sc'],
                'per': per['mean_per'],
                'users_per_item_per_turn': users_per_item['mean_users_per_item_per_turn'],
                'extra': {
                    'sc_max': sc['max_sc'],
                    'sc_std': sc.get('std_sc', 0),
                    'per_max': per['max_per'],
                    'per_std': per.get('std_per', 0),
                    'users_per_item_max': users_per_item['max_users_per_item_per_turn'],
                    'users_per_item_std': users_per_item['std_users_per_item_per_turn'],
                }
            }
        
        topo_results = results['topologies'][topo_name]
        print(f"    CPP: {topo_results['cpp']:.4f}")
        print(f"    SC:  {topo_results['sc']:.4f}")
        print(f"    PER: {topo_results['per']:.4f}")
        if 'users_per_item_per_turn' in topo_results:
            print(f"    Avg users per item per turn: {topo_results['users_per_item_per_turn']:.2f}")
        if topo_name in results['item_degrees']:
            ds = results['item_degrees'][topo_name]
            print(f"    Item degree (pos):     mean={ds['pos_degree_mean']:.2f}  median={ds['pos_degree_median']:.1f}  max={ds['pos_degree_max']:.0f}  min={ds['pos_degree_min']:.0f}")
            print(f"    Item degree (neg):     mean={ds['neg_degree_mean']:.2f}  median={ds['neg_degree_median']:.1f}  max={ds['neg_degree_max']:.0f}  min={ds['neg_degree_min']:.0f}")
            print(f"    Item degree (overall): mean={ds['overall_degree_mean']:.2f}  median={ds['overall_degree_median']:.1f}  max={ds['overall_degree_max']:.0f}  min={ds['overall_degree_min']:.0f}")
    
    return results, dm


def generate_latex_table(all_results: List[Dict]) -> str:
    """Generate LaTeX table from results."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{NetSafe-Adapted Metrics Across Datasets and Topologies}
\label{tab:dataset_metrics}
\begin{tabular}{l|ccc|ccc|ccc|ccc|ccc}
\toprule
& \multicolumn{3}{c|}{Tree} & \multicolumn{3}{c|}{Cand-1} & \multicolumn{3}{c|}{Cand-2} & \multicolumn{3}{c|}{Cand-3} & \multicolumn{3}{c}{Cand-4} \\
Dataset & CPP & SC & PER & CPP & SC & PER & CPP & SC & PER & CPP & SC & PER & CPP & SC & PER \\
\midrule
"""
    
    for result in all_results:
        if result is None:
            continue
        
        name = result['dataset'].replace('_', r'\_').replace('-', r'-')
        # Shorten name for table
        short_name = name.replace('user', 'u').replace('dense', 'D').replace('sparse', 'S').replace('mini', 'M')
        
        row = [short_name]
        for topo in ['tree', 'cand_1', 'cand_2', 'cand_3', 'cand_4']:
            if topo in result['topologies']:
                t = result['topologies'][topo]
                row.extend([f"{t['cpp']:.3f}", f"{t['sc']:.3f}", f"{t['per']:.3f}"])
            else:
                row.extend(['-', '-', '-'])
        
        latex += ' & '.join(row) + r' \\' + '\n'
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(all_results: List[Dict]) -> str:
    """Generate Markdown table from results."""
    md = "| Dataset | Users | Items | Density | Avg Users/Item | "
    md += "Tree CPP | Tree SC | Tree PER | "
    md += "C1 CPP | C1 SC | C1 PER | C1 Users/Item/Turn | "
    md += "C2 CPP | C2 SC | C2 PER | C2 Users/Item/Turn | "
    md += "C3 CPP | C3 SC | C3 PER | C3 Users/Item/Turn | "
    md += "C4 CPP | C4 SC | C4 PER | C4 Users/Item/Turn |\n"
    md += "|" + "|".join(["---"]*24) + "|\n"
    
    for result in all_results:
        if result is None:
            continue
        
        stats = result['basic_stats']
        row = [
            result['dataset'],
            str(stats['n_users']),
            str(stats['n_items']),
            f"{stats['density']:.5f}",
            f"{stats['avg_users_per_item']:.2f}",
        ]
        
        for topo in ['tree', 'cand_1', 'cand_2', 'cand_3', 'cand_4']:
            if topo in result['topologies']:
                t = result['topologies'][topo]
                row.extend([f"{t['cpp']:.4f}", f"{t['sc']:.4f}", f"{t['per']:.4f}"])
                if 'users_per_item_per_turn' in t:
                    row.append(f"{t['users_per_item_per_turn']:.2f}")
            else:
                row.extend(['-', '-', '-'])
                if topo != 'tree':
                    row.append('-')
        
        md += "| " + " | ".join(row) + " |\n"
    
    return md


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute NetSafe metrics for datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to process (default: all)')
    parser.add_argument('--output', type=str, 
                       default=str(Path(__file__).parent / 'dataset_metrics_report.md'),
                       help='Output file for report')
    args = parser.parse_args()
    
    # Get all datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [
            'CDs-100user-dense',
            'CDs-100user-sparse',
            'CDs-20-user-dense',
            'CDs-20-user-sparse',
            'ml-100k-100user-dense',
            'ml-100k-100user-sparse',
            'ml-100k-20-user-dense',
            'ml-100k-20-user-sparse',
        ]
    
    print("=" * 70)
    print("NetSafe-Adapted Metrics Computation")
    print("=" * 70)
    print(f"\nDatasets to process: {len(datasets)}")
    print(f"Topologies: tree, cand_1, cand_2, cand_3, cand_4")
    
    all_results = []
    all_dm = {}  # dataset_name -> DatasetMetrics instance
    for dataset in datasets:
        result, dm = compute_all_metrics(dataset)
        if result:
            all_results.append(result)
            all_dm[dataset] = dm
    
    # --- Generate interaction matrix visuals with paired item-axis ---
    # Detect dense/sparse pairs: same prefix minus the density suffix.
    # Sparse always has more items, so we use its item count as the shared max.
    import re
    print("\n" + "=" * 70)
    print("Generating Interaction Matrix Visuals (paired item axis)")
    print("=" * 70)

    def _pair_key(name: str) -> Optional[str]:
        """Extract the pairing key, e.g. 'ml-100k-20-user' from 'ml-100k-20-user-dense'."""
        m = re.match(r'^(.+-user)-(dense|sparse)$', name)
        return m.group(1) if m else None

    pairs = defaultdict(dict)  # pair_key -> {'dense': name, 'sparse': name}
    for name in all_dm:
        pk = _pair_key(name)
        if pk:
            variant = 'dense' if name.endswith('-dense') else 'sparse'
            pairs[pk][variant] = name

    plotted = set()
    for pk, variants in pairs.items():
        if 'dense' in variants and 'sparse' in variants:
            dm_sparse = all_dm[variants['sparse']]
            dm_dense = all_dm[variants['dense']]
            max_items = len(dm_sparse.all_items)
            print(f"\n  Pair: {variants['dense']} / {variants['sparse']}  →  shared item axis = {max_items}")
            plot_paired_interaction_matrix(dm_dense, dm_sparse)
            plotted.add(variants['dense'])
            plotted.add(variants['sparse'])

    # Plot any unpaired datasets with their natural axis
    for name, dm in all_dm.items():
        if name not in plotted:
            dm.plot_interaction_matrix()
    
    # Generate reports
    print("\n" + "=" * 70)
    print("Generating Reports")
    print("=" * 70)
    
    # Markdown report
    md_report = "# NetSafe-Adapted Metrics Report\n\n"
    md_report += "## Metrics Definitions\n\n"
    md_report += "- **CPP (Collaborative Propagation Potential)**: Measures how efficiently signals propagate through the graph. Higher = faster contamination spread.\n"
    md_report += "- **SC (Susceptibility Centrality)**: Measures user vulnerability based on network position. Higher = more susceptible to influence.\n"
    md_report += "- **PER (Probabilistic Exposure Risk)**: Probability of exposure to attacker content. Higher = more likely to be contaminated.\n"
    md_report += "- **Users/Item/Turn**: Expected number of users who will see each item in one turn. This varies by k (number of candidates sampled per turn).\n\n"
    md_report += "## Topology Configurations\n\n"
    md_report += "- **Tree**: Hub-and-spoke with manager, neighbor_count=3, frozen recruitment\n"
    md_report += "- **Cand-1**: Binary classification (k=1 item per round)\n"
    md_report += "- **Cand-2**: Pairwise comparison (k=2 items per round)\n"
    md_report += "- **Cand-3**: 3-candidate ranking (k=3 items per round)\n"
    md_report += "- **Cand-4**: Full ranking (k=4 items per round)\n\n"
    md_report += "## Key Clarifications\n\n"
    md_report += "- **Items**: Refers to the total item catalog size (e.g., 118 items means there are 118 unique items in the dataset)\n"
    md_report += "- **Avg Users/Item**: Average number of users who have each item in their interaction history\n"
    md_report += "- **Users/Item/Turn**: Expected number of users who will sample/see each item in one turn, given k candidates per turn\n"
    md_report += "  - For k=1: Each user samples 1 item, so fewer users see each item per turn\n"
    md_report += "  - For k=2: Each user samples 2 items, so more users see each item per turn\n"
    md_report += "  - For k=3: Each user samples 3 items, so more users see each item per turn\n"
    md_report += "  - For k=4: Each user samples 4 items, so even more users see each item per turn\n"
    md_report += "  - Note: Different k values affect negative sampling strategies and item exposure patterns\n\n"
    md_report += "## Results\n\n"
    md_report += generate_markdown_table(all_results)
    
    # Add detailed stats
    md_report += "\n## Detailed Statistics\n\n"
    for result in all_results:
        if result is None:
            continue
        stats = result['basic_stats']
        md_report += f"### {result['dataset']}\n\n"
        md_report += f"- Users: {stats['n_users']}\n"
        md_report += f"- Items: {stats['n_items']} (total item catalog)\n"
        md_report += f"- Total interactions: {stats['total_interactions']}\n"
        md_report += f"- Density: {stats['density']:.6f}\n"
        md_report += f"- Avg user degree (history length): {stats['avg_user_degree']:.2f}\n"
        md_report += f"- Avg users per item (pos only): {stats['avg_users_per_item']:.2f}\n"
        md_report += f"- Avg user overlap: {stats['avg_user_overlap']:.2f}\n"
        
        # Add per-topology users per item per turn
        md_report += f"\n**Users per item per turn (expected exposure):**\n"
        for topo in ['cand_1', 'cand_2', 'cand_3', 'cand_4']:
            if topo in result['topologies'] and 'users_per_item_per_turn' in result['topologies'][topo]:
                val = result['topologies'][topo]['users_per_item_per_turn']
                md_report += f"- {topo}: {val:.2f} users see each item on average per turn\n"
        
        # Add per-topology item degree breakdown
        if result.get('item_degrees'):
            md_report += f"\n**Item degree breakdown (pos / neg / overall):**\n\n"
            md_report += "| Topology | Pos Mean | Pos Median | Neg Mean | Neg Median | Overall Mean | Overall Median | Overall Max |\n"
            md_report += "|----------|----------|------------|----------|------------|--------------|----------------|-------------|\n"
            for topo in ['cand_1', 'cand_2', 'cand_3', 'cand_4']:
                if topo in result['item_degrees']:
                    ds = result['item_degrees'][topo]
                    md_report += f"| {topo} | {ds['pos_degree_mean']:.2f} | {ds['pos_degree_median']:.1f} | {ds['neg_degree_mean']:.2f} | {ds['neg_degree_median']:.1f} | {ds['overall_degree_mean']:.2f} | {ds['overall_degree_median']:.1f} | {ds['overall_degree_max']:.0f} |\n"
        
        md_report += "\n"
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(md_report)
    print(f"\nMarkdown report saved to: {args.output}")
    
    # Save JSON results
    json_output = args.output.replace('.md', '.json')
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved to: {json_output}")
    
    # Print LaTeX table
    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)
    print(generate_latex_table(all_results))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
