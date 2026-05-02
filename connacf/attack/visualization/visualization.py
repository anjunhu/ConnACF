"""
Visualization utilities for ConnaCF attack analysis

Creates plots and graphs to visualize attack propagation, contamination,
and system performance degradation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime


class AttackVisualization:
    """Visualization utilities for attack analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_directory', 'attack_results')
        self.plot_format = config.get('plot_format', 'png')
        self.save_high_res = config.get('save_high_res', True)
        self.dpi = 300 if self.save_high_res else 100
        
        # Fixed layout cache for consistent node positioning across ALL rounds
        # This will be initialized once with ALL possible agents
        self.fixed_layout = None
        self.layout_initialized = False
        self.n_users = None
        self.n_items = None
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def initialize_fixed_layout(self, n_users: int, n_items: int, seed: int = 42):
        """
        Initialize fixed positions for ALL agents upfront.
        This ensures consistent positioning across all interaction graph visualizations.
        
        Args:
            n_users: Total number of users in the system
            n_items: Total number of items in the system
            seed: Random seed for reproducible layout
        """
        if self.layout_initialized and self.n_users == n_users and self.n_items == n_items:
            print(f"[VISUALIZATION] Fixed layout already initialized for {n_users} users, {n_items} items")
            return
        
        print(f"[VISUALIZATION] Initializing fixed layout for {n_users} users, {n_items} items")
        
        self.n_users = n_users
        self.n_items = n_items
        self.fixed_layout = {}
        
        np.random.seed(seed)
        
        # Create a bipartite-style layout with users on left, items on right
        # This provides clear visual separation and consistent positioning
        
        # User positions: arranged in a column on the left side
        # Spread users vertically from y=0 to y=1
        for user_id in range(n_users):
            # X position: left side (0.0 to 0.3) with some randomness
            x = 0.1 + np.random.uniform(-0.05, 0.05)
            # Y position: evenly distributed vertically
            y = user_id / max(n_users - 1, 1) if n_users > 1 else 0.5
            # Add small random offset to prevent perfect alignment
            y += np.random.uniform(-0.02, 0.02)
            self.fixed_layout[f"U{user_id:02d}"] = np.array([x, y])
        
        # Item positions: arranged in a column on the right side
        # Spread items vertically from y=0 to y=1
        for item_id in range(n_items):
            # X position: right side (0.7 to 1.0) with some randomness
            x = 0.9 + np.random.uniform(-0.05, 0.05)
            # Y position: evenly distributed vertically
            y = item_id / max(n_items - 1, 1) if n_items > 1 else 0.5
            # Add small random offset to prevent perfect alignment
            y += np.random.uniform(-0.02, 0.02)
            self.fixed_layout[f"I{item_id:02d}"] = np.array([x, y])
        
        self.layout_initialized = True
        print(f"[VISUALIZATION] Fixed layout initialized: {len(self.fixed_layout)} positions cached")
    
    def get_node_position(self, node_id: str) -> np.ndarray:
        """
        Get the fixed position for a node.
        Falls back to a deterministic position if node wasn't in initial layout.
        """
        if node_id in self.fixed_layout:
            return self.fixed_layout[node_id]
        
        # Fallback: generate deterministic position based on node ID
        # This handles any nodes that weren't in the initial layout
        np.random.seed(hash(node_id) % (2**32))
        if node_id.startswith('U'):
            x = 0.1 + np.random.uniform(-0.05, 0.05)
        else:
            x = 0.9 + np.random.uniform(-0.05, 0.05)
        y = np.random.uniform(0, 1)
        
        # Cache for future use
        self.fixed_layout[node_id] = np.array([x, y])
        return self.fixed_layout[node_id]
        
    def plot_contamination_timeline(self, metrics_collector, experiment_name: str):
        """Plot contamination progression over rounds"""
        timeline = metrics_collector.get_contamination_timeline()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # User contamination
        ax1.plot(timeline['rounds'], timeline['user_contamination'], 
                marker='o', linewidth=2, label='Contaminated Users')
        ax1.set_xlabel('Interaction Round')
        ax1.set_ylabel('Number of Contaminated Users')
        ax1.set_title('User Agent Contamination Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Item contamination
        ax2.plot(timeline['rounds'], timeline['item_contamination'], 
                marker='s', linewidth=2, color='orange', label='Contaminated Items')
        ax2.set_xlabel('Interaction Round')
        ax2.set_ylabel('Number of Contaminated Items')
        ax2.set_title('Item Agent Contamination Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_contamination_timeline")
        plt.close()
    
    def plot_performance_degradation(self, metrics_collector, experiment_name: str):
        """Plot system performance degradation over rounds"""
        timeline = metrics_collector.get_performance_timeline()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(timeline['rounds'], timeline['accuracy'], 
                       marker='o', linewidth=2, color='red')
        axes[0, 0].set_title('Recommendation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall@5
        axes[0, 1].plot(timeline['rounds'], timeline['recall_at_5'], 
                       marker='s', linewidth=2, color='blue')
        axes[0, 1].set_title('Recall@5')
        axes[0, 1].set_ylabel('Recall@5')
        axes[0, 1].grid(True, alpha=0.3)
        
        # NDCG@5
        axes[1, 0].plot(timeline['rounds'], timeline['ndcg_at_5'], 
                       marker='^', linewidth=2, color='green')
        axes[1, 0].set_title('NDCG@5')
        axes[1, 0].set_xlabel('Interaction Round')
        axes[1, 0].set_ylabel('NDCG@5')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined performance
        axes[1, 1].plot(timeline['rounds'], timeline['accuracy'], 
                       marker='o', linewidth=2, label='Accuracy')
        axes[1, 1].plot(timeline['rounds'], timeline['recall_at_5'], 
                       marker='s', linewidth=2, label='Recall@5')
        axes[1, 1].plot(timeline['rounds'], timeline['ndcg_at_5'], 
                       marker='^', linewidth=2, label='NDCG@5')
        axes[1, 1].set_title('Combined Performance Metrics')
        axes[1, 1].set_xlabel('Interaction Round')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_performance_degradation")
        plt.close()
    
    def plot_contamination_heatmap(self, metrics_collector, experiment_name: str):
        """Plot contamination heatmap across agents and rounds"""
        # Extract user contamination data
        user_contamination_matrix = []
        rounds = []
        
        for user_id, user_metrics in metrics_collector.user_metrics.items():
            if user_id not in metrics_collector.attacker_user_indices:  # Only clean agents
                # Use contamination score if available (attack mode), otherwise use evolution drift
                contamination_scores = []
                for m in user_metrics:
                    if m.get('contamination'):
                        contamination_scores.append(m['contamination']['contamination_score'])
                    else:
                        contamination_scores.append(m.get('evolution', {}).get('text_drift', 0.0))
                
                if not rounds:  # First user, extract rounds
                    rounds = [m['round'] for m in user_metrics]
                user_contamination_matrix.append(contamination_scores)
        
        if user_contamination_matrix:
            contamination_df = pd.DataFrame(
                user_contamination_matrix,
                columns=[f'Round {r}' for r in rounds],
                index=[f'User {i}' for i in range(len(user_contamination_matrix))]
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(contamination_df, annot=False, cmap='Reds', 
                       cbar_kws={'label': 'Contamination Score'}, ax=ax)
            ax.set_title('User Agent Contamination Heatmap')
            ax.set_xlabel('Interaction Round')
            ax.set_ylabel('User Agent')
            
            plt.tight_layout()
            self._save_plot(fig, f"{experiment_name}_contamination_heatmap")
            plt.close()
    
    def plot_interaction_graph(self, interaction_controller, round_num: int, 
                             contamination_scores: Dict[int, float], experiment_name: str):
        """Plot agent interaction graph with contamination-based coloring.
        
        DEPRECATED: This method generates misleading graphs because it only shows
        edges from attack propagation (user-user, item-item similarity) but not
        the actual user-item interactions from the forward pass.
        
        User-item edges are now properly logged in the forward pass via
        interaction_controller.record_interaction() with type='user_item'.
        
        This method is kept for backwards compatibility but should not be called.
        """
        import warnings
        warnings.warn(
            "plot_interaction_graph is deprecated and generates misleading graphs. "
            "User-item edges are now logged in the forward pass.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create a graph from CUMULATIVE interaction history (all rounds up to current)
        G = nx.Graph()
        
        # Add nodes (users and items)
        n_users = interaction_controller.n_users
        n_items = interaction_controller.n_items
        
        # Get CUMULATIVE interactions from all rounds up to current round
        # This shows the full interaction network, not just the current batch
        all_interactions = []
        for r in range(round_num + 1):
            round_interactions = interaction_controller._interaction_history.get(r, [])
            all_interactions.extend(round_interactions)
        
        print(f"[VISUALIZATION] Round {round_num}: Collected {len(all_interactions)} cumulative interactions from rounds 0-{round_num}")
        
        # Collect all agents involved in interactions
        involved_users = set()
        involved_items = set()
        
        for interaction in all_interactions:
            agent1_id = interaction['agent1']
            agent2_id = interaction['agent2']
            agent_type = interaction['type']
            
            if agent_type == 'user':
                involved_users.add(agent1_id)
                involved_users.add(agent2_id)
            elif agent_type == 'item':
                involved_items.add(agent1_id)
                involved_items.add(agent2_id)
        
        print(f"[VISUALIZATION] Cumulative network: {len(involved_users)} users, {len(involved_items)} items")
        
        # If no interactions yet, fall back to showing subset of agents
        if not involved_users and not involved_items:
            print(f"[VISUALIZATION] No interactions recorded yet, showing subset of agents")
            involved_users = set(range(min(20, n_users)))
            involved_items = set(range(min(10, n_items)))
        
        # Add user nodes (limit for visualization to avoid clutter)
        # Show up to 30 users to capture more of the network
        for user_id in list(involved_users)[:30]:
            contamination = contamination_scores.get(user_id, 0.0)
            is_attacker = user_id in interaction_controller.attacker_user_indices
            
            G.add_node(f"U{user_id:02d}", 
                      node_type='user',
                      contamination=contamination,
                      is_attacker=is_attacker,
                      agent_id=user_id)
        
        # Add item nodes (limit for visualization)
        # Show up to 15 items to capture more of the network
        for item_id in list(involved_items)[:15]:
            contamination = contamination_scores.get(f"item_{item_id}", 0.0)
            is_attacker = item_id in interaction_controller.attacker_item_indices
            
            G.add_node(f"I{item_id:02d}",
                      node_type='item', 
                      contamination=contamination,
                      is_attacker=is_attacker,
                      agent_id=item_id)
        
        # Add edges based on CUMULATIVE interactions
        edges_added = 0
        for interaction in all_interactions:
            agent1_id = interaction['agent1']
            agent2_id = interaction['agent2']
            agent_type = interaction['type']
            
            if agent_type == 'user':
                node1 = f"U{agent1_id:02d}"
                node2 = f"U{agent2_id:02d}"
            elif agent_type == 'item':
                node1 = f"I{agent1_id:02d}"
                node2 = f"I{agent2_id:02d}"
            else:
                continue
            
            # Only add edge if both nodes exist in graph
            if node1 in G.nodes() and node2 in G.nodes():
                if not G.has_edge(node1, node2):  # Avoid duplicate edges
                    G.add_edge(node1, node2)
                    edges_added += 1
        
        print(f"[VISUALIZATION] Round {round_num}: {len(G.nodes())} nodes, {edges_added} edges from cumulative interactions")
        
        # If still no edges (early rounds), add some user-item connections for visualization
        if edges_added == 0:
            user_nodes = [n for n in G.nodes() if n.startswith('U')]
            item_nodes = [n for n in G.nodes() if n.startswith('I')]
            
            np.random.seed(42 + round_num)
            for user_node in user_nodes[:10]:
                connected_items = np.random.choice(item_nodes, size=min(3, len(item_nodes)), replace=False)
                for item_node in connected_items:
                    G.add_edge(user_node, item_node)
            print(f"[VISUALIZATION] Added {len(G.edges())} placeholder edges for visualization")
        
        # Save plain text representation
        self._save_graph_text_representation(G, round_num, experiment_name)
        
        # Initialize fixed layout if not already done
        # This creates positions for ALL possible agents upfront
        if not self.layout_initialized:
            self.initialize_fixed_layout(n_users, n_items, seed=42)
        
        # Use the fixed layout - get positions only for nodes in current graph
        pos = {}
        for node in G.nodes():
            pos[node] = self.get_node_position(node)
        
        print(f"[VISUALIZATION] Using fixed layout for {len(pos)} nodes (positions are consistent across rounds)")
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Separate nodes by type and attacker status
        user_clean = [n for n in G.nodes() if n.startswith('U') and not G.nodes[n]['is_attacker']]
        user_attacker = [n for n in G.nodes() if n.startswith('U') and G.nodes[n]['is_attacker']]
        item_clean = [n for n in G.nodes() if n.startswith('I') and not G.nodes[n]['is_attacker']]
        item_attacker = [n for n in G.nodes() if n.startswith('I') and G.nodes[n]['is_attacker']]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Draw nodes with contamination-based coloring - FIX: Ensure proper coloring
        if user_clean:
            user_contamination = [G.nodes[n]['contamination'] for n in user_clean]
            print(f"[DEBUG] User contamination scores: {user_contamination}")  # Debug output
            
            # Use explicit colors based on contamination levels
            node_colors = []
            for contamination in user_contamination:
                if contamination > 0.7:
                    node_colors.append('darkblue')  # Highly contaminated
                elif contamination > 0.3:
                    node_colors.append('blue')      # Partially contaminated
                elif contamination > 0.0:
                    node_colors.append('lightblue') # Low contamination
                else:
                    node_colors.append('lightcyan') # Clean
            
            nx.draw_networkx_nodes(G, pos, nodelist=user_clean, 
                                 node_color=node_colors, node_shape='o',
                                 node_size=300, alpha=0.8, ax=ax)
        
        if user_attacker:
            nx.draw_networkx_nodes(G, pos, nodelist=user_attacker,
                                 node_color='red', node_shape='o',
                                 node_size=400, alpha=0.9, ax=ax)
        
        if item_clean:
            item_contamination = [G.nodes[n]['contamination'] for n in item_clean]
            print(f"[DEBUG] Item contamination scores: {item_contamination}")  # Debug output
            
            # Use explicit colors based on contamination levels
            node_colors = []
            for contamination in item_contamination:
                if contamination > 0.7:
                    node_colors.append('darkgreen')  # Highly contaminated
                elif contamination > 0.3:
                    node_colors.append('green')      # Partially contaminated
                elif contamination > 0.0:
                    node_colors.append('lightgreen') # Low contamination
                else:
                    node_colors.append('palegreen')  # Clean
                
            nx.draw_networkx_nodes(G, pos, nodelist=item_clean,
                                 node_color=node_colors, node_shape='s',
                                 node_size=200, alpha=0.8, ax=ax)
        
        if item_attacker:
            nx.draw_networkx_nodes(G, pos, nodelist=item_attacker,
                                 node_color='darkred', node_shape='s',
                                 node_size=300, alpha=0.9, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f'Agent Interaction Graph - Round {round_num}')
        ax.axis('off')
        
        # Add legend with contamination info
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcyan', 
                      markersize=10, label='Benign U (Clean)', alpha=0.8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Benign U (Low Contamination)', alpha=0.8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Compromised U (Partial)', alpha=0.8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                      markersize=10, label='Compromised U (High)', alpha=0.8),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Attacker U', alpha=0.9),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='palegreen', 
                      markersize=8, label='Benign I (Clean)', alpha=0.8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
                      markersize=8, label='Benign I (Low Contamination)', alpha=0.8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=8, label='Compromised I (Partial)', alpha=0.8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgreen', 
                      markersize=8, label='Compromised I (High)', alpha=0.8),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', 
                      markersize=8, label='Attacker I', alpha=0.9)
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add contamination statistics text
        stats_text = self._get_contamination_stats_text(G)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_interaction_graph_round_{round_num}")
        plt.close()
        
        # Save adjacency matrix visualization
        self._save_adjacency_matrix_plot(G, round_num, experiment_name)
        
        # Save mesh interaction plot (3D visualization)
        self._save_mesh_interaction_plot(G, round_num, experiment_name, contamination_scores)
        
        print(f"[INFO] Interaction graph saved for round {round_num}")
        print(f"[INFO] Adjacency matrix plot saved for round {round_num}")
        print(f"[INFO] Mesh interaction plot saved for round {round_num}")
        print(f"[INFO] Text representation saved for round {round_num}")
    
    def _get_contamination_stats_text(self, G: nx.Graph) -> str:
        """Generate contamination statistics text for the plot"""
        user_nodes = [n for n in G.nodes() if n.startswith('U')]
        item_nodes = [n for n in G.nodes() if n.startswith('I')]
        
        user_contamination = [G.nodes[n]['contamination'] for n in user_nodes]
        item_contamination = [G.nodes[n]['contamination'] for n in item_nodes]
        
        user_attackers = sum(1 for n in user_nodes if G.nodes[n]['is_attacker'])
        item_attackers = sum(1 for n in item_nodes if G.nodes[n]['is_attacker'])
        
        user_compromised = sum(1 for c in user_contamination if c > 0.3)
        item_compromised = sum(1 for c in item_contamination if c > 0.3)
        
        stats = f"""Graph Statistics:
Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}

Users: {len(user_nodes)} total
- Attackers: {user_attackers}
- Compromised: {user_compromised}
- Avg Contamination: {np.mean(user_contamination):.3f}

Items: {len(item_nodes)} total  
- Attackers: {item_attackers}
- Compromised: {item_compromised}
- Avg Contamination: {np.mean(item_contamination):.3f}"""
        
        return stats
    
    def _save_graph_text_representation(self, G: nx.Graph, round_num: int, experiment_name: str):
        """Save plain text representation of the interaction graph"""
        os.makedirs(self.output_dir, exist_ok=True)
        text_file = os.path.join(self.output_dir, f"{experiment_name}_interaction_graph_round_{round_num}.txt")
        
        with open(text_file, 'w+') as f:
            f.write(f"AGENT INTERACTION GRAPH - ROUND {round_num}\n")
            f.write("=" * 50 + "\n\n")
            
            # Graph statistics
            f.write(f"Graph Statistics:\n")
            f.write(f"  - Total Nodes: {G.number_of_nodes()}\n")
            f.write(f"  - Total Edges: {G.number_of_edges()}\n")
            f.write(f"  - Graph Density: {nx.density(G):.4f}\n\n")
            
            # Node information
            f.write("NODES:\n")
            f.write("-" * 30 + "\n")
            
            user_nodes = [n for n in G.nodes() if n.startswith('U')]
            item_nodes = [n for n in G.nodes() if n.startswith('I')]
            
            f.write(f"\nUser Agents ({len(user_nodes)}):\n")
            for node in sorted(user_nodes):
                node_data = G.nodes[node]
                contamination = node_data['contamination']
                # Determine status based on both attacker flag and contamination level
                if node_data['is_attacker']:
                    status = "ATTACKER"
                elif contamination > 0.7:
                    status = "HIGHLY_CONTAMINATED"
                elif contamination > 0.3:
                    status = "CONTAMINATED"
                elif contamination > 0.1:
                    status = "SLIGHTLY_CONTAMINATED"
                else:
                    status = "CLEAN"
                f.write(f"  {node}: {status}, Contamination={contamination:.4f}\n")
            
            f.write(f"\nItem Agents ({len(item_nodes)}):\n")
            for node in sorted(item_nodes):
                node_data = G.nodes[node]
                contamination = node_data['contamination']
                # Determine status based on both attacker flag and contamination level
                if node_data['is_attacker']:
                    status = "ATTACKER"
                elif contamination > 0.7:
                    status = "HIGHLY_CONTAMINATED"
                elif contamination > 0.3:
                    status = "CONTAMINATED"
                elif contamination > 0.1:
                    status = "SLIGHTLY_CONTAMINATED"
                else:
                    status = "CLEAN"
                f.write(f"  {node}: {status}, Contamination={contamination:.4f}\n")
            
            # Edge information
            f.write(f"\nEDGES ({G.number_of_edges()}):\n")
            f.write("-" * 30 + "\n")
            for edge in sorted(G.edges()):
                node1, node2 = edge
                node1_data = G.nodes[node1]
                node2_data = G.nodes[node2]
                
                # Determine interaction type based on contamination and attacker status
                if node1_data['is_attacker'] or node2_data['is_attacker']:
                    interaction_type = "ATTACKER_INVOLVED"
                elif node1_data['contamination'] > 0.3 or node2_data['contamination'] > 0.3:
                    interaction_type = "CONTAMINATED"
                else:
                    interaction_type = "CLEAN"
                
                f.write(f"  {node1} <--> {node2} ({interaction_type})\n")
            
            # Contamination summary
            f.write(f"\nCONTAMINATION SUMMARY:\n")
            f.write("-" * 30 + "\n")
            
            victim_users = [n for n in user_nodes if not G.nodes[n]['is_attacker']]
            victim_items = [n for n in item_nodes if not G.nodes[n]['is_attacker']]
            
            if victim_users:
                user_contaminations = [G.nodes[n]['contamination'] for n in victim_users]
                f.write(f"Victim Users (non-attacker):\n")
                f.write(f"  - Total: {len(victim_users)}\n")
                f.write(f"  - Average Contamination: {np.mean(user_contaminations):.4f}\n")
                f.write(f"  - Max Contamination: {np.max(user_contaminations):.4f}\n")
                f.write(f"  - Highly Contaminated (>0.7): {sum(1 for c in user_contaminations if c > 0.7)}\n")
                f.write(f"  - Contaminated (>0.3): {sum(1 for c in user_contaminations if c > 0.3)}\n")
                f.write(f"  - Slightly Contaminated (>0.1): {sum(1 for c in user_contaminations if c > 0.1)}\n")
                f.write(f"  - Clean (<=0.1): {sum(1 for c in user_contaminations if c <= 0.1)}\n")
            
            if victim_items:
                item_contaminations = [G.nodes[n]['contamination'] for n in victim_items]
                f.write(f"Victim Items (non-attacker):\n")
                f.write(f"  - Total: {len(victim_items)}\n")
                f.write(f"  - Average Contamination: {np.mean(item_contaminations):.4f}\n")
                f.write(f"  - Max Contamination: {np.max(item_contaminations):.4f}\n")
                f.write(f"  - Highly Contaminated (>0.7): {sum(1 for c in item_contaminations if c > 0.7)}\n")
                f.write(f"  - Contaminated (>0.3): {sum(1 for c in item_contaminations if c > 0.3)}\n")
                f.write(f"  - Slightly Contaminated (>0.1): {sum(1 for c in item_contaminations if c > 0.1)}\n")
                f.write(f"  - Clean (<=0.1): {sum(1 for c in item_contaminations if c <= 0.1)}\n")
            
            # Network analysis
            f.write(f"\nNETWORK ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            f.write(f"Top 5 Most Connected Nodes:\n")
            for node, centrality in top_central_nodes:
                node_data = G.nodes[node]
                status = "ATTACKER" if node_data['is_attacker'] else "CLEAN"
                f.write(f"  {node} ({status}): {centrality:.4f}\n")
            
            # Connected components
            components = list(nx.connected_components(G))
            f.write(f"\nConnected Components: {len(components)}\n")
            for i, component in enumerate(components):
                f.write(f"  Component {i+1}: {len(component)} nodes\n")
            
            # Adjacency matrix (for smaller graphs)
            if G.number_of_nodes() <= 30:
                f.write(f"\nADJACENCY MATRIX:\n")
                f.write("-" * 30 + "\n")
                
                all_nodes = sorted(G.nodes())
                f.write("     " + " ".join(f"{node:>4}" for node in all_nodes) + "\n")
                
                for node1 in all_nodes:
                    row = f"{node1:>4} "
                    for node2 in all_nodes:
                        if G.has_edge(node1, node2):
                            row += "   1"
                        else:
                            row += "   0"
                    f.write(row + "\n")
            
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Text representation saved: {text_file}")
    
    def _get_contamination_stats_text(self, G: nx.Graph) -> str:
        """Get contamination statistics text for the plot"""
        user_nodes = [n for n in G.nodes() if n.startswith('U')]
        item_nodes = [n for n in G.nodes() if n.startswith('I')]
        
        clean_users = [n for n in user_nodes if not G.nodes[n]['is_attacker']]
        clean_items = [n for n in item_nodes if not G.nodes[n]['is_attacker']]
        
        attacker_users = len([n for n in user_nodes if G.nodes[n]['is_attacker']])
        attacker_items = len([n for n in item_nodes if G.nodes[n]['is_attacker']])
        
        stats = f"Graph Stats:\n"
        stats += f"Users: {len(clean_users)} clean, {attacker_users} attackers\n"
        stats += f"Items: {len(clean_items)} clean, {attacker_items} attackers\n"
        stats += f"Edges: {G.number_of_edges()}\n"
        
        if clean_users:
            user_contaminations = [G.nodes[n]['contamination'] for n in clean_users]
            avg_user_cont = np.mean(user_contaminations)
            stats += f"Avg User Contamination: {avg_user_cont:.3f}\n"
        
        if clean_items:
            item_contaminations = [G.nodes[n]['contamination'] for n in clean_items]
            avg_item_cont = np.mean(item_contaminations)
            stats += f"Avg Item Contamination: {avg_item_cont:.3f}"
        
        return stats
    
    def plot_attack_effectiveness_comparison(self, results_dict: Dict[str, Any], experiment_name: str):
        """Compare attack effectiveness across different scenarios"""
        scenarios = list(results_dict.keys())
        metrics = ['accuracy_degradation', 'user_contamination_rate', 'item_contamination_rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[scenario].get(metric, 0) for scenario in scenarios]
            
            bars = axes[i].bar(range(len(scenarios)), values, alpha=0.7)
            axes[i].set_xlabel('Attack Scenario')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} by Scenario')
            axes[i].set_xticks(range(len(scenarios)))
            axes[i].set_xticklabels(scenarios, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_attack_effectiveness_comparison")
        plt.close()
    
    def create_summary_dashboard(self, metrics_collector, experiment_name: str):
        """Create comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Contamination timeline
        ax1 = fig.add_subplot(gs[0, :2])
        timeline = metrics_collector.get_contamination_timeline()
        ax1.plot(timeline['rounds'], timeline['user_contamination'], 
                marker='o', label='Users', linewidth=2)
        ax1.plot(timeline['rounds'], timeline['item_contamination'], 
                marker='s', label='Items', linewidth=2)
        ax1.set_title('Contamination Timeline')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Contaminated Agents')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance degradation
        ax2 = fig.add_subplot(gs[0, 2:])
        perf_timeline = metrics_collector.get_performance_timeline()
        ax2.plot(perf_timeline['rounds'], perf_timeline['accuracy'], 
                marker='o', label='Accuracy', linewidth=2)
        ax2.plot(perf_timeline['rounds'], perf_timeline['recall_at_5'], 
                marker='s', label='Recall@5', linewidth=2)
        ax2.set_title('Performance Degradation')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Summary statistics
        ax3 = fig.add_subplot(gs[1, :2])
        summary = metrics_collector.get_summary_statistics()
        
        stats_text = f"""
        Total Rounds: {summary.get('total_rounds', 0)}
        Initial Accuracy: {summary.get('initial_accuracy', 0):.3f}
        Final Accuracy: {summary.get('final_accuracy', 0):.3f}
        Accuracy Degradation: {summary.get('accuracy_degradation', 0):.3f}
        
        User Contamination Rate: {summary.get('user_contamination_rate', 0):.3f}
        Item Contamination Rate: {summary.get('item_contamination_rate', 0):.3f}
        
        Total Attackers: {summary.get('total_attackers', 0)}
        Attacker User Ratio: {summary.get('attacker_user_ratio', 0):.3f}
        Attacker Item Ratio: {summary.get('attacker_item_ratio', 0):.3f}
        """
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        ax3.set_title('Summary Statistics')
        ax3.axis('off')
        
        # 4. Contamination distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Get final round contamination scores
        final_user_contamination = []
        for user_id, user_metrics in metrics_collector.user_metrics.items():
            if user_id not in metrics_collector.attacker_user_indices and user_metrics:
                latest = user_metrics[-1]
                if latest.get('contamination'):
                    final_user_contamination.append(latest['contamination']['contamination_score'])
                else:
                    final_user_contamination.append(latest.get('evolution', {}).get('text_drift', 0.0))
        
        if final_user_contamination:
            ax4.hist(final_user_contamination, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_title('Final Contamination Score Distribution')
            ax4.set_xlabel('Contamination Score')
            ax4.set_ylabel('Number of Agents')
            ax4.grid(True, alpha=0.3)
        
        # 5. Attack progression heatmap (simplified)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Create simplified heatmap data
        rounds = timeline['rounds']
        contamination_data = np.array([timeline['user_contamination'], 
                                     timeline['item_contamination']])
        
        im = ax5.imshow(contamination_data, cmap='Reds', aspect='auto')
        ax5.set_title('Attack Progression Heatmap')
        ax5.set_xlabel('Round')
        ax5.set_ylabel('Agent Type')
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Users', 'Items'])
        ax5.set_xticks(range(0, len(rounds), max(1, len(rounds)//10)))
        ax5.set_xticklabels([rounds[i] for i in range(0, len(rounds), max(1, len(rounds)//10))])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Contaminated Agents')
        
        plt.suptitle(f'Attack Analysis Dashboard - {experiment_name}', fontsize=16)
        self._save_plot(fig, f"{experiment_name}_summary_dashboard")
        plt.close()
    
    def plot_round_interaction_graph(self, round_num: int, agents_data: Dict, 
                                   interactions: List[Dict], experiment_name: str):
        """Create detailed interaction graph for specific round"""
        G = nx.Graph()
        
        # Add nodes with contamination data
        for agent_id, agent_info in agents_data.items():
            node_type = agent_info.get('agent_type', 'unknown')
            contamination = agent_info.get('contamination', 0.0)
            is_attacker = agent_info.get('is_attacker', False)
            
            G.add_node(agent_id,
                      node_type=node_type,
                      contamination=contamination,
                      is_attacker=is_attacker)
        
        # Add edges from interactions
        for interaction in interactions:
            agent1 = interaction.get('agent1_id')
            agent2 = interaction.get('agent2_id')
            contamination_transfer = interaction.get('contamination_transfer', 0.0)
            
            if agent1 and agent2 and agent1 in G.nodes() and agent2 in G.nodes():
                G.add_edge(agent1, agent2, weight=contamination_transfer)
        
        # Use fixed layout for consistent positioning
        # Convert agent IDs to standard format and get positions
        pos = {}
        for node in G.nodes():
            # Try to get position from fixed layout
            # Handle different node ID formats (user_X, item_X, UXX, IXX)
            if node.startswith('user_'):
                user_id = int(node.replace('user_', ''))
                standard_node = f"U{user_id:02d}"
            elif node.startswith('item_'):
                item_id = int(node.replace('item_', ''))
                standard_node = f"I{item_id:02d}"
            else:
                standard_node = node
            
            pos[node] = self.get_node_position(standard_node)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Separate nodes by type and status
        user_clean = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'user' and not G.nodes[n].get('is_attacker')]
        user_attacker = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'user' and G.nodes[n].get('is_attacker')]
        item_clean = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'item' and not G.nodes[n].get('is_attacker')]
        item_attacker = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'item' and G.nodes[n].get('is_attacker')]
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v].get('weight', 0.1) for u, v in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=weights, edge_color='gray', ax=ax)
        
        # Draw nodes with contamination-based coloring
        if user_clean:
            user_contamination = [G.nodes[n]['contamination'] for n in user_clean]
            nx.draw_networkx_nodes(G, pos, nodelist=user_clean,
                                 node_color=user_contamination, node_shape='o',
                                 node_size=400, cmap='Reds', vmin=0, vmax=1, ax=ax)
        
        if user_attacker:
            nx.draw_networkx_nodes(G, pos, nodelist=user_attacker,
                                 node_color='darkred', node_shape='o',
                                 node_size=500, ax=ax)
        
        if item_clean:
            item_contamination = [G.nodes[n]['contamination'] for n in item_clean]
            nx.draw_networkx_nodes(G, pos, nodelist=item_clean,
                                 node_color=item_contamination, node_shape='s',
                                 node_size=300, cmap='Blues', vmin=0, vmax=1, ax=ax)
        
        if item_attacker:
            nx.draw_networkx_nodes(G, pos, nodelist=item_attacker,
                                 node_color='darkblue', node_shape='s',
                                 node_size=400, ax=ax)
        
        # Draw labels
        labels = {node: node.replace('user_', 'U').replace('item_', 'I') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Agent Interaction Network - Round {round_num}\n'
                    f'Nodes colored by contamination level', fontsize=14)
        ax.axis('off')
        
        # Add colorbar for contamination
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Contamination Level', rotation=270, labelpad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=12, label='Clean Users'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
                      markersize=12, label='Attacker Users'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                      markersize=10, label='Clean Items'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue',
                      markersize=10, label='Attacker Items')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_detailed_interaction_round_{round_num}")
        plt.close()
    
    def plot_agent_contamination_progression(self, agent_metrics: Dict, experiment_name: str):
        """Plot individual agent contamination over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot user agent contamination progression
        user_agents = {k: v for k, v in agent_metrics.items() if v['agent_type'] == 'user'}
        for agent_id, data in list(user_agents.items())[:10]:  # Limit to first 10 for readability
            ax1.plot(data['rounds'], data['contamination'], 
                    marker='o', linewidth=2, alpha=0.7, label=agent_id)
        
        ax1.set_title('User Agent Contamination Progression')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Contamination Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot item agent contamination progression
        item_agents = {k: v for k, v in agent_metrics.items() if v['agent_type'] == 'item'}
        for agent_id, data in list(item_agents.items())[:10]:  # Limit to first 10 for readability
            ax2.plot(data['rounds'], data['contamination'], 
                    marker='s', linewidth=2, alpha=0.7, label=agent_id)
        
        ax2.set_title('Item Agent Contamination Progression')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Contamination Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_agent_contamination_progression")
        plt.close()
    
    def plot_training_vs_eval_performance(self, training_metrics: List, 
                                        eval_metrics: List, experiment_name: str):
        """Compare training vs evaluation performance over rounds"""
        if not training_metrics and not eval_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Extract data
        train_rounds = [m['round'] for m in training_metrics if m.get('phase') == 'training']
        train_accuracy = [m['accuracy'] for m in training_metrics if m.get('phase') == 'training']
        train_recall = [m['recall@5'] for m in training_metrics if m.get('phase') == 'training']
        
        eval_rounds = [m['round'] for m in eval_metrics if m.get('phase') == 'evaluation']
        eval_accuracy = [m['accuracy'] for m in eval_metrics if m.get('phase') == 'evaluation']
        eval_recall = [m['recall@5'] for m in eval_metrics if m.get('phase') == 'evaluation']
        
        # Plot accuracy comparison
        if train_rounds and train_accuracy:
            axes[0, 0].plot(train_rounds, train_accuracy, 'o-', label='Training', linewidth=2)
        if eval_rounds and eval_accuracy:
            axes[0, 0].plot(eval_rounds, eval_accuracy, 's-', label='Evaluation', linewidth=2)
        axes[0, 0].set_title('Accuracy: Training vs Evaluation')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot recall comparison
        if train_rounds and train_recall:
            axes[0, 1].plot(train_rounds, train_recall, 'o-', label='Training', linewidth=2)
        if eval_rounds and eval_recall:
            axes[0, 1].plot(eval_rounds, eval_recall, 's-', label='Evaluation', linewidth=2)
        axes[0, 1].set_title('Recall@5: Training vs Evaluation')
        axes[0, 1].set_ylabel('Recall@5')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot performance degradation
        if train_accuracy and eval_accuracy:
            degradation = [t - e for t, e in zip(train_accuracy, eval_accuracy) if t and e]
            degradation_rounds = train_rounds[:len(degradation)]
            axes[1, 0].plot(degradation_rounds, degradation, 'r^-', linewidth=2)
            axes[1, 0].set_title('Performance Gap (Training - Evaluation)')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Accuracy Gap')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot combined metrics
        all_metrics = training_metrics + eval_metrics
        if all_metrics:
            rounds = [m['round'] for m in all_metrics]
            accuracy = [m['accuracy'] for m in all_metrics]
            phases = [m.get('phase', 'unknown') for m in all_metrics]
            
            train_mask = [p == 'training' for p in phases]
            eval_mask = [p == 'evaluation' for p in phases]
            
            if any(train_mask):
                train_r = [r for r, m in zip(rounds, train_mask) if m]
                train_a = [a for a, m in zip(accuracy, train_mask) if m]
                axes[1, 1].scatter(train_r, train_a, c='blue', marker='o', s=50, label='Training')
            
            if any(eval_mask):
                eval_r = [r for r, m in zip(rounds, eval_mask) if m]
                eval_a = [a for a, m in zip(accuracy, eval_mask) if m]
                axes[1, 1].scatter(eval_r, eval_a, c='red', marker='s', s=50, label='Evaluation')
            
            axes[1, 1].set_title('Performance Over Time')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, f"{experiment_name}_training_vs_eval_performance")
        plt.close()
    
    def create_animated_attack_progression(self, all_round_data: List[Dict], experiment_name: str):
        """Create animated visualization of attack spreading (placeholder for now)"""
        # This would create an animated GIF or video showing attack progression
        # For now, create a series of static plots
        
        print(f"[VISUALIZATION] Creating attack progression series for {len(all_round_data)} rounds...")
        
        for round_data in all_round_data:
            round_num = round_data['round']
            contamination_scores = round_data['contamination_scores']
            
            # Create a simplified progression plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract contamination data
            user_contamination = [v for k, v in contamination_scores.items() if k.startswith('user_')]
            item_contamination = [v for k, v in contamination_scores.items() if k.startswith('item_')]
            
            # Create bar plot
            categories = ['Users', 'Items']
            avg_contamination = [
                np.mean(user_contamination) if user_contamination else 0,
                np.mean(item_contamination) if item_contamination else 0
            ]
            max_contamination = [
                np.max(user_contamination) if user_contamination else 0,
                np.max(item_contamination) if item_contamination else 0
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, avg_contamination, width, label='Average Contamination', alpha=0.7)
            ax.bar(x + width/2, max_contamination, width, label='Max Contamination', alpha=0.7)
            
            ax.set_xlabel('Agent Type')
            ax.set_ylabel('Contamination Score')
            ax.set_title(f'Attack Progression - Round {round_num}')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._save_plot(fig, f"{experiment_name}_progression_round_{round_num:02d}")
            plt.close()
        
        print(f"[VISUALIZATION] Attack progression series created for experiment: {experiment_name}")
    
    def _save_adjacency_matrix_plot(self, G: nx.Graph, round_num: int, experiment_name: str):
        """Save adjacency matrix as a heatmap visualization"""
        try:
            # Get sorted node list - sort by node type first, then by numeric ID
            def sort_key(node):
                # Extract node type (U or I) and numeric ID
                node_type = node[0]  # 'U' or 'I'
                node_id = int(node[1:])  # Extract numeric part
                return (node_type, node_id)
            
            all_nodes = sorted(G.nodes(), key=sort_key)
            n_nodes = len(all_nodes)
            
            # Skip if graph is too large
            if n_nodes > 100:
                print(f"[ADJACENCY] Skipping adjacency matrix plot - graph too large ({n_nodes} nodes)")
                return
            
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(G, nodelist=all_nodes).todense()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(10, n_nodes * 0.3), max(8, n_nodes * 0.25)))
            
            # Plot heatmap
            im = ax.imshow(adj_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(n_nodes))
            ax.set_yticks(range(n_nodes))
            ax.set_xticklabels(all_nodes, rotation=90, fontsize=8)
            ax.set_yticklabels(all_nodes, fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Connection (0=No Edge, 1=Edge)', rotation=270, labelpad=20)
            
            # Add title
            ax.set_title(f'Adjacency Matrix - Round {round_num}\n{n_nodes} nodes, {G.number_of_edges()} edges')
            ax.set_xlabel('Agent ID')
            ax.set_ylabel('Agent ID')
            
            # Add grid
            ax.set_xticks(np.arange(n_nodes) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_nodes) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            self._save_plot(fig, f"{experiment_name}_adjacency_matrix_round_{round_num}")
            plt.close()
            
            print(f"[ADJACENCY] Adjacency matrix plot saved for round {round_num}")
            
        except Exception as e:
            print(f"[ADJACENCY] Error creating adjacency matrix plot: {e}")
    
    def _save_mesh_interaction_plot(self, G: nx.Graph, round_num: int, 
                                   experiment_name: str, contamination_scores: Dict):
        """Save 3D mesh interaction plot showing contamination spread"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Skip if graph is too large
            if G.number_of_nodes() > 100:
                print(f"[MESH] Skipping mesh plot - graph too large ({G.number_of_nodes()} nodes)")
                return
            
            # Create 3D figure
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get 2D layout positions - use fixed layout for consistency
            pos = {}
            for node in G.nodes():
                pos[node] = self.get_node_position(node)
            
            # Prepare node data
            user_nodes = [n for n in G.nodes() if n.startswith('U')]
            item_nodes = [n for n in G.nodes() if n.startswith('I')]
            
            # Plot user nodes
            for node in user_nodes:
                x, y = pos[node]
                z = G.nodes[node]['contamination']  # Use contamination as height
                
                if G.nodes[node]['is_attacker']:
                    color = 'red'
                    size = 100
                    marker = '^'
                else:
                    # Color based on contamination level
                    if z > 0.7:
                        color = 'darkblue'
                    elif z > 0.3:
                        color = 'blue'
                    elif z > 0.0:
                        color = 'lightblue'
                    else:
                        color = 'lightcyan'
                    size = 60
                    marker = 'o'
                
                ax.scatter(x, y, z, c=color, marker=marker, s=size, alpha=0.8, edgecolors='black', linewidths=0.5)
            
            # Plot item nodes
            for node in item_nodes:
                x, y = pos[node]
                z = G.nodes[node]['contamination']  # Use contamination as height
                
                if G.nodes[node]['is_attacker']:
                    color = 'darkred'
                    size = 80
                    marker = 's'
                else:
                    # Color based on contamination level
                    if z > 0.7:
                        color = 'darkgreen'
                    elif z > 0.3:
                        color = 'green'
                    elif z > 0.0:
                        color = 'lightgreen'
                    else:
                        color = 'palegreen'
                    size = 50
                    marker = 's'
                
                ax.scatter(x, y, z, c=color, marker=marker, s=size, alpha=0.8, edgecolors='black', linewidths=0.5)
            
            # Draw edges as 3D lines
            for edge in G.edges():
                node1, node2 = edge
                x1, y1 = pos[node1]
                z1 = G.nodes[node1]['contamination']
                x2, y2 = pos[node2]
                z2 = G.nodes[node2]['contamination']
                
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'gray', alpha=0.3, linewidth=0.5)
            
            # Set labels and title
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Contamination Level')
            ax.set_title(f'3D Mesh Interaction Plot - Round {round_num}\nHeight = Contamination Level')
            
            # Set z-axis limits
            ax.set_zlim(0, 1)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                          markersize=10, label='Attacker User', alpha=0.8),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                          markersize=8, label='Highly Contaminated User', alpha=0.8),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                          markersize=8, label='Contaminated User', alpha=0.8),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcyan', 
                          markersize=8, label='Clean User', alpha=0.8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', 
                          markersize=8, label='Attacker Item', alpha=0.8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkgreen', 
                          markersize=7, label='Highly Contaminated Item', alpha=0.8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                          markersize=7, label='Contaminated Item', alpha=0.8),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='palegreen', 
                          markersize=7, label='Clean Item', alpha=0.8),
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # Adjust viewing angle
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            self._save_plot(fig, f"{experiment_name}_mesh_interaction_3d_round_{round_num}")
            plt.close()
            
            print(f"[MESH] 3D mesh interaction plot saved for round {round_num}")
            
        except ImportError:
            print(f"[MESH] 3D plotting not available - install matplotlib with 3D support")
        except Exception as e:
            print(f"[MESH] Error creating mesh interaction plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_plot(self, fig, filename: str):
        """Save plot to file"""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{filename}.{self.plot_format}")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"Plot saved: {filepath}")


def create_attack_visualizations(metrics_collector, interaction_controller, 
                               config: Dict[str, Any], experiment_name: str):
    """Create all attack visualizations"""
    viz = AttackVisualization(config.get('visualization', {}))
    
    # Generate all plots
    viz.plot_contamination_timeline(metrics_collector, experiment_name)
    viz.plot_performance_degradation(metrics_collector, experiment_name)
    viz.plot_contamination_heatmap(metrics_collector, experiment_name)
    
    # NOTE: plot_interaction_graph removed - was generating misleading graphs
    # The old implementation only showed attack propagation edges (user-user, item-item)
    # but not the actual user-item interactions from the forward pass.
    # User-item edges are now properly logged in the forward pass via record_interaction().
    
    # Create summary dashboard
    viz.create_summary_dashboard(metrics_collector, experiment_name)
    
    print(f"All visualizations created for experiment: {experiment_name}")