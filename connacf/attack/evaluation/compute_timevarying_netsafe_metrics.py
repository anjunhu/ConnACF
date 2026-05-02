#!/usr/bin/env python3
"""
NetSafe Time-Varying Metrics Computation for ConnaCF

Computes NetSafe metrics at each turn as the communication graph evolves:
1. Network Efficiency (NE): Global efficiency - changes as edges form/break
2. Eigenvector Centrality (EC): Attacker influence - changes with topology
3. Attack Path Vulnerability (APV): Path compromise - changes with graph structure

Unlike compute_static_netsafe_metrics.py which computes metrics once for fixed topologies,
this version tracks how metrics evolve over time in the dynamic ConnaCF system.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import json
import os


@dataclass
class TimeVaryingNetSafeMetrics:
    """Container for NetSafe metrics at a specific turn."""
    turn: int
    topology_snapshot: str  # Description of graph at this turn
    num_nodes: int
    num_edges: int
    num_attackers: int
    graph_density: float
    
    # The three core time-varying metrics
    network_efficiency: float  # NE(t)
    eigenvector_centrality: Dict[str, float] = field(default_factory=dict)  # EC(t) per node
    mean_attacker_centrality: float = 0.0  # Mean EC(t) for attackers
    attack_path_vulnerability: float = 0.0  # APV(t)
    
    # Additional time-varying metrics
    attacker_degree: int = 0  # Number of connections attacker has
    attacker_betweenness: float = 0.0  # Betweenness centrality of attacker
    
    def to_dict(self) -> Dict:
        return {
            'turn': self.turn,
            'topology_snapshot': self.topology_snapshot,
            'graph_properties': {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'num_attackers': self.num_attackers,
                'graph_density': round(self.graph_density, 4)
            },
            'netsafe_metrics': {
                'network_efficiency_NE': round(self.network_efficiency, 4),
                'mean_attacker_eigenvector_centrality_EC': round(self.mean_attacker_centrality, 4),
                'attack_path_vulnerability_APV': round(self.attack_path_vulnerability, 4),
                'attacker_degree': self.attacker_degree,
                'attacker_betweenness': round(self.attacker_betweenness, 4)
            },
            'per_node_eigenvector_centrality': {k: round(v, 4) for k, v in self.eigenvector_centrality.items()}
        }


def compute_shortest_paths(adj: np.ndarray) -> np.ndarray:
    """Compute all-pairs shortest paths using Floyd-Warshall."""
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                dist[i, j] = 1
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    return dist


def compute_network_efficiency(dist_matrix: np.ndarray) -> float:
    """
    Compute Network Efficiency (NE) at time t.
    
    E_NE(G_t) = (1 / |V|(|V|-1)) * Σ_{i≠j} (1 / d_ij(t))
    
    TIME-VARYING: Changes as edges are added/removed in the communication graph.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.0
    
    total_efficiency = 0.0
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i, j] < np.inf and dist_matrix[i, j] > 0:
                total_efficiency += 1.0 / dist_matrix[i, j]
    
    return total_efficiency / (n * (n - 1))


def compute_eigenvector_centrality(adj: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """
    Compute Eigenvector Centrality at time t.
    
    EC(t) measures node influence based on the graph structure at time t.
    
    TIME-VARYING: As the adjacency matrix changes, so does the principal eigenvector.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    
    A = adj.astype(float)
    
    # Check if graph is disconnected or has no edges
    if np.sum(A) == 0:
        return np.zeros(n)
    
    # Use numpy's eigenvalue solver for accuracy
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Get index of largest eigenvalue
        max_idx = np.argmax(np.real(eigenvalues))
        x = np.real(eigenvectors[:, max_idx])
        # Ensure all positive (eigenvector can be negated)
        if np.sum(x) < 0:
            x = -x
    except:
        # Fallback to power iteration
        x = np.ones(n) / np.sqrt(n)
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-10:
                break
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
    
    # Normalize to [0, 1]
    x = np.abs(x)  # Ensure positive
    if np.max(x) > 0:
        x = x / np.max(x)
    
    return x


def compute_betweenness_centrality(dist_matrix: np.ndarray, node_idx: int) -> float:
    """
    Compute betweenness centrality for a specific node at time t.
    
    BC(v,t) = Σ_{s≠v≠t} (σ_st(v) / σ_st)
    
    Where σ_st is the number of shortest paths from s to t,
    and σ_st(v) is the number of those paths passing through v.
    
    TIME-VARYING: Changes as shortest paths change with graph evolution.
    """
    n = dist_matrix.shape[0]
    if n <= 2:
        return 0.0
    
    betweenness = 0.0
    
    for s in range(n):
        for t in range(n):
            if s == t or s == node_idx or t == node_idx:
                continue
            if dist_matrix[s, t] == np.inf:
                continue
            
            # Check if node_idx is on shortest path from s to t
            if abs(dist_matrix[s, node_idx] + dist_matrix[node_idx, t] - dist_matrix[s, t]) < 1e-9:
                betweenness += 1.0
    
    # Normalize
    max_betweenness = (n - 1) * (n - 2)
    return betweenness / max_betweenness if max_betweenness > 0 else 0.0


def compute_apv(dist_matrix: np.ndarray, attacker_indices: Set[int]) -> float:
    """
    Compute Attack Path Vulnerability (APV) at time t.
    
    APV(t) = (Σ_{i≠j} δ_atk(d_ij(t))) / (|V|(|V|-1))
    
    TIME-VARYING: As the graph structure changes, different paths may pass through attackers.
    """
    n = dist_matrix.shape[0]
    if n <= 1 or not attacker_indices:
        return 0.0
    
    compromised_paths = 0
    total_paths = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dist_matrix[i, j] == np.inf:
                continue
            
            total_paths += 1
            
            # Path is compromised if source or target is attacker
            if i in attacker_indices or j in attacker_indices:
                compromised_paths += 1
                continue
            
            # Check intermediate nodes on shortest path
            path_compromised = False
            for k in attacker_indices:
                if k == i or k == j:
                    continue
                if abs(dist_matrix[i, k] + dist_matrix[k, j] - dist_matrix[i, j]) < 1e-9:
                    path_compromised = True
                    break
            
            if path_compromised:
                compromised_paths += 1
    
    return compromised_paths / total_paths if total_paths > 0 else 0.0


def compute_metrics_for_turn(
    turn: int,
    adj: np.ndarray,
    agent_names: List[str],
    attacker_indices: Set[int],
    topology_description: Optional[str] = None
) -> TimeVaryingNetSafeMetrics:
    """
    Compute all NetSafe metrics for a specific turn.
    
    This captures the instantaneous state of the communication graph at turn t.
    """
    n = adj.shape[0]
    num_edges = int(np.sum(adj) // 2)
    graph_density = (2 * num_edges) / (n * (n - 1)) if n > 1 else 0
    
    # Compute shortest paths at this turn
    dist_matrix = compute_shortest_paths(adj)
    
    # 1. Network Efficiency (NE) at time t
    ne = compute_network_efficiency(dist_matrix)
    
    # 2. Eigenvector Centrality (EC) at time t
    ec_scores = compute_eigenvector_centrality(adj)
    ec_dict = {agent_names[i]: float(ec_scores[i]) for i in range(n)}
    
    # Mean attacker centrality
    attacker_ec = [ec_scores[i] for i in attacker_indices]
    mean_attacker_ec = np.mean(attacker_ec) if attacker_ec else 0.0
    
    # 3. Attack Path Vulnerability (APV) at time t
    apv = compute_apv(dist_matrix, attacker_indices)
    
    # Additional metrics
    attacker_degree = int(np.sum(adj[list(attacker_indices)[0]])) if attacker_indices else 0
    attacker_betweenness = compute_betweenness_centrality(dist_matrix, list(attacker_indices)[0]) if attacker_indices else 0.0
    
    if topology_description is None:
        topology_description = f"Turn {turn} snapshot"
    
    return TimeVaryingNetSafeMetrics(
        turn=turn,
        topology_snapshot=topology_description,
        num_nodes=n,
        num_edges=num_edges,
        num_attackers=len(attacker_indices),
        graph_density=graph_density,
        network_efficiency=ne,
        eigenvector_centrality=ec_dict,
        mean_attacker_centrality=mean_attacker_ec,
        attack_path_vulnerability=apv,
        attacker_degree=attacker_degree,
        attacker_betweenness=attacker_betweenness
    )


def analyze_metric_evolution(metrics_over_time: List[TimeVaryingNetSafeMetrics]) -> Dict:
    """
    Analyze how metrics evolve over time.
    
    Returns statistics about metric changes:
    - Mean, std, min, max for each metric
    - Trend (increasing/decreasing/stable)
    - Volatility (how much it changes)
    """
    if not metrics_over_time:
        return {}
    
    ne_values = [m.network_efficiency for m in metrics_over_time]
    ec_values = [m.mean_attacker_centrality for m in metrics_over_time]
    apv_values = [m.attack_path_vulnerability for m in metrics_over_time]
    degree_values = [m.attacker_degree for m in metrics_over_time]
    
    def compute_stats(values):
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable',
            'volatility': float(np.std(np.diff(values))) if len(values) > 1 else 0.0
        }
    
    return {
        'network_efficiency': compute_stats(ne_values),
        'attacker_centrality': compute_stats(ec_values),
        'attack_path_vulnerability': compute_stats(apv_values),
        'attacker_degree': compute_stats(degree_values),
        'num_turns': len(metrics_over_time)
    }


def load_adjacency_matrices_from_results(results_dir: str, task_id: int = 0) -> List[Tuple[int, np.ndarray]]:
    """
    Load adjacency matrices from ConnaCF attack results.
    
    Looks for turn_*.json files and extracts adjacency matrices.
    Returns list of (turn_number, adjacency_matrix) tuples.
    """
    task_dir = os.path.join(results_dir, f"task_{task_id}")
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    adjacency_matrices = []
    
    # Find all turn files
    turn_files = sorted([f for f in os.listdir(task_dir) if f.startswith('turn_') and f.endswith('.json')])
    
    for turn_file in turn_files:
        turn_num = int(turn_file.split('_')[1].split('.')[0])
        
        with open(os.path.join(task_dir, turn_file), 'r') as f:
            turn_data = json.load(f)
        
        # Extract adjacency matrix if available
        if 'adjacency_matrix' in turn_data:
            adj = np.array(turn_data['adjacency_matrix'])
            adjacency_matrices.append((turn_num, adj))
    
    return adjacency_matrices


def main():
    """
    Example: Compute time-varying NetSafe metrics from ConnaCF results.
    
    This demonstrates how metrics evolve as the communication graph changes.
    """
    print("=" * 70)
    print("TIME-VARYING NETSAFE METRICS FOR CONNACF")
    print("=" * 70)
    print()
    print("This script computes NetSafe metrics at each turn as the")
    print("communication graph evolves in the ConnaCF system.")
    print()
    
    # Example: Load from attack results
    # results_dir = "attack_results_subset_small/CDs-100user-dense"
    # task_id = 0
    
    # For demonstration, create synthetic time-varying graphs
    agent_names = ["Manager", "Analyst", "Searcher", "Reflector"]
    n = len(agent_names)
    attacker_indices = {1}  # Analyst is attacker
    
    print(f"Agents: {agent_names}")
    print(f"Attacker: {agent_names[1]} (index {1})")
    print()
    
    # Simulate graph evolution over 5 turns
    print("Simulating graph evolution over 5 turns...")
    print()
    
    metrics_over_time = []
    
    # Turn 0: Star topology (initial)
    adj_t0 = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ])
    metrics_t0 = compute_metrics_for_turn(0, adj_t0, agent_names, attacker_indices, "Star (initial)")
    metrics_over_time.append(metrics_t0)
    
    # Turn 1: Analyst connects to Searcher
    adj_t1 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    metrics_t1 = compute_metrics_for_turn(1, adj_t1, agent_names, attacker_indices, "Analyst→Searcher added")
    metrics_over_time.append(metrics_t1)
    
    # Turn 2: Analyst connects to Reflector
    adj_t2 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ])
    metrics_t2 = compute_metrics_for_turn(2, adj_t2, agent_names, attacker_indices, "Analyst→Reflector added")
    metrics_over_time.append(metrics_t2)
    
    # Turn 3: Searcher connects to Reflector (forming more paths)
    adj_t3 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ])
    metrics_t3 = compute_metrics_for_turn(3, adj_t3, agent_names, attacker_indices, "Searcher→Reflector added")
    metrics_over_time.append(metrics_t3)
    
    # Turn 4: Full mesh
    adj_t4 = np.ones((n, n), dtype=int)
    np.fill_diagonal(adj_t4, 0)
    metrics_t4 = compute_metrics_for_turn(4, adj_t4, agent_names, attacker_indices, "Full mesh")
    metrics_over_time.append(metrics_t4)
    
    # Print results for each turn
    for metrics in metrics_over_time:
        print("-" * 70)
        print(f"TURN {metrics.turn}: {metrics.topology_snapshot}")
        print("-" * 70)
        print(f"Edges: {metrics.num_edges}, Density: {metrics.graph_density:.4f}")
        print(f"\nNetSafe Metrics:")
        print(f"  NE(t={metrics.turn}):  {metrics.network_efficiency:.4f}")
        print(f"  EC(t={metrics.turn}):  {metrics.mean_attacker_centrality:.4f} (attacker)")
        print(f"  APV(t={metrics.turn}): {metrics.attack_path_vulnerability:.4f}")
        print(f"  Attacker degree: {metrics.attacker_degree}")
        print(f"  Attacker betweenness: {metrics.attacker_betweenness:.4f}")
        print()
    
    # Analyze evolution
    print("=" * 70)
    print("METRIC EVOLUTION ANALYSIS")
    print("=" * 70)
    evolution = analyze_metric_evolution(metrics_over_time)
    
    for metric_name, stats in evolution.items():
        if metric_name == 'num_turns':
            continue
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Trend: {stats['trend']}")
        print(f"  Volatility: {stats['volatility']:.4f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
TIME-VARYING BEHAVIOR:

1. Network Efficiency (NE):
   - Increases as more edges are added (graph becomes more connected)
   - Reflects how quickly information can spread at each turn
   
2. Eigenvector Centrality (EC):
   - Changes as attacker's position in network evolves
   - Initially low (peripheral), increases as attacker gains connections
   
3. Attack Path Vulnerability (APV):
   - Increases as attacker becomes more central
   - Shows how many communication paths are compromised at each turn

RECOMMENDATION: Track these metrics over time to understand:
- When the attacker becomes most dangerous (high EC, high APV)
- How network structure affects attack propagation
- Whether defensive measures reduce vulnerability over time
""")
    
    # Save results
    output_file = "attack_metrics/timevarying_netsafe_example.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'agents': agent_names,
        'attacker': agent_names[1],
        'attacker_index': 1,
        'metrics_per_turn': [m.to_dict() for m in metrics_over_time],
        'evolution_analysis': evolution,
        'description': 'Time-varying NetSafe metrics showing how graph structure affects vulnerability over time'
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
