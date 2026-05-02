#!/usr/bin/env python3
"""
Communication Matrix Visualization Tool

Plots the communication adjacency matrix between user and item agents
at each turn or cumulatively across the experiment.

Usage:
    python tools/plot_communication_matrix.py \
        --graph_dir macf_output/experiment/communication_graph \
        --output communication_matrix.png \
        --turn 5  # Optional: specific turn, or omit for cumulative
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_turn_data(graph_dir: str, turn: Optional[int] = None) -> Dict:
    """
    Load communication graph data from turn JSON files.
    
    Args:
        graph_dir: Path to communication_graph directory
        turn: Specific turn to load, or None for all turns
        
    Returns:
        Dict with aggregated edge data
    """
    graph_path = Path(graph_dir)
    
    if turn is not None:
        # Load specific turn
        turn_file = graph_path / f"turn_{turn:04d}.json"
        if not turn_file.exists():
            logger.error(f"Turn file not found: {turn_file}")
            return {}
        
        with open(turn_file, 'r') as f:
            data = json.load(f)
        
        return {
            'turns': [data],
            'global_turn': turn
        }
    else:
        # Load all turns
        turn_files = sorted(graph_path.glob("turn_*.json"))
        if not turn_files:
            logger.error(f"No turn files found in {graph_dir}")
            return {}
        
        turns = []
        for tf in turn_files:
            with open(tf, 'r') as f:
                turns.append(json.load(f))
        
        return {
            'turns': turns,
            'global_turn': None  # Cumulative
        }


def build_adjacency_matrix(
    data: Dict,
    cumulative: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build adjacency matrix from turn data.
    
    Args:
        data: Turn data from load_turn_data
        cumulative: If True, aggregate all turns. If False, use last turn only.
        
    Returns:
        Tuple of (matrix, user_labels, item_labels)
    """
    turns = data.get('turns', [])
    if not turns:
        return np.array([]), [], []
    
    # Collect all unique agents
    all_users = set()
    all_items = set()
    
    for turn in turns:
        all_users.update(turn.get('active_user_agents', []))
        all_items.update(turn.get('active_item_agents', []))
    
    user_list = sorted(list(all_users))
    item_list = sorted(list(all_items))
    
    if not user_list or not item_list:
        return np.array([]), [], []
    
    # Create user/item index mappings
    user_idx = {u: i for i, u in enumerate(user_list)}
    item_idx = {it: i for i, it in enumerate(item_list)}
    
    # Build matrix (users as rows, items as columns)
    matrix = np.zeros((len(user_list), len(item_list)))
    
    # Aggregate edges
    turns_to_process = turns if cumulative else [turns[-1]]
    
    for turn in turns_to_process:
        edges = turn.get('edges', [])
        for edge in edges:
            src = edge.get('source_agent_id', '')
            tgt = edge.get('target_agent_id', '')
            src_type = edge.get('source_type', '')
            tgt_type = edge.get('target_type', '')
            
            # User -> Item edge
            if src_type == 'user' and tgt_type == 'item':
                if src in user_idx and tgt in item_idx:
                    matrix[user_idx[src], item_idx[tgt]] += 1
            
            # Item -> User edge (transpose direction)
            elif src_type == 'item' and tgt_type == 'user':
                if tgt in user_idx and src in item_idx:
                    matrix[user_idx[tgt], item_idx[src]] += 1
    
    # Simplify labels (extract IDs)
    user_labels = [u.replace('user_agent_', 'U') for u in user_list]
    item_labels = [it.replace('item_agent_', 'I') for it in item_list]
    
    return matrix, user_labels, item_labels


def plot_communication_matrix(
    matrix: np.ndarray,
    user_labels: List[str],
    item_labels: List[str],
    output_file: str,
    title: str = "Agent Communication Matrix",
    global_turn: Optional[int] = None
):
    """
    Plot the communication adjacency matrix as a heatmap.
    
    Args:
        matrix: Adjacency matrix (users x items)
        user_labels: Labels for user agents (rows)
        item_labels: Labels for item agents (columns)
        output_file: Output file path
        title: Plot title
        global_turn: Turn number for title (None for cumulative)
    """
    if matrix.size == 0:
        logger.error("Empty matrix, cannot plot")
        return
    
    # Determine figure size based on matrix dimensions
    n_users, n_items = matrix.shape
    fig_width = max(10, min(20, n_items * 0.3 + 2))
    fig_height = max(8, min(16, n_users * 0.3 + 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create custom colormap (white -> light blue -> dark blue)
    colors = ['#ffffff', '#e3f2fd', '#90caf9', '#42a5f5', '#1976d2', '#0d47a1']
    cmap = LinearSegmentedColormap.from_list('comm', colors)
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Communication Count', fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(item_labels)))
    ax.set_yticks(np.arange(len(user_labels)))
    
    # Only show labels if not too many
    if len(item_labels) <= 50:
        ax.set_xticklabels(item_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel(f'Item Agents ({len(item_labels)} total)', fontsize=11)
    
    if len(user_labels) <= 50:
        ax.set_yticklabels(user_labels, fontsize=8)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel(f'User Agents ({len(user_labels)} total)', fontsize=11)
    
    # Labels
    if len(item_labels) <= 50:
        ax.set_xlabel('Item Agents', fontsize=11)
    if len(user_labels) <= 50:
        ax.set_ylabel('User Agents', fontsize=11)
    
    # Title
    if global_turn is not None:
        full_title = f"{title}\nTurn {global_turn}"
    else:
        full_title = f"{title}\n(Cumulative)"
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(-.5, len(item_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(user_labels), 1), minor=True)
    ax.grid(which='minor', color='#e0e0e0', linestyle='-', linewidth=0.5)
    
    # Add statistics annotation
    total_edges = int(matrix.sum())
    non_zero = np.count_nonzero(matrix)
    density = non_zero / matrix.size if matrix.size > 0 else 0
    
    stats_text = f"Total edges: {total_edges}\nActive pairs: {non_zero}\nDensity: {density:.2%}"
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Communication matrix saved to: {output_file}")
    plt.close()


def plot_temporal_evolution(
    graph_dir: str,
    output_file: str,
    max_turns: int = 20
):
    """
    Plot how communication density evolves over turns.
    
    Args:
        graph_dir: Path to communication_graph directory
        output_file: Output file path
        max_turns: Maximum number of turns to show
    """
    graph_path = Path(graph_dir)
    turn_files = sorted(graph_path.glob("turn_*.json"))[:max_turns]
    
    if not turn_files:
        logger.error("No turn files found")
        return
    
    turns = []
    edge_counts = []
    user_counts = []
    item_counts = []
    densities = []
    
    for tf in turn_files:
        with open(tf, 'r') as f:
            data = json.load(f)
        
        turn_num = data.get('global_turn', 0)
        n_edges = data.get('num_edges', len(data.get('edges', [])))
        n_users = data.get('num_user_agents', len(data.get('active_user_agents', [])))
        n_items = data.get('num_item_agents', len(data.get('active_item_agents', [])))
        
        turns.append(turn_num)
        edge_counts.append(n_edges)
        user_counts.append(n_users)
        item_counts.append(n_items)
        
        # Calculate density
        max_edges = n_users * n_items * 2  # Bidirectional
        density = n_edges / max_edges if max_edges > 0 else 0
        densities.append(density)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Edge count over time
    ax1 = axes[0, 0]
    ax1.plot(turns, edge_counts, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Turn', fontsize=11)
    ax1.set_ylabel('Number of Edges', fontsize=11)
    ax1.set_title('Communication Edges per Turn', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Agent counts over time
    ax2 = axes[0, 1]
    ax2.plot(turns, user_counts, 'g-s', linewidth=2, markersize=6, label='User Agents')
    ax2.plot(turns, item_counts, 'r-^', linewidth=2, markersize=6, label='Item Agents')
    ax2.set_xlabel('Turn', fontsize=11)
    ax2.set_ylabel('Number of Agents', fontsize=11)
    ax2.set_title('Active Agents per Turn', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Density over time
    ax3 = axes[1, 0]
    ax3.plot(turns, densities, 'm-d', linewidth=2, markersize=6)
    ax3.set_xlabel('Turn', fontsize=11)
    ax3.set_ylabel('Communication Density', fontsize=11)
    ax3.set_title('Graph Density per Turn', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative edges
    ax4 = axes[1, 1]
    cumulative_edges = np.cumsum(edge_counts)
    ax4.fill_between(turns, cumulative_edges, alpha=0.3, color='blue')
    ax4.plot(turns, cumulative_edges, 'b-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Turn', fontsize=11)
    ax4.set_ylabel('Cumulative Edges', fontsize=11)
    ax4.set_title('Cumulative Communication', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Communication Graph Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Temporal evolution plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot communication matrix between MACF agents'
    )
    parser.add_argument(
        '--graph_dir', type=str, required=True,
        help='Path to communication_graph directory'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='communication_matrix.png',
        help='Output file path'
    )
    parser.add_argument(
        '--turn', type=int, default=None,
        help='Specific turn to visualize (omit for cumulative)'
    )
    parser.add_argument(
        '--temporal', action='store_true',
        help='Plot temporal evolution instead of matrix'
    )
    parser.add_argument(
        '--title', type=str, default='Agent Communication Matrix',
        help='Plot title'
    )
    
    args = parser.parse_args()
    
    if args.temporal:
        # Plot temporal evolution
        plot_temporal_evolution(args.graph_dir, args.output)
    else:
        # Load data
        data = load_turn_data(args.graph_dir, args.turn)
        if not data:
            logger.error("Failed to load data")
            return
        
        # Build matrix
        matrix, user_labels, item_labels = build_adjacency_matrix(
            data, cumulative=(args.turn is None)
        )
        
        if matrix.size == 0:
            logger.error("No communication data found")
            return
        
        # Plot
        plot_communication_matrix(
            matrix, user_labels, item_labels,
            args.output, args.title, args.turn
        )
        
        # Print summary
        print(f"\nMatrix shape: {matrix.shape}")
        print(f"Total edges: {int(matrix.sum())}")
        print(f"Non-zero pairs: {np.count_nonzero(matrix)}")
        print(f"Density: {np.count_nonzero(matrix) / matrix.size:.2%}")


if __name__ == "__main__":
    main()
