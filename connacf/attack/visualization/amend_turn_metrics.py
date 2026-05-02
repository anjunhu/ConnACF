#!/usr/bin/env python3
"""
Amend turn_*.json files with active agent counts and create MBD stacked area plots.

This script:
1. Reads user_metrics.json and item_metrics.json to get actual active agent counts
2. Amends all turn_*.json files with the correct denominators
3. Creates stacked area plots for the MBD (Bias & Misinformation Dissemination) row
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_metrics_files(task_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load user_metrics, item_metrics, and summary files."""
    # Find the metrics files
    user_metrics_files = glob.glob(os.path.join(task_dir, "*_user_metrics.json"))
    item_metrics_files = glob.glob(os.path.join(task_dir, "*_item_metrics.json"))
    summary_files = glob.glob(os.path.join(task_dir, "*_summary.json"))
    
    user_metrics = {}
    item_metrics = {}
    summary = {}
    
    if user_metrics_files:
        with open(user_metrics_files[0]) as f:
            user_metrics = json.load(f)
    
    if item_metrics_files:
        with open(item_metrics_files[0]) as f:
            item_metrics = json.load(f)
    
    if summary_files:
        with open(summary_files[0]) as f:
            summary = json.load(f)
    
    return user_metrics, item_metrics, summary


def get_attacker_indices(summary: Dict, user_metrics: Dict, item_metrics: Dict) -> Tuple[set, set]:
    """Extract attacker indices from summary or metrics files."""
    attacker_user_indices = set()
    attacker_item_indices = set()
    
    # Try to get from summary config first
    if 'config' in summary:
        config = summary['config']
        attacker_user_indices = set(config.get('attacker_user_indices', []))
        attacker_item_indices = set(config.get('attacker_item_indices', []))
    
    # If not found in summary, extract from metrics files
    if not attacker_user_indices and user_metrics:
        for user_id, metrics_list in user_metrics.items():
            if metrics_list and metrics_list[0].get('is_attacker', False):
                attacker_user_indices.add(int(user_id))
    
    if not attacker_item_indices and item_metrics:
        for item_id, metrics_list in item_metrics.items():
            if metrics_list and metrics_list[0].get('is_attacker', False):
                attacker_item_indices.add(int(item_id))
    
    return attacker_user_indices, attacker_item_indices


def amend_turn_files(task_dir: str, user_metrics: Dict, item_metrics: Dict, 
                     attacker_user_indices: set, attacker_item_indices: set) -> List[Dict]:
    """Amend all turn_*.json files with active agent counts and rename fields."""
    turn_files = sorted(glob.glob(os.path.join(task_dir, "turn_*.json")), 
                       key=lambda x: int(os.path.basename(x).replace('turn_', '').replace('.json', '')))
    
    # Calculate active agent counts
    active_users_total = len(user_metrics)
    active_items_total = len(item_metrics)
    
    # Count attackers that are actually in the metrics (active attackers)
    active_attacker_users = len(set(int(k) for k in user_metrics.keys()) & attacker_user_indices)
    active_attacker_items = len(set(int(k) for k in item_metrics.keys()) & attacker_item_indices)
    
    active_users_non_attacker = active_users_total - active_attacker_users
    active_items_non_attacker = active_items_total - active_attacker_items
    
    print(f"Active users: {active_users_total} (attackers: {active_attacker_users}, non-attacker: {active_users_non_attacker})")
    print(f"Active items: {active_items_total} (attackers: {active_attacker_items}, non-attacker: {active_items_non_attacker})")
    
    amended_metrics = []
    
    for turn_file in turn_files:
        with open(turn_file) as f:
            turn_data = json.load(f)
        
        # Rename 'contamination' to 'natural_semantic_evolution' if it exists
        if 'contamination' in turn_data and turn_data['contamination']:
            old_contam = turn_data['contamination']
            
            # Create new natural_semantic_evolution section with renamed fields
            natural_evolution = {
                # Renamed fields
                'user_evolution_count': old_contam.get('total_compromised_users', 0),
                'item_evolution_count': old_contam.get('total_compromised_items', 0),
                'user_evolution_percentage': old_contam.get('compromised_user_percentage', 0),
                'item_evolution_percentage': old_contam.get('compromised_item_percentage', 0),
                # Active agent counts
                'active_users_total': active_users_total,
                'active_items_total': active_items_total,
                'active_users_non_attacker': active_users_non_attacker,
                'active_items_non_attacker': active_items_non_attacker,
                'num_attacker_users': active_attacker_users,
                'num_attacker_items': active_attacker_items,
            }
            
            # Recalculate percentages with correct denominators
            evolution_users = natural_evolution['user_evolution_count']
            evolution_items = natural_evolution['item_evolution_count']
            
            natural_evolution['user_evolution_percentage'] = min(100.0, (evolution_users / max(1, active_users_non_attacker)) * 100)
            natural_evolution['item_evolution_percentage'] = min(100.0, (evolution_items / max(1, active_items_non_attacker)) * 100)
            
            # Replace old contamination with new natural_semantic_evolution
            turn_data['natural_semantic_evolution'] = natural_evolution
            del turn_data['contamination']
        
        # Handle case where natural_semantic_evolution already exists (just update counts)
        elif 'natural_semantic_evolution' in turn_data and turn_data['natural_semantic_evolution']:
            nse = turn_data['natural_semantic_evolution']
            nse['active_users_total'] = active_users_total
            nse['active_items_total'] = active_items_total
            nse['active_users_non_attacker'] = active_users_non_attacker
            nse['active_items_non_attacker'] = active_items_non_attacker
            nse['num_attacker_users'] = active_attacker_users
            nse['num_attacker_items'] = active_attacker_items
            
            # Recalculate percentages
            evolution_users = nse.get('user_evolution_count', 0)
            evolution_items = nse.get('item_evolution_count', 0)
            nse['user_evolution_percentage'] = min(100.0, (evolution_users / max(1, active_users_non_attacker)) * 100)
            nse['item_evolution_percentage'] = min(100.0, (evolution_items / max(1, active_items_non_attacker)) * 100)
        
        # Save amended file
        with open(turn_file, 'w') as f:
            json.dump(turn_data, f, indent=2)
        
        amended_metrics.append(turn_data)
        print(f"Amended {os.path.basename(turn_file)}")
    
    return amended_metrics


def create_mbd_stacked_area_plots(task_dir: str, metrics: List[Dict], 
                                   attacker_user_indices: set, attacker_item_indices: set):
    """
    Create stacked area plots for MBD (Bias & Misinformation Dissemination) row.
    
    Uses LLM Judge contamination counts (semantic detection), NOT pattern-based metrics.
    
    Layout:
    - Row 1: User counts (stacked area) | User percentages (stacked area)
    - Row 2: Item counts (stacked area) | Item percentages (stacked area)
    
    Colors:
    - Pink/translucent red: Attackers (bottom layer)
    - Light translucent orange: LLM-detected contaminated victims (stacked on top)
    """
    turns = [m['turn'] for m in metrics]
    
    # Extract data
    num_attacker_users = len(attacker_user_indices)
    num_attacker_items = len(attacker_item_indices)
    
    # Get active counts from first metric with natural_semantic_evolution data
    active_users_total = 0
    active_items_total = 0
    for m in metrics:
        nse = m.get('natural_semantic_evolution') or m.get('contamination')
        if nse:
            active_users_total = nse.get('active_users_total', 50)
            active_items_total = nse.get('active_items_total', 133)
            break
    
    # Extract LLM Judge contamination counts per turn (NOT pattern-based)
    llm_contaminated_users = []
    llm_contaminated_items = []
    
    for m in metrics:
        llm_judge = m.get('llm_judge', {})
        user_data = llm_judge.get('user', {})
        item_data = llm_judge.get('item', {})
        llm_contaminated_users.append(user_data.get('contaminated_count', 0) or 0)
        llm_contaminated_items.append(item_data.get('contaminated_count', 0) or 0)
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bias & Misinformation Dissemination (LLM Judge) - Stacked Area Plots', fontsize=14, fontweight='bold')
    
    # Colors
    attacker_color = '#FF6B6B'  # Pink/red for attackers
    attacker_alpha = 0.6
    victim_color = '#FFA500'    # Orange for LLM-detected victims
    victim_alpha = 0.6
    
    # === Plot 1: User Counts (Stacked Area) ===
    ax1 = axes[0, 0]
    
    # Attackers are constant (bottom layer)
    attacker_users_line = [num_attacker_users] * len(turns)
    
    # Stack: attackers at bottom, LLM-detected victims on top
    ax1.fill_between(turns, 0, attacker_users_line, 
                     color=attacker_color, alpha=attacker_alpha, label='Attackers')
    ax1.fill_between(turns, attacker_users_line, 
                     [a + v for a, v in zip(attacker_users_line, llm_contaminated_users)],
                     color=victim_color, alpha=victim_alpha, label='LLM-Detected Victims')
    
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('User Count')
    ax1.set_title('User Contamination (Count)')
    ax1.set_ylim(0, active_users_total)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: User Percentages (Stacked Area) ===
    ax2 = axes[0, 1]
    
    # Percentages relative to total active users
    attacker_users_pct = [(num_attacker_users / max(1, active_users_total)) * 100] * len(turns)
    victim_users_pct = [(v / max(1, active_users_total)) * 100 for v in llm_contaminated_users]
    
    ax2.fill_between(turns, 0, attacker_users_pct,
                     color=attacker_color, alpha=attacker_alpha, label='Attackers %')
    ax2.fill_between(turns, attacker_users_pct,
                     [a + v for a, v in zip(attacker_users_pct, victim_users_pct)],
                     color=victim_color, alpha=victim_alpha, label='LLM-Detected Victims %')
    
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('Percentage of System')
    ax2.set_title('User Contamination (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Item Counts (Stacked Area) ===
    ax3 = axes[1, 0]
    
    attacker_items_line = [num_attacker_items] * len(turns)
    
    ax3.fill_between(turns, 0, attacker_items_line,
                     color=attacker_color, alpha=attacker_alpha, label='Attackers')
    ax3.fill_between(turns, attacker_items_line,
                     [a + v for a, v in zip(attacker_items_line, llm_contaminated_items)],
                     color=victim_color, alpha=victim_alpha, label='LLM-Detected Victims')
    
    ax3.set_xlabel('Turn')
    ax3.set_ylabel('Item Count')
    ax3.set_title('Item Contamination (Count)')
    ax3.set_ylim(0, active_items_total)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Item Percentages (Stacked Area) ===
    ax4 = axes[1, 1]
    
    attacker_items_pct = [(num_attacker_items / max(1, active_items_total)) * 100] * len(turns)
    victim_items_pct = [(v / max(1, active_items_total)) * 100 for v in llm_contaminated_items]
    
    ax4.fill_between(turns, 0, attacker_items_pct,
                     color=attacker_color, alpha=attacker_alpha, label='Attackers %')
    ax4.fill_between(turns, attacker_items_pct,
                     [a + v for a, v in zip(attacker_items_pct, victim_items_pct)],
                     color=victim_color, alpha=victim_alpha, label='LLM-Detected Victims %')
    
    ax4.set_xlabel('Turn')
    ax4.set_ylabel('Percentage of System')
    ax4.set_title('Item Contamination (%)')
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(task_dir, 'mbd_stacked_area_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved MBD stacked area plots to {output_path}")


def main(task_dir: str):
    """Main function to amend turn files and create plots."""
    print(f"Processing task directory: {task_dir}")
    
    # Load metrics files
    user_metrics, item_metrics, summary = load_metrics_files(task_dir)
    
    if not user_metrics or not item_metrics:
        print("Error: Could not load user_metrics or item_metrics files")
        return
    
    # Get attacker indices
    attacker_user_indices, attacker_item_indices = get_attacker_indices(summary, user_metrics, item_metrics)
    
    print(f"Attacker user indices: {len(attacker_user_indices)}")
    print(f"Attacker item indices: {len(attacker_item_indices)}")
    
    # Amend turn files
    amended_metrics = amend_turn_files(task_dir, user_metrics, item_metrics,
                                        attacker_user_indices, attacker_item_indices)
    
    # Create MBD stacked area plots
    if amended_metrics:
        create_mbd_stacked_area_plots(task_dir, amended_metrics, 
                                       attacker_user_indices, attacker_item_indices)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task_dir = sys.argv[1]
    else:
        # Default path
        task_dir = "connacf/attack_output/netsafe/misinfo_subset_small/CDs-100user-dense/task_0"
    
    main(task_dir)
