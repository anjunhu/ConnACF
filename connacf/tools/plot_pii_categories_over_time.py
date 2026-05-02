#!/usr/bin/env python3
"""
Plot PII leakage rates by category over time for MAMA attack experiments.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_turn_data(experiment_dir: str) -> list[dict]:
    """Load all turn JSON files from an experiment directory."""
    turns = []
    turn_idx = 0
    while True:
        turn_file = os.path.join(experiment_dir, f"turn_{turn_idx}.json")
        if not os.path.exists(turn_file):
            break
        with open(turn_file, 'r') as f:
            turns.append(json.load(f))
        turn_idx += 1
    return turns


def plot_pii_categories_over_time(experiment_dir: str, output_path: str = None):
    """
    Plot PII leakage rates for each category separately over time.
    
    Args:
        experiment_dir: Path to experiment output directory containing turn_*.json files
        output_path: Optional output path for the plot. Defaults to experiment_dir/pii_categories_over_time.png
    """
    turns = load_turn_data(experiment_dir)
    if not turns:
        print(f"No turn data found in {experiment_dir}")
        return
    
    # Extract category rates over time
    categories = ['identity', 'contact', 'location', 'temporal', 'regulated']
    category_data = {cat: [] for cat in categories}
    rounds = []
    
    for turn in turns:
        rounds.append(turn['turn'])
        privacy = turn.get('privacy', {})
        cat_rates = privacy.get('category_rates', {})
        for cat in categories:
            category_data[cat].append(cat_rates.get(cat, 0.0))
    
    # Also extract overall PII leakage rate
    overall_rates = [turn.get('privacy', {}).get('pii_leakage_rate', 0.0) for turn in turns]
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Color scheme for categories
    colors = {
        'identity': '#e74c3c',    # Red
        'contact': '#3498db',     # Blue
        'location': '#2ecc71',    # Green
        'temporal': '#9b59b6',    # Purple
        'regulated': '#f39c12',   # Orange
    }
    
    # Plot each category in its own subplot
    for idx, cat in enumerate(categories):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        ax.plot(rounds, category_data[cat], 
                color=colors[cat], marker='o', linewidth=2, markersize=8,
                label=cat.capitalize())
        ax.fill_between(rounds, category_data[cat], alpha=0.3, color=colors[cat])
        
        ax.set_xlabel('Turn', fontsize=11)
        ax.set_ylabel('Leak Rate', fontsize=11)
        ax.set_title(f'{cat.capitalize()} PII Leakage', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_rate = category_data[cat][-1]
        ax.annotate(f'{final_rate:.1%}', 
                   xy=(rounds[-1], final_rate),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=colors[cat])
    
    # Plot 6: Combined view with all categories
    ax = axes[1, 2]
    for cat in categories:
        ax.plot(rounds, category_data[cat], 
                color=colors[cat], marker='o', linewidth=2, markersize=6,
                label=cat.capitalize(), alpha=0.8)
    
    # Add overall rate as dashed line
    ax.plot(rounds, overall_rates, 
            color='black', marker='s', linewidth=2, markersize=6,
            linestyle='--', label='Overall', alpha=0.7)
    
    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel('Leak Rate', fontsize=11)
    ax.set_title('All Categories Combined', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xticks(rounds)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    task_id = os.path.basename(experiment_dir)
    fig.suptitle(f'PII Leakage by Category Over Time\nExperiment: {task_id}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    if output_path is None:
        output_path = os.path.join(experiment_dir, 'pii_categories_over_time.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PII category plot to: {output_path}")
    
    # Print summary
    print("\nPII Leakage Summary by Category:")
    print("-" * 50)
    for cat in categories:
        rates = category_data[cat]
        print(f"  {cat.capitalize():12s}: {rates[0]:.1%} -> {rates[-1]:.1%} (final)")
    print(f"  {'Overall':12s}: {overall_rates[0]:.1%} -> {overall_rates[-1]:.1%} (final)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_pii_categories_over_time.py <experiment_dir> [output_path]")
        print("\nExample:")
        print("  python plot_pii_categories_over_time.py connacf/attack_output/mama/mama_medium_5cand/ml-100k-20-user-dense/260213131931")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_pii_categories_over_time(experiment_dir, output_path)
