"""
Plot AAS and Compromised Nodes vs Turns from per-turn JSON metrics

Usage:
    python plot_turn_metrics.py --task_dir attack_results/task_0 --output_dir plots
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import glob


def load_turn_metrics(task_dir: str) -> List[Dict]:
    """Load all turn metrics from a task directory"""
    turn_files = sorted(glob.glob(os.path.join(task_dir, "turn_*.json")))
    
    metrics = []
    for turn_file in turn_files:
        with open(turn_file, 'r') as f:
            metrics.append(json.load(f))
    
    return metrics


def plot_aas_vs_turns(metrics: List[Dict], output_path: str):
    """Plot Average Attack Success (AAS) vs Turns"""
    turns = [m['turn'] for m in metrics]
    aas_scores = [m['attack_success']['aas_score'] for m in metrics]
    accuracy_deg = [m['attack_success']['accuracy_degradation'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AAS Score
    ax1.plot(turns, aas_scores, linewidth=2.2, color='red', label='AAS Score', alpha=0.75)
    ax1.set_xlabel('Turn / Hop', fontsize=12)
    ax1.set_ylabel('Average Attack Success (AAS)', fontsize=12)
    ax1.set_title('Attack Success vs Interaction Turns', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy Degradation
    ax2.plot(turns, accuracy_deg, linewidth=2.2, color='darkred', label='Accuracy Degradation', alpha=0.75)
    ax2.set_xlabel('Turn / Hop', fontsize=12)
    ax2.set_ylabel('Accuracy Degradation', fontsize=12)
    ax2.set_title('Accuracy Degradation vs Interaction Turns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved AAS plot to {output_path}")
    plt.close()


def plot_compromised_vs_turns(metrics: List[Dict], output_path: str):
    """Plot % of Compromised Nodes vs Turns"""
    turns = [m['turn'] for m in metrics]
    user_pct = [m['contamination']['compromised_user_percentage'] for m in metrics]
    item_pct = [m['contamination']['compromised_item_percentage'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # User Contamination
    ax1.plot(turns, user_pct, linewidth=2.2, color='blue', label='User Agents', alpha=0.75)
    ax1.set_xlabel('Turn / Hop', fontsize=12)
    ax1.set_ylabel('% Compromised Users', fontsize=12)
    ax1.set_title('User Agent Contamination vs Turns', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 100])
    
    # Item Contamination
    ax2.plot(turns, item_pct, linewidth=2.2, color='green', label='Item Agents', alpha=0.75)
    ax2.set_xlabel('Turn / Hop', fontsize=12)
    ax2.set_ylabel('% Compromised Items', fontsize=12)
    ax2.set_title('Item Agent Contamination vs Turns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved compromised nodes plot to {output_path}")
    plt.close()


def plot_combined_metrics(metrics: List[Dict], output_path: str):
    """Plot combined view of all key metrics"""
    turns = [m['turn'] for m in metrics]
    aas_scores = [m['attack_success']['aas_score'] for m in metrics]
    user_pct = [m['contamination']['compromised_user_percentage'] for m in metrics]
    item_pct = [m['contamination']['compromised_item_percentage'] for m in metrics]
    accuracy = [m['system_performance']['accuracy'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AAS Score
    axes[0, 0].plot(turns, aas_scores, linewidth=2.2, color='red', alpha=0.75)
    axes[0, 0].set_ylabel('AAS Score', fontsize=11)
    axes[0, 0].set_title('Average Attack Success', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(turns, accuracy, linewidth=2.2, color='purple', alpha=0.75)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('System Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # User Contamination
    axes[1, 0].plot(turns, user_pct, linewidth=2.2, color='blue', alpha=0.75)
    axes[1, 0].set_xlabel('Turn / Hop', fontsize=11)
    axes[1, 0].set_ylabel('% Compromised Users', fontsize=11)
    axes[1, 0].set_title('User Agent Contamination', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 100])
    
    # Item Contamination
    axes[1, 1].plot(turns, item_pct, linewidth=2.2, color='green', alpha=0.75)
    axes[1, 1].set_xlabel('Turn / Hop', fontsize=11)
    axes[1, 1].set_ylabel('% Compromised Items', fontsize=11)
    axes[1, 1].set_title('Item Agent Contamination', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined metrics plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot turn-based metrics from ConnaCF attack analysis')
    parser.add_argument('--task_dir', type=str, required=True, help='Path to task directory (e.g., attack_results/task_0)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.task_dir}...")
    metrics = load_turn_metrics(args.task_dir)
    print(f"Loaded {len(metrics)} turns of data")
    
    if not metrics:
        print("No metrics found!")
        return
    
    # Generate plots
    task_name = os.path.basename(args.task_dir)
    
    plot_aas_vs_turns(
        metrics, 
        os.path.join(args.output_dir, f'{task_name}_aas_vs_turns.png')
    )
    
    plot_compromised_vs_turns(
        metrics,
        os.path.join(args.output_dir, f'{task_name}_compromised_vs_turns.png')
    )
    
    plot_combined_metrics(
        metrics,
        os.path.join(args.output_dir, f'{task_name}_combined_metrics.png')
    )
    
    print("\nAll plots generated successfully!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    main()
