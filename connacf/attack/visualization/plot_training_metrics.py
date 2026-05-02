#!/usr/bin/env python3
"""
Plot Training Metrics - Visualize per-turn metrics during/after training

This script reads the per-turn metrics saved during training and generates
plots similar to what you see during inference.

Usage:
    python plot_training_metrics.py --task_dir attack_results/task_0
    python plot_training_metrics.py --task_dir attack_results/task_0 --output_dir plots
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def load_turn_metrics(task_dir: str) -> List[Dict]:
    """Load all turn metrics from task directory"""
    turn_files = sorted(glob.glob(os.path.join(task_dir, "turn_*.json")))
    
    metrics = []
    for turn_file in turn_files:
        with open(turn_file, 'r') as f:
            metrics.append(json.load(f))
    
    return metrics


def extract_metrics_for_plotting(turn_metrics: List[Dict]) -> Dict[str, List]:
    """Extract metrics arrays for plotting"""
    turns = []
    accuracies = []
    aas_scores = []
    contaminated_users = []
    contaminated_items = []
    compromised_user_pct = []
    compromised_item_pct = []
    
    for metric in turn_metrics:
        turns.append(metric['turn'])
        accuracies.append(metric['system_performance'].get('accuracy', 0.0))
        aas_scores.append(metric['attack_success'].get('aas_score', 0.0))
        contaminated_users.append(metric['contamination'].get('total_compromised_users', 0))
        contaminated_items.append(metric['contamination'].get('total_compromised_items', 0))
        compromised_user_pct.append(metric['contamination'].get('compromised_user_percentage', 0.0))
        compromised_item_pct.append(metric['contamination'].get('compromised_item_percentage', 0.0))
    
    return {
        'turns': turns,
        'accuracies': accuracies,
        'aas_scores': aas_scores,
        'contaminated_users': contaminated_users,
        'contaminated_items': contaminated_items,
        'compromised_user_pct': compromised_user_pct,
        'compromised_item_pct': compromised_item_pct,
    }


def plot_training_metrics(metrics: Dict[str, List], output_dir: str, task_name: str = "task_0"):
    """Generate comprehensive training metrics plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics - {task_name}', fontsize=16, fontweight='bold')
    
    turns = metrics['turns']
    
    # Plot 1: Accuracy vs Turns
    axes[0, 0].plot(turns, metrics['accuracies'], linewidth=2.2, color='blue', alpha=0.75)
    axes[0, 0].set_xlabel('Turn', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('System Accuracy vs Turns', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.0])
    
    # Plot 2: AAS Score vs Turns
    axes[0, 1].plot(turns, metrics['aas_scores'], linewidth=2.2, color='red', alpha=0.75)
    axes[0, 1].set_xlabel('Turn', fontsize=12)
    axes[0, 1].set_ylabel('AAS Score', fontsize=12)
    axes[0, 1].set_title('Attack Success Score vs Turns', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Contaminated Users (Count)
    axes[0, 2].plot(turns, metrics['contaminated_users'], linewidth=2.2, color='green', alpha=0.75)
    axes[0, 2].set_xlabel('Turn', fontsize=12)
    axes[0, 2].set_ylabel('Contaminated Users', fontsize=12)
    axes[0, 2].set_title('Contaminated Users vs Turns', fontsize=13, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Contaminated Items (Count)
    axes[1, 0].plot(turns, metrics['contaminated_items'], linewidth=2.2, color='purple', alpha=0.75)
    axes[1, 0].set_xlabel('Turn', fontsize=12)
    axes[1, 0].set_ylabel('Contaminated Items', fontsize=12)
    axes[1, 0].set_title('Contaminated Items vs Turns', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Compromised User Percentage
    axes[1, 1].plot(turns, metrics['compromised_user_pct'], linewidth=2.2, color='cyan', alpha=0.75)
    axes[1, 1].set_xlabel('Turn', fontsize=12)
    axes[1, 1].set_ylabel('Compromised %', fontsize=12)
    axes[1, 1].set_title('Compromised User % vs Turns', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    # Plot 6: Compromised Item Percentage
    axes[1, 2].plot(turns, metrics['compromised_item_pct'], linewidth=2.2, color='orange', alpha=0.75)
    axes[1, 2].set_xlabel('Turn', fontsize=12)
    axes[1, 2].set_ylabel('Compromised %', fontsize=12)
    axes[1, 2].set_title('Compromised Item % vs Turns', fontsize=13, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{task_name}_training_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive plot to: {output_file}")
    plt.close()
    
    # Create individual plots for key metrics
    _plot_individual_metric(turns, metrics['accuracies'], 'Accuracy', 'Turn', 
                           'System Accuracy vs Turns', output_dir, f'{task_name}_accuracy.png', 'blue')
    _plot_individual_metric(turns, metrics['aas_scores'], 'AAS Score', 'Turn',
                           'Attack Success Score vs Turns', output_dir, f'{task_name}_aas.png', 'red')
    _plot_individual_metric(turns, metrics['compromised_user_pct'], 'Compromised %', 'Turn',
                           'Compromised User % vs Turns', output_dir, f'{task_name}_compromised_users_pct.png', 'green')


def _plot_individual_metric(x_data: List, y_data: List, ylabel: str, xlabel: str,
                            title: str, output_dir: str, filename: str, color: str = 'blue'):
    """Plot individual metric"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, linewidth=2.2, color=color, label=ylabel, alpha=0.75)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {ylabel} plot to: {output_file}")
    plt.close()


def print_metrics_summary(metrics: Dict[str, List]):
    """Print summary of metrics"""
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(f"Total Turns: {len(metrics['turns'])}")
    print(f"Initial Accuracy: {metrics['accuracies'][0]:.3f}")
    print(f"Final Accuracy: {metrics['accuracies'][-1]:.3f}")
    print(f"Accuracy Degradation: {metrics['accuracies'][0] - metrics['accuracies'][-1]:.3f}")
    print(f"Final AAS Score: {metrics['aas_scores'][-1]:.3f}")
    print(f"Final Contaminated Users: {metrics['contaminated_users'][-1]}")
    print(f"Final Contaminated Items: {metrics['contaminated_items'][-1]}")
    print(f"Final Compromised User %: {metrics['compromised_user_pct'][-1]:.2f}%")
    print(f"Final Compromised Item %: {metrics['compromised_item_pct'][-1]:.2f}%")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from per-turn data')
    parser.add_argument('--task_dir', type=str, required=True, 
                       help='Path to task directory (e.g., attack_results/task_0)')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Validate task directory
    if not os.path.exists(args.task_dir):
        print(f"❌ Error: Task directory not found: {args.task_dir}")
        return
    
    # Load turn metrics
    print(f"Loading turn metrics from: {args.task_dir}")
    turn_metrics = load_turn_metrics(args.task_dir)
    
    if not turn_metrics:
        print(f"❌ Error: No turn metrics found in {args.task_dir}")
        return
    
    print(f"✅ Loaded {len(turn_metrics)} turn metrics")
    
    # Extract metrics for plotting
    metrics = extract_metrics_for_plotting(turn_metrics)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Generate plots
    task_name = os.path.basename(args.task_dir)
    plot_training_metrics(metrics, args.output_dir, task_name)
    
    print(f"\n✅ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
