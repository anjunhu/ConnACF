"""
Plot MACF Temporal Metrics with Task Boundaries

Plots per-turn metrics across global turns with translucent grey vertical
lines showing task boundaries (where new tasks start).

Usage:
    python plot_temporal_metrics.py --input macf_output/temporal_metrics.json --output plots/
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import numpy as np


def load_temporal_metrics(input_path: str) -> Dict:
    """Load temporal metrics from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def plot_ranking_metrics_with_boundaries(
    data: Dict,
    output_path: str,
    title_suffix: str = ""
):
    """
    Plot Hit@10 and NDCG@10 vs global turns with task boundaries.
    
    Task boundaries are shown as translucent light grey vertical lines.
    """
    metrics = data['metrics']
    task_boundaries = data['task_boundaries']
    
    if not metrics:
        print("No metrics to plot!")
        return
    
    global_turns = [m['global_turn'] for m in metrics]
    hit_at_10 = [m['hit_at_10'] for m in metrics]
    ndcg_at_10 = [m['ndcg_at_10'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # === Plot Hit@10 ===
    ax1.plot(global_turns, hit_at_10, linewidth=2.2, 
             color='#2196F3', label='Hit@10', alpha=0.75)
    ax1.set_ylabel('Hit@10', fontsize=12)
    ax1.set_title(f'Per-Turn Hit@10 vs Global Turns{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_ylim([-0.05, 1.05])
    
    # Add task boundary lines
    for boundary in task_boundaries:
        ax1.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # === Plot NDCG@10 ===
    ax2.plot(global_turns, ndcg_at_10, linewidth=2.2,
             color='#4CAF50', label='NDCG@10', alpha=0.75)
    ax2.set_xlabel('Global Turn', fontsize=12)
    ax2.set_ylabel('NDCG@10', fontsize=12)
    ax2.set_title(f'Per-Turn NDCG@10 vs Global Turns{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_ylim([-0.05, 1.05])
    
    # Add task boundary lines
    for boundary in task_boundaries:
        ax2.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Add legend for task boundaries
    ax2.axvline(x=-100, color='grey', linestyle='-', linewidth=1.5, alpha=0.3, 
                label='Task Boundary')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ranking metrics plot to {output_path}")
    plt.close()


def plot_combined_dashboard(
    data: Dict,
    output_path: str,
    title_suffix: str = ""
):
    """
    Plot combined dashboard with all key metrics and task boundaries.
    
    4-panel view:
    - Hit@10 over time
    - NDCG@10 over time  
    - Draft list size / consensus
    - LLM calls / duration
    """
    metrics = data['metrics']
    task_boundaries = data['task_boundaries']
    
    if not metrics:
        print("No metrics to plot!")
        return
    
    global_turns = [m['global_turn'] for m in metrics]
    hit_at_10 = [m['hit_at_10'] for m in metrics]
    ndcg_at_10 = [m['ndcg_at_10'] for m in metrics]
    draft_sizes = [m['draft_list_size'] for m in metrics]
    consensus_scores = [m['consensus_score'] for m in metrics]
    llm_calls = [m['llm_calls'] for m in metrics]
    durations = [m['turn_duration'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # === Panel 1: Hit@10 ===
    ax = axes[0, 0]
    ax.plot(global_turns, hit_at_10, linewidth=2.2,
            color='#2196F3', alpha=0.75)
    ax.set_ylabel('Hit@10', fontsize=11)
    ax.set_title('Per-Turn Hit@10', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # === Panel 2: NDCG@10 ===
    ax = axes[0, 1]
    ax.plot(global_turns, ndcg_at_10, linewidth=2.2,
            color='#4CAF50', alpha=0.75)
    ax.set_ylabel('NDCG@10', fontsize=11)
    ax.set_title('Per-Turn NDCG@10', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # === Panel 3: Draft Size & Consensus ===
    ax = axes[1, 0]
    ax.plot(global_turns, draft_sizes, linewidth=2.2,
            color='#FF9800', alpha=0.75, label='Draft Size')
    ax.set_xlabel('Global Turn', fontsize=11)
    ax.set_ylabel('Draft List Size', fontsize=11, color='#FF9800')
    ax.tick_params(axis='y', labelcolor='#FF9800')
    ax.set_title('Draft Size & Consensus Score', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary y-axis for consensus
    ax2 = ax.twinx()
    ax2.plot(global_turns, consensus_scores, linewidth=2.2,
             color='#9C27B0', alpha=0.75, label='Consensus')
    ax2.set_ylabel('Consensus Score', fontsize=11, color='#9C27B0')
    ax2.tick_params(axis='y', labelcolor='#9C27B0')
    ax2.set_ylim([-0.05, 1.05])
    
    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # === Panel 4: Resources ===
    ax = axes[1, 1]
    ax.bar(global_turns, llm_calls, color='#E91E63', alpha=0.6, label='LLM Calls')
    ax.set_xlabel('Global Turn', fontsize=11)
    ax.set_ylabel('LLM Calls', fontsize=11, color='#E91E63')
    ax.tick_params(axis='y', labelcolor='#E91E63')
    ax.set_title('Resource Usage', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary y-axis for duration
    ax2 = ax.twinx()
    ax2.plot(global_turns, durations, linewidth=2.2,
             color='#00BCD4', alpha=0.75, label='Duration')
    ax2.set_ylabel('Turn Duration (s)', fontsize=11, color='#00BCD4')
    ax2.tick_params(axis='y', labelcolor='#00BCD4')
    
    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Add overall title
    fig.suptitle(
        f"MACF Temporal Metrics Dashboard{title_suffix}\n"
        f"(Grey lines = Task Boundaries, {len(task_boundaries)} tasks, {len(metrics)} turns)",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined dashboard to {output_path}")
    plt.close()


def plot_per_task_improvement(
    data: Dict,
    output_path: str,
    title_suffix: str = ""
):
    """
    Plot per-task improvement: how Hit@10/NDCG@10 improve within each task.
    
    Shows that recommendations should improve across N turns within each task.
    """
    metrics = data['metrics']
    task_boundaries = data['task_boundaries']
    
    if not metrics or len(task_boundaries) < 2:
        print("Not enough data for per-task improvement plot!")
        return
    
    # Group metrics by task
    task_metrics = {}
    for m in metrics:
        task_id = m['task_id']
        if task_id not in task_metrics:
            task_metrics[task_id] = []
        task_metrics[task_id].append(m)
    
    # Plot improvement within each task
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(task_metrics)))
    
    for idx, (task_id, task_data) in enumerate(sorted(task_metrics.items())):
        turns_in_task = [m['turn_in_task'] for m in task_data]
        hit_at_10 = [m['hit_at_10'] for m in task_data]
        ndcg_at_10 = [m['ndcg_at_10'] for m in task_data]
        
        ax1.plot(turns_in_task, hit_at_10, linewidth=2.2,
                 color=colors[idx], alpha=0.75, label=f'Task {task_id}' if idx < 5 else None)
        ax2.plot(turns_in_task, ndcg_at_10, linewidth=2.2,
                 color=colors[idx], alpha=0.75, label=f'Task {task_id}' if idx < 5 else None)
    
    ax1.set_xlabel('Turn Within Task', fontsize=12)
    ax1.set_ylabel('Hit@10', fontsize=12)
    ax1.set_title(f'Hit@10 Improvement Within Tasks{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    if len(task_metrics) <= 10:
        ax1.legend(loc='upper left', fontsize=8)
    
    ax2.set_xlabel('Turn Within Task', fontsize=12)
    ax2.set_ylabel('NDCG@10', fontsize=12)
    ax2.set_title(f'NDCG@10 Improvement Within Tasks{title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])
    if len(task_metrics) <= 10:
        ax2.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-task improvement plot to {output_path}")
    plt.close()


def plot_llm_judge_results(
    data: Dict,
    output_path: str,
    title_suffix: str = ""
):
    """
    Plot LLM judge contamination scores over time.
    
    Only plots turns where LLM judge was evaluated (batched).
    """
    metrics = data['metrics']
    task_boundaries = data['task_boundaries']
    
    # Filter to only evaluated turns
    evaluated = [m for m in metrics if m.get('llm_judge_evaluated', False)]
    
    if not evaluated:
        print("No LLM judge evaluations to plot!")
        return
    
    global_turns = [m['global_turn'] for m in evaluated]
    contamination_scores = [m.get('contamination_score', 0) or 0 for m in evaluated]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(global_turns, contamination_scores, linewidth=2.2,
            color='#F44336', label='Contamination Score', alpha=0.75)
    
    ax.set_xlabel('Global Turn', fontsize=12)
    ax.set_ylabel('Contamination Score', fontsize=12)
    ax.set_title(f'LLM Judge Contamination Scores{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left')
    
    # Add task boundary lines
    for boundary in task_boundaries:
        ax.axvline(x=boundary, color='grey', linestyle='-', linewidth=1.5, alpha=0.3)
    
    # Add annotation for evaluation interval
    llm_judge_interval = data.get('llm_judge_interval', 10)
    ax.annotate(
        f'Evaluated every {llm_judge_interval} turns',
        xy=(0.98, 0.98), xycoords='axes fraction',
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved LLM judge plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot MACF temporal metrics with task boundaries'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to temporal_metrics.json file'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--title-suffix', type=str, default='',
        help='Suffix to add to plot titles'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"Loading temporal metrics from {args.input}...")
    data = load_temporal_metrics(args.input)
    
    experiment_name = data.get('experiment_name', 'macf')
    total_turns = data.get('total_global_turns', len(data.get('metrics', [])))
    total_tasks = data.get('total_tasks', len(data.get('task_boundaries', [])))
    
    print(f"Experiment: {experiment_name}")
    print(f"Total global turns: {total_turns}")
    print(f"Total tasks: {total_tasks}")
    print(f"Task boundaries: {data.get('task_boundaries', [])[:10]}...")
    
    title_suffix = args.title_suffix or f" ({experiment_name})"
    
    # Generate plots
    plot_ranking_metrics_with_boundaries(
        data,
        os.path.join(args.output, f'{experiment_name}_ranking_metrics.png'),
        title_suffix
    )
    
    plot_combined_dashboard(
        data,
        os.path.join(args.output, f'{experiment_name}_dashboard.png'),
        title_suffix
    )
    
    plot_per_task_improvement(
        data,
        os.path.join(args.output, f'{experiment_name}_per_task_improvement.png'),
        title_suffix
    )
    
    plot_llm_judge_results(
        data,
        os.path.join(args.output, f'{experiment_name}_llm_judge.png'),
        title_suffix
    )
    
    print(f"\nAll plots saved to {args.output}/")


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    main()
