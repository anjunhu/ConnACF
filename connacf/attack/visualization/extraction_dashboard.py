"""
Extraction Dashboard Visualization for Privacy/Reverse-Engineering Attacks

For attacks where information flows FROM victims TO attacker:
- MASLeak: IP extraction (prompts, topology, agent count)
- MAMA: PII leakage
- MASTER: System probing and trait extraction

Contrast with dissemination attacks (NetSafe, Cheat, Drunk) where
contamination flows FROM attacker TO victims.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class ExtractionDashboard:
    """
    3x3 dashboard for extraction/privacy attacks.
    
    Layout:
    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │ Cumulative ER       │ Per-Target Extract  │ Worm Propagation    │
    │ (line over turns)   │ (stacked area)      │ (responses/turn)    │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ IP Target Breakdown │ U-I Topology Leak   │ Per-User Heatmap    │
    │ (grouped bars)      │ (edges over time)   │ (users × turns)     │
    ├─────────────────────┼─────────────────────┼─────────────────────┤
    │ Summary Statistics  │ Confidence Dist     │ Recovery by Type    │
    │ (text box)          │ (histogram)         │ (horizontal bars)   │
    └─────────────────────┴─────────────────────┴─────────────────────┘
    """
    
    def __init__(self, output_dir: str, attack_type: str = 'MASLeak'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attack_type = attack_type
        self.dpi = 300
        
        # Per-turn metrics storage
        self.per_turn_metrics: List[Dict[str, Any]] = []
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def record_turn_metrics(self, turn: int, metrics: Dict[str, Any]):
        """
        Record metrics for a single turn.
        
        Expected metrics keys:
        - extract_rate: Overall ER_MAS
        - ss_system_prompt: System prompt semantic similarity
        - sm_system_prompt: System prompt substring match
        - ss_task_instructions: Task instructions SS
        - sm_task_instructions: Task instructions SM
        - f1_agent_count: Agent count F1
        - f1_comm_density: Communication density F1
        - ui_topology_recall: U-I edge recall
        - ui_items_leaked: Number of items leaked
        - ui_items_total: Total items in ground truth
        - responses_collected: Cumulative responses
        - per_user_scores: List of per-user extraction scores
        """
        record = {'turn': turn, 'timestamp': datetime.now().isoformat()}
        record.update(metrics)
        self.per_turn_metrics.append(record)
    
    def create_dashboard(self, final_metrics: Dict[str, Any], 
                        experiment_name: str = 'extraction_attack'):
        """Create the full 3x3 extraction dashboard."""
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Row 1
        self._plot_cumulative_er(fig.add_subplot(gs[0, 0]))
        self._plot_per_target_extraction(fig.add_subplot(gs[0, 1]))
        self._plot_worm_propagation(fig.add_subplot(gs[0, 2]))
        
        # Row 2
        self._plot_ip_target_breakdown(fig.add_subplot(gs[1, 0]), final_metrics)
        self._plot_ui_topology_leakage(fig.add_subplot(gs[1, 1]))
        self._plot_per_user_heatmap(fig.add_subplot(gs[1, 2]), final_metrics)
        
        # Row 3
        self._plot_summary_statistics(fig.add_subplot(gs[2, 0]), final_metrics)
        self._plot_confidence_distribution(fig.add_subplot(gs[2, 1]), final_metrics)
        self._plot_recovery_by_type(fig.add_subplot(gs[2, 2]), final_metrics)
        
        # Title
        fig.suptitle(f'{self.attack_type} Extraction Dashboard - {experiment_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        save_path = self.output_dir / f'{experiment_name}_extraction_dashboard.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"[EXTRACTION] Dashboard saved to {save_path}")
        return save_path
    
    # ═══════════════════════════════════════════════════════════════════════
    # Row 1: Temporal Extraction Progress
    # ═══════════════════════════════════════════════════════════════════════
    
    def _plot_cumulative_er(self, ax):
        """Plot 1,1: Cumulative Extraction Rate over turns."""
        if not self.per_turn_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Cumulative Extraction Rate')
            return
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        er_values = [m.get('extract_rate', 0) for m in self.per_turn_metrics]
        
        ax.plot(turns, er_values, marker='o', linewidth=2, markersize=4, 
               color='#2E86AB', label='ER_MAS')
        ax.fill_between(turns, er_values, alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('Turn', fontsize=10)
        ax.set_ylabel('Extraction Rate', fontsize=10)
        ax.set_title('Cumulative Extraction Rate (ER_MAS)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    def _plot_per_target_extraction(self, ax):
        """Plot 1,2: Per-target extraction as stacked area."""
        if not self.per_turn_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Per-Target Extraction')
            return
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        
        # Extract per-target metrics
        targets = {
            'SS_sys': [m.get('ss_system_prompt', 0) for m in self.per_turn_metrics],
            'SS_task': [m.get('ss_task_instructions', 0) for m in self.per_turn_metrics],
            'F1_num': [m.get('f1_agent_count', 0) for m in self.per_turn_metrics],
            'F1_comm': [m.get('f1_comm_density', 0) for m in self.per_turn_metrics],
            'UI_topo': [m.get('ui_topology_recall', 0) for m in self.per_turn_metrics],
        }
        
        colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A']
        
        ax.stackplot(turns, targets.values(), labels=targets.keys(), 
                    colors=colors, alpha=0.8)
        
        ax.set_xlabel('Turn', fontsize=10)
        ax.set_ylabel('Cumulative Score', fontsize=10)
        ax.set_title('Per-Target Extraction Progress', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_worm_propagation(self, ax):
        """Plot 1,3: Worm propagation (responses collected over time)."""
        if not self.per_turn_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Worm Propagation')
            return
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        responses = [m.get('responses_collected', 0) for m in self.per_turn_metrics]
        
        ax.bar(turns, responses, color='#6A0572', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add cumulative line
        ax2 = ax.twinx()
        cumulative = np.cumsum(responses) if responses else []
        if len(cumulative) > 0:
            ax2.plot(turns, cumulative, color='#FF6B6B', linewidth=2, 
                    marker='s', markersize=3, label='Cumulative')
            ax2.set_ylabel('Cumulative Responses', fontsize=10, color='#FF6B6B')
            ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        ax.set_xlabel('Turn', fontsize=10)
        ax.set_ylabel('Responses This Turn', fontsize=10)
        ax.set_title('Worm Propagation (Response Collection)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # ═══════════════════════════════════════════════════════════════════════
    # Row 2: Detailed Extraction Metrics
    # ═══════════════════════════════════════════════════════════════════════
    
    def _plot_ip_target_breakdown(self, ax, final_metrics: Dict[str, Any]):
        """Plot 2,1: IP target breakdown as grouped bars."""
        # Paper metrics: SS and SM for prompts/tasks, F1 for counts
        metrics_pairs = [
            ('System\nPrompt', final_metrics.get('ss_system_prompt', 0), 
             final_metrics.get('sm_system_prompt', 0)),
            ('Task\nInstr.', final_metrics.get('ss_task_instructions', 0),
             final_metrics.get('sm_task_instructions', 0)),
            ('Agent\nCount', final_metrics.get('f1_agent_count', 0), 0),
            ('Comm\nDensity', final_metrics.get('f1_comm_density', 0), 0),
            ('U-I\nTopology', final_metrics.get('ui_topology_recall', 0),
             final_metrics.get('ui_topology_precision', 0)),
        ]
        
        labels = [m[0] for m in metrics_pairs]
        ss_values = [m[1] for m in metrics_pairs]
        sm_values = [m[2] for m in metrics_pairs]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ss_values, width, label='SS/F1/Recall', 
                      color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, sm_values, width, label='SM/Precision',
                      color='#E74C3C', alpha=0.8)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('IP Target Extraction Breakdown', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_ui_topology_leakage(self, ax):
        """Plot 2,2: U-I topology leakage over time."""
        if not self.per_turn_metrics:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('U-I Topology Leakage')
            return
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        leaked = [m.get('ui_items_leaked', 0) for m in self.per_turn_metrics]
        total = [m.get('ui_items_total', 1) for m in self.per_turn_metrics]
        
        # Leak rate over time
        leak_rate = [l/t if t > 0 else 0 for l, t in zip(leaked, total)]
        
        ax.fill_between(turns, leak_rate, alpha=0.4, color='#9B59B6')
        ax.plot(turns, leak_rate, marker='o', linewidth=2, markersize=4,
               color='#9B59B6', label='Leak Rate')
        
        # Add leaked count as secondary axis
        ax2 = ax.twinx()
        ax2.bar(turns, leaked, alpha=0.3, color='#27AE60', label='Items Leaked')
        ax2.set_ylabel('Items Leaked', fontsize=10, color='#27AE60')
        ax2.tick_params(axis='y', labelcolor='#27AE60')
        
        ax.set_xlabel('Turn', fontsize=10)
        ax.set_ylabel('Leak Rate', fontsize=10)
        ax.set_title('U-I Topology Leakage Over Time', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_per_user_heatmap(self, ax, final_metrics: Dict[str, Any]):
        """Plot 2,3: Per-user extraction heatmap."""
        per_user_scores = final_metrics.get('ui_topology_scores', [])
        user_ids = final_metrics.get('user_ids', [])
        
        if not per_user_scores or len(per_user_scores) == 0:
            ax.text(0.5, 0.5, 'No per-user data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Per-User Extraction')
            return
        
        # If we have temporal data, create a proper heatmap
        if self.per_turn_metrics and len(self.per_turn_metrics) > 1:
            # Build matrix: users × turns
            n_users = len(per_user_scores)
            n_turns = len(self.per_turn_metrics)
            
            # Limit display for readability
            max_users = min(n_users, 30)
            max_turns = min(n_turns, 20)
            
            matrix = np.zeros((max_users, max_turns))
            for t_idx, m in enumerate(self.per_turn_metrics[:max_turns]):
                scores = m.get('per_user_scores', per_user_scores)
                for u_idx in range(min(len(scores), max_users)):
                    matrix[u_idx, t_idx] = scores[u_idx]
            
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xlabel('Turn', fontsize=10)
            ax.set_ylabel('User Index', fontsize=10)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Extraction Score', fontsize=9)
        else:
            # Single snapshot: bar chart
            max_users = min(len(per_user_scores), 30)
            ax.barh(range(max_users), per_user_scores[:max_users], 
                   color='#E74C3C', alpha=0.7)
            ax.set_xlabel('Extraction Score', fontsize=10)
            ax.set_ylabel('User Index', fontsize=10)
            ax.set_xlim(0, 1)
        
        ax.set_title('Per-User Extraction Scores', fontsize=11, fontweight='bold')
    
    # ═══════════════════════════════════════════════════════════════════════
    # Row 3: Summary and Distributions
    # ═══════════════════════════════════════════════════════════════════════
    
    def _plot_summary_statistics(self, ax, final_metrics: Dict[str, Any]):
        """Plot 3,1: Summary statistics text box."""
        ax.axis('off')
        
        # Build summary text
        er_mas = final_metrics.get('extract_rate', 0)
        er_v2 = final_metrics.get('extract_rate_v2', er_mas)
        n_responses = final_metrics.get('num_responses_collected', 0)
        elapsed = final_metrics.get('elapsed_time', 0)
        
        ui_leaked = final_metrics.get('ui_items_leaked', 0)
        ui_total = final_metrics.get('ui_items_total', 0)
        ui_recall = final_metrics.get('ui_topology_recall', 0)
        
        summary_text = f"""
╔══════════════════════════════════════╗
║  {self.attack_type} EXTRACTION SUMMARY       ║
╠══════════════════════════════════════╣
║                                      ║
║  Overall ER_MAS:     {er_mas:>6.3f}         ║
║  ER_MAS (v2):        {er_v2:>6.3f}         ║
║                                      ║
║  Responses Collected: {n_responses:>5}          ║
║  Elapsed Time:       {elapsed:>6.1f}s        ║
║                                      ║
║  ─── U-I Topology ───                ║
║  Items Leaked:       {ui_leaked:>5}/{ui_total:<5}     ║
║  Topology Recall:    {ui_recall:>6.3f}         ║
║                                      ║
║  ─── Paper Metrics ───               ║
║  SS_sys:  {final_metrics.get('ss_system_prompt', 0):>5.3f}                    ║
║  SM_sys:  {final_metrics.get('sm_system_prompt', 0):>5.3f}                    ║
║  SS_task: {final_metrics.get('ss_task_instructions', 0):>5.3f}                    ║
║  SM_task: {final_metrics.get('sm_task_instructions', 0):>5.3f}                    ║
║  F1_num:  {final_metrics.get('f1_agent_count', 0):>5.3f}                    ║
║  F1_comm: {final_metrics.get('f1_comm_density', 0):>5.3f}                    ║
╚══════════════════════════════════════╝
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))
        
        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold')
    
    def _plot_confidence_distribution(self, ax, final_metrics: Dict[str, Any]):
        """Plot 3,2: Confidence/score distribution histogram."""
        per_user_scores = final_metrics.get('ui_topology_scores', [])
        
        if not per_user_scores or len(per_user_scores) == 0:
            ax.text(0.5, 0.5, 'No score data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Score Distribution')
            return
        
        # Histogram
        ax.hist(per_user_scores, bins=20, color='#3498DB', alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_score = np.mean(per_user_scores)
        median_score = np.median(per_user_scores)
        std_score = np.std(per_user_scores)
        
        ax.axvline(mean_score, color='#E74C3C', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='#27AE60', linestyle=':', linewidth=2,
                  label=f'Median: {median_score:.3f}')
        
        ax.set_xlabel('Extraction Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Per-User Score Distribution', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats text
        stats_text = f'σ = {std_score:.3f}\nn = {len(per_user_scores)}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_recovery_by_type(self, ax, final_metrics: Dict[str, Any]):
        """Plot 3,3: Recovery rate by leakage type (horizontal bars)."""
        # Define leakage types and their scores
        leakage_types = [
            ('System Prompt (SS)', final_metrics.get('ss_system_prompt', 0)),
            ('System Prompt (SM)', final_metrics.get('sm_system_prompt', 0)),
            ('Task Instr. (SS)', final_metrics.get('ss_task_instructions', 0)),
            ('Task Instr. (SM)', final_metrics.get('sm_task_instructions', 0)),
            ('Agent Count (F1)', final_metrics.get('f1_agent_count', 0)),
            ('Comm Density (F1)', final_metrics.get('f1_comm_density', 0)),
            ('U-I Topology', final_metrics.get('ui_topology_recall', 0)),
        ]
        
        labels = [lt[0] for lt in leakage_types]
        values = [lt[1] for lt in leakage_types]
        
        # Color based on value
        colors = ['#27AE60' if v > 0.5 else '#F39C12' if v > 0.2 else '#E74C3C' 
                 for v in values]
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Recovery Score', fontsize=10)
        ax.set_title('Recovery by Leakage Type', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27AE60', label='High (>0.5)'),
            Patch(facecolor='#F39C12', label='Medium (0.2-0.5)'),
            Patch(facecolor='#E74C3C', label='Low (<0.2)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # ═══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════════════
    
    def save_per_turn_metrics(self, filename: str = 'extraction_per_turn_metrics.json'):
        """Save per-turn metrics to JSON for later analysis."""
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.per_turn_metrics, f, indent=2, default=str)
        print(f"[EXTRACTION] Per-turn metrics saved to {save_path}")
        return save_path
    
    def load_per_turn_metrics(self, filepath: str):
        """Load per-turn metrics from JSON."""
        with open(filepath, 'r') as f:
            self.per_turn_metrics = json.load(f)
        print(f"[EXTRACTION] Loaded {len(self.per_turn_metrics)} turn records")
    
    def create_temporal_plots(self, experiment_name: str = 'extraction_attack'):
        """Create individual temporal plots (in addition to dashboard)."""
        if not self.per_turn_metrics:
            print("[EXTRACTION] No temporal data to plot")
            return
        
        # 1. ER over time (detailed)
        self._create_er_timeline_plot(experiment_name)
        
        # 2. Per-target breakdown over time
        self._create_target_breakdown_plot(experiment_name)
        
        # 3. Topology leakage progression
        self._create_topology_progression_plot(experiment_name)
    
    def _create_er_timeline_plot(self, experiment_name: str):
        """Create detailed ER timeline plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        er_values = [m.get('extract_rate', 0) for m in self.per_turn_metrics]
        er_v2 = [m.get('extract_rate_v2', m.get('extract_rate', 0)) 
                for m in self.per_turn_metrics]
        
        ax.plot(turns, er_values, marker='o', linewidth=2, markersize=5,
               color='#2E86AB', label='ER_MAS (7 metrics)')
        ax.plot(turns, er_v2, marker='s', linewidth=2, markersize=5,
               color='#E63946', label='ER_MAS v2 (with U-I)')
        
        ax.fill_between(turns, er_values, alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Extraction Rate', fontsize=12)
        ax.set_title(f'{self.attack_type} Extraction Rate Over Time', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / f'{experiment_name}_er_timeline.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"[EXTRACTION] ER timeline saved to {save_path}")
    
    def _create_target_breakdown_plot(self, experiment_name: str):
        """Create per-target extraction breakdown plot."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        
        targets = [
            ('ss_system_prompt', 'SS_sys (System Prompt)', '#E63946'),
            ('sm_system_prompt', 'SM_sys (System Prompt)', '#F4A261'),
            ('ss_task_instructions', 'SS_task (Task Instructions)', '#2A9D8F'),
            ('f1_agent_count', 'F1_num (Agent Count)', '#264653'),
            ('f1_comm_density', 'F1_comm (Comm Density)', '#E9C46A'),
            ('ui_topology_recall', 'UI_topo (Topology Recall)', '#9B59B6'),
        ]
        
        for ax, (key, title, color) in zip(axes, targets):
            values = [m.get(key, 0) for m in self.per_turn_metrics]
            ax.plot(turns, values, marker='o', linewidth=2, markersize=4, color=color)
            ax.fill_between(turns, values, alpha=0.3, color=color)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Turn', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.attack_type} Per-Target Extraction Progress',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'{experiment_name}_target_breakdown.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"[EXTRACTION] Target breakdown saved to {save_path}")
    
    def _create_topology_progression_plot(self, experiment_name: str):
        """Create U-I topology leakage progression plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        turns = [m['turn'] for m in self.per_turn_metrics]
        leaked = [m.get('ui_items_leaked', 0) for m in self.per_turn_metrics]
        total = [m.get('ui_items_total', 1) for m in self.per_turn_metrics]
        recall = [m.get('ui_topology_recall', 0) for m in self.per_turn_metrics]
        precision = [m.get('ui_topology_precision', 0) for m in self.per_turn_metrics]
        
        # Left: Items leaked
        ax1.bar(turns, leaked, color='#9B59B6', alpha=0.7, label='Items Leaked')
        ax1.axhline(y=total[0] if total else 0, color='#E74C3C', linestyle='--',
                   linewidth=2, label=f'Total Items ({total[0] if total else 0})')
        ax1.set_xlabel('Turn', fontsize=12)
        ax1.set_ylabel('Number of Items', fontsize=12)
        ax1.set_title('U-I Items Leaked Over Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right: Recall and Precision
        ax2.plot(turns, recall, marker='o', linewidth=2, markersize=5,
                color='#27AE60', label='Recall')
        ax2.plot(turns, precision, marker='s', linewidth=2, markersize=5,
                color='#3498DB', label='Precision')
        ax2.set_xlabel('Turn', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Topology Recall & Precision', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.attack_type} U-I Topology Leakage Progression',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'{experiment_name}_topology_progression.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"[EXTRACTION] Topology progression saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory function for creating appropriate dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def create_extraction_visualizations(attacker, output_dir: str, experiment_name: str):
    """
    Create extraction visualizations for MASLeak/MAMA/MASTER attacks.
    
    Args:
        attacker: The attacker instance (MASLeakAttacker, MAMAAttacker, etc.)
        output_dir: Directory to save visualizations
        experiment_name: Name prefix for saved files
    """
    # Determine attack type
    attack_type = 'MASLeak'
    if hasattr(attacker, '__class__'):
        class_name = attacker.__class__.__name__
        if 'MAMA' in class_name:
            attack_type = 'MAMA'
        elif 'MASTER' in class_name:
            attack_type = 'MASTER'
    
    dashboard = ExtractionDashboard(output_dir, attack_type)
    
    # Get final metrics
    if hasattr(attacker, 'metrics'):
        final_metrics = attacker.metrics
    elif hasattr(attacker, 'evaluate'):
        final_metrics = attacker.evaluate()
    else:
        final_metrics = {}
    
    # Get per-turn metrics if available
    if hasattr(attacker, 'per_turn_metrics'):
        dashboard.per_turn_metrics = attacker.per_turn_metrics
    elif hasattr(attacker, 'all_responses'):
        # Build per-turn metrics from responses
        _build_per_turn_from_responses(dashboard, attacker)
    
    # Create dashboard
    dashboard.create_dashboard(final_metrics, experiment_name)
    
    # Create individual temporal plots
    dashboard.create_temporal_plots(experiment_name)
    
    # Save per-turn metrics
    dashboard.save_per_turn_metrics(f'{experiment_name}_per_turn_metrics.json')
    
    return dashboard


def _build_per_turn_from_responses(dashboard: ExtractionDashboard, attacker):
    """Build per-turn metrics from attacker's response history."""
    if not hasattr(attacker, 'all_responses') or not attacker.all_responses:
        return
    
    # Group responses by turn
    from collections import defaultdict
    by_turn = defaultdict(list)
    for resp in attacker.all_responses:
        turn = resp.get('turn', 0)
        by_turn[turn].append(resp)
    
    # Build cumulative metrics per turn
    cumulative_extracted = {}
    for turn in sorted(by_turn.keys()):
        responses = by_turn[turn]
        
        # Merge extracted data
        for resp in responses:
            extracted = resp.get('extracted', {})
            for k, v in extracted.items():
                if v and k not in cumulative_extracted:
                    cumulative_extracted[k] = v
        
        # Compute metrics at this turn (simplified)
        metrics = {
            'responses_collected': len(responses),
            'cumulative_responses': sum(len(by_turn[t]) for t in by_turn if t <= turn),
            'fields_extracted': len(cumulative_extracted),
        }
        
        dashboard.record_turn_metrics(turn, metrics)
