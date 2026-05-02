"""
Visualization tools for MASLeak results
"""

from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from .leakage_types import LeakageType, LeakageMetrics


class MASLeakVisualizer:
    """Visualize MASLeak attack results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def plot_recovery_rates(self, metrics: Dict[LeakageType, LeakageMetrics], 
                           save_path: Optional[Path] = None):
        """Plot recovery rates for each leakage type"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = [lt.value for lt in metrics.keys()]
        rates = [m.recovery_rate for m in metrics.values()]
        
        bars = ax.bar(types, rates, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Recovery Rate', fontsize=12)
        ax.set_xlabel('Leakage Type', fontsize=12)
        ax.set_title('Information Recovery Rates by Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'recovery_rates.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved recovery rates plot to {save_path}")
    
    def plot_accuracy_comparison(self, metrics: Dict[LeakageType, LeakageMetrics],
                                save_path: Optional[Path] = None):
        """Plot accuracy metrics comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        types = [lt.value for lt in metrics.keys()]
        recovery = [m.recovery_rate for m in metrics.values()]
        accuracy = [m.exact_match_accuracy for m in metrics.values()]
        
        x = np.arange(len(types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, recovery, width, label='Recovery Rate', 
                      color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, accuracy, width, label='Exact Match Accuracy',
                      color='coral', alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Leakage Type', fontsize=12)
        ax.set_title('Recovery Rate vs Accuracy by Leakage Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'accuracy_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved accuracy comparison plot to {save_path}")
    
    def plot_confidence_distribution(self, observations_file: Path,
                                    save_path: Optional[Path] = None):
        """Plot confidence score distributions"""
        # Load observations
        with open(observations_file, 'r') as f:
            observations = json.load(f)
        
        confidences = [obs['confidence'] for obs in observations]
        
        if not confidences:
            print("No confidence scores to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2.boxplot(confidences, vert=True)
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence Score Statistics', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {np.mean(confidences):.3f}\n"
        stats_text += f"Median: {np.median(confidences):.3f}\n"
        stats_text += f"Std: {np.std(confidences):.3f}"
        ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confidence distribution plot to {save_path}")
    
    def plot_recovery_over_time(self, metrics_history: Dict[str, List[Dict]],
                               save_path: Optional[Path] = None):
        """Plot how recovery rate changes over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for leakage_type, history in metrics_history.items():
            if not history:
                continue
            
            recovery_rates = [m['recovery_rate'] for m in history]
            timesteps = list(range(len(recovery_rates)))
            
            ax.plot(timesteps, recovery_rates, marker='o', label=leakage_type, linewidth=2)
        
        ax.set_xlabel('Observation Step', fontsize=12)
        ax.set_ylabel('Recovery Rate', fontsize=12)
        ax.set_title('Information Recovery Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'recovery_over_time.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved recovery over time plot to {save_path}")
    
    def plot_leakage_heatmap(self, metrics: Dict[LeakageType, LeakageMetrics],
                            save_path: Optional[Path] = None):
        """Plot heatmap of different metrics across leakage types"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = [lt.value for lt in metrics.keys()]
        metric_names = ['Recovery\nRate', 'Exact Match\nAccuracy', 'Jaccard\nSimilarity', 
                       'Avg\nConfidence']
        
        # Build data matrix
        data = []
        for m in metrics.values():
            row = [
                m.recovery_rate,
                m.exact_match_accuracy,
                m.jaccard_similarity,
                m.avg_confidence
            ]
            data.append(row)
        
        data = np.array(data)
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(types)))
        ax.set_xticklabels(metric_names)
        ax.set_yticklabels(types)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(types)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Leakage Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'leakage_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved leakage heatmap to {save_path}")
    
    def create_summary_dashboard(self, summary: Dict[str, Any],
                                save_path: Optional[Path] = None):
        """Create comprehensive summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall statistics (top left)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        overall_text = "MASLeak Attack Summary\n\n"
        overall_text += f"Total Observations: {summary['total_observations']}\n"
        overall_text += f"Elapsed Time: {summary['elapsed_time']:.2f}s\n"
        overall_text += f"Avg Recovery Rate: {summary['overall_stats']['avg_recovery_rate']:.2%}\n"
        overall_text += f"Avg Accuracy: {summary['overall_stats']['avg_accuracy']:.2%}"
        
        ax1.text(0.5, 0.5, overall_text, transform=ax1.transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Recovery rates bar chart (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        types = list(summary['leakage_metrics'].keys())
        rates = [m['recovery_rate'] for m in summary['leakage_metrics'].values()]
        ax2.barh(types, rates, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Recovery Rate')
        ax2.set_title('Recovery Rates')
        ax2.set_xlim(0, 1)
        
        # Accuracy bar chart (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        accuracies = [m['exact_match_accuracy'] for m in summary['leakage_metrics'].values()]
        ax3.barh(types, accuracies, color='coral', alpha=0.7)
        ax3.set_xlabel('Accuracy')
        ax3.set_title('Exact Match Accuracy')
        ax3.set_xlim(0, 1)
        
        # Observations by type (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        obs_counts = list(summary['overall_stats']['total_observations_by_type'].values())
        ax4.bar(range(len(types)), obs_counts, color='green', alpha=0.7)
        ax4.set_xticks(range(len(types)))
        ax4.set_xticklabels(types, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Count')
        ax4.set_title('Observations by Type')
        
        # Detailed metrics table (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        table_data = []
        for lt, metrics in summary['leakage_metrics'].items():
            row = [
                lt,
                f"{metrics['recovery_rate']:.2%}",
                f"{metrics['exact_match_accuracy']:.2%}",
                f"{metrics['recovered_items']}/{metrics['total_items']}"
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Type', 'Recovery', 'Accuracy', 'Items'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.suptitle('MASLeak Information Leakage Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path is None:
            save_path = self.output_dir / 'summary_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary dashboard to {save_path}")
    
    def generate_all_plots(self, results_dir: Path):
        """Generate all visualization plots from saved results"""
        results_dir = Path(results_dir)
        
        # Load summary
        with open(results_dir / 'masleak_summary.json', 'r') as f:
            summary = json.load(f)
        
        # Convert string keys back to LeakageType
        metrics = {}
        for lt_str, m_dict in summary['leakage_metrics'].items():
            lt = LeakageType(lt_str)
            metrics[lt] = LeakageMetrics(leakage_type=lt)
            for key, value in m_dict.items():
                if hasattr(metrics[lt], key):
                    setattr(metrics[lt], key, value)
        
        # Generate plots
        self.plot_recovery_rates(metrics)
        self.plot_accuracy_comparison(metrics)
        self.plot_leakage_heatmap(metrics)
        self.create_summary_dashboard(summary)
        
        # Plot confidence distributions for each type
        for lt in metrics.keys():
            obs_file = results_dir / f'observations_{lt.value}.json'
            if obs_file.exists():
                save_path = self.output_dir / f'confidence_{lt.value}.png'
                self.plot_confidence_distribution(obs_file, save_path)
        
        # Plot recovery over time if history available
        history_file = results_dir / 'metrics_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            self.plot_recovery_over_time(history)
        
        print(f"\nAll visualizations saved to {self.output_dir}")
