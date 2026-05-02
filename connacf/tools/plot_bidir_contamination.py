#!/usr/bin/env python3
"""
Bidir attack: contamination drop + reverse-engineering metrics (MASTER / TOMA).

Row 1: User & Item contamination rates
Row 2: RE metric panels (topology recall/precision/F1, extraction rate, etc.)

Usage:
    python tools/plot_bidir_contamination.py --last_epoch 30 \\
        --title "MASTER x ConnaCF x MovieLens100 (InterMat Density, 2c)" \\
        --output "../figure/master_InterMat.png" \\
        --auto_label \\
        --connacf_dirs \\
            attack_output/master/master_2cand/ml-100k-100user-sparse200/260227084528 \\
            attack_output/master/master_2cand/ml-100k-100user-medium100/260226233137 \\
        --labels "200i" "100i"
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from _plot_common import (
    base_parser, load_datasets, group_by_label,
    get_color_for_label, MARKERS, RE_METRIC_KEYS,
    plot_group, annotate_finals, safe_max, last_valid,
    infer_title_suffix,
)
import _plot_common as _pc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RE_LABELS = {
    'ui_topology_recall':    'U-I Topology Recall',
    'ui_topology_precision': 'U-I Topology Precision',
    'ui_topology_f1':        'U-I Topology F1',
    'extract_rate':          'Overall Extraction Rate',
    'ss_system_prompt':      'System Prompt Similarity',
    'ss_task_instructions':  'Task Instr. Similarity',
}


def plot_reveng(datasets, output_file, title=None):
    if not datasets:
        logger.error("No datasets to plot")
        return

    # Determine which RE metrics have actual data
    active_re = []
    for k in RE_METRIC_KEYS:
        for d in datasets:
            vals = d.get('reverse_engineering', {}).get(k, [])
            if any(not np.isnan(v) for v in vals):
                active_re.append(k)
                break

    n_re = len(active_re)
    n_cols = max(2, n_re)

    if n_re:
        fig, all_axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10))
        if n_cols == 1:
            all_axes = all_axes.reshape(2, 1)
        axes_r1 = all_axes[0]
        axes_r2 = all_axes[1]
        for i in range(2, n_cols):
            axes_r1[i].set_visible(False)
        for i in range(n_re, n_cols):
            axes_r2[i].set_visible(False)
    else:
        fig, axes_r1 = plt.subplots(1, 2, figsize=(14, 5))
        axes_r2 = None

    fig.suptitle(title + infer_title_suffix(datasets) if title else
                 f'ConnaCF Reverse Engineering Metrics{infer_title_suffix(datasets)}',
                 fontsize=14, fontweight='bold')

    label_order, label_groups = group_by_label(datasets)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}

    # Row 1: victim rates
    max_user, max_item = 0, 0
    for i, label in enumerate(label_order):
        group = label_groups[label]
        color = label_colors[label]
        marker = MARKERS[i % len(MARKERS)]
        max_user = max(max_user, plot_group(axes_r1[0], group, label, color, marker, 'user_victim_pct'))
        max_item = max(max_item, plot_group(axes_r1[1], group, label, color, marker, 'item_victim_pct'))

    for ax, ylabel, max_val in [
        (axes_r1[0], 'User Victim Rate (%)', max_user),
        (axes_r1[1], 'Item Victim Rate (%)', max_item),
    ]:
        ax.set_xlabel('Turn', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel.replace(' (%)', ' Contamination Rate'), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(100, max_val * 1.1) if max_val > 0 else 100)

    annotate_finals(axes_r1[0], label_order, label_groups, label_colors, 'user_victim_pct')
    annotate_finals(axes_r1[1], label_order, label_groups, label_colors, 'item_victim_pct')

    # Row 2: RE metrics
    if active_re and axes_r2 is not None:
        for pi, re_key in enumerate(active_re):
            ax = axes_r2[pi]
            max_val = 0
            for i, label in enumerate(label_order):
                group = label_groups[label]
                color = label_colors[label]
                marker = MARKERS[i % len(MARKERS)]
                max_val = max(max_val, plot_group(ax, group, label, color, marker,
                                                  'reverse_engineering', nested_key=re_key))
            ax.set_xlabel('Turn', fontsize=11)
            ax.set_ylabel(RE_LABELS.get(re_key, re_key), fontsize=11)
            ax.set_title(RE_LABELS.get(re_key, re_key), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(1.05, max_val * 1.1) if max_val > 0 else 1.05)
            annotate_finals(ax, label_order, label_groups, label_colors,
                            'reverse_engineering', nested_key=re_key, fmt='.3f', unit='')

    handles, labels = axes_r1[0].get_legend_handles_labels()
    if handles and not _pc._pub_mode:
        fig.legend(handles, labels, loc='lower center', ncol=min(len(label_order), 5),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 if active_re else 0.15)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    parser = base_parser('Plot bidir contamination drop + RE metrics (MASTER / TOMA)')
    args = parser.parse_args()
    datasets = load_datasets(args)
    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets found.")
        return
    plot_reveng(valid, args.output, args.title)


if __name__ == '__main__':
    main()
