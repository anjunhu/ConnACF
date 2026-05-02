#!/usr/bin/env python3
"""
Reverse-engineering only plot for MASLeak (and similar RE-focused attacks).

Layout: 3 rows x 2 cols — all 6 RE metrics, no contamination rows.

  [0,0] U-I Topology Recall      [0,1] U-I Topology Precision
  [1,0] U-I Topology F1          [1,1] Overall Extraction Rate
  [2,0] System Prompt Similarity [2,1] Task Instr. Similarity

Usage:
    python tools/plot_reveng.py --last_epoch 30 \\
        --title "MASLeak x ConnaCF x MovieLens100 (InterMat Density, 2c)" \\
        --output "../figure/masleak_InterMat.png" \\
        --auto_label \\
        --connacf_dirs \\
            attack_output/masleak/masleak_2cand/ml-100k-100user-sparse200/260226233305 \\
            attack_output/masleak/masleak_2cand/ml-100k-100user-medium100/260220125121 \\
            attack_output/masleak/masleak_2cand/ml-100k-100user-dense50/260226233247 \\
        --labels "200i" "100i" "50i"
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from _plot_common import _pub_figsize, _pub_savefig, SUBPLOT_W, SUBPLOT_H
import _plot_common as _pc
from _plot_common import (
    base_parser, load_datasets, group_by_label,
    get_color_for_label, sort_labels_for_legend, apply_bg, MARKERS, RE_METRIC_KEYS,
    plot_group, annotate_finals, safe_max,
    infer_title_suffix,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Full 3x2 layout for PNG — all 6 RE metrics
GRID_FULL = [
    ('ui_topology_recall',    'U-I Topology Recall'),
    ('ui_topology_precision', 'U-I Topology Precision'),
    ('ui_topology_f1',        'U-I Topology F1'),
    ('extract_rate',          'Overall Extraction Rate'),
    ('ss_system_prompt',      'Sys Prompt Sim.'),
    ('ss_task_instructions',  'Task Instr. Similarity'),
]
# Compact 1x3 layout for PDF (pub mode) — key metrics only, single row
GRID_PUB = [
    ('ui_topology_recall',    'U-I Topology Recall'),
    ('ui_topology_f1',        'U-I Topology F1'),
    ('ss_system_prompt',      'Sys Prompt Sim.'),
]


def plot_reveng(datasets, output_file, title=None):
    if not datasets:
        logger.error("No datasets to plot")
        return

    # PDF: compact 1×3; PNG: full 3×2
    grid_def = GRID_PUB if _pc._pub_mode else GRID_FULL
    ncols = 3 if _pc._pub_mode else 2

    active_grid = [(k, lbl) for k, lbl in grid_def
                   if any(any(not np.isnan(v)
                              for v in d.get('reverse_engineering', {}).get(k, []))
                          for d in datasets)]
    if not active_grid:
        logger.error("No plottable RE metrics found")
        return

    act_rows = (len(active_grid) + ncols - 1) // ncols
    fig, axes = plt.subplots(act_rows, ncols,
                             figsize=_pc._pub_figsize(ncols * SUBPLOT_W, act_rows * SUBPLOT_H,
                                                      nrows=act_rows, row_h=SUBPLOT_H),
                             squeeze=False)

    if not _pc._pub_mode:
        fig.suptitle(
            (title + infer_title_suffix(datasets)) if title else
            f'ConnaCF Reverse Engineering Metrics{infer_title_suffix(datasets)}',
            fontsize=14, fontweight='bold',
        )

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}
    apply_bg(fig, axes, label_order)

    for idx, (re_key, re_label) in enumerate(active_grid):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        max_val = 0
        for i, label in enumerate(label_order):
            group = label_groups[label]
            color = label_colors[label]
            marker = MARKERS[i % len(MARKERS)]
            max_val = max(max_val, plot_group(
                ax, group, label, color, marker,
                'reverse_engineering', nested_key=re_key,
            ))

        ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(re_label, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
        else:
            if col == 0:
                ax.set_ylabel(re_label, fontsize=_pc.AXIS_FONTSIZE)
        ax.grid(True, alpha=0.3)
        if col > 0:
            ax.tick_params(labelleft=False)
        ax.set_ylim(0, max(1.05, max_val * 1.1) if max_val > 0 else 1.05)
        if not _pc._pub_mode:
            annotate_finals(ax, label_order, label_groups, label_colors,
                            'reverse_engineering', nested_key=re_key, fmt='.3f', unit='')

    # Hide unused cells
    for idx in range(len(active_grid), act_rows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.08)

    _pub_savefig(output_file)
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    parser = base_parser('Plot RE-only metrics for MASLeak (3x2 grid)')
    args = parser.parse_args()
    datasets = load_datasets(args)
    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets found.")
        return
    plot_reveng(valid, args.output, args.title)


if __name__ == '__main__':
    main()
