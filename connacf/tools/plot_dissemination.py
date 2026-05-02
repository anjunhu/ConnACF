#!/usr/bin/env python3
"""
Plot user/item victim (contamination) rates for dissemination-focused attacks.
(CheatAgent, DrunkAgent, NetSafe, TOMA, MAMA, ...)

No reverse-engineering row — RE metrics are collected passively but are not
the focus of these attacks. Use plot_reveng.py for MASLeak / MASTER.

Usage:
    python tools/plot_dissemination.py --last_epoch 250 \\
        --title "DrunkAgent x ConnaCF x MovieLens100 (Inference, 101i, 50%atk)" \\
        --output "../figure/drunk_inference_101i.png" \\
        --auto_label \\
        --connacf_dirs \\
            attack_output/drunk/drunk_50percent_1cand/ml-100k-100user-medium100/260219121824 \\
            attack_output/drunk/drunk_50percent/ml-100k-100user-medium100/... \\
            attack_output/drunk/drunk_50percent_3cand/ml-100k-100user-medium100/260219121912 \\
        --labels "Sparse (1c)" "Medium (2c)" "Dense (3c)"
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from _plot_common import _pub_figsize, _pub_savefig, SUBPLOT_H
import _plot_common as _pc
from _plot_common import (
    base_parser, load_datasets, group_by_label,
    get_color_for_label, sort_labels_for_legend, apply_bg, MARKERS,
    plot_group, annotate_finals, safe_max, last_valid,
    infer_title_suffix, maybe_legend,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def plot_dissemination(datasets, output_file, title=None, normalize_turns=False,
                       figsize=None, title_fontsize=None):
    if not datasets:
        logger.error("No datasets to plot")
        return

    is_png = not str(output_file).endswith('.pdf')

    # Always 2 panels: user victim + item victim (no NDCG)
    ncols = 2
    base_w, base_h = _pub_figsize(7, 2.4, nrows=1, row_h=SUBPLOT_H * 1.5)
    # For PNG, add extra height to accommodate the legend below the plots
    fig_h = base_h + 0.7 if is_png else base_h
    if figsize is not None:
        base_w, fig_h = figsize
    fig, axes = plt.subplots(1, ncols, figsize=(base_w, fig_h), sharey=True)
    _title_fs = title_fontsize if title_fontsize is not None else _pc.TITLE_FONTSIZE
    axes = list(axes)
    auto_suffix = infer_title_suffix(datasets)
    if not _pc._pub_mode:
        fig.suptitle(title if title else
                     f'ConnaCF Victim Rate Comparison\n(% of Innocent Agents Contaminated){auto_suffix}',
                     fontsize=14, fontweight='bold')

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}
    apply_bg(fig, axes, label_order)

    xlabel = 'Equivalent Turn (1 unit = full cohort pass)' if normalize_turns else 'Turn'

    specs = [
        ('user_victim_pct', None,        'User ASR',  100),
        ('item_victim_pct', None,        'Item ASR',  100),
        ('system_performance', 'ndcg_at_10', 'NDCG@10',          1.05),
    ]

    for col, (ax, (metric, nested, ylabel, ymax)) in enumerate(zip(axes, specs[:ncols])):
        max_val = 0
        for i, label in enumerate(label_order):
            group = label_groups[label]
            color = label_colors[label]
            marker = MARKERS[i % len(MARKERS)]
            max_val = max(max_val, plot_group(ax, group, label, color, marker, metric, nested_key=nested))

        ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(ylabel, fontsize=_title_fs, fontweight='bold')
        else:
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=_pc.AXIS_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(ymax, max_val * 1.1) if max_val > 0 else ymax)
        if not _pc._pub_mode:
            annotate_finals(ax, label_order, label_groups, label_colors, metric,
                            nested_key=nested, fmt='.1f' if ymax == 100 else '.3f',
                            unit='%' if ymax == 100 else '')

    # ── Legend ──
    # Pull handles from the first axis (all axes share the same series)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        if is_png:
            maybe_legend(fig, handles, leg_labels,
                       loc='lower center',
                       ncol=min(len(handles), 6),
                       fontsize=9 if not _pc._pub_mode else 7,
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.18, wspace=0.08)
        else:
            maybe_legend(fig, handles, leg_labels,
                       loc='lower center',
                       ncol=min(len(handles), 6),
                       fontsize=9 if not _pc._pub_mode else 7,
                       bbox_to_anchor=(0.5, 0.0),
                       frameon=True)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.22, wspace=0.08)
    else:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.08)

    _pub_savefig(output_file)
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    parser = base_parser('Plot dissemination (victim rate) comparison across ConnaCF runs')
    args = parser.parse_args()
    datasets = load_datasets(args)
    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets found.")
        return
    plot_dissemination(valid, args.output, args.title, normalize_turns=args.normalize_turns)


if __name__ == '__main__':
    main()
