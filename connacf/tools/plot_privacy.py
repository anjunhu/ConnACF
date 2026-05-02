#!/usr/bin/env python3
"""
Plot stealth / privacy metrics for PromptInfection and InjecAgent attacks.

Plots a single custom metric per turn (dot-notation path into turn JSON).

Usage:
    python tools/plot_privacy.py --last_epoch 150 \\
        --metric dissemination.infection.pi_infection_rate_forward \\
        --title "PromptInfection x ConnaCF x MovieLens100 (Infection Rate)" \\
        --output "../figure/pi_infection.png" \\
        --connacf_dirs \\
            attack_output/prompt_infection/pi_canonical_1cand/ml-100k-100user-dense/task_0 \\
            attack_output/prompt_infection/pi_canonical_2cand/ml-100k-100user-dense/task_0 \\
            attack_output/prompt_infection/pi_canonical_3cand/ml-100k-100user-dense/task_0 \\
        --labels "Sparse (1c)" "Medium (2c)" "Dense (3c)"
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from _plot_common import (
    base_parser, load_datasets, group_by_label,
    interpolate_nans, last_valid,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def plot_privacy(datasets, metric_name, output_file, title=None):
    if not datasets:
        logger.error("No datasets to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    fig.suptitle(title or f'{metric_name} per Turn', fontsize=13, fontweight='bold')

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    label_order, label_groups = group_by_label(datasets)
    y_offset = 0.97

    for i, label in enumerate(label_order):
        group = label_groups[label]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        if len(group) == 1:
            d = group[0]
            vals = d.get('custom_metric', [])
            if d['turns'] and vals:
                t, v, mask = interpolate_nans(d['turns'], vals)
                ax.plot(t, v, linewidth=2.2, color=color, label=label, alpha=0.75)
                final = v[mask][-1] if mask.any() else None
                if final is not None:
                    ax.text(0.97, y_offset, f"{label}: {final:.3f}",
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    y_offset -= 0.06
        else:
            all_turns = sorted({t for d in group for t in d['turns']})
            turn_idx = {t: i for i, t in enumerate(all_turns)}
            mat = np.full((len(group), len(all_turns)), np.nan)
            for di, d in enumerate(group):
                vals = d.get('custom_metric', [])
                for ti, (t, v) in enumerate(zip(d['turns'], vals)):
                    if t in turn_idx:
                        mat[di, turn_idx[t]] = v
            with np.errstate(all='ignore'):
                mean_v = np.nanmean(mat, axis=0)
                min_v = np.nanmin(mat, axis=0)
                max_v = np.nanmax(mat, axis=0)
            valid = ~np.isnan(mean_v)
            if valid.any():
                ax.fill_between(all_turns, min_v, max_v, color=color, alpha=0.2)
                ax.plot(all_turns, mean_v, linewidth=2.2, color=color,
                        label=f'{label} (n={len(group)})', alpha=0.75)

    ax.set_xlabel('Turn', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    parser = base_parser('Plot stealth/privacy metrics (PromptInfection, InjecAgent)')
    parser.add_argument('--metric', type=str, required=True,
                        help='Dot-notation metric path, e.g. stealth.ia_asr_valid')
    args = parser.parse_args()
    datasets = load_datasets(args, custom_metric=args.metric)
    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets found.")
        return
    plot_privacy(valid, args.metric, args.output, args.title)


if __name__ == '__main__':
    main()
