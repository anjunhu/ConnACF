#!/usr/bin/env python3
"""
CORBA plotter — 3-panel layout: User ASR, Item ASR, DoS Rate.

Usage:
    python tools/plot_corba.py \\
        --output ../figure/corba_2c_intermat.pdf \\
        --connacf_dirs \\
            attack_output/corba/corba_canonical_2cand/ml-100k-100user-dense50/... \\
            attack_output/corba/corba_canonical_2cand/ml-100k-100user-medium100/... \\
            attack_output/corba/corba_canonical_2cand/ml-100k-100user-sparse200/... \\
        --labels "dense50" "medium100" "sparse200"
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import _plot_common as _pc
from _plot_common import (
    base_parser, load_datasets, group_by_label,
    get_color_for_label, _pub_figsize, _pub_savefig,
    infer_title_suffix, apply_bg, sort_labels_for_legend, SUBPLOT_H,
    LINE_W, LINE_A,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PANELS = [
    ('resources.corba_infection_rate_memory',      'User ASR',  'User ASR'),
    ('resources.corba_infection_rate_memory_item', 'Item ASR',  'Item ASR'),
    ('resources.corba_blocking_rate',              'DoS Rate',  'DoS Rate'),
]

_BASELINE_METRICS = {'resources.corba_infection_rate_memory_item'}
_SMOOTH_METRICS   = {'resources.corba_blocking_rate', 'resources.corba_iteration_exhaustion_rate'}


def _get_nested(d, dotpath, default=0.0):
    for key in dotpath.split('.'):
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d if d is not None else default


def load_series(run_dir, metric_path):
    from _plot_common import MAX_DISPLAY_TURNS
    turn_files = sorted(Path(run_dir).glob('turn_*.json'),
                        key=lambda f: int(f.stem.split('_')[1]))[:MAX_DISPLAY_TURNS]
    turns, values = [], []
    for tf in turn_files:
        try:
            with open(tf) as f:
                data = json.load(f)
            turns.append(int(tf.stem.split('_')[1]))
            values.append(float(_get_nested(data, metric_path)))
        except Exception as e:
            logger.warning(f"Skipping {tf}: {e}")
    return turns, values


def plot_corba(datasets, output_file, title=None):
    n = len(PANELS)
    fig, axes = plt.subplots(1, n,
                             figsize=_pub_figsize(_pc.SUBPLOT_W * n, SUBPLOT_H,
                                                  nrows=1, row_h=SUBPLOT_H),
                             sharey=True)

    auto_suffix = infer_title_suffix(datasets)
    if not _pc._pub_mode:
        fig.suptitle(title or f'CORBA Metrics{auto_suffix}', fontsize=13, fontweight='bold')

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}
    apply_bg(fig, list(axes), label_order)

    for ax, (metric_path, ylabel, panel_title) in zip(axes, PANELS):
        panel_series = []
        for i, label in enumerate(label_order):
            all_turns, all_vals = [], []
            for ds in label_groups[label]:
                t, v = load_series(ds['path'], metric_path)
                all_turns.extend(t)
                all_vals.extend(v)
            if not all_turns:
                continue
            by_turn = defaultdict(list)
            for t, v in zip(all_turns, all_vals):
                by_turn[t].append(v)
            ts = sorted(by_turn)
            vals = [np.mean(by_turn[t]) for t in ts]
            stds = [np.std(by_turn[t])  for t in ts]
            panel_series.append((label, label_colors[label], ts, vals, stds))

        if not panel_series:
            ax.text(0.5, 0.5, 'not instrumented', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray', style='italic')
        else:
            # Baseline-correct attacker-seeded item metrics
            if metric_path in _BASELINE_METRICS:
                t0_vals = [v[0] for _, _, _, v, _ in panel_series if v]
                if t0_vals and np.median(t0_vals) > 0.05:
                    panel_series = [(lbl, col, ts, [x - v[0] for x in v], s)
                                    for lbl, col, ts, v, s in panel_series]

            global_max = max(
                (max((x for x in v if not np.isnan(x)), default=0.0)
                 for _, _, _, v, _ in panel_series),
                default=1.0,
            )
            for label, color, ts, vals, stds in panel_series:
                norm = [x / global_max if global_max > 0 else x for x in vals]
                if metric_path in _SMOOTH_METRICS and len(norm) >= 5:
                    norm = list(np.convolve(norm, np.ones(7) / 7, mode='same'))
                ls = _pc.get_linestyle_for_label(label)
                mk = _pc.get_marker_for_label(label)
                every = max(1, len(ts) // 8)
                ax.plot(ts, norm, color=color, linewidth=LINE_W, alpha=LINE_A,
                        label=label, linestyle=ls,
                        marker=mk, markevery=every, markersize=9, markeredgewidth=1.2,
                        markerfacecolor=(*color[:3], 0.55) if len(color) == 3 else color)
                norm_std = [s / global_max if global_max > 0 else s for s in stds]
                if any(s > 0 for s in norm_std):
                    ax.fill_between(ts,
                                    np.array(norm) - np.array(norm_std),
                                    np.array(norm) + np.array(norm_std),
                                    color=color, alpha=0.15)

        ax.set_xlabel('Turn', fontsize=11)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(panel_title, fontsize=22, fontweight='bold')
        else:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('#888888')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)
    _pub_savefig(output_file)
    logger.info(f"Saved: {output_file}")


def main():
    parser = base_parser(__doc__)
    args = parser.parse_args()

    if getattr(args, 'pub', False):
        _pc._pub_figsize_override = (10.5, 2.8)

    datasets = load_datasets(args)
    if not datasets:
        logger.error("No datasets loaded -- check --connacf_dirs")
        return

    plot_corba(datasets, args.output, args.title)


if __name__ == '__main__':
    main()
