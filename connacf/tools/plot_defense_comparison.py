#!/usr/bin/env python3
"""
Defense comparison plot: G-Safeguard and BlindGuard vs unguarded Qwen3 baseline.
Rows: 1cand, 2cand, 3cand  |  Cols: User victim %, Item victim %

Usage (from connacf/):
    python tools/plot_defense_comparison.py [--pub] [--output figure/defense_comparison.png]
"""
import argparse
import json
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _plot_common as _pc
from _plot_common import _pub_figsize, _pub_savefig, SUBPLOT_H, apply_bg

BASE = os.path.join(os.path.dirname(__file__), '..', 'attack_output', 'netsafe')

RUNS = {
    1: {
        'unguarded':   'misinfo_1cand/ml-100k-100user-medium100-seed43/260327094928',
        'G-Safeguard': 'misinfo_1cand_g_safeguard/ml-100k-100user-medium100/260331172356',
        'BlindGuard':  'misinfo_1cand_blindguard/ml-100k-100user-medium100/260331172110',
    },
    2: {
        'unguarded':   'misinfo_2cand/ml-100k-100user-medium100-seed43/260324165825',
        'G-Safeguard': 'misinfo_2cand_g_safeguard/ml-100k-100user-medium100/260331172359',
        'BlindGuard':  'misinfo_2cand_blindguard/ml-100k-100user-medium100/260331172359',
    },
    3: {
        'unguarded':   None,  # no Qwen3 medium100 baseline for 3c yet
        'G-Safeguard': 'misinfo_3cand_g_safeguard/ml-100k-100user-medium100/260331172356',
        'BlindGuard':  'misinfo_3cand_blindguard/ml-100k-100user-medium100/260331172401',
    },
}

COLORS = {
    'unguarded':   '#d62728',   # red
    'G-Safeguard': '#2ca02c',   # green
    'BlindGuard':  '#1f77b4',   # blue
}
LINESTYLES = {
    'unguarded':   '-',
    'G-Safeguard': '--',
    'BlindGuard':  ':',
}


def load_series(rel_path):
    """Return (turns[], user_pct[], item_pct[]) from a run dir."""
    path = os.path.join(BASE, rel_path)
    files = glob.glob(os.path.join(path, 'turn_*.json'))
    if not files:
        return [], [], []

    def tnum(p):
        return int(os.path.basename(p).replace('turn_', '').replace('.json', ''))

    turns, user_pcts, item_pcts = [], [], []
    for f in sorted(files, key=tnum):
        d = json.load(open(f))
        t = tnum(f)
        uj = d.get('dissemination', {}).get('llm_judge', {}).get('user', {})
        ij = d.get('dissemination', {}).get('llm_judge', {}).get('item', {})
        uc, un = uj.get('contaminated_count'), uj.get('num_agents_evaluated')
        ic, in_ = ij.get('contaminated_count'), ij.get('num_agents_evaluated')
        if uc is not None and un:
            turns.append(t)
            user_pcts.append(uc / un * 100)
            item_pcts.append(ic / in_ * 100 if ic is not None and in_ else 0.0)
    return turns, user_pcts, item_pcts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pub', action='store_true')
    parser.add_argument('--output', default='figure/defense_comparison.png')
    args = parser.parse_args()

    if args.pub:
        _pc._pub_mode = True

    nrows, ncols = 3, 2
    base_w, base_h = _pub_figsize(14, nrows, nrows=nrows, row_h=SUBPLOT_H)
    fig, axes = plt.subplots(nrows, ncols, figsize=(base_w, base_h + 0.5), squeeze=False)

    if not _pc._pub_mode:
        fig.suptitle('NetSafe Defense Comparison: G-Safeguard & BlindGuard vs Unguarded (Qwen3)',
                     fontsize=12, fontweight='bold', y=0.98)

    col_titles = ['User Victim Rate (%)', 'Item Victim Rate (%)']
    row_labels = ['1 candidate', '2 candidates', '3 candidates']

    for row, ncand in enumerate([1, 2, 3]):
        # Load all series for this row
        all_series = {}
        for defense, rel_path in RUNS[ncand].items():
            if rel_path is None:
                continue
            t, u, i = load_series(rel_path)
            if t:
                all_series[defense] = (t, u, i)

        # Shared x-limit: unguarded range if available, else max of all
        xlim = max(all_series['unguarded'][0]) if 'unguarded' in all_series else max(max(v[0]) for v in all_series.values())
        xlim *= 1.02

        for col, metric_idx in enumerate([0, 1]):
            ax = axes[row][col]
            apply_bg(fig, [ax], list(COLORS.keys()))

            for defense, (turns, user_pcts, item_pcts) in all_series.items():
                vals = user_pcts if metric_idx == 0 else item_pcts
                ax.plot(turns, vals,
                        label=defense,
                        color=COLORS[defense],
                        linestyle=LINESTYLES[defense],
                        linewidth=1.8,
                        marker='o' if len(turns) < 20 else None,
                        markersize=4)

            fs = 9 if _pc._pub_mode else 10
            ax.set_ylim(0, 105)
            ax.set_xlim(0, xlim)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=fs - 1)
            if row == nrows - 1:
                ax.set_xlabel('Turn', fontsize=fs)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=fs, fontweight='bold')
            if 'unguarded' not in all_series and col == 0:
                ax.text(0.97, 0.95, '(no unguarded baseline yet)',
                        transform=ax.transAxes, fontsize=7,
                        ha='right', va='top', color='gray', style='italic')

    # Manual layout: reserve left margin for row labels, bottom for legend
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.12, hspace=0.45, wspace=0.28)
    fig.canvas.draw()
    for row, label in enumerate(row_labels):
        bbox = axes[row][0].get_position()
        fig.text(0.01, (bbox.y0 + bbox.y1) / 2, label,
                 ha='center', va='center', fontsize=9, fontweight='bold', rotation=90)

    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    if handles and not _pc._pub_mode:
        fig.legend(handles, leg_labels, loc='lower center', ncol=3,
                   fontsize=9, bbox_to_anchor=(0.5, 0.0))

    out = args.output
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else '.', exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
