#!/usr/bin/env python3
"""
MASTER plotter — 3-col × 2-row layout.

  Row 1: User Mean TIVS, Item Mean TIVS, ASR
  Row 2: COOR, SS System Prompt, F1 Agent Count

No individual subplot legends. Colour = candidate count (warm→cool: 1c→3c).
Linestyle = interaction-matrix density (solid=dense, dashed=medium, dotted=sparse).

Usage (from connacf/):
    python tools/plot_master.py --pub \\
        --title "MASTER: Inference-Time Density" \\
        --output ../figure/pdf/master_UIDensity.pdf \\
        --auto_label \\
        --dirs \\
            attack_output/master/master_1cand/ml-100k-100user-medium100/260311113801 \\
            attack_output/master/master_2cand/ml-100k-100user-medium100/260311091426 \\
            attack_output/master/master_3cand/ml-100k-100user-medium100/260311113752 \\
        --labels "1c" "2c" "3c" --last_epoch 100
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
from _plot_common import (
    _pub_figsize, _pub_savefig, apply_pub_style,
    group_by_label, get_color_for_label, get_linestyle_for_label,
    sort_labels_for_legend, apply_bg, MARKERS, SUBPLOT_W, SUBPLOT_H,
    interpolate_nans, safe_max, last_valid,
    infer_title_suffix, generate_auto_label, plot_group,
)
import _plot_common as _pc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

NCOLS, NROWS = 3, 2

# (section, metric_key, label, ylim)
PLOT_GRID = [
    [
        ('master_attack',    'asr',              'ASR',                     (0, 1.05)),
        ('master_attack',    'coor',             'COOR',                    (0, 1.05)),
        ('tics',             'tivs_user_mean',   'User TIVS',               None),
    ],
    [
        ('master_extraction','ss_system_prompt', 'Sys Prompt CosSim',         (0, 1.05)),
        ('master_extraction','f1_agent_count',   'Topo Sim F1',          (0, 1.05)),
        ('tics',             'tivs_item_mean',   'Item TIVS',               None),
    ],
]

# Flat order for 1×4 pub single-row layout: extraction LHS, dissemination RHS
# (consistent with TOMA: extraction cols 0-1, dissemination cols 2-3)
PLOT_GRID_FLAT = [
    ('master_extraction','ss_system_prompt', 'Sys Prompt CosSim',  (0, 1.05)),
    ('master_extraction','f1_agent_count',   'Topo Sim F1',   (0, 1.05)),
    ('master_attack',    'asr',              'ASR',              (0, 1.05)),
    ('master_attack',    'coor',             'COOR',             (0, 1.05)),
]

ALL_TICS_KEYS       = ['tivs_user_mean', 'tivs_item_mean']
ALL_ATTACK_KEYS     = ['asr', 'coor']
ALL_EXTRACTION_KEYS = ['ss_system_prompt', 'f1_agent_count']


# ── Data loading ─────────────────────────────────────────────────────────────

def _f(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan


def load_master_experiment(exp_dir: str) -> dict:
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        return _empty(f'Directory not found: {exp_dir}')

    turn_files = sorted(exp_path.glob('turn_*.json'),
                        key=lambda x: int(x.stem.split('_')[1]))
    if not turn_files:
        return _empty(f'No turn files in: {exp_dir}')

    turn_numbers = [int(f.stem.split('_')[1]) for f in turn_files]
    min_t, max_t = min(turn_numbers), max(turn_numbers)
    t2f = {int(f.stem.split('_')[1]): f for f in turn_files}

    metadata = {'num_items': None, 'num_candidates': None, 'attacker_ratio': None}
    cfg = exp_path / 'experiment_config.json'
    if not cfg.exists():
        cfg = exp_path.parent / 'experiment_config.json'
    if cfg.exists():
        try:
            with open(cfg) as f:
                c = json.load(f)
            ac = c.get('attack_config', {})
            metadata['num_candidates'] = ac.get('num_candidates')
            metadata['attacker_ratio'] = ac.get('attacker_ratio')
        except Exception:
            pass
    try:
        with open(turn_files[0]) as f:
            first = json.load(f)
        metadata['num_items'] = first.get('natural_semantic_evolution', {}).get('active_items_total')
    except Exception:
        pass

    turns = []
    tics_ts   = {k: [] for k in ALL_TICS_KEYS}
    attack_ts = {k: [] for k in ALL_ATTACK_KEYS}
    extrac_ts = {k: [] for k in ALL_EXTRACTION_KEYS}

    for t in range(min_t, max_t + 1):
        turns.append(t - min_t)
        if t not in t2f:
            for d in (tics_ts, attack_ts, extrac_ts):
                for k in d: d[k].append(np.nan)
            continue
        try:
            with open(t2f[t]) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f'Error loading {t2f[t]}: {e}')
            for d in (tics_ts, attack_ts, extrac_ts):
                for k in d: d[k].append(np.nan)
            continue

        pd_u = data.get('pattern_detection', {}).get('user', {})
        pd_i = data.get('pattern_detection', {}).get('item', {})
        tics_ts['tivs_user_mean'].append(_f(pd_u.get('mean_tivs')))
        tics_ts['tivs_item_mean'].append(_f(pd_i.get('mean_tivs')))

        master = data.get('reverse_engineering', {}).get('master', {})
        for k in ALL_ATTACK_KEYS:
            v = _f(master.get(k))
            if k == 'coor' and not np.isnan(v):
                v = v / 100.0
            attack_ts[k].append(v)
        for k in ALL_EXTRACTION_KEYS:
            extrac_ts[k].append(_f(master.get(k)))

    return {
        'turns': turns,
        'tics': tics_ts,
        'master_attack': attack_ts,
        'master_extraction': extrac_ts,
        'metadata': metadata,
        'name': 'Experiment',
    }


def _empty(error=None):
    r = {
        'turns': [],
        'tics':              {k: [] for k in ALL_TICS_KEYS},
        'master_attack':     {k: [] for k in ALL_ATTACK_KEYS},
        'master_extraction': {k: [] for k in ALL_EXTRACTION_KEYS},
        'metadata': {},
        'name': 'Experiment',
    }
    if error: r['error'] = error
    return r


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _has_data(datasets, section, key):
    return any(
        any(not np.isnan(v) for v in d.get(section, {}).get(key, []))
        for d in datasets
    )


def _drop_outliers(values, window=5):
    """Replace outliers (IQR) then apply rolling-median smoothing."""
    arr = [v for v in values if v is not None and not np.isnan(v)]
    if len(arr) < 4:
        return values
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    cleaned = [np.nan if (v is not None and not np.isnan(v) and (v < lo or v > hi)) else v
               for v in values]
    # Rolling median over a small window to suppress remaining spikes
    half = window // 2
    result = []
    for i, v in enumerate(cleaned):
        if v is None or np.isnan(v):
            result.append(v)
        else:
            neighbours = [cleaned[j] for j in range(max(0, i - half), min(len(cleaned), i + half + 1))
                          if cleaned[j] is not None and not np.isnan(cleaned[j])]
            result.append(float(np.median(neighbours)) if neighbours else v)
    return result


def _plot_series(ax, d, section, key, color, marker, label, drop_outliers=False):
    from _plot_common import LINE_W, LINE_A
    vals = d.get(section, {}).get(key, [])
    if drop_outliers:
        vals = _drop_outliers(vals)
    if not d['turns'] or not vals:
        return 0
    ls = get_linestyle_for_label(label)
    t, v, mask = interpolate_nans(d['turns'], vals)
    mk = _pc.get_marker_for_label(label); every = max(1, len(t) // 8); ax.plot(t, v, linewidth=LINE_W, color=color, label=label, alpha=LINE_A, linestyle=ls, marker=mk, markevery=every, markersize=9, markeredgewidth=1.2, markerfacecolor=(*color[:3], 0.55) if len(color)==3 else color)
    return safe_max(vals)


# ── Main plot ────────────────────────────────────────────────────────────────

def plot_master(datasets, output_file, title=None, full=False):
    if not datasets:
        logger.error('No datasets to plot')
        return

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}

    if _pc._pub_mode and not full:
        # 1×4 single-row layout; all panels share y-scale (0–1), drop y-axis on cols 1–3
        n = len(PLOT_GRID_FLAT)
        fig, axes_flat = plt.subplots(
            1, n,
            figsize=_pc._pub_figsize(n * SUBPLOT_W, SUBPLOT_H, nrows=1, row_h=SUBPLOT_H),
            squeeze=True,
            sharey=True,
        )
        apply_bg(fig, [axes_flat], label_order)

        for col_idx, (section, mk, ylabel, ylim) in enumerate(PLOT_GRID_FLAT):
            ax = axes_flat[col_idx]
            if not _has_data(datasets, section, mk):
                ax.set_visible(False)
                continue
            max_val = 0
            for i, lbl in enumerate(label_order):
                color = label_colors[lbl]
                marker = MARKERS[i % len(MARKERS)]
                mv = plot_group(ax, label_groups[lbl], lbl, color, marker, section, mk)
                max_val = max(max_val, mv)
            ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
            if _pc.USE_SUBPLOT_TITLES:
                ax.set_title(ylabel, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
            else:
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=_pc.AXIS_FONTSIZE)
            ax.grid(True, alpha=0.3)
            if col_idx > 0:
                ax.tick_params(labelleft=False)
            if ylim:
                ax.set_ylim(*ylim)
            elif max_val > 0:
                ax.set_ylim(0, max_val * 1.15)

        plt.tight_layout()
        handles = [plt.Line2D([0], [0], color=label_colors[lbl], linewidth=2, label=lbl)
                   for lbl in label_order]
        if not _pc._pub_mode:
            fig.legend(handles=handles, loc='lower center', ncol=len(label_order),
                       fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
            plt.subplots_adjust(bottom=0.12, wspace=0.08)
        else:
            plt.subplots_adjust(wspace=0.08)
        _pub_savefig(output_file)
        logger.info(f'Saved: {output_file}')
        plt.close()
        return

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=_pc._pub_figsize(NCOLS * SUBPLOT_W, NROWS * SUBPLOT_H, nrows=NROWS, row_h=SUBPLOT_H),
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.35, hspace=0.45)
    apply_bg(fig, axes, label_order)

    if not _pc._pub_mode:
        auto_suffix = infer_title_suffix(datasets)
        fig.suptitle(title or f'MASTER Attack Metrics{auto_suffix}',
                     fontsize=14, fontweight='bold')

    for row_idx, row_specs in enumerate(PLOT_GRID):
        for col_idx, (section, mk, ylabel, ylim) in enumerate(row_specs):
            ax = axes[row_idx][col_idx]
            if not _has_data(datasets, section, mk):
                ax.set_visible(False)
                continue

            max_val = 0
            for i, lbl in enumerate(label_order):
                color = label_colors[lbl]
                marker = MARKERS[i % len(MARKERS)]
                mv = plot_group(ax, label_groups[lbl], lbl, color, marker, section, mk)
                max_val = max(max_val, mv)

            if row_idx == NROWS - 1:
                ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
            if _pc.USE_SUBPLOT_TITLES:
                ax.set_title(ylabel, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
            else:
                if col_idx == 0:
                    ax.set_ylabel(ylabel, fontsize=_pc.AXIS_FONTSIZE)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            elif max_val > 0:
                ax.set_ylim(0, max_val * 1.15)

    plt.tight_layout()
    # Figure-level legend below the subplots
    handles = [plt.Line2D([0], [0], color=label_colors[lbl], linewidth=2, label=lbl)
               for lbl in label_order]
    if not _pc._pub_mode:
        fig.legend(handles=handles, loc='lower center', ncol=len(label_order),
                   fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
    plt.subplots_adjust(bottom=0.12 if not _pc._pub_mode else 0.04)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot MASTER attack metrics (1×4 compact layout)')
    parser.add_argument('--dirs', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--auto_label', action='store_true', default=False)
    parser.add_argument('--output', '-o', type=str, default='master_plot.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--sub_round', type=str, default='0', choices=['0', '1', 'both'])
    parser.add_argument('--pub', action='store_true', default=False)
    parser.add_argument('--full', action='store_true', default=False,
                        help='Full 2-row layout even in pub mode.')
    args = parser.parse_args()

    if args.pub:
        _pc._pub_figsize_override, _pc._pub_dpi_override = apply_pub_style()

    def smooth(values, alpha):
        if alpha <= 0 or not values: return values
        alpha = min(alpha, 0.99)
        out, prev = [], None
        for v in values:
            if np.isnan(v): out.append(np.nan)
            elif prev is None: out.append(v); prev = v
            else:
                s = alpha * prev + (1 - alpha) * v
                out.append(s); prev = s
        return out

    def trim(data, le, alpha):
        from _plot_common import MAX_DISPLAY_TURNS
        le = le or len(data['turns'])
        raw = data['turns'][:le]
        if args.sub_round != 'both':
            p = int(args.sub_round)
            indices = [i for i, t in enumerate(raw) if t == 0 or t % 2 == p]
        else:
            indices = list(range(len(raw)))
        indices = indices[:MAX_DISPLAY_TURNS]
        display = list(range(len(indices))) if args.sub_round != 'both' else [raw[i] for i in indices]

        def _pick(lst):
            return [smooth(lst[:le], alpha)[i] for i in indices] if lst else []

        def _pick_coor(lst):
            # light smoothing for COOR: outlier removal + mild EMA
            cleaned = _drop_outliers(lst[:le]) if lst else lst
            # forward-fill leading NaNs produced by outlier removal
            if cleaned:
                first_valid_val = next((v for v in cleaned if v is not None and not np.isnan(v)), None)
                if first_valid_val is not None:
                    filled = []
                    seen_valid = False
                    for v in cleaned:
                        if not seen_valid and (v is None or np.isnan(v)):
                            filled.append(first_valid_val)
                        else:
                            seen_valid = True
                            filled.append(v)
                    cleaned = filled
            return [smooth(cleaned, max(alpha, 0.2))[i] for i in indices] if cleaned else []

        out = {'turns': display, 'name': data['name'], 'metadata': data.get('metadata', {})}
        if 'error' in data: out['error'] = data['error']
        for sec in ('tics', 'master_attack', 'master_extraction'):
            out[sec] = {k: (_pick_coor(v) if k == 'coor' else _pick(v))
                        for k, v in data.get(sec, {}).items()}
        return out

    datasets = []
    for i, d in enumerate(args.dirs):
        logger.info(f'Loading: {d}')
        data = load_master_experiment(d)
        data = trim(data, args.last_epoch, args.smoothing)
        base = args.labels[i] if args.labels and i < len(args.labels) \
               else Path(d).parent.name + '/' + Path(d).name
        data['name'] = generate_auto_label(data, base) if args.auto_label else base
        logger.info(f'  {len(data["turns"])} turns, label: {data["name"]}')
        datasets.append(data)

    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error('No valid datasets')
        return
    dropped = [d for d in datasets if not d['turns']]
    for d in dropped:
        logger.warning(f"Dataset '{d['name']}' has no turns — check dir exists and contains turn_*.json. Error: {d.get('error', 'unknown')}")
    plot_master(valid, args.output, args.title, full=args.full)

    print('\n' + '=' * 60)
    print('MASTER SUMMARY')
    print('=' * 60)
    for d in valid:
        print(f'\n{d["name"]} ({len(d["turns"])} turns):')
        for sec, keys in [('master_attack', ALL_ATTACK_KEYS),
                          ('master_extraction', ALL_EXTRACTION_KEYS),
                          ('tics', ALL_TICS_KEYS)]:
            for k in keys:
                v = last_valid(d.get(sec, {}).get(k, []))
                if v is not None:
                    print(f'  {k:28s} {v:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
