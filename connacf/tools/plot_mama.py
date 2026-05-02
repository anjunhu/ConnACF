#!/usr/bin/env python3
"""
Dedicated MAMA privacy plotter — 2-row layout (no reverse engineering row).

  Row 1: Privacy metrics  — pii_leakage_rate, pii_exposure_count, category rates
  Row 2: System performance — accuracy, NDCG@5, Recall@5

Usage (from connacf/):
    python tools/plot_mama.py \\
        --dirs \\
            attack_output/mama/mama_1cand/ml-100k-100user-medium100/260225113804 \\
            attack_output/mama/mama_2cand/ml-100k-100user-medium100/260220125121 \\
            attack_output/mama/mama_3cand/ml-100k-100user-medium100/260226115345 \\
        --labels "1c" "2c" "3c" \\
        --output ../figure/mama_UIDensity.png
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent))
from _plot_common import _pub_figsize, _pub_savefig, apply_pub_style, SUBPLOT_W, SUBPLOT_H
import _plot_common as _pc
from _plot_common import (
    group_by_label, get_color_for_label, get_linestyle_for_label,
    sort_labels_for_legend, apply_bg, MARKERS,
    interpolate_nans, safe_max, last_valid,
    infer_title_suffix, generate_auto_label,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Metric definitions ───────────────────────────────────────────────────────

PRIVACY_METRICS = {
    'pii_leakage_rate':   {'label': 'PII Leakage Rate',   'ylim': (0, 1.05)},
    'pii_exposure_count': {'label': 'PII Exposure Count',  'ylim': None},
    'category_identity':  {'label': 'Identity Leak Rate',  'ylim': (0, 1.05)},
    'category_contact':   {'label': 'Contact Leak Rate',   'ylim': (0, 1.05)},
    'category_location':  {'label': 'Location Leak Rate',  'ylim': (0, 1.05)},
    'category_temporal':  {'label': 'Temporal Leak Rate',   'ylim': (0, 1.05)},
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _f(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan


def load_mama_experiment(exp_dir: str) -> dict:
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

    # Metadata
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
        nse = first.get('natural_semantic_evolution', {})
        metadata['num_items'] = nse.get('active_items_total')
    except Exception:
        pass

    turns = []
    priv_ts = {k: [] for k in PRIVACY_METRICS}

    for t in range(min_t, max_t + 1):
        turns.append(t)
        if t not in t2f:
            for k in priv_ts: priv_ts[k].append(np.nan)
            continue
        try:
            with open(t2f[t]) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f'Error loading {t2f[t]}: {e}')
            for k in priv_ts: priv_ts[k].append(np.nan)
            continue

        priv = data.get('privacy', {})
        cat = priv.get('category_rates', {})
        priv_ts['pii_leakage_rate'].append(_f(priv.get('pii_leakage_rate')))
        priv_ts['pii_exposure_count'].append(_f(priv.get('pii_exposure_count')))
        priv_ts['category_identity'].append(_f(cat.get('identity')))
        priv_ts['category_contact'].append(_f(cat.get('contact')))
        priv_ts['category_location'].append(_f(cat.get('location')))
        priv_ts['category_temporal'].append(_f(cat.get('temporal')))

    return {
        'turns': turns,
        'privacy': priv_ts,
        'metadata': metadata,
        'name': 'Experiment',
    }


def _empty(error=None):
    r = {
        'turns': [],
        'privacy': {k: [] for k in PRIVACY_METRICS},
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


def _plot_line(ax, turns, values, color, marker, label):
    from _plot_common import LINE_W, LINE_A
    ls = get_linestyle_for_label(label)
    t, v, mask = interpolate_nans(turns, values)
    mk = _pc.get_marker_for_label(label); every = max(1, len(t) // 8); ax.plot(t, v, linewidth=LINE_W, color=color, label=label, alpha=LINE_A, linestyle=ls, marker=mk, markevery=every, markersize=9, markeredgewidth=1.2, markerfacecolor=(*color[:3], 0.55) if len(color)==3 else color)


def _plot_band(ax, group_datasets, section, key, color, marker, label):
    from _plot_common import LINE_W, LINE_A
    ls = get_linestyle_for_label(label)
    all_turns = sorted({t for d in group_datasets for t in d['turns']})
    if not all_turns: return 0
    t2i = {t: i for i, t in enumerate(all_turns)}
    turns = np.array(all_turns, dtype=float)
    mat = np.full((len(group_datasets), len(turns)), np.nan)
    for di, d in enumerate(group_datasets):
        vals = d.get(section, {}).get(key, [])
        if not vals: continue
        _, iv, _ = interpolate_nans(d['turns'], vals)
        for tv, vv in zip(d['turns'], iv):
            if tv in t2i: mat[di, t2i[tv]] = vv
    with np.errstate(all='ignore'):
        mean_v = np.nanmean(mat, axis=0)
        min_v  = np.nanmin(mat, axis=0)
        max_v  = np.nanmax(mat, axis=0)
    valid = ~np.isnan(mean_v)
    if not valid.any(): return 0
    ax.fill_between(turns, min_v, max_v, color=color, alpha=0.2,
                    label=f'{label} (n={len(group_datasets)})')
    mk = _pc.get_marker_for_label(label); every = max(1, int(valid.sum()) // 8); ax.plot(turns, mean_v, linewidth=LINE_W, color=color, alpha=LINE_A, linestyle=ls, marker=mk, markevery=every, markersize=9, markeredgewidth=1.2, markerfacecolor=(*color[:3], 0.55) if len(color)==3 else color)
    return float(np.nanmax(max_v))


def _plot_group(ax, group, section, key, color, marker, label):
    if len(group) == 1:
        d = group[0]
        vals = d.get(section, {}).get(key, [])
        if d['turns'] and vals:
            _plot_line(ax, d['turns'], vals, color, marker, label)
            return safe_max(vals)
        return 0
    return _plot_band(ax, group, section, key, color, marker, label)


def _best_annotation_corner(ax, datasets, section, key):
    """Pick the axes corner (x, y, ha, va) with the least line density."""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1.0

    # Collect all final values (last 20% of turns) normalised to [0,1]
    tail_vals = []
    for d in datasets:
        vals = d.get(section, {}).get(key, [])
        if not vals:
            continue
        n = max(1, len(vals) // 5)
        tail = [v for v in vals[-n:] if not np.isnan(v)]
        tail_vals.extend([(v - ylim[0]) / y_range for v in tail])

    if not tail_vals:
        return 0.97, 0.97, 'right', 'top'

    mean_tail = np.mean(tail_vals)

    # Also check early values for top-left / bottom-left corners
    head_vals = []
    for d in datasets:
        vals = d.get(section, {}).get(key, [])
        if not vals:
            continue
        n = max(1, len(vals) // 5)
        head = [v for v in vals[:n] if not np.isnan(v)]
        head_vals.extend([(v - ylim[0]) / y_range for v in head])

    mean_head = np.mean(head_vals) if head_vals else mean_tail

    # Score each corner by how far the data is from it (higher = less crowded)
    corners = {
        'top-right':    (1 - mean_tail) + (1 - mean_tail),
        'top-left':     (1 - mean_head) + (1 - mean_head),
        'bottom-right': mean_tail + mean_tail,
        'bottom-left':  mean_head + mean_head,
    }
    best = max(corners, key=corners.get)
    return {
        'top-right':    (0.97, 0.97, 'right', 'top'),
        'top-left':     (0.03, 0.97, 'left',  'top'),
        'bottom-right': (0.97, 0.15, 'right', 'bottom'),
        'bottom-left':  (0.03, 0.15, 'left',  'bottom'),
    }[best]


def _annotate(ax, label_order, label_groups, label_colors, section, key, fmt='.3f'):
    if _pc._pub_mode:
        return
    all_datasets = [d for lbl in label_order for d in label_groups[lbl]]
    ax_x, ax_y, ha, va = _best_annotation_corner(ax, all_datasets, section, key)
    step = 0.06 if va == 'top' else -0.06

    y = ax_y
    for lbl in label_order:
        grp = label_groups[lbl]
        color = label_colors[lbl]
        if len(grp) == 1:
            vals = grp[0].get(section, {}).get(key, [])
            v = last_valid(vals) if vals else None
            if v is not None:
                ax.text(ax_x, y, f'{lbl}: {v:{fmt}}',
                        transform=ax.transAxes, ha=ha, va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= step
        else:
            finals = [last_valid(d.get(section, {}).get(key, [])) for d in grp]
            finals = [v for v in finals if v is not None]
            if finals:
                ax.text(ax_x, y,
                        f'{lbl}: {np.mean(finals):{fmt}} [{np.min(finals):{fmt}}-{np.max(finals):{fmt}}]',
                        transform=ax.transAxes, ha=ha, va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= step


# ── Main plot ────────────────────────────────────────────────────────────────

NROWS, NCOLS = 2, 3  # 6 privacy metrics in a 2×3 grid

def plot_mama(datasets, output_file, title=None, lower_row_only=False):
    ncols = 3 if lower_row_only else NCOLS
    if not datasets:
        logger.error('No datasets to plot')
        return

    _lower_keys = {'category_contact', 'category_location', 'category_temporal'}
    active_priv = {k: v for k, v in PRIVACY_METRICS.items()
                   if _has_data(datasets, 'privacy', k)
                   and (not lower_row_only or k in _lower_keys)}
    if not active_priv:
        logger.error('No plottable privacy metrics found')
        return

    metric_keys = list(active_priv.keys())
    act_rows = (len(metric_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(act_rows, ncols,
                             figsize=_pub_figsize(ncols * SUBPLOT_W, act_rows * SUBPLOT_H,
                                                  nrows=act_rows, row_h=SUBPLOT_H),
                             squeeze=False)

    auto_suffix = infer_title_suffix(datasets)
    if not _pc._pub_mode:
        fig.suptitle(
            title if title else f'MAMA Privacy Leakage Metrics{auto_suffix}',
            fontsize=14, fontweight='bold'
        )

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}
    apply_bg(fig, axes, label_order)

    for idx in range(act_rows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if idx >= len(metric_keys):
            ax.set_visible(False)
            continue
        mk = metric_keys[idx]
        meta = active_priv[mk]
        max_val = 0
        for i, lbl in enumerate(label_order):
            color = label_colors[lbl]
            marker = MARKERS[i % len(MARKERS)]
            mv = _plot_group(ax, label_groups[lbl], 'privacy', mk, color, marker, lbl)
            max_val = max(max_val, mv)

        ax.set_xlabel('Turn', fontsize=11)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(meta['label'], fontsize=22, fontweight="bold")
        else:
            if col == 0:
                ax.set_ylabel(meta['label'], fontsize=11)
        ax.grid(True, alpha=0.3)
        if col > 0:
            ax.tick_params(labelleft=False)
        if meta.get('ylim'):
            ax.set_ylim(*meta['ylim'])
        elif max_val > 0:
            ax.set_ylim(0, max_val * 1.15)
        fmt = '.0f' if mk == 'pii_exposure_count' else '.3f'
        _annotate(ax, label_order, label_groups, label_colors, 'privacy', mk, fmt=fmt)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(hspace=0.45, wspace=0.08)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot MAMA privacy leakage metrics (no reverse engineering row)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dirs', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--auto_label', action='store_true', default=False)
    parser.add_argument('--output', '-o', type=str, default='mama_plot.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--pub', action='store_true', default=False,
                        help='Publication preset: 8pt font, figsize=(7,2.8), dpi=300.')
    parser.add_argument('--lower-row-only', action='store_true', default=False,
                        help='Plot only the lower row (category leak rates), matching CORBA figsize.')
    parser.add_argument(
        '--sub_round', type=str, default='0', choices=['0', '1', 'both'],
        help='0=pre-backward (even turns), 1=post-backward (odd turns, default), both=all turns.',
    )
    args = parser.parse_args()

    if args.pub:
        import _plot_common as _pc
        _pc._pub_figsize_override, _pc._pub_dpi_override = apply_pub_style(figsize=(10.5, 4.4))

    if args.lower_row_only:
        import _plot_common as _pc
        _pc._pub_figsize_override = (10.5, 2.8)
        if args.pub:
            _pc._pub_dpi_override = 300

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

    def _filter_sub_round(turns, sub_round):
        if sub_round == 'both' or not turns:
            return turns, list(range(len(turns)))
        keep_parity = int(sub_round)
        indices = [i for i, t in enumerate(turns) if t == 0 or t % 2 == keep_parity]
        kept = [turns[i] for i in indices]
        return kept, indices

    def trim(data, le, alpha):
        le = le or len(data['turns'])
        raw_turns = data['turns'][:le]
        kept_turns, indices = _filter_sub_round(raw_turns, args.sub_round)
        display_turns = list(range(len(kept_turns))) if args.sub_round != 'both' else kept_turns

        def _pick(lst):
            return [smooth(lst, alpha)[i] for i in indices] if lst else []

        out = {'turns': display_turns, 'name': data['name'],
               'metadata': data.get('metadata', {})}
        if 'error' in data: out['error'] = data['error']
        for sec in ('privacy',):
            out[sec] = {k: _pick(v[:le]) for k, v in data.get(sec, {}).items()}
        return out

    datasets = []
    for i, d in enumerate(args.dirs):
        logger.info(f'Loading: {d}')
        data = load_mama_experiment(d)
        data = trim(data, args.last_epoch, args.smoothing)

        base = args.labels[i] if args.labels and i < len(args.labels) \
               else Path(d).parent.name + '/' + Path(d).name
        data['name'] = base  # metadata goes in title suffix via infer_title_suffix, not per-label

        logger.info(f'  {len(data["turns"])} turns, label: {data["name"]}')
        datasets.append(data)

    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error('No valid datasets')
        return
    plot_mama(valid, args.output, args.title, lower_row_only=args.lower_row_only)

    # Summary
    print('\n' + '=' * 60)
    print('MAMA PRIVACY SUMMARY')
    print('=' * 60)
    for d in valid:
        print(f'\n{d["name"]} ({len(d["turns"])} turns):')
        priv = d.get('privacy', {})
        plr = last_valid(priv.get('pii_leakage_rate', []))
        pec = last_valid(priv.get('pii_exposure_count', []))
        if plr is not None: print(f'  PII Leakage Rate:   {plr:.4f}')
        if pec is not None: print(f'  PII Exposure Count: {pec:.0f}')
        for ck in ('category_identity', 'category_contact', 'category_location', 'category_temporal'):
            cv = last_valid(priv.get(ck, []))
            if cv is not None:
                print(f'  {PRIVACY_METRICS[ck]["label"]:22s} {cv:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
