#!/usr/bin/env python3
"""
Defense comparison plotter.

For each guarded run, plots the unguarded baseline as a solid red-ish shaded
region and the defended run as a green hatched region, making the reduction
immediately visible.

Metric: user_victim_pct and item_victim_pct from the LLM judge (common to
all attack types: NetSafe, TOMA, MASTER, etc.).

Usage (from connacf/):
    python tools/plot_defense.py \
        --pairs \
            "attack_output/toma/toma_2cand/.../260307181255 : attack_output/toma/toma_2cand_tguard/.../260310163459" \
            "attack_output/netsafe/misinfo_repro_2cand/.../task_0 : attack_output/netsafe/misinfo_2cand_g_safeguard/.../260309235256" \
        --pair_labels "TOMA 2c — T-Guard" "NetSafe 2c — G-Safeguard" \
        --output ../figure/defense_comparison.png
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
from _plot_common import _pub_figsize, _pub_savefig, apply_pub_style, apply_bg
import _plot_common as _pc
from _plot_common import (
    load_connacf_data, interpolate_nans, safe_max, trim_data,
    get_color_for_label,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Colours
RED   = '#b72f14'   # Wire1c
GREEN = '#1860a8'   # Wire3c
RED_LIGHT   = '#c94a30'
GREEN_LIGHT = '#3a7fc1'


def _load_group(dirs, last_epoch, smoothing):
    """Load one or more experiment dirs and return list of trimmed datasets."""
    datasets = []
    for d in dirs:
        data = load_connacf_data(d.strip())
        if data['turns']:
            data = trim_data(data, last_epoch, smoothing)
            datasets.append(data)
        else:
            logger.warning(f'No turn data in {d}')
    return datasets


def _aggregate(datasets, key):
    """Align multiple datasets on a common turn axis and return (turns, mean, lo, hi)."""
    if not datasets:
        return np.array([]), np.array([]), np.array([]), np.array([])
    all_turns = sorted({t for d in datasets for t in d['turns']})
    t2i = {t: i for i, t in enumerate(all_turns)}
    turns = np.array(all_turns, dtype=float)
    mat = np.full((len(datasets), len(turns)), np.nan)
    for di, d in enumerate(datasets):
        vals = d.get(key, [])
        if not vals:
            continue
        _, iv, _ = interpolate_nans(d['turns'], vals)
        for tv, vv in zip(d['turns'], iv):
            if tv in t2i:
                mat[di, t2i[tv]] = vv
    with np.errstate(all='ignore'):
        mean_v = np.nanmean(mat, axis=0)
        lo = np.nanmin(mat, axis=0)
        hi = np.nanmax(mat, axis=0)
    return turns, mean_v, lo, hi


def _plot_pair(ax, unguarded, guarded, metric_key, ylabel, col=0, title_fontsize=_pc.TITLE_FONTSIZE, show_xlabel=True):
    """Plot one metric: unguarded and guarded as mean lines + seed range shading."""
    from _plot_common import LINE_W, LINE_A
    ut, um, ulo, uhi = _aggregate(unguarded, metric_key)
    gt, gm, glo, ghi = _aggregate(guarded, metric_key)

    max_val = 0.0

    if ut.size:
        n_ug = len(unguarded)
        lbl_ug = f'Unguarded (n={n_ug})' if n_ug > 1 else 'Unguarded'
        ax.fill_between(ut, ulo, uhi, color=RED, alpha=0.15, label=lbl_ug)
        ax.plot(ut, um, color=RED, linewidth=LINE_W, alpha=LINE_A)
        max_val = max(max_val, float(np.nanmax(uhi)))

    if gt.size:
        n_g = len(guarded)
        lbl_g = f'Guarded (n={n_g})' if n_g > 1 else 'Guarded'
        ax.fill_between(gt, glo, ghi, color=GREEN, alpha=0.15, label=lbl_g)
        ax.plot(gt, gm, color=GREEN, linewidth=LINE_W, alpha=LINE_A, linestyle='--')
        max_val = max(max_val, float(np.nanmax(ghi)))

    if _pc.USE_SUBPLOT_TITLES:
        ax.set_title(ylabel, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
    elif col == 0:
        ax.set_ylabel(ylabel, fontsize=_pc.AXIS_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
    ax.grid(True, alpha=0.3)
    if max_val > 0:
        ax.set_ylim(0, max(100, max_val * 1.15))

    y_ann = 0.92
    for vals, color, name in [(um, RED, 'Unguarded'), (gm, GREEN, 'Guarded')]:
        if vals is not None and len(vals):
            last = next((v for v in reversed(vals) if not np.isnan(v)), None)
            if last is not None:
                ax.text(0.97, y_ann, f'{name}: {last:.1f}%',
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_ann -= 0.07




def _compute_reduction(unguarded, guarded, metric_key):
    """Return (turns, mean_diff, lo_diff, hi_diff) where diff = unguarded − guarded."""
    ut, um, _, _ = _aggregate(unguarded, metric_key)
    gt, gm, _, _ = _aggregate(guarded, metric_key)
    if not ut.size or not gt.size:
        return np.array([]), np.array([]), np.array([]), np.array([])
    common = sorted(set(ut.tolist()) & set(gt.tolist()))
    if not common:
        return np.array([]), np.array([]), np.array([]), np.array([])
    u_idx = {t: i for i, t in enumerate(ut.tolist())}
    g_idx = {t: i for i, t in enumerate(gt.tolist())}
    turns = np.array(common, dtype=float)
    u_vals = np.array([um[u_idx[t]] for t in common])
    g_vals = np.array([gm[g_idx[t]] for t in common])
    diff = u_vals - g_vals
    # Envelope from all pairwise diffs across individual runs
    all_u, all_g = [], []
    for d in unguarded:
        vals = d.get(metric_key, [])
        if vals:
            _, iv, _ = interpolate_nans(d['turns'], vals)
            t2v = dict(zip(d['turns'], iv))
            all_u.append([t2v.get(t, np.nan) for t in common])
    for d in guarded:
        vals = d.get(metric_key, [])
        if vals:
            _, iv, _ = interpolate_nans(d['turns'], vals)
            t2v = dict(zip(d['turns'], iv))
            all_g.append([t2v.get(t, np.nan) for t in common])
    if all_u and all_g:
        u_arr = np.array(all_u)
        g_arr = np.array(all_g)
        pw = u_arr[:, np.newaxis, :] - g_arr[np.newaxis, :, :]
        pw2d = pw.reshape(-1, len(common))
        with np.errstate(all='ignore'):
            lo = np.nanmin(pw2d, axis=0)
            hi = np.nanmax(pw2d, axis=0)
    else:
        lo = hi = diff
    return turns, diff, lo, hi


def plot_defense(pairs, pair_labels, output_file, title, last_epoch, smoothing, ncols=2):
    """
    ncols=2: classic layout — each pair is a row, User col 0, Item col 1.
    ncols=4: wide layout — each pair is a column-pair (User|Item), pairs side by side.
             Rows: 0=Unguarded/Guarded, 1=Gap, 2=Reduction.
             (Matches the 4-col 2-row trim style used in the paper.)
    """
    n_pairs = len(pairs)
    
    # ── Load all pair data ────────────────────────────────────────────────
    pair_data = []
    for pair_str, plabel in zip(pairs, pair_labels):
        parts = pair_str.split(':')
        if len(parts) < 2:
            logger.error(f'Invalid pair: {pair_str}')
            pair_data.append(([], [], plabel))
            continue
        ug_dirs = [p.strip() for p in parts[0].split(',') if p.strip()]
        g_dirs  = [p.strip() for p in parts[1].split(',') if p.strip()]
        unguarded = _load_group(ug_dirs, last_epoch, smoothing)
        guarded   = _load_group(g_dirs, last_epoch, smoothing)
        pair_data.append((unguarded, guarded, plabel))

    if ncols == 4:
        # 4-col layout: cols = [pair0_user, pair0_item, pair1_user, pair1_item, ...]
        # Rows: 0=Unguarded/Guarded, 1=Gap, 2=Reduction
        n_grid_cols = n_pairs * 2
        n_grid_rows = 3
        fig, axes = plt.subplots(
            n_grid_rows, n_grid_cols,
            figsize=_pub_figsize(7 * n_grid_cols, n_grid_rows * 2.2,
                                 nrows=n_grid_rows, row_h=2.2),
            squeeze=False,
        )
        if not _pc._pub_mode:
            fig.suptitle(title or 'Defense Comparison', fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')

        metric_labels = [('user_victim_pct', 'User ASR'),
                         ('item_victim_pct', 'Item ASR')]

        for pi, (ug, g, plabel) in enumerate(pair_data):
            for mi, (metric_key, mlabel) in enumerate(metric_labels):
                gc = pi * 2 + mi  # grid column

                # Row 0: Unguarded vs Guarded
                ax = axes[0][gc]
                _plot_pair(ax, ug, g, metric_key, f'{plabel} {mlabel}',
                           col=gc, title_fontsize=_pc.TITLE_FONTSIZE, show_xlabel=True)

                # Row 1: Gap band (mean only)
                ax = axes[1][gc]
                ut, um, _, _ = _aggregate(ug, metric_key)
                gt, gm, _, _ = _aggregate(g, metric_key)
                max_val = 0.0
                if ut.size and gt.size:
                    common = sorted(set(ut.tolist()) & set(gt.tolist()))
                    if common:
                        u_idx = {t: i for i, t in enumerate(ut.tolist())}
                        g_idx = {t: i for i, t in enumerate(gt.tolist())}
                        turns = np.array(common, dtype=float)
                        u_vals = np.array([um[u_idx[t]] for t in common])
                        g_vals = np.array([gm[g_idx[t]] for t in common])
                        color = get_color_for_label(plabel, pi)
                        from _plot_common import LINE_W, LINE_A
                        ax.fill_between(turns, np.minimum(u_vals, g_vals),
                                        np.maximum(u_vals, g_vals),
                                        color=color, alpha=0.20)
                        ax.plot(turns, u_vals, color=color, linewidth=LINE_W,
                                alpha=LINE_A, linestyle='-')
                        ax.plot(turns, g_vals, color=color, linewidth=LINE_W,
                                alpha=LINE_A, linestyle='--')
                        max_val = float(np.nanmax(np.maximum(u_vals, g_vals)))
                ax.set_title(f'ASR Gap ({mlabel})', fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
                if max_val > 0:
                    ax.set_ylim(0, max(100, max_val * 1.15))

                # Row 2: Reduction (mean only)
                ax = axes[2][gc]
                turns, diff, _, _ = _compute_reduction(ug, g, metric_key)
                max_abs = 0.0
                if turns.size:
                    color = get_color_for_label(plabel, pi)
                    from _plot_common import LINE_W, LINE_A
                    ax.plot(turns, diff, color=color, linewidth=LINE_W, alpha=LINE_A)
                    max_abs = float(np.nanmax(np.abs(diff)))
                ax.axhline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.5)
                ax.set_title(f'ASR Reduction ({mlabel})', fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
                if max_abs > 0:
                    ax.set_ylim(-max_abs * 0.6, max_abs * 1.25)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.35)
        _pub_savefig(output_file)
        logger.info(f'Saved: {output_file}')
        plt.close()
        return

    # ── Classic 2-col layout ──────────────────────────────────────────────
    n_rows = n_pairs + 2
    from _plot_common import SUBPLOT_W, SUBPLOT_H
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=_pub_figsize(2 * SUBPLOT_W, n_rows * SUBPLOT_H, nrows=n_rows, row_h=SUBPLOT_H),
                             squeeze=False)

    if not _pc._pub_mode:
        fig.suptitle(title or 'Defense Comparison: Unguarded vs Guarded',
                     fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
    apply_bg(fig, axes, pair_labels)

    for row, (ug, g, plabel) in enumerate(pair_data):
        if not ug and not g:
            logger.warning(f'No data for pair: {plabel}')
            continue
        ax_user = axes[row][0]
        ax_item = axes[row][1]
        _plot_pair(ax_user, ug, g, 'user_victim_pct', 'ASR',
                   col=0, title_fontsize=_pc.TITLE_FONTSIZE, show_xlabel=True)
        _plot_pair(ax_item, ug, g, 'item_victim_pct', 'ASR',
                   col=1, title_fontsize=_pc.TITLE_FONTSIZE, show_xlabel=True)
        ax_user.set_title(f'{plabel} User ASR', fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
        ax_item.set_title(f'{plabel} Item ASR', fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')

    # Gap row
    for col, (metric_key, ylabel) in enumerate([
        ('user_victim_pct', 'User ASR'),
        ('item_victim_pct', 'Item ASR'),
    ]):
        ax = axes[n_pairs][col]
        max_val = 0.0
        for pi, (ug, g, plabel) in enumerate(pair_data):
            if not ug or not g:
                continue
            color = get_color_for_label(plabel, pi)
            ut, um, _, _ = _aggregate(ug, metric_key)
            gt, gm, _, _ = _aggregate(g, metric_key)
            if not ut.size or not gt.size:
                continue
            common = sorted(set(ut.tolist()) & set(gt.tolist()))
            if not common:
                continue
            u_idx = {t: i for i, t in enumerate(ut.tolist())}
            g_idx = {t: i for i, t in enumerate(gt.tolist())}
            turns = np.array(common, dtype=float)
            u_vals = np.array([um[u_idx[t]] for t in common])
            g_vals = np.array([gm[g_idx[t]] for t in common])
            bottom = np.minimum(u_vals, g_vals)
            top    = np.maximum(u_vals, g_vals)
            ax.fill_between(turns, bottom, top, color=color, alpha=0.20, label=plabel)
            from _plot_common import LINE_W, LINE_A
            ax.plot(turns, u_vals, color=color, linewidth=LINE_W, alpha=LINE_A, linestyle='-')
            ax.plot(turns, g_vals, color=color, linewidth=LINE_W, alpha=LINE_A, linestyle='--')
            with np.errstate(all='ignore'):
                max_val = max(max_val, float(np.nanmax(top)))
            last_u = next((v for v in reversed(u_vals) if not np.isnan(v)), None)
            last_g = next((v for v in reversed(g_vals) if not np.isnan(v)), None)
            if last_u is not None and last_g is not None and not _pc._pub_mode:
                gap = last_u - last_g
                mid = (last_u + last_g) / 2
                ax.text(turns[-1], mid, f' Δ{gap:+.1f}pp', fontsize=7, color=color, va='center')
        ax.set_title(f'User ASR Gap' if col == 0 else 'Item ASR Gap',
                     fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
        if max_val > 0:
            ax.set_ylim(0, max(100, max_val * 1.15))

    # Reduction row
    for col, (metric_key, ylabel) in enumerate([
        ('user_victim_pct', 'User ASR Reduction'),
        ('item_victim_pct', 'Item ASR Reduction'),
    ]):
        ax = axes[n_pairs + 1][col]
        max_abs = 0.0
        for pi, (ug, g, plabel) in enumerate(pair_data):
            if not ug or not g:
                continue
            color = get_color_for_label(plabel, pi)
            turns, diff, lo, hi = _compute_reduction(ug, g, metric_key)
            if not turns.size:
                continue
            from _plot_common import LINE_W, LINE_A
            ax.plot(turns, diff, color=color, linewidth=LINE_W, alpha=LINE_A)
            with np.errstate(all='ignore'):
                max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
            last_d = next((v for v in reversed(diff) if not np.isnan(v)), None)
            if last_d is not None and not _pc._pub_mode:
                ax.text(turns[-1], last_d, f' {last_d:+.1f}pp', fontsize=7, color=color, va='bottom')
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.5)
        ax.set_title(f'User Safety Gain' if col == 0 else 'Item Safety Gain',
                     fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
        if max_abs > 0:
            ax.set_ylim(min(-max_abs * 0.6, -15), max_abs * 1.25)

    plt.tight_layout(h_pad=2.5)
    plt.subplots_adjust(top=0.93)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot defense comparison (unguarded vs guarded)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--pairs', type=str, nargs='+', required=True,
        help='Each pair is "unguarded_dir[,dir2,...] : guarded_dir[,dir2,...]"',
    )
    parser.add_argument(
        '--pair_labels', type=str, nargs='+', required=True,
        help='Label for each pair row',
    )
    parser.add_argument('--output', '-o', type=str, default='defense_comparison.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--ncols', type=int, default=2, choices=[2, 4],
                        help='2=classic row-per-pair, 4=wide col-per-pair layout')
    parser.add_argument('--pub', action='store_true', default=False,
                        help='Publication preset: 8pt font, figsize scaled by rows, dpi=300.')
    parser.add_argument('--split-figures', action='store_true', default=False,
                        help='Save one full-width figure per pair instead of a combined multi-row figure.')
    args = parser.parse_args()

    if args.pub:
        import _plot_common as _pc
        from _plot_common import SUBPLOT_W, SUBPLOT_H
        _pc._pub_figsize_override, _pc._pub_dpi_override = apply_pub_style()
        if args.ncols == 2:
            _pc._pub_figsize_override = (2 * 10.5 / 4 * 1.56, SUBPLOT_H)  # 2 cols at TOMA per-subplot width
        import matplotlib
        matplotlib.rcParams['axes.titlesize'] = 26
        _pc.TITLE_FONTSIZE = 26

    if len(args.pairs) != len(args.pair_labels):
        logger.error('--pairs and --pair_labels must have the same length')
        return

    if args.split_figures:
        from pathlib import Path as _Path
        stem = _Path(args.output)
        for i, (pair, label) in enumerate(zip(args.pairs, args.pair_labels)):
            safe_label = label.replace(' ', '_').replace('/', '-').replace('—', '-')
            out = stem.with_name(f"{stem.stem}_{i:02d}_{safe_label}{stem.suffix}")
            plot_defense([pair], [label], str(out),
                         args.title, args.last_epoch, args.smoothing, ncols=args.ncols)
    else:
        plot_defense(args.pairs, args.pair_labels, args.output,
                     args.title, args.last_epoch, args.smoothing, ncols=args.ncols)


if __name__ == '__main__':
    main()
