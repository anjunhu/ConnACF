#!/usr/bin/env python3
"""
Compact 1×4 dissemination plot: [UID-User | UID-Item | IM-User | IM-Item]
with shared Y axis across all 4 panels.

Usage:
    python tools/plot_dissemination_compact.py --pub \\
        --title "NetSafe" \\
        --output "../figure/pdf/netsafe_compact.pdf" \\
        --uid_dirs DIR1 DIR2 DIR3 \\
        --uid_labels "1c" "2c" "3c" \\
        --im_dirs  DIR4 DIR5 DIR6 \\
        --im_labels "50i" "100i" "200i"
"""
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from _plot_common import (
    _pub_figsize, _pub_savefig, SUBPLOT_H,
    load_connacf_data, trim_data, group_by_label,
    get_color_for_label, sort_labels_for_legend, apply_bg, MARKERS,
    plot_group, annotate_finals, maybe_legend, generate_auto_label,
)
import _plot_common as _pc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_group(dirs, labels, last_epoch, smoothing, auto_label):
    datasets = []
    for i, d in enumerate(dirs):
        data = load_connacf_data(d)
        data = trim_data(data, last_epoch, smoothing, sub_round=0)
        base = labels[i] if labels and i < len(labels) else str(i)
        data['name'] = generate_auto_label(data, base) if auto_label else base
        datasets.append(data)
    return datasets


def _fill_row(axes_row, uid_datasets, im_datasets, row_title=None):
    """Fill one row of 4 axes with UID-User | UID-Item | IM-User | IM-Item panels."""
    specs = [
        ('user_victim_pct', None, 'User ASR'),
        ('item_victim_pct', None, 'Item ASR'),
    ]
    all_datasets = [uid_datasets, uid_datasets, im_datasets, im_datasets]
    panel_specs  = [specs[0], specs[1], specs[0], specs[1]]
    bg_colors    = ['#EEF4FF', '#EEF4FF', '#FFF0EE', '#FFF0EE']

    for col, (ax, datasets, (metric, nested, ylabel), bg) in enumerate(
            zip(axes_row, all_datasets, panel_specs, bg_colors)):

        ax.set_facecolor(bg)
        label_order, label_groups = group_by_label(datasets)
        label_order = sort_labels_for_legend(label_order)
        label_colors = {lbl: get_color_for_label(lbl, i)
                        for i, lbl in enumerate(label_order)}

        max_val = 0
        for i, label in enumerate(label_order):
            marker = MARKERS[i % len(MARKERS)]
            max_val = max(max_val, plot_group(
                ax, label_groups[label], label, label_colors[label], marker,
                metric, nested_key=nested))

        fs = _pc.AXIS_FONTSIZE
        ax.set_xlabel('Turn', fontsize=fs)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(ylabel, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')
        else:
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 1)
        ax.set_ylim(0, max(100, max_val * 1.1) if max_val > 0 else 100)

    # Row title on the left of the first axis
    if row_title:
        axes_row[0].set_ylabel(row_title, fontsize=_pc.AXIS_FONTSIZE + 1,
                               fontweight='bold', labelpad=8)


def plot_compact(uid_datasets, im_datasets, output_file, title=None):
    fig, axes = plt.subplots(1, 4, figsize=_pub_figsize(14, 2.4, nrows=1, row_h=SUBPLOT_H * 1.5),
                             sharey=True)
    plt.subplots_adjust(wspace=0.05)
    _fill_row(axes, uid_datasets, im_datasets)

    if title and not _pc._pub_mode:
        fig.suptitle(title, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


def plot_multi_attack(attacks, output_file, title=None):
    """
    attacks: list of dicts with keys:
        uid_datasets, im_datasets, row_title
    Produces one row per attack, 4 panels per row.
    """
    n_rows = len(attacks)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=_pub_figsize(14, n_rows * 2.4, nrows=n_rows, row_h=SUBPLOT_H * 1.5),
        sharey=False,
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.45)

    for row, atk in enumerate(attacks):
        _fill_row(axes[row], atk['uid_datasets'], atk['im_datasets'],
                  row_title=atk.get('row_title'))

    if title and not _pc._pub_mode:
        fig.suptitle(title, fontsize=_pc.TITLE_FONTSIZE, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.45)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--title', default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--uid_dirs', nargs='+', default=None)
    p.add_argument('--uid_labels', nargs='+', default=None)
    p.add_argument('--im_dirs', nargs='+', default=None)
    p.add_argument('--im_labels', nargs='+', default=None)
    p.add_argument('--last_epoch', type=int, default=None)
    p.add_argument('--smoothing', type=float, default=0.0)
    p.add_argument('--auto_label', action='store_true')
    p.add_argument('--pub', action='store_true')
    p.add_argument('--split', action='store_true',
                   help='Also save separate _UIDensity and _InterMat outputs alongside --output')
    # Multi-attack stacked layout
    p.add_argument('--attacks', nargs='+', default=None,
                   help='JSON strings, one per attack row. Each must have keys: '
                        'uid_dirs, uid_labels, im_dirs, im_labels, row_title. '
                        'Example: \'{"uid_dirs":["d1"],"uid_labels":["1c"],'
                        '"im_dirs":["d2"],"im_labels":["100i"],'
                        '"row_title":"NetSafe ($\\\\alpha_U{>}0,\\\\alpha_I{>}0$)"}\'')
    args = p.parse_args()

    if args.pub:
        figsize, dpi = _pc.apply_pub_style()
        _pc._pub_figsize_override = figsize
        _pc._pub_dpi_override = dpi

    # ── Multi-attack stacked layout ──────────────────────────────────────
    if args.attacks:
        import json as _json
        attacks = []
        for spec_str in args.attacks:
            spec = _json.loads(spec_str)
            uid_ds = load_group(spec['uid_dirs'], spec.get('uid_labels'), args.last_epoch,
                                args.smoothing, args.auto_label)
            im_ds  = load_group(spec['im_dirs'],  spec.get('im_labels'),  args.last_epoch,
                                args.smoothing, args.auto_label)
            attacks.append({'uid_datasets': uid_ds, 'im_datasets': im_ds,
                            'row_title': spec.get('row_title', '')})
        if not attacks:
            logger.error('No attack specs loaded'); return
        plot_multi_attack(attacks, args.output, args.title)
        return

    # ── Single-attack compact layout ─────────────────────────────────────
    if not args.uid_dirs or not args.im_dirs:
        p.error('--uid_dirs and --im_dirs are required (or use --attacks for multi-attack mode)')

    uid_ds = load_group(args.uid_dirs, args.uid_labels, args.last_epoch, args.smoothing, args.auto_label)
    im_ds  = load_group(args.im_dirs,  args.im_labels,  args.last_epoch, args.smoothing, args.auto_label)

    if not uid_ds or not im_ds:
        logger.error('No datasets loaded'); return

    plot_compact(uid_ds, im_ds, args.output, args.title)

    if args.split:
        from pathlib import Path
        from plot_dissemination import plot_dissemination
        stem = Path(args.output)
        uid_out = stem.with_name(stem.stem + '_UIDensity' + stem.suffix)
        im_out  = stem.with_name(stem.stem + '_InterMat'  + stem.suffix)
        plot_dissemination(uid_ds, uid_out, args.title)
        plot_dissemination(im_ds,  im_out,  args.title)


if __name__ == '__main__':
    main()
