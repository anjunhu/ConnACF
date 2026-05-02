#!/usr/bin/env python3
"""
Plot TOMA-specific topology inference metrics across experiments.

TOMA (Topology-Oriented Membership Attack) infers the underlying U-I interaction
graph by observing recommendation outputs. Key metrics (all under
reverse_engineering.toma in each turn_*.json):

  - ccr               : Cumulative Contamination Rate (fraction of non-attacker users reached)
  - edge_recovery_rate: Fraction of true edges correctly inferred
  - node_coverage     : Fraction of nodes observed at least once
  - topology_similarity: Structural similarity between inferred and true graph
  - false_edge_rate   : Fraction of inferred edges that are wrong

Row 1: Attack effectiveness  — ccr, edge_recovery_rate, node_coverage
Row 2: Inference quality     — topology_similarity, false_edge_rate
Row 3: System performance    — accuracy, ndcg_at_5, recall_at_5
Usage (from connacf/):
    python tools/plot_toma.py \\
        --dirs \\
            attack_output/toma/toma_1cand/ml-100k-100user-medium100/260226093439 \\
            attack_output/toma/toma_1cand/ml-100k-100user-medium100/260226102130 \\
            attack_output/toma/toma_2cand/ml-100k-100user-medium100/260225170950 \\
        --labels "Sparse" "Sparse" "Medium" \\
        --auto_label \\
        --output ../figure/toma_MovieLens100_topology.png
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
    base_parser, group_by_label, get_color_for_label, get_linestyle_for_label,
    sort_labels_for_legend, apply_bg, MARKERS,
    interpolate_nans, safe_max, last_valid, annotate_finals,
    infer_title_suffix, generate_auto_label, load_connacf_data,
    trim_data,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Metric definitions ───────────────────────────────────────────────────────

TOMA_ATTACK_METRICS = {
    'edge_recovery_rate':{'label': 'Recovery Rate'},
    'node_coverage':     {'label': 'Node Coverage'},
}

# CCR (user memory contamination) + LLM-judge item victim rate — shown together, both normalised 0–1
DISSEMINATION_METRICS = {
    'ccr':             {'label': 'User ASR'},
    'item_victim_pct': {'label': 'Item ASR'},
}

TOMA_QUALITY_METRICS = {
    'topology_similarity': {'label': 'Topo Sim F1'},
    'false_edge_rate':     {'label': 'Edge FPR'},
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_toma_experiment(exp_dir: str) -> dict:
    """Load turn files and extract TOMA-specific + performance metrics."""
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

    # Load metadata upfront so attacker_ratio is available during the turn loop
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
    if turn_files:
        try:
            with open(turn_files[0]) as f:
                first = json.load(f)
            nse = first.get('natural_semantic_evolution', {})
            metadata['num_items'] = nse.get('active_items_total')
        except Exception:
            pass

    turns = []
    attack_ts  = {k: [] for k in TOMA_ATTACK_METRICS}
    quality_ts = {k: [] for k in TOMA_QUALITY_METRICS}
    dissem_ts  = {k: [] for k in DISSEMINATION_METRICS}

    for t in range(min_t, max_t + 1):
        turns.append(t)
        if t not in t2f:
            for d in (attack_ts, quality_ts, dissem_ts):
                for k in d: d[k].append(np.nan)
            continue
        try:
            with open(t2f[t]) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f'Error loading {t2f[t]}: {e}')
            for d in (attack_ts, quality_ts, dissem_ts):
                for k in d: d[k].append(np.nan)
            continue

        toma = data.get('reverse_engineering', {}).get('toma', {})
        for k in TOMA_ATTACK_METRICS:
            attack_ts[k].append(_f(toma.get(k)))
        for k in TOMA_QUALITY_METRICS:
            quality_ts[k].append(_f(toma.get(k)))

        # CCR (user memory) + LLM-judge item victim rate — paired for comparison, both 0–1
        llm = data.get('llm_judge', {})
        nse = data.get('natural_semantic_evolution', {})
        ij = llm.get('item', {})
        ic = ij.get('contaminated_count') if isinstance(ij, dict) else None
        ai_non = nse.get('active_items_non_attacker') or (ij.get('num_agents_evaluated') if isinstance(ij, dict) else None)

        # Normalize CCR: raw CCR starts at attacker_ratio (attackers always "contaminated"),
        # so rescale to [0,1] over the non-attacker population.
        raw_ccr = _f(toma.get('ccr'))
        ar = _f(metadata.get('attacker_ratio'))
        if not np.isnan(raw_ccr) and not np.isnan(ar) and ar < 1.0:
            norm_ccr = max(0.0, (raw_ccr - ar) / (1.0 - ar))
        else:
            norm_ccr = raw_ccr  # fallback if attacker_ratio unknown
        dissem_ts['ccr'].append(norm_ccr)

        # Item victim rate: convert from percentage to fraction
        item_vr = (ic / max(1, ai_non)) if ic is not None and ai_non else np.nan
        dissem_ts['item_victim_pct'].append(item_vr)

    return {
        'turns': turns,
        'toma_attack': attack_ts,
        'toma_quality': quality_ts,
        'dissemination': dissem_ts,
        'metadata': metadata,
        'name': 'Experiment',
    }


def _f(v):
    if v is None: return np.nan
    try: return float(v)
    except: return np.nan


def _empty(error=None):
    r = {
        'turns': [],
        'toma_attack':  {k: [] for k in TOMA_ATTACK_METRICS},
        'toma_quality': {k: [] for k in TOMA_QUALITY_METRICS},
        'dissemination': {k: [] for k in DISSEMINATION_METRICS},
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
    if not all_turns:
        return 0
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


def _annotate(ax, label_order, label_groups, label_colors, section, key, fmt='.3f'):
    if _pc._pub_mode:
        return
    y = 0.95
    for lbl in label_order:
        grp = label_groups[lbl]
        color = label_colors[lbl]
        if len(grp) == 1:
            vals = grp[0].get(section, {}).get(key, [])
            v = last_valid(vals) if vals else None
            if v is not None:
                ax.text(0.95, y, f'{lbl}: {v:{fmt}}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= 0.06
        else:
            finals = [last_valid(d.get(section, {}).get(key, [])) for d in grp]
            finals = [v for v in finals if v is not None]
            if finals:
                ax.text(0.95, y,
                        f'{lbl}: {np.mean(finals):{fmt}} [{np.min(finals):{fmt}}-{np.max(finals):{fmt}}]',
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= 0.06


# ── Main plot ────────────────────────────────────────────────────────────────

NROWS, NCOLS = 2, 3  # 6 metrics in a 2×3 grid (same layout as MAMA)

# Flat ordered list: row-major into the 2×3 grid
ALL_TOMA_METRICS = [
    # Row 1
    ('toma_attack',  'edge_recovery_rate', TOMA_ATTACK_METRICS['edge_recovery_rate']),
    ('toma_attack',  'node_coverage',      TOMA_ATTACK_METRICS['node_coverage']),
    ('dissemination','ccr',                DISSEMINATION_METRICS['ccr']),
    # Row 2
    ('toma_quality', 'topology_similarity',TOMA_QUALITY_METRICS['topology_similarity']),
    ('toma_quality', 'false_edge_rate',    TOMA_QUALITY_METRICS['false_edge_rate']),
    ('dissemination','item_victim_pct',    DISSEMINATION_METRICS['item_victim_pct']),
]

# Single-row 4-metric layout (MASTER-style): topology_similarity, false_edge_rate,
# user victim rate (ccr), item victim rate
TOMA_SINGLE_ROW_METRICS = [
    ('toma_quality', 'topology_similarity', TOMA_QUALITY_METRICS['topology_similarity']),
    ('toma_quality', 'false_edge_rate',     TOMA_QUALITY_METRICS['false_edge_rate']),
    ('dissemination','ccr',                 DISSEMINATION_METRICS['ccr']),
    ('dissemination','item_victim_pct',     DISSEMINATION_METRICS['item_victim_pct']),
]


def plot_toma(datasets, output_file, title=None, single_row=False):
    if not datasets:
        logger.error('No datasets to plot')
        return

    metric_list = TOMA_SINGLE_ROW_METRICS if single_row else ALL_TOMA_METRICS
    ncols = len(metric_list) if single_row else NCOLS

    active = [(s, k, m) for s, k, m in metric_list
              if _has_data(datasets, s, k)]
    if not active:
        logger.error('No plottable metrics found')
        return

    if single_row:
        n_active = len(active)
        fig, axes_row = plt.subplots(1, n_active,
                                     figsize=_pc._pub_figsize(n_active * SUBPLOT_W, SUBPLOT_H,
                                                              nrows=1, row_h=SUBPLOT_H),
                                     squeeze=False)
        axes = axes_row  # shape (1, n_active)
        act_rows = 1
        _ncols = n_active
    else:
        n_active = len(active)
        act_rows = (n_active + NCOLS - 1) // NCOLS
        fig, axes = plt.subplots(act_rows, NCOLS,
                                 figsize=_pc._pub_figsize(NCOLS * SUBPLOT_W, act_rows * SUBPLOT_H,
                                                          nrows=act_rows, row_h=SUBPLOT_H),
                                 squeeze=False)
        _ncols = NCOLS

    auto_suffix = infer_title_suffix(datasets)
    if not _pc._pub_mode:
        fig.suptitle(
            title if title else f'TOMA Topology Inference Metrics{auto_suffix}',
            fontsize=14, fontweight='bold'
        )

    label_order, label_groups = group_by_label(datasets)
    label_order = sort_labels_for_legend(label_order)
    label_colors = {lbl: get_color_for_label(lbl, i) for i, lbl in enumerate(label_order)}
    apply_bg(fig, axes, label_order)

    for idx in range(act_rows * _ncols):
        row, col = divmod(idx, _ncols)
        ax = axes[row][col]
        if idx >= len(active):
            ax.set_visible(False)
            continue
        section, mk, meta = active[idx]
        max_val = 0
        for i, lbl in enumerate(label_order):
            color = label_colors[lbl]
            marker = MARKERS[i % len(MARKERS)]
            mv = _plot_group(ax, label_groups[lbl], section, mk, color, marker, lbl)
            max_val = max(max_val, mv)

        if row == act_rows - 1 or act_rows == 1:
            ax.set_xlabel('Turn', fontsize=_pc.AXIS_FONTSIZE)
        if _pc.USE_SUBPLOT_TITLES:
            ax.set_title(meta['label'], fontsize=_pc.TITLE_FONTSIZE, fontweight="bold")
        else:
            if col == 0:
                ax.set_ylabel(meta['label'], fontsize=_pc.AXIS_FONTSIZE)
        ax.grid(True, alpha=0.3)
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.15)
        _annotate(ax, label_order, label_groups, label_colors, section, mk)

    plt.tight_layout()
    if single_row:
        plt.subplots_adjust(wspace=0.08)
    else:
        plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.08)
    _pub_savefig(output_file)
    logger.info(f'Saved: {output_file}')
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot TOMA topology inference metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dirs', type=str, nargs='+', required=True)
    parser.add_argument('--labels', type=str, nargs='+', default=None)
    parser.add_argument('--auto_label', action='store_true', default=False)
    parser.add_argument('--output', '-o', type=str, default='toma_plot.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument(
        '--sub_round', type=str, default='0', choices=['0', '1', 'both'],
        help='0=pre-backward (even turns), 1=post-backward (odd turns, default), both=all turns.',
    )
    parser.add_argument('--pub', action='store_true', default=False,
                        help='Publication preset: 8pt font, figsize scaled by rows, dpi=300.')
    parser.add_argument('--single-row', action='store_true', default=False,
                        help='Single-row 4-metric layout (topology_similarity, false_edge_rate, user/item victim rate).')
    args = parser.parse_args()

    if args.pub:
        import _plot_common as _pc
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

    def _filter_sub_round(turns, sub_round):
        if sub_round == 'both' or not turns:
            return turns, list(range(len(turns)))
        keep_parity = int(sub_round)
        indices = [i for i, t in enumerate(turns) if t == 0 or t % 2 == keep_parity]
        kept = [turns[i] for i in indices]
        return kept, indices

    def trim(data, le, alpha):
        from _plot_common import MAX_DISPLAY_TURNS
        le = le or len(data['turns'])
        raw_turns = data['turns'][:le]
        kept_turns, indices = _filter_sub_round(raw_turns, args.sub_round)
        # Cap at MAX_DISPLAY_TURNS after sub_round filtering
        kept_turns = kept_turns[:MAX_DISPLAY_TURNS]
        indices = indices[:MAX_DISPLAY_TURNS]
        display_turns = list(range(len(kept_turns))) if args.sub_round != 'both' else kept_turns

        def _pick(lst):
            return [smooth(lst, alpha)[i] for i in indices] if lst else []

        out = {'turns': display_turns, 'name': data['name'],
               'metadata': data.get('metadata', {})}
        if 'error' in data: out['error'] = data['error']
        for sec in ('toma_attack', 'toma_quality', 'dissemination'):
            out[sec] = {k: _pick(v[:le]) for k, v in data.get(sec, {}).items()}
        return out

    datasets = []
    for i, d in enumerate(args.dirs):
        logger.info(f'Loading: {d}')
        data = load_toma_experiment(d)
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
    plot_toma(valid, args.output, args.title, single_row=args.single_row)


if __name__ == '__main__':
    main()
