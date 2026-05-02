#!/usr/bin/env python3
"""
Plot comparison of leakage-driven attack metrics across experiments.

Unlike plot_comparison.py (which focuses on dissemination/contamination victim rates),
this script is designed for leakage-driven attacks (MASLeak, MAMA, etc.) and plots:

  Row 1: Privacy metrics (PII leakage rate, exposure count, category rates)
  Row 2: Reverse engineering metrics (extract_rate, system prompt similarity, etc.)
  Row 3: System performance impact (accuracy, NDCG@5) — stealth/utility tradeoff

Usage:
    # Single run
    python tools/plot_leakage.py \
        --dirs attack_output/masleak/masleak_2cand/ml-100k-20-user-dense/260216104607 \
        --labels "MASLeak 2cand"

    # Compare multiple runs
    python tools/plot_leakage.py \
        --dirs attack_output/mama/mama_medium_5cand/ml-100k-20-user-dense/260216111655 \
               attack_output/masleak/masleak_2cand/ml-100k-20-user-dense/260216104607 \
        --labels "MAMA" "MASLeak" \
        --output ../figure/leakage_comparison.png
"""

import os
import sys
import json
import argparse
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import logging

# Allow importing _plot_common from the same tools/ directory
sys.path.insert(0, str(Path(__file__).parent))
import _plot_common as _pc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Metric definitions ──────────────────────────────────────────────────────

# Privacy metrics (Row 1)
PRIVACY_METRICS = {
    'pii_leakage_rate':  {'label': 'PII Leakage Rate',   'ylim': (0, 1.05)},
    'pii_exposure_count': {'label': 'PII Exposure Count', 'ylim': None},
    'category_identity':  {'label': 'Identity Leak Rate', 'ylim': (0, 1.05)},
    'category_contact':   {'label': 'Contact Leak Rate',  'ylim': (0, 1.05)},
    'category_location':  {'label': 'Location Leak Rate', 'ylim': (0, 1.05)},
    'category_temporal':  {'label': 'Temporal Leak Rate', 'ylim': (0, 1.05)},
}

# Reverse engineering metrics (Row 2)
# Prioritise dynamic U-I topology metrics over static similarity scores
RE_METRICS = {
    'ui_topology_recall':    {'label': 'U-I Topology Recall',        'ylim': (0, 1.05)},
    'ui_topology_precision': {'label': 'U-I Topology Precision',     'ylim': (0, 1.05)},
    'ui_topology_f1':        {'label': 'U-I Topology F1',            'ylim': (0, 1.05)},
    'extract_rate':          {'label': 'Overall Extraction Rate',    'ylim': (0, 1.05)},
    'ss_system_prompt':      {'label': 'System Prompt Similarity',   'ylim': (0, 1.05)},
    'ss_task_instructions':  {'label': 'Task Instr. Similarity',     'ylim': (0, 1.05)},
}

# System performance metrics (Row 3)
PERF_METRICS = {
    'accuracy':   {'label': 'Accuracy',  'ylim': (0, 1.05)},
    'ndcg_at_5':  {'label': 'NDCG@5',    'ylim': (0, 1.05)},
    'recall_at_5': {'label': 'Recall@5', 'ylim': (0, 1.05)},
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_experiment(exp_dir: str) -> dict:
    """Load all turn files from an experiment directory and extract leakage metrics.

    Returns dict with:
        turns: list[int]
        privacy.*: list[float]          (per-turn)
        reverse_engineering.*: list[float]
        performance.*: list[float]
        metadata: dict
        name: str (placeholder)
    """
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        logger.warning(f"Directory does not exist: {exp_dir}")
        return _empty_result(error=f'Directory not found: {exp_dir}')

    turn_files = sorted(exp_path.glob("turn_*.json"),
                        key=lambda x: int(x.stem.split('_')[1]))
    if not turn_files:
        logger.warning(f"No turn_*.json files in: {exp_dir}")
        return _empty_result(error=f'No turn files in: {exp_dir}')

    turn_numbers = [int(f.stem.split('_')[1]) for f in turn_files]
    min_turn, max_turn = min(turn_numbers), max(turn_numbers)
    turn_to_file = {int(f.stem.split('_')[1]): f for f in turn_files}

    # Initialise time-series containers
    turns = []
    privacy_ts = {k: [] for k in PRIVACY_METRICS}
    re_ts = {k: [] for k in RE_METRICS}
    perf_ts = {k: [] for k in PERF_METRICS}

    for t in range(min_turn, max_turn + 1):
        turns.append(t)
        if t not in turn_to_file:
            for d in (privacy_ts, re_ts, perf_ts):
                for k in d:
                    d[k].append(np.nan)
            continue

        try:
            with open(turn_to_file[t]) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading {turn_to_file[t]}: {e}")
            for d in (privacy_ts, re_ts, perf_ts):
                for k in d:
                    d[k].append(np.nan)
            continue

        # ── Privacy ──
        priv = data.get('privacy', {})
        cat_rates = priv.get('category_rates', {})
        privacy_ts['pii_leakage_rate'].append(_float(priv.get('pii_leakage_rate')))
        privacy_ts['pii_exposure_count'].append(_float(priv.get('pii_exposure_count')))
        privacy_ts['category_identity'].append(_float(cat_rates.get('identity')))
        privacy_ts['category_contact'].append(_float(cat_rates.get('contact')))
        privacy_ts['category_location'].append(_float(cat_rates.get('location')))
        privacy_ts['category_temporal'].append(_float(cat_rates.get('temporal')))

        # ── Reverse engineering ──
        re = data.get('reverse_engineering', {})
        for k in RE_METRICS:
            re_ts[k].append(_float(re.get(k)))

        # ── System performance ──
        sp = data.get('system_performance', {})
        for k in PERF_METRICS:
            perf_ts[k].append(_float(sp.get(k)))

    # ── Metadata ──
    metadata = {}
    config_path = exp_path / 'experiment_config.json'
    if not config_path.exists():
        config_path = exp_path.parent / 'experiment_config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            ac = cfg.get('attack_config', {})
            metadata['num_candidates'] = ac.get('num_candidates')
            metadata['attacker_ratio'] = ac.get('attacker_ratio')
            metadata['attack_type'] = ac.get('attack_type')
        except Exception:
            pass

    return {
        'turns': turns,
        'privacy': privacy_ts,
        'reverse_engineering': re_ts,
        'performance': perf_ts,
        'metadata': metadata,
        'name': 'Experiment',
    }


def _float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _empty_result(error=None):
    r = {
        'turns': [],
        'privacy': {k: [] for k in PRIVACY_METRICS},
        'reverse_engineering': {k: [] for k in RE_METRICS},
        'performance': {k: [] for k in PERF_METRICS},
        'metadata': {},
        'name': 'Experiment',
    }
    if error:
        r['error'] = error
    return r


# ── Plotting ─────────────────────────────────────────────────────────────────

def _has_any_data(datasets, section_key, metric_key):
    """Return True if at least one dataset has non-NaN data for this metric."""
    for d in datasets:
        vals = d.get(section_key, {}).get(metric_key, [])
        if any(not np.isnan(v) for v in vals):
            return True
    return False


def _get_color(label: str, idx: int) -> tuple:
    """Deterministic colour per label, with slight variation by index."""
    ll = label.lower()
    if 'mama' in ll:
        base_hue, base_sat, base_light = 0 / 360, 0.75, 0.50
    elif 'masleak' in ll or 'mas' in ll:
        base_hue, base_sat, base_light = 210 / 360, 0.70, 0.45
    elif 'toma' in ll:
        base_hue, base_sat, base_light = 120 / 360, 0.65, 0.40
    elif 'master' in ll:
        base_hue, base_sat, base_light = 280 / 360, 0.65, 0.45
    elif 'sparse' in ll:
        base_hue, base_sat, base_light = 200 / 360, 0.70, 0.45
    elif 'medium' in ll:
        base_hue, base_sat, base_light = 40 / 360, 0.75, 0.45
    elif 'dense' in ll:
        base_hue, base_sat, base_light = 0 / 360, 0.75, 0.65
    else:
        base_hue = (idx * 0.15) % 1.0
        base_sat, base_light = 0.55, 0.45

    hue = (base_hue + (idx * 0.03) % 0.1 - 0.05) % 1.0
    sat = max(0.3, min(1.0, base_sat + (idx * 0.05) % 0.15 - 0.075))
    light = max(0.25, min(0.65, base_light + (idx * 0.04) % 0.12 - 0.06))
    return colorsys.hls_to_rgb(hue, light, sat)


MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


def _interpolate_nans(turns, values):
    turns = np.array(turns, dtype=float)
    values = np.array(values, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return turns, values, mask
    interp = np.interp(turns, turns[mask], values[mask])
    first, last = np.where(mask)[0][0], np.where(mask)[0][-1]
    interp[:first] = np.nan
    interp[last + 1:] = np.nan
    return turns, interp, mask


def _last_valid(values):
    for v in reversed(values):
        if not np.isnan(v):
            return v
    return None


def _safe_max(values, default=0):
    valid = [v for v in values if not np.isnan(v)]
    return max(valid) if valid else default


def _plot_line(ax, turns, values, color, marker, label):
    """Plot a single time-series line (no markers)."""
    t, v, mask = _interpolate_nans(turns, values)
    ax.plot(t, v, linewidth=2.2, color=color, label=label, alpha=0.75)


def _plot_band(ax, group_datasets, section_key, metric_key, color, marker, label):
    """Plot mean + min/max band for grouped datasets."""
    all_turns = set()
    for d in group_datasets:
        if d['turns']:
            all_turns.update(d['turns'])
    if not all_turns:
        return 0

    turns = np.array(sorted(all_turns), dtype=float)
    n = len(turns)
    t2i = {t: i for i, t in enumerate(turns)}

    mat = np.full((len(group_datasets), n), np.nan)
    for di, d in enumerate(group_datasets):
        vals = d.get(section_key, {}).get(metric_key, [])
        if not d['turns'] or not vals:
            continue
        _, iv, _ = _interpolate_nans(d['turns'], vals)
        for t_val, v_val in zip(d['turns'], iv):
            if t_val in t2i:
                mat[di, t2i[t_val]] = v_val

    with np.errstate(all='ignore'):
        mean_v = np.nanmean(mat, axis=0)
        min_v = np.nanmin(mat, axis=0)
        max_v = np.nanmax(mat, axis=0)

    valid = ~np.isnan(mean_v)
    if not valid.any():
        return 0

    ax.fill_between(turns, min_v, max_v, color=color, alpha=0.2,
                    label=f'{label} (n={len(group_datasets)})')
    ax.plot(turns, mean_v, linewidth=2.2, color=color, alpha=0.75)
    return float(np.nanmax(max_v))


def _plot_group(ax, group, section_key, metric_key, color, marker, label):
    """Dispatch to single-line or band depending on group size."""
    if len(group) == 1:
        d = group[0]
        vals = d.get(section_key, {}).get(metric_key, [])
        if d['turns'] and vals:
            _plot_line(ax, d['turns'], vals, color, marker, label)
            return _safe_max(vals)
        return 0
    return _plot_band(ax, group, section_key, metric_key, color, marker, label)


def _annotate(ax, label_order, label_groups, label_colors, section_key, metric_key, fmt='.3f'):
    """Add final-value annotations in the top-right corner."""
    y_off = 0.95
    for label in label_order:
        group = label_groups[label]
        color = label_colors[label]
        if len(group) == 1:
            vals = group[0].get(section_key, {}).get(metric_key, [])
            fv = _last_valid(vals) if vals else None
            if fv is not None:
                ax.text(0.95, y_off, f"{label}: {fv:{fmt}}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_off -= 0.06
        else:
            finals = []
            for d in group:
                vals = d.get(section_key, {}).get(metric_key, [])
                fv = _last_valid(vals) if vals else None
                if fv is not None:
                    finals.append(fv)
            if finals:
                ax.text(0.95, y_off,
                        f"{label}: {np.mean(finals):{fmt}} [{np.min(finals):{fmt}}-{np.max(finals):{fmt}}]",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_off -= 0.06


def plot_leakage_comparison(datasets: list, output_file: str, title: str = None):
    """Main plotting function.

    Dynamically builds rows based on which metric sections have data:
      - Row for privacy metrics (if any non-NaN)
      - Row for reverse-engineering metrics (if any non-NaN)
      - Row for system performance (always shown)
    """
    if not datasets:
        logger.error("No datasets to plot")
        return None

    # ── Determine which metric groups have data ──
    rows = []  # list of (section_key, metrics_dict, row_title)

    # Privacy row
    active_priv = {k: v for k, v in PRIVACY_METRICS.items()
                   if _has_any_data(datasets, 'privacy', k)}
    if active_priv:
        rows.append(('privacy', active_priv, 'Privacy / PII Leakage'))

    # Reverse engineering row
    active_re = {k: v for k, v in RE_METRICS.items()
                 if _has_any_data(datasets, 'reverse_engineering', k)}
    if active_re:
        rows.append(('reverse_engineering', active_re, 'Reverse Engineering'))

    # Performance row
    active_perf = {k: v for k, v in PERF_METRICS.items()
                   if _has_any_data(datasets, 'performance', k)}
    if active_perf:
        rows.append(('performance', active_perf, 'System Performance (Utility)'))

    if not rows:
        logger.error("No plottable metrics found in any dataset")
        return None

    n_rows = len(rows)
    n_cols = max(len(m) for _, m, _ in rows)

    fig, all_axes = plt.subplots(n_rows, n_cols,
                                 figsize=(min(6 * n_cols, 36), 4.5 * n_rows),
                                 squeeze=False)

    fig.suptitle(title or 'Leakage Attack Metrics Comparison',
                 fontsize=14, fontweight='bold')

    # ── Group datasets by label ──
    label_groups = defaultdict(list)
    label_order = []
    for d in datasets:
        lbl = d['name']
        if lbl not in label_groups:
            label_order.append(lbl)
        label_groups[lbl].append(d)

    label_colors = {lbl: _get_color(lbl, i) for i, lbl in enumerate(label_order)}

    # ── Plot each row ──
    for row_idx, (section_key, metrics, row_title) in enumerate(rows):
        metric_keys = list(metrics.keys())
        for col_idx in range(n_cols):
            ax = all_axes[row_idx][col_idx]
            if col_idx >= len(metric_keys):
                ax.set_visible(False)
                continue

            mk = metric_keys[col_idx]
            meta = metrics[mk]
            max_val = 0

            for i, lbl in enumerate(label_order):
                grp = label_groups[lbl]
                color = label_colors[lbl]
                marker = MARKERS[i % len(MARKERS)]
                mv = _plot_group(ax, grp, section_key, mk, color, marker, lbl)
                max_val = max(max_val, mv)

            ax.set_xlabel('Turn', fontsize=10)
            ax.set_ylabel(meta['label'], fontsize=10)
            ax.set_title(meta['label'], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if meta.get('ylim'):
                ax.set_ylim(*meta['ylim'])
            elif max_val > 0:
                ax.set_ylim(0, max_val * 1.15)

            fmt = '.0f' if mk == 'pii_exposure_count' else '.3f'
            _annotate(ax, label_order, label_groups, label_colors,
                      section_key, mk, fmt=fmt)

        # Row label on the left-most visible axis
        all_axes[row_idx][0].set_ylabel(
            f'{row_title}\n{metrics[metric_keys[0]]["label"]}', fontsize=10)

    # ── Legend ──
    handles, labels_leg = all_axes[0][0].get_legend_handles_labels()
    if handles and not _pc._pub_mode:
        fig.legend(handles, labels_leg, loc='lower center',
                   ncol=min(len(label_order), 6), fontsize=9,
                   bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, top=0.93, hspace=0.35)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Leakage comparison plot saved to: {output_file}")
    plt.close()
    return output_file


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot leakage-driven attack metrics comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single MASLeak run
  python tools/plot_leakage.py \\
      --dirs attack_output/masleak/masleak_2cand/ml-100k-20-user-dense/260216104607 \\
      --labels "MASLeak"

  # Compare MAMA vs MASLeak
  python tools/plot_leakage.py \\
      --dirs attack_output/mama/mama_medium_5cand/ml-100k-20-user-dense/260216111655 \\
             attack_output/masleak/masleak_2cand/ml-100k-20-user-dense/260216104607 \\
      --labels "MAMA" "MASLeak" \\
      --output ../figure/leakage_mama_vs_masleak.png
        """,
    )
    parser.add_argument('--dirs', type=str, nargs='+', required=True,
                        help='Experiment output directories (each containing turn_*.json)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Custom labels (must match number of --dirs)')
    parser.add_argument('--output', '-o', type=str, default='leakage_plot.png',
                        help='Output file path')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--last_epoch', type=int, default=None,
                        help='Trim data to this many turns')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='EMA smoothing factor (0-1). 0=none.')
    parser.add_argument('--auto_label', action='store_true', default=False,
                        help='Append inferred metadata (Ni, Nc, X%%atk) to bare labels')

    args = parser.parse_args()

    # ── Smoothing helper ──
    def smooth(values, alpha):
        if alpha <= 0 or not values:
            return values
        alpha = min(alpha, 0.99)
        out, prev = [], None
        for v in values:
            if np.isnan(v):
                out.append(np.nan)
            elif prev is None:
                out.append(v); prev = v
            else:
                s = alpha * prev + (1 - alpha) * v
                out.append(s); prev = s
        return out

    def trim(data, last_epoch, alpha):
        if not data['turns'] or last_epoch is None:
            le = len(data['turns'])
        else:
            le = last_epoch
        trimmed = {
            'turns': data['turns'][:le],
            'name': data['name'],
            'metadata': data.get('metadata', {}),
        }
        if 'error' in data:
            trimmed['error'] = data['error']
        for section in ('privacy', 'reverse_engineering', 'performance'):
            sec = data.get(section, {})
            trimmed[section] = {}
            for k, v in sec.items():
                tv = v[:le]
                if alpha > 0:
                    tv = smooth(tv, alpha)
                trimmed[section][k] = tv
        return trimmed

    # ── Load ──
    datasets = []
    errors = []
    for i, d in enumerate(args.dirs):
        logger.info(f"Loading: {d}")
        data = load_experiment(d)
        data = trim(data, args.last_epoch, args.smoothing)

        if args.labels and i < len(args.labels):
            data['name'] = args.labels[i]
        else:
            data['name'] = Path(d).parent.name + '/' + Path(d).name

        if args.auto_label:
            try:
                from _plot_common import generate_auto_label
                # Build a metadata dict compatible with generate_auto_label
                meta = data.get('metadata', {})
                compat = {
                    'metadata': {
                        'num_items': meta.get('num_candidates'),   # not available in leakage loader
                        'num_candidates': meta.get('num_candidates'),
                        'attacker_ratio': meta.get('attacker_ratio'),
                    }
                }
                # Try to get num_items from first turn file
                exp_path = Path(d)
                turn_files = sorted(exp_path.glob("turn_*.json"),
                                    key=lambda x: int(x.stem.split('_')[1]))
                if turn_files:
                    with open(turn_files[0]) as tf:
                        td = json.load(tf)
                    nse = td.get('natural_semantic_evolution', {})
                    compat['metadata']['num_items'] = nse.get('active_items_total')
                data['name'] = generate_auto_label(compat, data['name'])
            except Exception as e:
                logger.warning(f"auto_label failed: {e}")

        if 'error' in data:
            errors.append(f"  {data['name']}: {data['error']}")
        logger.info(f"  {len(data['turns'])} turns, label: {data['name']}")
        datasets.append(data)

    if errors:
        print("\n⚠️  WARNINGS:")
        for e in errors:
            print(e)
        print()

    valid = [d for d in datasets if d['turns']]
    if not valid:
        logger.error("No valid datasets to plot")
        return

    plot_leakage_comparison(valid, args.output, args.title)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("LEAKAGE ATTACK COMPARISON SUMMARY")
    print("=" * 70)

    for d in valid:
        print(f"\n{d['name']} ({len(d['turns'])} turns):")
        if 'error' in d:
            print(f"  ⚠️  {d['error']}")
            continue

        # Privacy
        priv = d.get('privacy', {})
        plr = _last_valid(priv.get('pii_leakage_rate', []))
        pec = _last_valid(priv.get('pii_exposure_count', []))
        if plr is not None:
            print(f"  PII Leakage Rate:    {plr:.4f}")
        if pec is not None:
            print(f"  PII Exposure Count:  {pec:.0f}")
        cats = ['category_identity', 'category_contact', 'category_location', 'category_temporal']
        for ck in cats:
            cv = _last_valid(priv.get(ck, []))
            if cv is not None:
                print(f"  {PRIVACY_METRICS[ck]['label']:22s} {cv:.4f}")

        # Reverse engineering
        re = d.get('reverse_engineering', {})
        re_any = False
        for k in RE_METRICS:
            rv = _last_valid(re.get(k, []))
            if rv is not None:
                if not re_any:
                    print(f"  Reverse Engineering:")
                    re_any = True
                print(f"    {RE_METRICS[k]['label']:28s} {rv:.4f}")

        # Performance
        perf = d.get('performance', {})
        perf_any = False
        for k in PERF_METRICS:
            pv = _last_valid(perf.get(k, []))
            if pv is not None:
                if not perf_any:
                    print(f"  System Performance:")
                    perf_any = True
                print(f"    {PERF_METRICS[k]['label']:28s} {pv:.4f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
