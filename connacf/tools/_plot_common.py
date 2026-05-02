#!/usr/bin/env python3
"""
Shared data-loading and label utilities for all plot_* scripts.
"""

import json
import colorsys
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import logging
import os
import matplotlib
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='paper', palette='deep')
    # Thin spines globally
    matplotlib.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#bbbbbb',
    })
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication-mode helpers (configured via CLI args, not env vars)
# ---------------------------------------------------------------------------

def apply_pub_style(fontsize=8, legend_fontsize=7, figsize=None, dpi=300):
    """Apply publication-ready rcParams. Call after parse_args when --pub is set."""
    global _pub_mode
    _pub_mode = True
    try:
        import seaborn as sns
        sns.set_theme(style='whitegrid', context='paper', font_scale=fontsize / 10, palette='deep')
    except ImportError:
        pass
    matplotlib.rcParams.update({
        'font.size': fontsize,
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'legend.fontsize': legend_fontsize,
        'figure.titlesize': fontsize + 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'legend.loc': 'best',
    })
    return figsize or (10.5, 2.2), dpi


_pub_figsize_override = None
_pub_dpi_override = None
_pub_mode = False          # set True when --pub is active; suppresses annotations/long labels
USE_SUBPLOT_TITLES = True  # True: label via ax.set_title; False: label via ax.set_ylabel (col 0 only)
MAX_DISPLAY_TURNS = 200    # cap applied after sub_round filtering


def _pub_figsize(default_w, default_h, nrows=None, row_h=None):
    """Return figsize, scaling height by nrows when pub override is active.

    When --pub is set, width is fixed to the override width (default 7 in).
    Height is computed as:
      - If nrows and row_h are given: nrows * row_h  (per-row height in inches)
      - Otherwise: scale the override height by (default_h / 2.8) so callers
        that pass a larger default_h still get proportionally taller figures.
    Without --pub, returns (default_w, default_h) unchanged.
    """
    if _pub_figsize_override:
        pub_w, pub_h_base = _pub_figsize_override
        if nrows is not None and row_h is not None:
            return (pub_w, nrows * row_h)
        # Scale proportionally: treat 2.5 as the 1-row baseline
        scale = default_h / 2.5
        return (pub_w, pub_h_base * scale)
    return (default_w, default_h)


def _pub_savefig(path, **kwargs):
    if _pub_dpi_override:
        kwargs['dpi'] = _pub_dpi_override
    # Use bbox_inches='tight' only for PNG; for PDF enforce exact figsize so
    # side-by-side \includegraphics[width=0.5\columnwidth]{...} panels are
    # the same physical height/width.
    p = str(path)
    if p.endswith('.pdf'):
        plt.savefig(path, bbox_inches=None, **kwargs)
    else:
        plt.savefig(path, bbox_inches='tight', **kwargs)




# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_nested_value(data, path):
    """Extract nested value from dict using dot notation path."""
    keys = path.split('.')
    value = data
    try:
        for key in keys:
            if value is None:
                return None
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    except Exception:
        return None


def interpolate_nans(turns, values):
    turns = np.array(turns)
    values = np.array(values)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return turns, values, mask
    interpolated = np.interp(turns, turns[mask], values[mask])
    first_valid = np.where(mask)[0][0]
    last_valid_idx = np.where(mask)[0][-1]
    interpolated[:first_valid] = np.nan
    interpolated[last_valid_idx + 1:] = np.nan
    return turns, interpolated, mask


def safe_max(values, default=0):
    valid = [v for v in values if not np.isnan(v)]
    return max(valid) if valid else default


def last_valid(values):
    for v in reversed(values):
        if not np.isnan(v):
            return v
    return None


def get_color_for_label(label: str, variant_idx: int = 0) -> tuple:
    """Return an RGB colour for a label.

    UIDensity plots (Nc labels): matched to Wire colours in sample-sigconf.tex.
      1c = RGB(183,47,20) red-orange, 2c = RGB(32,151,131) teal, 3c = RGB(24,96,169) blue.
    InterMat plots (Ni labels): three shades of green encoding density.
      dense (<75i) = dark green, medium (75-149i) = mid green, sparse (>=150i) = light green.
    Defense labels: green family (guarded) or red family (unprotected).
    """
    import re as _re
    label_lower = label.lower()

    # ── 1. Candidate-count label: "Nc" — Wire colours from sample-sigconf.tex ──
    # Check this FIRST so "TOMA 2c — T-Guard" gets Wire colour, not defense green.
    # Only skip if the label has an explicit Ni pattern (e.g. "2c_dense50i"),
    # meaning item-count is the primary dimension. Density keywords alone (e.g.
    # "1c (medium)") do NOT override — the Nc colour takes priority.
    m_c = _re.match(r'^(\d+)c\b', label_lower)
    if not m_c:
        # also match "... Nc ..." anywhere in label (e.g. "TOMA 2c — T-Guard")
        m_c = _re.search(r'\b(\d+)c\b', label_lower)
    # Only treat as InterMat (skip Nc branch) when Ni is the PRIMARY dimension,
    # i.e. the label starts with a number+i. Auto-label appends "(Ni, Nc)" metadata
    # to Nc-primary labels like "2c (100i, 2c)" — those must still use Wire colours.
    _ni_is_primary = bool(_re.match(r'^\d+i\b', label_lower))
    _ni_in_parens  = bool(_re.search(r'\(\s*\d+i\b', label_lower))
    if m_c and not _ni_is_primary and not _ni_in_parens:
        cands = int(m_c.group(1))
        _cand_rgb = {
            0: (0.55, 0.55, 0.55),               # grey — MACF/0c baseline
            1: (183/255, 47/255,  20/255),
            2: (0x47/255, 0x8D/255, 0x7E/255),   # #478D7E
            3: (24/255,  96/255,  169/255),
            4: colorsys.hls_to_rgb(280/360, 0.42, 0.70),
        }
        return _cand_rgb.get(cands, colorsys.hls_to_rgb((variant_idx * 0.15) % 1.0, 0.40, 0.55))

    # ── 0. Defense keywords ───────────────────────────────────────────────
    _guard_kws = ('guard', 'safeguard', 'shield', 'defend', 'protected')
    _unguard_kws = ('unprotected', 'unguard', 'baseline', 'no.def')
    if any(k in label_lower for k in _guard_kws) and not any(k in label_lower for k in _unguard_kws):
        h = (140 / 360) + (variant_idx * 0.02) % 0.06
        return colorsys.hls_to_rgb(h, 0.38, 0.75)
    if any(k in label_lower for k in _unguard_kws):
        h = (5 / 360) + (variant_idx * 0.02) % 0.06
        return colorsys.hls_to_rgb(h, 0.45, 0.80)

    # ── 2. Item-count label: "Ni" — three shades of green ────────────────
    # Check BEFORE guard keywords so "T-Guard (200i)" uses Ni color, not guard green.
    # dense (<75i) → dark green, medium (75-149i) → mid green, sparse (>=150i) → light green
    m_i = _re.search(r'(\d+)i\b', label_lower)
    if m_i and (_ni_is_primary or _ni_in_parens):
        n_items = int(m_i.group(1))
        if n_items >= 150:
            return (0x6A/255, 0xCB/255, 0xB7/255)
        elif n_items >= 75:
            return (0x47/255, 0x8D/255, 0x7E/255)
        else:
            return (0x1A/255, 0x5C/255, 0x3E/255)
        n_items = int(m_i.group(1))
        if n_items >= 150:
            return (0x6A/255, 0xCB/255, 0xB7/255)   # sparse → #6ACBB7 light
        elif n_items >= 75:
            return (0x47/255, 0x8D/255, 0x7E/255)   # medium → #478D7E mid
        else:
            return (0x1A/255, 0x5C/255, 0x3E/255)   # dense  → #1A5C3E dark

    # ── 3. Legacy keyword fallbacks ───────────────────────────────────────
    if 'sparse' in label_lower:
        return (0x6A/255, 0xCB/255, 0xB7/255)
    if 'medium' in label_lower or 'full' in label_lower:
        return (0x47/255, 0x8D/255, 0x7E/255)
    if 'dense' in label_lower:
        return (0x1A/255, 0x5C/255, 0x3E/255)

    # ── 4. Generic fallback ───────────────────────────────────────────────
    h = (variant_idx * 0.15) % 1.0
    return colorsys.hls_to_rgb(h, 0.40, 0.55)


MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', '<', '>']

# Background colours to distinguish UIDensity vs InterMat plots
BG_UIDENSITY = '#EBF4FB'   # very light blue
BG_INTERMAT  = '#FDF0F5'   # very light pink

# Uniform subplot size for consistent aspect ratio across all figures
SUBPLOT_W = 2.8   # inches per subplot column
SUBPLOT_H = 2.2   # inches per subplot row


def get_linestyle_for_label(label: str) -> object:
    """All lines are solid — density is encoded by colour and marker."""
    return 'solid'


import matplotlib.path as _mpath
import numpy as _np_markers

def _tall_diamond():
    """Squarish diamond."""
    verts = _np_markers.array([[0.0, 0.65], [0.5, 0.0], [0.0, -0.65], [-0.5, 0.0], [0.0, 0.65]])
    codes = [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 3 + [_mpath.Path.CLOSEPOLY]
    return _mpath.Path(verts, codes)

def _tall_x():
    """Stretched X: two thin crossing bars, tall aspect ratio."""
    w, h = 0.18, 0.9
    # Two rotated rectangles forming an X, approximated as two thin diamonds
    verts = _np_markers.array([
        # bar 1: top-left to bottom-right
        [-w, h], [w, h], [w, -h], [-w, -h], [-w, h],
    ])
    # Rotate 45 degrees
    angle = _np_markers.pi / 4
    c, s = _np_markers.cos(angle), _np_markers.sin(angle)
    R = _np_markers.array([[c, -s], [s, c]])
    b1 = (R @ _np_markers.array([[-w, h], [w, h], [w, -h], [-w, -h], [-w, h]]).T).T
    b2 = (R.T @ _np_markers.array([[-w, h], [w, h], [w, -h], [-w, -h], [-w, h]]).T).T
    verts = _np_markers.vstack([b1, [[_np_markers.nan, _np_markers.nan]], b2])
    codes = ([_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 3 + [_mpath.Path.CLOSEPOLY] +
             [_mpath.Path.MOVETO] +
             [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 3 + [_mpath.Path.CLOSEPOLY])
    return _mpath.Path(verts, codes)

def _tall_star():
    """Stretched spiky 6-point star, tall aspect ratio."""
    n = 6
    outer_r_x, outer_r_y = 0.35, 0.9
    inner_r_x, inner_r_y = 0.12, 0.35
    verts = []
    for i in range(n):
        angle_out = _np_markers.pi / 2 + 2 * _np_markers.pi * i / n
        angle_in  = angle_out + _np_markers.pi / n
        verts.append([outer_r_x * _np_markers.cos(angle_out), outer_r_y * _np_markers.sin(angle_out)])
        verts.append([inner_r_x * _np_markers.cos(angle_in),  inner_r_y * _np_markers.sin(angle_in)])
    verts.append(verts[0])
    codes = [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * (len(verts) - 2) + [_mpath.Path.CLOSEPOLY]
    return _mpath.Path(_np_markers.array(verts), codes)

_MARKER_TALL_DIAMOND = _tall_diamond()
_MARKER_TALL_STAR = _tall_star()  # kept for fallback

# Thick stretched ✖: two wide rectangles crossed at ±45°
def _tall_x_proper():
    w, h = 0.22, 0.65   # half-width, half-height of each bar
    bar = _np_markers.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]])
    angle = _np_markers.pi / 4
    c, s = _np_markers.cos(angle), _np_markers.sin(angle)
    R = _np_markers.array([[c, -s], [s, c]])
    b1 = (R @ bar.T).T
    b2 = (R.T @ bar.T).T
    v = _np_markers.vstack([b1, b2])
    codes = ([_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 3 + [_mpath.Path.CLOSEPOLY] +
             [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 3 + [_mpath.Path.CLOSEPOLY])
    return _mpath.Path(v, codes)

def _tall_hourglass():
    """Tall hourglass ⧗: two triangles meeting at pinch, stretched vertically."""
    verts = _np_markers.array([
        [-0.5,  0.65], [ 0.5,  0.65], [ 0.0,  0.0],
        [ 0.5, -0.65], [-0.5, -0.65], [ 0.0,  0.0],
        [-0.5,  0.65],
    ])
    codes = [_mpath.Path.MOVETO] + [_mpath.Path.LINETO] * 5 + [_mpath.Path.CLOSEPOLY]
    return _mpath.Path(verts, codes)

_MARKER_TALL_X         = _tall_x_proper()
_MARKER_TALL_HOURGLASS = _tall_hourglass()


def get_marker_for_label(label: str):
    """Return tall custom marker: oval (sparsest), stretched-X (medium), spiky star (densest)."""
    import re as _re
    label_lower = label.lower()
    _ni_is_primary = bool(_re.match(r'^\d+i\b', label_lower))
    _ni_in_parens  = bool(_re.search(r'\(\s*\d+i\b', label_lower))
    m_c = _re.match(r'^(\d+)c\b', label_lower) or _re.search(r'\b(\d+)c\b', label_lower)
    if m_c and not _ni_is_primary and not _ni_in_parens:
        k = int(m_c.group(1))
        if k >= 3:   return _MARKER_TALL_HOURGLASS
        elif k == 2: return _MARKER_TALL_X
        else:        return _MARKER_TALL_DIAMOND
    m_i = _re.search(r'(\d+)i\b', label_lower)
    if m_i:
        n = int(m_i.group(1))
        if n < 75:    return _MARKER_TALL_HOURGLASS
        elif n < 150: return _MARKER_TALL_X
        else:         return _MARKER_TALL_DIAMOND
    if 'dense' in label_lower:   return _MARKER_TALL_HOURGLASS
    if 'medium' in label_lower:  return _MARKER_TALL_X
    if 'sparse' in label_lower:  return _MARKER_TALL_DIAMOND
    return 'None'


def infer_bg_color(labels: list) -> str:
    """Return BG_UIDENSITY if labels look like Nc, BG_INTERMAT if Ni."""
    import re as _re
    for lbl in labels:
        if _re.search(r'\d+i\b', lbl.lower()):
            return BG_INTERMAT
        if _re.match(r'\d+c\b', lbl.lower()):
            return BG_UIDENSITY
    return BG_UIDENSITY


def apply_bg(fig, axes, label_order):
    """Set individual axes background based on label type (not the figure canvas)."""
    bg = infer_bg_color(label_order)
    for ax in np.array(axes).flat:
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(bg)
        for name, spine in ax.spines.items():
            if name in ('top', 'right'):
                spine.set_visible(False)
            else:
                spine.set_linewidth(0.6)
                spine.set_color('#bbbbbb')


def sort_labels_for_legend(label_order: list) -> list:
    """Sort labels: Nc ordered 1c→3c, Ni ordered 200i→50i (sparse→dense)."""
    import re as _re

    def _sort_key(lbl):
        m_c = _re.match(r'^(\d+)c\b', lbl.lower())
        if m_c:
            return (0, int(m_c.group(1)))
        m_i = _re.search(r'(\d+)i\b', lbl.lower())
        if m_i:
            return (1, -int(m_i.group(1)))  # descending: 200i < 100i < 50i
        return (2, lbl)

    try:
        return sorted(label_order, key=_sort_key)
    except Exception:
        return label_order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RE_METRIC_KEYS = ['ui_topology_recall', 'ui_topology_precision', 'ui_topology_f1',
                  'extract_rate', 'ss_system_prompt', 'ss_task_instructions']


def load_macf_data(macf_dir: str) -> dict:
    macf_path = Path(macf_dir)
    task_dirs = sorted(macf_path.glob("task_*"), key=lambda x: int(x.name.split('_')[1]))
    all_turn_data = {}
    global_turn = 0
    for task_dir in task_dirs:
        turn_files = sorted(task_dir.glob("turn_*.json"), key=lambda x: int(x.stem.split('_')[1]))
        local_nums = [int(f.stem.split('_')[1]) for f in turn_files]
        if local_nums:
            turn_to_file = {int(f.stem.split('_')[1]): f for f in turn_files}
            for local_turn in range(max(local_nums) + 1):
                if local_turn in turn_to_file:
                    all_turn_data[global_turn] = turn_to_file[local_turn]
                global_turn += 1
    if not all_turn_data:
        return {'turns': [], 'user_victim_pct': [], 'item_victim_pct': [], 'name': 'Tree (MACF)'}
    max_turn = max(all_turn_data.keys())
    turns, user_rates, item_rates = [], [], []
    for t in range(max_turn + 1):
        turns.append(t)
        if t not in all_turn_data:
            user_rates.append(np.nan)
            item_rates.append(np.nan)
            continue
        try:
            with open(all_turn_data[t]) as f:
                data = json.load(f)
            ur = get_nested_value(data, 'llm_judge.global_user_victim_rate') or \
                 get_nested_value(data, 'llm_judge.user_victim_rate')
            ir = get_nested_value(data, 'llm_judge.global_item_victim_rate') or \
                 get_nested_value(data, 'llm_judge.item_victim_rate')
            user_rates.append(ur * 100 if ur is not None else np.nan)
            item_rates.append(ir * 100 if ir is not None else np.nan)
        except Exception as e:
            logger.warning(f"Error loading {all_turn_data[t]}: {e}")
            user_rates.append(np.nan)
            item_rates.append(np.nan)
    return {'turns': turns, 'user_victim_pct': user_rates, 'item_victim_pct': item_rates,
            'name': 'Tree (MACF CheatAgent-Item)'}


def _infer_metadata_from_yaml(exp_path: Path) -> dict:
    """Try to recover num_candidates and attacker_ratio from an attack config YAML
    when experiment_config.json is absent (old-style task_0 runs).

    Strategy: walk the path parts to find the attack type and config name,
    then look for a matching YAML under attack_config/<attack_type>/.
    e.g. .../drunk/drunk_50percent_3cand/... -> attack_config/drunk/drunk_3cand.yaml
    """
    import re as _re
    try:
        import yaml as _yaml
    except ImportError:
        return {}

    parts = exp_path.parts
    # Find the repo root (contains attack_config/)
    root = exp_path
    for _ in range(8):
        if (root / 'attack_config').exists():
            break
        root = root.parent
    else:
        return {}

    # Find attack type dir name in path (e.g. 'drunk', 'netsafe', 'toma', ...)
    attack_type = None
    config_hint = None
    for i, part in enumerate(parts):
        if (root / 'attack_config' / part).is_dir():
            attack_type = part
            if i + 1 < len(parts):
                config_hint = parts[i + 1]  # e.g. drunk_50percent_3cand
            break

    if not attack_type:
        return {}

    # Extract num_candidates from config_hint name (e.g. _3cand -> 3, _2cand -> 2)
    n_cand = None
    if config_hint:
        m = _re.search(r'_(\d+)cand', config_hint)
        if m:
            n_cand = int(m.group(1))

    # Find best matching YAML in attack_config/<attack_type>/
    yaml_dir = root / 'attack_config' / attack_type
    best_yaml = None
    if n_cand is not None:
        # Prefer yaml with matching cand count in name
        for y in sorted(yaml_dir.glob('*.yaml')):
            m = _re.search(r'_(\d+)cand', y.stem)
            if m and int(m.group(1)) == n_cand:
                best_yaml = y
                break
    if best_yaml is None:
        yamls = sorted(yaml_dir.glob('*.yaml'))
        if yamls:
            best_yaml = yamls[0]

    if best_yaml is None:
        return {}

    try:
        with open(best_yaml) as f:
            cfg = _yaml.safe_load(f)
        # Search recursively for num_candidates and attacker_ratio
        def _find(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    r = _find(v, key)
                    if r is not None:
                        return r
            return None

        result = {}
        nc = _find(cfg, 'num_candidates')
        ar = _find(cfg, 'attacker_ratio')
        if nc is not None:
            result['num_candidates'] = nc
        if ar is not None:
            result['attacker_ratio'] = ar
        if result:
            logger.debug(f"Inferred metadata from {best_yaml.name}: {result}")
        return result
    except Exception as e:
        logger.warning(f"Could not load YAML {best_yaml}: {e}")
        return {}


def load_connacf_data(connacf_dir: str, custom_metric: str = None) -> dict:
    connacf_path = Path(connacf_dir)
    if not connacf_path.exists():
        return {'turns': [], 'user_victim_pct': [], 'item_victim_pct': [],
                'name': 'DenseMesh (ConnaCF)', 'error': f'Directory not found: {connacf_dir}'}

    turn_files = sorted(connacf_path.glob("turn_*.json"), key=lambda x: int(x.stem.split('_')[1]))
    if not turn_files:
        return {'turns': [], 'user_victim_pct': [], 'item_victim_pct': [],
                'name': 'DenseMesh (ConnaCF)', 'error': f'No turn files in: {connacf_dir}'}

    # Metadata
    metadata = {'num_items': None, 'num_users': None, 'num_candidates': None, 'attacker_ratio': None, 'api_batch': None}
    cfg = connacf_path / 'experiment_config.json'
    if not cfg.exists():
        cfg = connacf_path.parent / 'experiment_config.json'
    if cfg.exists():
        try:
            with open(cfg) as f:
                c = json.load(f)
            a = c.get('attack_config', {})
            metadata['num_candidates'] = a.get('num_candidates')
            metadata['attacker_ratio'] = a.get('attacker_ratio')
            # api_batch lives at top level of experiment_config (not inside attack_config)
            raw_batch = c.get('api_batch')
            if raw_batch is not None:
                metadata['api_batch'] = int(raw_batch)
        except Exception as e:
            logger.warning(f"Error loading experiment_config.json: {e}")
    else:
        # Fallback: infer from attack config YAML using the experiment dir name
        # e.g. drunk_50percent_3cand -> attack_config/drunk/drunk_3cand.yaml
        _yaml_meta = _infer_metadata_from_yaml(connacf_path)
        if _yaml_meta:
            metadata.update(_yaml_meta)
    try:
        with open(turn_files[0]) as f:
            first = json.load(f)
        nse = first.get('natural_semantic_evolution', {})
        metadata['num_items'] = nse.get('active_items_total')
        metadata['num_users'] = nse.get('active_users_total')
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")

    turn_numbers = [int(f.stem.split('_')[1]) for f in turn_files]
    min_turn, max_turn = min(turn_numbers), max(turn_numbers)
    turn_to_file = {int(f.stem.split('_')[1]): f for f in turn_files}

    turns, user_rates, item_rates = [], [], []
    re_metrics = {k: [] for k in RE_METRIC_KEYS}
    custom_vals = []

    for t in range(min_turn, max_turn + 1):
        turns.append(t)
        if t not in turn_to_file:
            user_rates.append(np.nan)
            item_rates.append(np.nan)
            for k in RE_METRIC_KEYS:
                re_metrics[k].append(np.nan)
            custom_vals.append(np.nan)
            continue
        try:
            with open(turn_to_file[t]) as f:
                data = json.load(f)
            llm = data.get('llm_judge', {})
            nse = data.get('natural_semantic_evolution', {})
            uj = llm.get('user', {})
            ij = llm.get('item', {})
            uc = uj.get('contaminated_count') if uj else None
            ic = ij.get('contaminated_count') if ij else None
            au_tot = nse.get('active_users_total')
            ai_tot = nse.get('active_items_total')
            au_non = nse.get('active_users_non_attacker') or (uj.get('num_agents_evaluated') if uj else None)
            ai_non = nse.get('active_items_non_attacker') or (ij.get('num_agents_evaluated') if ij else None)
            na_u = nse.get('num_attacker_users', 0)
            na_i = nse.get('num_attacker_items', 0)

            def _rate(contaminated, non_atk, total, n_atk):
                if contaminated is None:
                    return np.nan
                denom = non_atk if non_atk is not None else (total - n_atk if total is not None else None)
                return (contaminated / max(1, denom)) * 100 if denom is not None else np.nan

            user_rates.append(_rate(uc, au_non, au_tot, na_u))
            item_rates.append(_rate(ic, ai_non, ai_tot, na_i))

            re_data = data.get('reverse_engineering', {})
            master_sub = re_data.get('master', {})
            for k in RE_METRIC_KEYS:
                v = re_data.get(k)
                # Fall back to re.master subdict for MASTER runs where top-level fields
                # were not promoted (older runs before the integration fix).
                if (v is None or v == 0.0) and k in master_sub and master_sub[k] not in (None, 0.0):
                    v = master_sub[k]
                re_metrics[k].append(float(v) if v is not None else np.nan)

            if custom_metric:
                v = get_nested_value(data, custom_metric)
                custom_vals.append(float(v) if v is not None else np.nan)
            else:
                custom_vals.append(np.nan)
        except Exception as e:
            logger.warning(f"Error loading turn {t}: {e}")
            user_rates.append(np.nan)
            item_rates.append(np.nan)
            for k in RE_METRIC_KEYS:
                re_metrics[k].append(np.nan)
            custom_vals.append(np.nan)

    result = {'turns': turns, 'user_victim_pct': user_rates, 'item_victim_pct': item_rates,
              'name': 'DenseMesh (ConnaCF)', 'metadata': metadata}

    # Always include RE metrics when we have turn data — zeros are valid (attack not yet succeeded).
    # Previously this only included RE data when non-zero, which caused the second row to disappear
    # for runs where RE metrics are all zero (e.g. sparse/medium interaction matrices).
    if turns:
        result['reverse_engineering'] = re_metrics

    if custom_metric and any(not np.isnan(v) for v in custom_vals):
        result['custom_metric'] = custom_vals

    return result


def extract_label_from_path(path: str) -> str:
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in ['netsafe', 'cheat', 'drunk', 'rectextattack', 'toma', 'mama', 'masleak', 'master']:
            if i + 1 < len(parts):
                return parts[i + 1]
    for i, part in enumerate(parts):
        if part.startswith('task_') and i > 0:
            return parts[i - 1]
    return Path(path).name


def generate_auto_label(data: dict, base_label: str = None) -> str:
    if _pub_mode:
        return base_label or ''
    meta = data.get('metadata', {})
    parts = []
    if meta.get('num_items') is not None:
        parts.append(f"{meta['num_items']}i")
    if meta.get('num_candidates') is not None:
        parts.append(f"{meta['num_candidates']}c")
    if meta.get('attacker_ratio') is not None:
        parts.append(f"{int(meta['attacker_ratio'] * 100)}%atk")
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"{base_label}{suffix}" if base_label else suffix.strip(" ()")


def infer_title_suffix(datasets: list) -> str:
    """Build a parenthetical suffix from shared metadata across datasets.

    Collects unique values for num_items, num_candidates, attacker_ratio.
    If all datasets agree on a value it is shown as a fixed fact;
    if they differ it is omitted (the per-label auto_label handles it).
    """
    from collections import Counter
    items   = [d.get('metadata', {}).get('num_items')      for d in datasets if d.get('metadata', {}).get('num_items')      is not None]
    cands   = [d.get('metadata', {}).get('num_candidates') for d in datasets if d.get('metadata', {}).get('num_candidates') is not None]
    ratios  = [d.get('metadata', {}).get('attacker_ratio') for d in datasets if d.get('metadata', {}).get('attacker_ratio') is not None]

    parts = []
    if items  and len(set(items))  == 1: parts.append(f"{items[0]}i")
    if cands  and len(set(cands))  == 1: parts.append(f"{cands[0]}c")
    if ratios and len(set(ratios)) == 1: parts.append(f"{int(ratios[0]*100)}% Attackers")
    return f" ({', '.join(parts)})" if parts else ""


def normalize_turns(data: dict, reference_cohort: int = 100) -> dict:
    """Rescale turns so that 1 unit = one full pass over `reference_cohort` users.

    Uses metadata already present in the dataset dict:
      cohort  = active_users_total - 1  (subtract RecBole padding token)
      batch   = api_batch  (LLM calls issued per raw turn)

    equivalent_turn = raw_turn * (batch / cohort)

    If either value is missing the turns are returned unchanged.
    """
    meta = data.get('metadata', {})
    num_users = meta.get('num_users')   # active_users_total from turn_0.json
    api_batch = meta.get('api_batch')   # from experiment_config.json
    if num_users is None or api_batch is None:
        logger.warning("normalize_turns: missing num_users or api_batch in metadata — skipping")
        return data
    cohort = max(1, int(num_users) - 1)  # subtract RecBole padding token
    scale = api_batch / cohort
    new_turns = [t * scale for t in data['turns']]
    return {**data, 'turns': new_turns}


# ---------------------------------------------------------------------------
# Shared argparse base
# ---------------------------------------------------------------------------

def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--macf_dir', type=str, default=None)
    p.add_argument('--connacf_dirs', type=str, nargs='+', default=None)
    p.add_argument('--labels', type=str, nargs='+', default=None)
    p.add_argument('--auto_label', action='store_true', default=False)
    p.add_argument('--output', '-o', type=str, default='plot.png')
    p.add_argument('--visuals_dir', type=str, default=None,
                   help='One-off output directory override (e.g. ../visuals). Replaces dirname of --output.')
    p.add_argument('--title', type=str, default=None)
    p.add_argument('--last_epoch', type=int, default=100)
    p.add_argument('--smoothing', type=float, default=0.0)
    p.add_argument(
        '--pub', action='store_true', default=False,
        help='Publication preset: 8pt font, 7pt legend, figsize=(7,2.8), dpi=300.',
    )
    p.add_argument(
        '--large-titles', action='store_true', default=False,
        help='Increase subplot title size by 5pt (for ablation/guardrail figures).',
    )
    p.add_argument(
        '--normalize_turns', action='store_true', default=False,
        help=(
            'Rescale x-axis to equivalent turns: 1 unit = one full pass over the '
            'reference cohort (100 users). Uses api_batch / (active_users_total - 1) '
            'from logged metadata. Enables fair comparison across cohort sizes.'
        ),
    )
    p.add_argument(
        '--sub_round', type=str, default='0', choices=['0', '1', 'both'],
        help=(
            'Which sub-round to plot when all_update_rounds=2 produces two turns per batch. '
            '0 = pre-backward (even turns, cold profiles), '
            '1 = post-backward (odd turns, updated profiles, default), '
            'both = keep all turns (shows zigzag).'
        ),
    )
    return p


def smooth_values(values, alpha):
    if alpha <= 0 or not values:
        return values
    alpha = min(alpha, 0.99)
    smoothed, last = [], None
    for v in values:
        if np.isnan(v):
            smoothed.append(np.nan)
        elif last is None:
            smoothed.append(v)
            last = v
        else:
            new = alpha * last + (1 - alpha) * v
            smoothed.append(new)
            last = new
    return smoothed


def filter_sub_round(turns: list, sub_round: str) -> list:
    """Return the subset of turn indices to keep based on sub_round selection.

    With all_update_rounds=2 the training loop emits two turns per batch:
      even turns (0, 2, 4, ...) = pre-backward  (sub_round='0')
      odd  turns (1, 3, 5, ...) = post-backward (sub_round='1', default)

    Turn 0 is always kept regardless of parity — it is the baseline measurement
    before any learning has occurred and is needed for all plots.

    sub_round='both' keeps everything (shows the zigzag).
    """
    if sub_round == 'both' or not turns:
        return turns
    keep_parity = int(sub_round)  # 0 or 1
    result = []
    for t in turns:
        if t == 0 or (t % 2) == keep_parity:
            result.append(t)
    return result


def _remap_turns(turns: list) -> list:
    """Re-index a filtered turn list to 0, 1, 2, ... for clean x-axis display."""
    return list(range(len(turns)))


def trim_data(data: dict, last_epoch: int, smoothing: float = 0.0,
              sub_round: str = 'both') -> dict:
    if not data['turns'] or last_epoch is None:
        raw_turns = data['turns']
    else:
        raw_turns = data['turns'][:last_epoch]

    # Apply sub_round filtering
    kept = filter_sub_round(raw_turns, sub_round)
    # Cap displayed turns at MAX_DISPLAY_TURNS (after sub_round halving)
    # kept = kept[:MAX_DISPLAY_TURNS]  # removed — use --last_epoch exclusively
    kept_set = set(kept)
    indices = [i for i, t in enumerate(raw_turns) if t in kept_set]

    def _pick(lst):
        return [lst[i] for i in indices] if lst else []

    up = smooth_values(_pick(data['user_victim_pct']), smoothing)
    ip = smooth_values(_pick(data['item_victim_pct']), smoothing)
    display_turns = _remap_turns(kept) if sub_round != 'both' else kept

    result = {
        'turns': display_turns,
        'user_victim_pct': up,
        'item_victim_pct': ip,
        'name': data['name'],
        **{k: v for k, v in data.items()
           if k not in ('turns', 'user_victim_pct', 'item_victim_pct', 'name',
                        'reverse_engineering', 'custom_metric')}
    }
    if 'reverse_engineering' in data:
        result['reverse_engineering'] = {
            k: smooth_values(_pick(v), smoothing)
            for k, v in data['reverse_engineering'].items()
        }
    if 'custom_metric' in data:
        result['custom_metric'] = smooth_values(_pick(data['custom_metric']), smoothing)
    return result


def load_datasets(args, custom_metric=None):
    """Load all datasets from args, apply trim/smooth/sub_round, assign labels."""
    global _pub_figsize_override, _pub_dpi_override
    if getattr(args, 'pub', False):
        figsize, dpi = apply_pub_style()
        _pub_figsize_override = figsize
        _pub_dpi_override = dpi
    # One-off output directory override
    if getattr(args, 'visuals_dir', None):
        import os as _os
        _vd = args.visuals_dir
        _os.makedirs(_vd, exist_ok=True)
        args.output = _os.path.join(_vd, _os.path.basename(args.output))
    if getattr(args, 'large_titles', False):
        global TITLE_FONTSIZE
        TITLE_FONTSIZE = TITLE_FONTSIZE + 5
        matplotlib.rcParams['axes.titlesize'] = matplotlib.rcParams.get('axes.titlesize', 14) + 5

    datasets = []
    label_idx = 0
    sub_round = getattr(args, 'sub_round', 'both')

    if getattr(args, 'macf_dir', None):
        d = load_macf_data(args.macf_dir)
        d = trim_data(d, args.last_epoch, args.smoothing, sub_round)
        if args.labels and label_idx < len(args.labels):
            base = args.labels[label_idx]; label_idx += 1
            d['name'] = generate_auto_label(d, base) if args.auto_label else base
        datasets.append(d)

    for connacf_dir in (args.connacf_dirs or []):
        d = load_connacf_data(connacf_dir, custom_metric=custom_metric)
        d = trim_data(d, args.last_epoch, args.smoothing, sub_round)
        if getattr(args, 'normalize_turns', False):
            d = normalize_turns(d)
        if args.labels and label_idx < len(args.labels):
            base = args.labels[label_idx]; label_idx += 1
        else:
            base = extract_label_from_path(connacf_dir)
        d['name'] = generate_auto_label(d, base) if args.auto_label else base
        d['path'] = connacf_dir  # preserve raw path for attack-specific plotters
        datasets.append(d)

    return datasets


def group_by_label(datasets):
    groups = defaultdict(list)
    order = []
    for d in datasets:
        if d['name'] not in groups:
            order.append(d['name'])
        groups[d['name']].append(d)
    return order, groups


def compute_band_data(group_datasets, metric_key, nested_key=None):
    all_turns = sorted({t for d in group_datasets for t in d['turns']})
    if not all_turns:
        return None
    turn_idx = {t: i for i, t in enumerate(all_turns)}
    mat = np.full((len(group_datasets), len(all_turns)), np.nan)
    for di, d in enumerate(group_datasets):
        raw = d.get(metric_key, {}).get(nested_key, []) if nested_key else d.get(metric_key, [])
        if not raw:
            continue
        _, iv, _ = interpolate_nans(d['turns'], raw)
        for t, v in zip(d['turns'], iv):
            if t in turn_idx:
                mat[di, turn_idx[t]] = v
    with np.errstate(all='ignore'):
        return (np.array(all_turns),
                np.nanmean(mat, axis=0),
                np.nanmin(mat, axis=0),
                np.nanmax(mat, axis=0))


def maybe_legend(fig_or_ax, *args, **kwargs):
    """Add legend only when not in pub mode."""
    if _pub_mode:
        return
    fig_or_ax.legend(*args, **kwargs)


LINE_W = 2.2   # shared line width across all plot_* scripts
LINE_A = 0.75  # shared line alpha
TITLE_FONTSIZE = 22  # shared subplot title font size
AXIS_FONTSIZE  = 11  # shared axis label / tick font size


def plot_group(ax, group_datasets, label, color, marker, metric_key, nested_key=None):
    ls = get_linestyle_for_label(label)
    mk = get_marker_for_label(label)
    max_val = 0
    if len(group_datasets) == 1:
        d = group_datasets[0]
        vals = d.get(metric_key, {}).get(nested_key, []) if nested_key else d.get(metric_key, [])
        if d['turns'] and vals:
            ti, vi, mask = interpolate_nans(d['turns'], vals)
            every = max(1, len(ti) // 8)
            ax.plot(ti, vi, linewidth=LINE_W, color=color, label=label, alpha=LINE_A, linestyle=ls,
                    marker=mk, markevery=every, markersize=9, markeredgewidth=1.2,
                    markerfacecolor=(*color[:3], 0.55) if len(color) == 3 else color)
            max_val = safe_max(vals)
    else:
        bd = compute_band_data(group_datasets, metric_key, nested_key)
        if bd is not None:
            turns, mean_v, min_v, max_v = bd
            valid = ~np.isnan(mean_v)
            if valid.any():
                ax.fill_between(turns, min_v, max_v, color=color, alpha=0.2,
                                label=f'{label} (range, n={len(group_datasets)})')
                every = max(1, int(valid.sum()) // 8)
                ax.plot(turns, mean_v, linewidth=LINE_W, color=color, alpha=LINE_A, linestyle=ls,
                        marker=mk, markevery=every, markersize=9, markeredgewidth=1.2,
                        markerfacecolor=(*color[:3], 0.55) if len(color) == 3 else color)
                max_val = float(np.nanmax(max_v))
    return max_val


def annotate_finals(ax, label_order, label_groups, label_colors, metric_key,
                    nested_key=None, fmt='.1f', unit='%'):
    if _pub_mode:
        return   # suppress in pub — LaTeX captions carry the context
    y = 0.95
    for label in label_order:
        group = label_groups[label]
        color = label_colors[label]
        if len(group) == 1:
            vals = group[0].get(metric_key, {}).get(nested_key, []) if nested_key \
                   else group[0].get(metric_key, [])
            v = last_valid(vals) if vals else None
            if v is not None:
                ax.text(0.95, y, f"{label}: {v:{fmt}}{unit}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= 0.06
        else:
            fvals = []
            for d in group:
                vals = d.get(metric_key, {}).get(nested_key, []) if nested_key else d.get(metric_key, [])
                v = last_valid(vals) if vals else None
                if v is not None:
                    fvals.append(v)
            if fvals:
                mn, lo, hi = np.mean(fvals), np.min(fvals), np.max(fvals)
                ax.text(0.95, y, f"{label}: {mn:{fmt}}{unit} [{lo:{fmt}}-{hi:{fmt}}]",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y -= 0.06
