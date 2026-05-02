#!/usr/bin/env python3
"""
Plot comparison of victim rates between experiments.

Supports:
1. MACF vs ConnaCF comparison (original mode)
2. Multiple ConnaCF runs comparison (new mode)

Shows User % and Item % victim rates side-by-side for easy comparison.

Usage:
    # Compare MACF vs ConnaCF (original)
    python tools/plot_comparison.py \
        --macf_dir macf_output/cheat_50percent/CDs-100user-dense \
        --connacf_dir attack_output/netsafe/misinfo_50u_133i/CDs-100user-dense/task_0 \
        --output comparison_plot.png
    
    # Compare multiple ConnaCF runs
    python tools/plot_comparison.py \
        --connacf_dirs attack_output/netsafe/misinfo_sparse/CDs-100user-dense/task_0 \
                       attack_output/netsafe/misinfo_full/CDs-100user-dense/task_0 \
                       attack_output/netsafe/misinfo_dense/CDs-100user-dense/task_0 \
        --labels "Sparse" "Full" "Dense" \
        --output connacf_comparison.png
"""

import os
import json
import argparse
import glob
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_nested_value(data, path):
    """Extract nested value from dict using dot notation path"""
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
    except:
        return None


def load_macf_data(macf_dir: str) -> dict:
    """Load MACF turn data and extract victim rates."""
    macf_path = Path(macf_dir)
    
    # Find all task directories
    task_dirs = sorted(macf_path.glob("task_*"), key=lambda x: int(x.name.split('_')[1]))
    
    # First pass: collect all turn files with their global turn numbers
    all_turn_data = {}  # global_turn -> file_path
    global_turn = 0
    
    for task_dir in task_dirs:
        turn_files = sorted(task_dir.glob("turn_*.json"), key=lambda x: int(x.stem.split('_')[1]))
        local_turn_numbers = [int(f.stem.split('_')[1]) for f in turn_files]
        
        if local_turn_numbers:
            max_local_turn = max(local_turn_numbers)
            turn_to_file = {int(f.stem.split('_')[1]): f for f in turn_files}
            
            for local_turn in range(max_local_turn + 1):
                if local_turn in turn_to_file:
                    all_turn_data[global_turn] = turn_to_file[local_turn]
                # else: missing turn, will be NaN
                global_turn += 1
    
    if not all_turn_data:
        return {
            'turns': [],
            'user_victim_pct': [],
            'item_victim_pct': [],
            'name': 'Tree (MACF CheatAgent-Item)'
        }
    
    # Second pass: build arrays with NaN for missing turns
    max_turn = max(all_turn_data.keys())
    turns = []
    user_victim_rates = []
    item_victim_rates = []
    
    for turn_num in range(max_turn + 1):
        turns.append(turn_num)
        
        if turn_num not in all_turn_data:
            user_victim_rates.append(np.nan)
            item_victim_rates.append(np.nan)
            continue
        
        turn_file = all_turn_data[turn_num]
        try:
            with open(turn_file, 'r') as f:
                data = json.load(f)
            
            # Try to get global victim rates first, then fall back to regular victim rates
            user_rate = get_nested_value(data, 'llm_judge.global_user_victim_rate')
            item_rate = get_nested_value(data, 'llm_judge.global_item_victim_rate')
            
            # Fallback to non-global rates if global not available
            if user_rate is None:
                user_rate = get_nested_value(data, 'llm_judge.user_victim_rate')
            if item_rate is None:
                item_rate = get_nested_value(data, 'llm_judge.item_victim_rate')
            
            user_victim_rates.append(user_rate * 100 if user_rate is not None else np.nan)
            item_victim_rates.append(item_rate * 100 if item_rate is not None else np.nan)
            
        except Exception as e:
            logger.warning(f"Error loading {turn_file}: {e}")
            user_victim_rates.append(np.nan)
            item_victim_rates.append(np.nan)
    
    return {
        'turns': turns,
        'user_victim_pct': user_victim_rates,
        'item_victim_pct': item_victim_rates,
        'name': 'Tree (MACF CheatAgent-Item)'
    }


def load_connacf_data(connacf_dir: str, custom_metric: str = None) -> dict:
    """Load ConnaCF (netsafe) turn data and extract victim rates.
    
    Victim rate = contaminated_non_attackers / innocent_agents
    This matches MACF's victim_rate metric for proper comparison.
    
    Also extracts metadata: num_items, num_candidates, attacker_ratio for auto-labeling.
    
    Args:
        connacf_dir: Path to the task directory containing turn_*.json files.
        custom_metric: Optional dot-notation path to extract from each turn JSON,
                       e.g. "stealth.ia_asr_valid" or "resources.corba_blocking_rate".
    """
    connacf_path = Path(connacf_dir)
    
    # Check if directory exists
    if not connacf_path.exists():
        logger.warning(f"Directory does not exist: {connacf_dir}")
        return {
            'turns': [],
            'user_victim_pct': [],
            'item_victim_pct': [],
            'name': 'DenseMesh (ConnaCF)',
            'error': f'Directory not found: {connacf_dir}'
        }
    
    turn_files = sorted(connacf_path.glob("turn_*.json"), key=lambda x: int(x.stem.split('_')[1]))
    
    if not turn_files:
        logger.warning(f"No turn_*.json files found in: {connacf_dir}")
        return {
            'turns': [],
            'user_victim_pct': [],
            'item_victim_pct': [],
            'name': 'DenseMesh (ConnaCF)',
            'error': f'No turn files in: {connacf_dir}'
        }
    
    # === EXTRACT METADATA ===
    metadata = {
        'num_items': None,
        'num_users': None,
        'num_candidates': None,
        'attacker_ratio': None,
    }
    
    # Try to load experiment_config.json for num_candidates and attacker_ratio
    config_path = connacf_path / 'experiment_config.json'
    if not config_path.exists():
        # Try parent directory (for old task_0 structure)
        config_path = connacf_path.parent / 'experiment_config.json'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            attack_config = config.get('attack_config', {})
            metadata['num_candidates'] = attack_config.get('num_candidates')
            metadata['attacker_ratio'] = attack_config.get('attacker_ratio')
        except Exception as e:
            logger.warning(f"Error loading experiment_config.json: {e}")
    
    # Get num_items from first turn file's natural_semantic_evolution
    try:
        with open(turn_files[0], 'r') as f:
            first_turn = json.load(f)
        nse = first_turn.get('natural_semantic_evolution', {})
        metadata['num_items'] = nse.get('active_items_total')
        metadata['num_users'] = nse.get('active_users_total')
    except Exception as e:
        logger.warning(f"Error extracting metadata from first turn: {e}")
    
    # Find the range of turns (min to max)
    turn_numbers = [int(f.stem.split('_')[1]) for f in turn_files]
    min_turn = min(turn_numbers)
    max_turn = max(turn_numbers)
    
    # Create a mapping of turn number to file
    turn_to_file = {int(f.stem.split('_')[1]): f for f in turn_files}
    
    turns = []
    user_victim_rates = []
    item_victim_rates = []
    
    # Reverse engineering metrics (for MASLeak)
    RE_METRIC_KEYS = ['ui_topology_recall', 'ui_topology_precision', 'ui_topology_f1',
                      'extract_rate', 'ss_system_prompt', 'ss_task_instructions']
    re_metrics = {k: [] for k in RE_METRIC_KEYS}
    custom_metric_values = []  # For --metric flag
    
    for turn_num in range(min_turn, max_turn + 1):
        turns.append(turn_num)
        
        if turn_num not in turn_to_file:
            # Missing turn - use NaN
            user_victim_rates.append(np.nan)
            item_victim_rates.append(np.nan)
            for k in RE_METRIC_KEYS:
                re_metrics[k].append(np.nan)
            custom_metric_values.append(np.nan)
            continue
        
        turn_file = turn_to_file[turn_num]
        try:
            with open(turn_file, 'r') as f:
                data = json.load(f)
            
            # Get LLM Judge contamination counts (these are VICTIM counts, not including attackers)
            llm_judge = data.get('llm_judge', {})
            nse = data.get('natural_semantic_evolution', {})
            
            # Check if llm_judge data exists for user/item
            user_judge_data = llm_judge.get('user', {})
            item_judge_data = llm_judge.get('item', {})
            
            user_contaminated = user_judge_data.get('contaminated_count') if user_judge_data else None
            item_contaminated = item_judge_data.get('contaminated_count') if item_judge_data else None
            
            # Get total agent counts - try multiple sources
            # 1. From natural_semantic_evolution (preferred)
            active_users_total = nse.get('active_users_total')
            active_items_total = nse.get('active_items_total')
            active_users_non_attacker = nse.get('active_users_non_attacker')
            active_items_non_attacker = nse.get('active_items_non_attacker')
            
            # 2. Fallback: use num_agents_evaluated from llm_judge (these are non-attackers)
            if active_users_non_attacker is None and user_judge_data:
                active_users_non_attacker = user_judge_data.get('num_agents_evaluated')
            if active_items_non_attacker is None and item_judge_data:
                active_items_non_attacker = item_judge_data.get('num_agents_evaluated')
            
            # Get attacker counts
            num_attacker_users = nse.get('num_attacker_users', 0)
            num_attacker_items = nse.get('num_attacker_items', 0)
            
            # Calculate user victim rate
            user_rate = np.nan
            if user_contaminated is not None:
                # Prefer active_users_non_attacker if available
                if active_users_non_attacker is not None:
                    innocent_users = max(1, active_users_non_attacker)
                    user_rate = (user_contaminated / innocent_users) * 100
                elif active_users_total is not None:
                    innocent_users = max(1, active_users_total - 1 - num_attacker_users)  # -1 for PAD
                    user_rate = (user_contaminated / innocent_users) * 100
            
            # Calculate item victim rate
            item_rate = np.nan
            if item_contaminated is not None:
                # Prefer active_items_non_attacker if available
                if active_items_non_attacker is not None:
                    innocent_items = max(1, active_items_non_attacker)
                    item_rate = (item_contaminated / innocent_items) * 100
                elif active_items_total is not None:
                    innocent_items = max(1, active_items_total - 1 - num_attacker_items)  # -1 for PAD
                    item_rate = (item_contaminated / innocent_items) * 100
            
            user_victim_rates.append(user_rate)
            item_victim_rates.append(item_rate)
            
            # Extract reverse_engineering metrics
            re_data = data.get('reverse_engineering', {})
            for k in RE_METRIC_KEYS:
                val = re_data.get(k)
                re_metrics[k].append(float(val) if val is not None else np.nan)
            
            # Extract custom metric if requested
            if custom_metric:
                val = get_nested_value(data, custom_metric)
                custom_metric_values.append(float(val) if val is not None else np.nan)
            else:
                custom_metric_values.append(np.nan)
            
        except Exception as e:
            logger.warning(f"Error loading {turn_file}: {e}")
            user_victim_rates.append(np.nan)
            item_victim_rates.append(np.nan)
            for k in RE_METRIC_KEYS:
                re_metrics[k].append(np.nan)
            custom_metric_values.append(np.nan)
    
    result = {
        'turns': turns,
        'user_victim_pct': user_victim_rates,
        'item_victim_pct': item_victim_rates,
        'name': 'DenseMesh (ConnaCF)',
        'metadata': metadata,
    }
    
    # Only include reverse_engineering if any non-NaN values exist
    has_re = any(
        any(not np.isnan(v) for v in re_metrics[k])
        for k in RE_METRIC_KEYS
    )
    if has_re:
        result['reverse_engineering'] = re_metrics
    
    # Include custom metric if requested and has data
    if custom_metric and any(not np.isnan(v) for v in custom_metric_values):
        result['custom_metric'] = custom_metric_values
    
    return result


def extract_label_from_path(path: str) -> str:
    """Extract a meaningful label from an ConnaCF output path.
    
    Examples:
        attack_output/netsafe/misinfo_sparse/CDs-100user-dense/task_0 -> misinfo_sparse
        attack_output/cheat/cheat_user_50percent/CDs-100user-dense/task_0 -> cheat_user_50percent
    """
    parts = Path(path).parts
    # Look for the attack config name (usually 2-3 levels deep)
    for i, part in enumerate(parts):
        if part in ['netsafe', 'cheat', 'drunk', 'rectextattack']:
            if i + 1 < len(parts):
                return parts[i + 1]
    # Fallback: use parent of task_X
    for i, part in enumerate(parts):
        if part.startswith('task_'):
            if i > 0:
                return parts[i - 1]
    return Path(path).name


def generate_auto_label(data: dict, base_label: str = None) -> str:
    """Generate an informative label from metadata.
    
    Format: "{base_label} ({n_items}i, {n_cand}c, {atk_pct}%atk)"
    
    Args:
        data: Dict with 'metadata' key containing num_items, num_candidates, attacker_ratio
        base_label: Optional base label to prepend
    
    Returns:
        Auto-generated label string
    """
    metadata = data.get('metadata', {})
    
    parts = []
    
    # Number of items
    n_items = metadata.get('num_items')
    if n_items is not None:
        parts.append(f"{n_items}i")
    
    # Number of candidates (topology density)
    n_cand = metadata.get('num_candidates')
    if n_cand is not None:
        parts.append(f"{n_cand}c")
    
    # Attacker ratio
    atk_ratio = metadata.get('attacker_ratio')
    if atk_ratio is not None:
        parts.append(f"{int(atk_ratio * 100)}%atk")
    
    if parts:
        suffix = f" ({', '.join(parts)})"
    else:
        suffix = ""
    
    if base_label:
        return f"{base_label}{suffix}"
    else:
        return suffix.strip(" ()")


def plot_multi_connacf_comparison(datasets: list, output_file: str, title: str = None, show_re_row: bool = True):
    """
    Plot comparison of multiple ConnaCF runs.

    When multiple datasets share the same label, they are grouped together and displayed as:
    - A mean line (solid)
    - A translucent band showing the min-max range

    If any dataset contains reverse_engineering metrics (MASLeak), adds a second row
    with panels for those metrics.

    Args:
        datasets: List of dicts with 'turns', 'user_victim_pct', 'item_victim_pct', 'name'
        output_file: Output file path
        title: Optional custom title
    """
    if not datasets:
        logger.error("No datasets to plot")
        return None

    # Check if any dataset has reverse_engineering data
    RE_METRIC_KEYS = ['ui_topology_recall', 'ui_topology_precision', 'ui_topology_f1',
                      'extract_rate', 'ss_system_prompt', 'ss_task_instructions']
    RE_LABELS = {
        'ui_topology_recall': 'U-I Topology Recall',
        'ui_topology_precision': 'U-I Topology Precision',
        'ui_topology_f1': 'U-I Topology F1',
        'extract_rate': 'Overall Extraction Rate',
        'ss_system_prompt': 'System Prompt Similarity',
        'ss_task_instructions': 'Task Instr. Similarity',
    }

    has_re_data = show_re_row and any('reverse_engineering' in d for d in datasets)

    # Determine which RE metrics actually have data
    active_re_metrics = []
    if has_re_data:
        for k in RE_METRIC_KEYS:
            for d in datasets:
                re = d.get('reverse_engineering', {})
                vals = re.get(k, [])
                if any(not np.isnan(v) for v in vals):
                    active_re_metrics.append(k)
                    break

    # Layout: 1 row if no RE data, 2 rows if RE data exists
    if active_re_metrics:
        n_re_panels = len(active_re_metrics)
        # Row 1: 2 panels (user/item victim). Row 2: RE metric panels
        n_cols = max(2, n_re_panels)
        fig, all_axes = plt.subplots(2, n_cols, figsize=(7 * n_cols, 10))
        # Ensure 2D array
        if n_cols == 1:
            all_axes = all_axes.reshape(2, 1)
        axes_row1 = all_axes[0]
        axes_row2 = all_axes[1]
        # Hide unused panels in row 1 if n_cols > 2
        for i in range(2, n_cols):
            axes_row1[i].set_visible(False)
        # Hide unused panels in row 2 if n_re_panels < n_cols
        for i in range(n_re_panels, n_cols):
            axes_row2[i].set_visible(False)
    else:
        fig, axes_row1 = plt.subplots(1, 2, figsize=(14, 5))
        axes_row2 = None

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('ConnaCF Victim Rate Comparison\n(% of Innocent Agents Contaminated)',
                     fontsize=14, fontweight='bold')

    # Color families based on label keywords
    def get_color_for_label(label: str, variant_idx: int = 0) -> tuple:
        label_lower = label.lower()
        if 'tree' in label_lower:
            base_hue = 120 / 360
            base_sat, base_light = 0.65, 0.40
        elif 'sparse' in label_lower:
            base_hue = 200 / 360
            base_sat, base_light = 0.70, 0.45
        elif 'medium' in label_lower or 'full' in label_lower:
            base_hue = 40 / 360
            base_sat, base_light = 0.75, 0.45
        elif 'dense' in label_lower:
            base_hue = 0 / 360
            base_sat, base_light = 0.75, 0.65
        else:
            base_hue = (variant_idx * 0.15) % 1.0
            base_sat, base_light = 0.50, 0.50

        hue_shift = (variant_idx * 0.03) % 0.1 - 0.05
        sat_shift = (variant_idx * 0.05) % 0.15 - 0.075
        light_shift = (variant_idx * 0.04) % 0.12 - 0.06

        hue = (base_hue + hue_shift) % 1.0
        sat = max(0.3, min(1.0, base_sat + sat_shift))
        light = max(0.25, min(0.65, base_light + light_shift))

        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        return (r, g, b)

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    def safe_max(values, default=0):
        valid = [v for v in values if not np.isnan(v)]
        return max(valid) if valid else default

    def last_valid(values):
        for v in reversed(values):
            if not np.isnan(v):
                return v
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
        interpolated[last_valid_idx+1:] = np.nan
        return turns, interpolated, mask

    # Group datasets by label name
    label_groups = defaultdict(list)
    label_order = []
    for data in datasets:
        label = data['name']
        if label not in label_groups:
            label_order.append(label)
        label_groups[label].append(data)

    label_colors = {}
    for i, label in enumerate(label_order):
        label_colors[label] = get_color_for_label(label, i)

    def compute_band_data(group_datasets, metric_key, nested_key=None):
        """Compute mean, min, max for a group of datasets sharing the same label.

        Args:
            group_datasets: list of dataset dicts
            metric_key: top-level key in dataset dict (e.g. 'user_victim_pct' or 'reverse_engineering')
            nested_key: if metric_key is a dict, the sub-key to use (e.g. 'extract_rate')
        """
        if not group_datasets:
            return None

        all_turns = set()
        for data in group_datasets:
            if data['turns']:
                all_turns.update(data['turns'])

        if not all_turns:
            return None

        turns = sorted(all_turns)
        n_turns = len(turns)
        turn_to_idx = {t: i for i, t in enumerate(turns)}

        values_matrix = np.full((len(group_datasets), n_turns), np.nan)

        for d_idx, data in enumerate(group_datasets):
            if not data['turns']:
                continue
            if nested_key:
                raw_values = data.get(metric_key, {}).get(nested_key, [])
            else:
                raw_values = data.get(metric_key, [])
            if not raw_values:
                continue
            _, interp_values, _ = interpolate_nans(data['turns'], raw_values)
            for t_idx, (t, v) in enumerate(zip(data['turns'], interp_values)):
                if t in turn_to_idx:
                    values_matrix[d_idx, turn_to_idx[t]] = v

        with np.errstate(all='ignore'):
            mean_values = np.nanmean(values_matrix, axis=0)
            min_values = np.nanmin(values_matrix, axis=0)
            max_values = np.nanmax(values_matrix, axis=0)

        return np.array(turns), mean_values, min_values, max_values

    def plot_group(ax, group_datasets, label, color, marker, metric_key, nested_key=None):
        """Plot a group of datasets - either as band (multiple) or single line."""
        max_val = 0

        if len(group_datasets) == 1:
            data = group_datasets[0]
            if nested_key:
                values = data.get(metric_key, {}).get(nested_key, [])
            else:
                values = data.get(metric_key, [])
            if data['turns'] and values:
                turns_interp, values_interp, mask = interpolate_nans(data['turns'], values)
                ax.plot(turns_interp, values_interp, linewidth=2.2, color=color,
                        label=label, alpha=0.75)
                max_val = safe_max(values)
        else:
            band_data = compute_band_data(group_datasets, metric_key, nested_key)
            if band_data is not None:
                turns, mean_vals, min_vals, max_vals = band_data
                valid_mask = ~np.isnan(mean_vals)
                if valid_mask.any():
                    ax.fill_between(turns, min_vals, max_vals,
                                    color=color, alpha=0.2,
                                    label=f'{label} (range, n={len(group_datasets)})')
                    ax.plot(turns, mean_vals, linewidth=2.2, color=color, alpha=0.75)
                    max_val = np.nanmax(max_vals)

        return max_val

    def annotate_final_values(ax, label_order, label_groups, label_colors, metric_key, nested_key=None):
        """Add final value annotations to an axis."""
        y_offset = 0.95
        for label in label_order:
            group = label_groups[label]
            color = label_colors[label]

            if len(group) == 1:
                if nested_key:
                    values = group[0].get(metric_key, {}).get(nested_key, [])
                else:
                    values = group[0].get(metric_key, [])
                final_val = last_valid(values) if values else None
                if final_val is not None:
                    ax.text(0.95, y_offset, f"{label}: {final_val:.3f}",
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    y_offset -= 0.06
            else:
                final_vals = []
                for d in group:
                    if nested_key:
                        values = d.get(metric_key, {}).get(nested_key, [])
                    else:
                        values = d.get(metric_key, [])
                    v = last_valid(values) if values else None
                    if v is not None:
                        final_vals.append(v)
                if final_vals:
                    mean_final = np.mean(final_vals)
                    min_final = np.min(final_vals)
                    max_final = np.max(final_vals)
                    ax.text(0.95, y_offset, f"{label}: {mean_final:.3f} [{min_final:.3f}-{max_final:.3f}]",
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    y_offset -= 0.06

    # === Row 1, Left: User Victim Rate ===
    ax1 = axes_row1[0]
    max_user = 0

    for i, label in enumerate(label_order):
        group = label_groups[label]
        color = label_colors[label]
        marker = markers[i % len(markers)]
        max_val = plot_group(ax1, group, label, color, marker, 'user_victim_pct')
        max_user = max(max_user, max_val)

    ax1.set_xlabel('Turn', fontsize=11)
    ax1.set_ylabel('User Victim Rate (%)', fontsize=11)
    ax1.set_title('User Contamination Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(100, max_user * 1.1) if max_user > 0 else 100)

    # Add final values annotation
    y_offset = 0.95
    for label in label_order:
        group = label_groups[label]
        color = label_colors[label]

        if len(group) == 1:
            final_val = last_valid(group[0]['user_victim_pct'])
            if final_val is not None:
                ax1.text(0.95, y_offset, f"{label}: {final_val:.1f}%",
                        transform=ax1.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_offset -= 0.06
        else:
            final_vals = [last_valid(d['user_victim_pct']) for d in group]
            final_vals = [v for v in final_vals if v is not None]
            if final_vals:
                mean_final = np.mean(final_vals)
                min_final = np.min(final_vals)
                max_final = np.max(final_vals)
                ax1.text(0.95, y_offset, f"{label}: {mean_final:.1f}% [{min_final:.1f}-{max_final:.1f}]",
                        transform=ax1.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_offset -= 0.06

    # === Row 1, Right: Item Victim Rate ===
    ax2 = axes_row1[1]
    max_item = 0

    for i, label in enumerate(label_order):
        group = label_groups[label]
        color = label_colors[label]
        marker = markers[i % len(markers)]
        max_val = plot_group(ax2, group, label, color, marker, 'item_victim_pct')
        max_item = max(max_item, max_val)

    ax2.set_xlabel('Turn', fontsize=11)
    ax2.set_ylabel('Item Victim Rate (%)', fontsize=11)
    ax2.set_title('Item Contamination Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(100, max_item * 1.1) if max_item > 0 else 100)

    # Add final values annotation
    y_offset = 0.95
    for label in label_order:
        group = label_groups[label]
        color = label_colors[label]

        if len(group) == 1:
            final_val = last_valid(group[0]['item_victim_pct'])
            if final_val is not None:
                ax2.text(0.95, y_offset, f"{label}: {final_val:.1f}%",
                        transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_offset -= 0.06
        else:
            final_vals = [last_valid(d['item_victim_pct']) for d in group]
            final_vals = [v for v in final_vals if v is not None]
            if final_vals:
                mean_final = np.mean(final_vals)
                min_final = np.min(final_vals)
                max_final = np.max(final_vals)
                ax2.text(0.95, y_offset, f"{label}: {mean_final:.1f}% [{min_final:.1f}-{max_final:.1f}]",
                        transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_offset -= 0.06

    # === Row 2: Reverse Engineering Metrics (if present) ===
    if active_re_metrics and axes_row2 is not None:
        for panel_idx, re_key in enumerate(active_re_metrics):
            ax = axes_row2[panel_idx]
            max_val = 0

            for i, label in enumerate(label_order):
                group = label_groups[label]
                color = label_colors[label]
                marker = markers[i % len(markers)]
                val = plot_group(ax, group, label, color, marker, 'reverse_engineering', nested_key=re_key)
                max_val = max(max_val, val)

            ax.set_xlabel('Turn', fontsize=11)
            ax.set_ylabel(RE_LABELS.get(re_key, re_key), fontsize=11)
            ax.set_title(RE_LABELS.get(re_key, re_key), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            # These are 0-1 scores
            ax.set_ylim(0, max(1.05, max_val * 1.1) if max_val > 0 else 1.05)

            annotate_final_values(ax, label_order, label_groups, label_colors,
                                  'reverse_engineering', nested_key=re_key)

    # Single legend at the bottom of the figure
    handles, labels_legend = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', ncol=min(len(label_order), 5),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08 if active_re_metrics else 0.15)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Multi-comparison plot saved to: {output_file}")
    plt.close()

    return output_file


def plot_comparison(macf_data: dict, connacf_data: dict, output_file: str, title: str = None):
    """
    Plot side-by-side comparison of User % and Item % victim rates.
    
    Layout: 1 row x 2 columns
    - Left: User Victim Rate (%)
    - Right: Item Victim Rate (%)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Victim Rate Comparison: MACF vs ConnaCF\n(% of Innocent Agents Contaminated)', 
                     fontsize=14, fontweight='bold')
    
    # Colors
    macf_color = '#2196F3'  # Blue
    connacf_color = '#FF5722'  # Orange-red
    
    # Helper to get non-NaN values for max calculation
    def safe_max(values, default=0):
        valid = [v for v in values if not np.isnan(v)]
        return max(valid) if valid else default
    
    # Helper to get last non-NaN value
    def last_valid(values):
        for v in reversed(values):
            if not np.isnan(v):
                return v
        return None
    
    # Helper to interpolate NaN values for continuous line plotting
    def interpolate_nans(turns, values):
        """Linearly interpolate NaN values to connect data points across gaps."""
        turns = np.array(turns)
        values = np.array(values)
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return turns, values, mask  # Not enough points to interpolate
        # Interpolate only between valid points (not extrapolate)
        interpolated = np.interp(turns, turns[mask], values[mask])
        # Keep NaN for points outside the range of valid data
        first_valid = np.where(mask)[0][0]
        last_valid_idx = np.where(mask)[0][-1]
        interpolated[:first_valid] = np.nan
        interpolated[last_valid_idx+1:] = np.nan
        return turns, interpolated, mask
    
    # === Left: User Victim Rate ===
    ax1 = axes[0]
    
    if macf_data['turns'] and macf_data['user_victim_pct']:
        turns_interp, values_interp, mask = interpolate_nans(macf_data['turns'], macf_data['user_victim_pct'])
        ax1.plot(turns_interp, values_interp, linewidth=2.2, color=macf_color,
                 label=macf_data['name'], alpha=0.75)
    
    if connacf_data['turns'] and connacf_data['user_victim_pct']:
        turns_interp, values_interp, mask = interpolate_nans(connacf_data['turns'], connacf_data['user_victim_pct'])
        ax1.plot(turns_interp, values_interp, linewidth=2.2, color=connacf_color,
                 label=connacf_data['name'], alpha=0.75)
    
    ax1.set_xlabel('Turn', fontsize=11)
    ax1.set_ylabel('User Victim Rate (%)', fontsize=11)
    ax1.set_title('User Contamination Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Calculate y-axis limit using only valid (non-NaN) values
    max_user = max(safe_max(macf_data['user_victim_pct']), safe_max(connacf_data['user_victim_pct']))
    ax1.set_ylim(0, max(100, max_user * 1.1) if max_user > 0 else 100)
    
    # Add final values annotation (use last valid value)
    final_macf_user = last_valid(macf_data['user_victim_pct'])
    if final_macf_user is not None:
        ax1.text(0.95, 0.85, f"Tree: {final_macf_user:.1f}%", 
                transform=ax1.transAxes, ha='right', va='top', fontsize=9,
                color=macf_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    final_connacf_user = last_valid(connacf_data['user_victim_pct'])
    if final_connacf_user is not None:
        ax1.text(0.95, 0.75, f"DenseMesh: {final_connacf_user:.1f}%", 
                transform=ax1.transAxes, ha='right', va='top', fontsize=9,
                color=connacf_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # === Right: Item Victim Rate ===
    ax2 = axes[1]
    
    if macf_data['turns'] and macf_data['item_victim_pct']:
        turns_interp, values_interp, mask = interpolate_nans(macf_data['turns'], macf_data['item_victim_pct'])
        ax2.plot(turns_interp, values_interp, linewidth=2.2, color=macf_color,
                 label=macf_data['name'], alpha=0.75)
    
    if connacf_data['turns'] and connacf_data['item_victim_pct']:
        turns_interp, values_interp, mask = interpolate_nans(connacf_data['turns'], connacf_data['item_victim_pct'])
        ax2.plot(turns_interp, values_interp, linewidth=2.2, color=connacf_color,
                 label=connacf_data['name'], alpha=0.75)
    
    ax2.set_xlabel('Turn', fontsize=11)
    ax2.set_ylabel('Item Victim Rate (%)', fontsize=11)
    ax2.set_title('Item Contamination Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Calculate y-axis limit using only valid (non-NaN) values
    max_item = max(safe_max(macf_data['item_victim_pct']), safe_max(connacf_data['item_victim_pct']))
    ax2.set_ylim(0, max(100, max_item * 1.1) if max_item > 0 else 100)
    
    # Add final values annotation (use last valid value)
    final_macf_item = last_valid(macf_data['item_victim_pct'])
    if final_macf_item is not None:
        ax2.text(0.95, 0.85, f"Tree: {final_macf_item:.1f}%", 
                transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                color=macf_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    final_connacf_item = last_valid(connacf_data['item_victim_pct'])
    if final_connacf_item is not None:
        ax2.text(0.95, 0.75, f"DenseMesh: {final_connacf_item:.1f}%", 
                transform=ax2.transAxes, ha='right', va='top', fontsize=9,
                color=connacf_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Single legend at the bottom of the figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, 
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend at bottom
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plot saved to: {output_file}")
    plt.close()
    
    return output_file


def plot_custom_metric(datasets: list, metric_name: str, output_file: str, title: str = None):
    """
    Plot a single custom metric per turn across multiple ConnaCF runs.
    Used for attack-specific metrics like ia_asr_valid, corba_blocking_rate, etc.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    else:
        fig.suptitle(f'{metric_name} per Turn', fontsize=13, fontweight='bold')
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    colors = plt.cm.tab10.colors
    
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
    
    # Group by label for band plotting
    label_groups = defaultdict(list)
    label_order = []
    for data in datasets:
        label = data['name']
        if label not in label_groups:
            label_order.append(label)
        label_groups[label].append(data)
    
    y_offset = 0.97
    for i, label in enumerate(label_order):
        group = label_groups[label]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        if len(group) == 1:
            data = group[0]
            values = data.get('custom_metric', [])
            if data['turns'] and values:
                t, v, mask = interpolate_nans(data['turns'], values)
                ax.plot(t, v, linewidth=2.2, color=color, label=label, alpha=0.75)
                final = v[mask][-1] if mask.any() else None
                if final is not None:
                    ax.text(0.97, y_offset, f"{label}: {final:.3f}",
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    y_offset -= 0.06
        else:
            # Band: mean ± range
            all_turns = sorted({t for d in group for t in d['turns']})
            n = len(all_turns)
            turn_idx = {t: i for i, t in enumerate(all_turns)}
            mat = np.full((len(group), n), np.nan)
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
    logger.info(f"Custom metric plot saved to: {output_file}")
    plt.close()
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Plot comparison of victim rates between experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare MACF vs ConnaCF (original mode)
  python tools/plot_comparison.py \\
      --macf_dir macf_output/cheat_50percent/CDs-100user-dense \\
      --connacf_dir attack_output/netsafe/misinfo_full/CDs-100user-dense/task_0

  # Compare multiple ConnaCF runs
  python tools/plot_comparison.py \\
      --connacf_dirs attack_output/netsafe/misinfo_sparse/CDs-100user-dense/task_0 \\
                     attack_output/netsafe/misinfo_full/CDs-100user-dense/task_0 \\
                     attack_output/netsafe/misinfo_dense/CDs-100user-dense/task_0

  # With custom labels
  python tools/plot_comparison.py \\
      --connacf_dirs attack_output/cheat/cheat_user_50percent_sparse/CDs-100user-dense/task_0 \\
                     attack_output/cheat/cheat_user_50percent/CDs-100user-dense/task_0 \\
                     attack_output/cheat/cheat_user_50percent_dense/CDs-100user-dense/task_0 \\
      --labels "Sparse" "Medium" "Dense"

  # Combined: MACF + multiple ConnaCF runs with labels
  python tools/plot_comparison.py \\
      --macf_dir ./macf_output/cheat_user_50percent/CDs-100user-dense \\
      --connacf_dirs attack_output/cheat_user/cheat_user_50percent_sparse/CDs-100user-dense/task_0 \\
                     attack_output/cheat_user/cheat_user_50percent/CDs-100user-dense/task_0 \\
                     attack_output/cheat_user/cheat_user_50percent_dense/CDs-100user-dense/task_0 \\
      --labels "Tree" "Sparse" "Medium" "Dense"
        """
    )
    
    # MACF vs ConnaCF mode (original)
    parser.add_argument('--macf_dir', type=str, default=None,
                        help='Path to MACF output directory (for MACF vs ConnaCF comparison)')
    parser.add_argument('--connacf_dir', type=str, default=None,
                        help='Path to single ConnaCF output directory (for MACF vs ConnaCF comparison)')
    
    # Multi-ConnaCF mode (new)
    parser.add_argument('--connacf_dirs', type=str, nargs='+', default=None,
                        help='Paths to multiple ConnaCF output directories (for multi-run comparison)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Custom labels for each ConnaCF run (must match number of --connacf_dirs)')
    parser.add_argument('--auto_label', action='store_true', default=False,
                        help='Auto-generate labels from metadata (num_items, num_candidates, attacker_ratio). '
                             'Appends metadata to custom labels if both provided.')
    
    parser.add_argument('--no_re_row', action='store_true', default=False,
                        help='Suppress the reverse-engineering metrics row even if RE data is present. '
                             'Use for dissemination-focused attacks (DrunkAgent, TOMA, MAMA, etc.) '
                             'where RE metrics are collected passively but not the focus.')
                        help='Output file path')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--last_epoch', type=int, default=100,
                        help='Trim all data to this many turns (default: 100)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Smoothing factor (0-1). 0=no smoothing, higher=more smoothing. '
                             'Uses exponential moving average. Recommended: 0.3-0.7')
    parser.add_argument('--metric', type=str, default=None,
                        help='Custom metric to plot instead of victim rates. '
                             'Supports dot-notation for nested keys, e.g. '
                             '"stealth.ia_asr_valid", "resources.corba_blocking_rate", '
                             '"dissemination.infection.pi_infection_rate_forward". '
                             'When set, plots a single panel with this metric per turn.')
    
    args = parser.parse_args()
    
    def smooth_values(values, alpha):
        """Apply exponential moving average smoothing.
        
        Args:
            values: List of values (may contain NaN)
            alpha: Smoothing factor (0-1). Higher = more smoothing.
                   EMA formula: smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * values[t]
        
        Returns:
            Smoothed values (same length as input)
        """
        if alpha <= 0 or not values:
            return values
        
        alpha = min(alpha, 0.99)  # Cap at 0.99
        smoothed = []
        last_valid = None
        
        for v in values:
            if np.isnan(v):
                smoothed.append(np.nan)
            elif last_valid is None:
                smoothed.append(v)
                last_valid = v
            else:
                # EMA: new = alpha * old + (1-alpha) * current
                new_val = alpha * last_valid + (1 - alpha) * v
                smoothed.append(new_val)
                last_valid = new_val
        
        return smoothed
    
    def trim_data(data: dict, last_epoch: int, smoothing: float = 0.0) -> dict:
        """Trim data to last_epoch turns and optionally apply smoothing."""
        if not data['turns'] or last_epoch is None:
            return data
        
        user_pct = data['user_victim_pct'][:last_epoch]
        item_pct = data['item_victim_pct'][:last_epoch]
        
        # Apply smoothing if requested
        if smoothing > 0:
            user_pct = smooth_values(user_pct, smoothing)
            item_pct = smooth_values(item_pct, smoothing)
        
        result = {
            'turns': data['turns'][:last_epoch],
            'user_victim_pct': user_pct,
            'item_victim_pct': item_pct,
            'name': data['name'],
            **{k: v for k, v in data.items() if k not in ['turns', 'user_victim_pct', 'item_victim_pct', 'name', 'reverse_engineering']}
        }
        
        # Trim reverse_engineering metrics if present
        if 'reverse_engineering' in data:
            result['reverse_engineering'] = {
                k: v[:last_epoch] for k, v in data['reverse_engineering'].items()
            }
        
        return result
    
    # Determine mode
    if args.connacf_dirs:
        # Multi-comparison mode (optionally with MACF)
        has_macf = args.macf_dir is not None
        if has_macf:
            logger.info("Mode: MACF + Multi-ConnaCF comparison")
        else:
            logger.info("Mode: Multi-ConnaCF comparison")
        
        datasets = []
        errors = []
        label_idx = 0  # Track label index separately
        
        # Load MACF data first if provided
        if has_macf:
            logger.info(f"Loading MACF data from: {args.macf_dir}")
            macf_data = load_macf_data(args.macf_dir)
            macf_data = trim_data(macf_data, args.last_epoch, args.smoothing)
            
            # Set custom label if provided
            if args.labels and label_idx < len(args.labels):
                base_label = args.labels[label_idx]
                label_idx += 1
                if args.auto_label:
                    macf_data['name'] = generate_auto_label(macf_data, base_label)
                else:
                    macf_data['name'] = base_label
            
            logger.info(f"  Found {len(macf_data['turns'])} turns, label: {macf_data['name']}")
            datasets.append(macf_data)
        
        # Load ConnaCF data
        for connacf_dir in args.connacf_dirs:
            logger.info(f"Loading ConnaCF data from: {connacf_dir}")
            data = load_connacf_data(connacf_dir, custom_metric=args.metric)
            data = trim_data(data, args.last_epoch, args.smoothing)
            
            # Set label: custom label, auto-label, or extract from path
            if args.labels and label_idx < len(args.labels):
                base_label = args.labels[label_idx]
                label_idx += 1
            else:
                base_label = extract_label_from_path(connacf_dir)
            
            # Apply auto-labeling if requested
            if args.auto_label:
                data['name'] = generate_auto_label(data, base_label)
            else:
                data['name'] = base_label
            
            # Check for errors
            if 'error' in data:
                errors.append(f"  {data['name']}: {data['error']}")
            
            logger.info(f"  Found {len(data['turns'])} turns, label: {data['name']}")
            datasets.append(data)
        
        # Show errors if any
        if errors:
            print("\n⚠️  WARNINGS:")
            for err in errors:
                print(err)
            print()
        
        # Filter out datasets with no data for plotting
        valid_datasets = [d for d in datasets if d['turns']]
        if not valid_datasets:
            logger.error("No valid datasets to plot!")
            print("\n❌ No valid data found in any of the specified directories.")
            print("   Make sure the experiments have been run and output exists.")
            return
        
        # Plot multi-comparison (or custom metric if --metric specified)
        if args.metric:
            plot_custom_metric(valid_datasets, args.metric, args.output, args.title)
        else:
            plot_multi_connacf_comparison(valid_datasets, args.output, args.title, show_re_row=not args.no_re_row)
        # Print summary
        print("\n" + "="*60)
        if has_macf:
            print("MACF + MULTI-CONNACF COMPARISON SUMMARY")
        else:
            print("MULTI-CONNACF COMPARISON SUMMARY")
        print("="*60)
        
        def last_valid(values):
            for v in reversed(values):
                if not np.isnan(v):
                    return v
            return None
        
        # Group datasets by label for summary
        label_groups = defaultdict(list)
        label_order = []
        for data in datasets:
            label = data['name']
            if label not in label_groups:
                label_order.append(label)
            label_groups[label].append(data)
        
        RE_SUMMARY_KEYS = ['ss_system_prompt', 'ss_task_instructions', 'extract_rate',
                           'f1_agent_count', 'f1_comm_density', 'gs_topology']
        
        for label in label_order:
            group = label_groups[label]
            
            if len(group) == 1:
                data = group[0]
                print(f"\n{label} ({len(data['turns'])} turns):")
                if 'error' in data:
                    print(f"  ⚠️  {data['error']}")
                    continue
                final_user = last_valid(data['user_victim_pct'])
                final_item = last_valid(data['item_victim_pct'])
                if final_user is not None:
                    print(f"  Final User Victim Rate: {final_user:.2f}%")
                else:
                    print(f"  Final User Victim Rate: N/A")
                if final_item is not None:
                    print(f"  Final Item Victim Rate: {final_item:.2f}%")
                else:
                    print(f"  Final Item Victim Rate: N/A")
                # Print reverse_engineering metrics if present
                re = data.get('reverse_engineering', {})
                if re:
                    print(f"  Reverse Engineering Metrics:")
                    for k in RE_SUMMARY_KEYS:
                        vals = re.get(k, [])
                        fv = last_valid(vals) if vals else None
                        if fv is not None:
                            print(f"    {k}: {fv:.4f}")
            else:
                # Multiple datasets with same label - show aggregated stats
                print(f"\n{label} (n={len(group)} runs, aggregated):")
                
                # Collect final values
                user_finals = []
                item_finals = []
                for data in group:
                    if 'error' not in data:
                        u = last_valid(data['user_victim_pct'])
                        i = last_valid(data['item_victim_pct'])
                        if u is not None:
                            user_finals.append(u)
                        if i is not None:
                            item_finals.append(i)
                
                if user_finals:
                    mean_u = np.mean(user_finals)
                    min_u = np.min(user_finals)
                    max_u = np.max(user_finals)
                    print(f"  Final User Victim Rate: {mean_u:.2f}% (range: {min_u:.2f}% - {max_u:.2f}%)")
                else:
                    print(f"  Final User Victim Rate: N/A")
                
                if item_finals:
                    mean_i = np.mean(item_finals)
                    min_i = np.min(item_finals)
                    max_i = np.max(item_finals)
                    print(f"  Final Item Victim Rate: {mean_i:.2f}% (range: {min_i:.2f}% - {max_i:.2f}%)")
                else:
                    print(f"  Final Item Victim Rate: N/A")
                
                # Print reverse_engineering metrics if any dataset has them
                has_re = any('reverse_engineering' in d for d in group)
                if has_re:
                    print(f"  Reverse Engineering Metrics:")
                    for k in RE_SUMMARY_KEYS:
                        re_finals = []
                        for d in group:
                            vals = d.get('reverse_engineering', {}).get(k, [])
                            fv = last_valid(vals) if vals else None
                            if fv is not None:
                                re_finals.append(fv)
                        if re_finals:
                            print(f"    {k}: {np.mean(re_finals):.4f} (range: {np.min(re_finals):.4f} - {np.max(re_finals):.4f})")
        
        print("="*60)
        
    elif args.macf_dir and args.connacf_dir:
        # Original MACF vs ConnaCF mode
        logger.info("Mode: MACF vs ConnaCF comparison")
        
        logger.info(f"Loading MACF data from: {args.macf_dir}")
        macf_data = load_macf_data(args.macf_dir)
        macf_data = trim_data(macf_data, args.last_epoch, args.smoothing)
        logger.info(f"  Found {len(macf_data['turns'])} turns")
        
        logger.info(f"Loading ConnaCF data from: {args.connacf_dir}")
        connacf_data = load_connacf_data(args.connacf_dir)
        connacf_data = trim_data(connacf_data, args.last_epoch, args.smoothing)
        logger.info(f"  Found {len(connacf_data['turns'])} turns")
        
        # Plot comparison
        plot_comparison(macf_data, connacf_data, args.output, args.title)
        
        # Print summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        def last_valid(values):
            for v in reversed(values):
                if not np.isnan(v):
                    return v
            return None
        
        print(f"\nMACF ({len(macf_data['turns'])} turns):")
        final_macf_user = last_valid(macf_data['user_victim_pct'])
        final_macf_item = last_valid(macf_data['item_victim_pct'])
        if final_macf_user is not None:
            print(f"  Final User Victim Rate: {final_macf_user:.2f}%")
        else:
            print(f"  Final User Victim Rate: N/A (no valid data)")
        if final_macf_item is not None:
            print(f"  Final Item Victim Rate: {final_macf_item:.2f}%")
        else:
            print(f"  Final Item Victim Rate: N/A (no valid data)")
        
        print(f"\nConnaCF ({len(connacf_data['turns'])} turns):")
        final_connacf_user = last_valid(connacf_data['user_victim_pct'])
        final_connacf_item = last_valid(connacf_data['item_victim_pct'])
        if final_connacf_user is not None:
            print(f"  Final User Victim Rate: {final_connacf_user:.2f}%")
        else:
            print(f"  Final User Victim Rate: N/A (no valid data)")
        if final_connacf_item is not None:
            print(f"  Final Item Victim Rate: {final_connacf_item:.2f}%")
        else:
            print(f"  Final Item Victim Rate: N/A (no valid data)")
        print("="*60)
    
    else:
        parser.error("Must provide either --connacf_dirs for multi-run comparison, "
                     "or both --macf_dir and --connacf_dir for MACF vs ConnaCF comparison")


if __name__ == "__main__":
    main()
