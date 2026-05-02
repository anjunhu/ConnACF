#!/usr/bin/env python3
"""
Plot MASLeak reverse engineering metrics and pattern-based TIVS over time.

Includes:
- Reverse engineering similarity scores (static IP extraction)
- U-I topology recall/precision/F1 (item set reconstruction)
- Mean Jaccard Index (mIoU) of inferred vs ground-truth item sets per user
- Pattern-based TIVS contamination scores
- System performance

Usage:
    python tools/plot_masleak_reverse_engineering.py <run_dir> [--output plot.png]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_turns(run_dir):
    pattern = os.path.join(run_dir, "turn_*.json")
    files = sorted(glob.glob(pattern),
                   key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    turns = []
    for f in files:
        with open(f) as fh:
            turns.append(json.load(fh))
    return turns


def compute_mean_jaccard(extracted_ip, n_items):
    """
    Compute mean Jaccard Index (mIoU) across users.
    
    For each user, Jaccard = |inferred ∩ catalog| / |inferred ∪ catalog|.
    Since we don't have per-user ground truth interaction history in the JSON,
    we approximate: the "universe" for each user is the full item catalog (n_items).
    
    But more usefully, we compute pairwise Jaccard between users' inferred sets
    to measure how much the attacker has differentiated individual users' tastes.
    
    Returns:
        mean_jaccard: Mean per-user |inferred_ids| / |all_item_ids| (coverage)
        mean_pairwise_jaccard: Mean Jaccard between all user pairs (differentiation)
        per_user_coverage: List of per-user coverage fractions
    """
    mentioned_ids = extracted_ip.get('mentioned_item_ids_per_user', {})
    if not mentioned_ids:
        return 0.0, 0.0, []
    
    all_item_ids = set(range(1, n_items + 1))  # Items are 1-indexed
    
    # Per-user coverage: fraction of catalog inferred
    per_user_coverage = []
    user_sets = {}
    for uid, ids in mentioned_ids.items():
        id_set = set(ids) if isinstance(ids, list) else ids
        user_sets[uid] = id_set
        coverage = len(id_set) / len(all_item_ids) if all_item_ids else 0
        per_user_coverage.append(coverage)
    
    mean_coverage = np.mean(per_user_coverage) if per_user_coverage else 0.0
    
    # Pairwise Jaccard: how differentiated are the inferred user profiles?
    uids = sorted(user_sets.keys())
    pairwise = []
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            a, b = user_sets[uids[i]], user_sets[uids[j]]
            union = len(a | b)
            inter = len(a & b)
            pairwise.append(inter / union if union > 0 else 0)
    
    mean_pairwise = np.mean(pairwise) if pairwise else 0.0
    
    return float(mean_coverage), float(mean_pairwise), per_user_coverage


def extract_series(turns, n_items=20):
    """Extract time series from turn data."""
    data = {
        "turn": [],
        # Reverse engineering (static IP)
        "ss_system_prompt": [], "ss_task_instructions": [],
        "f1_agent_count": [], "f1_comm_density": [],
        "gs_topology": [], "extract_rate": [],
        "num_responses": [], "num_ip_fields": [],
        # U-I topology (item set reconstruction)
        "ui_recall": [], "ui_precision": [], "ui_f1": [],
        "ui_mean": [], "ui_leak_rate": [],
        "ui_items_leaked": [], "ui_items_total": [],
        "total_mentioned": [],
        # Jaccard / coverage
        "mean_catalog_coverage": [],
        "mean_pairwise_jaccard": [],
        # Pattern detection (TIVS)
        "tivs_user_mean": [], "tivs_item_mean": [],
        "tivs_user_max": [], "tivs_item_max": [],
        # System performance
        "accuracy": [], "ndcg_at_10": [],
    }

    for t in turns:
        data["turn"].append(t["turn"])

        re = t.get("reverse_engineering", {})
        data["ss_system_prompt"].append(re.get("ss_system_prompt", 0))
        data["ss_task_instructions"].append(re.get("ss_task_instructions", 0))
        data["f1_agent_count"].append(re.get("f1_agent_count", 0))
        data["f1_comm_density"].append(re.get("f1_comm_density", 0))
        data["gs_topology"].append(re.get("gs_topology", 0))
        data["extract_rate"].append(re.get("extract_rate", 0))
        data["num_responses"].append(re.get("num_responses_collected", 0))
        data["num_ip_fields"].append(re.get("num_ip_fields_extracted", 0))

        # U-I topology metrics
        data["ui_recall"].append(re.get("ui_topology_recall", 0))
        data["ui_precision"].append(re.get("ui_topology_precision", 0))
        data["ui_f1"].append(re.get("ui_topology_f1", 0))
        data["ui_mean"].append(re.get("ui_topology_mean", 0))
        data["ui_leak_rate"].append(re.get("ui_topology_leak_rate", 0))
        data["ui_items_leaked"].append(re.get("ui_items_leaked", 0))
        data["ui_items_total"].append(re.get("ui_items_total", 0))

        # Count total mentioned items
        extracted_ip = re.get("extracted_ip", {})
        mentioned = extracted_ip.get("mentioned_items_per_user", {})
        data["total_mentioned"].append(sum(len(v) for v in mentioned.values()))

        # Jaccard / coverage
        cov, pw_jacc, _ = compute_mean_jaccard(extracted_ip, n_items)
        data["mean_catalog_coverage"].append(cov)
        data["mean_pairwise_jaccard"].append(pw_jacc)

        # Pattern detection
        pd = t.get("dissemination", {}).get("pattern_detection", {})
        pd_u = pd.get("user", {})
        pd_i = pd.get("item", {})
        data["tivs_user_mean"].append(pd_u.get("mean_tivs", 0))
        data["tivs_item_mean"].append(pd_i.get("mean_tivs", 0))
        data["tivs_user_max"].append(pd_u.get("max_tivs", 0))
        data["tivs_item_max"].append(pd_i.get("max_tivs", 0))

        sp = t.get("system_performance", {})
        data["accuracy"].append(sp.get("accuracy", 0))
        data["ndcg_at_10"].append(sp.get("ndcg_at_10", 0))

    return data


def plot(data, output_path, run_dir):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    turns = np.array(data["turn"])

    parts = run_dir.rstrip("/").split("/")
    title_parts = [p for p in parts if "masleak" in p.lower() or "ml-" in p.lower()]
    fig.suptitle(
        f"MASLeak Reverse Engineering — {' / '.join(title_parts) or run_dir}",
        fontsize=14, fontweight="bold",
    )

    # ── Row 1, Col 1: Static IP Extraction ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(turns, data["ss_system_prompt"], "-", label="SS(system_prompt)", color="#2196F3", linewidth=2.2, alpha=0.75)
    ax1.plot(turns, data["ss_task_instructions"], "-", label="SS(task_instr)", color="#FF9800", linewidth=2.2, alpha=0.75)
    ax1.plot(turns, data["extract_rate"], "-", label="Extract Rate", color="#4CAF50", linewidth=2.2, alpha=0.75)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Score")
    ax1.set_title("Static IP Extraction")
    ax1.legend(fontsize=8); ax1.set_ylim(-0.05, 1.05); ax1.grid(True, alpha=0.3)

    # ── Row 1, Col 2: U-I Topology Reconstruction ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(turns, data["ui_recall"], "-", label="Recall", color="#4CAF50", linewidth=2.2, alpha=0.75)
    ax2.plot(turns, data["ui_precision"], "-", label="Precision", color="#F44336", linewidth=2.2, alpha=0.75)
    ax2.plot(turns, data["ui_f1"], "-", label="F1", color="#9C27B0", linewidth=2.2, alpha=0.75)
    ax2_r = ax2.twinx()
    ax2_r.plot(turns, data["total_mentioned"], "--", label="Total items mentioned",
               color="#795548", linewidth=1.5, alpha=0.5)
    ax2_r.set_ylabel("# Items mentioned", color="#795548", fontsize=9)
    ax2.set_xlabel("Turn"); ax2.set_ylabel("Score")
    ax2.set_title("U-I Topology: Recall / Precision / F1")
    ax2.legend(loc="center left", fontsize=8)
    ax2_r.legend(loc="center right", fontsize=8)
    ax2.set_ylim(-0.05, 1.05); ax2.grid(True, alpha=0.3)

    # ── Row 1, Col 3: Catalog Coverage & Pairwise Jaccard ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(turns, data["mean_catalog_coverage"], "-",
             label="Mean catalog coverage", color="#2196F3", linewidth=2.2, alpha=0.75)
    ax3.plot(turns, data["mean_pairwise_jaccard"], "-",
             label="Mean pairwise Jaccard", color="#FF5722", linewidth=2.2, alpha=0.75)
    ax3.set_xlabel("Turn"); ax3.set_ylabel("Score")
    ax3.set_title("Interaction History Reconstruction")
    ax3.legend(fontsize=8); ax3.set_ylim(-0.05, 1.05); ax3.grid(True, alpha=0.3)

    # ── Row 2, Col 1: Pattern-Based TIVS ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(turns, data["tivs_user_mean"], "-", label="User mean TIVS", color="#2196F3", linewidth=2.2, alpha=0.75)
    ax4.plot(turns, data["tivs_item_mean"], "-", label="Item mean TIVS", color="#FF5722", linewidth=2.2, alpha=0.75)
    ax4.plot(turns, data["tivs_user_max"], "--", label="User max TIVS", color="#2196F3", linewidth=1.5, alpha=0.4)
    ax4.plot(turns, data["tivs_item_max"], "--", label="Item max TIVS", color="#FF5722", linewidth=1.5, alpha=0.4)
    ax4.set_xlabel("Turn"); ax4.set_ylabel("TIVS Score")
    ax4.set_title("Pattern-Based Contamination (TIVS)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # ── Row 2, Col 2: System Performance ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(turns, data["accuracy"], "-", label="Accuracy", color="#4CAF50", linewidth=2.2, alpha=0.75)
    ax5.plot(turns, data["ndcg_at_10"], "-", label="NDCG@10", color="#FF9800", linewidth=2.2, alpha=0.75)
    ax5.set_xlabel("Turn"); ax5.set_ylabel("Score")
    ax5.set_title("System Performance"); ax5.legend(fontsize=8)
    ax5.set_ylim(0, 1.05); ax5.grid(True, alpha=0.3)

    # ── Row 2, Col 3: Structural Inference ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(turns, data["f1_agent_count"], "-", label="F1(agent_count)", color="#9C27B0", linewidth=2.2, alpha=0.75)
    ax6.plot(turns, data["f1_comm_density"], "-", label="F1(comm_density)", color="#E91E63", linewidth=2.2, alpha=0.75)
    ax6.plot(turns, data["gs_topology"], "-", label="GS(topology)", color="#607D8B", linewidth=2.2, alpha=0.75)
    ax6_r = ax6.twinx()
    ax6_r.plot(turns, data["num_responses"], "--", label="Responses collected",
               color="#795548", linewidth=1.5, alpha=0.6)
    ax6_r.set_ylabel("# Responses", color="#795548", fontsize=9)
    ax6.set_xlabel("Turn"); ax6.set_ylabel("Score")
    ax6.set_title("Structural Inference")
    ax6.legend(loc="upper left", fontsize=8)
    ax6_r.legend(loc="upper right", fontsize=8)
    ax6.set_ylim(-0.05, 1.05); ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot MASLeak reverse engineering metrics")
    parser.add_argument("run_dir", help="Path to run directory with turn_*.json files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path (default: <run_dir>/masleak_reverse_engineering.png)")
    parser.add_argument("--n-items", type=int, default=20,
                        help="Total items in catalog (for coverage computation)")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: {args.run_dir} is not a directory"); sys.exit(1)

    turns = load_turns(args.run_dir)
    if not turns:
        print(f"No turn_*.json files found in {args.run_dir}"); sys.exit(1)

    print(f"Loaded {len(turns)} turns from {args.run_dir}")
    data = extract_series(turns, n_items=args.n_items)
    output = args.output or os.path.join(args.run_dir, "masleak_reverse_engineering.png")
    plot(data, output, args.run_dir)


if __name__ == "__main__":
    main()
