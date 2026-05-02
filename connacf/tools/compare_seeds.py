#!/usr/bin/env python3
"""
Compare seed-42 / seed-43 / seed-44 variants of ml-100k-100user-medium100.

Outputs:
  connacf/tools/interaction_matrix_seeds.png  — 3-panel interaction matrix
  (metrics printed to stdout)
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr

DATASET_DIR = Path(__file__).parent.parent / "dataset"
OUT_PNG = Path(__file__).parent / "interaction_matrix_seeds.png"

SEEDS = [
    ("seed42", "ml-100k-100user-medium100-seed42"),
    ("seed43", "ml-100k-100user-medium100-seed43"),
    ("seed44", "ml-100k-100user-medium100-seed44"),
]


# ── data loading ────────────────────────────────────────────────────────────

def load_interactions(dataset_name: str):
    """Return user_items dict and item_users dict from train+valid+test."""
    base = DATASET_DIR / dataset_name
    user_items: dict[int, set[int]] = defaultdict(set)
    for suffix in [".train.inter", ".valid.inter", ".test.inter"]:
        for f in base.iterdir():
            if f.name.endswith(suffix):
                with open(f) as fh:
                    next(fh)
                    for line in fh:
                        parts = line.strip().split("\t")
                        if len(parts) < 3:
                            continue
                        uid = int(parts[0])
                        for tok in parts[1].split():
                            user_items[uid].add(int(tok))
                        user_items[uid].add(int(parts[2]))
    item_users: dict[int, set[int]] = defaultdict(set)
    for u, items in user_items.items():
        for i in items:
            item_users[i].add(u)
    return dict(user_items), dict(item_users)


def build_matrix(user_items, item_users):
    users = sorted(user_items, key=lambda u: len(user_items[u]))
    items = sorted(item_users, key=lambda i: len(item_users[i]), reverse=True)
    uidx = {u: r for r, u in enumerate(users)}
    iidx = {i: c for c, i in enumerate(items)}
    mat = np.zeros((len(users), len(items)), dtype=np.float32)
    for u, its in user_items.items():
        for i in its:
            if i in iidx:
                mat[uidx[u], iidx[i]] = 1.0
    return mat, users, items


# ── similarity metrics ───────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def edge_jaccard(ui_a: dict, ui_b: dict) -> float:
    edges_a = {(u, i) for u, its in ui_a.items() for i in its}
    edges_b = {(u, i) for u, its in ui_b.items() for i in its}
    return jaccard(edges_a, edges_b)


def degree_corr(ui_a: dict, ui_b: dict, kind="user") -> tuple[float, float]:
    """Spearman correlation of degree vectors over the union of entities."""
    if kind == "user":
        keys = sorted(set(ui_a) | set(ui_b))
        da = [len(ui_a.get(k, set())) for k in keys]
        db = [len(ui_b.get(k, set())) for k in keys]
    else:  # item
        # build item_users from user_items
        iu_a: dict[int, set] = defaultdict(set)
        for u, its in ui_a.items():
            for i in its:
                iu_a[i].add(u)
        iu_b: dict[int, set] = defaultdict(set)
        for u, its in ui_b.items():
            for i in its:
                iu_b[i].add(u)
        keys = sorted(set(iu_a) | set(iu_b))
        da = [len(iu_a.get(k, set())) for k in keys]
        db = [len(iu_b.get(k, set())) for k in keys]
    r, p = spearmanr(da, db)
    return float(r), float(p)


# ── plotting ─────────────────────────────────────────────────────────────────

CMAP = LinearSegmentedColormap.from_list("interact", ["#f5f5f5", "#1565c0", "#0d47a1"], N=3)
COLORS = ["#1A5C3E", "#478D7E", "#6ACBB7"]  # dark→light green per seed


def plot_panel(fig, subplot_spec, mat, label, color, stats_text):
    n_u, n_i = mat.shape
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=subplot_spec,
        width_ratios=[6, 1], height_ratios=[1.2, 5],
        wspace=0.04, hspace=0.06,
    )
    ax_top   = fig.add_subplot(inner[0, 0])
    ax_main  = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])
    ax_stat  = fig.add_subplot(inner[0, 1])

    ax_main.imshow(mat, aspect="auto", cmap=CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax_main.set_xlabel(f"Items (catalog = {n_i})", fontsize=8)
    ax_main.set_ylabel("Users (sorted by degree →)", fontsize=8)
    ax_main.tick_params(labelsize=6)
    ax_main.set_xticks(np.linspace(0, n_i - 1, min(6, n_i)).astype(int))
    ax_main.set_yticks(np.linspace(0, n_u - 1, min(6, n_u)).astype(int))

    # column-density overlay
    window = max(1, n_i // 15)
    col_d = np.convolve(mat.mean(axis=0), np.ones(window) / window, mode="same")
    ax2 = ax_main.twinx()
    ax2.fill_between(range(n_i), col_d, color="#ff6f00", alpha=0.18)
    ax2.plot(range(n_i), col_d, color="#e65100", lw=0.8, alpha=0.6)
    ax2.set_ylim(0, max(col_d.max() * 2.5, 0.01))
    ax2.set_yticks([])
    ax2.set_xlim(-0.5, n_i - 0.5)

    ax_top.bar(range(n_i), mat.sum(axis=0), width=1.0, color="#1565c0", alpha=0.7)
    ax_top.set_xlim(-0.5, n_i - 0.5)
    ax_top.set_ylabel("Pop", fontsize=7)
    ax_top.tick_params(labelsize=5, labelbottom=False)
    ax_top.set_title(label, fontsize=11, fontweight="bold", pad=4, color=color)

    ax_right.barh(range(n_u), mat.sum(axis=1), height=1.0, color="#b71c1c", alpha=0.65)
    ax_right.set_ylim(-0.5, n_u - 0.5)
    ax_right.set_xlabel("Deg", fontsize=7)
    ax_right.tick_params(labelsize=5, labelleft=False)
    ax_right.invert_yaxis()

    ax_stat.axis("off")
    ax_stat.text(0.5, 0.5, stats_text, ha="center", va="center",
                 fontsize=7.5, family="monospace", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#666", lw=0.8, alpha=0.9))


# ── main ─────────────────────────────────────────────────────────────────────

def load_random_negatives(dataset_name: str) -> dict[int, list[int]]:
    """Load .random file: user_id -> list of negative item ids."""
    base = DATASET_DIR / dataset_name
    result = {}
    for f in base.iterdir():
        if f.name.endswith(".random"):
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        uid = int(parts[0])
                        result[uid] = [int(x) for x in parts[1].split()]
    return result


def load_binary_candidates(dataset_name: str) -> dict:
    """Load binary_train_1cand.json."""
    import json
    p = DATASET_DIR / dataset_name / "binary_train_1cand.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)["data"]


def load_ranking_candidates(dataset_name: str, fname="ranking_train_3cand.json") -> dict:
    """Load ranking_train_3cand.json."""
    import json
    p = DATASET_DIR / dataset_name / fname
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)["data"]


def neg_jaccard(neg_a: dict, neg_b: dict) -> float:
    """Mean per-user Jaccard of negative item sets."""
    users = set(neg_a) & set(neg_b)
    if not users:
        return 0.0
    scores = [jaccard(set(neg_a[u]), set(neg_b[u])) for u in users]
    return float(np.mean(scores))


def ranking_candidate_jaccard(rc_a: dict, rc_b: dict) -> float:
    """Mean per-user Jaccard of candidate item sets across training samples."""
    users = set(rc_a) & set(rc_b)
    if not users:
        return 0.0
    scores = []
    for u in users:
        items_a = {int(x) for sample in rc_a[u] for x in sample}
        items_b = {int(x) for sample in rc_b[u] for x in sample}
        scores.append(jaccard(items_a, items_b))
    return float(np.mean(scores))


def main():
    datasets = {}
    for label, name in SEEDS:
        print(f"Loading {name} …")
        ui, iu = load_interactions(name)
        neg = load_random_negatives(name)
        rc  = load_ranking_candidates(name)
        bc  = load_binary_candidates(name)
        datasets[label] = (ui, iu, neg, rc, bc, name)

    labels = [s[0] for s in SEEDS]
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))]

    # ── interaction graph similarity ────────────────────────────────────────
    print("\n── Interaction Graph Similarity (identical across seeds) ───────")
    print(f"{'Pair':<20} {'UserJ':>7} {'ItemJ':>7} {'EdgeJ':>7} {'ItemDegρ':>9}")
    print("-" * 58)
    for la, lb in pairs:
        ui_a, iu_a, *_ = datasets[la]
        ui_b, iu_b, *_ = datasets[lb]
        user_j = jaccard(set(ui_a), set(ui_b))
        item_j = jaccard(set(iu_a), set(iu_b))
        edge_j = edge_jaccard(ui_a, ui_b)
        ideg_r, _ = degree_corr(ui_a, ui_b, "item")
        print(f"{la}↔{lb:<12} {user_j:>7.4f} {item_j:>7.4f} {edge_j:>7.4f} {ideg_r:>9.4f}")

    # ── negative sampling similarity ────────────────────────────────────────
    print("\n── Negative Sampling Similarity (.random file) ─────────────────")
    print(f"{'Pair':<20} {'NegJ(mean/user)':>16} {'NegJ(global)':>13}")
    print("-" * 52)
    for la, lb in pairs:
        _, _, neg_a, _, _, _ = datasets[la]
        _, _, neg_b, _, _, _ = datasets[lb]
        mean_j = neg_jaccard(neg_a, neg_b)
        # global: treat all negatives as one flat set per user, then average
        global_neg_a = {i for negs in neg_a.values() for i in negs}
        global_neg_b = {i for negs in neg_b.values() for i in negs}
        glob_j = jaccard(global_neg_a, global_neg_b)
        print(f"{la}↔{lb:<12} {mean_j:>16.4f} {glob_j:>13.4f}")

    # ── ranking candidate similarity ────────────────────────────────────────
    print("\n── Training Candidate Similarity (ranking_train_3cand.json) ────")
    print(f"{'Pair':<20} {'CandJ(mean/user)':>17}")
    print("-" * 40)
    for la, lb in pairs:
        _, _, _, rc_a, _, _ = datasets[la]
        _, _, _, rc_b, _, _ = datasets[lb]
        cj = ranking_candidate_jaccard(rc_a, rc_b)
        print(f"{la}↔{lb:<12} {cj:>17.4f}")

    print("\n── Training Candidate Similarity (binary_train_1cand.json) ─────")
    print(f"{'Pair':<20} {'CandJ(mean/user)':>17}")
    print("-" * 40)
    for la, lb in pairs:
        _, _, _, _, bc_a, _ = datasets[la]
        _, _, _, _, bc_b, _ = datasets[lb]
        cj = ranking_candidate_jaccard(bc_a, bc_b)
        print(f"{la}↔{lb:<12} {cj:>17.4f}")

    # ── per-dataset basic stats ─────────────────────────────────────────────
    print("\n── Dataset Stats ───────────────────────────────────────────────")
    print(f"{'Label':<10} {'Users':>6} {'Items':>6} {'Edges':>7} {'Density':>9} {'AvgUDeg':>8} {'AvgIDeg':>8}")
    print("-" * 60)
    mats = {}
    for label, (ui, iu, neg, rc, bc, name) in datasets.items():
        n_u = len(ui); n_i = len(iu)
        edges = sum(len(v) for v in ui.values())
        density = edges / (n_u * n_i) if n_u * n_i else 0
        avg_u = edges / n_u if n_u else 0
        avg_i = edges / n_i if n_i else 0
        print(f"{label:<10} {n_u:>6} {n_i:>6} {edges:>7} {density:>9.5f} {avg_u:>8.2f} {avg_i:>8.2f}")
        mat, users, items = build_matrix(ui, iu)
        mats[label] = (mat, f"{n_u}u × {n_i}i\nρ={density:.4f}\nnnz={edges}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 5.5), dpi=150)
    outer = gridspec.GridSpec(1, 3, wspace=0.12, figure=fig)

    for col, (label, color) in enumerate(zip(labels, COLORS)):
        mat, stats_text = mats[label]
        plot_panel(fig, outer[col], mat, label, color, stats_text)

    fig.suptitle("ml-100k-100user-medium100 — Seed 42 / 43 / 44 Interaction Matrices",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nPlot saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
