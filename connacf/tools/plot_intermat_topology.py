#!/usr/bin/env python3
"""
Generate a PDF figure showing three schematic interaction matrices
(Dense / Medium / Sparse) as small illustrative heatmaps.

Each panel is a miniature user×item binary matrix whose fill pattern
visually conveys the density contrast — analogous to the full
interaction_matrix_pair_*.png plots but compact enough for a paper figure.
Edge/fill colours match the three InterMat green shades used in metric plots.

Usage:
    python tools/plot_intermat_topology.py
    python tools/plot_intermat_topology.py --output path/to/out.pdf
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_DEFAULT_OUT = os.path.join(_REPO_ROOT, "figure", "intermat_topology.pdf")

# ---------------------------------------------------------------------------
# InterMat green palette — identical to _plot_common.get_color_for_label
# ---------------------------------------------------------------------------
def _hls(h_deg, l, s):
    return colorsys.hls_to_rgb(h_deg / 360, l, s)

# Exact values from _plot_common.get_color_for_label — legacy keyword branch
GREEN_DENSE  = (0x1A/255, 0x5C/255, 0x3E/255)   # #1A5C3E dark
GREEN_MEDIUM = (0x47/255, 0x8D/255, 0x7E/255)   # #478D7E mid
GREEN_SPARSE = (0x6A/255, 0xCB/255, 0xB7/255)   # #6ACBB7 light

# ---------------------------------------------------------------------------
# Schematic matrix data
# Each variant: (n_users, n_items, approximate fill fraction)
# We generate a reproducible pseudo-random binary matrix that looks right.
# ---------------------------------------------------------------------------
# Total filled cells is constant — density drops as catalog widens.
_N_U      = 12
_N_FILLED = int(_N_U * 15 * 0.55)   # ~99 filled cells in all three panels

VARIANTS = [
    dict(label="Sparse", color=GREEN_SPARSE, n_u=_N_U, n_i=55, density=_N_FILLED / (_N_U * 55)),
    dict(label="Medium", color=GREEN_MEDIUM, n_u=_N_U, n_i=30, density=_N_FILLED / (_N_U * 30)),
    dict(label="Dense",  color=GREEN_DENSE,  n_u=_N_U, n_i=15, density=_N_FILLED / (_N_U * 15)),
]


def _make_matrix(n_u, n_i, density, seed=42):
    """
    Generate a schematic binary interaction matrix.
    Users are sorted by degree (descending) so the top-left corner is
    densest — matching the visual convention of the real plots.
    """
    rng = np.random.default_rng(seed)
    # Give each user a degree drawn from a power-law-ish distribution
    # so high-degree users cluster at the top.
    degrees = np.sort(rng.integers(
        max(1, int(n_i * density * 0.3)),
        max(2, int(n_i * density * 1.8)),
        size=n_u
    ))[::-1]
    degrees = np.clip(degrees, 1, n_i)

    mat = np.zeros((n_u, n_i), dtype=np.uint8)
    for u, deg in enumerate(degrees):
        # Bias towards low item indices (popular items) for realism
        weights = np.exp(-np.arange(n_i) / (n_i * 0.35))
        weights /= weights.sum()
        chosen = rng.choice(n_i, size=int(deg), replace=False, p=weights)
        mat[u, chosen] = 1
    return mat


def _green_cmap(base_color):
    """White → base_color colormap for the heatmap cells."""
    return LinearSegmentedColormap.from_list(
        "green_intermat", ["#ffffff", base_color], N=2
    )


def main():
    parser = argparse.ArgumentParser(description="InterMat schematic PDF")
    parser.add_argument("--output", "-o", default=_DEFAULT_OUT)
    args = parser.parse_args()

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n = len(VARIANTS)
    # Each cell is CELL_SIZE inches square — panels scale with n_i / n_u
    CELL = 0.055   # inches per cell
    PAD  = 0.35    # extra inches per panel (title + labels)
    GAP  = 0.18    # separation between panels in inches

    panel_widths  = [v["n_i"] * CELL + PAD for v in VARIANTS]
    panel_heights = [v["n_u"] * CELL + PAD for v in VARIANTS]
    fig_w = sum(panel_widths) + GAP * (len(VARIANTS) - 1) + 0.1
    fig_h = max(panel_heights) + 0.1

    fig = plt.figure(figsize=(fig_w, fig_h))

    x_cursor = 0.0
    axes = []
    for v, pw, ph in zip(VARIANTS, panel_widths, panel_heights):
        left   = x_cursor / fig_w
        width  = pw / fig_w
        height = ph / fig_h
        bottom = (fig_h - ph) / 2 / fig_h
        ax = fig.add_axes([left, bottom, width, height])
        axes.append(ax)
        x_cursor += pw + GAP

    for ax, v in zip(axes, VARIANTS):
        mat = _make_matrix(v["n_u"], v["n_i"], v["density"])
        cmap = _green_cmap(v["color"])

        ax.imshow(mat, aspect="auto", cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest")

        # Thin grid lines to separate cells
        ax.set_xticks(np.arange(-0.5, v["n_i"], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, v["n_u"], 1), minor=True)
        ax.grid(which="minor", color="#cccccc", linewidth=0.3)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        # Axis labels: U (left) and I (bottom) only — no ticks
        ax.set_ylabel("U", fontsize=7, labelpad=-4, rotation=0, va="center",
                      color="#444444", ha="right")
        ax.set_xlabel("I", fontsize=7, labelpad=2, color="#444444")

        # Panel title in matching green
        ax.set_title(v["label"], fontsize=7, fontweight="bold",
                     color=v["color"], pad=3)

        # Thin coloured border matching the green shade
        for spine in ax.spines.values():
            spine.set_edgecolor(v["color"])
            spine.set_linewidth(1.2)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
