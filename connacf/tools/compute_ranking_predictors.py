#!/usr/bin/env python3
"""
Compute empirical Utr/Uss/Itr/Iss from attack output runs, then compute
static graph-structural predictors (E_U, E_I, R_U, R_I) and
report Kendall-tau rank correlation between predictors and empirical outcomes.

Transient (tr) vs steady-state (ss) split:
  - Fit a piecewise linear model (two segments) to the metric time series.
  - The breakpoint is the turn that minimises total residual (grid search).
  - tr = mean slope of segment 1 (turns 0..bp); ss = mean of segment 2 (turns bp..end).
  - Hyperparams: TR_MIN_TURNS (min turns before breakpoint), TR_MAX_FRAC (max fraction
    of run for transient), SMOOTH_WINDOW (Savitzky-Golay window for slope estimation).

Usage:
  python compute_ranking_predictors.py --config ranking_pred_config.yaml
  python compute_ranking_predictors.py  # uses built-in RUNS table

Output:
  - ranking_predictors_raw.csv   : per-run raw numbers
  - ranking_predictors_corr.csv  : Kendall-tau correlation table
  - ranking_predictors.txt       : human-readable summary
"""

import argparse
import json
import glob
import os
import math
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import minimize

# Single source of truth for the §7 SIS model and the coupled predictors.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sis_predictors as sp  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (override via --config yaml or CLI flags)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    TR_MIN_TURNS=3,        # minimum turns in transient segment
    TR_MAX_FRAC=0.6,       # max fraction of run length for transient
    SMOOTH_WINDOW=3,       # rolling mean window for slope estimation (odd int)
    ALPHA=0.25,            # attacker ratio (for R, E_U, E_I)
    GAMMA=0.1,             # recovery rate per benign contact (for R)
    R_LLM=0.05,            # LLM self-cleansing rate (for R)
    OUTPUT_DIR="tools/ranking_pred_output",
    BASE_DIR="attack_output",
)

# ---------------------------------------------------------------------------
# Metric extraction: which JSON path to use per attack family
# ---------------------------------------------------------------------------
# Each entry: (user_metric_path, item_metric_path)
# Path is dot-separated key into the turn JSON.
METRIC_PATHS = {
    "dissem":        ("llm_judge.user._asr",                     "llm_judge.item._asr"),
    "corba":         ("resources.corba_infection_rate_memory",    "resources.corba_infection_rate_memory_item"),
    "mama":          ("privacy.pii_leakage_rate",                 None),
    "masleak":       ("reverse_engineering.ui_topology_f1",       None),
    "masleak_recall":("reverse_engineering.ui_topology_recall",   None),
    "master_extract":("reverse_engineering.master.ss_system_prompt", None),
    "toma_extract":  ("reverse_engineering.toma.node_coverage",   None),
    "extract": ("privacy.pii_leakage_rate", None),
    "master":  ("reverse_engineering.master.asr", "reverse_engineering.extract_rate"),
}

# ---------------------------------------------------------------------------
# Run table — pinned to the exact directories used in plotting.sh.
# Format: (family, metric_type, label, dataset, run_dir, k_UI, n_items)
# ---------------------------------------------------------------------------
RUNS = [
    # ── Dissem (U): NetSafe Qwen3 25pct — UIDensity ──────────────────────
    ("netsafe", "dissem", "1c_medium100", "medium100",
     "netsafe/misinfo_1cand/ml-100k-100-user-medium100/260320125807",        1, 100),
    ("netsafe", "dissem", "1c_medium100", "medium100",
     "netsafe/misinfo_1cand/ml-100k-100-user-medium100-seed43/260327094928", 1, 100),
    ("netsafe", "dissem", "1c_medium100", "medium100",
     "netsafe/misinfo_1cand/ml-100k-100-user-medium100/260325160149",        1, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260320125814",        2, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260328132026",        2, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260331122849",        2, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260325142936",        2, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100-seed43/260324165825", 2, 100),
    ("netsafe", "dissem", "2c_medium100", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100-seed44/260324165817", 2, 100),
    ("netsafe", "dissem", "3c_medium100", "medium100",
     "netsafe/misinfo_3cand/ml-100k-100-user-medium100/260320125759",        3, 100),
    ("netsafe", "dissem", "3c_medium100", "medium100",
     "netsafe/misinfo_3cand/ml-100k-100-user-medium100/260325142954",        3, 100),
    ("netsafe", "dissem", "3c_medium100", "medium100",
     "netsafe/misinfo_3cand/ml-100k-100-user-medium100/260325142944",        3, 100),
    # ── Dissem (I): NetSafe Qwen3 25pct — InterMat ───────────────────────
    ("netsafe", "dissem", "2c_sparse200", "sparse200",
     "netsafe/misinfo_2cand/ml-100k-100-user-sparse200/260328131711",        2, 200),
    ("netsafe", "dissem", "2c_sparse200", "sparse200",
     "netsafe/misinfo_2cand/ml-100k-100-user-sparse200/260331122926",        2, 200),
    ("netsafe", "dissem", "2c_sparse200", "sparse200",
     "netsafe/misinfo_2cand/ml-100k-100-user-sparse200/260325142951",        2, 200),
    ("netsafe", "dissem", "2c_sparse200", "sparse200",
     "netsafe/misinfo_2cand/ml-100k-100-user-sparse200/260324114904",        2, 200),
    ("netsafe", "dissem", "2c_medium100_im", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260328132026",        2, 100),
    ("netsafe", "dissem", "2c_medium100_im", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260331122849",        2, 100),
    ("netsafe", "dissem", "2c_medium100_im", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260325142936",        2, 100),
    ("netsafe", "dissem", "2c_medium100_im", "medium100",
     "netsafe/misinfo_2cand/ml-100k-100-user-medium100/260323224440",        2, 100),
    ("netsafe", "dissem", "2c_dense50",   "dense50",
     "netsafe/misinfo_2cand/ml-100k-100-user-dense50/260328133146",          2, 50),
    ("netsafe", "dissem", "2c_dense50",   "dense50",
     "netsafe/misinfo_2cand/ml-100k-100-user-dense50/260331122854",          2, 50),
    ("netsafe", "dissem", "2c_dense50",   "dense50",
     "netsafe/misinfo_2cand/ml-100k-100-user-dense50/260325142924",          2, 50),
    ("netsafe", "dissem", "2c_dense50",   "dense50",
     "netsafe/misinfo_2cand/ml-100k-100-user-dense50/260324114830",          2, 50),
    # ── Dissem (U): CORBA Qwen3 — UIDensity ──────────────────────────────
    ("corba", "corba", "1c_medium100", "medium100",
     "corba/corba_canonical_1cand/ml-100k-100-user-medium100/260329160656",              1, 100),
    ("corba", "corba", "1c_medium100", "medium100",
     "corba/corba_canonical_1cand/ml-100k-100-user-medium100/260331114105",              1, 100),
    ("corba", "corba", "2c_medium100", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100-user-medium100/260328162719",              2, 100),
    ("corba", "corba", "2c_medium100", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100-user-medium100/260331114144",              2, 100),
    ("corba", "corba", "2c_medium100", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260404205957", 2, 100),
    ("corba", "corba", "2c_medium100", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260404205957", 2, 100),
    ("corba", "corba", "3c_medium100", "medium100",
     "corba/corba_canonical_3cand/ml-100k-100-user-medium100/260329160725",              3, 100),
    ("corba", "corba", "3c_medium100", "medium100",
     "corba/corba_canonical_3cand/ml-100k-100-user-medium100/260331114112",              3, 100),
    ("corba", "corba", "3c_medium100", "medium100",
     "corba/corba_canonical_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260404205957", 3, 100),
    ("corba", "corba", "3c_medium100", "medium100",
     "corba/corba_canonical_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260404205957", 3, 100),
    # ── Dissem (I): CORBA Qwen3 — InterMat ───────────────────────────────
    ("corba", "corba", "2c_sparse200", "sparse200",
     "corba/corba_canonical_2cand/ml-100k-100-user-sparse200/260329160753",              2, 200),
    ("corba", "corba", "2c_sparse200", "sparse200",
     "corba/corba_canonical_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260404210012", 2, 200),
    ("corba", "corba", "2c_sparse200", "sparse200",
     "corba/corba_canonical_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260404210010", 2, 200),
    ("corba", "corba", "2c_medium100_im", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100-user-medium100/260328162719",              2, 100),
    ("corba", "corba", "2c_medium100_im", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100-user-medium100/260331114144",              2, 100),
    ("corba", "corba", "2c_medium100_im", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260404205957", 2, 100),
    ("corba", "corba", "2c_medium100_im", "medium100",
     "corba/corba_canonical_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260404205957", 2, 100),
    ("corba", "corba", "2c_dense50",   "dense50",
     "corba/corba_canonical_2cand/ml-100k-100-user-dense50/260329160735",                2, 50),
    ("corba", "corba", "2c_dense50",   "dense50",
     "corba/corba_canonical_2cand/ml-100k-100-user-dense50/260331114113",                2, 50),
    ("corba", "corba", "2c_dense50",   "dense50",
     "corba/corba_canonical_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260404205946", 2, 50),
    ("corba", "corba", "2c_dense50",   "dense50",
     "corba/corba_canonical_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260404205942", 2, 50),
    # ── Bidir (U): TOMA Qwen3 — UIDensity ────────────────────────────────
    ("toma", "dissem", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191636", 1, 100),
    ("toma", "dissem", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191637", 1, 100),
    ("toma", "dissem", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115333", 1, 100),
    ("toma", "dissem", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115342", 1, 100),
    ("toma", "dissem", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100-user-medium100/260309113920",            2, 100),
    ("toma", "dissem", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191634", 2, 100),
    ("toma", "dissem", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191635", 2, 100),
    ("toma", "dissem", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115334", 2, 100),
    ("toma", "dissem", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115344", 2, 100),
    ("toma", "dissem", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191636", 3, 100),
    ("toma", "dissem", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191640", 3, 100),
    ("toma", "dissem", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115336", 3, 100),
    ("toma", "dissem", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115346", 3, 100),
    # ── Bidir (I): TOMA Qwen3 — InterMat ─────────────────────────────────
    ("toma", "dissem", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100-user-sparse200/260316133358",            2, 200),
    ("toma", "dissem", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed43/qwen3-235b/260405191657", 2, 200),
    ("toma", "dissem", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed44/qwen3-235b/260405191647", 2, 200),
    ("toma", "dissem", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed43/qwen3-235b/260410115348", 2, 200),
    ("toma", "dissem", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed44/qwen3-235b/260410115406", 2, 200),
    ("toma", "dissem", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100-user-medium100/260309113920",            2, 100),
    ("toma", "dissem", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191634", 2, 100),
    ("toma", "dissem", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191635", 2, 100),
    ("toma", "dissem", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115334", 2, 100),
    ("toma", "dissem", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115344", 2, 100),
    ("toma", "dissem", "2c_dense50",   "dense50",
     "toma/toma_2cand_b/ml-100k-100-user-dense50/260316133313",              2, 50),
    ("toma", "dissem", "2c_dense50",   "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed43/qwen3-235b/260405191629",   2, 50),
    ("toma", "dissem", "2c_dense50",   "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed44/qwen3-235b/260405191629",   2, 50),
    ("toma", "dissem", "2c_dense50",   "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed43/qwen3-235b/260410115327",   2, 50),
    ("toma", "dissem", "2c_dense50",   "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed44/qwen3-235b/260410115337",   2, 50),
    # ── Extraction: MAMA — UIDensity ─────────────────────────────────────
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100-user-medium100/260227130851",              1, 100),
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100-user-medium100/260307152719",              1, 100),
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115303", 1, 100),
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115314", 1, 100),
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 1, 100),
    ("mama", "mama", "1c_medium100", "medium100",
     "mama/mama_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175725", 1, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100-user-medium100/260227120348",              2, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100-user-medium100/260308234620",              2, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115305", 2, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115316", 2, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175725", 2, 100),
    ("mama", "mama", "2c_medium100", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175728", 2, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100-user-medium100/260227130831",              3, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100-user-medium100/260308234638",              3, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115306", 3, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115316", 3, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 3, 100),
    ("mama", "mama", "3c_medium100", "medium100",
     "mama/mama_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175729", 3, 100),
    # ── Extraction: MAMA — InterMat ──────────────────────────────────────
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100-user-sparse200/260226115538",              2, 200),
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100-user-sparse200/260308234647",              2, 200),
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260410115320", 2, 200),
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260410115338", 2, 200),
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260406175734", 2, 200),
    ("mama", "mama", "2c_sparse200", "sparse200",
     "mama/mama_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260406175740", 2, 200),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100-user-medium100/260227120348",              2, 100),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100-user-medium100/260308234620",              2, 100),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115305", 2, 100),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115316", 2, 100),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175725", 2, 100),
    ("mama", "mama", "2c_medium100_im", "medium100",
     "mama/mama_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175728", 2, 100),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100-user-dense50/260226115453",                2, 50),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100-user-dense50/260308234543",                2, 50),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260410115256", 2, 50),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260410115306", 2, 50),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260406175717", 2, 50),
    ("mama", "mama", "2c_dense50", "dense50",
     "mama/mama_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260406175732", 2, 50),
    # ── Extraction: MASLeak — UIDensity ──────────────────────────────────
    ("masleak", "masleak", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100-user-medium100/260307183334",        1, 100),
    ("masleak", "masleak", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 1, 100),
    ("masleak", "masleak", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175731", 1, 100),
    ("masleak", "masleak", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115307", 1, 100),
    ("masleak", "masleak", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115316", 1, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260309120036",        2, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260310131140",        2, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 2, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175738", 2, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115309", 2, 100),
    ("masleak", "masleak", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115318", 2, 100),
    ("masleak", "masleak", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100-user-medium100/260307183344",        3, 100),
    ("masleak", "masleak", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175725", 3, 100),
    ("masleak", "masleak", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175735", 3, 100),
    ("masleak", "masleak", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115312", 3, 100),
    ("masleak", "masleak", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115320", 3, 100),
    # ── Extraction: MASLeak — InterMat ───────────────────────────────────
    ("masleak", "masleak", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100-user-sparse200/260311091417",        2, 200),
    ("masleak", "masleak", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260406175737", 2, 200),
    ("masleak", "masleak", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260406175747", 2, 200),
    ("masleak", "masleak", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260410115323", 2, 200),
    ("masleak", "masleak", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260410115331", 2, 200),
    ("masleak", "masleak", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260310131140",        2, 100),
    ("masleak", "masleak", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 2, 100),
    ("masleak", "masleak", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175738", 2, 100),
    ("masleak", "masleak", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115309", 2, 100),
    ("masleak", "masleak", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115318", 2, 100),
    ("masleak", "masleak", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100-user-dense50/260310131150",          2, 50),
    ("masleak", "masleak", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260406175717", 2, 50),
    ("masleak", "masleak", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260406175726", 2, 50),
    ("masleak", "masleak", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260410115313", 2, 50),
    ("masleak", "masleak", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260410115310", 2, 50),
    # ── Extraction: MASLeak Recall — same dirs, different metric ──────────
    ("masleak", "masleak_recall", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100-user-medium100/260307183334",        1, 100),
    ("masleak", "masleak_recall", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 1, 100),
    ("masleak", "masleak_recall", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175731", 1, 100),
    ("masleak", "masleak_recall", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115307", 1, 100),
    ("masleak", "masleak_recall", "1c_medium100", "medium100",
     "masleak/masleak_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115316", 1, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260309120036",        2, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260310131140",        2, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175738", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115309", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115318", 2, 100),
    ("masleak", "masleak_recall", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100-user-medium100/260307183344",        3, 100),
    ("masleak", "masleak_recall", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175725", 3, 100),
    ("masleak", "masleak_recall", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175735", 3, 100),
    ("masleak", "masleak_recall", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115312", 3, 100),
    ("masleak", "masleak_recall", "3c_medium100", "medium100",
     "masleak/masleak_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115320", 3, 100),
    ("masleak", "masleak_recall", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100-user-sparse200/260311091417",        2, 200),
    ("masleak", "masleak_recall", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260406175737", 2, 200),
    ("masleak", "masleak_recall", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260406175747", 2, 200),
    ("masleak", "masleak_recall", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260410115323", 2, 200),
    ("masleak", "masleak_recall", "2c_sparse200", "sparse200",
     "masleak/masleak_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260410115331", 2, 200),
    ("masleak", "masleak_recall", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100-user-medium100/260310131140",        2, 100),
    ("masleak", "masleak_recall", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260406175723", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260406175738", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260410115309", 2, 100),
    ("masleak", "masleak_recall", "2c_medium100_im", "medium100",
     "masleak/masleak_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260410115318", 2, 100),
    ("masleak", "masleak_recall", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100-user-dense50/260310131150",          2, 50),
    ("masleak", "masleak_recall", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260406175717", 2, 50),
    ("masleak", "masleak_recall", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260406175726", 2, 50),
    ("masleak", "masleak_recall", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260410115313", 2, 50),
    ("masleak", "masleak_recall", "2c_dense50", "dense50",
     "masleak/masleak_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260410115310", 2, 50),
    # ── Extract (Bidir-U): TOMA extraction — UIDensity ────────────────────
    ("toma", "toma_extract", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191636", 1, 100),
    ("toma", "toma_extract", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191637", 1, 100),
    ("toma", "toma_extract", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115333", 1, 100),
    ("toma", "toma_extract", "1c_medium100", "medium100",
     "toma/toma_1cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115342", 1, 100),
    ("toma", "toma_extract", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100-user-medium100/260309113920",            2, 100),
    ("toma", "toma_extract", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191634", 2, 100),
    ("toma", "toma_extract", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191635", 2, 100),
    ("toma", "toma_extract", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115334", 2, 100),
    ("toma", "toma_extract", "2c_medium100", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115344", 2, 100),
    ("toma", "toma_extract", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191636", 3, 100),
    ("toma", "toma_extract", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191640", 3, 100),
    ("toma", "toma_extract", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115336", 3, 100),
    ("toma", "toma_extract", "3c_medium100", "medium100",
     "toma/toma_3cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115346", 3, 100),
    # ── Extract (Bidir-I): TOMA extraction — InterMat ─────────────────────
    ("toma", "toma_extract", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100-user-sparse200/260316133358",            2, 200),
    ("toma", "toma_extract", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed43/qwen3-235b/260405191657", 2, 200),
    ("toma", "toma_extract", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed44/qwen3-235b/260405191647", 2, 200),
    ("toma", "toma_extract", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed43/qwen3-235b/260410115348", 2, 200),
    ("toma", "toma_extract", "2c_sparse200", "sparse200",
     "toma/toma_2cand_b/ml-100k-100user-sparse200-seed44/qwen3-235b/260410115406", 2, 200),
    ("toma", "toma_extract", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100-user-medium100/260309113920",            2, 100),
    ("toma", "toma_extract", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260405191634", 2, 100),
    ("toma", "toma_extract", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260405191635", 2, 100),
    ("toma", "toma_extract", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed43/qwen3-235b/260410115334", 2, 100),
    ("toma", "toma_extract", "2c_medium100_im", "medium100",
     "toma/toma_2cand_b/ml-100k-100user-medium100-seed44/qwen3-235b/260410115344", 2, 100),
    ("toma", "toma_extract", "2c_dense50", "dense50",
     "toma/toma_2cand_b/ml-100k-100-user-dense50/260316133313",              2, 50),
    ("toma", "toma_extract", "2c_dense50", "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed43/qwen3-235b/260405191629",   2, 50),
    ("toma", "toma_extract", "2c_dense50", "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed44/qwen3-235b/260405191629",   2, 50),
    ("toma", "toma_extract", "2c_dense50", "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed43/qwen3-235b/260410115327",   2, 50),
    ("toma", "toma_extract", "2c_dense50", "dense50",
     "toma/toma_2cand_b/ml-100k-100user-dense50-seed44/qwen3-235b/260410115337",   2, 50),
    # ── Extract (Bidir-U): MASTER extraction — UIDensity ──────────────────
    ("master", "master_extract", "1c_medium100", "medium100",
     "master/master_1cand/ml-100k-100-user-medium100/260311113801",          1, 100),
    ("master", "master_extract", "1c_medium100", "medium100",
     "master/master_1cand/ml-100k-100user-medium100-seed43/qwen3-235b/260411123813", 1, 100),
    ("master", "master_extract", "1c_medium100", "medium100",
     "master/master_1cand/ml-100k-100user-medium100-seed44/qwen3-235b/260411123816", 1, 100),
    ("master", "master_extract", "2c_medium100", "medium100",
     "master/master_2cand/ml-100k-100-user-medium100/260311091426",          2, 100),
    ("master", "master_extract", "2c_medium100", "medium100",
     "master/master_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260411123816", 2, 100),
    ("master", "master_extract", "2c_medium100", "medium100",
     "master/master_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260411123813", 2, 100),
    ("master", "master_extract", "3c_medium100", "medium100",
     "master/master_3cand/ml-100k-100-user-medium100/260311113752",          3, 100),
    ("master", "master_extract", "3c_medium100", "medium100",
     "master/master_3cand/ml-100k-100user-medium100-seed43/qwen3-235b/260411123813", 3, 100),
    ("master", "master_extract", "3c_medium100", "medium100",
     "master/master_3cand/ml-100k-100user-medium100-seed44/qwen3-235b/260411123816", 3, 100),
    # ── Extract (Bidir-I): MASTER extraction — InterMat ───────────────────
    ("master", "master_extract", "2c_sparse200", "sparse200",
     "master/master_2cand/ml-100k-100-user-sparse200/260311113622",          2, 200),
    ("master", "master_extract", "2c_sparse200", "sparse200",
     "master/master_2cand/ml-100k-100user-sparse200-seed43/qwen3-235b/260411123825", 2, 200),
    ("master", "master_extract", "2c_sparse200", "sparse200",
     "master/master_2cand/ml-100k-100user-sparse200-seed44/qwen3-235b/260411123826", 2, 200),
    ("master", "master_extract", "2c_medium100_im", "medium100",
     "master/master_2cand/ml-100k-100-user-medium100/260311091426",          2, 100),
    ("master", "master_extract", "2c_medium100_im", "medium100",
     "master/master_2cand/ml-100k-100user-medium100-seed43/qwen3-235b/260411123816", 2, 100),
    ("master", "master_extract", "2c_medium100_im", "medium100",
     "master/master_2cand/ml-100k-100user-medium100-seed44/qwen3-235b/260411123813", 2, 100),
    ("master", "master_extract", "2c_dense50", "dense50",
     "master/master_2cand/ml-100k-100-user-dense50/260311113559",            2, 50),
    ("master", "master_extract", "2c_dense50", "dense50",
     "master/master_2cand/ml-100k-100user-dense50-seed43/qwen3-235b/260411123807",   2, 50),
    ("master", "master_extract", "2c_dense50", "dense50",
     "master/master_2cand/ml-100k-100user-dense50-seed44/qwen3-235b/260411123808",   2, 50),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_nested(d: dict, path: str, default=0.0):
    """Dot-path accessor."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if isinstance(d, (int, float)) else default


def load_series(run_dir: str, metric_path: str) -> List[float]:
    """Load a scalar metric time series from turn_*.json files.
    Special path 'llm_judge.user._asr' / 'llm_judge.item._asr' computes
    contaminated_count / non_attacker_total * 100 (U-ASR or I-ASR).
    """
    files = sorted(
        glob.glob(os.path.join(run_dir, "turn_*.json")),
        key=lambda x: int(os.path.basename(x).replace("turn_", "").replace(".json", ""))
    )
    series = []
    for f in files:
        try:
            d = json.load(open(f))
            if metric_path.endswith("._asr"):
                role = metric_path.split(".")[1]  # "user" or "item"
                llm = d.get("llm_judge", {})
                agent = (llm.get(role) or {})
                cc = agent.get("contaminated_count")
                nse = d.get("natural_semantic_evolution", {})
                if role == "user":
                    denom = nse.get("active_users_non_attacker") or \
                            (nse.get("active_users_total", 0) - nse.get("num_attacker_users", 0))
                else:
                    denom = nse.get("active_items_non_attacker") or \
                            (nse.get("active_items_total", 0) - nse.get("num_attacker_items", 0))
                series.append((cc / max(1, denom)) * 100 if cc is not None and denom else 0.0)
            elif metric_path.startswith("resources.corba_"):
                # CORBA metrics are already rates [0,1]; scale to % for consistency
                series.append(float(get_nested(d, metric_path, 0.0)) * 100)
            elif metric_path == "privacy.pii_leakage_rate":
                # MAMA PII leakage rate [0,1]; scale to %
                series.append(float(get_nested(d, metric_path, 0.0)) * 100)
            else:
                series.append(float(get_nested(d, metric_path, 0.0)))
        except Exception:
            series.append(0.0)
    return series


def smooth(series: List[float], window: int) -> List[float]:
    """Rolling mean with edge padding."""
    if window <= 1 or len(series) < window:
        return series
    arr = np.array(series, dtype=float)
    kernel = np.ones(window) / window
    return list(np.convolve(arr, kernel, mode="same"))


def find_breakpoint(series: List[float], tr_min: int, tr_max_frac: float,
                    attack: str = "") -> int:
    """
    Automatic transient/steady-state split by two-segment piecewise-LINEAR fit.

    For every candidate breakpoint bp in [tr_min, tr_max], fit a least-squares
    line to series[:bp] (the ramp) and another to series[bp:] (the plateau),
    and pick the bp that minimises the combined residual sum of squares. This
    is the method described in this module's header and it is fully data-driven
    -- it replaces the previous hand-tuned per-attack MANUAL cutoffs (read "from
    visual inspection"), which injected researcher degrees of freedom into the
    tr/ss numbers that feed every correlation. `attack` is accepted but unused,
    so the split no longer depends on the attack's identity.
    """
    n = len(series)
    if n < 4:
        return max(1, n // 2)
    tr_max = min(max(tr_min + 1, int(n * tr_max_frac)), n - 2)
    arr = np.array(series, dtype=float)
    x = np.arange(n, dtype=float)

    def _seg_sse(xs, ys):
        if len(xs) < 2:
            return 0.0
        a, b = np.polyfit(xs, ys, 1)
        resid = ys - (a * xs + b)
        return float(np.dot(resid, resid))

    best_bp, best_sse = tr_min, np.inf
    for bp in range(tr_min, tr_max + 1):
        sse = _seg_sse(x[:bp], arr[:bp]) + _seg_sse(x[bp:], arr[bp:])
        if sse < best_sse:
            best_sse, best_bp = sse, bp
    return best_bp


def compute_tr_ss(series: List[float], cfg: dict, attack: str = "") -> Tuple[float, float]:
    """
    Returns (tr_slope, ss_mean).
    tr_slope: mean first-difference in the transient segment (per turn).
    ss_mean:  mean value in the steady-state segment.
    """
    if not series:
        return 0.0, 0.0
    s = smooth(series, cfg["SMOOTH_WINDOW"])
    bp = find_breakpoint(s, cfg["TR_MIN_TURNS"], cfg["TR_MAX_FRAC"], attack)
    tr_seg = s[:bp]
    ss_seg = s[bp:]
    if len(tr_seg) >= 2:
        diffs = [tr_seg[i+1] - tr_seg[i] for i in range(len(tr_seg)-1)]
        tr_slope = float(np.mean(diffs))
    else:
        tr_slope = float(tr_seg[-1]) if tr_seg else 0.0
    ss_mean = float(np.mean(ss_seg)) if ss_seg else float(np.mean(s))
    return tr_slope, ss_mean


# ---------------------------------------------------------------------------
# Static predictors
# ---------------------------------------------------------------------------

def item_degree(rho: float, k: int) -> float:
    """Expected number of user contacts per item per turn.

    Each of the n_U users contacts k items per turn, so the total number of
    U-I contacts per turn is n_U*k, and the mean per item is
        n_U*k / n_I = rho*k,   with rho = n_U/n_I.

    This is the item-side counterpart of the user-side degree k, and is what
    must appear in E_I / R_I. Using rho alone makes the item-side predictors
    independent of k, which contradicts the observed k-dependence of item-side
    ASR (paper F3). Note that on the InterMat sweep k is held fixed at 2, so
    d -> 2d is a monotone reparameterisation there and the fitted (gamma, r)
    absorb it: all previously reported InterMat correlations are unchanged.
    The correction matters because it also makes R_I defined on the k sweep.
    """
    return rho * k


def compute_rho(n_items: int, n_users: int = 100) -> float:
    """Catalog concentration ρ = n_U / n_I (as defined in the paper)."""
    return n_users / n_items if n_items > 0 else 0.0


# ---------------------------------------------------------------------------
# ODE simulation and trajectory fitting
# ---------------------------------------------------------------------------

def simulate_sis(beta_U: float, beta_I: float, gamma: float, r: float,
                 alpha_U: float, alpha_I: float, k: int, rho: float,
                 n_turns: int) -> Tuple[List[float], List[float]]:
    """
    Integrate the bipartite mean-field SIS ODEs (one step = one turn).

    drho_U/dt = beta_U*(1-rho_U)*[1-(1-rho_I)^k]     - rho_U*[(1-rho_I)*k*gamma + r]
    drho_I/dt = beta_I*(1-rho_I)*[1-(1-rho_U)^(rho*k)] - rho_I*[(1-rho_U)*rho*k*gamma + r]

    rho = n_U/n_I (catalog concentration); rho*k is the item-side degree, i.e.
    the expected number of user contacts per item per turn (see item_degree).
    Initial conditions: rho_U(0) = alpha_U, rho_I(0) = alpha_I.
    Returns (rho_U_series, rho_I_series) as percentages [0,100].
    """
    rho_U, rho_I = alpha_U, alpha_I
    d_I = item_degree(rho, k)
    u_series, i_series = [rho_U * 100], [rho_I * 100]
    for _ in range(n_turns - 1):
        inf_U = beta_U * (1 - rho_U) * (1 - (1 - rho_I) ** k)
        rec_U = rho_U * ((1 - rho_I) * k * gamma + r)
        inf_I = beta_I * (1 - rho_I) * (1 - (1 - rho_U) ** max(d_I, 1e-9))
        rec_I = rho_I * ((1 - rho_U) * d_I * gamma + r)
        rho_U = float(np.clip(rho_U + inf_U - rec_U, 0.0, 1.0))
        rho_I = float(np.clip(rho_I + inf_I - rec_I, 0.0, 1.0))
        u_series.append(rho_U * 100)
        i_series.append(rho_I * 100)
    return u_series, i_series


def fit_sis_to_run(u_obs: List[float], i_obs: List[float],
                   alpha_U: float, alpha_I: float,
                   k: int, rho: float) -> dict:
    """
    Fit (beta_U, beta_I, gamma, r) to minimise MSE between ODE trajectory
    and observed (u_obs, i_obs) time series (both in %).
    rho = n_U/n_I (catalog concentration).
    """
    n = len(u_obs)
    u_arr = np.array(u_obs, dtype=float)
    i_arr = np.array(i_obs, dtype=float) if i_obs else None

    def _mse(params):
        beta_U, beta_I, gamma, r = [max(p, 1e-6) for p in params]
        u_sim, i_sim = simulate_sis(beta_U, beta_I, gamma, r,
                                    alpha_U, alpha_I, k, rho, n)
        loss = np.mean((np.array(u_sim) - u_arr) ** 2)
        if i_arr is not None and len(i_arr) == n:
            loss += np.mean((np.array(i_sim) - i_arr) ** 2)
        return float(loss)

    best_loss, best_p = np.inf, [0.5, 0.5, 0.1, 0.05]
    for bU in [0.1, 0.3, 0.6, 1.0, 2.0]:
        for bI in [0.1, 0.3, 0.6, 1.0, 2.0]:
            for g in [0.01, 0.1, 0.5, 1.0]:
                for rv in [0.01, 0.05, 0.2]:
                    v = _mse([bU, bI, g, rv])
                    if v < best_loss:
                        best_loss, best_p = v, [bU, bI, g, rv]

    res = minimize(_mse, best_p, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
    beta_U, beta_I, gamma, r = [max(p, 1e-6) for p in res.x]
    u_sim, i_sim = simulate_sis(beta_U, beta_I, gamma, r,
                                alpha_U, alpha_I, k, rho, n)
    mse_U = float(np.mean((np.array(u_sim) - u_arr) ** 2))
    mse_I = float(np.mean((np.array(i_sim) - np.array(i_obs)) ** 2)) if i_obs else float("nan")
    return {
        "beta_U": round(beta_U, 5),
        "beta_I": round(beta_I, 5),
        "gamma":  round(gamma,  5),
        "r":      round(r,      5),
        "mse_U":  round(mse_U,  4),
        "mse_I":  round(mse_I,  4) if not math.isnan(mse_I) else float("nan"),
    }


def fit_sis_per_attack(rows: List[dict], base_dir: str, cfg: dict) -> Dict[str, dict]:
    """
    For each (attack, metric_type, label) run, load the full time series and
    fit SIS parameters. Returns dict keyed by run_dir.
    """
    results = {}
    for entry in RUNS:
        attack, metric_type, label, dataset, pattern, k, n_items = entry[:7]
        alpha = entry[7] if len(entry) > 7 else cfg["ALPHA"]
        run_dir = resolve_run_dir(base_dir, pattern)
        if run_dir is None:
            continue
        u_path, i_path = METRIC_PATHS[metric_type]
        u_series = load_series(run_dir, u_path) if u_path else []
        i_series = load_series(run_dir, i_path) if i_path else []
        if len(u_series) < 5:
            continue
        rho_val = compute_rho(n_items)
        fitted = fit_sis_to_run(u_series, i_series, alpha, alpha, k, rho_val)
        fitted["run_dir"] = run_dir
        fitted["attack"] = attack
        fitted["label"] = label
        fitted["k_UI"] = k
        fitted["n_items"] = n_items
        results[run_dir] = fitted
        print(f"  SIS fit {attack:<12} {label:<20} "
              f"β_U={fitted['beta_U']:.4f} β_I={fitted['beta_I']:.4f} "
              f"γ={fitted['gamma']:.4f} r={fitted['r']:.4f} "
              f"MSE_U={fitted['mse_U']:.2f} MSE_I={fitted['mse_I']}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_run_dir(base_dir: str, pattern: str) -> Optional[str]:
    """Return the run directory: pattern is now a pinned relative path (no glob)."""
    path = os.path.join(base_dir, pattern)
    return path if os.path.isdir(path) else None


def process_run(entry: tuple, base_dir: str, cfg: dict) -> Optional[dict]:
    attack, metric_type, label, dataset, pattern, k, n_items = entry[:7]
    alpha = entry[7] if len(entry) > 7 else cfg["ALPHA"]  # per-run alpha override
    run_dir = resolve_run_dir(base_dir, pattern)
    if run_dir is None:
        print(f"  [SKIP] {attack}/{label}: no run found for {pattern}")
        return None

    u_path, i_path = METRIC_PATHS[metric_type]

    u_series = load_series(run_dir, u_path) if u_path else []
    i_series = load_series(run_dir, i_path) if i_path else []

    if not u_series:
        print(f"  [SKIP] {attack}/{label}: empty user series")
        return None

    u_tr, u_ss = compute_tr_ss(u_series, cfg, attack)
    i_tr, i_ss = compute_tr_ss(i_series, cfg, attack) if i_series else (0.0, 0.0)

    # Transient (one-hop exposure) predictor -- no free parameters. The
    # steady-state predictor (coupled rho*) and R0 are computed later, once
    # theta is fit per attack; see sis_predictors.evaluate.
    rho_val = sp.catalog_concentration(n_items)
    E_U = sp.exposure_U(alpha, k)
    E_I = sp.exposure_I(alpha, rho_val, k)

    return {
        "attack": attack,
        "metric_type": metric_type,
        "label": label,
        "dataset": dataset,
        "k_UI": k,
        "n_items": n_items,
        "alpha": alpha,
        "run_dir": run_dir,
        "n_turns": len(u_series),
        # Empirical outcomes
        "U_tr": round(u_tr, 5),
        "U_ss": round(u_ss, 5),
        "I_tr": round(i_tr, 5),
        "I_ss": round(i_ss, 5),
        # Transient predictor (exposure, no free params)
        "E_U": round(E_U, 5),
        "E_I": round(E_I, 5),
        "rho_val": round(rho_val, 3),
    }


ATTACK_CLASS = {
    ("netsafe",  "dissem"):         ("Dissemination", None),
    ("corba",    "corba"):          ("Dissemination", None),
    ("toma",     "dissem"):         ("Bidirectional", "dissem"),
    ("toma",     "toma_extract"):   ("Bidirectional[extract-TOMA]", None),
    ("master",   "master_extract"): ("Bidirectional[extract-MASTER]", None),
    ("mama",     "mama"):           ("Extraction-MAMA",    None),
    ("masleak",  "masleak"):        ("Extraction-MASLeak-F1", None),
    ("masleak",  "masleak_recall"): ("Extraction-MASLeak-Recall", None),
}

def sweep_axis(label: str) -> str:
    # InterMat labels contain density keywords or _im suffix; UIDensity labels end in plain _medium100
    if any(x in label for x in ["sparse", "dense50", "_im"]):
        return "InterMat"
    return "UIDensity"


def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    exclude = {"gamma_tr", "gamma_ss", "R_U_tr", "R_U_ss", "R_I_tr", "R_I_ss"}
    fields = [k for k in rows[0].keys() if k not in exclude]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}")


def write_raw_numbers_csv(rows: List[dict], cfg: dict, path: str):
    """Write the honest per-run CSV: observed outcomes + the parameter-free
    exposure predictor. The steady-state predictor (coupled rho*) and R0 are
    per-attack (they need the fitted theta), so they live in the corr CSV, not
    here. No max(R,E) flooring; no per-attack manual tr cutoffs."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "attack", "metric_type", "sweep", "label", "k_UI", "n_items", "alpha",
            "n_turns", "U_tr", "U_ss", "I_tr", "I_ss", "E_U", "E_I",
        ])
        for r in rows:
            w.writerow([
                r["attack"], r["metric_type"], sweep_axis(r["label"]), r["label"],
                r["k_UI"], r["n_items"], round(float(r["alpha"]), 4),
                r.get("n_turns", ""),
                round(r["U_tr"], 5), round(r["U_ss"], 5),
                round(r["I_tr"], 5), round(r["I_ss"], 5),
                round(r["E_U"], 5), round(r["E_I"], 5),
            ])
    print(f"  Wrote {path}")


def print_summary(rows, evaluation, loo):
    """Human-readable summary: raw per-run outcomes + the signed per-sweep
    correlation table from sis_predictors.evaluate, plus leave-one-attack-out."""
    print("\n" + "=" * 88)
    print("RAW NUMBERS (observed outcomes + parameter-free exposure)")
    print("=" * 88)
    cw = 10
    cols = ["U_tr", "U_ss", "I_tr", "I_ss", "E_U", "E_I"]
    header = f"{'attack':<14} {'label':<18} {'k':>3} {'ni':>4} " + \
             " ".join(f"{c:>{cw}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['attack']:<14} {r['label']:<18} {r['k_UI']:>3} {r['n_items']:>4} "
        line += " ".join(f"{r.get(c, 0.0):>{cw}.4f}" for c in cols)
        print(line)

    def _f(x):
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"

    print("\n" + "=" * 72)
    print("SEC7 PREDICTORS -- signed Spearman per sweep")
    print("  transient <- exposure (0 params);  steady <- coupled rho* (P2/RMP fix)")
    print("=" * 72)
    hdr = f"{'attack/metric':<24} {'sweep':<6} {'U_tr':>7} {'I_tr':>7} {'U_ss':>7} {'I_ss':>7}"
    print(hdr)
    print("-" * len(hdr))
    for atk, rr in evaluation.items():
        for i, (sw, s) in enumerate(rr["sweeps"].items()):
            name = atk if i == 0 else ""
            print(f"{name:<24} {sw:<6} {_f(s['U_tr']):>7} {_f(s['I_tr']):>7} "
                  f"{_f(s['U_ss']):>7} {_f(s['I_ss']):>7}")

    print("\nLeave-one-attack-out (out-of-sample steady-state ranking):")
    for atk, rr in loo.items():
        print(f"  {atk:<24} U_ss={_f(rr['U_ss']):>7}  I_ss={_f(rr['I_ss']):>7}")


def load_rows_from_csv(path):
    """Load per-run rows from a cached ranking_predictors_raw.csv (the raw
    per-turn trajectories are not shipped; the extracted tr/ss are). Recomputes
    the parameter-free exposure so E is always consistent with sis_predictors."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            k, ni, a = int(r["k_UI"]), int(r["n_items"]), float(r["alpha"])
            rho = sp.catalog_concentration(ni)
            rows.append({
                "attack": r["attack"], "metric_type": r["metric_type"],
                "label": r.get("label", ""), "dataset": r.get("dataset", ""),
                "k_UI": k, "n_items": ni, "alpha": a,
                "run_dir": r.get("run_dir", ""), "n_turns": r.get("n_turns", ""),
                "U_tr": float(r["U_tr"]), "U_ss": float(r["U_ss"]),
                "I_tr": float(r["I_tr"]), "I_ss": float(r["I_ss"]),
                "E_U": round(sp.exposure_U(a, k), 5),
                "E_I": round(sp.exposure_I(a, rho, k), 5),
                "rho_val": round(rho, 3),
            })
    return rows


def build_configs(rows):
    """Aggregate per-run rows (seed replicates) into per-attack Config lists for
    the coupled predictor / correlation stage in sis_predictors."""
    from collections import defaultdict as _dd
    agg = _dd(lambda: _dd(lambda: _dd(list)))
    alpha_of = {}
    for r in rows:
        key = (r["attack"], r["metric_type"])
        ck = (int(r["k_UI"]), int(r["n_items"]))
        for m in ("U_tr", "U_ss", "I_tr", "I_ss"):
            agg[key][ck][m].append(float(r[m]))
        alpha_of[key] = float(r["alpha"])
    out = {}
    for key, byck in agg.items():
        means = {ck: {m: float(np.mean(v)) for m, v in d.items()}
                 for ck, d in byck.items()}
        has_item = max((mv.get("I_ss", 0.0) + mv.get("I_tr", 0.0))
                       for mv in means.values()) > 1e-9
        a = alpha_of[key]
        cfgs = [sp.Config(k=k, n_items=ni, alpha_U=a, alpha_I=a,
                          U_tr=mv.get("U_tr", 0.0), U_ss=mv.get("U_ss", 0.0),
                          I_tr=mv.get("I_tr", 0.0), I_ss=mv.get("I_ss", 0.0),
                          has_item=has_item)
                for (k, ni), mv in sorted(means.items())]
        out["/".join(key)] = cfgs
    return out


def write_loo_csv(loo, path):
    """Write the leave-one-attack-out out-of-sample steady-state correlations."""
    def _c(x):
        return "" if (isinstance(x, float) and math.isnan(x)) else round(x, 4)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "U_ss_loao", "I_ss_loao",
                    "beta_U", "beta_I", "gamma", "r"])
        for atk, rr in loo.items():
            th = rr.get("theta_train", (float("nan"),) * 4)
            w.writerow([atk, _c(rr["U_ss"]), _c(rr["I_ss"]),
                        round(th[0], 5), round(th[1], 5),
                        round(th[2], 5), round(th[3], 5)])
    print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=DEFAULTS["OUTPUT_DIR"].replace(
        "tools/ranking_pred_output", "attack_output"))
    parser.add_argument("--output_dir", default=DEFAULTS["OUTPUT_DIR"])
    parser.add_argument("--tr_min_turns", type=int, default=DEFAULTS["TR_MIN_TURNS"])
    parser.add_argument("--tr_max_frac", type=float, default=DEFAULTS["TR_MAX_FRAC"])
    parser.add_argument("--smooth_window", type=int, default=DEFAULTS["SMOOTH_WINDOW"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["ALPHA"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["GAMMA"])
    parser.add_argument("--r_llm", type=float, default=DEFAULTS["R_LLM"])
    parser.add_argument("--fit_sis", action="store_true", default=False,
                        help="Fit full SIS ODE (beta_U, beta_I, gamma, r) per run to observed trajectories.")
    parser.add_argument("--from_raw_csv", default=None,
                        help="Skip raw-trajectory extraction; load per-run tr/ss from a cached "
                             "ranking_predictors_raw.csv and run the predictor stage only.")
    parser.add_argument("--config", default=None,
                        help="YAML config file overriding any of the above")
    args = parser.parse_args()

    cfg = dict(
        TR_MIN_TURNS=args.tr_min_turns,
        TR_MAX_FRAC=args.tr_max_frac,
        SMOOTH_WINDOW=args.smooth_window,
        ALPHA=args.alpha,
        GAMMA=args.gamma,
        R_LLM=args.r_llm,
    )

    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        cfg.update({k.upper(): v for k, v in overrides.items()})

    base_dir = args.base_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: {cfg}")

    if args.from_raw_csv:
        print(f"Loading cached per-run outcomes from {args.from_raw_csv}\n")
        rows = load_rows_from_csv(args.from_raw_csv)
    else:
        print(f"Base dir: {base_dir}")
        print(f"Processing {len(RUNS)} run entries...\n")
        rows = []
        for entry in RUNS:
            result = process_run(entry, base_dir, cfg)
            if result:
                rows.append(result)
                print(f"  OK  {result['attack']:<12} {result['label']:<20} "
                      f"turns={result['n_turns']:>3}  "
                      f"U_tr={result['U_tr']:.4f}  U_ss={result['U_ss']:.4f}  "
                      f"I_tr={result['I_tr']:.4f}  I_ss={result['I_ss']:.4f}")

    if not rows:
        print("No runs found. (Raw trajectories live under attack_output/ and are "
              "not shipped; use --from_raw_csv to run from cached outcomes.)")
        return

    # --- §7 predictive metrics (RMP-coupled, signed, honest) ------------------
    # The predictor/fit/correlation stage is delegated entirely to
    # sis_predictors: exposure ranks the transient slope; the COUPLED
    # steady state rho* (no partition frozen at alpha) ranks the steady-state
    # mean; theta=(beta_U,beta_I,gamma,r) is fit once per attack and validated
    # leave-one-attack-out; correlations are SIGNED Spearman with no max(R,E)
    # flooring and no short-circuit. See sis_predictors.py for the model.
    configs_by_attack = build_configs(rows)
    evaluation = sp.evaluate(configs_by_attack)
    loo = sp.leave_one_attack_out(configs_by_attack)

    write_csv(rows, os.path.join(output_dir, "ranking_predictors_raw.csv"))
    write_raw_numbers_csv(rows, cfg, os.path.join(output_dir, "ranking_predictors_per_run.csv"))
    sp.write_corr_csv(evaluation, os.path.join(output_dir, "ranking_predictors_corr.csv"))
    write_loo_csv(loo, os.path.join(output_dir, "ranking_predictors_loo.csv"))

    # Optional: fit full SIS ODE per run
    if args.fit_sis:
        print("\nFitting SIS ODE trajectories per run (beta_U, beta_I, gamma, r)...")
        sis_results = fit_sis_per_attack(rows, base_dir, cfg)
        sis_path = os.path.join(output_dir, "sis_fitted_params.csv")
        if sis_results:
            fields = ["attack", "label", "k_UI", "n_items",
                      "beta_U", "beta_I", "gamma", "r", "mse_U", "mse_I", "run_dir"]
            with open(sis_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(sis_results.values())
            print(f"  Wrote {sis_path}")

    summary_path = os.path.join(output_dir, "ranking_predictors.txt")
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_summary(rows, evaluation, loo)
    sys.stdout = old_stdout
    summary = buf.getvalue()
    print(summary)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()
