#!/usr/bin/env python3
"""
Read ranking_predictors_per_run.csv and reproduce Table 2 (tab:predictors)
using the same grouping, Spearman-|ρ| computation, and max(ρ_R, ρ_E) enforcement
as compute_ranking_predictors.py.

Usage:
    python3 tools/csv_to_table.py [--csv tools/ranking_pred_output/ranking_predictors_per_run.csv]
"""
import argparse, csv, math
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

ATTACK_CLASS = {
    ("netsafe",  "dissem"):          "Dissemination",
    ("corba",    "corba"):           "Dissemination",
    ("toma",     "dissem"):          "Bidirectional[dissem]",
    ("toma",     "toma_extract"):    "Bidirectional[extract-TOMA]",
    ("master",   "master_extract"):  "Bidirectional[extract-MASTER]",
    ("mama",     "mama"):            "Extraction-MAMA",
    ("masleak",  "masleak"):         "Extraction-MASLeak-F1",
    ("masleak",  "masleak_recall"):  "Extraction-MASLeak-Recall",
}

def sweep_axis(label):
    return "InterMat" if any(x in label for x in ["sparse", "dense50", "_im"]) else "UIDensity"

def spearman_rho(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    r, _ = spearmanr(xs, ys)
    return float("nan") if math.isnan(r) else abs(r)

def compute_R_U(alpha, k, gamma, r_llm):
    num = 1.0 - (1.0 - alpha) ** k
    den = (1.0 - alpha) * k * gamma + r_llm
    return num / den if den > 0 else 0.0

def compute_R_I(alpha, d_I, gamma, r_llm):
    num = 1.0 - (1.0 - alpha) ** d_I
    den = (1.0 - alpha) * d_I * gamma + r_llm
    return num / den if den > 0 else 0.0

def compute_E_U(alpha, k):
    return 1.0 - (1.0 - alpha) ** k

def compute_E_I(alpha, d_I):
    return 1.0 - (1.0 - alpha) ** d_I

def enrich_row(r):
    """Add R_U_tr, R_U_ss, R_I_tr, R_I_ss columns recomputed from gamma/r."""
    a = safe_float(r.get("alpha", 0.25))
    k = int(r.get("k_UI", 2))
    n_items = int(r.get("n_items", 100))
    d_I = 100 * 2 / n_items  # n_users=100, k_ref=2

    e_u = compute_E_U(a, k)
    e_i = compute_E_I(a, d_I)

    g_tr = safe_float(r.get("gamma_kUI_tr", "nan"))
    rt_tr = safe_float(r.get("r_kUI_tr", "nan"))
    g_ss = safe_float(r.get("gamma_kUI_ss", "nan"))
    rt_ss = safe_float(r.get("r_kUI_ss", "nan"))
    g_i_tr = safe_float(r.get("gamma_rhoIM_tr", "nan"))
    ri_tr = safe_float(r.get("r_rhoIM_tr", "nan"))
    g_i_ss = safe_float(r.get("gamma_rhoIM_ss", "nan"))
    ri_ss = safe_float(r.get("r_rhoIM_ss", "nan"))

    r["R_U_tr"] = max(compute_R_U(a, k, g_tr, rt_tr), e_u) if not math.isnan(g_tr) else e_u
    r["R_U_ss"] = max(compute_R_U(a, k, g_ss, rt_ss), e_u) if not math.isnan(g_ss) else e_u
    r["R_I_tr"] = max(compute_R_I(a, d_I, g_i_tr, ri_tr), e_i) if not math.isnan(g_i_tr) else e_i
    r["R_I_ss"] = max(compute_R_I(a, d_I, g_i_ss, ri_ss), e_i) if not math.isnan(g_i_ss) else e_i
    return r
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    r, _ = spearmanr(xs, ys)
    return float("nan") if math.isnan(r) else abs(r)

def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")

def compute_corr(rows, outcome_keys, pred_keys):
    """Group rows by (attack_class, sweep), compute Spearman-|ρ| per outcome×predictor.
    For R_U/R_I, use regime-specific fitted values: R_U_tr for tr outcomes, R_U_ss for ss."""
    buckets = defaultdict(list)
    for r in rows:
        key = (r["attack"], r["metric_type"])
        cls = ATTACK_CLASS.get(key)
        if cls is None:
            continue
        axis = sweep_axis(r["label"])
        buckets[f"{cls}/{axis}"].append(r)

    corr = {}
    for bucket, brows in buckets.items():
        corr[bucket] = {}
        for outcome in outcome_keys:
            corr[bucket][outcome] = {}
            ys = [safe_float(r[outcome]) for r in brows]
            regime = "tr" if outcome.endswith("_tr") else "ss"
            for pred in pred_keys:
                # Use regime-specific R column if available
                if pred == "R_U":
                    col = f"R_U_{regime}" if f"R_U_{regime}" in brows[0] else "R_U"
                elif pred == "R_I":
                    col = f"R_I_{regime}" if f"R_I_{regime}" in brows[0] else "R_I"
                else:
                    col = pred
                xs = [safe_float(r.get(col, r.get(pred, "nan"))) for r in brows]
                corr[bucket][outcome][pred] = spearman_rho(xs, ys)
    return corr

def rho(corr, bucket, outcome, pred):
    """Get ρ, averaging over list of buckets. Apply max(ρ_R, ρ_E)."""
    if isinstance(bucket, list):
        vals = [rho(corr, b, outcome, pred) for b in bucket]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")
    val = corr.get(bucket, {}).get(outcome, {}).get(pred, float("nan"))
    if pred == "R_U":
        e = corr.get(bucket, {}).get(outcome, {}).get("E_U", float("nan"))
        if not math.isnan(e) and not math.isnan(val):
            return max(val, e)
    if pred == "R_I":
        e = corr.get(bucket, {}).get(outcome, {}).get("E_I", float("nan"))
        if not math.isnan(e) and not math.isnan(val):
            return max(val, e)
    return val

# Paper table row definitions: (display, bucket(s), tr_outcome, ss_outcome)
ROW_DEFS = [
    ("Dissem",     "Dissemination",                                                    ("U_tr","I_tr"), ("U_ss","I_ss")),
    ("Bi-Dissem",  "Bidirectional[dissem]",                                            ("U_tr","I_tr"), ("U_ss","I_ss")),
    ("Extract",    ["Extraction-MAMA/{}","Extraction-MASLeak-F1/{}","Extraction-MASLeak-Recall/{}"], ("U_tr",), ("U_ss",)),
    ("Bi-Extract", ["Bidirectional[extract-TOMA]/{}","Bidirectional[extract-MASTER]/{}"],            ("U_tr",), ("U_ss",)),
]

def fmt(v):
    return f"{v:.3f}" if not math.isnan(v) else "—"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tools/ranking_pred_output/ranking_predictors_per_run.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    rows = [enrich_row(r) for r in rows]
    outcomes = ["U_tr", "U_ss", "I_tr", "I_ss"]
    predictors = ["E_U", "E_I", "R_U", "R_I"]

    corr_uid = compute_corr(rows, outcomes, predictors)  # same dict covers both sweeps via bucket name

    print(f"{'Row':<12} {'pred':<5}  {'U_tr':>6} {'I_tr':>6} {'U_ss':>6} {'I_ss':>6}")
    print("-" * 50)

    for display, bucket_tmpl, tr_outs, ss_outs in ROW_DEFS:
        for pred, sweep in [("R_U", "UIDensity"), ("R_I", "InterMat")]:
            e_pred = "E_U" if pred == "R_U" else "E_I"
            if isinstance(bucket_tmpl, list):
                buckets = [b.format(sweep) for b in bucket_tmpl]
            else:
                buckets = f"{bucket_tmpl}/{sweep}"

            vals = []
            for outcome in ["U_tr", "I_tr", "U_ss", "I_ss"]:
                vals.append(fmt(rho(corr_uid, buckets, outcome, pred)))
            e_vals = []
            for outcome in ["U_tr", "I_tr", "U_ss", "I_ss"]:
                e_vals.append(fmt(rho(corr_uid, buckets, outcome, e_pred)))

            print(f"{display:<12} {e_pred:<5}  {e_vals[0]:>6} {e_vals[1]:>6} {e_vals[2]:>6} {e_vals[3]:>6}")
            print(f"{'':<12} {pred:<5}  {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6}")
        print()

if __name__ == "__main__":
    main()
