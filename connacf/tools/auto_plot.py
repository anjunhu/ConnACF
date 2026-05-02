#!/usr/bin/env python3
"""
auto_plot.py — auto-scan attack_output for latest runs and plot with inferred labels.

Scans attack_output/<attack>/ for experiment_config.json files, infers connectivity
labels (1c/2c/3c, 50i/100i/200i) from the config, picks the latest run per
(num_candidates, dataset_density, seed, llm) combination, and calls the appropriate
plotting tool.

Usage:
    python tools/auto_plot.py --attack mama --pub --output ../figure/pdf/mama_auto.pdf
    python tools/auto_plot.py --attack netsafe --llm qwen3 --seeds 43 44 --pub
    python tools/auto_plot.py --attack corba --plot-type intermat --pub
    python tools/auto_plot.py --attack toma --seeds 43 --last_epoch 100
    python tools/auto_plot.py --attack netsafe --variant blindguard --llm qwen3

Supported attacks: mama, masleak, netsafe, toma, master, corba, pi, ia
Plot types: uidensity (default), intermat
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Map attack name → subdirectory under attack_output/
ATTACK_DIR_MAP = {
    "mama":    "mama",
    "masleak": "masleak",
    "netsafe": "netsafe",
    "toma":    "toma",
    "master":  "master",
    "corba":   "corba",
    "pi":      "prompt_infection",
    "ia":      "injecagent",
}

# Canonical config subdir prefixes per attack (exact match against the first path component
# under attack_output/<attack>/). Runs in subdirs that don't match are excluded unless
# --variant is given, which is appended as a suffix filter instead.
CANONICAL_SUBDIR = {
    "mama":    re.compile(r"^mama_\dcand$"),
    "masleak": re.compile(r"^masleak_\dcand$"),
    "netsafe": re.compile(r"^misinfo_\dcand$"),
    "toma":    re.compile(r"^toma_\dcand_b$"),   # _b = item-backward enabled (canonical)
    "master":  re.compile(r"^master_\dcand$"),
    "corba":   re.compile(r"^corba_canonical_\dcand$"),
    "pi":      re.compile(r"^pi_canonical_\dcand$"),
    "ia":      re.compile(r"^ia_memory_exfil_\dcand$"),
}

# Map attack → plotting tool and --attack flag (if needed)
PLOT_TOOL = {
    "mama":    ("tools/plot_mama.py",          None),
    "masleak": ("tools/plot_reveng.py",        None),
    "netsafe": ("tools/plot_dissemination.py", None),
    "toma":    ("tools/plot_toma.py",          None),
    "master":  ("tools/plot_master.py",        None),
    "corba":   ("tools/plot_pi_corba_ia.py",   "corba"),
    "pi":      ("tools/plot_pi_corba_ia.py",   "pi"),
    "ia":      ("tools/plot_pi_corba_ia.py",   "ia"),
}

# Attacks whose plotting tool uses --dirs instead of --connacf_dirs
USES_DIRS_FLAG = {"toma", "mama", "masleak", "master"}


def density_label(dataset_name: str) -> str:
    if "dense50"   in dataset_name or "-dense"   in dataset_name: return "50i"
    if "medium100" in dataset_name or "-medium"  in dataset_name: return "100i"
    if "sparse200" in dataset_name or "-sparse"  in dataset_name: return "200i"
    return "?"


def seed_label(dataset_name: str) -> Optional[str]:
    m = re.search(r"seed(\d+)", dataset_name)
    return m.group(1) if m else None


def llm_short(llm_model: str) -> str:
    s = llm_model.lower()
    if "qwen3"  in s: return "qwen3"
    if "claude" in s: return "claude"
    if "llama"  in s: return "llama"
    if "hf-"    in s: return llm_model.split("/")[-1]
    return llm_model.split(".")[-1][:10]


def scan_runs(attack: str, base_dir: str, llm_filter: Optional[str],
              seed_filter: list, cand_filter: list,
              density_filter: list, variant: Optional[str]) -> list:
    """Scan attack_output/<attack>/ and return list of run metadata dicts."""
    atk_dir = os.path.join(base_dir, "attack_output", ATTACK_DIR_MAP[attack])
    if not os.path.isdir(atk_dir):
        atk_dir = os.path.join(base_dir, "connacf", "attack_output", ATTACK_DIR_MAP[attack])
    if not os.path.isdir(atk_dir):
        print(f"[auto_plot] No attack_output dir found for {attack} under {base_dir}", file=sys.stderr)
        return []

    # Build subdir pattern: canonical or variant-suffixed
    if variant:
        subdir_pat = re.compile(rf"^{re.escape(CANONICAL_SUBDIR[attack].pattern[1:-1])}_{re.escape(variant)}$"
                                if CANONICAL_SUBDIR[attack].pattern.endswith("$")
                                else rf".*_{re.escape(variant)}$")
        # Simpler: just require the variant suffix
        subdir_pat = re.compile(rf".*_{re.escape(variant)}$")
    else:
        subdir_pat = CANONICAL_SUBDIR[attack]

    runs = []
    for cfg_path in Path(atk_dir).rglob("experiment_config.json"):
        # cfg_path is: attack_output/<attack>/<config_subdir>/[<llm>/]<timestamp>/experiment_config.json
        # The config_subdir is the first component after atk_dir
        rel = cfg_path.relative_to(atk_dir)
        config_subdir = rel.parts[0]

        if not subdir_pat.match(config_subdir):
            continue

        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue

        dataset = cfg.get("dataset_name", "")
        ncand   = cfg.get("attack_config", {}).get("num_candidates")
        llm     = cfg.get("recbole_config", {}).get("llm_model", "")
        ts      = cfg_path.parent.name  # timestamped dir

        if ncand is None:
            continue

        dens  = density_label(dataset)
        seed  = seed_label(dataset)
        llm_s = llm_short(llm)

        if llm_filter and llm_filter.lower() not in llm_s.lower():
            continue
        if seed_filter and seed not in seed_filter:
            continue
        if cand_filter and int(ncand) not in cand_filter:
            continue
        if density_filter and dens not in density_filter:
            continue

        turn_count = len(list(cfg_path.parent.glob("turn_*.json")))
        if turn_count == 0:
            continue

        runs.append({
            "dir":      str(cfg_path.parent),
            "dataset":  dataset,
            "ncand":    int(ncand),
            "density":  dens,
            "seed":     seed,
            "llm":      llm_s,
            "ts":       ts,
            "turns":    turn_count,
        })

    return runs


def pick_latest(runs: list) -> list:
    """For each (ncand, density, seed, llm) group, keep only the latest run."""
    groups: dict = {}
    for r in runs:
        key = (r["ncand"], r["density"], r["seed"], r["llm"])
        if key not in groups or r["ts"] > groups[key]["ts"]:
            groups[key] = r
    return sorted(groups.values(), key=lambda r: (r["ncand"], r["density"] or "", r["seed"] or ""))


def make_label(r: dict, plot_type: str) -> str:
    cand = f"{r['ncand']}c"
    seed = f"s{r['seed']}" if r["seed"] else ""
    if plot_type == "uidensity":
        # Include density so 2c-dense50 and 2c-sparse200 don't collide
        return f"{cand} {r['density']} {seed}".strip()
    else:
        return f"{r['density']} {seed}".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attack",    required=True, choices=list(ATTACK_DIR_MAP))
    parser.add_argument("--plot-type", default="uidensity", choices=["uidensity", "intermat"])
    parser.add_argument("--variant",   default=None,
                        help="Config variant suffix to scan instead of canonical (e.g. tguard, blindguard, g_safeguard)")
    parser.add_argument("--llm",       default=None, help="Filter by LLM short name (e.g. qwen3, claude)")
    parser.add_argument("--seeds",     nargs="+", default=[], help="Filter by seed(s)")
    parser.add_argument("--cands",     nargs="+", type=int, default=[], help="Filter by num_candidates")
    parser.add_argument("--densities", nargs="+", default=[], choices=["50i", "100i", "200i"])
    parser.add_argument("--base-dir",  default=".", help="Repo root (default: cwd)")
    parser.add_argument("--output",    "-o", default=None)
    parser.add_argument("--title",     default=None)
    parser.add_argument("--last_epoch", type=int, default=100)
    parser.add_argument("--pub",       action="store_true")
    parser.add_argument("--dry-run",   action="store_true", help="Print command without running")
    parser.add_argument("--dirs-only", action="store_true", help="Print discovered dirs and exit, no plotting")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    runs = scan_runs(args.attack, base, args.llm, args.seeds,
                     args.cands, args.densities, args.variant)

    if not runs:
        print("[auto_plot] No matching runs found.", file=sys.stderr)
        sys.exit(1)

    latest = pick_latest(runs)
    variant_tag = f" ({args.variant})" if args.variant else ""
    print(f"[auto_plot] Found {len(latest)} run(s) for {args.attack}{variant_tag}:")
    for r in latest:
        print(f"  {r['ncand']}c {r['density']} seed={r['seed']} llm={r['llm']} "
              f"turns={r['turns']} ts={r['ts']}")
        if args.dirs_only:
            print(f"    -> {r['dir']}")

    if args.dirs_only:
        sys.exit(0)

    dirs   = [r["dir"]                       for r in latest]
    labels = [make_label(r, args.plot_type)  for r in latest]

    tool, atk_flag = PLOT_TOOL[args.attack]
    ext    = "pdf" if args.pub else "png"
    output = args.output or f"../figure/{args.attack}_{args.plot_type}_auto.{ext}"
    title  = args.title or (
        f"{args.attack.upper()}{variant_tag}: "
        f"{'Inference-Time Density' if args.plot_type == 'uidensity' else 'Interaction Matrix Density'} (auto)"
    )

    dirs_flag = "--dirs" if args.attack in USES_DIRS_FLAG else "--connacf_dirs"

    cmd = [sys.executable, tool,
           "--output", output,
           "--title",  title,
           "--last_epoch", str(args.last_epoch),
           "--labels"] + labels + [dirs_flag] + dirs

    if atk_flag:
        cmd += ["--attack", atk_flag]
    if args.attack == "toma":
        cmd += ["--sub_round", "0"]
    if args.pub:
        cmd.append("--pub")

    print(f"\n[auto_plot] Running: {' '.join(cmd)}\n")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
