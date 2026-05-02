#!/usr/bin/env python3
"""
G-Safeguard two-stage orchestrator.

Stage 1 (already running): gs_* tmux sessions run the attack with
  enable_defense: false — pure data collection, no GNN overhead.

Stage 2 (this script): once each session has >= TARGET_TURNS turn JSONs,
  - pool the labeled interaction graphs from turn JSONs
  - train the GNN
  - patch all gsafeguard YAMLs with the new checkpoint path
  - relaunch each gs_* session as a fresh guarded run

Run from connacf/:
    python gsafeguard_train_and_relaunch.py [--target-turns N] [--dry-run]
"""
import argparse
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE, "defense", "checkpoints", "gsafeguard")

# Each entry: (tmux_session, dataset, attack_config)
SESSIONS = [
    ("gs_1cand",    "ml-100k-100user-medium100", "attack_config/gsafeguard/netsafe_misinfo_1cand_g_safeguard.yaml"),
    ("gs_2cand",    "ml-100k-100user-medium100", "attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml"),
    ("gs_2cand_d50","ml-100k-100user-dense50",   "attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml"),
    ("gs_2cand_s200","ml-100k-100user-sparse200","attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml"),
    ("gs_3cand",    "ml-100k-100user-medium100", "attack_config/gsafeguard/netsafe_misinfo_3cand_g_safeguard.yaml"),
]

# Output-dir prefix per config (must match YAML output.output_directory)
CONFIG_OUTPUT_PREFIX = {
    "attack_config/gsafeguard/netsafe_misinfo_1cand_g_safeguard.yaml": "attack_output/netsafe/misinfo_1cand_g_safeguard",
    "attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml": "attack_output/netsafe/misinfo_2cand_g_safeguard",
    "attack_config/gsafeguard/netsafe_misinfo_3cand_g_safeguard.yaml": "attack_output/netsafe/misinfo_3cand_g_safeguard",
}


def latest_run_dir(config: str, dataset: str, min_timestamp: str = "") -> str | None:
    """Return the most-recently-created timestamped run dir for this config+dataset.

    If min_timestamp is given (e.g. '260406000000'), only dirs with a timestamp
    lexicographically >= min_timestamp are considered.
    """
    prefix = os.path.join(BASE, CONFIG_OUTPUT_PREFIX[config], dataset)
    candidates = sorted(glob.glob(os.path.join(prefix, "*", "*")))
    valid = [
        d for d in candidates
        if os.path.isdir(d)
        and glob.glob(os.path.join(d, "turn_*.json"))
        and os.path.basename(d) >= min_timestamp
    ]
    return valid[-1] if valid else None


def turn_count(run_dir: str) -> int:
    return len(glob.glob(os.path.join(run_dir, "turn_*.json")))


def all_ready(target: int, min_timestamp: str = "") -> dict[str, str] | None:
    """Return {session: run_dir} if all sessions have >= target turns, else None."""
    result = {}
    for session, dataset, config in SESSIONS:
        rd = latest_run_dir(config, dataset, min_timestamp)
        n = turn_count(rd) if rd else 0
        print(f"  {session}: {n} turns  ({rd or 'no run dir found'})", flush=True)
        if n < target:
            return None
        result[session] = rd
    return result


def build_training_graphs(run_dirs: dict[str, str]) -> list:
    """
    Build labeled PyG-compatible graph dicts from turn JSONs.

    Each turn JSON contains dissemination.llm_judge.all_contamination_scores
    (one per non-attacker agent) and communication_graph edges.  We use the
    contamination score > threshold as the attacker label proxy for training.
    """
    import numpy as np

    graphs = []
    CONTAM_THRESHOLD = 0.3  # score above this → treat as "attacker" for GNN training

    for session, run_dir in run_dirs.items():
        turn_files = sorted(glob.glob(os.path.join(run_dir, "turn_*.json")))
        for tf in turn_files:
            try:
                d = json.load(open(tf))
            except Exception:
                continue

            # --- node features: use contamination scores as proxy embeddings ---
            scores = (d.get("dissemination", {})
                       .get("llm_judge", {})
                       .get("all_contamination_scores", []))
            if not scores:
                continue

            n_nodes = len(scores)
            # Simple 1-D feature: contamination score (GNN will learn from topology)
            node_features = np.array(scores, dtype=np.float32).reshape(-1, 1)

            # --- labels ---
            labels = (node_features[:, 0] > CONTAM_THRESHOLD).astype(np.int64)

            # --- edges from communication_graph ---
            cg = d.get("communication_graph", {})
            # edges stored as list of [src, dst] pairs if present, else empty
            edge_index = cg.get("edges", [])
            if edge_index:
                ei = np.array(edge_index, dtype=np.int64).T  # shape (2, E)
            else:
                # fallback: fully connected among contaminated nodes
                ei = np.zeros((2, 0), dtype=np.int64)

            graphs.append({
                "x": node_features,
                "edge_index": ei,
                "y": labels,
                "turn": d.get("turn", 0),
                "session": session,
            })

    print(f"  Built {len(graphs)} graph snapshots from {len(run_dirs)} sessions.", flush=True)
    return graphs


def train(graphs: list, dry_run: bool) -> str:
    """Train GNN on pooled graphs, return checkpoint path."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    pooled_path = os.path.join(CKPT_DIR, "gsafeguard_pooled_data.pkl")
    with open(pooled_path, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {len(graphs)} graphs to {pooled_path}", flush=True)

    if dry_run:
        fake_ckpt = os.path.join(CKPT_DIR, "gsafeguard_dry_run.pth")
        open(fake_ckpt, "w").close()
        print(f"  [DRY RUN] Skipping GNN training, fake checkpoint: {fake_ckpt}", flush=True)
        return fake_ckpt

    sys.path.insert(0, os.path.dirname(BASE))  # add ConnaCF root so 'connacf' is importable
    from connacf.defense.train_gnn import train_gnn
    ckpt = train_gnn(
        dataset_path=pooled_path,
        save_dir=CKPT_DIR,
        hidden_dim=1024, num_heads=8, num_layers=2,
        dropout=0.2, epochs=30, device="cpu",
    )
    print(f"  Trained checkpoint: {ckpt}", flush=True)
    return ckpt


def patch_yamls(ckpt_path: str):
    """Enable defense and update gnn_checkpoint_path in all gsafeguard YAMLs."""
    yaml_dir = os.path.join(BASE, "attack_config", "gsafeguard")
    for fname in os.listdir(yaml_dir):
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(yaml_dir, fname)
        content = open(fpath).read()
        new_content = re.sub(
            r'gnn_checkpoint_path:.*',
            f'gnn_checkpoint_path: "{ckpt_path}"',
            content,
        )
        new_content = re.sub(
            r'enable_defense:\s*false',
            'enable_defense: true',
            new_content,
        )
        if new_content != content:
            open(fpath, "w").write(new_content)
            print(f"  Patched {fname}", flush=True)


def disable_defense_in_yamls():
    """Set enable_defense: false in all gsafeguard YAMLs (for stage-1 passive runs)."""
    yaml_dir = os.path.join(BASE, "attack_config", "gsafeguard")
    for fname in os.listdir(yaml_dir):
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(yaml_dir, fname)
        content = open(fpath).read()
        new_content = re.sub(
            r'enable_defense:\s*true',
            'enable_defense: false',
            content,
        )
        if new_content != content:
            open(fpath, "w").write(new_content)
            print(f"  Disabled defense in {fname}", flush=True)


def kill_sessions():
    for session, _, _ in SESSIONS:
        subprocess.run(["tmux", "send-keys", "-t", session, "C-c", ""], check=False)
    time.sleep(2)


def relaunch(dry_run: bool):
    for session, dataset, config in SESSIONS:
        cmd = f"cd {BASE} && python3 attack_connacf.py -q -d {dataset} -a {config}"
        print(f"  [{session}] {cmd}", flush=True)
        if not dry_run:
            # Create session if it doesn't exist, then send the command
            subprocess.run(["tmux", "new-session", "-d", "-s", session], check=False)
            subprocess.run(["tmux", "send-keys", "-t", session, cmd, "Enter"], check=False)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-turns", type=int, default=30,
                        help="Minimum turns each session must have before training (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build graphs and show plan but skip GNN training and relaunch")
    parser.add_argument("--stage1", action="store_true",
                        help="Launch stage-1 passive runs (disable defense, start sessions)")
    parser.add_argument("--skip-stage1", metavar="CKPT",
                        help="Skip stage-1 entirely: use this existing checkpoint for stage-2 directly")
    parser.add_argument("--min-timestamp", default="",
                        help="Only consider run dirs with timestamp >= this value (e.g. 260406000000)")
    args = parser.parse_args()

    if args.stage1:
        print("Stage 1: disabling defense in YAMLs and launching passive runs...", flush=True)
        disable_defense_in_yamls()
        relaunch(dry_run=False)
        print("Stage-1 runs launched. Re-run without --stage1 to watch and trigger stage 2.", flush=True)
        return

    if args.skip_stage1:
        ckpt = os.path.abspath(args.skip_stage1)
        if not os.path.isfile(ckpt):
            print(f"ERROR: checkpoint not found: {ckpt}", flush=True)
            sys.exit(1)
        print(f"Skipping stage-1. Using existing checkpoint: {ckpt}", flush=True)
        patch_yamls(ckpt)
        relaunch(dry_run=args.dry_run)
        print("Stage-2 guarded runs launched.", flush=True)
        return

    print(f"G-Safeguard orchestrator — target: {args.target_turns} turns per session", flush=True)

    min_ts = args.min_timestamp
    if not min_ts:
        # Default: only consider runs started today or later
        from datetime import datetime
        min_ts = datetime.now().strftime("%y%m%d") + "000000"
        print(f"  (auto min-timestamp: {min_ts})", flush=True)

    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] Checking sessions...", flush=True)
        run_dirs = all_ready(args.target_turns, min_ts)
        if run_dirs:
            break
        print(f"  Not ready yet. Sleeping 60s...", flush=True)
        time.sleep(60)

    print(f"\nAll sessions ready. Killing stage-1 runs...", flush=True)
    if not args.dry_run:
        kill_sessions()

    print("Building training graphs from turn JSONs...", flush=True)
    graphs = build_training_graphs(run_dirs)
    if not graphs:
        print("ERROR: no graphs built — check turn JSON structure.", flush=True)
        sys.exit(1)

    print("Training GNN...", flush=True)
    ckpt = train(graphs, args.dry_run)

    print("Patching YAML configs with checkpoint path and enabling defense...", flush=True)
    patch_yamls(ckpt)

    print("Relaunching guarded runs...", flush=True)
    relaunch(args.dry_run)

    print("\nDone. Stage-2 guarded runs launched.", flush=True)


if __name__ == "__main__":
    main()
