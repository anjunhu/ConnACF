#!/usr/bin/env python3
"""Automated Train-Then-Defend Pipeline for G-Safeguard / BlindGuard.

This script automates the full defense evaluation pipeline:
1. Phase 1: Run attack experiment to collect training data (no detection)
2. Phase 2: Train GNN on collected data
3. Phase 3: Run attack experiment with trained GNN (active defense)

Following the original G-Safeguard paper methodology, metrics from Phase 1
are NOT counted as defense results — they represent the attack baseline.
Only Phase 3 metrics represent actual defense performance.

Usage (matching attack_connacf.py interface):
    python3 -m connacf.defense.train_then_defend \\
        -q -d ml-100k-100user-dense \\
        -a attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml \\
        -l qwen.qwen3-235b-a22b-2507-v1:0

    # Skip Phase 1 if training data already exists:
    python3 -m connacf.defense.train_then_defend \\
        -d ml-100k-100user-dense \\
        -a attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml \\
        --training-data defense/training_data/existing_data.pkl \\
        --skip-phase1

    # Skip Phase 1 and 2 if checkpoint already exists:
    python3 -m connacf.defense.train_then_defend \\
        -d ml-100k-100user-dense \\
        -a attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml \\
        --checkpoint defense/checkpoints/existing_model.pth \\
        --skip-phase1 --skip-phase2
"""

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Ensure the repo root (parent of the connacf/ package) is on sys.path so that
# `import connacf` works whether this script is run from AgentCF/ or AgentCF/connacf/.
_here = os.path.dirname(os.path.abspath(__file__))          # .../connacf/defense/
_connacf_dir = os.path.dirname(_here)                        # .../connacf/
_repo_root = os.path.dirname(_connacf_dir)                   # .../AgentCF/
for _p in (_repo_root, _connacf_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_path: str) -> None:
    """Save config dict to YAML file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_attack_experiment(
    config_path: str,
    dataset: str,
    turns: int,
    phase_name: str,
    cwd: str = ".",
    quiet: bool = False,
    llm_model: str = None,
    gpu_id: int = None,
    judge_model: str = None,
    max_turns: int = None,
) -> str:
    """Run attack experiment via attack_connacf.py.
    
    Returns:
        Path to the experiment output directory (timestamped task dir).
    """
    logger.info(f"[{phase_name}] Starting attack experiment...")
    logger.info(f"[{phase_name}] Config: {config_path}")
    logger.info(f"[{phase_name}] Dataset: {dataset}")
    if max_turns is not None:
        logger.info(f"[{phase_name}] Max global turns: {max_turns} (epochs capped at 999)")
    logger.info(f"[{phase_name}] Turns (epochs): {turns}")
    
    # Use attack_connacf.py as the main entry point
    cmd = [
        sys.executable,
        "attack_connacf.py",
        "--model", "ConnaCF",
        "-d", dataset,
        "-e", "999" if max_turns is not None else str(turns),
        "-a", config_path,
    ]
    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])
    if quiet:
        cmd.append("-q")
    if llm_model:
        cmd.extend(["-l", llm_model])
    if gpu_id is not None:
        cmd.extend(["-g", str(gpu_id)])
    if judge_model:
        cmd.extend(["-j", judge_model])
    
    logger.info(f"[{phase_name}] Command: {' '.join(cmd)}")
    
    # Run with live output streaming for visibility
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    task_dir = None
    
    # Stream output and capture task directory
    for line in process.stdout:
        output_lines.append(line)
        print(f"  {line.rstrip()}")
        # Capture task directory
        if "Task directory (timestamped):" in line:
            task_dir = line.split(":")[-1].strip()
    
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"[{phase_name}] Experiment failed with return code {process.returncode}")
        logger.error(f"[{phase_name}] Last 50 lines of output:")
        for line in output_lines[-50:]:
            logger.error(f"  {line.rstrip()}")
        raise RuntimeError(f"Attack experiment failed in {phase_name}")
    
    logger.info(f"[{phase_name}] Experiment completed successfully")
    
    if task_dir:
        logger.info(f"[{phase_name}] Output directory: {task_dir}")
    
    return task_dir or ""


def train_gnn(
    dataset_path: str,
    save_dir: str,
    epochs: int = 20,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = None,
) -> str:
    """Train GNN model on collected data.
    
    Returns:
        Path to the saved checkpoint.
    """
    if device is None:
        from connacf.utils.gpu_utils import free_device
        device = free_device()
    logger.info("[PHASE 2] Starting GNN training...")
    logger.info(f"[PHASE 2] Dataset: {dataset_path}")
    logger.info(f"[PHASE 2] Save dir: {save_dir}")
    logger.info(f"[PHASE 2] Epochs: {epochs}")
    
    # Import here to avoid circular imports
    from connacf.defense.train_gnn import train_gnn as _train_gnn
    
    checkpoint_path = _train_gnn(
        dataset_path=dataset_path,
        save_dir=save_dir,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        weight_decay=0.0002,
        batch_size=batch_size,
        dropout=dropout,
        device=device,
    )
    
    logger.info(f"[PHASE 2] Training complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def create_phase1_config(base_config: dict, output_dir: str) -> dict:
    """Create Phase 1 config: data collection only, no active detection."""
    config = copy.deepcopy(base_config)

    if "attack" not in config:
        config["attack"] = {}
    if "defense" not in config["attack"]:
        config["attack"]["defense"] = {}

    defense = config["attack"]["defense"]
    # Disable detection; keep collect_training_data as-is (must be true in G-Safeguard configs)
    defense["enable_defense"] = False
    defense["gnn_checkpoint_path"] = ""

    if "output" not in config["attack"]:
        config["attack"]["output"] = {}
    config["attack"]["output"]["output_directory"] = os.path.join(
        output_dir, "phase1_data_collection"
    )
    return config


def create_phase3_config(
    base_config: dict,
    checkpoint_path: str,
    output_dir: str,
) -> dict:
    """Create Phase 3 config: active defense with trained GNN."""
    config = copy.deepcopy(base_config)
    
    # Ensure defense section exists
    if "attack" not in config:
        config["attack"] = {}
    if "defense" not in config["attack"]:
        config["attack"]["defense"] = {}
    
    defense = config["attack"]["defense"]
    
    # Phase 3: Active defense with trained checkpoint
    defense["enable_defense"] = True
    defense["gnn_checkpoint_path"] = checkpoint_path
    defense["collect_training_data"] = False  # Don't collect, just defend
    
    # Update output directory to indicate Phase 3
    if "output" not in config["attack"]:
        config["attack"]["output"] = {}
    config["attack"]["output"]["output_directory"] = os.path.join(
        output_dir, "phase3_defense_eval"
    )
    
    return config


def find_training_data(search_dir: str, filename_pattern: str = "*.pkl") -> Optional[str]:
    """Search for training data pickle file in a directory tree."""
    import glob
    
    # Try direct path first
    if os.path.isfile(search_dir):
        return search_dir
    
    # Search recursively
    pattern = os.path.join(search_dir, "**", filename_pattern)
    matches = glob.glob(pattern, recursive=True)
    
    # Filter for defense/training_data paths
    defense_matches = [m for m in matches if "training_data" in m or "defense" in m]
    if defense_matches:
        return defense_matches[0]
    
    if matches:
        return matches[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Automated Train-Then-Defend Pipeline for G-Safeguard/BlindGuard"
    )
    parser.add_argument(
        "-a", "--attack_config",
        type=str,
        required=True,
        help="Path to base attack config YAML file",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., ml-100k-100user-dense)",
    )
    parser.add_argument(
        "-e", "--epochs", "--turns",
        type=int,
        default=1,
        dest="turns",
        help="Number of attack epochs/turns for Phase 3 (default: 1)",
    )
    parser.add_argument(
        "--phase1-turns",
        type=int,
        default=30,
        dest="phase1_turns",
        help="Number of global simulation turns for Phase 1 clean data collection (default: 30)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce logging to WARNING level",
    )
    parser.add_argument(
        "-l", "--llm_model",
        type=str,
        default=None,
        help="Override LLM model (passed through to attack_connacf.py)",
    )
    parser.add_argument(
        "-g", "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID (passed through to attack_connacf.py)",
    )
    parser.add_argument(
        "-j", "--judge_model",
        type=str,
        default=None,
        help="LLM judge model (passed through to attack_connacf.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for all phases (default: auto-generated)",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to existing training data pickle (skips Phase 1 data collection)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to existing GNN checkpoint (skips Phase 1 and 2)",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (requires --training-data or --checkpoint)",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (requires --checkpoint)",
    )
    parser.add_argument(
        "--gnn-epochs",
        type=int,
        default=20,
        help="GNN training epochs (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for GNN training (default: cuda:<CUDA_VISIBLE_DEVICES> or cpu)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for running experiments (default: directory containing attack_connacf.py)",
    )
    
    args = parser.parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    if args.device is None:
        from connacf.utils.gpu_utils import free_device
        args.device = free_device()
    
    # Validate arguments
    if args.skip_phase1 and not args.training_data and not args.checkpoint:
        parser.error("--skip-phase1 requires --training-data or --checkpoint")
    if args.skip_phase2 and not args.checkpoint:
        parser.error("--skip-phase2 requires --checkpoint")
    
    # Determine working directory (where attack_connacf.py lives)
    if args.cwd:
        cwd = args.cwd
    else:
        # Always resolve cwd as the connacf/ package dir (where attack_connacf.py lives),
        # regardless of where the script is launched from.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)  # defense/ -> connacf/
        if not os.path.exists(os.path.join(cwd, "attack_connacf.py")):
            raise FileNotFoundError(
                f"Cannot find attack_connacf.py in {cwd}. "
                "Check that train_then_defend.py is inside connacf/defense/."
            )
    
    logger.info(f"Working directory: {os.path.abspath(cwd)}")
    
    # Load base config
    base_config = load_config(args.attack_config)
    
    # Generate output directory — always relative to repo root (parent of connacf/)
    # __file__ is connacf/defense/train_then_defend.py → dirname×3 = repo root
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_repo_root, args.output_dir)
    else:
        defense_method = base_config.get("attack", {}).get("defense", {}).get(
            "defense_method", "g-safeguard"
        )
        output_dir = os.path.join(_repo_root, f"attack_output_defense/{defense_method}/{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Pipeline output directory: {output_dir}")
    
    # Save pipeline metadata
    metadata = {
        "timestamp": timestamp,
        "base_config": args.attack_config,
        "dataset": args.dataset,
        "turns": args.turns,
        "phase1_turns": args.phase1_turns,
        "gnn_epochs": args.gnn_epochs,
        "device": args.device,
        "skip_phase1": args.skip_phase1,
        "skip_phase2": args.skip_phase2,
    }
    with open(os.path.join(output_dir, "pipeline_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    training_data_path = args.training_data
    checkpoint_path = args.checkpoint
    
    # ==================== PHASE 1: Data Collection ====================
    if not args.skip_phase1:
        logger.info("=" * 60)
        logger.info("PHASE 1: Training Data Collection (No Detection)")
        logger.info("=" * 60)
        
        # Create Phase 1 config
        phase1_config = create_phase1_config(base_config, output_dir)
        phase1_config_path = os.path.join(output_dir, "phase1_config.yaml")
        save_config(phase1_config, phase1_config_path)
        
        # Run Phase 1 experiment — stop after exactly phase1_turns global turns
        phase1_task_dir = run_attack_experiment(
            config_path=phase1_config_path,
            dataset=args.dataset,
            turns=args.phase1_turns,
            max_turns=args.phase1_turns,
            phase_name="PHASE 1",
            cwd=cwd,
            quiet=args.quiet,
            llm_model=args.llm_model,
            gpu_id=args.gpu_id,
            judge_model=args.judge_model,
        )
        
        # training_data_output in the config is relative to cwd (the connacf/ dir)
        training_data_rel = phase1_config["attack"]["defense"].get(
            "training_data_output",
            "defense/training_data/dataset.pkl",
        )
        training_data_path = os.path.join(cwd, training_data_rel)

        if not os.path.exists(training_data_path):
            # The pkl is saved relative to the task_dir inside phase1_data_collection,
            # not relative to cwd. Search output_dir first, then cwd as fallback.
            import glob
            pkl_name = os.path.basename(training_data_rel)
            search_roots = [output_dir, cwd]
            matches = []
            for root in search_roots:
                matches = glob.glob(os.path.join(root, "**", pkl_name), recursive=True)
                if matches:
                    break
            if matches:
                training_data_path = max(matches, key=os.path.getmtime)
                logger.info(f"[PHASE 1] Found training data via glob: {training_data_path}")

        logger.info(f"[PHASE 1] Training data path: {training_data_path}")
    else:
        logger.info("PHASE 1: SKIPPED (using existing training data)")
    
    # ==================== PHASE 2: GNN Training ====================
    if not args.skip_phase2:
        logger.info("=" * 60)
        logger.info("PHASE 2: GNN Model Training")
        logger.info("=" * 60)
        
        # Find training data if path doesn't exist directly
        if not training_data_path or not os.path.exists(training_data_path):
            logger.info(f"[PHASE 2] Training data not found at: {training_data_path}")
            logger.info(f"[PHASE 2] Searching in output directory...")
            
            found_path = find_training_data(output_dir)
            if found_path:
                training_data_path = found_path
                logger.info(f"[PHASE 2] Found training data at: {training_data_path}")
            else:
                raise FileNotFoundError(
                    f"Training data not found in {output_dir}. "
                    f"Run Phase 1 first or provide --training-data."
                )
        
        # Get GNN config from base config
        gnn_config = base_config.get("attack", {}).get("defense", {}).get(
            "gnn_config", {}
        )
        defense_method = base_config.get("attack", {}).get("defense", {}).get(
            "defense_method", "g-safeguard"
        )

        checkpoint_dir = os.path.join(output_dir, "checkpoints")

        if defense_method == "blindguard":
            from connacf.defense.train_blindguard import train_blindguard_from_dataset
            checkpoint_path = train_blindguard_from_dataset(
                dataset_path=training_data_path,
                save_dir=checkpoint_dir,
                input_dim=gnn_config.get("embedding_dim", 384),
                hidden_dim=gnn_config.get("hidden_dim", 512),
                output_dim=gnn_config.get("hidden_dim", 256),
                n_epochs=args.gnn_epochs,
            )
        else:  # g-safeguard (supervised)
            checkpoint_path = train_gnn(
                dataset_path=training_data_path,
                save_dir=checkpoint_dir,
                epochs=args.gnn_epochs,
                hidden_dim=gnn_config.get("hidden_dim", 1024),
                num_heads=gnn_config.get("num_heads", 8),
                num_layers=gnn_config.get("num_layers", 2),
                dropout=gnn_config.get("dropout", 0.2),
                device=args.device,
            )
        
        logger.info(f"[PHASE 2] Checkpoint saved to: {checkpoint_path}")
    else:
        logger.info("PHASE 2: SKIPPED (using existing checkpoint)")
    
    # ==================== PHASE 3: Defense Evaluation ====================
    logger.info("=" * 60)
    logger.info("PHASE 3: Defense Evaluation (Active Detection)")
    logger.info("=" * 60)
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"GNN checkpoint not found: {checkpoint_path}. "
            f"Run Phase 2 first or provide --checkpoint."
        )
    
    # Create Phase 3 config
    phase3_config = create_phase3_config(base_config, checkpoint_path, output_dir)
    phase3_config_path = os.path.join(output_dir, "phase3_config.yaml")
    save_config(phase3_config, phase3_config_path)
    
    # Run Phase 3 experiment
    phase3_output = run_attack_experiment(
        config_path=phase3_config_path,
        dataset=args.dataset,
        turns=args.turns,
        phase_name="PHASE 3",
        cwd=cwd,
        quiet=args.quiet,
        llm_model=args.llm_model,
        gpu_id=args.gpu_id,
        judge_model=args.judge_model,
    )
    
    # ==================== Summary ====================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training data: {training_data_path}")
    logger.info(f"GNN checkpoint: {checkpoint_path}")
    logger.info(f"Defense evaluation output: {phase3_output}")
    logger.info("")
    logger.info("NOTE: Only Phase 3 metrics represent actual defense performance.")
    logger.info("Phase 1 metrics are baseline (attack without defense).")
    
    # Save final summary
    summary = {
        "output_dir": output_dir,
        "training_data_path": training_data_path,
        "checkpoint_path": checkpoint_path,
        "phase3_output": phase3_output,
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
