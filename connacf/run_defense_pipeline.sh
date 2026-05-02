#!/bin/bash
# Convenience wrapper for the train-then-defend pipeline
#
# Usage:
#   ./run_defense_pipeline.sh <config> <dataset> [turns] [epochs]
#
# Examples:
#   ./run_defense_pipeline.sh attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml ml-100k-100user-dense
#   ./run_defense_pipeline.sh attack_config/gsafeguard/netsafe_misinfo_2cand_g_safeguard.yaml ml-100k-100user-dense 5 20

set -e

CONFIG=${1:?"Usage: $0 <config> <dataset> [turns] [epochs]"}
DATASET=${2:?"Usage: $0 <config> <dataset> [turns] [epochs]"}
TURNS=${3:-5}
EPOCHS=${4:-20}

echo "=============================================="
echo "G-Safeguard / BlindGuard Defense Pipeline"
echo "=============================================="
echo "Config:  $CONFIG"
echo "Dataset: $DATASET"
echo "Turns:   $TURNS"
echo "Epochs:  $EPOCHS"
echo "=============================================="

# Resolve the repo root (one level up from this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

python -m connacf.defense.train_then_defend \
    --config "connacf/$CONFIG" \
    --dataset "$DATASET" \
    --turns "$TURNS" \
    --epochs "$EPOCHS"

echo ""
echo "Pipeline complete! Check attack_output_defense/ for results."
