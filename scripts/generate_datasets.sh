#!/usr/bin/env bash
# Generate all per-seed datasets without launching any experiments.
# Usage: bash scripts/generate_datasets.sh [connacf_dir] [seed1 seed2 ...] [--shuffle-graph] [--force]
# Example: bash scripts/generate_datasets.sh . 42 43 44
set -e

CONNACF="${1:-$(cd "$(cd "$(dirname "$0")/.." && pwd)/connacf" && pwd)}"
shift 1 2>/dev/null || true
SHUFFLE_FLAG=""; FORCE_FLAG=""
SEEDS=()
for arg in "$@"; do
  case "$arg" in
    --shuffle-graph) SHUFFLE_FLAG="--shuffle-graph" ;;
    --force)         FORCE_FLAG="--force" ;;
    *)               SEEDS+=("$arg") ;;
  esac
done
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(42 43 44); fi

echo "CONNACF=$CONNACF  SEEDS=${SEEDS[*]}"
cd "$CONNACF"

for seed in "${SEEDS[@]}"; do
  echo "=== Seed $seed ==="
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items  50 --label dense  --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 100 --label medium --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 200 --label sparse --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
done

echo "=== Done: datasets generated for seeds ${SEEDS[*]} ==="
