#!/usr/bin/env bash
# Launch M-Guard (MASTER + built-in defense subsystem) for a list of seeds.
# Entry point: attack_connacf.py (same as MASTER baseline — no GNN training phase needed).
# Uses attack_config/mguard/ configs (NOT master_with_defenses_*.yaml which uses the
# old MASTER-internal defenses key; mguard/ configs use the defense: subsystem).
# 5 canonical configs: 1c-medium, 2c-dense, 2c-medium, 2c-sparse, 3c-medium
#
# Usage: bash launch_mguard.sh [connacf_dir] [llm_model] [seed1 seed2 ...]
# Example: bash launch_mguard.sh . qwen.qwen3-235b-a22b-2507-v1:0 43 44
set -e

CONNACF="${1:-$(cd "$(cd "$(dirname "$0")/.." && pwd)/connacf" && pwd)}"
LLM="${2:-qwen.qwen3-235b-a22b-2507-v1:0}"
shift 2 2>/dev/null || true
SHUFFLE_FLAG=""; FORCE_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --shuffle-graph) SHUFFLE_FLAG="--shuffle-graph" ;;
    --force)         FORCE_FLAG="--force" ;;
    *)               SEEDS+=("$arg") ;;
  esac
done
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(43 44); fi

echo "CONNACF=$CONNACF  LLM=$LLM  SEEDS=${SEEDS[*]}"

# ── 1. Generate datasets ─────────────────────────────────────────────────────
echo "=== Generating datasets ==="
cd "$CONNACF"
for seed in "${SEEDS[@]}"; do
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items  50 --label dense  --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 100 --label medium --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 200 --label sparse --seed "$seed" $SHUFFLE_FLAG $FORCE_FLAG
done
echo "Datasets ready."

# ── 2. Helper ────────────────────────────────────────────────────────────────
launch() {
  local session="$1"; local cmd="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    local pane_pid py_pid
    pane_pid=$(tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null | head -1)
    py_pid=$(pstree -p "$pane_pid" 2>/dev/null | grep -oP 'python3\(\K[0-9]+' | head -1 || true)
    if [[ -n "$py_pid" ]]; then
      echo "  $session: already running (pid=$py_pid) — skipped"
      return 0
    fi
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  tmux new-session -d -s "$session"
  sleep 1
  tmux send-keys -t "$session" "conda activate connacf && cd $CONNACF && $cmd" Enter
  echo "  $session: launched"
}

# ── 3. M-Guard per seed ──────────────────────────────────────────────────────
# Config note: only 1/2/3cand mguard yamls exist (no density variants).
# 2c-dense and 2c-sparse reuse master_2cand_mguard.yaml with different -d datasets.
for seed in "${SEEDS[@]}"; do
  echo "=== Seed $seed ==="
  launch "mguard-1c-100-${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -a attack_config/mguard/master_1cand_mguard.yaml -l $LLM"
  launch "mguard-2c-050-${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-dense50-seed${seed}   -a attack_config/mguard/master_2cand_mguard.yaml -l $LLM"
  launch "mguard-2c-100-${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -a attack_config/mguard/master_2cand_mguard.yaml -l $LLM"
  launch "mguard-2c-200-${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-sparse200-seed${seed} -a attack_config/mguard/master_2cand_mguard.yaml -l $LLM"
  launch "mguard-3c-100-${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -a attack_config/mguard/master_3cand_mguard.yaml -l $LLM"
done

echo "=== All done: $((${#SEEDS[@]} * 5)) sessions × ${#SEEDS[@]} seeds ==="
