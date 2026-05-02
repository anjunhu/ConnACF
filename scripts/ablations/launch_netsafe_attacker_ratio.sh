#!/usr/bin/env bash
# NetSafe attacker-ratio ablation: 10%, 50%, 75% × full matrix
# Full matrix: 1c-100, 2c-050, 2c-100, 2c-200, 3c-100 (mirrors root launchers)
# Usage: bash launch_netsafe_attacker_ratio.sh [connacf_dir] [llm_model] [seed1 seed2 ...]
set -euo pipefail

CONNACF="${1:-$(cd "$(dirname "$0")/../.." && pwd)/connacf}"
LLM="${2:-qwen.qwen3-235b-a22b-2507-v1:0}"
shift 2 2>/dev/null || true
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(43 44); fi

echo "CONNACF=$CONNACF  LLM=$LLM  SEEDS=${SEEDS[*]}"

cd "$CONNACF"
echo "=== Generating datasets ==="
for seed in "${SEEDS[@]}"; do
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items  50 --label dense  --seed "$seed"
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 100 --label medium --seed "$seed"
  python3 tools/generate_ml100k_data_sparse_dense.py --source ml-100k --num-users 100 --target-items 200 --label sparse --seed "$seed"
done
echo "Datasets ready."

launch() {
  local session="$1" cmd="$2"
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
  tmux send-keys -t "$session" "cd $CONNACF && $cmd" Enter
  echo "  $session: launched"
}

echo "=== Launching ratio ablation ==="
for seed in "${SEEDS[@]}"; do
  for pct in 10pct 50pct 75pct; do
    launch "ns_1c_100_${pct}_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $LLM -a attack_config/netsafe/misinfo_1cand_${pct}.yaml"
    launch "ns_2c_050_${pct}_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-dense50-seed${seed}   -l $LLM -a attack_config/netsafe/misinfo_2cand_${pct}.yaml"
    launch "ns_2c_100_${pct}_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $LLM -a attack_config/netsafe/misinfo_2cand_${pct}.yaml"
    launch "ns_2c_200_${pct}_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-sparse200-seed${seed} -l $LLM -a attack_config/netsafe/misinfo_2cand_${pct}.yaml"
    launch "ns_3c_100_${pct}_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $LLM -a attack_config/netsafe/misinfo_3cand_${pct}.yaml"
  done
done

echo "=== Done: $((${#SEEDS[@]} * 15)) sessions launched ==="
