#!/usr/bin/env bash
# LLM consistency ablation: NetSafe full matrix × 5 non-flagship models
# Flagship Qwen3 covered by launch_netsafe_corba_seeds.sh
# Usage: bash launch_netsafe_llm_ablation.sh [connacf_dir] [seed1 seed2 ...]
set -euo pipefail

CONNACF="${1:-$(cd "$(dirname "$0")/../.." && pwd)/connacf}"
shift 1 2>/dev/null || true
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(43 44); fi

echo "CONNACF=$CONNACF  SEEDS=${SEEDS[*]}"

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

echo "=== Launching LLM ablation ==="
for seed in "${SEEDS[@]}"; do
  for alias_model in \
    "haiku:us.anthropic.claude-haiku-4-5-20251001-v1:0" \
    "llama4:us.meta.llama4-maverick-17b-instruct-v1:0" \
    "mixtral:mistral.mixtral-8x7b-instruct-v0:1" \
    "sonnet46:us.anthropic.claude-sonnet-4-6"; do
    alias="${alias_model%%:*}"
    model="${alias_model#*:}"
    launch "llm_${alias}_1c_100_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $model -a attack_config/netsafe/misinfo_1cand.yaml"
    launch "llm_${alias}_2c_050_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-dense50-seed${seed}   -l $model -a attack_config/netsafe/misinfo_2cand.yaml"
    launch "llm_${alias}_2c_100_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $model -a attack_config/netsafe/misinfo_2cand.yaml"
    launch "llm_${alias}_2c_200_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-sparse200-seed${seed} -l $model -a attack_config/netsafe/misinfo_2cand.yaml"
    launch "llm_${alias}_3c_100_${seed}" "python3 attack_connacf.py -q -d ml-100k-100user-medium100-seed${seed} -l $model -a attack_config/netsafe/misinfo_3cand.yaml"
  done
done

echo "=== Done: $((${#SEEDS[@]} * 25)) sessions launched ==="
