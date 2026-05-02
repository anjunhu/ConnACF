#!/usr/bin/env bash
# Role-split ablation: CheatAgent, DrunkAgent, RecTextAttack × full matrix
# (NetSafe covered by launch_netsafe_corba_seeds.sh)
# Usage: bash launch_netsafe_role_split_ablation.sh [connacf_dir] [llm_model] [seed1 seed2 ...]
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

echo "=== Launching role-split ablation ==="
for seed in "${SEEDS[@]}"; do
  for atk_cfg in \
    "cheat:cheat/cheat_1cand.yaml:1c_100" \
    "cheat:cheat/cheat_2cand.yaml:2c_050" \
    "cheat:cheat/cheat_2cand.yaml:2c_100" \
    "cheat:cheat/cheat_2cand.yaml:2c_200" \
    "cheat:cheat/cheat_3cand.yaml:3c_100" \
    "drunk:drunk/drunk_1cand.yaml:1c_100" \
    "drunk:drunk/drunk_2cand.yaml:2c_050" \
    "drunk:drunk/drunk_2cand.yaml:2c_100" \
    "drunk:drunk/drunk_2cand.yaml:2c_200" \
    "drunk:drunk/drunk_3cand.yaml:3c_100" \
    "rta:rectextattack/rectextattack_textfooler_1cand.yaml:1c_100" \
    "rta:rectextattack/rectextattack_textfooler_2cand.yaml:2c_050" \
    "rta:rectextattack/rectextattack_textfooler_2cand.yaml:2c_100" \
    "rta:rectextattack/rectextattack_textfooler_2cand.yaml:2c_200" \
    "rta:rectextattack/rectextattack_textfooler_3cand.yaml:3c_100"; do
    name="${atk_cfg%%:*}"; rest="${atk_cfg#*:}"; cfg="${rest%%:*}"; label="${rest#*:}"
    case "$label" in
      *050*) D="ml-100k-100user-dense50-seed${seed}" ;;
      *200*) D="ml-100k-100user-sparse200-seed${seed}" ;;
      *)     D="ml-100k-100user-medium100-seed${seed}" ;;
    esac
    launch "${name}_${label}_${seed}" "python3 attack_connacf.py -q -d $D -l $LLM -a attack_config/$cfg"
  done
done

echo "=== Done: $((${#SEEDS[@]} * 15)) sessions launched ==="
