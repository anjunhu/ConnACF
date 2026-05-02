#!/usr/bin/env bash
# Run ONE ConnaCF experiment on the first free GPU (Bedrock-free: local agent + Anthropic judge).
#
# Usage:
#   bash scripts/run_local.sh <session_name> <dataset> <attack_config> [hf_alias] [judge_alias]
#
# Examples:
#   bash scripts/run_local.sh netsafe-1c-42 ml-100k-100user-medium100-seed42 attack_config/netsafe/misinfo_1cand.yaml
#   bash scripts/run_local.sh corba-1c-42   ml-100k-100user-medium100-seed42 attack_config/corba/corba_canonical_1cand.yaml
#   bash scripts/run_local.sh netsafe-1c-42 ml-100k-100user-medium100-seed42 attack_config/netsafe/misinfo_1cand.yaml hf-qwen3-1.7b anthropic-sonnet-4-6
set -euo pipefail

SESSION="${1:?Usage: $0 <session> <dataset> <attack_config> [hf_alias] [judge_alias] [gpu_id]}"
DATASET="${2:?}"
CONFIG="${3:?}"
HF_ALIAS="${4:-hf-qwen3-8b}"
JUDGE="${5:-anthropic-sonnet-4-5}"
GPU_ARG="${6:-}"

# Locate connacf/
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$_dir/../connacf" ]]; then
  CONNACF="$(cd "$_dir/.." && pwd)/connacf"
elif [[ -d "$PWD/connacf" ]]; then
  CONNACF="$PWD/connacf"
else
  echo "ERROR: cannot locate connacf/. Set CONNACF env var." >&2; exit 1
fi

# Use explicit GPU if provided, otherwise pick first with >17000 MiB free
FREE_GPU=""
if [[ -n "$GPU_ARG" ]]; then
  FREE_GPU="$GPU_ARG"
elif command -v nvidia-smi &>/dev/null; then
  while IFS=', ' read -r idx free; do
    if [[ "$free" -gt 17000 ]]; then FREE_GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
fi

if [[ -z "$FREE_GPU" ]]; then
  echo "ERROR: no free GPU found (>17000 MiB)" >&2; exit 1
fi

CMD="conda activate connacf && CUDA_VISIBLE_DEVICES=$FREE_GPU python3 attack_connacf.py -q -d $DATASET -l $HF_ALIAS -j $JUDGE -a $CONFIG"

echo "Session : $SESSION"
echo "GPU     : $FREE_GPU"
echo "Agent   : $HF_ALIAS"
echo "Judge   : $JUDGE"
echo "Dataset : $DATASET"
echo "Config  : $CONFIG"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$CONNACF"
tmux send-keys -t "$SESSION" "$CMD" Enter
echo "Launched tmux session: $SESSION"
