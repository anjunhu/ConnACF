#!/usr/bin/env bash
# plot_ratio_robustness.sh
# Generates Figure 12: Attacker Ratio Robustness (transposed).
# One panel per topology config; curves = 10% / 25% / 50% / 75% attackers.
#
# Run from the ConnaCF repo root:
#   bash scripts/plot_ratio_robustness.sh [--pub] [--png]
#
# Outputs: figure/pdf/ablations/ablation_ratio_robustness_{cfg}.pdf  (or .png)

set -euo pipefail
cd "$(dirname "$0")/../connacf"

PUB=""
EXT="pdf"
for arg in "$@"; do
    [[ "$arg" == "--pub" ]] && PUB="--pub"
    [[ "$arg" == "--png" ]] && EXT="png"
done

OUTDIR="../figure/${EXT}/ablations"
mkdir -p "$OUTDIR"

# Resolve latest run dir matching attack/dataset glob
run() {
    local atk_dir="$1" ds_glob="$2" llm_glob="$3"
    find "$atk_dir" -maxdepth 3 -type d -name "$llm_glob" 2>/dev/null \
        | grep -F "$ds_glob" | sort | tail -1
}

# 25pct baseline dirs (from main qwen3 runs)
dirs_25pct() {
    case "$1" in
        1c-100) echo "NETSAFE_1C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_1C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_1C_MEDIUM100_25PCT_SEED44_DIR" ;;
        2c-050) echo "NETSAFE_2C_DENSE50_25PCT_SEED42_DIR NETSAFE_2C_DENSE50_25PCT_SEED43_DIR NETSAFE_2C_DENSE50_25PCT_SEED44_DIR" ;;
        2c-100) echo "NETSAFE_2C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_2C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_2C_MEDIUM100_25PCT_SEED44_DIR" ;;
        2c-200) echo "NETSAFE_2C_SPARSE200_25PCT_SEED42_DIR NETSAFE_2C_SPARSE200_25PCT_SEED43_DIR NETSAFE_2C_SPARSE200_25PCT_SEED44_DIR" ;;
        3c-100) echo "NETSAFE_3C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_3C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_3C_MEDIUM100_25PCT_SEED44_DIR" ;;
    esac
}

# cfg_spec: "cfg_key:attack_subdir:dataset_label"
for cfg_spec in \
    "1c-100:misinfo_1cand:medium100" \
    "2c-050:misinfo_2cand:dense50" \
    "2c-100:misinfo_2cand:medium100" \
    "2c-200:misinfo_2cand:sparse200" \
    "3c-100:misinfo_3cand:medium100"; do

    cfg="${cfg_spec%%:*}"; rest="${cfg_spec#*:}"
    atk="${rest%%:*}"; ds_label="${rest##*:}"

    DIRS=(); LABELS=()

    # 10pct / 50pct / 75pct from seeded runs
    for pct in 10pct 50pct 75pct; do
        for seed in 43 44; do
            d=NETSAFE_${cand}_${ds_label^^}_${pct}_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("$pct")
        done
    done

    # Insert 25pct between 10pct and 50pct
    DIRS_FINAL=(); LABELS_FINAL=()
    for i in "${!LABELS[@]}"; do
        [[ "${LABELS[$i]}" == "10pct" ]] && DIRS_FINAL+=("${DIRS[$i]}") && LABELS_FINAL+=("10pct")
    done
    for d in $(dirs_25pct "$cfg"); do
        [[ -d "$d" ]] && DIRS_FINAL+=("$d") && LABELS_FINAL+=("25pct")
    done
    for pct in 50pct 75pct; do
        for i in "${!LABELS[@]}"; do
            [[ "${LABELS[$i]}" == "$pct" ]] && DIRS_FINAL+=("${DIRS[$i]}") && LABELS_FINAL+=("$pct")
        done
    done

    if [[ ${#DIRS_FINAL[@]} -eq 0 ]]; then
        echo "SKIP $cfg — no data found"
        continue
    fi

    echo "Plotting $cfg (${#DIRS_FINAL[@]} runs)..."
    python3 tools/plot_dissemination.py --last_epoch 300 $PUB \
        --title "NetSafe: Attacker Ratio Robustness (${cfg})" \
        --output "$OUTDIR/ablation_ratio_robustness_${cfg}.$EXT" \
        --connacf_dirs "${DIRS_FINAL[@]}" \
        --labels "${LABELS_FINAL[@]}"
done

echo "Done. Outputs in $OUTDIR/"
