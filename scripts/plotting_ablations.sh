#!/usr/bin/env bash
# plotting_ablations.sh — publication figures for ablation studies
# Usage: bash plotting_ablations.sh [--pub]
#
# Ablation studies covered:
#   1. Attacker ratio  (10% / 50% / 75%) × {1c-100, 2c-050, 2c-100, 2c-200, 3c-100} × seeds 43,44
#      → 30 runs, output: ATTACK_OUTPUT_DIR_PLACEHOLDER
#
#   2. LLM model       (haiku / llama4 / mixtral / sonnet46) × 5 configs × seeds 43,44
#      → 50 runs, output: ATTACK_OUTPUT_DIR_PLACEHOLDER
#
#   3. Role-split      (CheatAgent / DrunkAgent / RecTextAttack) × 5 configs × seeds 43,44

set -euo pipefail
cd "$(dirname "$0")/../connacf" || exit 1

PUB=""
OUTDIR="../figure/ablations"
EXT="png"
if [[ "${1:-}" == "--pub" ]]; then
    PUB="--pub"
    OUTDIR="../figure/pdf/ablations"
    EXT="pdf"
fi
mkdir -p "$OUTDIR"

# ── X-axis cutoff (matches plotting.sh NETSAFE_EPOCHS) ──────
NETSAFE_EPOCHS=80

# latest DIR_GLOB — returns the most recent timestamped subdir matching a glob
latest() { ls -dt $1 2>/dev/null | head -1 || true; }

##############################################################
# HELPER: resolve latest run for a given config+dataset+llm
# Usage: run ATTACK_DIR DATASET LLM_GLOB
# e.g.:  run ATTACK_OUTPUT_DIR_PLACEHOLDER ml-100k-100user-medium100-seed43 "*haiku*"
##############################################################
run() {
    local base="$1" dataset="$2" llm_glob="$3"
    latest "${base}/${dataset}/${llm_glob}/*"
}

##############################################################
# 1. ATTACKER RATIO ABLATION
#    3 pcts × {1c-100, 2c-050, 2c-100, 2c-200, 3c-100} × 2 seeds = 30 runs
#    Plot: U-I density (1c/2c/3c on medium100) and InterMat (2c on dense/medium/sparse)
#    per attacker percentage, averaged over seeds
##############################################################

for pct in 10pct 50pct 75pct; do
    # U-I density: 1c / 2c / 3c on medium100, both seeds
    DIRS=()
    LABELS=()
    for cand in 1cand 2cand 3cand; do
        for seed in 43 44; do
            d=NETSAFE_${cand}_MEDIUM100_${pct}_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("${cand%cand}c")
        done
    done
    if [[ ${#DIRS[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: U-I Density (${pct} attackers)" \
            --output "$OUTDIR/ablation_ratio_${pct}_UIDensity.$EXT" \
            --connacf_dirs "${DIRS[@]}" --labels "${LABELS[@]}"
    fi

    # InterMat density: 2c on dense50 / medium100 / sparse200, both seeds
    DIRS=()
    LABELS=()
    for dataset_label in "dense50:50i" "medium100:100i" "sparse200:200i"; do
        ds="${dataset_label%%:*}"; lbl="${dataset_label##*:}"
        for seed in 43 44; do
            d=NETSAFE_2C_${ds^^}_${pct}_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("${lbl}")
        done
    done
    if [[ ${#DIRS[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: InterMat Density (2c, ${pct} attackers)" \
            --output "$OUTDIR/ablation_ratio_${pct}_InterMat.$EXT" \
            --connacf_dirs "${DIRS[@]}" --labels "${LABELS[@]}"
    fi
done

##############################################################
# 1b. ATTACKER RATIO ROBUSTNESS (transposed view)
#     One plot per config (1c-100, 2c-050, 2c-100, 2c-200, 3c-100)
#     4 curves per plot: 10pct / 25pct (default) / 50pct / 75pct attackers
##############################################################

# 25% baseline dirs (from main plotting.sh qwen3 runs, hardcoded)
# bash 3.2-compatible lookup for 25pct baseline dirs (no declare -A)
get_dirs_25pct() {
    case "$1" in
        1c-100) echo "NETSAFE_1C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_1C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_1C_MEDIUM100_25PCT_SEED44_DIR" ;;
        2c-050) echo "NETSAFE_2C_DENSE50_25PCT_SEED42_DIR NETSAFE_2C_DENSE50_25PCT_SEED43_DIR NETSAFE_2C_DENSE50_25PCT_SEED44_DIR" ;;
        2c-100) echo "NETSAFE_2C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_2C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_2C_MEDIUM100_25PCT_SEED44_DIR" ;;
        2c-200) echo "NETSAFE_2C_SPARSE200_25PCT_SEED42_DIR NETSAFE_2C_SPARSE200_25PCT_SEED43_DIR NETSAFE_2C_SPARSE200_25PCT_SEED44_DIR" ;;
        3c-100) echo "NETSAFE_3C_MEDIUM100_25PCT_SEED42_DIR NETSAFE_3C_MEDIUM100_25PCT_SEED43_DIR NETSAFE_3C_MEDIUM100_25PCT_SEED44_DIR" ;;
    esac
}

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
            d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                    "ml-100k-100user-${ds_label}-seed${seed}" "*")
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("$pct")
        done
    done

    # 25pct from hardcoded baseline dirs (insert between 10pct and 50pct)
    DIRS_FINAL=(); LABELS_FINAL=()
    # collect 10pct entries first
    for i in "${!LABELS[@]}"; do
        [[ "${LABELS[$i]}" == "10pct" ]] && DIRS_FINAL+=("${DIRS[$i]}") && LABELS_FINAL+=("10pct")
    done
    # insert 25pct
    for d in $(get_dirs_25pct "$cfg"); do
        [[ -d "$d" ]] && DIRS_FINAL+=("$d") && LABELS_FINAL+=("25pct")
    done
    # collect 50pct and 75pct
    for pct in 50pct 75pct; do
        for i in "${!LABELS[@]}"; do
            [[ "${LABELS[$i]}" == "$pct" ]] && DIRS_FINAL+=("${DIRS[$i]}") && LABELS_FINAL+=("$pct")
        done
    done

    if [[ ${#DIRS_FINAL[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: Attacker Ratio Robustness (${cfg})" \
            --output "$OUTDIR/ablation_ratio_robustness_${cfg}.$EXT" \
            --connacf_dirs "${DIRS_FINAL[@]}" --labels "${LABELS_FINAL[@]}"
    fi
done


#    5 models × {1c-100, 2c-050, 2c-100, 2c-200, 3c-100} × 2 seeds = 50 runs
#    Plot: U-I density (1c/2c/3c on medium100) per model
##############################################################

llm_glob() {
    case "$1" in
        haiku)    echo "*haiku*" ;;
        llama4)   echo "*llama4*" ;;
        mixtral)  echo "*mixtral*" ;;
        sonnet46) echo "*sonnet*4-6*" ;;
    esac
}

# U-I density + InterMat: one pair of plots per model, 3 curves each (seeds merged)
for model in haiku llama4 mixtral sonnet46; do
    glob=$(llm_glob "$model")

    # UIDensity: 1c/2c/3c on medium100, both seeds → 3 curves
    DIRS=(); LABELS=()
    for cand in 1cand 2cand 3cand; do
        for seed in 43 44; do
            d=NETSAFE_${cand}_MEDIUM100_${model^^}_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("${cand%cand}c")
        done
    done
    if [[ ${#DIRS[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: U-I Density (${model})" \
            --output "$OUTDIR/ablation_llm_${model}_UIDensity.$EXT" \
            --connacf_dirs "${DIRS[@]}" --labels "${LABELS[@]}"
    fi

    # InterMat: 2c on dense50/medium100/sparse200, both seeds → 3 curves
    DIRS=(); LABELS=()
    for ds_lbl in "dense50:50i" "medium100:100i" "sparse200:200i"; do
        ds="${ds_lbl%%:*}"; lbl="${ds_lbl##*:}"
        for seed in 43 44; do
            d=NETSAFE_2C_${ds^^}_${model^^}_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("$lbl")
        done
    done
    if [[ ${#DIRS[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: InterMat Density (${model})" \
            --output "$OUTDIR/ablation_llm_${model}_InterMat.$EXT" \
            --connacf_dirs "${DIRS[@]}" --labels "${LABELS[@]}"
    fi
done

# Cross-model comparison: 2c medium100, all models side-by-side (seed43 only for clarity)
##############################################################
# LLM COMPARISON: 5 configs × 6 models (5 ablation + Qwen3 baseline)
# Each plot = one config, 6 curves (one per model), seeds merged
# Configs: 1c-100, 2c-050, 2c-100, 2c-200, 3c-100
##############################################################

# Map config label → (attack_dir, dataset)
# qwen3 uses old-style path (no llm slug subdir, timestamp directly under dataset)
for cfg_spec in \
    "1c-100:misinfo_1cand:medium100" \
    "2c-050:misinfo_2cand:dense50" \
    "2c-100:misinfo_2cand:medium100" \
    "2c-200:misinfo_2cand:sparse200" \
    "3c-100:misinfo_3cand:medium100"; do
    cfg="${cfg_spec%%:*}"; rest="${cfg_spec#*:}"
    atk="${rest%%:*}"; ds_label="${rest##*:}"

    DIRS=(); LABELS=()
    for seed in 43 44; do
        dataset="ml-100k-100user-${ds_label}-seed${seed}"

        # Qwen3: timestamp sits directly under dataset dir (no llm slug)
        d=NETSAFE_${cfg}_QWEN3_SEED${seed}_DIR  # replace with actual path
        [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("qwen3")

        # 5 ablation models
        for model_slug in "haiku:claude-haiku-4-5" \
                          "llama4:llama4-maverick-17b" "mixtral:mixtral-8x7b" \
                          "sonnet46:claude-sonnet-4-6"; do
            label="${model_slug%%:*}"; slug="${model_slug##*:}"
            d=NETSAFE_${cfg}_QWEN3_SEED${seed}_DIR  # replace with actual path
            [[ -n "$d" ]] && DIRS+=("$d") && LABELS+=("$label")
        done
    done

    if [[ ${#DIRS[@]} -gt 0 ]]; then
        python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
            --title "NetSafe: LLM Robustness (${cfg})" \
            --output "$OUTDIR/ablation_llm_robustness_${cfg}.$EXT" \
            --connacf_dirs "${DIRS[@]}" --labels "${LABELS[@]}"
    fi
done

##############################################################
# 3. ROLE-SPLIT ABLATION
#    NetSafe / CheatAgent / DrunkAgent / RecTextAttack × 5 configs × 2 seeds
#    One combined figure: 1 attack per row, 4 panels per row
#    Row titles: attack name + alpha annotation
##############################################################

# Collect dirs for each attack into bash arrays, then emit JSON specs
# NetSafe — reuse main experiment dirs (same as plotting.sh)
NS_UID_DIRS=(
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
)
NS_UID_LABELS=("1c" "1c" "2c" "2c" "2c" "3c" "3c")
NS_IM_DIRS=(
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
    NETSAFE_${cfg}_${label^^}_SEED${seed}_DIR  # replace with actual path
)
NS_IM_LABELS=("200i" "100i" "50i" "200i" "100i" "50i")

# CheatAgent
CA_UID_DIRS=(); CA_UID_LABELS=(); CA_IM_DIRS=(); CA_IM_LABELS=()
for cand in 1cand 2cand 3cand; do
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-medium100-seed${seed}" "*")
        [[ -n "$d" ]] && CA_UID_DIRS+=("$d") && CA_UID_LABELS+=("${cand%cand}c")
    done
done
for ds_lbl in "dense50:50i" "medium100:100i" "sparse200:200i"; do
    ds="${ds_lbl%%:*}"; lbl="${ds_lbl##*:}"
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-${ds}-seed${seed}" "*")
        [[ -n "$d" ]] && CA_IM_DIRS+=("$d") && CA_IM_LABELS+=("$lbl")
    done
done

# DrunkAgent
DA_UID_DIRS=(); DA_UID_LABELS=(); DA_IM_DIRS=(); DA_IM_LABELS=()
for cand in 1cand 2cand 3cand; do
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-medium100-seed${seed}" "*")
        [[ -n "$d" ]] && DA_UID_DIRS+=("$d") && DA_UID_LABELS+=("${cand%cand}c")
    done
done
for ds_lbl in "dense50:50i" "medium100:100i" "sparse200:200i"; do
    ds="${ds_lbl%%:*}"; lbl="${ds_lbl##*:}"
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-${ds}-seed${seed}" "*")
        [[ -n "$d" ]] && DA_IM_DIRS+=("$d") && DA_IM_LABELS+=("$lbl")
    done
done

# RecTextAttack
RTA_UID_DIRS=(); RTA_UID_LABELS=(); RTA_IM_DIRS=(); RTA_IM_LABELS=()
for cand in 1cand 2cand 3cand; do
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-medium100-seed${seed}" "*")
        [[ -n "$d" ]] && RTA_UID_DIRS+=("$d") && RTA_UID_LABELS+=("${cand%cand}c")
    done
done
for ds_lbl in "dense50:50i" "medium100:100i" "sparse200:200i"; do
    ds="${ds_lbl%%:*}"; lbl="${ds_lbl##*:}"
    for seed in 43 44; do
        d=NETSAFE_${atk}_${ds_label}_${pct}_SEED${seed}_DIR  # replace with actual path  #
                "ml-100k-100user-${ds}-seed${seed}" "*")
        [[ -n "$d" ]] && RTA_IM_DIRS+=("$d") && RTA_IM_LABELS+=("$lbl")
    done
done

# Helper: build JSON spec string for one attack row
# Usage: make_spec ROW_TITLE UID_DIRS_ARRAY UID_LABELS_ARRAY IM_DIRS_ARRAY IM_LABELS_ARRAY
make_json_array() {
    local arr=("$@")
    local out="["
    for i in "${!arr[@]}"; do
        [[ $i -gt 0 ]] && out+=","
        out+="\"${arr[$i]}\""
    done
    out+="]"
    echo "$out"
}

# Build --attacks specs (only include rows that have data)
ATTACK_SPECS=()

if [[ ${#NS_UID_DIRS[@]} -gt 0 && ${#NS_IM_DIRS[@]} -gt 0 ]]; then
    NS_SPEC=$(printf '{"uid_dirs":%s,"uid_labels":%s,"im_dirs":%s,"im_labels":%s,"row_title":"NetSafe (aU>0, aI>0)"}' \
        "$(make_json_array "${NS_UID_DIRS[@]}")" \
        "$(make_json_array "${NS_UID_LABELS[@]}")" \
        "$(make_json_array "${NS_IM_DIRS[@]}")" \
        "$(make_json_array "${NS_IM_LABELS[@]}")")
    ATTACK_SPECS+=("$NS_SPEC")
fi

if [[ ${#CA_UID_DIRS[@]} -gt 0 && ${#CA_IM_DIRS[@]} -gt 0 ]]; then
    CA_SPEC=$(printf '{"uid_dirs":%s,"uid_labels":%s,"im_dirs":%s,"im_labels":%s,"row_title":"CheatAgent (aU>0, aI=0)"}' \
        "$(make_json_array "${CA_UID_DIRS[@]}")" \
        "$(make_json_array "${CA_UID_LABELS[@]}")" \
        "$(make_json_array "${CA_IM_DIRS[@]}")" \
        "$(make_json_array "${CA_IM_LABELS[@]}")")
    ATTACK_SPECS+=("$CA_SPEC")
fi

if [[ ${#DA_UID_DIRS[@]} -gt 0 && ${#DA_IM_DIRS[@]} -gt 0 ]]; then
    DA_SPEC=$(printf '{"uid_dirs":%s,"uid_labels":%s,"im_dirs":%s,"im_labels":%s,"row_title":"DrunkAgent (aU=0, aI>0)"}' \
        "$(make_json_array "${DA_UID_DIRS[@]}")" \
        "$(make_json_array "${DA_UID_LABELS[@]}")" \
        "$(make_json_array "${DA_IM_DIRS[@]}")" \
        "$(make_json_array "${DA_IM_LABELS[@]}")")
    ATTACK_SPECS+=("$DA_SPEC")
fi

if [[ ${#RTA_UID_DIRS[@]} -gt 0 && ${#RTA_IM_DIRS[@]} -gt 0 ]]; then
    RTA_SPEC=$(printf '{"uid_dirs":%s,"uid_labels":%s,"im_dirs":%s,"im_labels":%s,"row_title":"RecTextAttack"}' \
        "$(make_json_array "${RTA_UID_DIRS[@]}")" \
        "$(make_json_array "${RTA_UID_LABELS[@]}")" \
        "$(make_json_array "${RTA_IM_DIRS[@]}")" \
        "$(make_json_array "${RTA_IM_LABELS[@]}")")
    ATTACK_SPECS+=("$RTA_SPEC")
fi

if [[ ${#ATTACK_SPECS[@]} -gt 0 ]]; then
    python3 tools/plot_dissemination_compact.py $PUB \
        --output "$OUTDIR/ablation_rolesplit_combined.$EXT" \
        --last_epoch $NETSAFE_EPOCHS \
        --attacks "${ATTACK_SPECS[@]}"
fi

# Individual compact plots per attack (referenced by amlc_main.tex)
if [[ ${#NS_UID_DIRS[@]} -gt 0 && ${#NS_IM_DIRS[@]} -gt 0 ]]; then
    python3 tools/plot_dissemination_compact.py $PUB \
        --output "$OUTDIR/ablation_rolesplit_netsafe.$EXT" \
        --last_epoch $NETSAFE_EPOCHS --split \
        --uid_dirs "${NS_UID_DIRS[@]}" --uid_labels "${NS_UID_LABELS[@]}" \
        --im_dirs  "${NS_IM_DIRS[@]}"  --im_labels  "${NS_IM_LABELS[@]}"
fi
if [[ ${#CA_UID_DIRS[@]} -gt 0 && ${#CA_IM_DIRS[@]} -gt 0 ]]; then
    python3 tools/plot_dissemination_compact.py $PUB \
        --output "$OUTDIR/ablation_rolesplit_cheat.$EXT" \
        --last_epoch $NETSAFE_EPOCHS --split \
        --uid_dirs "${CA_UID_DIRS[@]}" --uid_labels "${CA_UID_LABELS[@]}" \
        --im_dirs  "${CA_IM_DIRS[@]}"  --im_labels  "${CA_IM_LABELS[@]}"
fi
if [[ ${#DA_UID_DIRS[@]} -gt 0 && ${#DA_IM_DIRS[@]} -gt 0 ]]; then
    python3 tools/plot_dissemination_compact.py $PUB \
        --output "$OUTDIR/ablation_rolesplit_drunk.$EXT" \
        --last_epoch $NETSAFE_EPOCHS --split \
        --uid_dirs "${DA_UID_DIRS[@]}" --uid_labels "${DA_UID_LABELS[@]}" \
        --im_dirs  "${DA_IM_DIRS[@]}"  --im_labels  "${DA_IM_LABELS[@]}"
fi
if [[ ${#RTA_UID_DIRS[@]} -gt 0 && ${#RTA_IM_DIRS[@]} -gt 0 ]]; then
    python3 tools/plot_dissemination_compact.py $PUB \
        --output "$OUTDIR/ablation_rolesplit_rta.$EXT" \
        --last_epoch $NETSAFE_EPOCHS --split \
        --uid_dirs "${RTA_UID_DIRS[@]}" --uid_labels "${RTA_UID_LABELS[@]}" \
        --im_dirs  "${RTA_IM_DIRS[@]}"  --im_labels  "${RTA_IM_LABELS[@]}"
fi

echo ""
echo "Done. Ablation figures written to $OUTDIR/"
