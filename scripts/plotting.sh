#!/usr/bin/env bash
# plotting.sh — merged from plotting.sh + plotting_260410.sh
# Usage: bash plotting.sh [--pub]
#   --pub  Publication preset: PDF output to figure/pdf/, 8pt font, figsize=(7,2.8), dpi=300
#
# Two plots per attack: (1) UIDensity — varies k on fixed medium100
#                       (2) InterMat  — varies rho at fixed k=2
# TOMA/MASTER additionally have full-layout variants (--single-row / --full).
# Placeholders: replace ATTACK_CAND_DENSITY_SEEDXX_DIR with actual run directories.

set -e
cd "$(dirname "$0")/../connacf" || exit 1

PUB=""
OUTDIR="../figure"
EXT="png"
if [[ "$1" == "--pub" ]]; then
    PUB="--pub"
    OUTDIR="../figure/pdf"
    EXT="pdf"
    mkdir -p "$OUTDIR"
fi

NETSAFE_EPOCHS=80
DEFENSE_EPOCHS=40

##############################################################
# DISSEMINATION: NetSafe
##############################################################

python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/netsafe_UIDensity_qwen3.$EXT" \
    --connacf_dirs \
        NETSAFE_1C_MEDIUM100_SEED42_DIR \
        NETSAFE_1C_MEDIUM100_SEED43_DIR \
        NETSAFE_1C_MEDIUM100_SEED44_DIR \
        NETSAFE_2C_MEDIUM100_SEED42_DIR \
        NETSAFE_2C_MEDIUM100_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_SEED44_DIR \
        NETSAFE_3C_MEDIUM100_SEED42_DIR \
        NETSAFE_3C_MEDIUM100_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c"

python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/netsafe_InterMat_qwen3.$EXT" \
    --connacf_dirs \
        NETSAFE_2C_SPARSE200_SEED42_DIR \
        NETSAFE_2C_SPARSE200_SEED43_DIR \
        NETSAFE_2C_SPARSE200_SEED44_DIR \
        NETSAFE_2C_MEDIUM100_SEED42_DIR \
        NETSAFE_2C_MEDIUM100_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_SEED44_DIR \
        NETSAFE_2C_DENSE50_SEED42_DIR \
        NETSAFE_2C_DENSE50_SEED43_DIR \
        NETSAFE_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i"


##############################################################
# DISSEMINATION: CORBA
##############################################################

python3 tools/plot_corba.py $PUB \
    --title "CORBA: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/corba_UIDensity_qwen3.$EXT" \
    --connacf_dirs \
        CORBA_1C_MEDIUM100_SEED42_DIR \
        CORBA_1C_MEDIUM100_SEED43_DIR \
        CORBA_1C_MEDIUM100_SEED44_DIR \
        CORBA_2C_MEDIUM100_SEED42_DIR \
        CORBA_2C_MEDIUM100_SEED43_DIR \
        CORBA_2C_MEDIUM100_SEED44_DIR \
        CORBA_3C_MEDIUM100_SEED42_DIR \
        CORBA_3C_MEDIUM100_SEED43_DIR \
        CORBA_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c"

python3 tools/plot_corba.py $PUB \
    --title "CORBA: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/corba_InterMat_qwen3.$EXT" \
    --connacf_dirs \
        CORBA_2C_SPARSE200_SEED42_DIR \
        CORBA_2C_SPARSE200_SEED43_DIR \
        CORBA_2C_SPARSE200_SEED44_DIR \
        CORBA_2C_MEDIUM100_SEED42_DIR \
        CORBA_2C_MEDIUM100_SEED43_DIR \
        CORBA_2C_MEDIUM100_SEED44_DIR \
        CORBA_2C_DENSE50_SEED42_DIR \
        CORBA_2C_DENSE50_SEED43_DIR \
        CORBA_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i"


##############################################################
# EXTRACTION: MAMA
##############################################################

python3 tools/plot_mama.py $PUB \
    --title "MAMA: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/mama_UIDensity_qwen3.$EXT" \
    --dirs \
        MAMA_1C_MEDIUM100_SEED42_DIR \
        MAMA_1C_MEDIUM100_SEED43_DIR \
        MAMA_1C_MEDIUM100_SEED44_DIR \
        MAMA_2C_MEDIUM100_SEED42_DIR \
        MAMA_2C_MEDIUM100_SEED43_DIR \
        MAMA_2C_MEDIUM100_SEED44_DIR \
        MAMA_3C_MEDIUM100_SEED42_DIR \
        MAMA_3C_MEDIUM100_SEED43_DIR \
        MAMA_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --last_epoch 100

python3 tools/plot_mama.py $PUB \
    --title "MAMA: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/mama_InterMat_qwen3.$EXT" \
    --dirs \
        MAMA_2C_SPARSE200_SEED42_DIR \
        MAMA_2C_SPARSE200_SEED43_DIR \
        MAMA_2C_SPARSE200_SEED44_DIR \
        MAMA_2C_MEDIUM100_SEED42_DIR \
        MAMA_2C_MEDIUM100_SEED43_DIR \
        MAMA_2C_MEDIUM100_SEED44_DIR \
        MAMA_2C_DENSE50_SEED42_DIR \
        MAMA_2C_DENSE50_SEED43_DIR \
        MAMA_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --last_epoch 100


##############################################################
# EXTRACTION: MASLeak
##############################################################

python3 tools/plot_reveng.py $PUB \
    --title "MASLeak: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/masleak_UIDensity_qwen3.$EXT" \
    --connacf_dirs \
        MASLEAK_1C_MEDIUM100_SEED42_DIR \
        MASLEAK_1C_MEDIUM100_SEED43_DIR \
        MASLEAK_1C_MEDIUM100_SEED44_DIR \
        MASLEAK_2C_MEDIUM100_SEED42_DIR \
        MASLEAK_2C_MEDIUM100_SEED43_DIR \
        MASLEAK_2C_MEDIUM100_SEED44_DIR \
        MASLEAK_3C_MEDIUM100_SEED42_DIR \
        MASLEAK_3C_MEDIUM100_SEED43_DIR \
        MASLEAK_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --last_epoch 60

python3 tools/plot_reveng.py $PUB \
    --title "MASLeak: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/masleak_InterMat_qwen3.$EXT" \
    --connacf_dirs \
        MASLEAK_2C_SPARSE200_SEED42_DIR \
        MASLEAK_2C_SPARSE200_SEED43_DIR \
        MASLEAK_2C_SPARSE200_SEED44_DIR \
        MASLEAK_2C_MEDIUM100_SEED42_DIR \
        MASLEAK_2C_MEDIUM100_SEED43_DIR \
        MASLEAK_2C_MEDIUM100_SEED44_DIR \
        MASLEAK_2C_DENSE50_SEED42_DIR \
        MASLEAK_2C_DENSE50_SEED43_DIR \
        MASLEAK_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --last_epoch 60


##############################################################
# BIDIRECTIONAL: TOMA
##############################################################

python3 tools/plot_toma.py $PUB \
    --title "TOMA: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/toma_UIDensity_qwen3.$EXT" \
    --dirs \
        TOMA_1C_MEDIUM100_SEED42_DIR \
        TOMA_1C_MEDIUM100_SEED43_DIR \
        TOMA_1C_MEDIUM100_SEED44_DIR \
        TOMA_2C_MEDIUM100_SEED42_DIR \
        TOMA_2C_MEDIUM100_SEED43_DIR \
        TOMA_2C_MEDIUM100_SEED44_DIR \
        TOMA_3C_MEDIUM100_SEED42_DIR \
        TOMA_3C_MEDIUM100_SEED43_DIR \
        TOMA_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --sub_round 0 --last_epoch 100

python3 tools/plot_toma.py $PUB \
    --title "TOMA: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/toma_InterMat_qwen3.$EXT" \
    --dirs \
        TOMA_2C_SPARSE200_SEED42_DIR \
        TOMA_2C_SPARSE200_SEED43_DIR \
        TOMA_2C_SPARSE200_SEED44_DIR \
        TOMA_2C_MEDIUM100_SEED42_DIR \
        TOMA_2C_MEDIUM100_SEED43_DIR \
        TOMA_2C_MEDIUM100_SEED44_DIR \
        TOMA_2C_DENSE50_SEED42_DIR \
        TOMA_2C_DENSE50_SEED43_DIR \
        TOMA_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --sub_round 0 --last_epoch 100

python3 tools/plot_toma.py $PUB --single-row \
    --title "TOMA: Inference-Time Density — 4-panel (Qwen3)" \
    --output "$OUTDIR/toma_UIDensity_qwen3_4panel.$EXT" \
    --dirs \
        TOMA_1C_MEDIUM100_SEED42_DIR \
        TOMA_1C_MEDIUM100_SEED43_DIR \
        TOMA_1C_MEDIUM100_SEED44_DIR \
        TOMA_2C_MEDIUM100_SEED42_DIR \
        TOMA_2C_MEDIUM100_SEED43_DIR \
        TOMA_2C_MEDIUM100_SEED44_DIR \
        TOMA_3C_MEDIUM100_SEED42_DIR \
        TOMA_3C_MEDIUM100_SEED43_DIR \
        TOMA_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --sub_round 0 --last_epoch 100

python3 tools/plot_toma.py $PUB --single-row \
    --title "TOMA: Interaction Matrix Density — 4-panel (2c, Qwen3)" \
    --output "$OUTDIR/toma_InterMat_qwen3_4panel.$EXT" \
    --dirs \
        TOMA_2C_SPARSE200_SEED42_DIR \
        TOMA_2C_SPARSE200_SEED43_DIR \
        TOMA_2C_SPARSE200_SEED44_DIR \
        TOMA_2C_MEDIUM100_SEED42_DIR \
        TOMA_2C_MEDIUM100_SEED43_DIR \
        TOMA_2C_MEDIUM100_SEED44_DIR \
        TOMA_2C_DENSE50_SEED42_DIR \
        TOMA_2C_DENSE50_SEED43_DIR \
        TOMA_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --sub_round 0 --last_epoch 100


##############################################################
# BIDIRECTIONAL: MASTER
##############################################################

python3 tools/plot_master.py $PUB \
    --title "MASTER: Inference-Time Density (Qwen3)" \
    --output "$OUTDIR/master_UIDensity_qwen3.$EXT" \
    --dirs \
        MASTER_1C_MEDIUM100_SEED42_DIR \
        MASTER_1C_MEDIUM100_SEED43_DIR \
        MASTER_1C_MEDIUM100_SEED44_DIR \
        MASTER_2C_MEDIUM100_SEED42_DIR \
        MASTER_2C_MEDIUM100_SEED43_DIR \
        MASTER_2C_MEDIUM100_SEED44_DIR \
        MASTER_3C_MEDIUM100_SEED42_DIR \
        MASTER_3C_MEDIUM100_SEED43_DIR \
        MASTER_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --last_epoch 100

python3 tools/plot_master.py $PUB \
    --title "MASTER: Interaction Matrix Density (2c, Qwen3)" \
    --output "$OUTDIR/master_InterMat_qwen3.$EXT" \
    --dirs \
        MASTER_2C_SPARSE200_SEED42_DIR \
        MASTER_2C_SPARSE200_SEED43_DIR \
        MASTER_2C_SPARSE200_SEED44_DIR \
        MASTER_2C_MEDIUM100_SEED42_DIR \
        MASTER_2C_MEDIUM100_SEED43_DIR \
        MASTER_2C_MEDIUM100_SEED44_DIR \
        MASTER_2C_DENSE50_SEED42_DIR \
        MASTER_2C_DENSE50_SEED43_DIR \
        MASTER_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --last_epoch 100

python3 tools/plot_master.py $PUB --full \
    --title "MASTER: Inference-Time Density — full (Qwen3)" \
    --output "$OUTDIR/master_UIDensity_qwen3_full.$EXT" \
    --dirs \
        MASTER_1C_MEDIUM100_SEED42_DIR \
        MASTER_1C_MEDIUM100_SEED43_DIR \
        MASTER_1C_MEDIUM100_SEED44_DIR \
        MASTER_2C_MEDIUM100_SEED42_DIR \
        MASTER_2C_MEDIUM100_SEED43_DIR \
        MASTER_2C_MEDIUM100_SEED44_DIR \
        MASTER_3C_MEDIUM100_SEED42_DIR \
        MASTER_3C_MEDIUM100_SEED43_DIR \
        MASTER_3C_MEDIUM100_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c" --last_epoch 100

python3 tools/plot_master.py $PUB --full \
    --title "MASTER: Interaction Matrix Density — full (2c, Qwen3)" \
    --output "$OUTDIR/master_InterMat_qwen3_full.$EXT" \
    --dirs \
        MASTER_2C_SPARSE200_SEED42_DIR \
        MASTER_2C_SPARSE200_SEED43_DIR \
        MASTER_2C_SPARSE200_SEED44_DIR \
        MASTER_2C_MEDIUM100_SEED42_DIR \
        MASTER_2C_MEDIUM100_SEED43_DIR \
        MASTER_2C_MEDIUM100_SEED44_DIR \
        MASTER_2C_DENSE50_SEED42_DIR \
        MASTER_2C_DENSE50_SEED43_DIR \
        MASTER_2C_DENSE50_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i" --last_epoch 100


##############################################################
# DEFENSES
##############################################################

python3 tools/plot_defense.py $PUB \
    --title "G-Safeguard: Inference-Time Density (NetSafe)" \
    --output "$OUTDIR/defense_gsafeguard_UIDensity.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "NETSAFE_1C_MEDIUM100_SEED42_DIR,NETSAFE_1C_MEDIUM100_SEED43_DIR,NETSAFE_1C_MEDIUM100_SEED44_DIR : GSAFEGUARD_1C_MEDIUM100_SEED42_DIR,GSAFEGUARD_1C_MEDIUM100_SEED43_DIR,GSAFEGUARD_1C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_2C_MEDIUM100_SEED42_DIR,NETSAFE_2C_MEDIUM100_SEED43_DIR,NETSAFE_2C_MEDIUM100_SEED44_DIR : GSAFEGUARD_2C_MEDIUM100_SEED42_DIR,GSAFEGUARD_2C_MEDIUM100_SEED43_DIR,GSAFEGUARD_2C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_3C_MEDIUM100_SEED42_DIR,NETSAFE_3C_MEDIUM100_SEED43_DIR,NETSAFE_3C_MEDIUM100_SEED44_DIR : GSAFEGUARD_3C_MEDIUM100_SEED42_DIR,GSAFEGUARD_3C_MEDIUM100_SEED43_DIR,GSAFEGUARD_3C_MEDIUM100_SEED44_DIR" \
    --pair_labels \
        "1c" \
        "2c" \
        "3c"

python3 tools/plot_defense.py $PUB \
    --title "G-Safeguard: Interaction Matrix Density (NetSafe, 2c)" \
    --output "$OUTDIR/defense_gsafeguard_InterMat.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "NETSAFE_2C_SPARSE200_SEED42_DIR,NETSAFE_2C_SPARSE200_SEED43_DIR,NETSAFE_2C_SPARSE200_SEED44_DIR : GSAFEGUARD_2C_SPARSE200_SEED42_DIR,GSAFEGUARD_2C_SPARSE200_SEED43_DIR,GSAFEGUARD_2C_SPARSE200_SEED44_DIR" \
        "NETSAFE_2C_MEDIUM100_SEED42_DIR,NETSAFE_2C_MEDIUM100_SEED43_DIR,NETSAFE_2C_MEDIUM100_SEED44_DIR : GSAFEGUARD_2C_MEDIUM100_SEED42_DIR,GSAFEGUARD_2C_MEDIUM100_SEED43_DIR,GSAFEGUARD_2C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_2C_DENSE50_SEED42_DIR,NETSAFE_2C_DENSE50_SEED43_DIR,NETSAFE_2C_DENSE50_SEED44_DIR : GSAFEGUARD_2C_DENSE50_SEED42_DIR,GSAFEGUARD_2C_DENSE50_SEED43_DIR,GSAFEGUARD_2C_DENSE50_SEED44_DIR" \
    --pair_labels \
        "200i" \
        "100i" \
        "50i"

python3 tools/plot_defense.py $PUB \
    --title "BlindGuard: Inference-Time Density (NetSafe)" \
    --output "$OUTDIR/defense_blindguard_UIDensity.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "NETSAFE_1C_MEDIUM100_SEED42_DIR,NETSAFE_1C_MEDIUM100_SEED43_DIR,NETSAFE_1C_MEDIUM100_SEED44_DIR : BLINDGUARD_1C_MEDIUM100_SEED42_DIR,BLINDGUARD_1C_MEDIUM100_SEED43_DIR,BLINDGUARD_1C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_2C_MEDIUM100_SEED42_DIR,NETSAFE_2C_MEDIUM100_SEED43_DIR,NETSAFE_2C_MEDIUM100_SEED44_DIR : BLINDGUARD_2C_MEDIUM100_SEED42_DIR,BLINDGUARD_2C_MEDIUM100_SEED43_DIR,BLINDGUARD_2C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_3C_MEDIUM100_SEED42_DIR,NETSAFE_3C_MEDIUM100_SEED43_DIR,NETSAFE_3C_MEDIUM100_SEED44_DIR : BLINDGUARD_3C_MEDIUM100_SEED42_DIR,BLINDGUARD_3C_MEDIUM100_SEED43_DIR,BLINDGUARD_3C_MEDIUM100_SEED44_DIR" \
    --pair_labels \
        "1c" \
        "2c" \
        "3c"

python3 tools/plot_defense.py $PUB \
    --title "BlindGuard: Interaction Matrix Density (NetSafe, 2c)" \
    --output "$OUTDIR/defense_blindguard_InterMat.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "NETSAFE_2C_SPARSE200_SEED42_DIR,NETSAFE_2C_SPARSE200_SEED43_DIR,NETSAFE_2C_SPARSE200_SEED44_DIR : BLINDGUARD_2C_SPARSE200_SEED42_DIR,BLINDGUARD_2C_SPARSE200_SEED43_DIR,BLINDGUARD_2C_SPARSE200_SEED44_DIR" \
        "NETSAFE_2C_MEDIUM100_SEED42_DIR,NETSAFE_2C_MEDIUM100_SEED43_DIR,NETSAFE_2C_MEDIUM100_SEED44_DIR : BLINDGUARD_2C_MEDIUM100_SEED42_DIR,BLINDGUARD_2C_MEDIUM100_SEED43_DIR,BLINDGUARD_2C_MEDIUM100_SEED44_DIR" \
        "NETSAFE_2C_DENSE50_SEED42_DIR,NETSAFE_2C_DENSE50_SEED43_DIR,NETSAFE_2C_DENSE50_SEED44_DIR : BLINDGUARD_2C_DENSE50_SEED42_DIR,BLINDGUARD_2C_DENSE50_SEED43_DIR,BLINDGUARD_2C_DENSE50_SEED44_DIR" \
    --pair_labels \
        "200i" \
        "100i" \
        "50i"

python3 tools/plot_defense.py $PUB \
    --title "T-Guard: Inference-Time Density (TOMA)" \
    --output "$OUTDIR/defense_tguard_UIDensity.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "TOMA_1C_MEDIUM100_SEED42_DIR,TOMA_1C_MEDIUM100_SEED43_DIR,TOMA_1C_MEDIUM100_SEED44_DIR : TGUARD_1C_MEDIUM100_SEED42_DIR,TGUARD_1C_MEDIUM100_SEED43_DIR,TGUARD_1C_MEDIUM100_SEED44_DIR" \
        "TOMA_2C_MEDIUM100_SEED42_DIR,TOMA_2C_MEDIUM100_SEED43_DIR,TOMA_2C_MEDIUM100_SEED44_DIR : TGUARD_2C_MEDIUM100_SEED42_DIR,TGUARD_2C_MEDIUM100_SEED43_DIR,TGUARD_2C_MEDIUM100_SEED44_DIR" \
        "TOMA_3C_MEDIUM100_SEED42_DIR,TOMA_3C_MEDIUM100_SEED43_DIR,TOMA_3C_MEDIUM100_SEED44_DIR : TGUARD_3C_MEDIUM100_SEED42_DIR,TGUARD_3C_MEDIUM100_SEED43_DIR,TGUARD_3C_MEDIUM100_SEED44_DIR" \
    --pair_labels \
        "1c" \
        "2c" \
        "3c"

python3 tools/plot_defense.py $PUB \
    --title "T-Guard: Interaction Matrix Density (TOMA, 2c)" \
    --output "$OUTDIR/defense_tguard_InterMat.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "TOMA_2C_SPARSE200_SEED42_DIR,TOMA_2C_SPARSE200_SEED43_DIR,TOMA_2C_SPARSE200_SEED44_DIR : TGUARD_2C_SPARSE200_SEED42_DIR,TGUARD_2C_SPARSE200_SEED43_DIR,TGUARD_2C_SPARSE200_SEED44_DIR" \
        "TOMA_2C_MEDIUM100_SEED42_DIR,TOMA_2C_MEDIUM100_SEED43_DIR,TOMA_2C_MEDIUM100_SEED44_DIR : TGUARD_2C_MEDIUM100_SEED42_DIR,TGUARD_2C_MEDIUM100_SEED43_DIR,TGUARD_2C_MEDIUM100_SEED44_DIR" \
        "TOMA_2C_DENSE50_SEED42_DIR,TOMA_2C_DENSE50_SEED43_DIR,TOMA_2C_DENSE50_SEED44_DIR : TGUARD_2C_DENSE50_SEED42_DIR,TGUARD_2C_DENSE50_SEED43_DIR,TGUARD_2C_DENSE50_SEED44_DIR" \
    --pair_labels \
        "200i" \
        "100i" \
        "50i"

python3 tools/plot_defense.py $PUB \
    --title "M-Guard: Inference-Time Density (MASTER)" \
    --output "$OUTDIR/defense_mguard_UIDensity.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "MASTER_1C_MEDIUM100_SEED42_DIR,MASTER_1C_MEDIUM100_SEED43_DIR,MASTER_1C_MEDIUM100_SEED44_DIR : MGUARD_1C_MEDIUM100_SEED42_DIR,MGUARD_1C_MEDIUM100_SEED43_DIR,MGUARD_1C_MEDIUM100_SEED44_DIR" \
        "MASTER_2C_MEDIUM100_SEED42_DIR,MASTER_2C_MEDIUM100_SEED43_DIR,MASTER_2C_MEDIUM100_SEED44_DIR : MGUARD_2C_MEDIUM100_SEED42_DIR,MGUARD_2C_MEDIUM100_SEED43_DIR,MGUARD_2C_MEDIUM100_SEED44_DIR" \
        "MASTER_3C_MEDIUM100_SEED42_DIR,MASTER_3C_MEDIUM100_SEED43_DIR,MASTER_3C_MEDIUM100_SEED44_DIR : MGUARD_3C_MEDIUM100_SEED42_DIR,MGUARD_3C_MEDIUM100_SEED43_DIR,MGUARD_3C_MEDIUM100_SEED44_DIR" \
    --pair_labels \
        "1c" \
        "2c" \
        "3c"

python3 tools/plot_defense.py $PUB \
    --title "M-Guard: Interaction Matrix Density (MASTER, 2c)" \
    --output "$OUTDIR/defense_mguard_InterMat.$EXT" \
    --last_epoch $DEFENSE_EPOCHS \
    --pairs \
        "MASTER_2C_SPARSE200_SEED42_DIR,MASTER_2C_SPARSE200_SEED43_DIR,MASTER_2C_SPARSE200_SEED44_DIR : MGUARD_2C_SPARSE200_SEED42_DIR,MGUARD_2C_SPARSE200_SEED43_DIR,MGUARD_2C_SPARSE200_SEED44_DIR" \
        "MASTER_2C_MEDIUM100_SEED42_DIR,MASTER_2C_MEDIUM100_SEED43_DIR,MASTER_2C_MEDIUM100_SEED44_DIR : MGUARD_2C_MEDIUM100_SEED42_DIR,MGUARD_2C_MEDIUM100_SEED43_DIR,MGUARD_2C_MEDIUM100_SEED44_DIR" \
        "MASTER_2C_DENSE50_SEED42_DIR,MASTER_2C_DENSE50_SEED43_DIR,MASTER_2C_DENSE50_SEED44_DIR : MGUARD_2C_DENSE50_SEED42_DIR,MGUARD_2C_DENSE50_SEED43_DIR,MGUARD_2C_DENSE50_SEED44_DIR" \
    --pair_labels \
        "200i" \
        "100i" \
        "50i"


##############################################################
# ABLATIONS: Attacker ratio
##############################################################

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: Attacker Ratio Ablation (k=1, medium100)" \
    --output "$OUTDIR/ablations/ablation_ratio_robustness_1c-100.$EXT" \
    --connacf_dirs \
        NETSAFE_1C_MEDIUM100_RATIO10_SEED43_DIR \
        NETSAFE_1C_MEDIUM100_RATIO25_SEED43_DIR \
        NETSAFE_1C_MEDIUM100_RATIO50_SEED43_DIR \
        NETSAFE_1C_MEDIUM100_RATIO75_SEED43_DIR \
    --labels "10%" "25%" "50%" "75%"

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: Attacker Ratio Ablation (k=2, medium100)" \
    --output "$OUTDIR/ablations/ablation_ratio_robustness_2c-100.$EXT" \
    --connacf_dirs \
        NETSAFE_2C_MEDIUM100_RATIO10_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_RATIO25_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_RATIO50_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_RATIO75_SEED43_DIR \
    --labels "10%" "25%" "50%" "75%"

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: Attacker Ratio Ablation (k=3, medium100)" \
    --output "$OUTDIR/ablations/ablation_ratio_robustness_3c-100.$EXT" \
    --connacf_dirs \
        NETSAFE_3C_MEDIUM100_RATIO10_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_RATIO25_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_RATIO50_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_RATIO75_SEED43_DIR \
    --labels "10%" "25%" "50%" "75%"


##############################################################
# ABLATIONS: Role split
##############################################################

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NETSAFE: Role Split — UIDensity" \
    --output "$OUTDIR/ablations/ablation_rolesplit_netsafe_UIDensity.$EXT" \
    --connacf_dirs \
        NETSAFE_1C_MEDIUM100_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_SEED43_DIR \
    --labels "1c" "2c" "3c"

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NETSAFE: Role Split — InterMat" \
    --output "$OUTDIR/ablations/ablation_rolesplit_netsafe_InterMat.$EXT" \
    --connacf_dirs \
        NETSAFE_2C_SPARSE200_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_SEED43_DIR \
        NETSAFE_2C_DENSE50_SEED43_DIR \
    --labels "200i" "100i" "50i"

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "CHEAT: Role Split — UIDensity" \
    --output "$OUTDIR/ablations/ablation_rolesplit_cheat_UIDensity.$EXT" \
    --connacf_dirs \
        CHEAT_1C_MEDIUM100_SEED43_DIR \
        CHEAT_2C_MEDIUM100_SEED43_DIR \
        CHEAT_3C_MEDIUM100_SEED43_DIR \
    --labels "1c" "2c" "3c" --last_epoch 150

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "CHEAT: Role Split — InterMat" \
    --output "$OUTDIR/ablations/ablation_rolesplit_cheat_InterMat.$EXT" \
    --connacf_dirs \
        CHEAT_2C_SPARSE200_SEED43_DIR \
        CHEAT_2C_MEDIUM100_SEED43_DIR \
        CHEAT_2C_DENSE50_SEED43_DIR \
    --labels "200i" "100i" "50i" --last_epoch 150

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "DRUNK: Role Split — UIDensity" \
    --output "$OUTDIR/ablations/ablation_rolesplit_drunk_UIDensity.$EXT" \
    --connacf_dirs \
        DRUNK_1C_MEDIUM100_SEED43_DIR \
        DRUNK_2C_MEDIUM100_SEED43_DIR \
        DRUNK_3C_MEDIUM100_SEED43_DIR \
    --labels "1c" "2c" "3c" --last_epoch 250

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "DRUNK: Role Split — InterMat" \
    --output "$OUTDIR/ablations/ablation_rolesplit_drunk_InterMat.$EXT" \
    --connacf_dirs \
        DRUNK_2C_SPARSE200_SEED43_DIR \
        DRUNK_2C_MEDIUM100_SEED43_DIR \
        DRUNK_2C_DENSE50_SEED43_DIR \
    --labels "200i" "100i" "50i" --last_epoch 250

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "RTA: Role Split — UIDensity" \
    --output "$OUTDIR/ablations/ablation_rolesplit_rta_UIDensity.$EXT" \
    --connacf_dirs \
        RTA_1C_MEDIUM100_SEED43_DIR \
        RTA_2C_MEDIUM100_SEED43_DIR \
        RTA_3C_MEDIUM100_SEED43_DIR \
    --labels "1c" "2c" "3c" --last_epoch 150

python3 tools/plot_dissemination.py --large-titles --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "RTA: Role Split — InterMat" \
    --output "$OUTDIR/ablations/ablation_rolesplit_rta_InterMat.$EXT" \
    --connacf_dirs \
        RTA_2C_SPARSE200_SEED43_DIR \
        RTA_2C_MEDIUM100_SEED43_DIR \
        RTA_2C_DENSE50_SEED43_DIR \
    --labels "200i" "100i" "50i" --last_epoch 150


##############################################################
# ABLATIONS: Cross-LLM
##############################################################

for MODEL in haiku sonnet46 mixtral llama4; do
python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: UIDensity (${MODEL})" \
    --output "$OUTDIR/ablations/ablation_llm_${MODEL}_UIDensity.$EXT" \
    --connacf_dirs \
        NETSAFE_1C_MEDIUM100_${MODEL^^}_SEED42_DIR \
        NETSAFE_1C_MEDIUM100_${MODEL^^}_SEED43_DIR \
        NETSAFE_1C_MEDIUM100_${MODEL^^}_SEED44_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED42_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED44_DIR \
        NETSAFE_3C_MEDIUM100_${MODEL^^}_SEED42_DIR \
        NETSAFE_3C_MEDIUM100_${MODEL^^}_SEED43_DIR \
        NETSAFE_3C_MEDIUM100_${MODEL^^}_SEED44_DIR \
    --labels "1c" "1c" "1c" "2c" "2c" "2c" "3c" "3c" "3c"

python3 tools/plot_dissemination.py --last_epoch $NETSAFE_EPOCHS $PUB \
    --title "NetSafe: InterMat (${MODEL})" \
    --output "$OUTDIR/ablations/ablation_llm_${MODEL}_InterMat.$EXT" \
    --connacf_dirs \
        NETSAFE_2C_SPARSE200_${MODEL^^}_SEED42_DIR \
        NETSAFE_2C_SPARSE200_${MODEL^^}_SEED43_DIR \
        NETSAFE_2C_SPARSE200_${MODEL^^}_SEED44_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED42_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED43_DIR \
        NETSAFE_2C_MEDIUM100_${MODEL^^}_SEED44_DIR \
        NETSAFE_2C_DENSE50_${MODEL^^}_SEED42_DIR \
        NETSAFE_2C_DENSE50_${MODEL^^}_SEED43_DIR \
        NETSAFE_2C_DENSE50_${MODEL^^}_SEED44_DIR \
    --labels "200i" "200i" "200i" "100i" "100i" "100i" "50i" "50i" "50i"
done

echo ""
echo "Done. Output written to $OUTDIR/"
