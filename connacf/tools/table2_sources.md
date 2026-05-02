> **Superseded — see [`PREDICTIVE_METRICS.md`](PREDICTIVE_METRICS.md).**
> The predictor was corrected: the frozen one-hop `R_U/R_I` below is replaced by
> the coupled steady state `rho*` (no partition frozen) for the steady-state
> mean, parameter-free exposure for the transient slope, and `R0 = sqrt(R_U*R_I)`
> as the severity scalar; correlations are now **signed** Spearman (no `max(R,E)`).
> The current outputs are `ranking_predictors_corr.csv` (per-sweep, signed) and
> `ranking_predictors_loo.csv` (leave-one-attack-out), produced by
> `sis_predictors.py` / `compute_ranking_predictors.py`. The mapping below is
> retained only for historical reference to the original submission.

# Table 2 Source Mapping

Rows = attack groups. Columns = predictor ($R_U$ or $R_I$) × outcome temporal regime.
$R_U$ is correlated against the **kUI sweep**; $R_I$ against the **rhoIM sweep**.

| Row | $R_U$: $U_{\text{tr}}$ | $R_U$: $I_{\text{tr}}$ | $R_U$: $U_{\text{ss}}$ | $R_U$: $I_{\text{ss}}$ | $R_I$: $U_{\text{tr}}$ | $R_I$: $I_{\text{tr}}$ | $R_I$: $U_{\text{ss}}$ | $R_I$: $I_{\text{ss}}$ |
|---|---|---|---|---|---|---|---|---|
| **Dissem** | NetSafe+CORBA user ASR transient slope, kUI sweep | NetSafe+CORBA item ASR transient slope, kUI sweep | NetSafe+CORBA user ASR steady-state mean, kUI sweep | NetSafe+CORBA item ASR steady-state mean, kUI sweep | NetSafe+CORBA user ASR transient slope, rhoIM sweep | NetSafe+CORBA item ASR transient slope, rhoIM sweep | NetSafe+CORBA user ASR steady-state mean, rhoIM sweep | NetSafe+CORBA item ASR steady-state mean, rhoIM sweep |
| **Bi-Dissem** | TOMA user ASR transient slope, kUI sweep | TOMA item ASR transient slope, kUI sweep | TOMA user ASR steady-state mean, kUI sweep | TOMA item ASR steady-state mean, kUI sweep | TOMA user ASR transient slope, rhoIM sweep | TOMA item ASR transient slope, rhoIM sweep | TOMA user ASR steady-state mean, rhoIM sweep | TOMA item ASR steady-state mean, rhoIM sweep |
| **Extract** | ← MAMA pii_leakage + MASLeak ui_topology_f1 + MASLeak ui_topology_recall transient slope, kUI sweep (U/I merged) | | ← MAMA pii_leakage + MASLeak ui_topology_f1 + MASLeak ui_topology_recall steady-state mean, kUI sweep (U/I merged) | | ← MAMA pii_leakage + MASLeak ui_topology_f1 + MASLeak ui_topology_recall transient slope, rhoIM sweep (U/I merged, low variance) | | ← MAMA pii_leakage + MASLeak ui_topology_f1 + MASLeak ui_topology_recall steady-state mean, rhoIM sweep (U/I merged, low variance) | |
| **Bi-Extract** | ← TOMA toma.node_coverage + MASTER master.ss_system_prompt transient slope, kUI sweep (U/I merged) | | ← TOMA toma.node_coverage + MASTER master.ss_system_prompt steady-state mean, kUI sweep (U/I merged) | | ← TOMA toma.node_coverage + MASTER master.ss_system_prompt transient slope, rhoIM sweep (U/I merged) | | ← TOMA toma.node_coverage + MASTER master.ss_system_prompt steady-state mean, rhoIM sweep (U/I merged) | |

## CSV Columns

```
attack, metric_type, sweep, label, k_UI, n_items, alpha,
tr_cutoff,
U_tr, U_ss, I_tr, I_ss,
E_U, E_I, R_U, R_I,
gamma_kUI_tr, r_kUI_tr, gamma_kUI_ss, r_kUI_ss,
gamma_rhoIM_tr, r_rhoIM_tr, gamma_rhoIM_ss, r_rhoIM_ss,
run_dir, n_turns
```
