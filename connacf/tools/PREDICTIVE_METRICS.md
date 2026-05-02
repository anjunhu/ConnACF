# Predictive Metrics (§7) — Model, Predictors, and Implementation

This documents the corrected implementation of the paper's secondary
contribution: **ranking connectivity configurations by expected attack outcome
from graph-structural quantities, without running the full agent simulation.**

Everything here is implemented in [`sis_predictors.py`](sis_predictors.py) (the
single source of truth) and consumed by
[`compute_ranking_predictors.py`](compute_ranking_predictors.py).

The theory is grounded in the paper's primary reference:
Pastor-Satorras, Castellano, Van Mieghem & Vespignani, *Epidemic processes in
complex networks*, **Rev. Mod. Phys. 87, 925 (2015)** — cited below as **RMP**.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| `U`, `I` | user / item partitions (bipartite mesh) |
| `k` ∈ {1,2,3} | candidate count — items each user contacts per turn (**user degree**) |
| `ρ = n_U/n_I` ∈ {0.5, 1, 2} | catalog concentration (`n_U=100`; `n_I` ∈ {200,100,50}) |
| `ρ·k` | **item degree** — expected users contacting an item per turn (`n_U·k/n_I`) |
| `α_U, α_I` | seeded attacker fractions (main runs 0.25; MASTER 0.1; MASLeak 0.5) |
| `ρ_U, ρ_I` | contaminated fraction of each partition (the ASR being modelled) |
| `β_U, β_I` | per-partition susceptibility (infection scale) |
| `γ` | recovery rate **per benign contact** (alignment-driven self-correction) |
| `r` | spontaneous LLM self-cleansing rate (baseline recovery) |
| `θ = (β_U, β_I, γ, r)` | the four fitted constants |

---

## 2. Final model — coupled bipartite contact-SIS

Two coupled populations on a bipartite mesh. One Euler step = one turn.

```
drho_U/dt = beta_U (1-rho_U)[1-(1-rho_I)^k]      - rho_U[(1-rho_I) k     gamma + r]
drho_I/dt = beta_I (1-rho_I)[1-(1-rho_U)^(rho k)] - rho_I[(1-rho_U) rho k gamma + r]
```

Initial condition `rho_U(0)=α_U`, `rho_I(0)=α_I` (seeded attackers).
Implemented in `simulate_sis(...)`.

**Infection term** — `1-(1-ρ_other)^degree` is the **discrete-time contact
form** (Gómez et al., 2010, cited by the paper): the probability that at least
one of `degree` contacts transmits. It reduces to the mass-action `degree·ρ` for
small `ρ` (RMP Eq. 6). The user sees `k` item contacts; the item sees `ρk` user
contacts (RMP §VII.B.5, bipartite networks).

**Recovery term** — recovery is **contact-mediated**, scaling with the number of
*benign* contacts `(1-ρ_other)·degree`, plus a spontaneous baseline `r`. This is
**SIS-inspired, not textbook SIS**: standard SIS recovers spontaneously at a
constant rate `μρ` (RMP Eq. 6). The contact-mediated form is the paper's
deliberate *alignment-driven recovery* extension — more benign interactions give
more chances to reflect and self-correct. It is kept on purpose; a useful
consequence is that the reproduction number **saturates in `k`** (see §3.3),
reproducing the observed plateaus (paper F1/F3) with no extra fitting.

---

## 3. Final predictors

All three are derived from the **one** model above. Each targets a different
outcome regime.

### 3.1 Transient slope ← exposure (no free parameters)

The initial per-turn growth rate `dρ/dt|_{ρ=0}` is just the one-hop exposure:

```
E_U = 1 - (1 - alpha_I)^k          # exposure_U(alpha_I, k)
E_I = 1 - (1 - alpha_U)^(rho*k)     # exposure_I(alpha_U, rho, k)
```

Because it has **no fitted parameters**, its correlation with the observed
transient slope is an honest, non-circular number. On the ρ-sweep `k=2` is
fixed, so `E_U` is constant and its correlation is (correctly) undefined — the
user-side transient does not respond to catalog concentration.

### 3.2 Steady-state mean ← coupled fixed point ρ\* (the P2 / RMP fix)

The steady state is the **coupled** fixed point of the full system, obtained by
integrating both partitions to equilibrium with **neither partition frozen**:

```
(rho_U*, rho_I*) = sis_steady_state(beta_U, beta_I, gamma, r, alpha_U, alpha_I, k, rho)
```

This is the correction demanded by the reference. RMP Eq. 77 (Gómez-Gardeñes et
al., 2008) shows the bipartite SIS threshold **couples both partitions
multiplicatively** — reducing one partition's rate alone cannot rule out an
outbreak. The paper's original predictor instead *froze* the opposite partition
at its seeded prevalence `α` (a one-hop approximation), which decouples exactly
what the reference says is coupled. Consequence of the fix: on the ρ-sweep the
**user** steady state now moves with `ρ` through the coupling, where the frozen
`R_U` was identically constant.

### 3.3 Severity scalar ← coupled reproduction number R₀

A single scalar per configuration = spectral radius of the 2×2 next-generation
matrix of the linearised system (RMP Eq. 11 homogeneous, Eq. 77 bipartite):

```
mu_U = k*gamma + r ,   mu_I = rho*k*gamma + r
R0   = sqrt( beta_U * beta_I * rho * k^2 / (mu_U * mu_I) )   ==  sqrt(R_U * R_I)
```

Implemented in `sis_R0(...)`. It couples both partitions multiplicatively (it is
literally the geometric mean of the two per-partition factors), and **saturates
to `sqrt(β_U β_I)/γ` as `k → ∞`** because the contact-mediated recovery grows
with `k` in lockstep with exposure. `sis_growth_rate(...)` returns the
associated dominant Jacobian eigenvalue (`>0 ⇔ R0>1`; RMP Eq. 8).

### Predictor → outcome map

| Outcome | Predictor | Varies with | Free params |
|---|---|---|---|
| transient slope `U_tr`, `I_tr` | exposure `E_U`, `E_I` | `k` (both), `ρ` (item only) | none |
| steady-state mean `U_ss`, `I_ss` | coupled `ρ_U*`, `ρ_I*` | `k` and `ρ` (both, via coupling) | `θ` |
| overall severity | coupled `R₀` | `k`, `ρ` | `θ` |

---

## 4. Grounding in the reference (RMP 2015)

| RMP | Result | Use here |
|---|---|---|
| Eq. 6 | homogeneous mean-field SIS `dρ/dt = βρ(1−ρ) − μρ` | ODE skeleton |
| Eq. 6 | recovery is spontaneous `μρ` | we deliberately deviate (contact-mediated) — documented |
| Eq. 8 | early growth `dρ/dt ≈ (β−μ)ρ` | transient predictor = exposure |
| Eq. 11 | `R₀ = ⟨k⟩β/μ` | R₀ scalar (homogeneous) |
| Eq. 77 | bipartite threshold `λ_U λ_I = ⟨k⟩_U⟨k⟩_I/⟨k²⟩_U⟨k²⟩_I` — **coupled** | ρ\* and R₀ keep both partitions coupled (P2 fix) |
| Gómez 2010 | discrete-time contact form `1−(1−λ)^d` | infection term |

---

## 5. Fitting & evaluation protocol (honest by construction)

* **One `θ` per attack**, shared across both partitions, both regimes and both
  sweeps. Only the steady-state predictor `ρ*` uses `θ`; exposure uses none.
  Sharing `θ` over many targets prevents per-point overfitting.
* **Signed Spearman.** A predictor that ranks configurations *backwards* scores
  **negative** — it is a failure, not a success. (The previous code maximised
  `|ρ|`, which rewarded getting the order backwards.)
* **Constant predictors are penalised**, so the fit cannot "win" by driving a
  partition to a flat, unrankable prediction and dropping it from the average.
* **Leave-one-attack-out** (`leave_one_attack_out`): fit `θ` on all attacks but
  one, score the held-out attack. This is the out-of-sample number, reported
  next to the in-sample fit.
* **No `max(R, E)` flooring** and **no `if E is good then R:=E` short-circuit** —
  both previously guaranteed the recovery-aware predictor could never look worse
  than the naive one, making the comparison meaningless. Removed.
* **Automatic transient/steady breakpoint** by two-segment piecewise-linear fit
  (`find_breakpoint`), replacing per-attack cutoffs read "from visual
  inspection."
* **Distinct `α_U`, `α_I`** threaded throughout (equal in the shipped configs,
  but no longer hardcoded to one value).

---

## 6. What was broken, and what changed

| Tag | Issue (before) | Fix |
|---|---|---|
| **P2** | steady-state predictor froze the opposite partition at `α` → decoupled | coupled fixed point `ρ*` (RMP Eq. 77) |
| **B1** | item degree hardcoded `ρ·2` (k_ref=2) → item predictor ignored `k` (ConnACF) | `item_degree = ρ·k` |
| **B3** | single `alpha` used for both partitions | distinct `α_U`, `α_I` |
| **B4** | `max(R,E)` flooring + `E≥0.8 ⇒ R:=E` short-circuit | removed; report R honestly |
| **B5** | fit maximised `|Spearman|` (sign-blind) | **signed** Spearman |
| **B6** | `(γ,r)` refit in-sample per attack×regime×sweep on the same 3 points | shared `θ` + leave-one-attack-out |
| **B7** | tr/ss breakpoints hand-set per attack | automatic piecewise-linear fit |
| **B8** | three divergent SIS parameterisations; ODE unused for ranking | one model in `sis_predictors.py`, imported everywhere |
| **P1** | recovery is not standard SIS | **kept on purpose** — SIS-*inspired*, documented (§2) |

---

## 7. Files & how to run

* [`sis_predictors.py`](sis_predictors.py) — the model, the three predictors,
  the fit, and the signed evaluation. Run it directly to reproduce the §7 table
  from the cached per-run outcomes (no raw trajectories needed):

  ```bash
  python tools/sis_predictors.py --raw_csv tools/ranking_pred_output/ranking_predictors_raw.csv
  python tools/sis_predictors.py --compare   # head-to-head predictive power (§9)
  ```

  Writes `ranking_predictors_corr_signed.csv`.

* [`compute_ranking_predictors.py`](compute_ranking_predictors.py) — full
  pipeline: extracts per-turn `tr`/`ss` from raw `attack_output/` runs, then
  delegates the predictor/fit/correlation stage to `sis_predictors`.

---

## 8. Example output (from the shipped cached outcomes)

Signed Spearman; `nan` = predictor or target has no rank variance (meaningful,
e.g. `U_tr` on the ρ-sweep is constant because the user degree `k` is fixed).

```
attack/metric            sweep     U_tr    I_tr    U_ss    I_ss
netsafe/dissem           kUI        1.0    -0.5   0.866   0.866
                         rhoIM      nan     1.0     nan     nan
corba/corba              kUI        0.5     1.0     1.0     1.0
                         rhoIM      nan     0.5     0.5     0.5
toma/dissem              kUI        0.5     1.0     1.0     1.0
                         rhoIM      nan     1.0     0.5     1.0
mama/mama                kUI        1.0     nan     1.0     nan
masleak/masleak          kUI        1.0     nan     1.0     nan
```

Leave-one-attack-out (out-of-sample steady-state ranking): dissemination
generalises well (netsafe U/I = +0.6/+0.9, corba +0.9/+0.8, toma +0.9/+1.0);
extraction is honestly weaker (mama +0.1, toma_extract −0.3) — consistent with
the paper's statement that the first-order predictor is uneven across attacker
goals, and now shown rather than floored away.
```
netsafe/dissem    U_ss=+0.6  I_ss=+0.9
corba/corba       U_ss=+0.9  I_ss=+0.8
toma/dissem       U_ss=+0.9  I_ss=+1.0
mama/mama         U_ss=+0.1  I_ss= nan
toma/toma_extract U_ss=-0.3  I_ss= nan
```

---

## 9. Predictive power — which predictor is actually better?

Reproduce with:

```bash
python tools/sis_predictors.py --compare
```

### 9.1 How this is evaluated (and why it matters)

With only 3–5 configurations per attack, **in-sample** Spearman is nearly
meaningless: a 4-parameter model can interpolate the points. So candidates are
judged on two honest axes:

* **Leave-one-attack-out (LOAO)** — fit the constants on the *other*
  dissemination attacks, then predict the held-out attack's configuration
  ranking. This is the real test of the paper's claim: a *universal*,
  structure-based severity predictor. A model that only wins in-sample is
  overfitting, not predicting.
* **Parsimony** — every fitted model must clear the **0-parameter exposure
  baseline**. If a 4-param model ties exposure, exposure wins.

All candidates are fit under the *same* discipline (maximise signed Spearman of
predictor vs steady-state ASR, pooled over U and I) and scored with **signed**
Spearman.

### 9.2 Result (mean signed Spearman over netsafe, corba, toma dissemination)

| Predictor | params | U_ss (in) | I_ss (in) | **U_ss (LOAO)** | **I_ss (LOAO)** |
|---|---|---|---|---|---|
| Exposure `E` | 0 | +0.52 | +0.89 | +0.52 | **+0.89** |
| Frozen one-hop `R` (paper's original) | 2 | +0.52 | +0.92 | +0.52 | +0.85 |
| **Coupled `ρ*`** | 4 | +0.83 | +0.89 | **+0.80** | +0.90 |
| Coupled `R₀` scalar | 4 | +0.70 | +0.87 | +0.70 | +0.87 |

The user-side `ρ*` advantage is consistent across **all three** attacks and in
both Spearman and Kendall τ (per-attack LOAO U_ss: netsafe +0.60, corba +0.90,
toma +0.90), and does not overfit (in 0.83 → LOAO 0.80).

### 9.3 What is better

1. **Item side → exposure alone wins (parsimony).** Everything sits at
   ~0.85–0.92; the 4-param models do not beat the 0-param baseline. Item ASR is
   essentially a function of item degree `ρk`, which `exposure_I` already
   encodes. *No fitting needed* — report this as a result.
2. **User side → coupled `ρ*` is genuinely better** (+0.80 vs +0.52) and
   generalises. This is exactly where the reference says coupling must matter:
   on the ρ-sweep the user degree `k` is fixed, so `exposure_U` is constant and
   blind; user contamination moves only through item→user feedback, which only
   the coupled fixed point sees. (E.g. corba's ρ-sweep U_ss = {86.9, 72.8, 79.3}
   at fixed `k=2`, where exposure is identical for all three.)
3. **The paper's frozen one-hop `R` adds parameters for no out-of-sample gain** —
   it ties exposure on the user side (both +0.52), because freezing the item
   partition removes the very feedback that carries the ρ-signal. Direct
   empirical justification for the P2 fix.
4. **`R₀`** is a solid single coupled scalar but is dominated by per-partition
   `ρ*`; use `R₀` for coarse overall severity, `ρ*` for per-partition.

**Recommendation:** report **exposure** (transient), **coupled `ρ*`**
(steady-state, per-partition), and **`R₀`** (severity scalar); **drop the frozen
one-hop `R`** — it never beats the free baseline out of sample. Surface the
asymmetry as a finding: coupling earns its keep precisely on the user side, as
the bipartite theory predicts, and is unnecessary on the item side.

**Caveat:** n = 3 attacks × 5 configs is low-powered — this supports a
*consistent direction*, not a precise effect size.
