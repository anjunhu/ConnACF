#!/usr/bin/env python3
"""
sis_predictors.py -- single source of truth for the §7 "Predictive Metrics".

This module replaces three previously-divergent SIS parameterisations
(compute_ranking_predictors' static odds, its 4-param ODE fit, and
plot_*_sis' 6-param fit) with ONE coupled bipartite model and ONE set of
predictors, so every tool that ranks connectivity configurations by expected
attack outcome agrees on the maths.

Model (SIS-inspired, NOT textbook SIS -- see note on recovery):
-------------------------------------------------------------
Two coupled populations, users (U) and items (I), on a bipartite mesh.
rho_U, rho_I are contaminated fractions in [0,1]. A user contacts k items per
turn; an item therefore contacts rho*k users per turn, with rho = n_U/n_I the
catalog concentration (this is the item-side degree; Pastor-Satorras et al.,
Rev. Mod. Phys. 87, 925 (2015), Sec. VII.B.5, bipartite networks).

    drho_U/dt = beta_U (1-rho_U)[1-(1-rho_I)^k]      - rho_U[(1-rho_I) k     gamma + r]
    drho_I/dt = beta_I (1-rho_I)[1-(1-rho_U)^(rho k)] - rho_I[(1-rho_U) rho k gamma + r]

* Infection uses the discrete-time contact form 1-(1-rho)^d (Gomez et al.,
  2010): probability that at least one of d infected-or-not contacts transmits.
  Reduces to the mass-action d*rho for small rho (RMP Eq. 6).
* Recovery is CONTACT-MEDIATED, not the spontaneous mu*rho of standard SIS
  (RMP Eq. 6). Each benign contact gives a chance to self-correct, so recovery
  scales with the number of clean contacts (1-rho_other)*degree, plus a
  spontaneous LLM self-cleansing baseline r. This is the paper's deliberate
  "alignment-driven recovery" extension. A useful consequence: the coupled
  reproduction number saturates as k -> inf (see sis_R0), reproducing the
  observed plateaus (paper F1/F3) without any extra fitting.

Predictors derived from this ONE model:
  * exposure_U / exposure_I : one-hop per-turn exposure = initial growth rate
        (no free parameters); ranks the TRANSIENT slope.
  * sis_steady_state        : the COUPLED fixed point (rho_U*, rho_I*) obtained
        by integrating the full system to equilibrium -- NO freezing of the
        other partition. This is the P2 fix: RMP (Eq. 77) shows the two
        partitions are coupled multiplicatively, so the steady state must be
        solved jointly, not with rho_other frozen at alpha. Ranks the
        STEADY-STATE mean.
  * sis_R0                  : the coupled basic reproduction number, i.e. the
        spectral radius of the 2x2 next-generation matrix of the linearised
        system (RMP Eq. 11 homogeneous, Eq. 77 bipartite). A single severity
        scalar per configuration; equals sqrt(R_U * R_I).

Fitting is honest: theta=(beta_U,beta_I,gamma,r) is shared across partitions,
regimes and sweeps (fit once per attack, plus a global fit and a
leave-one-attack-out out-of-sample number), and correlations are SIGNED
Spearman -- a predictor that ranks configurations backwards is a failure, not
a success. There is no max(R,E) flooring and no "if E is already good, set
R=E" short-circuit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

N_USERS = 100  # fixed across all ml-100k runs


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def catalog_concentration(n_items: int, n_users: int = N_USERS) -> float:
    """rho = n_U / n_I (paper Sec. 3.3)."""
    return n_users / n_items if n_items > 0 else 0.0


def item_degree(rho: float, k: int) -> float:
    """Expected user contacts per item per turn = rho*k.

    n_U users each contact k items => n_U*k U-I contacts => n_U*k/n_I = rho*k
    per item. This is the item-side counterpart of the user degree k and is
    what enters the item-side exposure and recovery. (ConnACF used a hardcoded
    rho*2 here, which made item predictors ignore k on the k-sweep -- fixed.)
    """
    return rho * k


# ---------------------------------------------------------------------------
# One-hop exposure  ==  initial per-turn growth rate  ==  transient predictor
# ---------------------------------------------------------------------------

def exposure_U(alpha_I: float, k: int) -> float:
    """User exposure: prob >=1 of k item-contacts is a seeded attacker.
    1 - (1-alpha_I)^k.  (No free parameters.)"""
    return 1.0 - (1.0 - alpha_I) ** k


def exposure_I(alpha_U: float, rho: float, k: int) -> float:
    """Item exposure: 1 - (1-alpha_U)^(rho*k).  (No free parameters.)"""
    return 1.0 - (1.0 - alpha_U) ** item_degree(rho, k)


# ---------------------------------------------------------------------------
# Coupled bipartite ODE (the model)
# ---------------------------------------------------------------------------

def simulate_sis(beta_U: float, beta_I: float, gamma: float, r: float,
                 alpha_U: float, alpha_I: float, k: int, rho: float,
                 n_turns: int) -> Tuple[List[float], List[float]]:
    """Integrate the coupled bipartite SIS (one Euler step = one turn).

    Initial conditions rho_U(0)=alpha_U, rho_I(0)=alpha_I (seeded attackers).
    Returns (rho_U_series, rho_I_series) in percent [0,100].
    Both partitions evolve -- neither is frozen (P2/RMP coupling).
    """
    d_I = item_degree(rho, k)
    rho_U, rho_I = alpha_U, alpha_I
    u_series, i_series = [rho_U * 100.0], [rho_I * 100.0]
    for _ in range(max(n_turns - 1, 0)):
        inf_U = beta_U * (1.0 - rho_U) * (1.0 - (1.0 - rho_I) ** k)
        rec_U = rho_U * ((1.0 - rho_I) * k * gamma + r)
        inf_I = beta_I * (1.0 - rho_I) * (1.0 - (1.0 - rho_U) ** max(d_I, 1e-9))
        rec_I = rho_I * ((1.0 - rho_U) * d_I * gamma + r)
        rho_U = float(np.clip(rho_U + inf_U - rec_U, 0.0, 1.0))
        rho_I = float(np.clip(rho_I + inf_I - rec_I, 0.0, 1.0))
        u_series.append(rho_U * 100.0)
        i_series.append(rho_I * 100.0)
    return u_series, i_series


def steady_state_batch(theta: Tuple[float, float, float, float],
                       ks: "np.ndarray", rhos: "np.ndarray",
                       alpha_Us: "np.ndarray", alpha_Is: "np.ndarray",
                       max_turns: int = 800, tol: float = 1e-7
                       ) -> Tuple["np.ndarray", "np.ndarray"]:
    """Vectorised coupled steady state for many configurations at once.

    Integrates every configuration's ODE in lock-step as numpy vectors and
    early-stops when the largest per-config change falls below tol. This is the
    hot path used during fitting; the scalar sis_steady_state wraps it.
    """
    bU, bI, g, r = theta
    ks = np.asarray(ks, dtype=float)
    d_I = np.asarray(rhos, dtype=float) * ks
    d_I = np.maximum(d_I, 1e-9)
    rho_U = np.asarray(alpha_Us, dtype=float).copy()
    rho_I = np.asarray(alpha_Is, dtype=float).copy()
    for _ in range(max_turns):
        inf_U = bU * (1.0 - rho_U) * (1.0 - np.power(1.0 - rho_I, ks))
        rec_U = rho_U * ((1.0 - rho_I) * ks * g + r)
        inf_I = bI * (1.0 - rho_I) * (1.0 - np.power(1.0 - rho_U, d_I))
        rec_I = rho_I * ((1.0 - rho_U) * d_I * g + r)
        nu = np.clip(rho_U + inf_U - rec_U, 0.0, 1.0)
        ni = np.clip(rho_I + inf_I - rec_I, 0.0, 1.0)
        if np.max(np.abs(nu - rho_U)) < tol and np.max(np.abs(ni - rho_I)) < tol:
            rho_U, rho_I = nu, ni
            break
        rho_U, rho_I = nu, ni
    return rho_U, rho_I


def sis_steady_state(beta_U: float, beta_I: float, gamma: float, r: float,
                     alpha_U: float, alpha_I: float, k: int, rho: float,
                     max_turns: int = 800, tol: float = 1e-7
                     ) -> Tuple[float, float]:
    """Coupled steady state (rho_U*, rho_I*) in [0,1] by integrating to
    equilibrium -- the P2 fix, replacing the paper's "freeze rho_other=alpha"
    one-hop odds. Early-stops when both partitions stop changing."""
    u, i = steady_state_batch((beta_U, beta_I, gamma, r),
                              np.array([k], float), np.array([rho], float),
                              np.array([alpha_U], float), np.array([alpha_I], float),
                              max_turns=max_turns, tol=tol)
    return float(u[0]), float(i[0])


def sis_R0(beta_U: float, beta_I: float, gamma: float, r: float,
           k: int, rho: float) -> float:
    """Coupled basic reproduction number = spectral radius of the 2x2
    next-generation matrix of the linearised system (RMP Eq. 11 / Eq. 77):

        mu_U = k*gamma + r,  mu_I = rho*k*gamma + r
        R0   = sqrt( beta_U*beta_I * rho*k^2 / (mu_U * mu_I) )   ==  sqrt(R_U*R_I)

    Couples both partitions multiplicatively; saturates to sqrt(beta_U*beta_I)/gamma
    as k -> inf because the contact-mediated recovery grows with k too.
    """
    mu_U = k * gamma + r
    mu_I = item_degree(rho, k) * gamma + r
    if mu_U <= 0 or mu_I <= 0:
        return float("nan")
    return math.sqrt(beta_U * beta_I * rho * k * k / (mu_U * mu_I))


def sis_growth_rate(beta_U: float, beta_I: float, gamma: float, r: float,
                    k: int, rho: float) -> float:
    """Dominant eigenvalue of the Jacobian at the disease-free equilibrium
    (RMP Eq. 8 early-growth rate). >0 iff R0>1. Single coupled scalar."""
    mu_U = k * gamma + r
    mu_I = item_degree(rho, k) * gamma + r
    disc = ((mu_U - mu_I) / 2.0) ** 2 + beta_U * beta_I * rho * k * k
    return -(mu_U + mu_I) / 2.0 + math.sqrt(max(disc, 0.0))


def fit_trajectory(u_obs, i_obs, alpha_U: float, alpha_I: float, k: int, rho: float,
                   bounds=((0.02, 3.0), (0.02, 3.0), (1e-3, 3.0), (1e-3, 3.0))
                   ) -> Tuple[float, float, float, float]:
    """Fit theta=(beta_U, beta_I, gamma, r) of the coupled model to observed
    per-turn trajectories (u_obs, i_obs, both in percent) by MSE. This is the
    single shared 4-parameter trajectory fit used by the SIS-overlay plotters,
    replacing their old per-plot 6-parameter (gamma_U,gamma_I,r_U,r_I) fits --
    which also used the wrong item exponent (rho instead of rho*k)."""
    u = np.asarray(u_obs, dtype=float)
    n = len(u)
    iv = None
    if i_obs is not None and len(i_obs) == n and np.any(np.asarray(i_obs, float) > 0):
        iv = np.asarray(i_obs, dtype=float)

    def _mse(p):
        bU, bI, g, r = (max(x, 1e-6) for x in p)
        su, si = simulate_sis(bU, bI, g, r, alpha_U, alpha_I, k, rho, n)
        loss = float(np.mean((np.asarray(su) - u) ** 2))
        if iv is not None:
            loss += float(np.mean((np.asarray(si) - iv) ** 2))
        return loss

    best, bv = (0.4, 0.4, 0.1, 0.05), np.inf
    for bU in (0.1, 0.3, 0.6, 1.0, 2.0):
        for bI in (0.1, 0.3, 0.6, 1.0, 2.0):
            for g in (0.01, 0.1, 0.5, 1.0):
                for rr in (0.01, 0.05, 0.2):
                    v = _mse((bU, bI, g, rr))
                    if v < bv:
                        bv, best = v, (bU, bI, g, rr)
    res = minimize(_mse, np.array(best), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 4000})
    return tuple(float(max(x, lo)) for x, (lo, hi) in zip(res.x, bounds))  # type: ignore


# ---------------------------------------------------------------------------
# One-hop reduction (kept only for reference / diagnostics, NOT for ranking)
# ---------------------------------------------------------------------------

def one_hop_R(exposure: float, degree: float, alpha: float,
              gamma: float, r: float) -> float:
    """Paper's first-order R (rho*/(1-rho*) with the other partition frozen at
    alpha). Retained for comparison; the ranking pipeline uses the coupled
    sis_steady_state / sis_R0 instead. R0 == sqrt(R_U*R_I) recovers this pair.
    """
    den = (1.0 - alpha) * degree * gamma + r
    return exposure / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Config points and honest fitting
# ---------------------------------------------------------------------------

class Config:
    """One connectivity configuration with its observed (seed-averaged) outcomes."""
    __slots__ = ("k", "n_items", "rho", "alpha_U", "alpha_I",
                 "U_tr", "U_ss", "I_tr", "I_ss", "has_item")

    def __init__(self, k: int, n_items: int, alpha_U: float, alpha_I: float,
                 U_tr: float, U_ss: float, I_tr: float, I_ss: float,
                 has_item: bool):
        self.k = k
        self.n_items = n_items
        self.rho = catalog_concentration(n_items)
        self.alpha_U = alpha_U
        self.alpha_I = alpha_I
        self.U_tr, self.U_ss = U_tr, U_ss
        self.I_tr, self.I_ss = I_tr, I_ss
        self.has_item = has_item


def _signed_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Signed Spearman rho; nan if degenerate. Sign is kept on purpose: a
    predictor that ranks configurations backwards must score negative."""
    if len(x) < 2 or len(set(np.round(x, 12))) < 2 or len(set(np.round(y, 12))) < 2:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def _mean_signed(vals: List[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _config_arrays(configs: List[Config]):
    return (np.array([c.k for c in configs], float),
            np.array([c.rho for c in configs], float),
            np.array([c.alpha_U for c in configs], float),
            np.array([c.alpha_I for c in configs], float))


def steady_state_predictions(theta: Tuple[float, float, float, float],
                             configs: List[Config]) -> Tuple[List[float], List[float]]:
    """(rho_U*, rho_I*) for each config under parameters theta (vectorised)."""
    ks, rhos, aUs, aIs = _config_arrays(configs)
    u, i = steady_state_batch(theta, ks, rhos, aUs, aIs)
    return list(map(float, u)), list(map(float, i))


def fit_theta(configs: List[Config],
              bounds=((0.02, 5.0), (0.02, 5.0), (1e-3, 5.0), (1e-3, 5.0))
              ) -> Tuple[float, float, float, float]:
    """Fit theta=(beta_U,beta_I,gamma,r) to MAXIMISE the signed mean Spearman
    between the coupled steady state and the observed steady-state means,
    pooled over BOTH partitions and BOTH sweeps (one theta explains many
    targets -> not an in-sample-per-point overfit). Exposure (transient) has
    no parameters, so it is never fitted."""
    if len(configs) < 3:
        return (0.6, 0.6, 0.1, 0.05)

    ks, rhos, aUs, aIs = _config_arrays(configs)
    y_u = np.array([c.U_ss for c in configs], float)
    item_mask = np.array([c.has_item for c in configs], bool)
    y_i = np.array([c.I_ss for c in configs], float)

    u_has_var = len(set(np.round(y_u, 9))) > 1
    i_has_var = item_mask.any() and len(set(np.round(y_i[item_mask], 9))) > 1
    PENALTY = -0.5  # a constant predictor cannot rank -> treated as a failure,

    def _score(pred, y, has_var):
        if not has_var:
            return None  # target itself is constant; nothing to rank
        s = _signed_spearman(pred, y)
        return PENALTY if math.isnan(s) else s  # penalise constant predictors

    def neg_obj(theta):
        if any(t <= 0 for t in theta):
            return 1.0
        u, i = steady_state_batch(tuple(theta), ks, rhos, aUs, aIs)
        parts = [_score(u, y_u, u_has_var)]
        if i_has_var:
            parts.append(_score(i[item_mask], y_i[item_mask], True))
        parts = [p for p in parts if p is not None]
        return -float(np.mean(parts)) if parts else 1.0

    # coarse grid then Nelder-Mead refinement
    grid = [0.05, 0.2, 0.6, 1.5, 3.0]
    grid_g = [0.01, 0.05, 0.2, 0.8]
    best, best_v = (0.6, 0.6, 0.1, 0.05), 1.0
    for bU in grid:
        for bI in grid:
            for g in grid_g:
                for rr in grid_g:
                    v = neg_obj((bU, bI, g, rr))
                    if v < best_v:
                        best_v, best = v, (bU, bI, g, rr)
    res = minimize(neg_obj, np.array(best), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    theta = tuple(float(max(t, lo)) for t, (lo, hi) in zip(res.x, bounds))
    return theta  # type: ignore


# ---------------------------------------------------------------------------
# Correlation report (signed, no flooring, no short-circuit)
# ---------------------------------------------------------------------------

KUI_SWEEP = "kUI"      # k in {1,2,3} at n_items=100
RHOIM_SWEEP = "rhoIM"  # n_items in {50,100,200} at k=2


def _sweep_subset(configs: List[Config], sweep: str) -> List[int]:
    if sweep == KUI_SWEEP:
        return [j for j, c in enumerate(configs) if c.n_items == 100]
    return [j for j, c in enumerate(configs) if c.k == 2]


def evaluate(configs_by_attack: Dict[str, List[Config]]) -> Dict[str, dict]:
    """For each attack: fit ONE theta (pooled over both sweeps), then report
    SIGNED Spearman per sweep (kUI, rhoIM) x partition (U,I) x regime (tr,ss):
      * transient : exposure vs {U,I}_tr   (no free parameters)
      * steady    : coupled rho* vs {U,I}_ss   (the P2/RMP-coupled predictor)
    plus the coupled R0 severity ranking. Layout mirrors the paper's Table 2.
    """
    out: Dict[str, dict] = {}
    for attack, configs in configs_by_attack.items():
        if len(configs) < 3:
            continue
        theta = fit_theta(configs)
        us, is_ = steady_state_predictions(theta, configs)
        exp_u = [exposure_U(c.alpha_I, c.k) for c in configs]
        exp_i = [exposure_I(c.alpha_U, c.rho, c.k) for c in configs]

        def _corr(idx, pred, key):
            vals = [pred[j] for j in idx]
            y = [getattr(configs[j], key) for j in idx]
            return _signed_spearman(vals, y)

        rep = {"theta": theta, "n_configs": len(configs), "sweeps": {}}
        for sweep in (KUI_SWEEP, RHOIM_SWEEP):
            idx = _sweep_subset(configs, sweep)
            item_idx = [j for j in idx if configs[j].has_item]
            rep["sweeps"][sweep] = {
                "U_tr": _corr(idx, exp_u, "U_tr"),
                "I_tr": _corr(item_idx, exp_i, "I_tr") if item_idx else float("nan"),
                "U_ss": _corr(idx, us, "U_ss"),
                "I_ss": _corr(item_idx, is_, "I_ss") if item_idx else float("nan"),
            }
        out[attack] = rep
    return out


def leave_one_attack_out(configs_by_attack: Dict[str, List[Config]]
                         ) -> Dict[str, dict]:
    """Out-of-sample check: fit theta on all attacks except one, then score the
    held-out attack's steady-state ranking with that theta. Reported alongside
    the in-sample numbers so the reader can see generalisation, not just fit."""
    attacks = [a for a, c in configs_by_attack.items() if len(c) >= 3]
    out = {}
    for held in attacks:
        train = [c for a in attacks if a != held for c in configs_by_attack[a]]
        if len(train) < 3:
            continue
        theta = fit_theta(train)
        cfgs = configs_by_attack[held]
        us, is_ = steady_state_predictions(theta, cfgs)
        item_idx = [j for j, c in enumerate(cfgs) if c.has_item]
        out[held] = {
            "theta_train": theta,
            "U_ss": _signed_spearman(us, [c.U_ss for c in cfgs]),
            "I_ss": (_signed_spearman([is_[j] for j in item_idx],
                                      [cfgs[j].I_ss for j in item_idx])
                     if item_idx else float("nan")),
        }
    return out


# ---------------------------------------------------------------------------
# Predictive-power comparison (which predictor is actually better?)
# ---------------------------------------------------------------------------

def fit_one_hop(configs: List[Config]) -> Tuple[float, float]:
    """Fit (gamma, r) for the paper's ORIGINAL frozen one-hop odds R, on the
    same footing as the coupled fit: maximise mean signed Spearman of R vs the
    steady-state ASR, pooled over U and I. Used only for the comparison."""
    yU = [c.U_ss for c in configs]
    yI = [c.I_ss for c in configs]
    has_i = any(c.has_item for c in configs)
    best, bv = (0.1, 0.05), 1.0
    for g in np.logspace(-3, 0.7, 24):
        for r in np.logspace(-3, 0.7, 24):
            U = [one_hop_R(exposure_U(c.alpha_I, c.k), c.k, c.alpha_I, g, r) for c in configs]
            parts = [_signed_spearman(U, yU)]
            if has_i:
                I = [one_hop_R(exposure_I(c.alpha_U, c.rho, c.k),
                               item_degree(c.rho, c.k), c.alpha_U, g, r) for c in configs]
                parts.append(_signed_spearman(I, yI))
            parts = [p for p in parts if not math.isnan(p)]
            v = -float(np.mean(parts)) if parts else 1.0
            if v < bv:
                bv, best = v, (g, r)
    return best


def _predict_pair(kind: str, params, configs: List[Config]):
    """(U_vals, I_vals) for a candidate steady-state predictor."""
    if kind == "E":       # exposure, 0 params
        return ([exposure_U(c.alpha_I, c.k) for c in configs],
                [exposure_I(c.alpha_U, c.rho, c.k) for c in configs])
    if kind == "R1hop":   # frozen one-hop odds, 2 params (paper's original)
        g, r = params
        return ([one_hop_R(exposure_U(c.alpha_I, c.k), c.k, c.alpha_I, g, r) for c in configs],
                [one_hop_R(exposure_I(c.alpha_U, c.rho, c.k),
                           item_degree(c.rho, c.k), c.alpha_U, g, r) for c in configs])
    if kind == "rho":     # coupled fixed point, 4 params
        u, i = steady_state_predictions(params, configs)
        return u, i
    if kind == "R0":      # coupled scalar, 4 params
        v = [sis_R0(*params, c.k, c.rho) for c in configs]
        return v, v
    raise ValueError(kind)


CANDIDATES = [("E(0p)", "E"), ("R_1hop(2p)", "R1hop"),
              ("rho*(4p)", "rho"), ("R0(4p)", "R0")]


def compare_predictors(configs_by_attack: Dict[str, List[Config]]
                       ) -> Tuple[Dict[str, dict], List[str]]:
    """Head-to-head steady-state ranking power of every candidate predictor,
    IN-SAMPLE vs LEAVE-ONE-ATTACK-OUT, over the attacks that carry both user
    and item contamination (dissemination). This is the honest test: fitted
    models must clear the 0-parameter exposure baseline OUT of sample, or they
    are overfitting rather than predicting. Returns (per-predictor accumulator
    of per-attack signed Spearman, attack list)."""
    attacks = [a for a, c in configs_by_attack.items()
               if len(c) >= 4 and any(x.has_item for x in c)]
    acc = {name: {(p, s): [] for p in ("in", "loao") for s in ("U", "I")}
           for name, _ in CANDIDATES}
    for held in attacks:
        cfgs = configs_by_attack[held]
        others = [c for a in attacks if a != held for c in configs_by_attack[a]]
        fit_in = {"R1hop": fit_one_hop(cfgs), "rho": fit_theta(cfgs)}
        fit_in["R0"] = fit_in["rho"]
        fit_lo = {"R1hop": fit_one_hop(others), "rho": fit_theta(others)}
        fit_lo["R0"] = fit_lo["rho"]
        yU, yI = [c.U_ss for c in cfgs], [c.I_ss for c in cfgs]
        for name, kind in CANDIDATES:
            for proto, fits in (("in", fit_in), ("loao", fit_lo)):
                params = None if kind == "E" else fits[kind]
                U, I = _predict_pair(kind, params, cfgs)
                acc[name][(proto, "U")].append(_signed_spearman(U, yU))
                acc[name][(proto, "I")].append(_signed_spearman(I, yI))
    return acc, attacks


def _nanmean(vals: List[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def print_comparison(acc: Dict[str, dict], attacks: List[str]) -> None:
    print("=" * 76)
    print("PREDICTIVE POWER -- steady-state ASR ranking, mean signed Spearman")
    print(f"over dissemination attacks {attacks}")
    print("(each attack: 5 configs = k-sweep + rho-sweep; per partition)")
    print("=" * 76)
    print(f"{'predictor':<12}   {'IN-SAMPLE':^15}    {'LEAVE-ONE-ATTACK-OUT':^15}")
    print(f"{'':<12}   {'U_ss':>6} {'I_ss':>6}    {'U_ss':>6} {'I_ss':>6}")
    print("-" * 62)
    for name, _ in CANDIDATES:
        d = acc[name]
        iu, ii = _nanmean(d[("in", "U")]), _nanmean(d[("in", "I")])
        lu, li = _nanmean(d[("loao", "U")]), _nanmean(d[("loao", "I")])
        print(f"{name:<12}   {iu:>+6.2f} {ii:>+6.2f}    {lu:>+6.2f} {li:>+6.2f}")
    print("-" * 62)
    print("Read the LEAVE-ONE-ATTACK-OUT columns: a fitted model that only wins")
    print("in-sample is overfitting. Exposure (0 params) is the bar to beat.")


# ---------------------------------------------------------------------------
# CSV I/O + CLI  (reproduces §7 from cached per-run tr/ss without raw traces)
# ---------------------------------------------------------------------------

def load_configs_from_raw_csv(path: str) -> Dict[str, List[Config]]:
    """Build per-attack Config lists from ranking_predictors_raw.csv, averaging
    seed replicates into one point per (attack, metric_type, k, n_items)."""
    import csv
    agg: Dict[Tuple[str, str], Dict[Tuple[int, int], Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    alpha_of: Dict[Tuple[str, str], float] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            key = (row["attack"], row["metric_type"])
            ck = (int(row["k_UI"]), int(row["n_items"]))
            for m in ("U_tr", "U_ss", "I_tr", "I_ss"):
                try:
                    agg[key][ck][m].append(float(row[m]))
                except (ValueError, KeyError):
                    pass
            try:
                alpha_of[key] = float(row["alpha"])
            except (ValueError, KeyError):
                alpha_of.setdefault(key, 0.25)
    out: Dict[str, List[Config]] = {}
    for key, byck in agg.items():
        means = {ck: {m: float(np.mean(v)) for m, v in d.items()} for ck, d in byck.items()}
        has_item = max((mv.get("I_ss", 0.0) + mv.get("I_tr", 0.0)) for mv in means.values()) > 1e-9
        a = alpha_of[key]
        cfgs = [Config(k=k, n_items=ni, alpha_U=a, alpha_I=a,
                       U_tr=mv.get("U_tr", 0.0), U_ss=mv.get("U_ss", 0.0),
                       I_tr=mv.get("I_tr", 0.0), I_ss=mv.get("I_ss", 0.0),
                       has_item=has_item)
                for (k, ni), mv in sorted(means.items())]
        out["/".join(key)] = cfgs
    return out


def write_corr_csv(evaluation: Dict[str, dict], path: str) -> None:
    """Write the corrected, SIGNED correlation table (supersedes the old
    max(R,E), |rho| ranking_predictors_corr.csv)."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attack", "sweep", "predictor_transient", "predictor_steady",
                    "U_tr", "I_tr", "U_ss", "I_ss",
                    "beta_U", "beta_I", "gamma", "r"])
        for atk, rr in evaluation.items():
            bU, bI, g, r = rr["theta"]
            for sw, s in rr["sweeps"].items():
                w.writerow([atk, sw, "exposure", "rho_star",
                            _fmt(s["U_tr"]), _fmt(s["I_tr"]),
                            _fmt(s["U_ss"]), _fmt(s["I_ss"]),
                            round(bU, 5), round(bI, 5), round(g, 5), round(r, 5)])


def _fmt(x: float) -> str:
    return "" if (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_csv", default=os.path.join(
        os.path.dirname(__file__), "ranking_pred_output", "ranking_predictors_raw.csv"))
    ap.add_argument("--out_csv", default=None,
                    help="Where to write the corrected corr CSV (default: alongside raw).")
    ap.add_argument("--compare", action="store_true",
                    help="Run the head-to-head predictive-power comparison "
                         "(exposure vs frozen one-hop R vs coupled rho* vs R0, "
                         "in-sample vs leave-one-attack-out) and exit.")
    args = ap.parse_args(argv)

    configs = load_configs_from_raw_csv(args.raw_csv)

    if args.compare:
        acc, attacks = compare_predictors(configs)
        print_comparison(acc, attacks)
        return

    ev = evaluate(configs)
    loo = leave_one_attack_out(configs)

    print("=" * 88)
    print("§7 predictive metrics -- SIGNED Spearman (coupled rho*; no |rho|, no max(R,E))")
    print("  transient <- exposure (no free params);  steady <- coupled fixed point rho*")
    print("=" * 88)
    hdr = f"{'attack/metric':<24} {'sweep':<6} {'U_tr':>7} {'I_tr':>7} {'U_ss':>7} {'I_ss':>7}"
    print(hdr); print("-" * len(hdr))
    for atk, rr in ev.items():
        for i, (sw, s) in enumerate(rr["sweeps"].items()):
            name = atk if i == 0 else ""
            print(f"{name:<24} {sw:<6} "
                  f"{_fmt(s['U_tr']) or 'nan':>7} {_fmt(s['I_tr']) or 'nan':>7} "
                  f"{_fmt(s['U_ss']) or 'nan':>7} {_fmt(s['I_ss']) or 'nan':>7}")
    print("\nLeave-one-attack-out (out-of-sample steady-state ranking):")
    for atk, rr in loo.items():
        print(f"  {atk:<24} U_ss={_fmt(rr['U_ss']) or 'nan':>7}  I_ss={_fmt(rr['I_ss']) or 'nan':>7}")

    out_csv = args.out_csv or os.path.join(
        os.path.dirname(args.raw_csv), "ranking_predictors_corr_signed.csv")
    write_corr_csv(ev, out_csv)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
