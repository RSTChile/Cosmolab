#!/usr/bin/env python3
"""
CG002 — excess_L2 dinámico vs línea base grain null (ℤ_n)
========================================================
Mide la MISMA métrica que grain_null_model.py sobre supervivientes
del motor de campo medio (misma física que constantes/exceso barrido).

Inicialización: ω_i ~ uniforme en Z_n (grano explícito).
Compatibilidad: c_ij = cos(2π(ω_i−ω_j)/n). Solo S evoluciona; ω fijo.

Compara exponente log-log en N con nulo −½ y prefactor √((n−1)/n).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parámetros (baseline CG002) ----
ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0
PASOS, THETA_CP, ALPHA = 240, 0.0, 1.0
N_PHASE = 8
NS = [250, 500, 1000, 2000, 4000]  # 8000 omitido (coste O(N·pasos); 4000 basta para exponente)
SEEDS_PER_N = 50

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_dynamic_l2_sweep.csv"
FITS_CSV = OUT_DIR / "cg002_dynamic_l2_fits.csv"
FIG_PATH = OUT_DIR / "cg002_dynamic_l2_vs_null.png"


def sat(S):
    return S / (1.0 + S / S_BAND)


def compat_matrix(omega: np.ndarray, n: int) -> np.ndarray:
    dphi = 2.0 * math.pi * (omega[:, None] - omega[None, :]) / n
    return np.cos(dphi)


def evolucion(omega: np.ndarray, n_phase: int):
    N = len(omega)
    C = compat_matrix(omega, n_phase)
    np.fill_diagonal(C, 0.0)
    # self-term analog: diagonal contribution from field medio with theta=0
    # g_{i<-i} not in graph; use mean field: dS_i = η m_i (Σ_j m_j c_ij) - η m_i² (self compat=1)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    for _ in range(PASOS):
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        coop = C @ m
        dS = ALPHA * ETA * m * coop - ALPHA * ETA * m * m
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
    return S, alive


def excess_L2_counts(counts: np.ndarray, n: int) -> float:
    N = counts.sum()
    if N == 0:
        return 0.0
    f = counts / N
    u = 1.0 / n
    return float(np.sqrt(((f - u) ** 2).sum()))


def excess_max_counts(counts: np.ndarray, n: int) -> float:
    N = counts.sum()
    if N == 0:
        return 0.0
    return float(counts.max() / N - 1.0 / n)


def orden_sphere(omega: np.ndarray, alive: np.ndarray, n: int) -> float:
    """Vector medio en ℝ² embebido (fase → coseno/seno) para comparar con exceso orden."""
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    ang = 2.0 * math.pi * omega[idx] / n
    u2 = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return float(np.linalg.norm(u2.mean(0)))


def nulo_geom_2d() -> float:
    return 2.0 / math.pi  # hemisferio uniforme en círculo (referencia)


def run_cosmos(N: int, n_phase: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    omega = rng.integers(0, n_phase, size=N)
    S, alive = evolucion(omega, n_phase)
    idx = np.where(alive)[0]
    n_surv = len(idx)
    counts = np.bincount(omega[idx], minlength=n_phase)
    el2 = excess_L2_counts(counts, n_phase)
    emax = excess_max_counts(counts, n_phase)
    orden = orden_sphere(omega, alive, n_phase)
    # nulo finito MC rápido para orden (hemisferio en círculo, n_surv puntos)
    nf_orden = nulo_orden_finito(n_surv, n_phase, seed=seed + 999)
    return dict(
        N=N, seed=seed, n_surv=n_surv, f_surv=n_surv / N,
        excess_L2=el2, excess_max=emax,
        orden=orden, orden_excess=orden - nf_orden,
        L2_null_allN=np.sqrt((n_phase - 1) / (n_phase * N)),
        L2_null_surv=np.sqrt((n_phase - 1) / (n_phase * max(n_surv, 1))),
    )


def nulo_orden_finito(n: int, n_phase: int, seed: int = 0) -> float:
    if n <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(200):
        ang = 2.0 * math.pi * rng.integers(0, n_phase, n) / n_phase
        u2 = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        # hemisferio: componente x > 0 tras rotar eje dominante aleatorio
        v = u2.mean(0)
        vals.append(np.linalg.norm(v))
    return float(np.mean(vals))


def loglog_fit(x, y):
    lx, ly = np.log(x.astype(float)), np.log(y.astype(float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, inter = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(np.exp(inter))


def main():
    t0 = time.monotonic()
    rows = []
    for ni, N in enumerate(NS):
        for seed in range(1, SEEDS_PER_N + 1):
            rows.append(run_cosmos(N, N_PHASE, seed))
        sub = [r for r in rows if r["N"] == N]
        m = np.mean([r["excess_L2"] for r in sub])
        ns = np.mean([r["n_surv"] for r in sub])
        null_s = np.sqrt((N_PHASE - 1) / (N_PHASE * ns))
        print(f"  [{ni+1}/{len(NS)}] N={N:5d}  L2_dyn={m:.4f}  L2_null_surv={null_s:.4f}  "
              f"ratio={m/null_s:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    # agregados por N
    agg = df.groupby("N").agg(
        excess_L2=("excess_L2", "mean"),
        excess_L2_sd=("excess_L2", "std"),
        excess_max=("excess_max", "mean"),
        orden_excess=("orden_excess", "mean"),
        n_surv=("n_surv", "mean"),
        f_surv=("f_surv", "mean"),
        L2_null_allN=("L2_null_allN", "first"),
        L2_null_surv=("L2_null_surv", "mean"),
    ).reset_index()

    fit_rows = []
    for col in ["excess_L2", "excess_max", "orden_excess"]:
        s, p = loglog_fit(agg.N.values, agg[col].values)
        fit_rows.append(dict(metric=col, exponente=s, prefactor=p, null_exp=-0.5))
    fitdf = pd.DataFrame(fit_rows)
    fitdf.to_csv(FITS_CSV, index=False)

    # null grain reference at same N (all nodes)
    null_ref = pd.read_csv(OUT_DIR / "grain_null_sweep.csv")
    null8 = null_ref[(null_ref.n == N_PHASE)].set_index("N")

    # ---- figura ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.loglog(agg.N, agg.excess_L2, "o-", color="crimson", lw=2, ms=7,
              label="dinámico L2 (supervivientes)")
    ax.loglog(agg.N, agg.L2_null_allN, "s--", color="steelblue", alpha=0.8,
              label="nulo grain (todos N, MC ref)")
    Ngrid = agg.N.values.astype(float)
    ax.loglog(Ngrid, np.sqrt((N_PHASE - 1) / (N_PHASE * Ngrid)), "k:",
              alpha=0.5, label="nulo analítico all-N")
    ax.loglog(agg.N, agg.L2_null_surv, "^--", color="teal", alpha=0.8,
              label="nulo analítico n_surv")
    ax.set_xlabel("N (nodos iniciales)")
    ax.set_ylabel("excess_L2")
    ax.set_title(f"CG002 dinámico vs grain null (Z_{N_PHASE})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ratio = agg.excess_L2 / agg.L2_null_surv
    ax2.semilogx(agg.N, ratio, "o-", color="darkgreen", lw=2)
    ax2.axhline(1.0, color="gray", ls="--", label="= nulo (solo grano)")
    ax2.set_xlabel("N")
    ax2.set_ylabel("dinámico / nulo(n_surv)")
    ax2.set_title("¿Se aparta de la línea base?")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    exp_l2 = fitdf[fitdf.metric == "excess_L2"].exponente.values[0]
    exp_ord = fitdf[fitdf.metric == "orden_excess"].exponente.values[0]
    ratio_2000 = ratio[agg.N == 2000].values[0]

    print("\n" + "=" * 60)
    print("CG002 dinámico L2 vs grain null")
    print("=" * 60)
    print(f"Exponente L2 dinámico: {exp_l2:+.4f}  (nulo = -0.500)")
    print(f"Exponente orden_excess:  {exp_ord:+.4f}")
    print(f"Ratio dinámico/nulo @ N=2000: {ratio_2000:.3f}")
    if abs(exp_l2 + 0.5) < 0.05:
        print("→ L2 decae como 1/√N: MISMO exponente que nulo combinatorio.")
    else:
        print("→ L2 NO sigue −½: posible firma estructural κ_Δ.")
    if ratio_2000 > 1.05:
        print(f"→ A N=2000 el dinámico está {100*(ratio_2000-1):.1f}% SOBRE nulo n_surv.")
    elif ratio_2000 < 0.95:
        print(f"→ A N=2000 el dinámico está {100*(1-ratio_2000):.1f}% BAJO nulo n_surv.")
    else:
        print("→ A N=2000 coincide con nulo n_surv (~solo grano).")
    print(f"\nCSV: {SWEEP_CSV}")
    print(f"Fig:  {FIG_PATH}")
    print(f"Tiempo: {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()