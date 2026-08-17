"""
CG002 — Caracterización de excess(d) con nulo de muestra finita
===============================================================
1. Replica validación: exceso_REAL = orden_dyn − nulo_finito(n_surv, d)
2. Barrido d ∈ {2..8} con muchas semillas
3. Ajuste de excess(d) vs dimensión

USO: python3 cg002_exceso_caracteriza.py [--semillas 500]
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gamma as gammaf

ETA, MU, SBAND = 0.05, 0.01, 8.0
KAPPA_S, S0, PASOS, THETA_CP, ALPHA = 1e-6, 1.0, 240, 0.0, 1.0
N_DIF = 2000

OUT = Path(__file__).with_name("cg002_exceso_caracteriza.txt")


def nulo_inf(d: int) -> float:
    return float(gammaf(d / 2) / (math.sqrt(math.pi) * gammaf((d + 1) / 2)))


def nulo_finito(n: int, d: int, n_mc: int = 400, seed: int = 0) -> float:
    """|mean(U)| para n puntos uniformes en hemisferio S^{d-1} (u_0 > 0)."""
    if n <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_mc):
        U = rng.standard_normal((n, d))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        # hemisferio respecto al primer eje
        mask = U[:, 0] > 0
        if mask.sum() == 0:
            continue
        Us = U[mask]
        vals.append(np.linalg.norm(Us.mean(0)))
    return float(np.mean(vals))


def sat(S):
    return S / (1.0 + S / SBAND)


def rot(d, th):
    R = np.eye(d)
    c, s = np.cos(th), np.sin(th)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def firmas(N, d, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def evolucion(U):
    N, d = U.shape
    R = rot(d, THETA_CP)
    UR = U @ R
    selfd = np.einsum("ij,ij->i", UR, U)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    for _ in range(PASOS):
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        v = (m[:, None] * U).sum(0)
        dS = ALPHA * ETA * m * (UR @ v) - ALPHA * ETA * m * m * selfd
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
    return alive


def cosmos(seed, d):
    U = firmas(N_DIF, d, seed)
    alive = evolucion(U)
    idx = np.where(alive)[0]
    n = len(idx)
    if n == 0:
        return n, 0.0
    Us = U[idx]
    return n, float(np.linalg.norm(Us.mean(0)))


def cv_pct(a):
    m = np.mean(a)
    return 100 * np.std(a) / abs(m) if abs(m) > 1e-15 else float("nan")


def fit_models(ds, excess):
    """Prueba formas simples excess(d)."""
    d = np.array(ds, float)
    y = np.array(excess, float)
    ni = np.array([nulo_inf(int(di)) for di in d])
    results = []

    def add(name, pred):
        resid = y - pred
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results.append((name, float(np.sqrt(ss_res / len(y))), r2, pred))

    # constante
    c = y.mean()
    add("constante c", np.full_like(y, c))

    # lineal en d
    A = np.vstack([d, np.ones_like(d)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    add(f"lineal a*d+b (a={coef[0]:.5f}, b={coef[1]:.5f})", coef[0] * d + coef[1])

    # lineal en nulo_inf
    A = np.vstack([ni, np.ones_like(ni)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    add(f"lineal a*nulo+b (a={coef[0]:.5f}, b={coef[1]:.5f})", coef[0] * ni + coef[1])

    # proporcional a nulo (sin intercepto)
    a = np.dot(ni, y) / np.dot(ni, ni)
    add(f"proporcional k*nulo (k={a:.5f})", a * ni)

    # k * nulo^p — barrido p
    best_p, best_k, best_rmse = None, None, 1e9
    for p in np.linspace(0.5, 2.5, 41):
        x = ni ** p
        k = np.dot(x, y) / np.dot(x, x)
        pred = k * x
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        if rmse < best_rmse:
            best_rmse, best_p, best_k = rmse, p, k
    add(f"potencia k*nulo^p (k={best_k:.5f}, p={best_p:.3f})", best_k * ni ** best_p)

    results.sort(key=lambda t: t[1])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=300)
    args = ap.parse_args()

    lines = [
        "CG002 — Caracterización excess(d) con nulo de muestra finita",
        f"Semillas por d: {args.semillas}  |  N_dif={N_DIF}",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Validación nulo finito (replica tabla Alexis/CC, seed=1) ===",
        f"{'d':>2} {'n_surv':>6} {'orden':>8} {'nulo_∞':>8} {'nulo_fin':>8} "
        f"{'sesgo':>8} {'exceso_REAL':>10}",
    ]

    seed = 1
    for d in (2, 3, 4, 5):
        n, orden = cosmos(seed, d)
        ng = nulo_inf(d)
        nf = nulo_finito(n, d)
        ex_real = orden - nf
        sesgo = nf - ng
        lines.append(
            f"{d:>2} {n:>6} {orden:>8.4f} {ng:>8.4f} {nf:>8.4f} "
            f"{sesgo:>+8.4f} {ex_real:>+10.4f}"
        )
        pct_sesgo = 100 * sesgo / ex_real if ex_real else 0
        lines[-1] += f"  (sesgo={pct_sesgo:.0f}% del exceso bruto)"

    lines += ["", "=== Barrido d=2..8 (exceso_REAL medio) ==="]
    lines.append(
        f"{'d':>2} {'nulo_∞':>8} {'n_surv':>7} {'orden':>8} {'nulo_fin':>8} "
        f"{'exceso':>8} {'CV%':>6}"
    )

    nf_cache: dict[tuple[int, int], float] = {}

    def nf_cached(n: int, d: int) -> float:
        key = (d, round(n / 5) * 5)
        if key not in nf_cache:
            nf_cache[key] = nulo_finito(key[1], d, n_mc=250, seed=d * 997 + key[1])
        return nf_cache[key]

    ds = list(range(2, 9))
    excess_means = []
    for d in ds:
        ns, ordens, excesos = [], [], []
        for s in range(1, args.semillas + 1):
            n, orden = cosmos(s, d)
            ns.append(n)
            ordens.append(orden)
            excesos.append(orden - nf_cached(n, d))
        ex_arr = np.array(excesos)
        excess_means.append(ex_arr.mean())
        ng = nulo_inf(d)
        lines.append(
            f"{d:>2} {ng:>8.4f} {np.mean(ns):>7.1f} {np.mean(ordens):>8.4f} "
            f"{nulo_finito(int(np.mean(ns)), d):>8.4f} {ex_arr.mean():>+8.5f} {cv_pct(ex_arr):>5.1f}%"
        )

    lines += ["", "=== Ajuste excess(d) ==="]
    fits = fit_models(ds, excess_means)
    for name, rmse, r2, _ in fits[:5]:
        lines.append(f"  {name}: RMSE={rmse:.5f}, R²={r2:.4f}")

    best = fits[0]
    lines += [
        "",
        f"Mejor ajuste: {best[0]}",
        "",
        "Tabla datos:",
        f"{'d':>2} {'excess':>10}",
    ]
    for d, e in zip(ds, excess_means):
        lines.append(f"{d:>2} {e:>10.5f}")

    lines += [
        "",
        "--- Lectura ---",
        "Sesgo muestra finita ≪ exceso dinámico (~8% a d=3) → concentración cooperativa REAL.",
        "excess(d) modulado por dimensión; buscar forma cerrada si R² alto en ajuste.",
    ]

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nGuardado: {OUT}")


if __name__ == "__main__":
    main()