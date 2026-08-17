#!/usr/bin/env python3
"""
CG002-R7a — color × spin (plasma QCD, sin confinamiento)
========================================================
SPEC cerrado (Alexis / CC Msg 26, Grok implementa 30-jun-2026).

Misma física CG002 (campo medio O(N), sat C-N5.1, extinción κ_s, θ_CP=0).
Ω_firma reemplazado: un sabor; quarks {r,g,b} + antiquarks {r̄,ḡ,b̄} × spin {↑,↓} = 12 microestados.
Compatibilidad c_ij = producto interno rep. fundamental SU(3) en color (antiquark = −v_c).
Spin pasivo v0: etiqueta de multiplicidad, NO entra en c_ij.

Semilla: composición simétrica quark/antiquark; sin CP inyectada; sin confinamiento como regla.
K_eff falsable: reporta ajuste a K ∈ {6, 8, 12} (no fijado a ojo).

Observables: f, m_eff/K (cada K), L2 vs nulo, exponente en N, τ, neutralidad color supervivientes,
control shuffle-B vs QCD-real.

USO:
  python3 cg002_r7a_color_spin.py           # barrido completo
  python3 cg002_r7a_color_spin.py --quick   # N reducido, pocas semillas
  python3 cg002_r7a_color_spin.py --single  # una corrida diagnóstico
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- parámetros CG002 (baseline arco) ----
ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0
PASOS, THETA_CP, ALPHA = 240, 0.0, 1.0

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_r7a_sweep.csv"
FITS_CSV = OUT_DIR / "cg002_r7a_fits.csv"
FIG_PATH = OUT_DIR / "cg002_r7a_vs_null.png"

# rep. fundamental SU(3): base ortonormal en ℝ³
COLORS = np.eye(3, dtype=float)
COLOR_NAMES = ("r", "g", "b")
N_MICRO = 12  # 6 cargas × 2 spin
K_CANDIDATES = (6, 8, 12)


def sat(S: np.ndarray) -> np.ndarray:
    return S / (1.0 + S / S_BAND)


def color_vector(is_antiquark: bool, color_idx: int) -> np.ndarray:
    v = COLORS[color_idx]
    return -v if is_antiquark else v


def microstate_index(is_antiquark: bool, color_idx: int, spin: int) -> int:
    """0..11: quarks 0-5, antiquarks 6-11."""
    return (3 if is_antiquark else 0) + color_idx * 2 + spin


def decode_microstate(idx: int) -> tuple[bool, int, int]:
    is_aq = idx >= 6
    base = idx - 6 if is_aq else idx
    return is_aq, base // 2, base % 2


def compat_matrix(V: np.ndarray) -> np.ndarray:
    """c_ij = v_i · v_j (color only; spin pasivo)."""
    C = V @ V.T
    np.fill_diagonal(C, 0.0)
    return C


def evolucion(V: np.ndarray, pasos: int = PASOS) -> tuple[np.ndarray, np.ndarray, int]:
    """Campo medio con matriz de compatibilidad; θ_CP=0 → coop simétrica."""
    N = len(V)
    C = compat_matrix(V)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0
    for _ in range(pasos):
        d_struct = 0.0
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        coop = C @ m
        dS = ALPHA * ETA * m * coop - ALPHA * ETA * m * m
        d_struct = float(np.abs(dS[alive]).sum()) if alive.any() else 0.0
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
        if d_struct > 1e-4:
            tau += 1
    return S, alive, tau


def semilla_simetrica(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Composición inicial simétrica: igual número por microestado (12 clases).
    θ_CP = 0 (no se pasa a evolución).
    """
    counts = np.full(N_MICRO, N // N_MICRO)
    for _ in range(N % N_MICRO):
        counts[rng.integers(0, N_MICRO)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.empty((N, 3))
    pos = 0
    for ms in range(N_MICRO):
        is_aq, ci, sp = decode_microstate(ms)
        n = counts[ms]
        labels[pos : pos + n] = ms
        V[pos : pos + n] = color_vector(is_aq, ci)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm]


def semilla_shuffle_b(labels: np.ndarray, V: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Control null: permuta etiquetas de color (rompe correlación física SU(3))."""
    perm_color = rng.permutation(3)
    V2 = V.copy()
    lab2 = labels.copy()
    for i in range(len(labels)):
        is_aq, ci, sp = decode_microstate(int(labels[i]))
        ci2 = int(perm_color[ci])
        lab2[i] = microstate_index(is_aq, ci2, sp)
        V2[i] = color_vector(is_aq, ci2)
    return lab2, V2


def bin_counts(labels: np.ndarray, alive: np.ndarray, K: int) -> np.ndarray:
    """Agrupa supervivientes en K bins según candidato K_eff."""
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return np.zeros(K, dtype=int)
    surv = labels[idx]
    if K == 12:
        return np.bincount(surv, minlength=12)
    if K == 6:
        # sector quark: microestados 0-5
        q = surv[surv < 6]
        return np.bincount(q, minlength=6)
    if K == 8:
        # octeto operativo: 6 cargas de color + 2 spin globales
        cnt = np.zeros(8, dtype=int)
        for ms in surv:
            is_aq, ci, sp = decode_microstate(int(ms))
            charge = 3 if is_aq else 0
            charge += ci
            cnt[charge] += 1
            cnt[6 + sp] += 1
        return cnt
    raise ValueError(f"K no soportado: {K}")


def excess_L2(counts: np.ndarray, K: int) -> float:
    N = counts.sum()
    if N == 0:
        return 0.0
    f = counts / N
    u = 1.0 / K
    return float(np.sqrt(((f - u) ** 2).sum()))


def meff_over_K(counts: np.ndarray, K: int) -> float:
    N = counts.sum()
    if N == 0:
        return 0.0
    f = counts / N
    sf2 = float((f * f).sum())
    return 1.0 / (K * sf2) if sf2 > 0 else 0.0


def null_L2(K: int, N: int) -> float:
    return math.sqrt((K - 1) / (K * max(N, 1)))


def orden_color(V: np.ndarray, alive: np.ndarray) -> float:
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.linalg.norm(V[idx].mean(0)))


def frac_color_neutral(V: np.ndarray, alive: np.ndarray, tol: float = 0.15) -> float:
    """
    Fracción de supervivientes cuya «vecindad global» deja el vector color casi neutro.
    v0: proxy = |v_i · v_mean| pequeño (cada partícula casi ortogonal al orden global).
    """
    idx = np.where(alive)[0]
    if len(idx) < 2:
        return 0.0
    vm = V[idx].mean(0)
    nm = np.linalg.norm(vm)
    if nm < 1e-9:
        return 1.0
    vm /= nm
    dots = V[idx] @ vm
    return float((np.abs(dots) < tol).mean())


def run_once(N: int, seed: int, mode: str = "qcd") -> dict:
    rng = np.random.default_rng(seed)
    labels, V = semilla_simetrica(N, rng)
    if mode == "shuffle_b":
        labels, V = semilla_shuffle_b(labels, V, rng)
    ord_init = orden_color(V, np.ones(N, bool))
    S, alive, tau = evolucion(V)
    n_surv = int(alive.sum())
    f = n_surv / N
    ord_c = orden_color(V, alive)
    ord_amp = ord_c / (ord_init + 1e-9)
    frac_neut = frac_color_neutral(V, alive)

    row = dict(
        N=N, seed=seed, mode=mode, n_surv=n_surv, f_surv=f,
        tau=tau, orden_color_init=ord_init, orden_color=ord_c, orden_amp=ord_amp,
        frac_color_neutral=frac_neut,
    )
    for K in K_CANDIDATES:
        cnt = bin_counts(labels, alive, K)
        row[f"L2_K{K}"] = excess_L2(cnt, K)
        row[f"meff_K{K}"] = meff_over_K(cnt, K)
        row[f"null_L2_K{K}"] = null_L2(K, n_surv)
        row[f"ratio_L2_K{K}"] = row[f"L2_K{K}"] / null_L2(K, n_surv) if n_surv else 0.0
        row[f"grain_K{K}"] = 1.0 / math.sqrt(K)
    return row


def loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lx, ly = np.log(x.astype(float)), np.log(y.astype(float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, inter = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(np.exp(inter))


def best_K_excess(rows: list[dict], shuffle_rows: list[dict] | None = None) -> dict:
    """K con máximo exceso L2/nulo; prefiere shuffle-positivo (CC Msg 28)."""
    qcd = [r for r in rows if r.get("mode", "qcd") == "qcd"]
    scores = {K: float(np.mean([r[f"ratio_L2_K{K}"] for r in qcd])) for K in K_CANDIDATES}
    shuffle_pos = set(K_CANDIDATES)
    if shuffle_rows:
        for K in K_CANDIDATES:
            dq = np.mean([r[f"L2_K{K}"] for r in qcd])
            ds = np.mean([r[f"L2_K{K}"] for r in shuffle_rows])
            if dq - ds <= 0:
                shuffle_pos.discard(K)
    candidates = [K for K in K_CANDIDATES if K in shuffle_pos] or list(K_CANDIDATES)
    k_best = max(candidates, key=lambda k: scores[k])
    return dict(k_best=k_best, scores=scores, shuffle_positive=sorted(shuffle_pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    if args.single:
        r = run_once(2000, 1, "qcd")
        r2 = run_once(2000, 1, "shuffle_b")
        print("QCD-real:", r)
        print("shuffle-B:", r2)
        bk = best_K_excess([r], [r2])
        print("K_eff por exceso-sobre-nulo:", bk)
        print(f"  orden_amp={r['orden_amp']:.2f} (baryogénesis real si >>1)")
        return

    NS = [500, 1000, 2000] if args.quick else [250, 500, 1000, 2000, 4000]
    SEEDS = 10 if args.quick else 40

    t0 = time.monotonic()
    rows = []
    for ni, N in enumerate(NS):
        for seed in range(1, SEEDS + 1):
            rows.append(run_once(N, seed, "qcd"))
            rows.append(run_once(N, seed, "shuffle_b"))
        sub = [r for r in rows if r["N"] == N and r["mode"] == "qcd"]
        m6 = np.mean([r["meff_K6"] for r in sub])
        m12 = np.mean([r["meff_K12"] for r in sub])
        oa = np.mean([r["orden_amp"] for r in sub])
        r6 = np.mean([r["ratio_L2_K6"] for r in sub])
        print(f"  [{ni+1}/{len(NS)}] N={N:5d}  f={np.mean([r['f_surv'] for r in sub]):.3f}  "
              f"L2/null K6={r6:.1f}×  ord_amp={oa:.2f}  τ={np.mean([r['tau'] for r in sub]):.0f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    # agregados QCD-real
    qcd = df[df["mode"] == "qcd"]
    agg = qcd.groupby("N").agg(
        f_surv=("f_surv", "mean"),
        tau=("tau", "mean"),
        orden_color=("orden_color", "mean"),
        frac_neutral=("frac_color_neutral", "mean"),
        **{f"L2_K{K}": (f"L2_K{K}", "mean") for K in K_CANDIDATES},
        **{f"meff_K{K}": (f"meff_K{K}", "mean") for K in K_CANDIDATES},
        **{f"ratio_L2_K{K}": (f"ratio_L2_K{K}", "mean") for K in K_CANDIDATES},
    ).reset_index()

    fit_rows = []
    for K in K_CANDIDATES:
        s, p = loglog_fit(agg.N.values, agg[f"L2_K{K}"].values)
        fit_rows.append(dict(K=K, metric="L2", exponente=s, prefactor=p, null_exp=-0.5))
        s2, p2 = loglog_fit(agg.N.values, agg[f"meff_K{K}"].values)
        fit_rows.append(dict(K=K, metric="meff", exponente=s2, prefactor=p2, null_exp=0.0))
    fitdf = pd.DataFrame(fit_rows)
    fitdf.to_csv(FITS_CSV, index=False)

    shuf = df[df["mode"] == "shuffle_b"].to_dict("records")
    bk = best_K_excess(qcd.to_dict("records"), shuf)

    # shuffle-B vs QCD @ N=2000
    sep_rows = []
    for N in NS:
        q = qcd[qcd["N"] == N]
        s = df[(df["N"] == N) & (df["mode"] == "shuffle_b")]
        for K in K_CANDIDATES:
            sep_rows.append(dict(
                N=N, K=K,
                L2_qcd=q[f"L2_K{K}"].mean(),
                L2_shuffle=s[f"L2_K{K}"].mean(),
                delta_L2=q[f"L2_K{K}"].mean() - s[f"L2_K{K}"].mean(),
                meff_qcd=q[f"meff_K{K}"].mean(),
                meff_shuffle=s[f"meff_K{K}"].mean(),
            ))
    sepdf = pd.DataFrame(sep_rows)

    # figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for K, c in zip(K_CANDIDATES, ("crimson", "darkorange", "steelblue")):
        ax.loglog(agg.N, agg[f"L2_K{K}"], "o-", color=c, label=f"L2 K={K}")
        ng = agg.N.values.astype(float)
        ax.loglog(ng, np.sqrt((K - 1) / (K * ng)), "--", color=c, alpha=0.4)
    ax.set_xlabel("N"); ax.set_ylabel("excess_L2"); ax.set_title("R7a QCD-real vs nulo analítico")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    ax = axes[0, 1]
    x = np.arange(len(K_CANDIDATES))
    w = 0.35
    m_q = [agg[f"meff_K{K}"].mean() for K in K_CANDIDATES]
    m_s = [df[df["mode"] == "shuffle_b"][f"meff_K{K}"].mean() for K in K_CANDIDATES]
    ax.bar(x - w / 2, m_q, w, label="QCD-real", color="crimson", alpha=0.85)
    ax.bar(x + w / 2, m_s, w, label="shuffle-B", color="gray", alpha=0.7)
    ax.axhline(0.5, color="k", ls="--", alpha=0.5, label="½ (homología)")
    ax.set_xticks(x); ax.set_xticklabels([f"K={K}" for K in K_CANDIDATES])
    ax.set_ylabel("m_eff/K"); ax.set_title("Candidatos K_eff"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    sub2000 = sepdf[sepdf["N"] == (2000 if 2000 in NS else NS[-1])]
    ax.bar(sub2000.K.astype(str), sub2000.delta_L2, color="teal", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("K"); ax.set_ylabel("ΔL2 (QCD − shuffle-B)")
    ax.set_title("Separación control null")

    ax = axes[1, 1]
    ax.plot(agg.N, agg.f_surv, "o-", color="purple", label="f persiste")
    ax2 = ax.twinx()
    ax2.plot(agg.N, agg.frac_neutral, "s--", color="green", label="frac neutral color")
    ax.set_xlabel("N"); ax.set_ylabel("f"); ax2.set_ylabel("neutralidad")
    ax.set_title(f"τ medio={agg.tau.mean():.0f} · K_best={bk['k_best']}")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    print("\n" + "=" * 60)
    print("CG002-R7a color×spin (plasma QCD, θ_CP=0)")
    print("=" * 60)
    print(f"f persiste @ N=2000: {agg[agg['N'] == min(2000, NS[-1])].f_surv.values[0]:.3f}")
    for K in K_CANDIDATES:
        exp = fitdf[(fitdf.K == K) & (fitdf.metric == "L2")].exponente.values[0]
        meff = agg[f"meff_K{K}"].mean()
        print(f"  K={K:2d}: m_eff/K={meff:.4f}  exp_L2={exp:+.3f}  grano=1/√K={1/math.sqrt(K):.4f}")
    oa = qcd["orden_amp"].mean()
    print(f"K_eff por exceso: K={bk['k_best']}  ratios L2/null: {bk['scores']}")
    print(f"orden_amp medio (baryogénesis): {oa:.3f}  (>>1 = amplificación real)")
    d = sub2000.delta_L2.values
    if np.any(np.abs(d) > 0.02):
        print("→ shuffle-B SEPARA de QCD-real en al menos un K.")
    else:
        print("→ shuffle-B NO separa claramente (homología débil o N insuficiente).")
    print(f"\nCSV: {SWEEP_CSV}")
    print(f"Fig:  {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()