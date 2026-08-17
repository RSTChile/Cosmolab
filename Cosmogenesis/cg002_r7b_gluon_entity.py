#!/usr/bin/env python3
"""
CG002-R7b — gluón como entidad (octeto SU(3) adjunto)
=====================================================
Extiende R7a: mismo motor CG002, θ_CP=0, semilla simétrica, sin confinamiento.

R7a: gluón = ley de acoplamiento (solo quarks/antiquarks, c_ij = v_i·v_j).
R7b: añade **8 gluones** como entidades en el lote (labels 12–19, spin-1 etiqueta).

Compatibilidad (color manda):
  · quark–quark / antiquark:  c_ij = v_i·v_j  (rep. fundamental, como R7a)
  · gluón–gluón:              c_ij = g_i·g_j  (8D, base octeto ortonormal)
  · quark–gluón:              c_ij = Re(v_i^T λ_a v_i)  (Gell-Mann, sin a mano)

K_eff falsable: K ∈ {8, 12, 20}  (octeto gluón / sector quark / espacio completo)
Controles: QCD-real, shuffle-B (color quark), shuffle-G (octeto gluón)
Comparación lado a lado con R7a (cg002_r7a_sweep.csv si existe).

USO:
  python3 cg002_r7b_gluon_entity.py
  python3 cg002_r7b_gluon_entity.py --quick
  python3 cg002_r7b_gluon_entity.py --single
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
SWEEP_CSV = OUT_DIR / "cg002_r7b_sweep.csv"
FITS_CSV = OUT_DIR / "cg002_r7b_fits.csv"
FIG_PATH = OUT_DIR / "cg002_r7b_vs_r7a.png"
R7A_SWEEP = OUT_DIR / "cg002_r7a_sweep.csv"

N_QUARK_MICRO = 12
N_GLUON_MICRO = 8
N_MICRO = N_QUARK_MICRO + N_GLUON_MICRO  # 20
K_CANDIDATES = (8, 12, 20)

COLORS = np.eye(3, dtype=float)

# Gell-Mann λ_1..λ_8 (Hermitian, convención PDG)
LAMBDA = np.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], float) / math.sqrt(3),
    ],
    dtype=complex,
)

# Octeto adjunto: 8 vectores ortonormales en ℝ⁸ (etiqueta gluón)
GLUON_BASIS = np.eye(N_GLUON_MICRO, dtype=float)


def sat(S: np.ndarray) -> np.ndarray:
    return S / (1.0 + S / S_BAND)


def is_gluon(label: int) -> bool:
    return label >= N_QUARK_MICRO


def decode_quark(ms: int) -> tuple[bool, int, int]:
    is_aq = ms >= 6
    base = ms - 6 if is_aq else ms
    return is_aq, base // 2, base % 2


def color_vector(is_antiquark: bool, color_idx: int) -> np.ndarray:
    v = COLORS[color_idx]
    return -v if is_antiquark else v


def quark_label(is_antiquark: bool, color_idx: int, spin: int) -> int:
    return (3 if is_antiquark else 0) + color_idx * 2 + spin


def gluon_label(octet_idx: int) -> int:
    return N_QUARK_MICRO + octet_idx


def quark_gluon_coupling(v: np.ndarray, octet_idx: int) -> float:
    """Acoplamiento quark–gluón vía generador λ_a (parte real, rep. fundamental)."""
    La = LAMBDA[octet_idx]
    return float(np.real(v @ La @ v))


def build_compat(V: np.ndarray, G: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Matriz N×N vectorizada: bloques qq, gg y acoplamiento q–g vía Gell-Mann."""
    N = len(labels)
    is_g = labels >= N_QUARK_MICRO
    C = np.zeros((N, N))
    idx_q = np.where(~is_g)[0]
    idx_g = np.where(is_g)[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    if len(idx_g):
        Gg = G[idx_g]
        C[np.ix_(idx_g, idx_g)] = Gg @ Gg.T
    if len(idx_q) and len(idx_g):
        for a in range(N_GLUON_MICRO):
            jg = idx_g[labels[idx_g] == gluon_label(a)]
            if len(jg) == 0:
                continue
            La = LAMBDA[a]
            coup = np.real(np.einsum("ni,ij,nj->n", V[idx_q], La, V[idx_q]))
            C[np.ix_(idx_q, jg)] = coup[:, None]
            C[np.ix_(jg, idx_q)] = coup[None, :]
    np.fill_diagonal(C, 0.0)
    return C


def evolucion(V: np.ndarray, G: np.ndarray, labels: np.ndarray, pasos: int = PASOS) -> tuple[np.ndarray, np.ndarray, int]:
    N = len(labels)
    C = build_compat(V, G, labels)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0
    for _ in range(pasos):
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


def semilla_simetrica(N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Igual número por microestado: 12 quark + 8 gluón = 20 clases."""
    counts = np.full(N_MICRO, N // N_MICRO)
    for _ in range(N % N_MICRO):
        counts[rng.integers(0, N_MICRO)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.zeros((N, 3))
    G = np.zeros((N, N_GLUON_MICRO))
    pos = 0
    for ms in range(N_MICRO):
        n = counts[ms]
        if ms < N_QUARK_MICRO:
            is_aq, ci, sp = decode_quark(ms)
            cv = color_vector(is_aq, ci)
            labels[pos : pos + n] = ms
            V[pos : pos + n] = cv
        else:
            oi = ms - N_QUARK_MICRO
            labels[pos : pos + n] = ms
            G[pos : pos + n] = GLUON_BASIS[oi]
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm], G[perm]


def semilla_shuffle_b(labels: np.ndarray, V: np.ndarray, G: np.ndarray, rng: np.random.Generator):
    perm_color = rng.permutation(3)
    lab2, V2, G2 = labels.copy(), V.copy(), G.copy()
    for i in range(len(labels)):
        if is_gluon(int(labels[i])):
            continue
        is_aq, ci, sp = decode_quark(int(labels[i]))
        ci2 = int(perm_color[ci])
        lab2[i] = quark_label(is_aq, ci2, sp)
        V2[i] = color_vector(is_aq, ci2)
    return lab2, V2, G2


def semilla_shuffle_g(labels: np.ndarray, G: np.ndarray, rng: np.random.Generator):
    perm_g = rng.permutation(N_GLUON_MICRO)
    lab2, G2 = labels.copy(), G.copy()
    for i in range(len(labels)):
        if not is_gluon(int(labels[i])):
            continue
        oi2 = int(perm_g[int(labels[i]) - N_QUARK_MICRO])
        lab2[i] = gluon_label(oi2)
        G2[i] = GLUON_BASIS[oi2]
    return lab2, G2


def bin_counts(labels: np.ndarray, alive: np.ndarray, K: int) -> np.ndarray:
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return np.zeros(K, dtype=int)
    surv = labels[idx]
    if K == 20:
        return np.bincount(surv, minlength=N_MICRO)
    if K == 12:
        q = surv[surv < N_QUARK_MICRO]
        return np.bincount(q, minlength=N_QUARK_MICRO)
    if K == 8:
        g = surv[surv >= N_QUARK_MICRO] - N_QUARK_MICRO
        return np.bincount(g, minlength=N_GLUON_MICRO)
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


def orden_color(V: np.ndarray, labels: np.ndarray, alive: np.ndarray) -> float:
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    mask = np.array([not is_gluon(int(labels[i])) for i in idx])
    if not mask.any():
        return 0.0
    return float(np.linalg.norm(V[idx][mask].mean(0)))


def frac_species(labels: np.ndarray, alive: np.ndarray) -> tuple[float, float]:
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    surv = labels[idx]
    fq = float((surv < N_QUARK_MICRO).mean())
    fg = float((surv >= N_QUARK_MICRO).mean())
    return fq, fg


def run_once(N: int, seed: int, mode: str = "qcd") -> dict:
    rng = np.random.default_rng(seed)
    labels, V, G = semilla_simetrica(N, rng)
    if mode == "shuffle_b":
        labels, V, G = semilla_shuffle_b(labels, V, G, rng)
    elif mode == "shuffle_g":
        labels, G = semilla_shuffle_g(labels, G, rng)
    ord_init = orden_color(V, labels, np.ones(N, bool))
    S, alive, tau = evolucion(V, G, labels)
    n_surv = int(alive.sum())
    f = n_surv / N
    ord_c = orden_color(V, labels, alive)
    ord_amp = ord_c / (ord_init + 1e-9)
    fq, fg = frac_species(labels, alive)

    row = dict(
        N=N, seed=seed, mode=mode, n_surv=n_surv, f_surv=f,
        tau=tau, orden_color_init=ord_init, orden_color=ord_c, orden_amp=ord_amp,
        frac_quark=fq, frac_gluon=fg,
    )
    for K in K_CANDIDATES:
        cnt = bin_counts(labels, alive, K)
        row[f"L2_K{K}"] = excess_L2(cnt, K)
        row[f"meff_K{K}"] = meff_over_K(cnt, K)
        row[f"null_L2_K{K}"] = null_L2(K, int(cnt.sum()))
        row[f"ratio_L2_K{K}"] = row[f"L2_K{K}"] / null_L2(K, int(cnt.sum())) if cnt.sum() else 0.0
        row[f"grain_K{K}"] = 1.0 / math.sqrt(K)
    return row


def loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y = y.astype(float)
    if np.std(y) < 1e-9:
        return 0.0, float(np.mean(y))
    lx, ly = np.log(x.astype(float)), np.log(np.maximum(y, 1e-12))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, inter = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(np.exp(inter))


def best_K_excess(rows: list[dict], shuffle_rows: list[dict] | None = None) -> dict:
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


def compare_r7a(qcd_df: pd.DataFrame) -> pd.DataFrame | None:
    if not R7A_SWEEP.exists():
        return None
    r7a = pd.read_csv(R7A_SWEEP)
    r7a = r7a[r7a["mode"] == "qcd"].groupby("N").agg(
        f_r7a=("f_surv", "mean"),
        meff12_r7a=("meff_K12", "mean"),
    ).reset_index()
    r7b = qcd_df.groupby("N").agg(
        f_r7b=("f_surv", "mean"),
        meff12_r7b=("meff_K12", "mean"),
        meff8_r7b=("meff_K8", "mean"),
        meff20_r7b=("meff_K20", "mean"),
    ).reset_index()
    return r7a.merge(r7b, on="N")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    if args.single:
        for mode in ("qcd", "shuffle_b", "shuffle_g"):
            r = run_once(1500, 1, mode)
            print(mode, f"={r['f_surv']:.3f}", f"q={r['frac_quark']:.2f}", f"g={r['frac_gluon']:.2f}",
                  f"meff8={r['meff_K8']:.3f}", f"meff20={r['meff_K20']:.3f}")
        print("K_best:", best_K_excess([run_once(1500, 1, "qcd")]))
        return

    NS = [500, 1000, 2000] if args.quick else [250, 500, 1000, 2000, 4000]
    SEEDS = 10 if args.quick else 40
    MODES = ("qcd", "shuffle_b", "shuffle_g")

    t0 = time.monotonic()
    rows = []
    for ni, N in enumerate(NS):
        for seed in range(1, SEEDS + 1):
            for mode in MODES:
                rows.append(run_once(N, seed, mode))
        sub = [r for r in rows if r["N"] == N and r["mode"] == "qcd"]
        print(
            f"  [{ni+1}/{len(NS)}] N={N:5d}  f={np.mean([r['f_surv'] for r in sub]):.3f}  "
            f"q/g={np.mean([r['frac_quark'] for r in sub]):.2f}/{np.mean([r['frac_gluon'] for r in sub]):.2f}  "
            f"meff/K8={np.mean([r['meff_K8'] for r in sub]):.3f}  "
            f"meff/K20={np.mean([r['meff_K20'] for r in sub]):.3f}  "
            f"τ={np.mean([r['tau'] for r in sub]):.0f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    qcd = df[df["mode"] == "qcd"]
    agg = qcd.groupby("N").agg(
        f_surv=("f_surv", "mean"),
        frac_quark=("frac_quark", "mean"),
        frac_gluon=("frac_gluon", "mean"),
        tau=("tau", "mean"),
        **{f"L2_K{K}": (f"L2_K{K}", "mean") for K in K_CANDIDATES},
        **{f"meff_K{K}": (f"meff_K{K}", "mean") for K in K_CANDIDATES},
    ).reset_index()

    fit_rows = []
    for K in K_CANDIDATES:
        s, p = loglog_fit(agg["N"].values, agg[f"L2_K{K}"].values)
        fit_rows.append(dict(K=K, metric="L2", exponente=s, prefactor=p))
        s2, p2 = loglog_fit(agg["N"].values, agg[f"meff_K{K}"].values)
        fit_rows.append(dict(K=K, metric="meff", exponente=s2, prefactor=p2))
    fitdf = pd.DataFrame(fit_rows)
    fitdf.to_csv(FITS_CSV, index=False)

    shuf = df[df["mode"] == "shuffle_b"].to_dict("records")
    bk = best_K_excess(qcd.to_dict("records"), shuf)
    cmp = compare_r7a(qcd)

    # figura: R7b vs R7a
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for K, c in zip(K_CANDIDATES, ("darkorange", "crimson", "steelblue")):
        ax.plot(agg["N"], agg[f"meff_K{K}"], "o-", color=c, label=f"m_eff/K, K={K}")
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("N"); ax.set_ylabel("m_eff/K"); ax.set_title("R7b QCD-real")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(agg["N"], agg["frac_quark"], "o-", label="frac quark")
    ax.plot(agg["N"], agg["frac_gluon"], "s-", label="frac gluón")
    ax.set_xlabel("N"); ax.set_ylabel("fracción supervivientes"); ax.legend(fontsize=8)
    ax.set_title("Especies que persisten"); ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if cmp is not None:
        ax.plot(cmp["N"], cmp["f_r7a"], "o--", color="gray", label="f R7a")
        ax.plot(cmp["N"], cmp["f_r7b"], "o-", color="purple", label="f R7b")
        ax.set_xlabel("N"); ax.set_ylabel("f persiste"); ax.legend(fontsize=8)
        ax.set_title("R7b vs R7a — supervivencia global")
    else:
        ax.text(0.5, 0.5, "sin cg002_r7a_sweep.csv", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    n_ref = 2000 if 2000 in NS else NS[-1]
    for mode, c, lab in (("shuffle_b", "teal", "Δ meff K=8 shuffle-B"),
                         ("shuffle_g", "goldenrod", "Δ meff K=8 shuffle-G")):
        sub = df[(df["N"] == n_ref) & (df["mode"] == mode)]
        qcd_sub = df[(df["N"] == n_ref) & (df["mode"] == "qcd")]
        delta = qcd_sub["meff_K8"].mean() - sub["meff_K8"].mean()
        ax.bar(lab, delta, color=c, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"Δ m_eff/K₈ vs QCD @ N={n_ref}")
    ax.set_title("Controles null")

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    print("\n" + "=" * 60)
    print("CG002-R7b gluón entidad (octeto + quarks, θ_CP=0)")
    print("=" * 60)
    n_show = min(2000, NS[-1])
    row = agg[agg["N"] == n_show].iloc[0]
    print(f"f @ N={n_show}: {row.f_surv:.3f}  (quark {row.frac_quark:.2f} · gluón {row.frac_gluon:.2f})")
    for K in K_CANDIDATES:
        meff = agg[f"meff_K{K}"].mean()
        exp = fitdf[(fitdf.K == K) & (fitdf.metric == "L2")].exponente.values[0]
        print(f"  K={K:2d}: m_eff/K={meff:.4f}  exp_L2={exp:+.3f}")
    print(f"K_eff por exceso: K={bk['k_best']}  ratios: {bk['scores']}")
    print(f"orden_amp medio: {qcd['orden_amp'].mean():.3f}")
    if cmp is not None:
        c = cmp[cmp["N"] == n_show].iloc[0]
        print(f"vs R7a @ N={n_show}: f {c.f_r7a:.3f}→{c.f_r7b:.3f}  meff/K12 {c.meff12_r7a:.3f}→{c.meff12_r7b:.3f}")
    print(f"\nCSV: {SWEEP_CSV}")
    print(f"Fig:  {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()