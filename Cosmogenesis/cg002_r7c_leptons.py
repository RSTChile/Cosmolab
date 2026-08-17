#!/usr/bin/env python3
"""
CG002-R7c — leptones (singlete de color) + quarks R7a
=====================================================
Mismo motor CG002. Quarks/antiquarks con c_ij color SU(3); leptones NO sienten
fuerte fuerte: c(quark,lepton)=0. Leptón-leptón: v_i·v_j con v=±1 (partícula/antipartícula).

Semilla simétrica 24 microestados (12 quark + 12 leptón). θ_CP=0.
K candidatos: K=6 (quark), K=12 (leptón), K=24 (completo).

USO:
  python3 cg002_r7c_leptons.py [--quick|--single]
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

ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0
PASOS, ALPHA = 240, 1.0

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_r7c_sweep.csv"
FITS_CSV = OUT_DIR / "cg002_r7c_fits.csv"
FIG_PATH = OUT_DIR / "cg002_r7c_leptons.png"
R7A_SWEEP = OUT_DIR / "cg002_r7a_sweep.csv"

N_QUARK = 12
N_LEPTON = 12
N_MICRO = N_QUARK + N_LEPTON
K_CANDIDATES = (6, 12, 24)

COLORS = np.eye(3, dtype=float)


def sat(S):
    return S / (1.0 + S / S_BAND)


def is_lepton(label: int) -> bool:
    return label >= N_QUARK


def decode_quark(ms: int) -> tuple[bool, int, int]:
    is_aq = ms >= 6
    base = ms - 6 if is_aq else ms
    return is_aq, base // 2, base % 2


def decode_lepton(ms: int) -> tuple[int, int]:
    """3 gen × 2 especies × 2 conjugación; v=±1 escalar."""
    base = ms - N_QUARK
    sign = 1 if base % 2 == 0 else -1
    return base // 2, sign


def color_vector(is_antiquark: bool, color_idx: int) -> np.ndarray:
    v = COLORS[color_idx]
    return -v if is_antiquark else v


def lepton_scalar(gen: int, sign: int) -> float:
    return float(sign)


def build_compat(V: np.ndarray, L: np.ndarray, labels: np.ndarray) -> np.ndarray:
    N = len(labels)
    is_l = labels >= N_QUARK
    C = np.zeros((N, N))
    idx_q = np.where(~is_l)[0]
    idx_l = np.where(is_l)[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    if len(idx_l):
        Ll = L[idx_l]
        C[np.ix_(idx_l, idx_l)] = np.outer(Ll, Ll)
    np.fill_diagonal(C, 0.0)
    return C


def evolucion(V: np.ndarray, L: np.ndarray, labels: np.ndarray):
    C = build_compat(V, L, labels)
    N = len(labels)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0
    for _ in range(PASOS):
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


def semilla_simetrica(N: int, rng: np.random.Generator):
    counts = np.full(N_MICRO, N // N_MICRO)
    for _ in range(N % N_MICRO):
        counts[rng.integers(0, N_MICRO)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.zeros((N, 3))
    L = np.zeros(N)
    pos = 0
    for ms in range(N_MICRO):
        n = counts[ms]
        if ms < N_QUARK:
            is_aq, ci, sp = decode_quark(ms)
            cv = color_vector(is_aq, ci)
            labels[pos : pos + n] = ms
            V[pos : pos + n] = cv
        else:
            gen, sign = decode_lepton(ms)
            labels[pos : pos + n] = ms
            L[pos : pos + n] = lepton_scalar(gen, sign)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm], L[perm]


def semilla_shuffle_b(labels, V, L, rng):
    perm_color = rng.permutation(3)
    lab2, V2 = labels.copy(), V.copy()
    for i in range(len(labels)):
        if is_lepton(int(labels[i])):
            continue
        is_aq, ci, sp = decode_quark(int(labels[i]))
        ci2 = int(perm_color[ci])
        lab2[i] = (3 if is_aq else 0) + ci2 * 2 + sp
        V2[i] = color_vector(is_aq, ci2)
    return lab2, V2, L


def semilla_shuffle_l(labels, L, rng):
    perm = rng.permutation(N_LEPTON)
    lab2, L2 = labels.copy(), L.copy()
    for i in range(len(labels)):
        if not is_lepton(int(labels[i])):
            continue
        new_slot = int(perm[int(labels[i]) - N_QUARK])
        lab2[i] = N_QUARK + new_slot
        g, s = decode_lepton(lab2[i])
        L2[i] = lepton_scalar(g, s)
    return lab2, L2


def bin_counts(labels, alive, K, sector="auto"):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return np.zeros(K, dtype=int)
    surv = labels[idx]
    if K == 24:
        return np.bincount(surv, minlength=N_MICRO)
    if K == 12:
        if sector == "quark":
            q = surv[surv < N_QUARK]
            return np.bincount(q, minlength=N_QUARK)
        q = surv[surv >= N_QUARK] - N_QUARK
        return np.bincount(q, minlength=N_LEPTON)
    if K == 6:
        q = surv[surv < 6]
        return np.bincount(q, minlength=6)
    raise ValueError(K)


def excess_L2(counts, K):
    N = counts.sum()
    if N == 0:
        return 0.0
    f = counts / N
    return float(np.sqrt(((f - 1 / K) ** 2).sum()))


def null_L2(K, N):
    return math.sqrt((K - 1) / (K * max(N, 1)))


def orden_color(V, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    return float(np.linalg.norm(V[idx].mean(0)))


def frac_species(labels, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    surv = labels[idx]
    return float((surv < N_QUARK).mean()), float((surv >= N_QUARK).mean())


def best_K_excess(rows, shuffle_rows=None):
    qcd = [r for r in rows if r.get("mode") == "qcd"]
    scores = {K: float(np.mean([r[f"ratio_L2_K{K}"] for r in qcd])) for K in K_CANDIDATES}
    shuffle_pos = set(K_CANDIDATES)
    if shuffle_rows:
        for K in K_CANDIDATES:
            if np.mean([r[f"L2_K{K}"] for r in qcd]) - np.mean([r[f"L2_K{K}"] for r in shuffle_rows]) <= 0:
                shuffle_pos.discard(K)
    candidates = [K for K in K_CANDIDATES if K in shuffle_pos] or list(K_CANDIDATES)
    return dict(k_best=max(candidates, key=lambda k: scores[k]), scores=scores)


def run_once(N, seed, mode="qcd"):
    rng = np.random.default_rng(seed)
    labels, V, L = semilla_simetrica(N, rng)
    if mode == "shuffle_b":
        labels, V, L = semilla_shuffle_b(labels, V, L, rng)
    elif mode == "shuffle_l":
        labels, L = semilla_shuffle_l(labels, L, rng)
    ord_init = orden_color(V, np.ones(N, bool))
    S, alive, tau = evolucion(V, L, labels)
    n_surv = int(alive.sum())
    ord_c = orden_color(V, alive)
    fq, fl = frac_species(labels, alive)
    row = dict(
        N=N, seed=seed, mode=mode, n_surv=n_surv, f_surv=n_surv / N, tau=tau,
        orden_color_init=ord_init, orden_color=ord_c, orden_amp=ord_c / (ord_init + 1e-9),
        frac_quark=fq, frac_lepton=fl,
    )
    for K in K_CANDIDATES:
        if K == 12:
            cnt_q = bin_counts(labels, alive, 12, "quark")
            cnt_l = bin_counts(labels, alive, 12, "lepton")
            row["L2_K12_q"] = excess_L2(cnt_q, N_QUARK)
            row["ratio_L2_K12_q"] = row["L2_K12_q"] / null_L2(N_QUARK, int(cnt_q.sum())) if cnt_q.sum() else 0.0
            row["L2_K12_l"] = excess_L2(cnt_l, N_LEPTON)
            row["ratio_L2_K12_l"] = row["L2_K12_l"] / null_L2(N_LEPTON, int(cnt_l.sum())) if cnt_l.sum() else 0.0
            cnt = cnt_l
            k_eff = N_LEPTON
        else:
            cnt = bin_counts(labels, alive, K)
            k_eff = K
        row[f"L2_K{K}"] = excess_L2(cnt, k_eff)
        row[f"null_L2_K{K}"] = null_L2(k_eff, int(cnt.sum()))
        row[f"ratio_L2_K{K}"] = row[f"L2_K{K}"] / row[f"null_L2_K{K}"] if cnt.sum() else 0.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    if args.single:
        r = run_once(2000, 1, "qcd")
        print(r)
        print("K_best:", best_K_excess([r]))
        return

    NS = [500, 1000, 2000] if args.quick else [250, 500, 1000, 2000, 4000]
    SEEDS = 10 if args.quick else 40
    MODES = ("qcd", "shuffle_b", "shuffle_l")
    t0 = time.monotonic()
    rows = []
    for ni, N in enumerate(NS):
        for seed in range(1, SEEDS + 1):
            for mode in MODES:
                rows.append(run_once(N, seed, mode))
        sub = [r for r in rows if r["N"] == N and r["mode"] == "qcd"]
        print(
            f"  [{ni+1}/{len(NS)}] N={N:5d}  f={np.mean([r['f_surv'] for r in sub]):.3f}  "
            f"q/L={np.mean([r['frac_quark'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_lepton'] for r in sub]):.2f}  "
            f"ord_amp={np.mean([r['orden_amp'] for r in sub]):.2f}  "
            f"K6×={np.mean([r['ratio_L2_K6'] for r in sub]):.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    qcd = df[df["mode"] == "qcd"]
    bk = best_K_excess(qcd.to_dict("records"), df[df["mode"] == "shuffle_b"].to_dict("records"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    agg = qcd.groupby("N").mean(numeric_only=True)
    ax[0].plot(agg.index, agg.f_surv, "o-", label="f total")
    ax[0].plot(agg.index, agg.frac_quark, "s--", label="frac quark")
    ax[0].plot(agg.index, agg.frac_lepton, "^--", label="frac leptón")
    ax[0].set_xlabel("N"); ax[0].legend(fontsize=8); ax[0].set_title("R7c supervivencia por especie")
    ax[1].plot(agg.index, agg.ratio_L2_K6, "o-", color="crimson", label="K6 quark")
    ax[1].plot(agg.index, agg.get("ratio_L2_K12_q", agg.ratio_L2_K12), "s-", label="K12 quark sector")
    ax[1].axhline(1, color="gray", ls="--")
    ax[1].set_xlabel("N"); ax[1].legend(fontsize=8); ax[1].set_title(f"K_eff={bk['k_best']}")
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    print("\n" + "=" * 60)
    print("CG002-R7c leptones (singlete) + quarks")
    print(f"K_eff por exceso: K={bk['k_best']}  {bk['scores']}")
    print(f"orden_amp medio: {qcd['orden_amp'].mean():.3f}")
    print(f"frac leptón superv.: {qcd['frac_lepton'].mean():.3f}")
    print(f"CSV: {SWEEP_CSV}  Fig: {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()