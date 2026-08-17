#!/usr/bin/env python3
"""
CG002-R7e — generaciones ×3 (multiplicidad de sabor)
====================================================
Extiende R7d: 3 generaciones de quarks con isospín u/d; leptones 3 gen (R7c).

Microestados: 72 quarks (3 gen × 2 sabor × 3 color × 2 spin × q/q̄)
              + 12 leptones = 84 total.

Canales (igual R7d): c_ij = c_color_ij + Q_i Q_j
  u: ±2/3, d: ∓1/3; leptón cargado ±1; ν: 0.
Generación y sabor: etiqueta de multiplicidad (no entran en c_ij salvo vía Q).

K candidatos: K=3 (carga), K=6 (color×spin), K=18 (gen×color×spin),
              K=72 (quark completo), K=84 (espacio total).

Controles: shuffle-B, shuffle-G (generación), shuffle-F (sabor u↔d),
           shuffle-L, shuffle-E.

USO:
  python3 cg002_r7e_generations.py [--quick|--single]
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
SWEEP_CSV = OUT_DIR / "cg002_r7e_sweep.csv"
FIG_PATH = OUT_DIR / "cg002_r7e_generations.png"

N_GEN = 3
N_QUARK = 72
N_LEPTON = 12
N_MICRO = N_QUARK + N_LEPTON
K_CANDIDATES = (3, 6, 18, 72, 84)

COLORS = np.eye(3, dtype=float)


def sat(S):
    return S / (1.0 + S / S_BAND)


def is_lepton(label: int) -> bool:
    return label >= N_QUARK


def encode_quark(is_aq: bool, gen: int, flavor: int, color: int, spin: int) -> int:
    base = gen * 12 + flavor * 6 + color * 2 + spin
    return 36 + base if is_aq else base


def decode_quark(ms: int) -> tuple[bool, int, int, int, int]:
    is_aq = ms >= 36
    base = ms - 36 if is_aq else ms
    gen = base // 12
    rem = base % 12
    flavor = rem // 6
    rem2 = rem % 6
    color = rem2 // 2
    spin = rem2 % 2
    return is_aq, gen, flavor, color, spin


def decode_lepton(ms: int) -> tuple[int, int, int]:
    base = ms - N_QUARK
    conj = base % 2
    spec = (base // 2) % 2
    gen = base // 4
    return gen, spec, conj


def color_vector(is_antiquark: bool, color_idx: int) -> np.ndarray:
    v = COLORS[color_idx]
    return -v if is_antiquark else v


def charge_quark(is_aq: bool, flavor: int) -> float:
    q = 2.0 / 3.0 if flavor == 0 else -1.0 / 3.0
    return -q if is_aq else q


def charge_lepton(spec: int, conj: int) -> float:
    if spec == 1:
        return 0.0
    return -1.0 if conj == 0 else 1.0


def microstate_charge(ms: int) -> float:
    if ms < N_QUARK:
        is_aq, _, flavor, _, _ = decode_quark(ms)
        return charge_quark(is_aq, flavor)
    _, spec, conj = decode_lepton(ms)
    return charge_lepton(spec, conj)


def build_compat(V: np.ndarray, Q: np.ndarray, labels: np.ndarray) -> np.ndarray:
    N = len(labels)
    is_l = labels >= N_QUARK
    C = np.zeros((N, N))
    idx_q = np.where(~is_l)[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    C += np.outer(Q, Q)
    np.fill_diagonal(C, 0.0)
    return C


def evolucion(V: np.ndarray, Q: np.ndarray, labels: np.ndarray):
    C = build_compat(V, Q, labels)
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
    Q = np.zeros(N)
    pos = 0
    for ms in range(N_MICRO):
        n = counts[ms]
        labels[pos : pos + n] = ms
        if ms < N_QUARK:
            is_aq, _, _, ci, _ = decode_quark(ms)
            V[pos : pos + n] = color_vector(is_aq, ci)
        Q[pos : pos + n] = microstate_charge(ms)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm], Q[perm]


def semilla_shuffle_b(labels, V, Q, rng):
    perm_color = rng.permutation(3)
    lab2, V2 = labels.copy(), V.copy()
    for i in range(len(labels)):
        if is_lepton(int(labels[i])):
            continue
        is_aq, gen, flavor, ci, sp = decode_quark(int(labels[i]))
        ci2 = int(perm_color[ci])
        lab2[i] = encode_quark(is_aq, gen, flavor, ci2, sp)
        V2[i] = color_vector(is_aq, ci2)
    return lab2, V2, Q


def semilla_shuffle_g(labels, V, Q, rng):
    perm_gen = rng.permutation(N_GEN)
    lab2, V2, Q2 = labels.copy(), V.copy(), Q.copy()
    for i in range(len(labels)):
        if is_lepton(int(labels[i])):
            continue
        is_aq, gen, flavor, ci, sp = decode_quark(int(labels[i]))
        gen2 = int(perm_gen[gen])
        lab2[i] = encode_quark(is_aq, gen2, flavor, ci, sp)
        Q2[i] = microstate_charge(lab2[i])
    return lab2, V2, Q2


def semilla_shuffle_f(labels, Q, rng):
    """Permuta sabor u↔d rompiendo correlación isospín–carga."""
    perm_flavor = rng.permutation(2)
    lab2, Q2 = labels.copy(), Q.copy()
    for i in range(len(labels)):
        if is_lepton(int(labels[i])):
            continue
        is_aq, gen, flavor, ci, sp = decode_quark(int(labels[i]))
        flavor2 = int(perm_flavor[flavor])
        lab2[i] = encode_quark(is_aq, gen, flavor2, ci, sp)
        Q2[i] = microstate_charge(lab2[i])
    return lab2, Q2


def semilla_shuffle_l(labels, Q, rng):
    perm = rng.permutation(N_LEPTON)
    lab2, Q2 = labels.copy(), Q.copy()
    for i in range(len(labels)):
        if not is_lepton(int(labels[i])):
            continue
        new_slot = int(perm[int(labels[i]) - N_QUARK])
        lab2[i] = N_QUARK + new_slot
        Q2[i] = microstate_charge(lab2[i])
    return lab2, Q2


def semilla_shuffle_e(labels, Q, rng):
    Q2 = Q.copy()
    perm = rng.permutation(len(Q))
    Q2[:] = Q[perm]
    return labels.copy(), Q2


def quark_bin_index(ms: int, K: int) -> int:
    is_aq, gen, flavor, color, spin = decode_quark(ms)
    if K == 72:
        return ms
    if K == 18:
        return gen * 6 + color * 2 + spin
    if K == 6:
        return color * 2 + spin
    raise ValueError(K)


def bin_counts(labels, alive, K):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return np.zeros(K, dtype=int)
    surv = labels[idx]
    if K == 84:
        return np.bincount(surv, minlength=N_MICRO)
    if K == 72:
        q = surv[surv < N_QUARK]
        return np.bincount(q, minlength=N_QUARK)
    if K in (6, 18):
        cnt = np.zeros(K, dtype=int)
        for ms in surv:
            if ms >= N_QUARK:
                continue
            cnt[quark_bin_index(int(ms), K)] += 1
        return cnt
    raise ValueError(K)


def bin_charge_counts(Q, alive) -> np.ndarray:
    idx = np.where(alive)[0]
    cnt = np.zeros(3, dtype=int)
    if len(idx) == 0:
        return cnt
    for q in Q[idx]:
        if q < -0.1:
            cnt[0] += 1
        elif q > 0.1:
            cnt[2] += 1
        else:
            cnt[1] += 1
    return cnt


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


def orden_charge(Q, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    return float(abs(Q[idx].mean()))


def frac_species(labels, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    surv = labels[idx]
    return float((surv < N_QUARK).mean()), float((surv >= N_QUARK).mean())


def frac_generations(labels, alive):
    idx = np.where(alive)[0]
    q = labels[idx][labels[idx] < N_QUARK]
    if len(q) == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    gens = np.array([decode_quark(int(ms))[1] for ms in q])
    c = np.bincount(gens, minlength=N_GEN).astype(float)
    c /= c.sum()
    return tuple(c)


def frac_flavors(labels, alive):
    idx = np.where(alive)[0]
    q = labels[idx][labels[idx] < N_QUARK]
    if len(q) == 0:
        return 0.5, 0.5
    fl = np.array([decode_quark(int(ms))[2] for ms in q])
    u = float((fl == 0).mean())
    return u, 1.0 - u


def frac_charge_neutral(Q, alive) -> float:
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    return float((np.abs(Q[idx]) < 0.1).mean())


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
    labels, V, Q = semilla_simetrica(N, rng)
    if mode == "shuffle_b":
        labels, V, Q = semilla_shuffle_b(labels, V, Q, rng)
    elif mode == "shuffle_g":
        labels, V, Q = semilla_shuffle_g(labels, V, Q, rng)
    elif mode == "shuffle_f":
        labels, Q = semilla_shuffle_f(labels, Q, rng)
    elif mode == "shuffle_l":
        labels, Q = semilla_shuffle_l(labels, Q, rng)
    elif mode == "shuffle_e":
        labels, Q = semilla_shuffle_e(labels, Q, rng)

    alive_all = np.ones(N, bool)
    ord_c_init = orden_color(V, alive_all)
    ord_q_init = orden_charge(Q, alive_all)
    S, alive, tau = evolucion(V, Q, labels)
    n_surv = int(alive.sum())
    ord_c = orden_color(V, alive)
    ord_q = orden_charge(Q, alive)
    fq, fl = frac_species(labels, alive)
    g0, g1, g2 = frac_generations(labels, alive)
    fu, fd = frac_flavors(labels, alive)
    row = dict(
        N=N,
        seed=seed,
        mode=mode,
        n_surv=n_surv,
        f_surv=n_surv / N,
        tau=tau,
        orden_color_init=ord_c_init,
        orden_color=ord_c,
        orden_charge_init=ord_q_init,
        orden_charge=ord_q,
        frac_quark=fq,
        frac_lepton=fl,
        frac_gen0=g0,
        frac_gen1=g1,
        frac_gen2=g2,
        frac_u=fu,
        frac_d=fd,
        frac_Q0=frac_charge_neutral(Q, alive),
        mean_Q_surv=float(Q[alive].mean()) if n_surv else 0.0,
    )
    cnt_q3 = bin_charge_counts(Q, alive)
    row["L2_K3"] = excess_L2(cnt_q3, 3)
    row["null_L2_K3"] = null_L2(3, int(cnt_q3.sum()))
    row["ratio_L2_K3"] = row["L2_K3"] / row["null_L2_K3"] if cnt_q3.sum() else 0.0

    for K in (6, 18, 72, 84):
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
    MODES = ("qcd", "shuffle_b", "shuffle_g", "shuffle_f", "shuffle_l", "shuffle_e")
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
            f"g0/1/2={np.mean([r['frac_gen0'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_gen1'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_gen2'] for r in sub]):.2f}  "
            f"u/d={np.mean([r['frac_u'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_d'] for r in sub]):.2f}  "
            f"K3×={np.mean([r['ratio_L2_K3'] for r in sub]):.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    qcd = df[df["mode"] == "qcd"]
    sh_b = df[df["mode"] == "shuffle_b"]
    bk = best_K_excess(qcd.to_dict("records"), sh_b.to_dict("records"))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    agg = qcd.groupby("N").mean(numeric_only=True)
    ax[0].plot(agg.index, agg.f_surv, "o-", label="f total")
    ax[0].plot(agg.index, agg.frac_quark, "s--", label="frac quark")
    ax[0].plot(agg.index, agg.frac_gen0, ":", label="gen0")
    ax[0].plot(agg.index, agg.frac_gen1, ":", label="gen1")
    ax[0].plot(agg.index, agg.frac_gen2, ":", label="gen2")
    ax[0].set_xlabel("N")
    ax[0].legend(fontsize=6)
    ax[0].set_title("R7e supervivencia")
    ax[1].plot(agg.index, agg.ratio_L2_K3, "o-", color="darkorange", label="K3 carga")
    ax[1].plot(agg.index, agg.ratio_L2_K18, "s-", color="crimson", label="K18 gen×color")
    ax[1].axhline(1, color="gray", ls="--")
    ax[1].set_xlabel("N")
    ax[1].legend(fontsize=7)
    ax[1].set_title("Exceso L2/nulo")
    ax[2].plot(agg.index, agg.frac_u, "o-", label="frac u")
    ax[2].plot(agg.index, agg.frac_d, "s--", label="frac d")
    ax[2].axhline(0.5, color="gray", ls="--")
    ax[2].set_xlabel("N")
    ax[2].legend(fontsize=7)
    ax[2].set_title(f"K_eff={bk['k_best']}")
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    def delta_l2(mode, K):
        return float(qcd[f"L2_K{K}"].mean() - df[df["mode"] == mode][f"L2_K{K}"].mean())

    print("\n" + "=" * 60)
    print("CG002-R7e generaciones ×3 + U(1) + leptones")
    print(f"K_eff por exceso (shuffle-B+): K={bk['k_best']}  {bk['scores']}")
    print(f"orden_color medio: {qcd['orden_color'].mean():.3f}")
    print(f"orden_charge medio: {qcd['orden_charge'].mean():.3f}")
    print(f"gen0/1/2: {qcd['frac_gen0'].mean():.3f}/{qcd['frac_gen1'].mean():.3f}/{qcd['frac_gen2'].mean():.3f}")
    print(f"u/d: {qcd['frac_u'].mean():.3f}/{qcd['frac_d'].mean():.3f}")
    print(f"ΔL2 shuffle-G K18: {delta_l2('shuffle_g', 18):.4f}")
    print(f"ΔL2 shuffle-F K3: {delta_l2('shuffle_f', 3):.4f}")
    print(f"CSV: {SWEEP_CSV}  Fig: {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()