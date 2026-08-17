#!/usr/bin/env python3
"""
CG002-R7f — γ / W± / Z + Higgs (masa emergente, nunca input)
=============================================================
Extiende R7e: 84 fermiones + 5 bosones (γ, W+, W−, Z, H).

Disciplina CC Msg 33:
  · Semilla uniforme S0=1 para TODOS (sin masa inyectada).
  · φ (VEV Higgs) emerge de sqrt(sat(S_H)) durante la evolución.
  · Masa emergente medida post-hoc: M_i = φ · √sat(S_i) — NO en semilla.

Canales c_ij (estáticos; fermiones = R7e + mediación bosón débil):
  1. Color SU(3) + Carga U(1)
  2. γ/Z mediación; W± antípodos; dobletes W (débil)
  3. φ emerge de S_H — observable; acoplamiento Yukawa dinámico pendiente Mesa

K candidatos: K=3,5,6,18,89.

USO:
  python3 cg002_r7f_higgs.py [--quick|--single]
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
W_COUP, PHI_SCALE = 0.1, 1.0

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_r7f_sweep.csv"
FIG_PATH = OUT_DIR / "cg002_r7f_higgs.png"

N_GEN = 3
N_QUARK = 72
N_LEPTON = 12
N_FERMION = 84
MS_GAMMA, MS_WPLUS, MS_WMINUS, MS_Z, MS_HIGGS = 84, 85, 86, 87, 88
N_BOSON = 5
N_MICRO = N_FERMION + N_BOSON
K_CANDIDATES = (3, 5, 6, 18, 89)

COLORS = np.eye(3, dtype=float)


def sat(S):
    return S / (1.0 + S / S_BAND)


def is_lepton(ms: int) -> bool:
    return ms < N_FERMION and ms >= N_QUARK


def is_fermion(ms: int) -> bool:
    return ms < N_FERMION


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


def charge_value(ms: int) -> float:
    if ms == MS_GAMMA:
        return 0.0
    if ms >= N_FERMION:
        return 0.0
    if ms < N_QUARK:
        is_aq, _, flavor, _, _ = decode_quark(ms)
        q = 2.0 / 3.0 if flavor == 0 else -1.0 / 3.0
        return -q if is_aq else q
    _, spec, conj = decode_lepton(ms)
    if spec == 1:
        return 0.0
    return -1.0 if conj == 0 else 1.0


def weak_t3(ms: int) -> float:
    if ms == MS_WPLUS:
        return 1.0
    if ms == MS_WMINUS:
        return -1.0
    if ms >= N_FERMION:
        return 0.0
    if is_lepton(ms):
        _, spec, conj = decode_lepton(ms)
        t3 = 0.5 if spec == 1 else -0.5
        return -t3 if conj else t3
    is_aq, _, flavor, _, _ = decode_quark(ms)
    t3 = 0.5 if flavor == 0 else -0.5
    return -t3 if is_aq else t3


def is_yukawa_active(ms: int) -> bool:
    """Fermiones acoplan al Higgs; σ uniforme (=1) — jerarquía NO en semilla."""
    return ms < N_FERMION


def fermion_generation(ms: int) -> int:
    if not is_fermion(ms):
        return -1
    if is_lepton(ms):
        return decode_lepton(ms)[0]
    return decode_quark(ms)[1]


def weak_doublet_partner(ms_i: int, ms_j: int) -> bool:
    if not is_fermion(ms_i) or not is_fermion(ms_j):
        return False
    gi, gj = fermion_generation(ms_i), fermion_generation(ms_j)
    if gi != gj or gi < 0:
        return False
    if is_lepton(ms_i) and is_lepton(ms_j):
        si, ci = decode_lepton(ms_i)[1:]
        sj, cj = decode_lepton(ms_j)[1:]
        return (si != sj) and (ci == cj)
    if not is_lepton(ms_i) and not is_lepton(ms_j):
        ai, _, fi, _, _ = decode_quark(ms_i)
        aj, _, fj, _, _ = decode_quark(ms_j)
        return (ai == aj) and (fi != fj)
    return False


def build_static_compat(V, Q, T3, labels) -> np.ndarray:
    N = len(labels)
    labs = labels.astype(int)
    is_f = labs < N_FERMION
    C = np.zeros((N, N))
    idx_q = np.where(is_f & (labs < N_QUARK))[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    C += np.outer(Q, Q)  # canal U(1) R7e — sin T3 global (evita cooperación espuria)

    idx_wp = np.where(labs == MS_WPLUS)[0]
    idx_wm = np.where(labs == MS_WMINUS)[0]
    if len(idx_wp) and len(idx_wm):
        C[np.ix_(idx_wp, idx_wm)] = -1.0
        C[np.ix_(idx_wm, idx_wp)] = -1.0

    idx_g = np.where(labs == MS_GAMMA)[0]
    idx_z = np.where(labs == MS_Z)[0]
    idx_h = np.where(labs == MS_HIGGS)[0]
    idx_ferm = np.where(is_f)[0]
    if len(idx_g):
        gcou = 0.15 * np.abs(Q[idx_ferm])
        C[idx_g[:, None], idx_ferm] = gcou
        C[idx_ferm, idx_g[:, None]] = gcou
    if len(idx_z):
        zcou = 0.15 * np.abs(T3[idx_ferm])
        C[idx_z[:, None], idx_ferm] = zcou
        C[idx_ferm, idx_z[:, None]] = zcou

    np.fill_diagonal(C, 0.0)
    return C, idx_h, idx_ferm


def build_compat(static_C) -> np.ndarray:
    """Dinámica fermiónica = R7e; bosones ya en static_C como mediación débil."""
    return static_C


def evolucion(V, Q, T3, labels):
    N = len(labels)
    labs = labels.astype(int)
    static_C, _, _ = build_static_compat(V, Q, T3, labels)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0
    phi_trace = []
    for _ in range(PASOS):
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        higgs_alive = alive & (labs == MS_HIGGS)
        phi = float(np.sqrt(sat(S[higgs_alive]).mean())) if higgs_alive.any() else 0.0
        phi_trace.append(phi)
        C = build_compat(static_C)
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
    phi_init = phi_trace[0] if phi_trace else 0.0
    phi_final = phi_trace[-1] if phi_trace else 0.0
    phi_peak = max(phi_trace) if phi_trace else 0.0
    return S, alive, tau, phi_init, phi_final, phi_peak, phi_trace


def semilla_simetrica(N: int, rng: np.random.Generator):
    counts = np.full(N_MICRO, N // N_MICRO)
    for _ in range(N % N_MICRO):
        counts[rng.integers(0, N_MICRO)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.zeros((N, 3))
    Q = np.zeros(N)
    T3 = np.zeros(N)
    pos = 0
    for ms in range(N_MICRO):
        n = counts[ms]
        labels[pos : pos + n] = ms
        if ms < N_QUARK:
            is_aq, _, _, ci, _ = decode_quark(ms)
            V[pos : pos + n] = color_vector(is_aq, ci)
        Q[pos : pos + n] = charge_value(ms)
        T3[pos : pos + n] = weak_t3(ms)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm], Q[perm], T3[perm]


def semilla_shuffle_b(labels, V, Q, T3, rng):
    perm_color = rng.permutation(3)
    lab2, V2 = labels.copy(), V.copy()
    for i in range(len(labels)):
        ms = int(labels[i])
        if not is_fermion(ms) or is_lepton(ms):
            continue
        is_aq, gen, flavor, ci, sp = decode_quark(ms)
        ci2 = int(perm_color[ci])
        lab2[i] = encode_quark(is_aq, gen, flavor, ci2, sp)
        V2[i] = color_vector(is_aq, ci2)
    return lab2, V2, Q, T3


def semilla_shuffle_e(labels, Q, T3, rng):
    Q2 = Q.copy()
    perm = rng.permutation(len(Q))
    Q2[:] = Q[perm]
    return labels.copy(), Q2, T3


def semilla_shuffle_w(labels, Q, T3, rng):
    """Rompe correlación T₃ física."""
    perm = rng.permutation(len(T3))
    T2 = T3.copy()
    T2[:] = T3[perm]
    return labels.copy(), Q, T2


def semilla_shuffle_h(labels, V, Q, T3, rng):
    """Rompe correlación Higgs–sector: permuta etiquetas de bosones entre sí."""
    boson_slots = np.where(labels >= N_FERMION)[0]
    if len(boson_slots) < 2:
        return labels.copy(), V, Q, T3
    lab2 = labels.copy()
    sub = lab2[boson_slots].copy()
    rng.shuffle(sub)
    lab2[boson_slots] = sub
    return lab2, V, Q, T3


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


def orden_t3(T3, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0
    return float(abs(T3[idx].mean()))


def frac_species(labels, alive):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0
    surv = labels[idx]
    nf = (surv < N_FERMION).sum()
    nq = ((surv < N_QUARK)).sum()
    nb = (surv >= N_FERMION).sum()
    ns = len(surv)
    return float(nq / ns), float((nf - nq) / ns), float(nb / ns)


def frac_generations(labels, alive):
    idx = np.where(alive)[0]
    ferm = labels[idx][labels[idx] < N_FERMION]
    if len(ferm) == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    gens = np.array([fermion_generation(int(ms)) for ms in ferm])
    c = np.bincount(gens, minlength=N_GEN).astype(float)
    c /= c.sum()
    return tuple(c)


def emergent_mass(S, labels, alive, phi: float):
    """M_i = sqrt(sat(S_i)) · φ — medida post-hoc, sin etiqueta de masa."""
    idx = np.where(alive & (labels < N_FERMION))[0]
    if len(idx) == 0 or phi <= 0:
        return np.zeros(3), 0.0, 0.0
    masses = np.sqrt(sat(S[idx])) * phi * PHI_SCALE
    gens = np.array([fermion_generation(int(labels[i])) for i in idx])
    by_gen = [float(masses[gens == g].mean()) if (gens == g).any() else 0.0 for g in range(N_GEN)]
    ratio = by_gen[2] / by_gen[0] if by_gen[0] > 1e-9 else 0.0
    spread = float(masses.std() / (masses.mean() + 1e-9))
    return by_gen, ratio, spread


def bin_counts(labels, alive, K):
    idx = np.where(alive)[0]
    if len(idx) == 0:
        return np.zeros(K, dtype=int)
    surv = labels[idx]
    if K == 89:
        return np.bincount(surv, minlength=N_MICRO)
    if K == 5:
        b = surv[surv >= N_FERMION] - N_FERMION
        return np.bincount(b, minlength=N_BOSON)
    if K == 18:
        cnt = np.zeros(18, dtype=int)
        for ms in surv:
            if ms >= N_FERMION:
                continue
            is_aq, gen, _, color, spin = decode_quark(int(ms))
            if not is_aq and ms < N_QUARK:
                cnt[gen * 6 + color * 2 + spin] += 1
        return cnt
    if K == 6:
        cnt = np.zeros(6, dtype=int)
        for ms in surv:
            if ms < N_QUARK:
                _, _, _, color, spin = decode_quark(int(ms))
                cnt[color * 2 + spin] += 1
        return cnt
    raise ValueError(K)


def bin_charge_counts(Q, alive):
    idx = np.where(alive)[0]
    cnt = np.zeros(3, dtype=int)
    for q in Q[idx]:
        if q < -0.1:
            cnt[0] += 1
        elif q > 0.1:
            cnt[2] += 1
        else:
            cnt[1] += 1
    return cnt


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
    labels, V, Q, T3 = semilla_simetrica(N, rng)
    if mode == "shuffle_b":
        labels, V, Q, T3 = semilla_shuffle_b(labels, V, Q, T3, rng)
    elif mode == "shuffle_e":
        labels, Q, T3 = semilla_shuffle_e(labels, Q, T3, rng)
    elif mode == "shuffle_w":
        labels, Q, T3 = semilla_shuffle_w(labels, Q, T3, rng)
    elif mode == "shuffle_h":
        labels, V, Q, T3 = semilla_shuffle_h(labels, V, Q, T3, rng)

    alive_all = np.ones(N, bool)
    ord_c0 = orden_color(V, alive_all)
    ord_q0 = orden_charge(Q, alive_all)
    ord_t0 = orden_t3(T3, alive_all)

    S, alive, tau, phi0, phi_f, phi_pk, _ = evolucion(V, Q, T3, labels)
    n_surv = int(alive.sum())
    ord_c = orden_color(V, alive)
    ord_q = orden_charge(Q, alive)
    ord_t = orden_t3(T3, alive)
    fq, fl, fb = frac_species(labels, alive)
    g0, g1, g2 = frac_generations(labels, alive)
    m_by_gen, m_ratio, m_spread = emergent_mass(S, labels, alive, phi_pk)

    row = dict(
        N=N,
        seed=seed,
        mode=mode,
        n_surv=n_surv,
        f_surv=n_surv / N,
        tau=tau,
        phi_init=phi0,
        phi_final=phi_f,
        phi_peak=phi_pk,
        phi_amp=phi_pk / (phi0 + 1e-9),
        frac_higgs_surv=float((alive & (labels == MS_HIGGS)).mean()),
        orden_color_init=ord_c0,
        orden_color=ord_c,
        orden_charge_init=ord_q0,
        orden_charge=ord_q,
        orden_t3_init=ord_t0,
        orden_t3=ord_t,
        frac_quark=fq,
        frac_lepton=fl,
        frac_boson=fb,
        frac_gen0=g0,
        frac_gen1=g1,
        frac_gen2=g2,
        mean_Q_surv=float(Q[alive].mean()) if n_surv else 0.0,
        mass_gen0=m_by_gen[0],
        mass_gen1=m_by_gen[1],
        mass_gen2=m_by_gen[2],
        mass_ratio_g2_g0=m_ratio,
        mass_spread=m_spread,
        L2_gen=excess_L2(np.array([g0, g1, g2]) * n_surv, 3) if n_surv else 0.0,
    )
    cnt_q3 = bin_charge_counts(Q, alive)
    row["L2_K3"] = excess_L2(cnt_q3, 3)
    row["null_L2_K3"] = null_L2(3, int(cnt_q3.sum()))
    row["ratio_L2_K3"] = row["L2_K3"] / row["null_L2_K3"] if cnt_q3.sum() else 0.0

    for K in K_CANDIDATES:
        if K == 3:
            continue
        cnt = bin_counts(labels, alive, K)
        row[f"L2_K{K}"] = excess_L2(cnt, K)
        row[f"null_L2_K{K}"] = null_L2(K, int(cnt.sum()))
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
    MODES = ("qcd", "shuffle_b", "shuffle_e", "shuffle_w", "shuffle_h")
    t0 = time.monotonic()
    rows = []
    for ni, N in enumerate(NS):
        for seed in range(1, SEEDS + 1):
            for mode in MODES:
                rows.append(run_once(N, seed, mode))
        sub = [r for r in rows if r["N"] == N and r["mode"] == "qcd"]
        print(
            f"  [{ni+1}/{len(NS)}] N={N:5d}  f={np.mean([r['f_surv'] for r in sub]):.3f}  "
            f"φ_f={np.mean([r['phi_final'] for r in sub]):.3f}  "
            f"ordQ={np.mean([r['orden_charge'] for r in sub]):.3f}  "
            f"g0/1/2={np.mean([r['frac_gen0'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_gen1'] for r in sub]):.2f}/"
            f"{np.mean([r['frac_gen2'] for r in sub]):.2f}  "
            f"m₂/m₀={np.mean([r['mass_ratio_g2_g0'] for r in sub]):.2f}  "
            f"K3×={np.mean([r['ratio_L2_K3'] for r in sub]):.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    qcd = df[df["mode"] == "qcd"]
    sh_e = df[df["mode"] == "shuffle_e"]
    sh_h = df[df["mode"] == "shuffle_h"]
    bk = best_K_excess(qcd.to_dict("records"), sh_e.to_dict("records"))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    agg = qcd.groupby("N").mean(numeric_only=True)
    ax[0].plot(agg.index, agg.orden_charge, "o-", label="orden_Q")
    ax[0].plot(agg.index, agg.orden_color, "s--", label="orden_color")
    ax[0].plot(agg.index, agg.phi_peak, "^:", label="φ peak")
    ax[0].set_xlabel("N")
    ax[0].legend(fontsize=7)
    ax[0].set_title("R7f polarización + VEV")
    ax[1].plot(agg.index, agg.frac_gen0, "o-", label="gen0")
    ax[1].plot(agg.index, agg.frac_gen1, "s-", label="gen1")
    ax[1].plot(agg.index, agg.frac_gen2, "^-", label="gen2")
    ax[1].axhline(1 / 3, color="gray", ls="--")
    ax[1].set_xlabel("N")
    ax[1].legend(fontsize=7)
    ax[1].set_title("Generaciones supervivientes")
    ax[2].plot(agg.index, agg.mass_ratio_g2_g0, "o-", color="darkgreen", label="m₂/m₀ emergente")
    ax[2].plot(agg.index, agg.ratio_L2_K3, "s--", color="darkorange", label="K3 carga")
    ax[2].axhline(1, color="gray", ls="--")
    ax[2].set_xlabel("N")
    ax[2].legend(fontsize=7)
    ax[2].set_title(f"K_eff={bk['k_best']}")
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    def delta_l2(mode, K):
        return float(qcd[f"L2_K{K}"].mean() - df[df["mode"] == mode][f"L2_K{K}"].mean())

    print("\n" + "=" * 60)
    print("CG002-R7f γ/W/Z + Higgs (masa emergente)")
    print(f"K_eff: K={bk['k_best']}  {bk['scores']}")
    print(f"orden_charge @ N=4000: {qcd[qcd.N==4000].orden_charge.mean():.3f}")
    print(f"phi_final @ N=4000: {qcd[qcd.N==4000].phi_final.mean():.3f}")
    print(f"mass_ratio g2/g0: {qcd['mass_ratio_g2_g0'].mean():.3f}")
    print(f"L2_gen (desigualdad): {qcd['L2_gen'].mean():.4f}")
    print(f"ΔL2 shuffle-H K5: {delta_l2('shuffle_h', 5):.4f}")
    print(f"ΔL2 shuffle-E K3: {delta_l2('shuffle_e', 3):.4f}")
    print(f"CSV: {SWEEP_CSV}  Fig: {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()