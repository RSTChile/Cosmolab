#!/usr/bin/env python3
"""
CG002-R7g — validación del primitivo de 3 puntos
================================================
Post-decisión Mesa (Alexis): extender c_ij con vértices ternarios.

Dos pruebas mínimas de la pared R7:
  · **gauge** — quark + gluón (R7b re-hecho con λ corregido + triplete q–g–q′)
  · **yukawa** — fermiones R7e + Higgs (triplete f–H–f′)

Controles: pareado-only (triplets vacíos) vs vertex3.

USO:
  python3 cg002_r7g_vertex3.py --mode gauge --single
  python3 cg002_r7g_vertex3.py --mode yukawa --quick
  python3 cg002_r7g_vertex3.py --mode both
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

from cg002_primitivo_vertex3 import (
    ALPHA,
    ETA,
    KAPPA_S,
    MU,
    PASOS,
    S0,
    S_BAND,
    build_gauge_triplets,
    build_yukawa_triplets,
    evolucion_vertex3,
    sat,
    vertex_coupling_strength,
)

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_r7g_sweep.csv"
FIG_PATH = OUT_DIR / "cg002_r7g_vertex3.png"

# --- gauge (R7a/b) ---
N_QUARK_MICRO = 12
N_GLUON_MICRO = 8
N_MICRO_GAUGE = 20
COLORS = np.eye(3, dtype=float)
GAUGE_SCALE = 0.35

# --- yukawa (R7e + H) ---
N_QUARK = 72
N_LEPTON = 12
N_FERMION = 84
MS_HIGGS = 84
N_MICRO_YUK = 85
YUK_SCALE = 0.06


def decode_quark_r7a(ms: int):
    is_aq = ms >= 6
    base = ms - 6 if is_aq else ms
    return is_aq, base // 2, base % 2


def color_vector(is_aq: bool, ci: int) -> np.ndarray:
    v = COLORS[ci]
    return -v if is_aq else v


def gluon_label(a: int) -> int:
    return N_QUARK_MICRO + a


def encode_quark_r7e(is_aq, gen, flavor, color, spin):
    base = gen * 12 + flavor * 6 + color * 2 + spin
    return 36 + base if is_aq else base


def is_lepton(ms: int) -> bool:
    return N_QUARK <= ms < N_FERMION


def charge_value(ms: int) -> float:
    if ms >= N_FERMION:
        return 0.0
    if ms < N_QUARK:
        is_aq = ms >= 36
        base = ms - 36 if is_aq else ms
        flavor = (base % 12) // 6
        q = 2.0 / 3.0 if flavor == 0 else -1.0 / 3.0
        return -q if is_aq else q
    base = ms - N_QUARK
    spec = (base // 2) % 2
    if spec == 1:
        return 0.0
    return -1.0 if base % 2 == 0 else 1.0


def semilla_gauge(N: int, rng: np.random.Generator):
    counts = np.full(N_MICRO_GAUGE, N // N_MICRO_GAUGE)
    for _ in range(N % N_MICRO_GAUGE):
        counts[rng.integers(0, N_MICRO_GAUGE)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.zeros((N, 3))
    pos = 0
    for ms in range(N_MICRO_GAUGE):
        n = counts[ms]
        labels[pos : pos + n] = ms
        if ms < N_QUARK_MICRO:
            is_aq, ci, _ = decode_quark_r7a(ms)
            V[pos : pos + n] = color_vector(is_aq, ci)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm]


def semilla_yukawa(N: int, rng: np.random.Generator):
    counts = np.full(N_MICRO_YUK, N // N_MICRO_YUK)
    for _ in range(N % N_MICRO_YUK):
        counts[rng.integers(0, N_MICRO_YUK)] += 1
    labels = np.empty(N, dtype=np.int8)
    V = np.zeros((N, 3))
    Q = np.zeros(N)
    pos = 0
    for ms in range(N_MICRO_YUK):
        n = counts[ms]
        labels[pos : pos + n] = ms
        if ms < N_QUARK:
            is_aq = ms >= 36
            base = ms - 36 if is_aq else ms
            ci = (base % 6) // 2
            V[pos : pos + n] = color_vector(is_aq, ci)
        Q[pos : pos + n] = charge_value(ms)
        pos += n
    perm = rng.permutation(N)
    return labels[perm], V[perm], Q[perm]


def build_C_pair_gauge(V, labels):
    N = len(labels)
    is_g = labels >= N_QUARK_MICRO
    C = np.zeros((N, N))
    idx_q = np.where(~is_g)[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    np.fill_diagonal(C, 0.0)
    return C


def build_C_pair_yukawa(V, Q, labels):
    N = len(labels)
    is_f = labels < N_FERMION
    C = np.zeros((N, N))
    idx_q = np.where(is_f & (labels < N_QUARK))[0]
    if len(idx_q):
        Vq = V[idx_q]
        C[np.ix_(idx_q, idx_q)] = Vq @ Vq.T
    C += np.outer(Q, Q)
    np.fill_diagonal(C, 0.0)
    return C


def orden_color(V, alive):
    idx = np.where(alive)[0]
    return float(np.linalg.norm(V[idx].mean(0))) if len(idx) else 0.0


def orden_charge(Q, alive):
    idx = np.where(alive)[0]
    return float(abs(Q[idx].mean())) if len(idx) else 0.0


def excess_L2(counts, K):
    N = counts.sum()
    if N == 0:
        return 0.0
    f = counts / N
    return float(np.sqrt(((f - 1 / K) ** 2).sum()))


def null_L2(K, N):
    return math.sqrt((K - 1) / (K * max(N, 1)))


def run_gauge(N, seed, use_vertex3: bool):
    rng = np.random.default_rng(seed)
    labels, V = semilla_gauge(N, rng)
    C = build_C_pair_gauge(V, labels)
    idx_q = np.where(labels < N_QUARK_MICRO)[0]
    idx_g = np.where(labels >= N_QUARK_MICRO)[0]
    triplets = []
    if use_vertex3:
        triplets = build_gauge_triplets(V, labels, idx_q, idx_g, gluon_label, GAUGE_SCALE)
    S, alive, tau = evolucion_vertex3(C, triplets, N)
    n_surv = int(alive.sum())
    surv_g = labels[alive] >= N_QUARK_MICRO
    n_gluon_surv = int(surv_g.sum()) if n_surv else 0
    n_gluon_init = int((labels >= N_QUARK_MICRO).sum())
    gluon_frac_surv = n_gluon_surv / max(n_gluon_init, 1)
    cnt8 = np.bincount(labels[alive][labels[alive] >= N_QUARK_MICRO] - N_QUARK_MICRO, minlength=8)
    return dict(
        mode="gauge",
        vertex3=use_vertex3,
        N=N,
        seed=seed,
        n_surv=n_surv,
        f_surv=n_surv / N,
        tau=tau,
        orden_color=orden_color(V, alive),
        gluon_frac_surv=gluon_frac_surv,
        n_triplets=len(triplets),
        coup_higgs=0.0,
        coup_gluon=vertex_coupling_strength(triplets, idx_g),
        L2_K8=excess_L2(cnt8, 8),
        ratio_L2_K8=excess_L2(cnt8, 8) / null_L2(8, int(cnt8.sum())) if cnt8.sum() else 0.0,
        phi_peak=0.0,
        phi_final=0.0,
        frac_higgs_surv=0.0,
        mass_ratio=0.0,
    )


def run_yukawa(N, seed, use_vertex3: bool):
    rng = np.random.default_rng(seed)
    labels, V, Q = semilla_yukawa(N, rng)
    C = build_C_pair_yukawa(V, Q, labels)
    idx_f = np.where(labels < N_FERMION)[0]
    idx_h = np.where(labels == MS_HIGGS)[0]
    triplets = []
    if use_vertex3:
        triplets = build_yukawa_triplets(idx_f, idx_h, YUK_SCALE, max_pairs=min(400, len(idx_f)))
    S, alive, tau = evolucion_vertex3(C, triplets, N)
    higgs_alive = alive & (labels == MS_HIGGS)
    phi_final = float(np.sqrt(sat(S[higgs_alive]).mean())) if higgs_alive.any() else 0.0
    phi_init = float(np.sqrt(sat(np.full(int(higgs_alive.sum()) or 1, S0)).mean()))
    phi_peak = phi_final  # v0: sin trace intra; report final
    m_by = []
    if phi_final > 0:
        idx = np.where(alive & (labels < N_FERMION))[0]
        masses = np.sqrt(sat(S[idx])) * phi_final
        for g in range(3):
            gens = np.array([(labels[i] % 36) // 12 if labels[i] < N_QUARK else (labels[i] - N_QUARK) // 4 for i in idx])
            m_by.append(float(masses[gens == g].mean()) if (gens == g).any() else 0.0)
    ratio = m_by[2] / m_by[0] if m_by and m_by[0] > 1e-9 else 0.0
    cnt3 = np.zeros(3)
    for q in Q[alive]:
        if q < -0.1:
            cnt3[0] += 1
        elif q > 0.1:
            cnt3[2] += 1
        else:
            cnt3[1] += 1
    n_surv = int(alive.sum())
    return dict(
        mode="yukawa",
        vertex3=use_vertex3,
        N=N,
        seed=seed,
        n_surv=n_surv,
        f_surv=n_surv / N,
        tau=tau,
        orden_color=orden_color(V, alive),
        orden_charge=orden_charge(Q, alive),
        gluon_frac_surv=0.0,
        n_triplets=len(triplets),
        coup_higgs=vertex_coupling_strength(triplets, idx_h),
        coup_gluon=0.0,
        L2_K8=0.0,
        ratio_L2_K8=0.0,
        phi_peak=phi_peak,
        phi_final=phi_final,
        frac_higgs_surv=float(higgs_alive.mean()) if len(idx_h) else 0.0,
        mass_ratio=ratio,
        L2_K3=excess_L2(cnt3, 3),
        ratio_L2_K3=excess_L2(cnt3, 3) / null_L2(3, int(cnt3.sum())) if cnt3.sum() else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gauge", "yukawa", "both"), default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    modes = ("gauge", "yukawa") if args.mode == "both" else (args.mode,)

    if args.single:
        for m in modes:
            fn = run_gauge if m == "gauge" else run_yukawa
            Ns = 800 if m == "gauge" else 2000
            print(m, "pareado:", fn(Ns, 1, False))
            print(m, "vertex3:", fn(Ns, 1, True))
        return

    NS = [500, 2000] if args.quick else [500, 1000, 2000, 4000]
    SEEDS = 8 if args.quick else 20
    t0 = time.monotonic()
    rows = []
    for m in modes:
        fn = run_gauge if m == "gauge" else run_yukawa
        for N in NS:
            for seed in range(1, SEEDS + 1):
                rows.append(fn(N, seed, False))
                rows.append(fn(N, seed, True))
            sub = [r for r in rows if r["mode"] == m and r["N"] == N and r["vertex3"]]
            print(
                f"  {m} N={N:4d}  f_v3={np.mean([r['f_surv'] for r in sub]):.3f}  "
                f"trip={np.mean([r['n_triplets'] for r in sub]):.0f}  "
                + (
                    f"g_surv={np.mean([r['gluon_frac_surv'] for r in sub]):.2f}  L2_K8×={np.mean([r['ratio_L2_K8'] for r in sub]):.2f}"
                    if m == "gauge"
                    else f"φ_f={np.mean([r['phi_final'] for r in sub]):.3f}  H_surv={np.mean([r['frac_higgs_surv'] for r in sub]):.2f}  coup_H={np.mean([r['coup_higgs'] for r in sub]):.1f}"
                ),
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mi, m in enumerate(modes):
        sub = df[(df["mode"] == m) & (df["vertex3"])]
        agg = sub.groupby("N").mean(numeric_only=True)
        if m == "gauge":
            ax[mi].plot(agg.index, agg.gluon_frac_surv, "o-", label="frac gluón superv.")
            ax[mi].plot(agg.index, agg.ratio_L2_K8, "s--", label="K8 exceso")
        else:
            ax[mi].plot(agg.index, agg.phi_final, "o-", label="φ final")
            ax[mi].plot(agg.index, agg.frac_higgs_surv, "s--", label="Higgs superv.")
        ax[mi].set_title(f"R7g {m} vertex3")
        ax[mi].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    print("\n" + "=" * 60)
    print("CG002-R7g primitivo 3 puntos")
    print(f"CSV: {SWEEP_CSV}  Fig: {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()