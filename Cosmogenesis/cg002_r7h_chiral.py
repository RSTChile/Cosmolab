#!/usr/bin/env python3
"""
CG002-R7h — Yukawa quiral (EIT3 R_θ trasplantado)
==================================================
Post CC Msg 44: visto bueno condicional con 3 correcciones aplicadas.

Batería --quick:
  chiral ON (ciclo límite) | vertex3 OFF | θ=0 | θ→−θ | β=0 | E=0 | shuffle_w

Caveat encabezado: receta EIT3 en vector de patas, no campo espacial 2D.
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

from cg002_primitivo_chiral import (
    ALPHA_LOOP,
    BETA_MEM,
    SIGMA_E,
    THETA_DEG,
    evolucion_chiral_vertex3,
)
from cg002_primitivo_vertex3 import (
    KAPPA_S,
    S0,
    S_BAND,
    build_yukawa_triplets,
    evolucion_vertex3,
    sat,
    vertex_coupling_strength,
)

OUT_DIR = Path(__file__).parent
SWEEP_CSV = OUT_DIR / "cg002_r7h_sweep.csv"
FIG_PATH = OUT_DIR / "cg002_r7h_chiral.png"

N_QUARK = 72
N_FERMION = 84
MS_HIGGS = 84
N_MICRO_YUK = 85
YUK_SCALE = 0.06

# Re-export semilla helpers from r7g
from cg002_r7g_vertex3 import (  # noqa: E402
    build_C_pair_yukawa,
    excess_L2,
    null_L2,
    orden_charge,
    orden_color,
    semilla_yukawa,
)


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


def run_yukawa_chiral(
    N: int,
    seed: int,
    *,
    use_chiral: bool = True,
    use_vertex3: bool = True,
    theta_deg: float = THETA_DEG,
    beta_mem: float = BETA_MEM,
    sigma_e: float = SIGMA_E,
    shuffle_signs: bool = False,
    cond: str = "chiral_on",
) -> dict:
    rng = np.random.default_rng(seed)
    labels, V, Q = semilla_yukawa(N, rng)
    C = build_C_pair_yukawa(V, Q, labels)
    idx_f = np.where(labels < N_FERMION)[0]
    idx_h = np.where(labels == MS_HIGGS)[0]
    triplets = []
    if use_vertex3:
        triplets = build_yukawa_triplets(idx_f, idx_h, YUK_SCALE, max_pairs=min(400, len(idx_f)))

    if use_chiral and use_vertex3 and triplets:
        S, alive, tau, M = evolucion_chiral_vertex3(
            C,
            triplets,
            Q,
            N,
            yuk_scale=YUK_SCALE,
            theta_deg=theta_deg,
            beta_mem=beta_mem,
            sigma_e=sigma_e,
            shuffle_signs=shuffle_signs,
            rng=rng,
        )
        mem_norm = float(np.mean([np.linalg.norm(v) for v in M.values()])) if M else 0.0
    else:
        S, alive, tau = evolucion_vertex3(C, triplets if use_vertex3 else [], N)
        mem_norm = 0.0

    higgs_alive = alive & (labels == MS_HIGGS)
    phi_final = float(np.sqrt(sat(S[higgs_alive]).mean())) if higgs_alive.any() else 0.0
    m_by = []
    if phi_final > 0:
        idx = np.where(alive & (labels < N_FERMION))[0]
        masses = np.sqrt(sat(S[idx])) * phi_final
        for g in range(3):
            gens = np.array(
                [(labels[i] % 36) // 12 if labels[i] < N_QUARK else (labels[i] - N_QUARK) // 4 for i in idx]
            )
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
        cond=cond,
        chiral=use_chiral and use_vertex3,
        vertex3=use_vertex3,
        theta_deg=theta_deg,
        beta_mem=beta_mem,
        sigma_e=sigma_e,
        shuffle_signs=shuffle_signs,
        N=N,
        seed=seed,
        n_surv=n_surv,
        f_surv=n_surv / N,
        tau=tau,
        orden_color=orden_color(V, alive),
        orden_charge=orden_charge(Q, alive),
        n_triplets=len(triplets),
        coup_higgs=vertex_coupling_strength(triplets, idx_h) if use_vertex3 else 0.0,
        phi_final=phi_final,
        frac_higgs_surv=float(higgs_alive.sum()) / max(len(idx_h), 1),
        mass_ratio=ratio,
        mem_norm=mem_norm,
        L2_K3=excess_L2(cnt3, 3),
        ratio_L2_K3=excess_L2(cnt3, 3) / null_L2(3, int(cnt3.sum())) if cnt3.sum() else 0.0,
        alpha_loop=ALPHA_LOOP,
    )


CONDITIONS = [
    ("chiral_on", dict(use_chiral=True, use_vertex3=True, theta_deg=THETA_DEG, beta_mem=BETA_MEM, sigma_e=SIGMA_E)),
    ("vertex3_off", dict(use_chiral=False, use_vertex3=False)),
    ("v3_symmetric", dict(use_chiral=False, use_vertex3=True)),
    ("theta_zero", dict(use_chiral=True, use_vertex3=True, theta_deg=0.0)),
    ("theta_mirror", dict(use_chiral=True, use_vertex3=True, theta_deg=-THETA_DEG)),
    ("beta_zero", dict(use_chiral=True, use_vertex3=True, beta_mem=0.0)),
    ("E_zero", dict(use_chiral=True, use_vertex3=True, sigma_e=0.0)),
    ("shuffle_w", dict(use_chiral=True, use_vertex3=True, shuffle_signs=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    if args.single:
        for name, kw in CONDITIONS[:3]:
            print(name, run_yukawa_chiral(2000, 1, cond=name, **kw))
        return

    NS = [500, 2000] if args.quick else [500, 1000, 2000, 4000]
    SEEDS = 4 if args.quick else 12
    t0 = time.monotonic()
    rows = []
    for cond_name, kw in CONDITIONS:
        for N in NS:
            for seed in range(1, SEEDS + 1):
                rows.append(run_yukawa_chiral(N, seed, cond=cond_name, **kw))
            sub = [r for r in rows if r["cond"] == cond_name and r["N"] == N]
            print(
                f"  {cond_name:14s} N={N:4d}  f={np.mean([r['f_surv'] for r in sub]):.3f}  "
                f"mass_r={np.mean([r['mass_ratio'] for r in sub]):.3f}  "
                f"φ={np.mean([r['phi_final'] for r in sub]):.3f}  "
                f"mem={np.mean([r['mem_norm'] for r in sub]):.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ax_i, metric in enumerate(("mass_ratio", "f_surv")):
        for cond in df["cond"].unique():
            sub = df[df["cond"] == cond].groupby("N")[metric].mean()
            ax[ax_i].plot(sub.index, sub.values, "o-", label=cond, ms=4)
        ax[ax_i].set_title(metric)
        ax[ax_i].legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)

    print("\n" + "=" * 60)
    print("CG002-R7h Yukawa quiral (EIT3 ciclo límite)")
    print(f"CSV: {SWEEP_CSV}  Fig: {FIG_PATH}")
    print(f"Tiempo: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()