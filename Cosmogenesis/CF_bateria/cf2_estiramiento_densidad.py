#!/usr/bin/env python3
"""
CF-2 — Enfriamiento por expansión / estiramiento del gradiente (barrido real)

Pre-registro: PROTOCOLO_CF2_estiramiento_densidad_PREREGISTRO.md (ANTES de producción).

Pregunta simple: al expandirse, ¿el gradiente físico se suaviza porque enfriar ES expandir
(estiramiento ∇_phys=∇_comov/a + dilución ρ∝a⁻³)?

Repara T7 del TEST_RHO viejo (1 semilla, sin barrido multi-seed de a).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "resultados_cf2"
OUT.mkdir(parents=True, exist_ok=True)

# --- sello pre-registrado (PROTOCOLO_CF2) ---
L = 64
PASOS = 400
RHO0 = 1.0
D0 = 0.12
W0 = 1.2
DT = 0.25
N_SUB = 2
SEEDS = (7, 42, 99, 777, 2025, 3141, 8191, 99991)
H_EXP_LIST = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

STRETCH_RATIO_MAX = 0.25
SMOOTH_WIDTH_MIN = 2.0
RHO_SEP_THR = 0.08
RATE_SEED_MIN = 0.70
N_H_PASS_MIN = 5
# al menos un H con a_final>50 y uno con a_final<20 entre los que pasan
A_HI, A_LO = 50.0, 20.0

PROTOCOL_ID = "CF2_ESTIRAMIENTO_DENSIDAD_2026-07-23"


def a_of(tg: float, H_EXP: float) -> float:
    return float(np.exp(H_EXP * tg))


def initial_T(L: int, w0: float) -> np.ndarray:
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    row = T.mean(axis=0)
    xs = np.arange(n, dtype=float)

    def cross(level: float) -> float:
        for i in range(n - 1):
            if (row[i] - level) * (row[i + 1] - level) <= 0:
                if row[i] == row[i + 1]:
                    return float(i)
                t = (level - row[i]) / (row[i + 1] - row[i])
                return float(i + t)
        return float(n // 2)

    w_comov = abs(cross(0.20) - cross(0.80))
    w_phys = w_comov * a
    return {
        "A_comov": A_comov,
        "A_phys": A_phys,
        "w_comov": w_comov,
        "w_phys": w_phys,
    }


def diffuse_step(T: np.ndarray, D: float) -> np.ndarray:
    if D <= 0:
        return T
    for _ in range(N_SUB):
        lap = (
            np.roll(T, 1, 0)
            + np.roll(T, -1, 0)
            + np.roll(T, 1, 1)
            + np.roll(T, -1, 1)
            - 4 * T
        )
        T = T + (D * DT / N_SUB) * lap
    return T


def run_arm(seed: int, H_EXP: float, mode: str) -> dict:
    """
    mode: REAL | NULL_RHO_FIXED | NULL_A_FIXED
    """
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    # ruido mínimo idéntico en estructura (solo fase del ruido)
    T = T + 1e-4 * rng.normal(size=T.shape)
    samples = []
    for step in range(PASOS):
        tg = step / max(PASOS - 1, 1)
        if mode == "NULL_A_FIXED":
            a = 1.0
            rho = RHO0
        else:
            a = a_of(tg, H_EXP)
            rho = RHO0 if mode == "NULL_RHO_FIXED" else RHO0 / (a**3)
        D = D0 * (rho / RHO0)
        T = diffuse_step(T, D)
        if step % 40 == 0 or step == PASOS - 1:
            m = grad_metrics(T, a)
            samples.append(
                {
                    "step": step,
                    "tg": tg,
                    "a": a,
                    "rho": rho,
                    "D": D,
                    **m,
                }
            )
    init, fin = samples[0], samples[-1]
    A_phys_ratio = fin["A_phys"] / max(init["A_phys"], 1e-12)
    w_phys_ratio = fin["w_phys"] / max(init["w_phys"], 1e-12)
    # mono: regresión log a → log A_phys (solo a>1.01)
    aa = np.array([s["a"] for s in samples if s["a"] > 1.01])
    Ap = np.array([s["A_phys"] for s in samples if s["a"] > 1.01])
    if len(aa) >= 3 and np.all(Ap > 0):
        slope = float(np.polyfit(np.log(aa), np.log(Ap), 1)[0])
    else:
        slope = 0.0  # a≈1: no hay estiramiento que medir
    return {
        "seed": seed,
        "H_EXP": H_EXP,
        "mode": mode,
        "a_final": fin["a"],
        "A_phys_init": init["A_phys"],
        "A_phys_final": fin["A_phys"],
        "A_comov_final": fin["A_comov"],
        "A_phys_ratio": float(A_phys_ratio),
        "w_phys_ratio": float(w_phys_ratio),
        "mono_slope_log": slope,
        "samples": samples,
    }


def evaluate_seed(seed: int, H_EXP: float) -> dict:
    real = run_arm(seed, H_EXP, "REAL")
    nrho = run_arm(seed, H_EXP, "NULL_RHO_FIXED")
    na = run_arm(seed, H_EXP, "NULL_A_FIXED")

    stretch = (real["A_phys_ratio"] < STRETCH_RATIO_MAX) and (
        real["w_phys_ratio"] > SMOOTH_WIDTH_MIN
    )
    denom = max(real["A_comov_final"], nrho["A_comov_final"], 1e-12)
    rho_sep = abs(real["A_comov_final"] - nrho["A_comov_final"]) / denom >= RHO_SEP_THR
    mono = real["mono_slope_log"] <= 0.0
    # NULL_A no debe "estirar" como REAL
    null_a_stretch = (na["A_phys_ratio"] < STRETCH_RATIO_MAX) and (
        na["w_phys_ratio"] > SMOOTH_WIDTH_MIN
    )

    return {
        "seed": seed,
        "H_EXP": H_EXP,
        "a_final": real["a_final"],
        "stretch": stretch,
        "rho_sep": rho_sep,
        "mono": mono,
        "null_a_stretch": null_a_stretch,
        "REAL": {k: real[k] for k in real if k != "samples"},
        "NULL_RHO_FIXED": {k: nrho[k] for k in nrho if k != "samples"},
        "NULL_A_FIXED": {k: na[k] for k in na if k != "samples"},
    }


def main():
    print(f"=== CF-2 {PROTOCOL_ID} ===\n")
    t0 = time.time()
    rows = []
    for H in H_EXP_LIST:
        print(f"--- H_EXP={H:.1f} a_final≈{np.exp(H):.1f} ---")
        for s in SEEDS:
            row = evaluate_seed(int(s), float(H))
            rows.append(row)
        sub = [r for r in rows if abs(r["H_EXP"] - H) < 1e-12]
        rs = np.mean([r["stretch"] for r in sub])
        rr = np.mean([r["rho_sep"] for r in sub])
        rm = np.mean([r["mono"] for r in sub])
        print(f"  rate_stretch={rs:.2f} rate_rho_sep={rr:.2f} rate_mono={rm:.2f}")

    # agregado por H
    by_H = []
    for H in H_EXP_LIST:
        sub = [r for r in rows if abs(r["H_EXP"] - H) < 1e-12]
        a_fin = float(np.mean([r["a_final"] for r in sub]))
        rate_stretch = float(np.mean([r["stretch"] for r in sub]))
        rate_rho = float(np.mean([r["rho_sep"] for r in sub]))
        rate_mono = float(np.mean([r["mono"] for r in sub]))
        rate_null_a_stretch = float(np.mean([r["null_a_stretch"] for r in sub]))
        pass_H = (
            rate_stretch >= RATE_SEED_MIN
            and rate_rho >= RATE_SEED_MIN
            and rate_mono >= RATE_SEED_MIN
        )
        by_H.append(
            {
                "H_EXP": H,
                "a_final_mean": a_fin,
                "rate_stretch": rate_stretch,
                "rate_rho_sep": rate_rho,
                "rate_mono": rate_mono,
                "rate_null_a_stretch": rate_null_a_stretch,
                "pass_H": pass_H,
                "mean_A_phys_ratio_REAL": float(
                    np.mean([r["REAL"]["A_phys_ratio"] for r in sub])
                ),
                "mean_A_phys_ratio_NULL_RHO": float(
                    np.mean([r["NULL_RHO_FIXED"]["A_phys_ratio"] for r in sub])
                ),
                "mean_A_phys_ratio_NULL_A": float(
                    np.mean([r["NULL_A_FIXED"]["A_phys_ratio"] for r in sub])
                ),
            }
        )

    n_pass_H = sum(1 for h in by_H if h["pass_H"])
    pass_hi = any(h["pass_H"] and h["a_final_mean"] > A_HI for h in by_H)
    pass_lo = any(h["pass_H"] and h["a_final_mean"] < A_LO for h in by_H)
    # NULL_A no debe estirar en masa: rate_null_a_stretch baja en H grandes
    null_a_ok = all(
        (h["rate_null_a_stretch"] <= 0.30) or (h["a_final_mean"] < 1.5)
        for h in by_H
    )

    global_pass = (
        n_pass_H >= N_H_PASS_MIN and pass_hi and pass_lo and null_a_ok
    )

    result = {
        "protocol_id": PROTOCOL_ID,
        "preregistro": "PROTOCOLO_CF2_estiramiento_densidad_PREREGISTRO.md",
        "seeds": list(SEEDS),
        "H_EXP_list": list(H_EXP_LIST),
        "thresholds": {
            "STRETCH_RATIO_MAX": STRETCH_RATIO_MAX,
            "SMOOTH_WIDTH_MIN": SMOOTH_WIDTH_MIN,
            "RHO_SEP_THR": RHO_SEP_THR,
            "RATE_SEED_MIN": RATE_SEED_MIN,
            "N_H_PASS_MIN": N_H_PASS_MIN,
            "A_HI": A_HI,
            "A_LO": A_LO,
        },
        "by_H": by_H,
        "rows": rows,
        "n_pass_H": n_pass_H,
        "pass_includes_a_gt_50": pass_hi,
        "pass_includes_a_lt_20": pass_lo,
        "null_a_ok": null_a_ok,
        "global_pass": global_pass,
        "verdict": "CF2_PASS" if global_pass else "CF2_FAIL",
        "elapsed_s": time.time() - t0,
    }

    path = OUT / "cf2_produccion_resultado.json"
    # rows are large; keep full for CS
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nJSON → {path}")
    print(f"verdict={result['verdict']} n_pass_H={n_pass_H}/8 hi={pass_hi} lo={pass_lo} null_a_ok={null_a_ok}")
    print(f"elapsed={result['elapsed_s']:.1f}s")

    # resumen corto
    md = [
        f"# CF-2 resultado crudo\n\n",
        f"**Protocolo:** `{PROTOCOL_ID}`\n\n",
        f"**Veredicto automático:** `{result['verdict']}`\n\n",
        f"| H_EXP | a_final | rate_stretch | rate_rho_sep | rate_mono | pass_H |\n",
        f"|-------|---------|--------------|--------------|-----------|--------|\n",
    ]
    for h in by_H:
        md.append(
            f"| {h['H_EXP']:.1f} | {h['a_final_mean']:.1f} | {h['rate_stretch']:.2f} | "
            f"{h['rate_rho_sep']:.2f} | {h['rate_mono']:.2f} | {h['pass_H']} |\n"
        )
    md.append(
        f"\nn_pass_H={n_pass_H} (min {N_H_PASS_MIN}); "
        f"incluye a>50: {pass_hi}; a<20: {pass_lo}; null_a_ok: {null_a_ok}\n"
    )
    (OUT / "RESUMEN_CF2_crudo.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
