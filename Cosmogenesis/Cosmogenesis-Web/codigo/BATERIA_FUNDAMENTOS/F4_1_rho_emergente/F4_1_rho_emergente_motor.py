#!/usr/bin/env python3
"""
F4_1_rho_emergente_motor.py — BATERÍA_FUNDAMENTOS, Enfoque 4, experimento F4-1

"Densidad emergente: ¿ρ cae sola al expandir, sin imponerlo?"

Diseño y criterio de PASS congelados en `PROTOCOLO_F4-1_PREREGISTRO.md` (mismo
directorio, mtime ANTERIOR a este archivo). No se toca el protocolo tras ver resultados
(T3). Este motor entrega números crudos; la adjudicación final es de CS.

Diferencia deliberada con `CF2_estiramiento_motor.py` (leído completo, NO editado, NO
importado): CF-2 impone `rho = RHO0/(a**3)` directamente en el código — exactamente la
ley que F4-1 debe medir, no imponer (T1). Aquí:

  1) El campo T (malla comóvil LxL, mismo "sello" físico que CF2: salto tanh, difusión
     de 5 puntos, DT=0.25, N_SUB=2, ORIGINAL_STEPS_PER_TG=399, H_EXP=6.0) se interpreta
     como una densidad de masa/energía LOCAL en unidades comóviles.
  2) El coeficiente de difusión D se mantiene FIJO (D0=0.12) para toda la corrida — NO
     depende de `a` ni de ninguna `rho` impuesta. La dinámica interna del campo es
     independiente de la expansión; sólo la LECTURA de su densidad física depende de
     `a`, y esa lectura es una conversión de volumen (V_fis=L²·a³), no una fórmula de
     densidad.
  3) En cada checkpoint se MIDE del estado actual: M_comov(t)=Σ T_i(t) (Método A, suma
     global) y M_banda(t)=Σ T_i(t) sobre el bloque central L/4..3L/4 (Método B, segundo
     observable independiente, evita re-medir lo mismo con otro nombre).
  4) ρ_eff_A(a) = M_comov(a) / (L²·a³);  ρ_eff_B(a) = M_banda(a) / ((L/2)²·a³).
     El NULL usa el MISMO M_comov(t)/M_banda(t) medido pero con divisor de volumen fijo
     en 1 (a≡1, sin expansión) — es la misma trayectoria física leída de dos formas, lo
     que aísla limpiamente el efecto de la conversión de volumen sin mezclar semillas
     distintas entre REAL y NULL.
  5) Ruido dinámico (T7): en cada subpaso de difusión se inyecta ruido gaussiano de
     amplitud `sigma_ruido` (barrida en log, no cosmético de semilla) ANTES del recorte
     a [0,1] final del paso. El recorte + el ruido son las ÚNICAS fuentes de
     no-conservación exacta de M_comov — reales, no disimuladas — y son lo que hace que
     el exponente medido pueda diferir de −3 en vez de ser una certeza aritmética.
  6) El exponente (pendiente log-log ρ_eff vs a) se AJUSTA por mínimos cuadrados, nunca
     se fija; se compara con −3 sólo al final, con incertidumbre entre semillas.

No edita `CF2_estiramiento_motor.py` ni ningún archivo de otro prefijo (F3_*, F4_2_*,
etc.). No topología. No commits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello físico heredado de CF2 (idéntico; NO se retoca, T1)
# ============================================================
L = 64
H_EXP = 6.0
D0 = 0.12
W0 = 1.2
DT = 0.25
N_SUB = 2
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F4-1_PREREGISTRO.md, sección 4)
# ============================================================
A_GRID = np.geomspace(1.0, 1.0e4, 12)
SIGMA_GRID = np.geomspace(1.0e-5, 1.0e-1, 8)
SEEDS_STANDARD = [
    7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321,
    13, 271828, 161803, 31415, 90210, 20260724,
]

# ============================================================
# Criterio de PASS pre-registrado (sección 6, congelado)
# ============================================================
MONO_TOL = 1e-6
SLOPE_DIFF_MIN = 0.3
PASS_RATE_MIN = 0.55
EXPECTED_SLOPE_3D = -3.0

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F4_1_rho_emergente"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical (idéntico a CF2): T≈1 a la izquierda, T≈0 a la derecha."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def diffuse_with_noise(
    T: np.ndarray, D: float, dt: float, n_sub: int, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Difusión de 5 puntos (idéntico núcleo a CF2.diffuse) + ruido dinámico Wiener
    inyectado EN CADA SUBPASO (T7: perturbación de la dinámica, no de la semilla),
    de amplitud `sigma` escalada por sqrt(dt_sub). D se mantiene FIJO — no depende de
    `a` ni de ninguna `rho` impuesta (a diferencia de CF2).
    """
    out = T
    dt_sub = dt / n_sub
    for _ in range(n_sub):
        lap = (
            np.roll(out, -1, 1)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 0)
            - 4.0 * out
        )
        out = out + dt_sub * D * lap
        if sigma > 0.0:
            out = out + sigma * np.sqrt(dt_sub) * rng.normal(size=out.shape)
    return out


def measure_state(T: np.ndarray) -> dict:
    """
    Método A (global) y Método B (bloque central, segundo observable independiente):
    contenido de masa/energía comóvil MEDIDO del estado actual, sin ninguna fórmula de
    `a` involucrada aquí.
    """
    M_comov = float(np.sum(T))
    lo, hi = L // 4, 3 * L // 4
    banda = T[lo:hi, lo:hi]
    M_banda = float(np.sum(banda))
    n_banda_cells = banda.shape[0] * banda.shape[1]
    return {"M_comov": M_comov, "M_banda": M_banda, "n_banda_cells": n_banda_cells}


def run_trajectory(seed: int, sigma: float, a_grid: np.ndarray) -> dict:
    """
    Integra difusión+ruido desde t_g=0 (D FIJO, independiente de `a`) y muestrea
    M_comov(t)/M_banda(t) en los checkpoints t_g(a)=ln(a)/H_EXP del barrido de `a`.
    Cadena markoviana sin look-ahead: muestrear checkpoints de una única trayectoria es
    idéntico a re-simular desde cero hasta cada t_g objetivo (mismo método de CF2/F3-1).
    """
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        m = measure_state(T)
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "M_comov": m["M_comov"],
                "M_banda": m["M_banda"],
                "n_banda_cells": m["n_banda_cells"],
            }
        )

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        T = diffuse_with_noise(T, D0, DT, N_SUB, sigma, rng)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    M_comov = np.array([c["M_comov"] for c in checkpoints])
    M_banda = np.array([c["M_banda"] for c in checkpoints])
    n_banda_cells = checkpoints[0]["n_banda_cells"]

    # --- Método A: densidad global ---
    rho_A_REAL = M_comov / ((L * L) * (a_vals**3))
    rho_A_NULL = M_comov / (L * L)  # divisor de volumen fijo en 1 (sin expansión)

    # --- Método B: densidad de bloque central ---
    rho_B_REAL = M_banda / (n_banda_cells * (a_vals**3))
    rho_B_NULL = M_banda / n_banda_cells

    return {
        "seed": seed,
        "sigma_ruido": sigma,
        "a_grid": a_vals.tolist(),
        "M_comov": M_comov.tolist(),
        "M_banda": M_banda.tolist(),
        "rho_A_REAL": rho_A_REAL.tolist(),
        "rho_A_NULL": rho_A_NULL.tolist(),
        "rho_B_REAL": rho_B_REAL.tolist(),
        "rho_B_NULL": rho_B_NULL.tolist(),
    }


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] * (1.0 + tol):
            return False
    return True


def loglog_slope(a_vals: np.ndarray, y_vals: np.ndarray) -> float:
    x = np.log(a_vals)
    y = np.log(np.clip(np.asarray(y_vals, dtype=float), 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def evaluate_trajectory(traj: dict) -> dict:
    a_vals = np.array(traj["a_grid"])

    out = {}
    for method in ("A", "B"):
        rho_real = np.array(traj[f"rho_{method}_REAL"])
        rho_null = np.array(traj[f"rho_{method}_NULL"])

        mono_real = monotonic_nonincreasing(rho_real)
        mono_null = monotonic_nonincreasing(rho_null)
        slope_real = loglog_slope(a_vals, rho_real)
        slope_null = loglog_slope(a_vals, rho_null)
        slope_diff = abs(slope_null - slope_real)

        cond_a = mono_real
        cond_b = (not mono_null) or (slope_diff >= SLOPE_DIFF_MIN)
        punto_pass = bool(cond_a and cond_b)

        out[method] = {
            "mono_REAL": bool(mono_real),
            "mono_NULL": bool(mono_null),
            "slope_REAL": slope_real,
            "slope_NULL": slope_null,
            "slope_diff_abs": float(slope_diff),
            "null_muerde": bool(cond_b),
            "punto_pass": punto_pass,
        }

    out["punto_pass"] = bool(out["A"]["punto_pass"] and out["B"]["punto_pass"])
    return out


def run_production(seeds: list[int], sigma_grid: np.ndarray, a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    per_sigma = {}

    for sigma in sigma_grid:
        sigma_key = f"{sigma:.6e}"
        per_seed = {}
        n_pass = 0
        for seed in seeds:
            traj = run_trajectory(seed, float(sigma), a_grid)
            ev = evaluate_trajectory(traj)
            per_seed[str(seed)] = {"trayectoria": traj, "evaluacion": ev}
            if ev["punto_pass"]:
                n_pass += 1
        rate = n_pass / len(seeds) if seeds else 0.0

        slopes_real_A = [per_seed[str(s)]["evaluacion"]["A"]["slope_REAL"] for s in seeds]
        slopes_real_B = [per_seed[str(s)]["evaluacion"]["B"]["slope_REAL"] for s in seeds]
        slopes_null_A = [per_seed[str(s)]["evaluacion"]["A"]["slope_NULL"] for s in seeds]
        slopes_null_B = [per_seed[str(s)]["evaluacion"]["B"]["slope_NULL"] for s in seeds]

        per_sigma[sigma_key] = {
            "sigma_ruido": float(sigma),
            "n_seeds_pass": n_pass,
            "n_seeds_total": len(seeds),
            "rate": rate,
            "robusto": bool(rate >= PASS_RATE_MIN),
            "exponente_A_REAL_media": float(np.mean(slopes_real_A)),
            "exponente_A_REAL_std": float(np.std(slopes_real_A)),
            "exponente_B_REAL_media": float(np.mean(slopes_real_B)),
            "exponente_B_REAL_std": float(np.std(slopes_real_B)),
            "exponente_A_NULL_media": float(np.mean(slopes_null_A)),
            "exponente_A_NULL_std": float(np.std(slopes_null_A)),
            "exponente_B_NULL_media": float(np.mean(slopes_null_B)),
            "exponente_B_NULL_std": float(np.std(slopes_null_B)),
            "resultados_por_semilla": per_seed,
        }

    todos_robustos = all(per_sigma[k]["robusto"] for k in per_sigma)
    verdict_label = "F4_1_ROBUSTO" if todos_robustos else "F4_1_FAIL_INESTABLE"

    todos_slopes_real_A = [
        per_sigma[k]["resultados_por_semilla"][str(s)]["evaluacion"]["A"]["slope_REAL"]
        for k in per_sigma
        for s in seeds
    ]
    todos_slopes_real_B = [
        per_sigma[k]["resultados_por_semilla"][str(s)]["evaluacion"]["B"]["slope_REAL"]
        for k in per_sigma
        for s in seeds
    ]

    payload = {
        "experimento": "F4-1 densidad emergente",
        "tag": tag,
        "sello": {
            "L": L,
            "H_EXP": H_EXP,
            "D0": D0,
            "W0": W0,
            "DT": DT,
            "N_SUB": N_SUB,
            "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "sigma_grid": sigma_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "EXPECTED_SLOPE_3D": EXPECTED_SLOPE_3D,
            "descripcion": (
                "punto_pass = punto_pass_A AND punto_pass_B; "
                "punto_pass_M = mono_REAL_M AND (NOT mono_NULL_M OR "
                "abs(slope_NULL_M - slope_REAL_M) >= SLOPE_DIFF_MIN), M en {A,B}. "
                "robusto(sigma) = rate(sigma) >= PASS_RATE_MIN. "
                "F4_1_ROBUSTO si robusto(sigma) para TODO sigma del barrido."
            ),
        },
        "resultados_por_sigma": per_sigma,
        "resumen": {
            "verdict": verdict_label,
            "todos_los_sigma_robustos": todos_robustos,
            "exponente_A_REAL_media_global": float(np.mean(todos_slopes_real_A)),
            "exponente_A_REAL_std_global": float(np.std(todos_slopes_real_A)),
            "exponente_B_REAL_media_global": float(np.mean(todos_slopes_real_B)),
            "exponente_B_REAL_std_global": float(np.std(todos_slopes_real_B)),
            "diferencia_con_menos_3_A": float(np.mean(todos_slopes_real_A) - EXPECTED_SLOPE_3D),
            "diferencia_con_menos_3_B": float(np.mean(todos_slopes_real_B) - EXPECTED_SLOPE_3D),
        },
        "runtime_seconds": time.time() - t0,
        "generated_at_unix": time.time(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion"], default="produccion", nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        seeds = SEEDS_STANDARD[:3]
        sigma_grid = SIGMA_GRID[::3]  # submuestra reducida
        a_grid = np.geomspace(1.0, 100.0, 5)
        tag = "smoke"
    else:
        seeds = SEEDS_STANDARD
        sigma_grid = SIGMA_GRID
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== F4-1 densidad emergente — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"sigma_grid={sigma_grid.tolist()}")
    print(f"a_grid={a_grid.tolist()}")

    payload = run_production(seeds, sigma_grid, a_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    for sigma_key, rec in payload["resultados_por_sigma"].items():
        print(
            f"  sigma={rec['sigma_ruido']:.3e}  rate={rec['rate']:.3f}  "
            f"robusto={rec['robusto']}  "
            f"slope_A_REAL={rec['exponente_A_REAL_media']:.4f}±{rec['exponente_A_REAL_std']:.4f}  "
            f"slope_B_REAL={rec['exponente_B_REAL_media']:.4f}±{rec['exponente_B_REAL_std']:.4f}  "
            f"slope_A_NULL={rec['exponente_A_NULL_media']:.4f}±{rec['exponente_A_NULL_std']:.4f}"
        )
    r = payload["resumen"]
    print(f"\nverdict={r['verdict']}")
    print(
        f"exponente global Método A: {r['exponente_A_REAL_media_global']:.4f} "
        f"± {r['exponente_A_REAL_std_global']:.4f}  (Δ vs -3 = {r['diferencia_con_menos_3_A']:.4f})"
    )
    print(
        f"exponente global Método B: {r['exponente_B_REAL_media_global']:.4f} "
        f"± {r['exponente_B_REAL_std_global']:.4f}  (Δ vs -3 = {r['diferencia_con_menos_3_B']:.4f})"
    )

    out_json = OUT_DIR / f"F4_1_rho_emergente_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
