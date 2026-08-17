#!/usr/bin/env python3
"""
F3_3_exponente_dilucion_motor.py — BATERÍA DE FUNDAMENTOS, experimento F3-3

"Tasa de dilución: ¿es a⁻³ especial o solo monótona?"

Generaliza CF2_estiramiento_motor.py (Cosmogenesis-Web/codigo/CF2_estiramiento/), que fijaba
ρ = ρ0/a³ (n=3) contra un único NULL de densidad fija. Aquí n es un parámetro BARRIDO explícito:

    ρ(a) = ρ0 · a^(−n),   D(a) = D0 · (ρ(a)/ρ0) = D0 · a^(−n),   n ∈ {0, 1, 2, 3, 4, 5}

  - n=0 reproduce exactamente el NULL_RHO_FIXED de CF2 (sin dilución) — es el NULL de F3-3.
  - n=1..5 son los 5 brazos "REAL" del barrido pre-registrado (ninguno privilegiado, T1).

Sustrato de campo (L, H_EXP, RHO0, D0, W0, DT, N_SUB, ORIGINAL_STEPS_PER_TG) heredado
LITERALMENTE de CF2_estiramiento_motor.py, sin retocar ningún valor (T1). No se importa el
módulo de CF2 para no acoplar rutas ni arriesgar tocarlo; se re-declara el mismo sello aquí.

Observable primario A_phys_max (idéntico al de CF2) + observable secundario A_phys_rms
(RMS en vez de máximo de la banda central) como segunda vía de verificación cruzada
independiente, según regla general de la batería (sección 0/1 del documento autoritativo).

Criterio de PASS pre-registrado y CONGELADO en PROTOCOLO_F3-3_PREREGISTRO.md (mtime anterior a
este archivo). Este script NO se auto-adjudica el veredicto cosmológico ni privilegia n=3 sobre
ningún otro punto del barrido — entrega la familia completa de curvas cruda; la lectura es de CS.

No edita CF2_estiramiento_motor.py, TEST_RHO_DISPERSION.py, ni ningún otro experimento de la
batería (prefijo propio F3_3_, carpeta propia).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado LITERAL de CF2_estiramiento_motor.py (T1: no se retoca)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2            # ancho comóvil inicial del salto (celdas)
DT = 0.25            # subpaso de difusión
N_SUB = 2             # subiteraciones de difusión por paso de reloj genético
ORIGINAL_STEPS_PER_TG = 399  # idéntico a CF2 / TEST_RHO_DISPERSION.py original

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F3-3_PREREGISTRO.md, sección 3)
# ============================================================
A_GRID = np.geomspace(1.0, 1000.0, 7)   # misma grilla EXACTA que CF2 (comparabilidad directa)
N_GRID = [0, 1, 2, 3, 4, 5]              # 0 = NULL (densidad fija); 1..5 = REAL barrido
N_REAL = [1, 2, 3, 4, 5]

SEEDS_STANDARD_CF2 = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_EXTENSION = [271828, 161803]       # dígitos de e y de φ; extensión a ≥12 semillas
SEEDS_F3_3 = SEEDS_STANDARD_CF2 + SEEDS_EXTENSION   # 12 semillas totales

# ============================================================
# Criterio de PASS pre-registrado (sección 6 del protocolo)
# ============================================================
MONO_TOL = 1e-9              # tolerancia de monotonicidad en a
SLOPE_DIFF_MIN = 0.05        # heredado literal de CF2
MONO_N_TOL = 1e-6            # tolerancia de monotonicidad en n (más laxa: solo 6 puntos)
SINGULARITY_FACTOR = 3.0     # curvatura en n=3 vs vecinos n=2,n=4
SINGULARITY_FLOOR = 1e-3     # piso absoluto anti-falso-positivo cuando la curva es ~lineal
PASS_RATE_MIN = 0.55         # idéntico umbral que CF2

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "F3_3_exponente_dilucion"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2_estiramiento_motor.py::initial_T."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Gradiente comóvil y físico, banda central (evita wrap-around periódico).
    Devuelve DOS observables independientes sobre la misma banda:
      - A_comov_max / A_phys_max: máximo |∂T/∂x| (idéntico al observable de CF2).
      - A_comov_rms / A_phys_rms: RMS |∂T/∂x| (observable secundario, forma completa del
        perfil, no solo el pico) — verificación cruzada de método (T2: ninguno de los dos
        comparte variable con el juez de otro experimento)."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov_max = float(g.max()) if g.size else 0.0
    A_comov_rms = float(np.sqrt(np.mean(g**2))) if g.size else 0.0
    a_safe = max(a, 1e-12)
    return {
        "A_comov_max": A_comov_max,
        "A_phys_max": A_comov_max / a_safe,
        "A_comov_rms": A_comov_rms,
        "A_phys_rms": A_comov_rms / a_safe,
    }


def diffuse(T: np.ndarray, D: float, dt: float, n_sub: int) -> np.ndarray:
    if D <= 0:
        return T
    out = T
    for _ in range(n_sub):
        lap = (
            np.roll(out, -1, 1)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 0)
            - 4.0 * out
        )
        out = out + (dt / n_sub) * D * lap
    return out


def run_sweep(n_exp: int, seed: int, a_grid: np.ndarray) -> dict:
    """
    Integra la difusión desde t_g=0 con ρ(a)=ρ0·a^(−n_exp), D(a)=D0·a^(−n_exp), y muestrea el
    estado del campo EXACTAMENTE en los instantes t_g(a)=ln(a)/H_EXP de cada punto del barrido
    de `a` (misma técnica de checkpointing markoviano que CF2_estiramiento_motor.py::run_sweep;
    válida aquí igual, porque la actualización de difusión no tiene look-ahead).

    n_exp=0 ⇒ ρ≡ρ0, D≡D0 (NULL_DENSIDAD_FIJA, idéntico al NULL_RHO_FIXED de CF2).
    n_exp=1..5 ⇒ brazos REAL del barrido pre-registrado.
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
        m = grad_metrics(T, a_now)
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "A_comov_max": m["A_comov_max"],
                "A_phys_max": m["A_phys_max"],
                "A_comov_rms": m["A_comov_rms"],
                "A_phys_rms": m["A_phys_rms"],
            }
        )

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        rho = RHO0 * (a ** (-n_exp))
        D = D0 * (rho / RHO0)  # = D0 * a^(-n_exp)

        T = diffuse(T, D, DT, N_SUB)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    return {
        "n_exp": n_exp,
        "seed": seed,
        "a_grid": a_vals.tolist(),
        "grad_fis_max": [c["A_phys_max"] for c in checkpoints],
        "grad_comov_max": [c["A_comov_max"] for c in checkpoints],
        "grad_fis_rms": [c["A_phys_rms"] for c in checkpoints],
        "grad_comov_rms": [c["A_comov_rms"] for c in checkpoints],
    }


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] * (1.0 + tol):
            return False
    return True


def monotonic_nondecreasing(vals: np.ndarray, tol: float = MONO_N_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] < vals[i] - tol:
            return False
    return True


def loglog_slope(a_vals: np.ndarray, grad_vals: np.ndarray) -> float:
    x = np.log(a_vals)
    y = np.log(np.clip(grad_vals, 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def evaluate_seed_family(runs_by_n: dict, observable_key: str) -> dict:
    """
    Evalúa los criterios pre-registrados para UNA semilla, sobre UN observable
    (observable_key ∈ {"grad_fis_max", "grad_fis_rms"}), a través de los 6 valores de n.
    """
    a_vals = np.array(runs_by_n[0]["a_grid"])

    mono = {}
    slope = {}
    for n in N_GRID:
        grad = np.array(runs_by_n[n][observable_key])
        mono[n] = monotonic_nonincreasing(grad)
        slope[n] = loglog_slope(a_vals, grad)

    # (3) NULL bites — generaliza el T4 de CF2 a través de todo el barrido de n
    null_bites = (not mono[0]) or any(
        abs(slope[n] - slope[0]) >= SLOPE_DIFF_MIN for n in N_REAL
    )

    # (4) monotonic_in_n — la verificación central de F3-3
    slope_real_seq = np.array([slope[n] for n in N_REAL])
    mono_in_n = monotonic_nondecreasing(slope_real_seq)
    # primer par consecutivo (si hay) donde se rompe la monotonicidad, para reporte honesto
    break_at = None
    for i in range(len(N_REAL) - 1):
        if slope_real_seq[i + 1] < slope_real_seq[i] - MONO_N_TOL:
            break_at = {"n_from": N_REAL[i], "n_to": N_REAL[i + 1],
                        "slope_from": float(slope_real_seq[i]), "slope_to": float(slope_real_seq[i + 1])}
            break

    # (5) n3_not_singular — curvatura discreta de slope(n) en n=2,3,4
    curv = {}
    for n in (2, 3, 4):
        curv[n] = slope[n - 1] - 2.0 * slope[n] + slope[n + 1]
    baseline = max(abs(curv[2]), abs(curv[4]), SINGULARITY_FLOOR)
    is_singular_n3 = abs(curv[3]) > SINGULARITY_FACTOR * baseline
    n3_not_singular = not is_singular_n3

    seed_pass = bool(null_bites and mono_in_n and n3_not_singular)

    return {
        "observable": observable_key,
        "mono_by_n": {str(n): bool(mono[n]) for n in N_GRID},
        "slope_by_n": {str(n): float(slope[n]) for n in N_GRID},
        "null_bites": bool(null_bites),
        "mono_in_n": bool(mono_in_n),
        "mono_in_n_break_at": break_at,
        "curvature_n2_n3_n4": {str(n): float(curv[n]) for n in (2, 3, 4)},
        "n3_singular": bool(is_singular_n3),
        "n3_not_singular": bool(n3_not_singular),
        "seed_pass": seed_pass,
    }


def run_production(seeds: list[int], a_grid: np.ndarray, n_grid: list[int], tag: str) -> dict:
    t0 = time.time()
    per_seed = {}
    n_pass_max = 0
    n_pass_rms = 0

    for seed in seeds:
        runs_by_n = {n: run_sweep(n, seed, a_grid) for n in n_grid}
        ev_max = evaluate_seed_family(runs_by_n, "grad_fis_max")
        ev_rms = evaluate_seed_family(runs_by_n, "grad_fis_rms")

        per_seed[str(seed)] = {
            "runs_por_n": {str(n): runs_by_n[n] for n in n_grid},
            "evaluation_max": ev_max,
            "evaluation_rms": ev_rms,
        }
        if ev_max["seed_pass"]:
            n_pass_max += 1
        if ev_rms["seed_pass"]:
            n_pass_rms += 1

    rate_max = n_pass_max / len(seeds) if seeds else 0.0
    rate_rms = n_pass_rms / len(seeds) if seeds else 0.0
    verdict_max = "F3_3_PASS" if rate_max >= PASS_RATE_MIN else "F3_3_FAIL"
    verdict_rms = "F3_3_PASS" if rate_rms >= PASS_RATE_MIN else "F3_3_FAIL"

    payload = {
        "experimento": "F3-3 tasa de dilución: ¿es a⁻³ especial o solo monótona?",
        "tag": tag,
        "sello": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0, "DT": DT,
            "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "n_grid": n_grid,
            "n_real": N_REAL,
            "n_null": 0,
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN,
            "MONO_N_TOL": MONO_N_TOL,
            "SINGULARITY_FACTOR": SINGULARITY_FACTOR,
            "SINGULARITY_FLOOR": SINGULARITY_FLOOR,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "seed_pass = null_bites(n=0 vs n=1..5) AND mono_in_n(slope(1..5) no-decreciente) "
                "AND n3_not_singular(curvatura(3) <= 3x max(curvatura(2),curvatura(4),1e-3)); "
                "evaluado independientemente sobre observable_max (pico) y observable_rms (RMS)."
            ),
        },
        "resultados_por_semilla": per_seed,
        "resumen": {
            "observable_max": {
                "n_seeds_pass": n_pass_max, "n_seeds_total": len(seeds),
                "rate": rate_max, "verdict": verdict_max,
            },
            "observable_rms": {
                "n_seeds_pass": n_pass_rms, "n_seeds_total": len(seeds),
                "rate": rate_rms, "verdict": verdict_rms,
            },
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
        seeds = SEEDS_F3_3[:3]
        a_grid = np.geomspace(1.0, 10.0, 3)
        n_grid = N_GRID  # evaluate_seed_family exige el barrido completo de n (0..5)
        tag = "smoke"
    else:
        seeds = SEEDS_F3_3
        a_grid = A_GRID
        n_grid = N_GRID
        tag = "produccion"

    print(f"=== F3-3 exponente de dilución — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"a_grid={a_grid.tolist()}")
    print(f"n_grid={n_grid}")

    payload = run_production(seeds, a_grid, n_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) — observable MAX ===")
    for seed_str, rec in payload["resultados_por_semilla"].items():
        ev = rec["evaluation_max"]
        slopes = ev["slope_by_n"]
        print(
            f"  seed={seed_str:>7}  null_bites={ev['null_bites']}  "
            f"mono_in_n={ev['mono_in_n']}  n3_singular={ev['n3_singular']}  "
            f"slope(n=0..5)=[" + ", ".join(f"{slopes[str(n)]:.4f}" for n in N_GRID) + "]  "
            f"seed_pass={ev['seed_pass']}"
        )
    resumen_max = payload["resumen"]["observable_max"]
    resumen_rms = payload["resumen"]["observable_rms"]
    print(f"\nrate_max={resumen_max['rate']:.3f}  verdict_max={resumen_max['verdict']}")
    print(f"rate_rms={resumen_rms['rate']:.3f}  verdict_rms={resumen_rms['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")

    out_json = OUT_DIR / f"F3_3_exponente_dilucion_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
