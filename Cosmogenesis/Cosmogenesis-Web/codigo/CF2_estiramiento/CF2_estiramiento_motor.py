#!/usr/bin/env python3
"""
CF2_estiramiento_motor.py — BATERÍA CF, experimento CF-2

"¿El enfriamiento por expansión suaviza el gradiente?"

Corrige el defecto T7 confirmado en el TEST_RHO_DISPERSION.py viejo
(codigo/test_rho_dispersion/TEST_RHO_DISPERSION.py): aquella corrida usaba
UNA sola semilla (SEED=2025) y CERO barrido de ningún parámetro — solo una
trayectoria temporal única de a=1→~403. Este motor:

  1) Trata `a` (factor de expansión) como parámetro BARRIDO explícito,
     log-espaciado en varias décadas (a ∈ {1, 3.16, 10, ..., 1000}).
  2) Corre sobre las 10 semillas estándar del proyecto (≥8 exigidas).
  3) Compara dos brazos que DEBEN poder diferir (NULL que muerde, T4):
       REAL          : ρ = ρ0/a³  (dilución real), D = D0·ρ/ρ0 = D0/a³
       NULL_RHO_FIXED: ρ ≡ ρ0     (sin dilución),  D ≡ D0
  4) El observable es ∇_fis(a) = ∇_comov(a) / a — geometría pura del campo
     T y de a, SIN ninguna variable de linaje/juez de otros experimentos (T2).
  5) El PASS se decide leyendo booleanos calculados en Python (monotonicidad
     + pendiente log-log), nunca por coincidencia de texto "PASS" in str (T7).
  6) El criterio de PASS está congelado en PROTOCOLO_CF-2_PREREGISTRO.md,
     escrito y con mtime ANTES de este script. No se toca aquí.

Este script NO se auto-adjudica "persiste/no persiste" la hipótesis más
amplia. Entrega números crudos; la adjudicación es de CS (Alexis) después.

No edita TEST_RHO_DISPERSION.py ni motor_1a7/pipeline.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado (idéntico a TEST_RHO_DISPERSION.py; T1: no se
# retoca para favorecer el resultado)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2           # ancho comóvil inicial del salto (celdas)
DT = 0.25          # subpaso de difusión (idéntico a TEST_RHO_DISPERSION.py, sin retocar)
N_SUB = 2          # subiteraciones de difusión por paso de reloj genético
# resolución del reloj genético: EXACTAMENTE la misma que el test viejo
# (PASOS_original=400 → 399 incrementos de t_g entre 0 y 1). Se extiende el
# número de pasos más allá de t_g=1 para alcanzar a=1000, pero el tamaño de
# paso dtg y el dt de difusión (DT) NO se retocan (T1).
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Barrido pre-registrado (PROTOCOLO_CF-2_PREREGISTRO.md, sección 3)
# ============================================================
A_GRID = np.geomspace(1.0, 1000.0, 7)   # 7 puntos, 3 décadas
SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]

# ============================================================
# Criterio de PASS pre-registrado (sección 6)
# ============================================================
MONO_TOL = 1e-9          # tolerancia numérica de monotonicidad
SLOPE_DIFF_MIN = 0.05    # separación mínima de pendiente log-log REAL vs NULL
PASS_RATE_MIN = 0.55     # umbral de tasa de semillas en PASS

MODES = ["REAL", "NULL_RHO_FIXED"]

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[1]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "CF2_estiramiento"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y)."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptness comóvil y física, banda central (evita wrap-around periódico)."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


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


def run_sweep(mode: str, seed: int, a_grid: np.ndarray) -> dict:
    """
    Integra la difusión desde t_g=0 y muestrea el estado del campo EXACTAMENTE
    en los instantes t_g(a)=ln(a)/H_EXP de cada punto del barrido de `a`.
    Como la actualización de difusión es una cadena markoviana sin look-ahead,
    muestrear checkpoints de una única trayectoria es idéntico a re-simular
    de cero hasta cada t_g objetivo (misma condición inicial y misma semilla).
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
                "A_comov": m["A_comov"],
                "A_phys": m["A_phys"],  # == grad_fis = grad_comov / a
            }
        )

    # checkpoint inicial (a = a_grid[0] = 1.0, tg=0)
    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        if mode == "NULL_RHO_FIXED":
            rho = RHO0
            D = D0
        else:  # REAL
            rho = RHO0 / (a**3)
            D = D0 * (rho / RHO0)  # = D0 / a^3

        # dt de difusión = DT (idéntico, sin escalar, al test viejo)
        T = diffuse(T, D, DT, N_SUB)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    # por si el bucle terminó sin cubrir todos los checkpoints por redondeo
    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    grad_fis = np.array([c["A_phys"] for c in checkpoints])
    grad_comov = np.array([c["A_comov"] for c in checkpoints])

    return {
        "mode": mode,
        "seed": seed,
        "a_grid": a_vals.tolist(),
        "grad_fis": grad_fis.tolist(),
        "grad_comov": grad_comov.tolist(),
    }


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] * (1.0 + tol):
            return False
    return True


def loglog_slope(a_vals: np.ndarray, grad_vals: np.ndarray) -> float:
    x = np.log(a_vals)
    y = np.log(np.clip(grad_vals, 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def evaluate_seed(real: dict, null: dict) -> dict:
    a_vals = np.array(real["a_grid"])
    grad_real = np.array(real["grad_fis"])
    grad_null = np.array(null["grad_fis"])

    mono_real = monotonic_nonincreasing(grad_real)
    mono_null = monotonic_nonincreasing(grad_null)
    slope_real = loglog_slope(a_vals, grad_real)
    slope_null = loglog_slope(a_vals, grad_null)
    slope_diff = abs(slope_null - slope_real)

    cond_a = mono_real
    cond_b = (not mono_null) or (slope_diff >= SLOPE_DIFF_MIN)
    seed_pass = bool(cond_a and cond_b)

    return {
        "mono_REAL": bool(mono_real),
        "mono_NULL_RHO_FIXED": bool(mono_null),
        "slope_REAL": slope_real,
        "slope_NULL_RHO_FIXED": slope_null,
        "slope_diff_abs": float(slope_diff),
        "cond_a_real_monotonic": bool(cond_a),
        "cond_b_null_differs": bool(cond_b),
        "seed_pass": seed_pass,
    }


def run_production(seeds: list[int], a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    per_seed = {}
    n_pass = 0
    for seed in seeds:
        real = run_sweep("REAL", seed, a_grid)
        null = run_sweep("NULL_RHO_FIXED", seed, a_grid)
        ev = evaluate_seed(real, null)
        per_seed[str(seed)] = {
            "REAL": real,
            "NULL_RHO_FIXED": null,
            "evaluation": ev,
        }
        if ev["seed_pass"]:
            n_pass += 1

    rate = n_pass / len(seeds) if seeds else 0.0
    verdict_label = "CF2_PASS" if rate >= PASS_RATE_MIN else "CF2_FAIL"

    payload = {
        "experimento": "CF-2 estiramiento por expansión",
        "tag": tag,
        "sello": {
            "L": L,
            "H_EXP": H_EXP,
            "RHO0": RHO0,
            "D0": D0,
            "W0": W0,
            "DT": DT,
            "N_SUB": N_SUB,
            "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "seed_pass = monotonic_nonincreasing(grad_fis_REAL) AND "
                "(NOT monotonic_nonincreasing(grad_fis_NULL) OR "
                "abs(slope_NULL - slope_REAL) >= SLOPE_DIFF_MIN)"
            ),
        },
        "resultados_por_semilla": per_seed,
        "resumen": {
            "n_seeds_pass": n_pass,
            "n_seeds_total": len(seeds),
            "rate": rate,
            "verdict": verdict_label,
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
        a_grid = np.geomspace(1.0, 10.0, 3)  # barrido reducido para smoke
        tag = "smoke"
    else:
        seeds = SEEDS_STANDARD
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== CF-2 estiramiento — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"a_grid={a_grid.tolist()}")

    payload = run_production(seeds, a_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    for seed_str, rec in payload["resultados_por_semilla"].items():
        ev = rec["evaluation"]
        print(
            f"  seed={seed_str:>6}  mono_REAL={ev['mono_REAL']}  "
            f"mono_NULL={ev['mono_NULL_RHO_FIXED']}  "
            f"slope_REAL={ev['slope_REAL']:.4f}  slope_NULL={ev['slope_NULL_RHO_FIXED']:.4f}  "
            f"|Δslope|={ev['slope_diff_abs']:.4f}  seed_pass={ev['seed_pass']}"
        )
    print(f"\nrate={payload['resumen']['rate']:.3f}  verdict={payload['resumen']['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")

    out_json = OUT_DIR / f"CF2_estiramiento_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
