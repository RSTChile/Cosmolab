#!/usr/bin/env python3
"""
F3_1_estiramiento_motor.py — BATERÍA_FUNDAMENTOS, Enfoque 3, experimento F3-1

"Estiramiento geométrico del gradiente, barrido amplio de a + ruido dinámico"

Repara el defecto T7 encontrado en CF-2
(Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py,
resultado en Cosmogenesis-Web/results/CF2_estiramiento/CF2_estiramiento_produccion_result.json):
CF-2 dio PASS 10/10 semillas, pero las 10 semillas resultaron casi idénticas —
la PDE de difusión es cuasi-determinista y la amplitud de ruido inicial estaba
FIJA en 1e-4, insuficiente para perturbar la dinámica. Un "10/10" así es
T7 disfrazado de robustez (barre semilla, no dinámica).

Este motor NO edita CF2_estiramiento_motor.py. Reutiliza su mismo núcleo físico
(idéntico sello: L, H_EXP, RHO0, D0, W0, DT, N_SUB, reloj genético, laplaciano
de 5 puntos, salto tanh, banda central anti-wrap, observable ∇_fis=∇_comov/a,
NULL_RHO_FIXED) y añade EXACTAMENTE lo que pide F3-1:

  1) `sigma_ruido` (amplitud del ruido inicial) pasa de constante a parámetro
     BARRIDO explícito, ≥6 puntos log en [1e-4, 1e-1] (produce 8 en este motor).
  2) `a` se barre en ≥12 puntos log en [1, 1e4] (produce 13).
  3) ≥12 semillas (produce 16: las 10 estándar del proyecto + 6 nuevas).
  4) Extensión declarada en el pre-registro (PROTOCOLO_F3-1_PREREGISTRO.md §4):
     además de la condición inicial, se ofrece una variante con ruido gaussiano
     tipo Wiener inyectado EN CADA SUBPASO de difusión (perturbación dinámica
     real, no solo de arranque). Se reportan AMBAS variantes por separado;
     el veredicto principal usa la variante literal de la spec (ruido solo en
     la condición inicial, con su amplitud barrida).
  5) El PASS se lee de booleanos de Python (monotonicidad + pendiente log-log),
     nunca de texto. Criterio congelado en el pre-registro ANTES de este script.

No se auto-adjudica el hallazgo más amplio. Entrega números crudos; la
adjudicación es de CS. No topología. No commits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello físico heredado de CF2_estiramiento_motor.py, IDÉNTICO
# (T1: no se retoca nada del núcleo para favorecer un resultado)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2
DT = 0.25
N_SUB = 2
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F3-1_PREREGISTRO.md §4)
# ============================================================
A_GRID = np.geomspace(1.0, 1.0e4, 13)          # 13 puntos, 4 décadas (≥12 exigidos)
SIGMA_GRID = np.geomspace(1.0e-4, 1.0e-1, 8)    # 8 puntos, 3 décadas (≥6 exigidos)
SEEDS_STANDARD_PROJECT = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_NUEVAS_F3_1 = [13, 271828, 161803, 31415, 90210, 20260724]
SEEDS = SEEDS_STANDARD_PROJECT + SEEDS_NUEVAS_F3_1  # 16 semillas (≥12 exigidos)

NOISE_VARIANTS = ["solo_inicial", "inicial_y_dinamico"]
MODES = ["REAL", "NULL_RHO_FIXED"]

# ============================================================
# Criterio de PASS pre-registrado (PROTOCOLO_F3-1_PREREGISTRO.md §6)
# ============================================================
MONO_TOL = 1e-9
SLOPE_DIFF_MIN = 0.05      # idéntico a CF-2, no re-elegido (T1)
PASS_RATE_MIN = 0.55       # idéntico a CF-2, no re-elegido (T1)
SLOPE_TOL = 0.15           # banda alrededor de -1 (estiramiento puro)
SLOPE_TARGET = -1.0
NULL_BITE_MIN = 0.70       # NULL debe morder en >=70% de las combinaciones

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F3_1_estiramiento_ruido"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2_estiramiento_motor.py."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptness comóvil y física, banda central (evita wrap-around periódico).
    Idéntico a CF2_estiramiento_motor.py."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


def diffuse(T: np.ndarray, D: float, dt: float, n_sub: int,
            rng: np.random.Generator | None, sigma_dyn: float) -> np.ndarray:
    """Difusión con laplaciano de 5 puntos (idéntico a CF2). Si sigma_dyn>0 y rng
    no es None, inyecta ruido gaussiano tipo Wiener EN CADA SUBPASO — la
    perturbación DINÁMICA que F1-5/F3-1 piden y que CF-2 no tenía. Con
    sigma_dyn=0 el comportamiento es bit-a-bit idéntico a CF2_estiramiento_motor.py."""
    out = T
    dt_sub = dt / n_sub
    for _ in range(n_sub):
        if D > 0:
            lap = (
                np.roll(out, -1, 1)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 0)
                - 4.0 * out
            )
            out = out + dt_sub * D * lap
        if sigma_dyn > 0.0 and rng is not None:
            out = out + sigma_dyn * np.sqrt(dt_sub) * rng.normal(size=out.shape)
    return out


def run_sweep(mode: str, seed: int, a_grid: np.ndarray, sigma_ruido: float,
              noise_variant: str) -> dict:
    """Integra desde t_g=0, muestreando ∇_fis en los checkpoints t_g(a)=ln(a)/H_EXP.
    Misma lógica de checkpointing markoviano que CF2_estiramiento_motor.py
    (una sola trayectoria por semilla, muestreada en los `a` objetivo).

    `sigma_ruido` reemplaza la constante 1e-4 de CF-2 y se aplica siempre a la
    condición inicial. Si `noise_variant == "inicial_y_dinamico"`, la MISMA
    amplitud se inyecta también en cada subpaso de difusión (ver `diffuse`).
    """
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + sigma_ruido * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    dyn_sigma = sigma_ruido if noise_variant == "inicial_y_dinamico" else 0.0

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        m = grad_metrics(T, a_now)
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "A_comov": m["A_comov"],
                "A_phys": m["A_phys"],
            }
        )

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
            D = D0 * (rho / RHO0)

        T = diffuse(T, D, DT, N_SUB, rng, dyn_sigma)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

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
        "sigma_ruido": sigma_ruido,
        "noise_variant": noise_variant,
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


def evaluate_point(real: dict, null: dict) -> dict:
    a_vals = np.array(real["a_grid"])
    grad_real = np.array(real["grad_fis"])
    grad_null = np.array(null["grad_fis"])

    mono_real = monotonic_nonincreasing(grad_real)
    mono_null = monotonic_nonincreasing(grad_null)
    slope_real = loglog_slope(a_vals, grad_real)
    slope_null = loglog_slope(a_vals, grad_null)
    slope_diff = abs(slope_null - slope_real)

    null_bites = bool((not mono_null) or (slope_diff >= SLOPE_DIFF_MIN))
    cond_a = mono_real
    point_pass = bool(cond_a and null_bites)
    slope_near_minus1 = bool(abs(slope_real - SLOPE_TARGET) <= SLOPE_TOL)

    return {
        "mono_REAL": bool(mono_real),
        "mono_NULL_RHO_FIXED": bool(mono_null),
        "slope_REAL": slope_real,
        "slope_NULL_RHO_FIXED": slope_null,
        "slope_diff_abs": float(slope_diff),
        "null_bites": null_bites,
        "slope_near_minus1": slope_near_minus1,
        "point_pass": point_pass,
    }


def run_production(seeds: list[int], a_grid: np.ndarray, sigma_grid: np.ndarray,
                    noise_variants: list[str], tag: str) -> dict:
    t0 = time.time()

    # resultados_crudos[variant][str(sigma)][str(seed)] = {"REAL":..,"NULL":..,"evaluation":..}
    resultados_crudos: dict = {}
    # curva[variant][str(sigma)] = {n_pass, n_total, rate, slopes_real:[...], ...}
    curva: dict = {}

    for variant in noise_variants:
        resultados_crudos[variant] = {}
        curva[variant] = {}
        for sigma in sigma_grid:
            sigma_key = f"{sigma:.6e}"
            resultados_crudos[variant][sigma_key] = {}
            n_pass = 0
            n_null_bites = 0
            n_slope_near = 0
            slopes_real = []
            slopes_null = []
            for seed in seeds:
                real = run_sweep("REAL", seed, a_grid, float(sigma), variant)
                null = run_sweep("NULL_RHO_FIXED", seed, a_grid, float(sigma), variant)
                ev = evaluate_point(real, null)
                resultados_crudos[variant][sigma_key][str(seed)] = {
                    "REAL": real,
                    "NULL_RHO_FIXED": null,
                    "evaluation": ev,
                }
                if ev["point_pass"]:
                    n_pass += 1
                if ev["null_bites"]:
                    n_null_bites += 1
                if ev["slope_near_minus1"]:
                    n_slope_near += 1
                slopes_real.append(ev["slope_REAL"])
                slopes_null.append(ev["slope_NULL_RHO_FIXED"])

            n_total = len(seeds)
            curva[variant][sigma_key] = {
                "sigma_ruido": float(sigma),
                "n_seeds_pass": n_pass,
                "n_seeds_total": n_total,
                "rate": n_pass / n_total if n_total else 0.0,
                "null_bite_rate": n_null_bites / n_total if n_total else 0.0,
                "slope_near_minus1_rate": n_slope_near / n_total if n_total else 0.0,
                "slope_REAL_mean": float(np.mean(slopes_real)) if slopes_real else None,
                "slope_REAL_std": float(np.std(slopes_real)) if slopes_real else None,
                "slope_NULL_mean": float(np.mean(slopes_null)) if slopes_null else None,
                "slope_NULL_std": float(np.std(slopes_null)) if slopes_null else None,
            }

    # ------------------------------------------------------------
    # Veredicto (leído de la variante literal de la spec: solo_inicial)
    # ------------------------------------------------------------
    variante_principal = "solo_inicial"
    curva_principal = curva[variante_principal]
    rates = [pt["rate"] for pt in curva_principal.values()]
    null_bite_rates = [pt["null_bite_rate"] for pt in curva_principal.values()]
    slope_means = [pt["slope_REAL_mean"] for pt in curva_principal.values()]

    estable = bool(min(rates) >= PASS_RATE_MIN) if rates else False
    null_muerde_global = bool(np.mean(null_bite_rates) >= NULL_BITE_MIN) if null_bite_rates else False
    slope_media_global = float(np.mean(slope_means)) if slope_means else None
    slope_en_banda = bool(
        slope_media_global is not None
        and abs(slope_media_global - SLOPE_TARGET) <= (SLOPE_TOL + 0.10)  # banda global un poco más laxa que la puntual
    )

    if estable and null_muerde_global and slope_en_banda:
        verdict_label = "F3_1_PASS_ESTABLE"
    elif (not estable) and null_muerde_global:
        verdict_label = "F3_1_FAIL_INESTABLE_EN_RUIDO"
    elif not null_muerde_global:
        verdict_label = "F3_1_FAIL_NULL_NO_MUERDE"
    else:
        verdict_label = "F3_1_FAIL_PENDIENTE_FUERA_DE_BANDA"

    payload = {
        "experimento": "F3-1 estiramiento geométrico del gradiente, barrido amplio de a + ruido dinámico",
        "enfoque": "ENFOQUE 3 — ¿enfriar es expandir?",
        "tag": tag,
        "sello_fisico_heredado_de_CF2": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "n_a": len(a_grid),
            "sigma_ruido_grid": sigma_grid.tolist(),
            "n_sigma": len(sigma_grid),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "noise_variants": noise_variants,
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "SLOPE_TOL": SLOPE_TOL,
            "SLOPE_TARGET": SLOPE_TARGET,
            "NULL_BITE_MIN": NULL_BITE_MIN,
            "descripcion_punto": (
                "point_pass = mono_REAL AND (NOT mono_NULL OR "
                "abs(slope_NULL-slope_REAL) >= SLOPE_DIFF_MIN)"
            ),
            "descripcion_veredicto": (
                "PASS_ESTABLE si min_sigma(rate(sigma)) >= PASS_RATE_MIN (variante "
                "solo_inicial, literal de la spec) AND NULL muerde en promedio "
                ">= NULL_BITE_MIN AND slope_REAL medio dentro de banda alrededor de -1"
            ),
        },
        "curva_por_amplitud_de_ruido": curva,
        "resultados_crudos_por_semilla": resultados_crudos,
        "veredicto": {
            "variante_usada_para_veredicto": variante_principal,
            "min_rate_sobre_sigma": float(min(rates)) if rates else None,
            "rates_por_sigma": rates,
            "estable_en_ruido": estable,
            "null_muerde_global": null_muerde_global,
            "slope_REAL_media_global": slope_media_global,
            "slope_en_banda": slope_en_banda,
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
        seeds = SEEDS[:3]
        a_grid = np.geomspace(1.0, 100.0, 5)
        sigma_grid = np.geomspace(1.0e-4, 1.0e-1, 3)
        variants = NOISE_VARIANTS
        tag = "smoke"
    else:
        seeds = SEEDS
        a_grid = A_GRID
        sigma_grid = SIGMA_GRID
        variants = NOISE_VARIANTS
        tag = "produccion"

    print(f"=== F3-1 estiramiento+ruido — modo={args.mode} ===")
    print(f"seeds({len(seeds)})={seeds}")
    print(f"a_grid({len(a_grid)})={a_grid.tolist()}")
    print(f"sigma_ruido_grid({len(sigma_grid)})={sigma_grid.tolist()}")
    print(f"noise_variants={variants}")

    payload = run_production(seeds, a_grid, sigma_grid, variants, tag)

    print("\n=== CURVA P(sigma_ruido) — variante solo_inicial (veredicto principal) ===")
    for sigma_key, pt in payload["curva_por_amplitud_de_ruido"]["solo_inicial"].items():
        print(
            f"  sigma={pt['sigma_ruido']:.2e}  rate={pt['rate']:.3f}  "
            f"null_bite_rate={pt['null_bite_rate']:.3f}  "
            f"slope_REAL_mean={pt['slope_REAL_mean']:.4f}±{pt['slope_REAL_std']:.4f}  "
            f"slope_NULL_mean={pt['slope_NULL_mean']:.4f}"
        )

    print("\n=== CURVA P(sigma_ruido) — variante inicial_y_dinamico (extra, no reemplaza veredicto) ===")
    for sigma_key, pt in payload["curva_por_amplitud_de_ruido"]["inicial_y_dinamico"].items():
        print(
            f"  sigma={pt['sigma_ruido']:.2e}  rate={pt['rate']:.3f}  "
            f"null_bite_rate={pt['null_bite_rate']:.3f}  "
            f"slope_REAL_mean={pt['slope_REAL_mean']:.4f}±{pt['slope_REAL_std']:.4f}  "
            f"slope_NULL_mean={pt['slope_NULL_mean']:.4f}"
        )

    v = payload["veredicto"]
    print(f"\nmin_rate_sobre_sigma={v['min_rate_sobre_sigma']:.3f}  "
          f"estable_en_ruido={v['estable_en_ruido']}  "
          f"null_muerde_global={v['null_muerde_global']}  "
          f"slope_media_global={v['slope_REAL_media_global']:.4f}  "
          f"slope_en_banda={v['slope_en_banda']}")
    print(f"VERDICT={v['verdict']}")

    out_json = OUT_DIR / f"F3_1_estiramiento_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
