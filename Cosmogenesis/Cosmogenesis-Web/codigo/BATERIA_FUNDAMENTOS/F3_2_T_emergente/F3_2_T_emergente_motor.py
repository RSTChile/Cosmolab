#!/usr/bin/env python3
"""
F3_2_T_emergente_motor.py — BATERÍA FUNDAMENTOS, Enfoque 3, experimento F3-2

"Enfriamiento como consecuencia medida, no impuesta: T leída del estado"

Pregunta: ¿la TEMPERATURA —leída directamente del propio estado del campo
(varianza espacial / energía cuadrática de gradiente de sus fluctuaciones),
SIN ningún término de enfriamiento puesto a mano— cae con el factor de
expansión `a` de forma EMERGENTE? Si cae, ¿sigue una ley de potencia
T(a) ~ a^(-n), midiendo n sin imponerlo?

Protocolo congelado ANTES de este motor: PROTOCOLO_F3-2_PREREGISTRO.md
(mismo directorio; verificar mtime < mtime de este archivo).

Diferencia deliberada con F3-1/CF2 (evita T2 — observable ≠ juez, y campo
≠ campo de F3-1):
  - Campo inicial: ruido gaussiano blanco de banda ancha (SIN salto/tanh
    macroscópico sembrado). No hay estructura de diseño que suavizar; el
    campo ES la fluctuación cuya "temperatura" se lee.
  - Sin clipping T∈[0,1] (a diferencia de CF2/F3-1): recortar introduciría
    un sesgo no lineal dependiente de sigma_ruido que contaminaría la
    propia medición de varianza/energía (violaría T2). Documentado en el
    protocolo, no se retoca aquí.
  - Forzamiento estocástico DINÁMICO en cada subpaso de difusión (no solo
    condición inicial) — necesario porque la ecuación de difusión es
    LINEAL: sin forzamiento, escalar sigma_ruido solo re-escala la curva
    sin cambiar forma/pendiente (perturbación cosmética, no dinámica,
    exactamente el defecto T7 que F3-1 identificó). El forzamiento está
    atado a la MISMA dilución que gobierna D (no es un parámetro libre
    nuevo elegido para dar un resultado, T1): decae con rho en REAL,
    constante en NULL_RHO_FIXED. NO es un baño externo (no empuja hacia
    un valor objetivo fijo T_bano; ver protocolo sección 3 para el
    razonamiento de por qué esto no colisiona con el control prohibido
    de F3-6).
  - Dos observables independientes (T2): T_energy (energía cuadrática de
    gradiente físico, con el /a^2 geométrico) y T_var (varianza espacial
    cruda, SIN división por a — verificación cruzada honesta: si sólo
    uno de los dos cae con la expansión, se reporta como hallazgo, no se
    fuerza el acuerdo).

Reutiliza SOLO la infraestructura de reloj genético de CF2
(CF2_estiramiento_motor.py: t_g -> a = exp(H_EXP*t_g), mismo dtg, mismo
mecanismo de checkpointing markoviano de una sola trayectoria muestreada
en los t_g(a) objetivo) y el patrón de brazos REAL vs NULL_RHO_FIXED
(misma trayectoria a(t), difiere solo densidad/difusividad/forzamiento).
NO reutiliza el perfil inicial (tanh) ni el observable (abruptancia
máxima de banda) de CF2/F3-1 — son propios de este experimento.

No edita CF2_estiramiento_motor.py, F3_1_estiramiento_motor.py, ni ningún
otro archivo existente. Prefijo exclusivo F3_2_.

Este script NO se auto-adjudica el veredicto de la batería. Entrega
números crudos; la adjudicación final es de CS (Alexis) después.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado de CF2 (idéntico, T1: no se retoca para
# favorecer el resultado) — SOLO el reloj genético y L
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
DT = 0.25          # subpaso de difusión, idéntico a CF2, sin retocar
N_SUB = 2           # subiteraciones de difusión por paso de reloj genético
ORIGINAL_STEPS_PER_TG = 399   # idéntico a CF2/TEST_RHO_DISPERSION.py

# ============================================================
# Parámetros PROPIOS de F3-2 (pre-registrados, PROTOCOLO_F3-2_PREREGISTRO.md)
# ============================================================
DYN_FRAC = 0.1      # fracción del forzamiento dinámico respecto a sigma_ruido
                     # (constante técnica fija, NO barrida, análoga a DT/N_SUB)

A_GRID = np.geomspace(1.0, 1.0e4, 14)          # 14 puntos, 4 décadas
SIGMA_RUIDO_GRID = np.geomspace(1.0e-4, 1.0e-1, 8)  # 8 puntos, 3 décadas

SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_EXTRA = [13, 271828, 161803, 31415, 90210, 20260724]
SEEDS_16 = SEEDS_STANDARD + SEEDS_EXTRA

# ============================================================
# Criterio de PASS pre-registrado (sección 7 del protocolo)
# ============================================================
MONO_TOL = 1e-9           # idéntico a CF2/F3-1
SLOPE_DIFF_MIN = 0.05      # idéntico a CF2/F3-1
R2_MIN = 0.70              # NUEVO, propio de F3-2 (ley de potencia)
PASS_RATE_MIN = 0.55       # idéntico a CF2/F3-1

MODES = ["REAL", "NULL_RHO_FIXED"]

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "F3_2_T_emergente"


# ============================================================
# Física
# ============================================================
def initial_noise_field(L: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Campo de ruido gaussiano blanco puro, media 0, SIN estructura
    macroscópica sembrada (a diferencia de CF2/F3-1)."""
    return sigma * rng.normal(size=(L, L))


def grad_energy_comov(T: np.ndarray) -> float:
    """<(dT/dx)^2 + (dT/dy)^2> comóvil, malla completa (campo homogéneo de
    ruido, sin artefacto de borde localizado que evitar con banda)."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    dTy = 0.5 * (np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0))
    return float(np.mean(dTx**2 + dTy**2))


def diffuse_forced(
    T: np.ndarray, D: float, forcing_amp: float, dt: float, n_sub: int, rng: np.random.Generator
) -> np.ndarray:
    """Difusión + forzamiento estocástico dinámico (Euler-Maruyama), cada
    subpaso. Sin clipping (ver protocolo, sección 3)."""
    out = T
    dt_sub = dt / n_sub
    sqrt_dt_sub = np.sqrt(dt_sub)
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
        if forcing_amp > 0:
            out = out + forcing_amp * sqrt_dt_sub * rng.normal(size=out.shape)
    return out


def run_sweep(mode: str, seed: int, sigma_ruido: float, a_grid: np.ndarray) -> dict:
    """Integra difusión+forzamiento desde t_g=0, muestreando el estado del
    campo EXACTAMENTE en los instantes t_g(a)=ln(a)/H_EXP de cada punto del
    barrido de `a` (mismo método de checkpointing markoviano que CF2/F3-1:
    una única trayectoria, sin re-simular desde cero por punto)."""
    rng = np.random.default_rng(seed)
    T = initial_noise_field(L, sigma_ruido, rng)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        e_comov = grad_energy_comov(T)
        t_energy = e_comov / max(a_now, 1e-12) ** 2
        t_var = float(np.var(T))
        checkpoints.append(
            {
                "a": float(a_now),
                "tg": float(tg_now),
                "grad_energy_comov": e_comov,
                "T_energy": t_energy,
                "T_var": t_var,
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
            forcing_amp = sigma_ruido * DYN_FRAC
        else:  # REAL
            rho = RHO0 / (a**3)
            D = D0 * (rho / RHO0)                     # = D0 / a^3
            forcing_amp = sigma_ruido * DYN_FRAC * np.sqrt(rho / RHO0)  # = sigma*DYN_FRAC*a^-1.5

        T = diffuse_forced(T, D, forcing_amp, DT, N_SUB, rng)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    t_energy = np.array([c["T_energy"] for c in checkpoints])
    t_var = np.array([c["T_var"] for c in checkpoints])

    return {
        "mode": mode,
        "seed": seed,
        "sigma_ruido": sigma_ruido,
        "a_grid": a_vals.tolist(),
        "T_energy": t_energy.tolist(),
        "T_var": t_var.tolist(),
    }


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] * (1.0 + tol) + tol:
            return False
    return True


def loglog_fit(a_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float, float]:
    """Ajuste log-log por mínimos cuadrados. Devuelve (pendiente, intercepto, R^2).
    Valores no positivos se recortan a un piso pequeño para poder tomar log
    (solo para evitar -inf; no altera el signo/monotonicidad del veredicto)."""
    x = np.log(a_vals)
    y = np.log(np.clip(y_vals, 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def evaluate_observable(a_vals: np.ndarray, y_real: np.ndarray, y_null: np.ndarray) -> dict:
    mono_real = monotonic_nonincreasing(y_real)
    mono_null = monotonic_nonincreasing(y_null)
    slope_real, intercept_real, r2_real = loglog_fit(a_vals, y_real)
    slope_null, intercept_null, r2_null = loglog_fit(a_vals, y_null)
    slope_diff = abs(slope_null - slope_real)

    # diagnóstico informativo (NO criterio de PASS): ajuste solo sobre el
    # tramo asintótico (últimos 60% de puntos de a), ver protocolo sección 7
    n = len(a_vals)
    k = max(int(np.ceil(0.4 * n)), 2)  # arranca al 40% del recorrido -> últimos 60%
    slope_real_asym, _, r2_real_asym = loglog_fit(a_vals[k:], y_real[k:])

    cond_mono = mono_real
    cond_r2 = r2_real >= R2_MIN
    cond_null_bites = (not mono_null) or (slope_diff >= SLOPE_DIFF_MIN)
    punto_pass = bool(cond_mono and cond_r2 and cond_null_bites)

    return {
        "mono_REAL": bool(mono_real),
        "mono_NULL_RHO_FIXED": bool(mono_null),
        "slope_REAL": slope_real,
        "slope_NULL_RHO_FIXED": slope_null,
        "R2_REAL": r2_real,
        "R2_NULL_RHO_FIXED": r2_null,
        "slope_diff_abs": float(slope_diff),
        "slope_REAL_asintotico_diagnostico": slope_real_asym,
        "R2_REAL_asintotico_diagnostico": r2_real_asym,
        "cond_mono_real": bool(cond_mono),
        "cond_r2_real_min": bool(cond_r2),
        "cond_null_muerde": bool(cond_null_bites),
        "punto_pass": punto_pass,
    }


def run_production(seeds: list[int], sigma_grid: np.ndarray, a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    per_combo = {}
    n_pass_energy = 0
    n_pass_var = 0
    n_null_bites = 0
    n_agree = 0
    n_total = 0

    # curva de robustez P(sigma_ruido) para cada observable
    rate_by_sigma_energy = {}
    rate_by_sigma_var = {}

    for sigma in sigma_grid:
        sigma_key = f"{sigma:.6e}"
        n_pass_this_sigma_energy = 0
        n_pass_this_sigma_var = 0
        per_combo[sigma_key] = {}
        for seed in seeds:
            real = run_sweep("REAL", seed, float(sigma), a_grid)
            null = run_sweep("NULL_RHO_FIXED", seed, float(sigma), a_grid)

            a_vals = np.array(real["a_grid"])
            ev_energy = evaluate_observable(
                a_vals, np.array(real["T_energy"]), np.array(null["T_energy"])
            )
            ev_var = evaluate_observable(
                a_vals, np.array(real["T_var"]), np.array(null["T_var"])
            )

            per_combo[sigma_key][str(seed)] = {
                "REAL": real,
                "NULL_RHO_FIXED": null,
                "evaluation_T_energy": ev_energy,
                "evaluation_T_var": ev_var,
            }

            n_total += 1
            if ev_energy["punto_pass"]:
                n_pass_energy += 1
                n_pass_this_sigma_energy += 1
            if ev_var["punto_pass"]:
                n_pass_var += 1
                n_pass_this_sigma_var += 1
            if ev_energy["cond_null_muerde"]:
                n_null_bites += 1
            if ev_energy["punto_pass"] == ev_var["punto_pass"]:
                n_agree += 1

        rate_by_sigma_energy[sigma_key] = n_pass_this_sigma_energy / len(seeds)
        rate_by_sigma_var[sigma_key] = n_pass_this_sigma_var / len(seeds)

    rate_energy = n_pass_energy / n_total if n_total else 0.0
    rate_var = n_pass_var / n_total if n_total else 0.0
    rate_null_bites = n_null_bites / n_total if n_total else 0.0
    rate_agree = n_agree / n_total if n_total else 0.0

    verdict = "F3_2_PASS" if (rate_energy >= PASS_RATE_MIN and rate_null_bites >= PASS_RATE_MIN) else "F3_2_FAIL"

    # exponente reportado: media +/- std de slope_REAL (T_energy) a través de
    # TODAS las combinaciones (sigma, semilla) -- tal como salga, sin comparar
    # contra un valor esperado para decidir PASS/FAIL.
    all_slopes_energy = []
    all_slopes_energy_asym = []
    all_slopes_var = []
    for sigma_key, by_seed in per_combo.items():
        for seed_key, rec in by_seed.items():
            all_slopes_energy.append(rec["evaluation_T_energy"]["slope_REAL"])
            all_slopes_energy_asym.append(rec["evaluation_T_energy"]["slope_REAL_asintotico_diagnostico"])
            all_slopes_var.append(rec["evaluation_T_var"]["slope_REAL"])

    exponente_T_energy = {
        "media": float(np.mean(all_slopes_energy)),
        "std": float(np.std(all_slopes_energy)),
        "min": float(np.min(all_slopes_energy)),
        "max": float(np.max(all_slopes_energy)),
        "n": len(all_slopes_energy),
    }
    exponente_T_energy_asintotico_diagnostico = {
        "media": float(np.mean(all_slopes_energy_asym)),
        "std": float(np.std(all_slopes_energy_asym)),
        "min": float(np.min(all_slopes_energy_asym)),
        "max": float(np.max(all_slopes_energy_asym)),
        "n": len(all_slopes_energy_asym),
    }
    exponente_T_var = {
        "media": float(np.mean(all_slopes_var)),
        "std": float(np.std(all_slopes_var)),
        "min": float(np.min(all_slopes_var)),
        "max": float(np.max(all_slopes_var)),
        "n": len(all_slopes_var),
    }

    payload = {
        "experimento": "F3-2 T emergente (enfriamiento como consecuencia medida)",
        "tag": tag,
        "sello_heredado_CF2": {
            "L": L,
            "H_EXP": H_EXP,
            "RHO0": RHO0,
            "D0": D0,
            "DT": DT,
            "N_SUB": N_SUB,
            "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "parametros_propios_F3_2": {
            "DYN_FRAC": DYN_FRAC,
            "clipping": False,
            "campo_inicial": "ruido gaussiano blanco puro N(0, sigma_ruido^2), sin estructura macroscopica",
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "sigma_ruido_grid": sigma_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_sigma": len(sigma_grid),
            "n_total_combinaciones": n_total,
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN,
            "R2_MIN": R2_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "punto_pass = mono_REAL AND (R2_REAL >= R2_MIN) AND "
                "(NOT mono_NULL OR abs(slope_NULL - slope_REAL) >= SLOPE_DIFF_MIN); "
                "veredicto principal basado en T_energy; PASS_F3-2 si "
                "rate_T_energy>=PASS_RATE_MIN Y rate_null_muerde>=PASS_RATE_MIN"
            ),
        },
        "resultados_por_sigma_y_semilla": per_combo,
        "curva_robustez_P_sigma": {
            "T_energy": rate_by_sigma_energy,
            "T_var": rate_by_sigma_var,
        },
        "resumen": {
            "n_total_combinaciones": n_total,
            "n_pass_T_energy": n_pass_energy,
            "n_pass_T_var": n_pass_var,
            "rate_T_energy": rate_energy,
            "rate_T_var": rate_var,
            "rate_NULL_muerde": rate_null_bites,
            "rate_acuerdo_T_energy_vs_T_var": rate_agree,
            "exponente_T_energy_rango_completo": exponente_T_energy,
            "exponente_T_energy_asintotico_diagnostico": exponente_T_energy_asintotico_diagnostico,
            "exponente_T_var_rango_completo": exponente_T_var,
            "verdict": verdict,
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
        sigma_grid = SIGMA_RUIDO_GRID[::3]  # reducido
        a_grid = np.geomspace(1.0, 100.0, 5)
        tag = "smoke"
    else:
        seeds = SEEDS_16
        sigma_grid = SIGMA_RUIDO_GRID
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== F3-2 T emergente — modo={args.mode} ===")
    print(f"seeds(n={len(seeds)})={seeds}")
    print(f"sigma_ruido_grid={sigma_grid.tolist()}")
    print(f"a_grid={a_grid.tolist()}")

    payload = run_production(seeds, sigma_grid, a_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    r = payload["resumen"]
    print(f"n_total_combinaciones={r['n_total_combinaciones']}")
    print(f"rate_T_energy={r['rate_T_energy']:.3f}  rate_T_var={r['rate_T_var']:.3f}")
    print(f"rate_NULL_muerde={r['rate_NULL_muerde']:.3f}")
    print(f"rate_acuerdo_T_energy_vs_T_var={r['rate_acuerdo_T_energy_vs_T_var']:.3f}")
    e = r["exponente_T_energy_rango_completo"]
    print(f"exponente_T_energy (rango completo): media={e['media']:.4f} std={e['std']:.4f} "
          f"min={e['min']:.4f} max={e['max']:.4f} (n={e['n']})")
    ea = r["exponente_T_energy_asintotico_diagnostico"]
    print(f"exponente_T_energy (asintotico, diagnostico): media={ea['media']:.4f} std={ea['std']:.4f}")
    ev = r["exponente_T_var_rango_completo"]
    print(f"exponente_T_var (rango completo): media={ev['media']:.4f} std={ev['std']:.4f}")
    print(f"\nverdict={r['verdict']}  (umbrales pre-registrados PASS_RATE_MIN={PASS_RATE_MIN}, R2_MIN={R2_MIN})")

    print("\ncurva_robustez P(sigma_ruido) [T_energy]:")
    for k, v in payload["curva_robustez_P_sigma"]["T_energy"].items():
        print(f"  sigma={k}  P={v:.3f}")

    out_json = OUT_DIR / f"F3_2_T_emergente_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
