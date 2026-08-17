#!/usr/bin/env python3
"""
F3_4_reversibilidad_termica_motor.py — BATERÍA DE FUNDAMENTOS, Enfoque 3, experimento F3-4

"Reversibilidad térmica: ¿se re-homogeneiza si se detiene la expansión?"

Pregunta (BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md, sección F3-4): si detenemos la expansión
(`a` fijo desde ese instante en adelante) y dejamos correr solo difusión después, ¿el
gradiente térmico se re-aplana (la difusión lo alcanza) o queda congelado? ¿Existe un
"tiempo de no-retorno"?

Hereda SIN MODIFICAR el sustrato físico de CF2_estiramiento_motor.py (L=64, H_EXP=6.0,
RHO0=1.0, D0=0.12, W0=1.2, DT=0.25, N_SUB=2, ORIGINAL_STEPS_PER_TG=399, ley de dilución
REAL rho=rho0/a**3, D=D0/a**3). Ese resultado (el transporte se apaga al expandirse) es el
PRESUPUESTO de este experimento, no se re-litiga aquí (T1: no se retoca para favorecer el
resultado).

Diseño (protocolo fechado y congelado ANTES de este script — verificar mtime de
PROTOCOLO_F3-4_PREREGISTRO.md < mtime de este archivo):

  1) Fase común de expansión: se integra desde t_g=0 hasta t_g_max=ln(1000)/H_EXP,
     muestreando (checkpointing markoviano, mismo truco que CF2) el campo EXACTO en cada
     punto de parada del barrido de `a`.
  2) Desde cada checkpoint se bifurca en DOS ramas que parten del MISMO campo:
       STOP     : `a` queda fijo (D queda fijo) — solo difusión — durante POST_STOP_TG.
       NULL     : "nunca parar" — la expansión CONTINÚA (D sigue cayendo) — durante la
                  MISMA ventana POST_STOP_TG. Es el control pre-registrado por el
                  documento autoritativo.
  3) POST_STOP_TG es UNA constante, la misma para los 9 puntos de parada (si no, la
     comparación entre puntos de parada no sería justa).
  4) Perturbación dinámica (regla T7 general, no solo semilla): ruido gaussiano aditivo en
     cada sub-paso de difusión, amplitud sigma barrida en {0.0, 1e-3, 5e-3}. sigma=0.0
     reproduce EXACTAMENTE la física de CF2 sin modificarla.
  5) Dos observables independientes (T2): (a) gradiente comóvil máximo en banda central
     (idéntico a CF2), (b) varianza espacial global del campo (medidor completamente
     distinto, no usa derivadas).

Este script NO se auto-adjudica el veredicto de la hipótesis más amplia. Entrega números
crudos; la adjudicación es de CS después.

No edita CF2_estiramiento_motor.py ni ningún archivo de otras carpetas F3_* en paralelo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado de CF2_estiramiento_motor.py (idéntico, T1: no se retoca)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2           # ancho comóvil inicial del salto (celdas)
DT = 0.25          # subpaso de difusión
N_SUB = 2          # subiteraciones de difusión por paso de reloj genético
ORIGINAL_STEPS_PER_TG = 399
DTG = 1.0 / ORIGINAL_STEPS_PER_TG

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F3-4_PREREGISTRO.md, secciones 3-4)
# ============================================================
A_STOP_GRID = np.geomspace(1.0, 1000.0, 9)          # 9 puntos de parada, mismo rango que CF2
TG_STOP_GRID = np.log(A_STOP_GRID) / H_EXP
TG_MAX = float(np.log(1000.0) / H_EXP)              # duración de TODA la fase de expansión
POST_STOP_TG = TG_MAX                                # ventana post-parada = misma duración

SEEDS_STANDARD_PROJECT = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_EXTENSION = [271828, 161803]  # dígitos de e y de phi (misma extensión que F3-3)
SEEDS = SEEDS_STANDARD_PROJECT + SEEDS_EXTENSION     # 12 semillas (≥12 exigidas)

RUIDO_DINAMICO_GRID = [0.0, 1e-3, 5e-3]              # perturbación DINÁMICA (T7), no solo semilla

# ============================================================
# Criterio de PASS pre-registrado (protocolo, sección 6)
# ============================================================
MONO_TOL = 0.05
REHOMOG_EARLY_MIN = 0.5
REHOMOG_LATE_MAX = 0.1
DIFF_MIN = 0.1
PASS_RATE_MIN = 0.55

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "F3_4_reversibilidad_termica"

PROTOCOLO_PATH = CODE_DIR / "PROTOCOLO_F3-4_PREREGISTRO.md"


# ============================================================
# Física (heredada de CF2, con ruido dinámico opcional añadido)
# ============================================================
def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y)."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Abruptness comóvil y física, banda central (evita wrap-around periódico). Idéntico a CF2."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


def var_metric(T: np.ndarray) -> float:
    """Segundo observable, independiente del gradiente: varianza espacial global (Método 2)."""
    return float(np.var(T))


def diffuse_step(T: np.ndarray, D: float, dt: float, n_sub: int,
                  rng: np.random.Generator | None, noise_amp: float) -> np.ndarray:
    """Un paso de reloj genético de difusión (N_SUB sub-iteraciones), con ruido dinámico
    aditivo OPCIONAL (Euler-Maruyama: escala por sqrt(dt/n_sub)). noise_amp=0.0 reproduce
    EXACTAMENTE diffuse() de CF2_estiramiento_motor.py sin modificación."""
    if D <= 0 and noise_amp <= 0:
        return T
    out = T
    sub_dt = dt / n_sub
    for _ in range(n_sub):
        if D > 0:
            lap = (
                np.roll(out, -1, 1)
                + np.roll(out, 1, 1)
                + np.roll(out, -1, 0)
                + np.roll(out, 1, 0)
                - 4.0 * out
            )
            out = out + sub_dt * D * lap
        if noise_amp > 0 and rng is not None:
            out = out + noise_amp * np.sqrt(sub_dt) * rng.normal(size=out.shape)
    return out


def integrate_common_phase(seed: int, noise_amp: float, tg_stop_grid: np.ndarray):
    """Fase común REAL (idéntica a CF2 REAL): D=D0/a**3, desde t_g=0 hasta max(tg_stop_grid).
    Devuelve, por cada t_g de parada, un checkpoint {T (copia), a, tg, A_comov, Var}."""
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    tg_max = float(tg_stop_grid[-1])
    n_steps = max(int(np.ceil(tg_max / DTG)), 1)

    checkpoints = []
    next_idx = 0

    def record(tg_now, a_now, T_now):
        m = grad_metrics(T_now, a_now)
        checkpoints.append(
            {
                "tg": float(tg_now),
                "a": float(a_now),
                "T": T_now.copy(),
                "A_comov": m["A_comov"],
                "Var": var_metric(T_now),
            }
        )

    if tg_stop_grid[0] <= 1e-15:
        record(0.0, float(np.exp(H_EXP * 0.0)), T)
        next_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * DTG
        a = float(np.exp(H_EXP * tg))
        D = D0 / (a ** 3)
        T = diffuse_step(T, D, DT, N_SUB, rng, noise_amp)
        T = np.clip(T, 0.0, 1.0)

        while next_idx < len(tg_stop_grid) and tg >= tg_stop_grid[next_idx] - 1e-9:
            record(tg, a, T)
            next_idx += 1

    while next_idx < len(tg_stop_grid):
        tg_target = float(tg_stop_grid[next_idx])
        a_target = float(np.exp(H_EXP * tg_target))
        record(tg_target, a_target, T)
        next_idx += 1

    return checkpoints


def integrate_branch_stop(T0: np.ndarray, a_stop: float, post_stop_tg: float,
                           rng: np.random.Generator, noise_amp: float) -> np.ndarray:
    """Rama STOP: a queda fijo en a_stop (D fijo en D0/a_stop**3) durante post_stop_tg."""
    D_fixed = D0 / (a_stop ** 3)
    n_steps = max(int(np.ceil(post_stop_tg / DTG)), 1)
    T = T0
    for _ in range(n_steps):
        T = diffuse_step(T, D_fixed, DT, N_SUB, rng, noise_amp)
        T = np.clip(T, 0.0, 1.0)
    return T


def integrate_branch_null(T0: np.ndarray, tg_stop: float, post_stop_tg: float,
                           rng: np.random.Generator, noise_amp: float) -> tuple[np.ndarray, float]:
    """Rama NULL ('nunca parar'): la expansión REAL continúa sin frenar durante post_stop_tg."""
    n_steps = max(int(np.ceil(post_stop_tg / DTG)), 1)
    T = T0
    tg = tg_stop
    for _ in range(n_steps):
        tg = tg + DTG
        a = float(np.exp(H_EXP * tg))
        D = D0 / (a ** 3)
        T = diffuse_step(T, D, DT, N_SUB, rng, noise_amp)
        T = np.clip(T, 0.0, 1.0)
    a_final = float(np.exp(H_EXP * tg))
    return T, a_final


def run_combo(seed: int, noise_amp: float, a_stop_grid: np.ndarray, tg_stop_grid: np.ndarray,
              post_stop_tg: float) -> dict:
    checkpoints = integrate_common_phase(seed, noise_amp, tg_stop_grid)

    per_point = []
    for idx, ckpt in enumerate(checkpoints):
        a_stop = ckpt["a"]
        tg_stop = ckpt["tg"]
        T0 = ckpt["T"]
        Acomov_stop = ckpt["A_comov"]
        Var_stop = ckpt["Var"]

        rng_stop = np.random.default_rng([seed, idx, 1])
        T_stop_final = integrate_branch_stop(T0, a_stop, post_stop_tg, rng_stop, noise_amp)
        m_stop = grad_metrics(T_stop_final, a_stop)
        Var_stop_final = var_metric(T_stop_final)

        rng_null = np.random.default_rng([seed, idx, 2])
        T_null_final, a_final_null = integrate_branch_null(T0, tg_stop, post_stop_tg, rng_null, noise_amp)
        m_null = grad_metrics(T_null_final, a_final_null)
        Var_null_final = var_metric(T_null_final)

        def reaplan(before, after):
            if before <= 1e-15:
                return float("nan")
            return float((before - after) / before)

        per_point.append(
            {
                "a_stop": float(a_stop),
                "tg_stop": float(tg_stop),
                "A_comov_stop": float(Acomov_stop),
                "Var_stop": float(Var_stop),
                "STOP": {
                    "A_comov_final": m_stop["A_comov"],
                    "Var_final": Var_stop_final,
                    "reaplan_grad": reaplan(Acomov_stop, m_stop["A_comov"]),
                    "reaplan_var": reaplan(Var_stop, Var_stop_final),
                },
                "NULL_never_stop": {
                    "a_final": a_final_null,
                    "A_comov_final": m_null["A_comov"],
                    "Var_final": Var_null_final,
                    "reaplan_grad": reaplan(Acomov_stop, m_null["A_comov"]),
                    "reaplan_var": reaplan(Var_stop, Var_null_final),
                },
            }
        )

    return {"seed": seed, "noise_amp": noise_amp, "per_point": per_point}


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] + tol:
            return False
    return True


def find_no_retorno(a_stop_grid: np.ndarray, reaplan_stop: np.ndarray) -> float | None:
    """Primer a_stop (ascendente) donde reaplan_stop <= REHOMOG_LATE_MAX de forma SOSTENIDA
    (no vuelve a subir por encima del umbral en ningún punto posterior)."""
    n = len(reaplan_stop)
    for i in range(n):
        if reaplan_stop[i] <= REHOMOG_LATE_MAX and np.all(reaplan_stop[i:] <= REHOMOG_LATE_MAX):
            return float(a_stop_grid[i])
    return None


def evaluate_combo(combo: dict) -> dict:
    a_vals = np.array([p["a_stop"] for p in combo["per_point"]])
    reaplan_stop = np.array([p["STOP"]["reaplan_grad"] for p in combo["per_point"]])
    reaplan_null = np.array([p["NULL_never_stop"]["reaplan_grad"] for p in combo["per_point"]])

    cond_a = monotonic_nonincreasing(reaplan_stop)
    cond_b = bool(reaplan_stop[0] >= REHOMOG_EARLY_MIN)
    cond_c = bool(reaplan_stop[-1] <= REHOMOG_LATE_MAX)
    diff_early = float(reaplan_stop[0] - reaplan_null[0])
    cond_d = bool(diff_early >= DIFF_MIN)

    seed_pass = bool(cond_a and cond_b and cond_c and cond_d)
    no_retorno_a = find_no_retorno(a_vals, reaplan_stop)
    no_retorno_tg = (float(np.log(no_retorno_a) / H_EXP) if no_retorno_a is not None else None)

    # segundo observable (varianza) — comparación cualitativa, no gate
    reaplan_stop_var = np.array([p["STOP"]["reaplan_var"] for p in combo["per_point"]])
    no_retorno_a_var = find_no_retorno(a_vals, reaplan_stop_var)

    return {
        "cond_a_monotonic": cond_a,
        "cond_b_early_rehomog": cond_b,
        "cond_c_late_frozen": cond_c,
        "cond_d_null_bites": cond_d,
        "diff_early_stop_minus_null": diff_early,
        "seed_pass": seed_pass,
        "no_retorno_a_grad": no_retorno_a,
        "no_retorno_tg_grad": no_retorno_tg,
        "no_retorno_a_var": no_retorno_a_var,
        "reaplan_stop_grad_curve": reaplan_stop.tolist(),
        "reaplan_null_grad_curve": reaplan_null.tolist(),
        "reaplan_stop_var_curve": reaplan_stop_var.tolist(),
    }


def run_production(seeds: list[int], noise_grid: list[float], a_stop_grid: np.ndarray,
                    tg_stop_grid: np.ndarray, post_stop_tg: float, tag: str) -> dict:
    t0 = time.time()
    combos = {}
    n_pass = 0
    n_total = 0
    for seed in seeds:
        for noise_amp in noise_grid:
            combo = run_combo(seed, noise_amp, a_stop_grid, tg_stop_grid, post_stop_tg)
            ev = evaluate_combo(combo)
            key = f"seed{seed}_noise{noise_amp}"
            combos[key] = {"seed": seed, "noise_amp": noise_amp, "combo": combo, "evaluation": ev}
            n_total += 1
            if ev["seed_pass"]:
                n_pass += 1

    rate = n_pass / n_total if n_total else 0.0
    verdict_label = "F3_4_PASS" if rate >= PASS_RATE_MIN else "F3_4_FAIL"

    payload = {
        "experimento": "F3-4 reversibilidad térmica (¿se re-homogeneiza al detener la expansión?)",
        "tag": tag,
        "sello_heredado_CF2": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0, "DT": DT,
            "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_stop_grid": a_stop_grid.tolist(),
            "tg_stop_grid": tg_stop_grid.tolist(),
            "post_stop_tg": post_stop_tg,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "ruido_dinamico_grid": noise_grid,
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "REHOMOG_EARLY_MIN": REHOMOG_EARLY_MIN,
            "REHOMOG_LATE_MAX": REHOMOG_LATE_MAX,
            "DIFF_MIN": DIFF_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
        },
        "resultados_por_combo": combos,
        "resumen": {
            "n_combos_pass": n_pass,
            "n_combos_total": n_total,
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

    proto_mtime = PROTOCOLO_PATH.stat().st_mtime if PROTOCOLO_PATH.exists() else None
    this_mtime = Path(__file__).resolve().stat().st_mtime
    if proto_mtime is not None and proto_mtime > this_mtime:
        print("ADVERTENCIA: el protocolo tiene mtime posterior a este script (inesperado).")

    if args.mode == "smoke":
        seeds = SEEDS[:3]
        noise_grid = RUIDO_DINAMICO_GRID[:2]
        a_stop_grid = np.geomspace(1.0, 1000.0, 4)
        tg_stop_grid = np.log(a_stop_grid) / H_EXP
        post_stop_tg = TG_MAX
        tag = "smoke"
    else:
        seeds = SEEDS
        noise_grid = RUIDO_DINAMICO_GRID
        a_stop_grid = A_STOP_GRID
        tg_stop_grid = TG_STOP_GRID
        post_stop_tg = POST_STOP_TG
        tag = "produccion"

    print(f"=== F3-4 reversibilidad térmica — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"a_stop_grid={a_stop_grid.tolist()}")
    print(f"ruido_dinamico_grid={noise_grid}")
    print(f"post_stop_tg={post_stop_tg:.6f} (== duración total de la fase de expansión)")

    payload = run_production(seeds, noise_grid, a_stop_grid, tg_stop_grid, post_stop_tg, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    for key, rec in payload["resultados_por_combo"].items():
        ev = rec["evaluation"]
        print(
            f"  {key:>22}  cond_a={ev['cond_a_monotonic']}  cond_b={ev['cond_b_early_rehomog']}  "
            f"cond_c={ev['cond_c_late_frozen']}  cond_d={ev['cond_d_null_bites']}  "
            f"diff_early={ev['diff_early_stop_minus_null']:.4f}  "
            f"no_retorno_a={ev['no_retorno_a_grad']}  seed_pass={ev['seed_pass']}"
        )
    print(f"\nrate={payload['resumen']['rate']:.3f}  verdict={payload['resumen']['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")
    print(f"runtime_seconds={payload['runtime_seconds']:.2f}")

    out_json = OUT_DIR / f"F3_4_reversibilidad_termica_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
