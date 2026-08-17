#!/usr/bin/env python3
"""
E5_4_3_reversibilidad_exergia_motor.py — ENFOQUE 5 (Energía·Exergía·Entropía), Tema 4,
experimento E5.4-3

"Reversibilidad: si se detiene la expansión, ¿la exergía se re-degrada?"

Pregunta (BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md, E5.4-3): parando la
expansión a mitad de camino, ¿la difusión mata la exergía ganada, o queda congelada? ¿Existe
un "tiempo de no-retorno"?

Metodología (reusada CONCEPTUALMENTE, no como código, de
BATERIA_FUNDAMENTOS/F3_4_reversibilidad_termica/F3_4_reversibilidad_termica_motor.py — mismo
truco de bifurcación STOP-vs-NULL desde un checkpoint exacto compartido, aplicado aquí al
observable de EXERGÍA en vez de al gradiente térmico):

  1) Fase común de expansión REAL (D=D0/a**3, heredado sin retocar de CF2/F3_4): se integra
     desde t_g=0, muestreando (checkpointing markoviano) el campo EXACTO en cada punto del
     barrido de `a` de parada.
  2) Desde cada checkpoint se bifurca en DOS ramas que parten del MISMO campo:
       STOP : `a` (y D) quedan fijos — solo difusión (+ruido) — durante POST_STOP_TG.
       NULL : "nunca parar" — la expansión REAL continúa (D sigue cayendo) — durante la
              MISMA ventana. Control pre-registrado por el documento autoritativo
              ("NULL=nunca parar").
  3) Dos observables de exergía, independientes entre sí (T2):
       X_var  = Σ(T-μ)²      (tipo energía-disponible/APE, cuadrático)
       X_info = Σ[ln2 - S_bin(T)]  (informacional, entropía binaria por celda)
  4) Guardián E1 (conservación del presupuesto E_total=ΣT) verificado en CADA checkpoint de
     CADA rama (T6) — no se asume, se mide la deriva.
  5) `ε` barre la AMPLITUD de la diferencia inicial (T0 = 0.5 + ε·(perfil-0.5)), no una
     cantidad hallada (T1). Ver PROTOCOLO_E5.4-3_PREREGISTRO.md §2.2-2.3 para la justificación
     física de por qué se barre pese a que la difusión pura es lineal en ε.

Protocolo fechado y congelado ANTES de este script — verificar mtime de
PROTOCOLO_E5.4-3_PREREGISTRO.md < mtime de este archivo.

Este script NO se auto-adjudica el veredicto de la hipótesis más amplia. Entrega números
crudos; la adjudicación es de CS después.

No edita CF2_estiramiento_motor.py, F3_4_reversibilidad_termica_motor.py, ni ninguna carpeta
E5_4_1/2/4/5_* de los agentes en paralelo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado de CF2/F3_4 (idéntico, T1: no se retoca)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2            # ancho comóvil inicial del salto (celdas)
DT = 0.25           # subpaso de difusión
N_SUB = 2            # subiteraciones de difusión por paso de reloj genético
ORIGINAL_STEPS_PER_TG = 399
DTG = 1.0 / ORIGINAL_STEPS_PER_TG

# ============================================================
# Barrido pre-registrado (PROTOCOLO_E5.4-3_PREREGISTRO.md, sección 4)
# ============================================================
A_STOP_GRID = np.geomspace(1.0, 1000.0, 10)          # 10 puntos de parada, 3 décadas
TG_STOP_GRID = np.log(A_STOP_GRID) / H_EXP
TG_MAX = float(np.log(1000.0) / H_EXP)               # duración de TODA la fase de expansión
POST_STOP_TG = TG_MAX                                 # ventana post-parada = misma duración

EPS_GRID = np.geomspace(1e-6, 1.0, 9)                # amplitud de la diferencia inicial, 6 déc.
RUIDO_DINAMICO_GRID = [0.0, 1e-3, 5e-3, 1e-2]        # perturbación dinámica (T7)

SEEDS_STANDARD_PROJECT = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_EXTENSION = [271828, 161803]                   # dígitos de e y de phi (misma extensión F3-3/F3-4)
SEEDS = SEEDS_STANDARD_PROJECT + SEEDS_EXTENSION      # 12 semillas (>=12 exigidas)

# ============================================================
# Criterio de PASS pre-registrado (protocolo, sección 5)
# ============================================================
MONO_TOL = 0.05
REDEGRAD_EARLY_MIN = 0.5
REDEGRAD_LATE_MAX = 0.1
DIFF_MIN = 0.1
PASS_RATE_MIN = 0.55

X_STOP_FLOOR = 1e-12   # por debajo de esto, X_parada se considera numéricamente degenerado -> NaN

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "E5_4_3_reversibilidad_exergia"

PROTOCOLO_PATH = CODE_DIR / "PROTOCOLO_E5.4-3_PREREGISTRO.md"


# ============================================================
# Física (heredada de CF2/F3_4: T(x,y) + difusión + expansión, con ruido dinámico opcional)
# ============================================================
def initial_T(L: int, w0: float, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Salto tanh de amplitud eps alrededor de mu=0.5 (eps=1 => salto completo CF2/F3_4;
    eps->0 => campo casi uniforme, mu=0.5). Mismo ruido de semilla que CF2/F3_4 (1e-4, NO
    escalado por eps, para no degenerar el estado inicial incluso en eps=0)."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    T0 = 0.5 + eps * (np.tile(profile, (L, 1)) - 0.5)
    T0 = T0 + 1e-4 * rng.normal(size=T0.shape)
    return np.clip(T0, 0.0, 1.0)


def exergy_metrics(T: np.ndarray) -> dict:
    """Los dos observables independientes de exergía (protocolo §2.4) + el guardián E1 (§2.5)."""
    E_total = float(np.sum(T))
    mu = float(np.mean(T))
    X_var = float(np.sum((T - mu) ** 2))

    Tc = np.clip(T, 1e-12, 1.0 - 1e-12)
    S_local = -Tc * np.log(Tc) - (1.0 - Tc) * np.log(1.0 - Tc)
    X_info = float(np.sum(np.log(2.0) - S_local))

    return {"E_total": E_total, "mu": mu, "X_var": X_var, "X_info": X_info}


def diffuse_step(T: np.ndarray, D: float, dt: float, n_sub: int,
                  rng: np.random.Generator | None, noise_amp: float) -> np.ndarray:
    """Un paso de reloj genético de difusión (N_SUB sub-iteraciones), con ruido dinámico
    aditivo OPCIONAL (Euler-Maruyama: escala por sqrt(dt/n_sub)). noise_amp=0.0 reproduce
    exactamente la difusión de 4-vecinos de CF2/F3_4 sin modificación."""
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


def integrate_common_phase(seed: int, eps: float, noise_amp: float, tg_stop_grid: np.ndarray):
    """Fase común REAL (idéntica en espíritu a CF2/F3_4 REAL): D=D0/a**3, desde t_g=0 hasta
    max(tg_stop_grid). Devuelve, por cada t_g de parada, un checkpoint con el campo exacto y
    sus métricas de exergía/energía."""
    rng = np.random.default_rng([seed, 0, int(round(eps * 1e9)) & 0xFFFFFFFF, 0])
    T = initial_T(L, W0, eps, rng)

    tg_max = float(tg_stop_grid[-1])
    n_steps = max(int(np.ceil(tg_max / DTG)), 1)

    checkpoints = []
    next_idx = 0

    def record(tg_now, a_now, T_now):
        m = exergy_metrics(T_now)
        checkpoints.append({
            "tg": float(tg_now),
            "a": float(a_now),
            "T": T_now.copy(),
            **m,
        })

    if tg_stop_grid[0] <= 1e-15:
        record(0.0, 1.0, T)
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
                           rng: np.random.Generator, noise_amp: float) -> np.ndarray:
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
    return T


def redegrad(before: float, after: float) -> float:
    """Fracción de exergía re-degradada. NaN explícito si `before` es numéricamente
    degenerado (protocolo §3) -- no se sustituye por 0 ni se descarta en silencio."""
    if before < X_STOP_FLOOR:
        return float("nan")
    return float((before - after) / before)


def run_combo(seed: int, eps: float, noise_amp: float, a_stop_grid: np.ndarray,
              tg_stop_grid: np.ndarray, post_stop_tg: float) -> dict:
    checkpoints = integrate_common_phase(seed, eps, noise_amp, tg_stop_grid)

    per_point = []
    e_total_0 = checkpoints[0]["E_total"]
    max_e_drift = 0.0

    for idx, ckpt in enumerate(checkpoints):
        a_stop = ckpt["a"]
        tg_stop = ckpt["tg"]
        T0 = ckpt["T"]
        Xvar_stop = ckpt["X_var"]
        Xinfo_stop = ckpt["X_info"]
        Etot_stop = ckpt["E_total"]
        max_e_drift = max(max_e_drift, abs(Etot_stop - e_total_0))

        rng_stop = np.random.default_rng([seed, idx, int(round(eps * 1e9)) & 0xFFFFFFFF, 1])
        T_stop_final = integrate_branch_stop(T0, a_stop, post_stop_tg, rng_stop, noise_amp)
        m_stop = exergy_metrics(T_stop_final)
        max_e_drift = max(max_e_drift, abs(m_stop["E_total"] - e_total_0))

        rng_null = np.random.default_rng([seed, idx, int(round(eps * 1e9)) & 0xFFFFFFFF, 2])
        T_null_final = integrate_branch_null(T0, tg_stop, post_stop_tg, rng_null, noise_amp)
        m_null = exergy_metrics(T_null_final)
        max_e_drift = max(max_e_drift, abs(m_null["E_total"] - e_total_0))

        per_point.append({
            "a_stop": float(a_stop),
            "tg_stop": float(tg_stop),
            "X_var_stop": float(Xvar_stop),
            "X_info_stop": float(Xinfo_stop),
            "E_total_stop": float(Etot_stop),
            "STOP": {
                "X_var_final": m_stop["X_var"],
                "X_info_final": m_stop["X_info"],
                "E_total_final": m_stop["E_total"],
                "redegrad_Xvar": redegrad(Xvar_stop, m_stop["X_var"]),
                "redegrad_Xinfo": redegrad(Xinfo_stop, m_stop["X_info"]),
            },
            "NULL_never_stop": {
                "X_var_final": m_null["X_var"],
                "X_info_final": m_null["X_info"],
                "E_total_final": m_null["E_total"],
                "redegrad_Xvar": redegrad(Xvar_stop, m_null["X_var"]),
                "redegrad_Xinfo": redegrad(Xinfo_stop, m_null["X_info"]),
            },
        })

    return {
        "seed": seed, "eps": eps, "noise_amp": noise_amp,
        "per_point": per_point,
        "E_total_0": float(e_total_0),
        "max_abs_E_drift": float(max_e_drift),
    }


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isnan(a) or np.isnan(b):
            return False
        if b > a + tol:
            return False
    return True


def find_no_retorno(a_stop_grid: np.ndarray, redegrad_stop: np.ndarray) -> float | None:
    """Primer a_stop (ascendente) donde redegrad_stop <= REDEGRAD_LATE_MAX de forma SOSTENIDA
    (no vuelve a subir por encima del umbral en ningún punto posterior). NaN cuenta como
    'no cumple' (no se puede afirmar congelamiento sobre un punto degenerado)."""
    n = len(redegrad_stop)
    for i in range(n):
        tail = redegrad_stop[i:]
        if np.any(np.isnan(tail)):
            continue
        if redegrad_stop[i] <= REDEGRAD_LATE_MAX and np.all(tail <= REDEGRAD_LATE_MAX):
            return float(a_stop_grid[i])
    return None


def evaluate_combo(combo: dict) -> dict:
    a_vals = np.array([p["a_stop"] for p in combo["per_point"]])
    redegrad_stop = np.array([p["STOP"]["redegrad_Xvar"] for p in combo["per_point"]])
    redegrad_null = np.array([p["NULL_never_stop"]["redegrad_Xvar"] for p in combo["per_point"]])

    cond_a = monotonic_nonincreasing(redegrad_stop)
    cond_b = bool((not np.isnan(redegrad_stop[0])) and redegrad_stop[0] >= REDEGRAD_EARLY_MIN)
    cond_c = bool((not np.isnan(redegrad_stop[-1])) and redegrad_stop[-1] <= REDEGRAD_LATE_MAX)
    if np.isnan(redegrad_stop[0]) or np.isnan(redegrad_null[0]):
        diff_early = float("nan")
        cond_d = False
    else:
        diff_early = float(redegrad_stop[0] - redegrad_null[0])
        cond_d = bool(diff_early >= DIFF_MIN)

    seed_pass = bool(cond_a and cond_b and cond_c and cond_d)
    no_retorno_a = find_no_retorno(a_vals, redegrad_stop)
    no_retorno_tg = (float(np.log(no_retorno_a) / H_EXP) if no_retorno_a is not None else None)

    # segundo observable (X_info) — comparación cualitativa, no gate (T3: no se sustituye el juez)
    redegrad_stop_info = np.array([p["STOP"]["redegrad_Xinfo"] for p in combo["per_point"]])
    no_retorno_a_info = find_no_retorno(a_vals, redegrad_stop_info)

    n_nan_stop = int(np.sum(np.isnan(redegrad_stop)))

    return {
        "cond_a_monotonic": cond_a,
        "cond_b_early_redegrad": cond_b,
        "cond_c_late_frozen": cond_c,
        "cond_d_null_bites": cond_d,
        "diff_early_stop_minus_null": diff_early,
        "seed_pass": seed_pass,
        "n_nan_points_Xvar": n_nan_stop,
        "no_retorno_a_Xvar": no_retorno_a,
        "no_retorno_tg_Xvar": no_retorno_tg,
        "no_retorno_a_Xinfo": no_retorno_a_info,
        "redegrad_stop_Xvar_curve": redegrad_stop.tolist(),
        "redegrad_null_Xvar_curve": redegrad_null.tolist(),
        "redegrad_stop_Xinfo_curve": redegrad_stop_info.tolist(),
        "max_abs_E_drift": combo["max_abs_E_drift"],
        "E_total_0": combo["E_total_0"],
    }


def run_production(seeds: list[int], eps_grid: np.ndarray, noise_grid: list[float],
                    a_stop_grid: np.ndarray, tg_stop_grid: np.ndarray, post_stop_tg: float,
                    tag: str) -> dict:
    t0 = time.time()
    combos = {}
    n_pass = 0
    n_total = 0
    max_e_drift_global = 0.0

    for seed in seeds:
        for eps in eps_grid:
            for noise_amp in noise_grid:
                combo = run_combo(seed, float(eps), noise_amp, a_stop_grid, tg_stop_grid, post_stop_tg)
                ev = evaluate_combo(combo)
                key = f"seed{seed}_eps{eps:.3e}_noise{noise_amp}"
                combos[key] = {
                    "seed": seed, "eps": float(eps), "noise_amp": noise_amp,
                    "per_point": combo["per_point"],
                    "evaluation": ev,
                }
                max_e_drift_global = max(max_e_drift_global, ev["max_abs_E_drift"])
                n_total += 1
                if ev["seed_pass"]:
                    n_pass += 1

    rate = n_pass / n_total if n_total else 0.0
    verdict_label = "E5_4_3_PASS" if rate >= PASS_RATE_MIN else "E5_4_3_FAIL"

    # desglose por eps y por ruido (protocolo §5: "no solo el agregado")
    rate_by_eps = {}
    for eps in eps_grid:
        keys = [k for k, v in combos.items() if abs(v["eps"] - float(eps)) < 1e-15]
        p = sum(1 for k in keys if combos[k]["evaluation"]["seed_pass"])
        rate_by_eps[f"{float(eps):.3e}"] = {"n_pass": p, "n_total": len(keys), "rate": p / len(keys) if keys else 0.0}

    rate_by_noise = {}
    for noise_amp in noise_grid:
        keys = [k for k, v in combos.items() if v["noise_amp"] == noise_amp]
        p = sum(1 for k in keys if combos[k]["evaluation"]["seed_pass"])
        rate_by_noise[str(noise_amp)] = {"n_pass": p, "n_total": len(keys), "rate": p / len(keys) if keys else 0.0}

    payload = {
        "experimento": "E5.4-3 Reversibilidad: si se detiene la expansion, la exergia se re-degrada?",
        "tag": tag,
        "sello_heredado_CF2_F3_4": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0, "DT": DT,
            "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_stop_grid": a_stop_grid.tolist(),
            "tg_stop_grid": tg_stop_grid.tolist(),
            "post_stop_tg": post_stop_tg,
            "eps_grid": eps_grid.tolist(),
            "ruido_dinamico_grid": noise_grid,
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL,
            "REDEGRAD_EARLY_MIN": REDEGRAD_EARLY_MIN,
            "REDEGRAD_LATE_MAX": REDEGRAD_LATE_MAX,
            "DIFF_MIN": DIFF_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "X_STOP_FLOOR": X_STOP_FLOOR,
        },
        "resultados_por_combo": combos,
        "resumen": {
            "n_combos_pass": n_pass,
            "n_combos_total": n_total,
            "rate": rate,
            "verdict": verdict_label,
            "rate_by_eps": rate_by_eps,
            "rate_by_noise": rate_by_noise,
            "max_abs_E_drift_global": max_e_drift_global,
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
        seeds = SEEDS[:2]
        eps_grid = EPS_GRID[::4]          # subgrid reducido
        noise_grid = RUIDO_DINAMICO_GRID[:2]
        a_stop_grid = np.geomspace(1.0, 1000.0, 4)
        tg_stop_grid = np.log(a_stop_grid) / H_EXP
        post_stop_tg = TG_MAX
        tag = "smoke"
    else:
        seeds = SEEDS
        eps_grid = EPS_GRID
        noise_grid = RUIDO_DINAMICO_GRID
        a_stop_grid = A_STOP_GRID
        tg_stop_grid = TG_STOP_GRID
        post_stop_tg = POST_STOP_TG
        tag = "produccion"

    print(f"=== E5.4-3 reversibilidad de exergia -- modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"a_stop_grid={a_stop_grid.tolist()}")
    print(f"eps_grid={eps_grid.tolist()}")
    print(f"ruido_dinamico_grid={noise_grid}")
    print(f"post_stop_tg={post_stop_tg:.6f} (== duracion total de la fase de expansion)")
    n_combos = len(seeds) * len(eps_grid) * len(noise_grid)
    print(f"n_combos={n_combos} (cada uno con {len(a_stop_grid)} puntos de parada x 2 ramas)")

    payload = run_production(seeds, eps_grid, noise_grid, a_stop_grid, tg_stop_grid, post_stop_tg, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    n_print = 0
    for key, rec in payload["resultados_por_combo"].items():
        ev = rec["evaluation"]
        if n_print < 40 or args.mode == "smoke":
            print(
                f"  {key:>34}  cond_a={ev['cond_a_monotonic']}  cond_b={ev['cond_b_early_redegrad']}  "
                f"cond_c={ev['cond_c_late_frozen']}  cond_d={ev['cond_d_null_bites']}  "
                f"no_retorno_a={ev['no_retorno_a_Xvar']}  n_nan={ev['n_nan_points_Xvar']}  "
                f"seed_pass={ev['seed_pass']}"
            )
        n_print += 1
    if n_print > 40 and args.mode != "smoke":
        print(f"  ... ({n_print - 40} combos más, ver JSON completo) ...")

    print(f"\nrate={payload['resumen']['rate']:.3f}  verdict={payload['resumen']['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")
    print(f"max_abs_E_drift_global={payload['resumen']['max_abs_E_drift_global']:.3e}  (guardian E1)")
    print("rate_by_eps:")
    for k, v in payload["resumen"]["rate_by_eps"].items():
        print(f"    eps={k:>10}  rate={v['rate']:.3f}  ({v['n_pass']}/{v['n_total']})")
    print("rate_by_noise:")
    for k, v in payload["resumen"]["rate_by_noise"].items():
        print(f"    noise={k:>8}  rate={v['rate']:.3f}  ({v['n_pass']}/{v['n_total']})")
    print(f"runtime_seconds={payload['runtime_seconds']:.2f}")

    out_json = OUT_DIR / f"E5_4_3_reversibilidad_exergia_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
