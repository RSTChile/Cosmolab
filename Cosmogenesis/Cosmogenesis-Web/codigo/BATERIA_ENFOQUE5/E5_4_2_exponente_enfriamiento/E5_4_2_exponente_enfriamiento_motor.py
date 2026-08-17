#!/usr/bin/env python3
"""
E5_4_2_exponente_enfriamiento_motor.py — BATERIA ENFOQUE 5, Tema 4, experimento E5.4-2

"Exponente de enfriamiento emergente: ¿T∝a^-n, con qué n?"

Pregunta (spec literal, BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md):
barrido a ∈ [1..1e6] (6 decadas) x epsilon x >=12 semillas. Observable: n medido en
T∝a^-n (SALIDA, nunca impuesto). NULL: sin expansion. PASS: n emerge y se reporta -
NO se fija a n=2 ni n=3 aunque la fisica los sugiera.

Protocolo congelado ANTES de este motor: PROTOCOLO_E5.4-2_PREREGISTRO.md (mismo
directorio; verificar mtime < mtime de este archivo).

=====================================================================
COMO SE EVITA EL DEFECTO DE F3-2 (leer antes de tocar nada)
=====================================================================
F3_2_T_emergente_motor.py midio T_energy(a) = grad_energy_comov(a) / a^2 EN AMBAS
ramas (REAL y NULL_RHO_FIXED). Su propio JSON de produccion
(results/F3_2_T_emergente/F3_2_T_emergente_produccion_result.json) muestra que el
ajuste asintotico converge a slope_REAL = -1.999999... (el exponente EXACTO de la
propia division geometrica /a^2, no una medida fisica independiente), y que en la
combinacion inspeccionada (sigma=1e-4, seed=7) slope_NULL=-2.1876 esta a solo 0.065
de slope_REAL=-2.1225 - pasa el umbral de "el NULL muerde" (0.05) por un margen
minimo. Causa: la cantidad COMOVIL sin dividir se estabiliza en un piso casi
identico en ambas ramas; al dividir ambas por el mismo a^2, el "-2" es aritmetica
de la conversion, no fisica de enfriamiento.

Este motor evita esa clase de defecto por construccion: el observable de
temperatura NUNCA divide por ninguna potencia de `a`. Se simula un gas ideal de
particulas libres en una caja fisica cuya pared REAL se aleja como
Lh(t)=0.5*L0*a(t) (colision elastica real con la pared movil, formula exacta
v <- 2*Vw - v); la temperatura se mide directo de las velocidades FISICAS,
T = mean(v^2), sin ningun factor a^-n en su definicion. Si aparece enfriamiento,
es porque las colisiones fisicas realmente le quitan energia cinetica al gas
cuando la pared retrocede - no porque se dividio una cantidad comovil por a^n.

El NULL_SIN_EXPANSION es, por esto mismo, literal: pared fija en a=1 SIEMPRE (no
"misma trayectoria de a(t) con densidad fija", la reinterpretacion que F3-2 tuvo
que adoptar porque su observable si necesitaba a(t) para dividir). Aqui el eje x
(a_grid) es solo la ETIQUETA del checkpoint de reloj genetico en que se lee T, no
un divisor - el NULL puede ser el caso mas fuerte posible (pared que nunca se
mueve) sin romper nada.

Ver PROTOCOLO_E5.4-2_PREREGISTRO.md secciones 0-1 para el razonamiento completo,
y la seccion "veredicto F3-2" del payload JSON (mas abajo) para la comparacion
numerica explicita que este motor calcula sobre sus PROPIOS resultados.

Reutiliza SOLO el reloj genetico de CF2/F3-2 (t_g -> a=exp(H_EXP*t_g), H_EXP=6.0,
dtg=1/399, checkpointing markoviano de una sola trayectoria muestreada en los
t_g(a) objetivo). NO reutiliza el campo de difusion, el perfil inicial ni el
observable de CF2/F3-1/F3-2 - el sustrato fisico (gas de particulas en caja con
pared movil) es propio de este experimento.

No edita CF2_estiramiento_motor.py, F3_2_T_emergente_motor.py, ni ningun otro
archivo existente. Prefijo exclusivo E5_4_2_.

Este script NO se auto-adjudica el veredicto de la bateria. Entrega numeros
crudos; la adjudicacion final es de CS (Alexis) despues.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Reloj genetico heredado de CF2/F3-2 (identico, T1: no se
# retoca para favorecer el resultado)
# ============================================================
H_EXP = 6.0
DTG = 1.0 / 399  # ORIGINAL_STEPS_PER_TG = 399, idéntico a CF2/F3-2

# ============================================================
# Parametros PROPIOS de E5.4-2 (pre-registrados,
# PROTOCOLO_E5.4-2_PREREGISTRO.md seccion 4)
# ============================================================
L0 = 5.0e-4       # ancho inicial de caja por eje (elegido por factibilidad, sec.3-4)
V0 = 1.0          # sigma de velocidad inicial por eje (T0=1)
N_SUB = 16        # subpasos de integracion por paso de reloj genetico (convergencia
                  # verificada contra N_SUB=128: T(a) coincide <0.5%, ledger~1e-15 en
                  # ambos, ver protocolo sec.4 "prueba de convergencia numerica")
N_PART = 2000     # particulas por replica
MAX_COLLISION_ITERS = 2  # cota fija de re-chequeo de cruce por subpaso (colas rapidas);
                  # el conteo de colisiones no cambia entre 1 y 4 en las pruebas

A_GRID = np.geomspace(1.0, 1.0e6, 25)          # 25 puntos, 6 decadas (spec literal)
EPS_GRID = np.geomspace(1.0e-12, 1.0, 8)        # 8 puntos, 12 decadas (mismo rango
                                                  # que E5.1-1/E5.3-1 hermanos)

SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_EXTRA = [13, 271828]
SEEDS_12 = SEEDS_STANDARD + SEEDS_EXTRA          # 12, minimo exigido por la spec

# ============================================================
# Criterio de PASS pre-registrado (protocolo seccion 6)
# ============================================================
MONO_TOL = 1e-6
R2_MIN = 0.50
SLOPE_DIFF_MIN = 0.3
PASS_RATE_MIN = 0.55

MODES = ["REAL", "NULL_SIN_EXPANSION"]

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "E5_4_2_exponente_enfriamiento"


# ============================================================
# Fisica: gas ideal en caja isotropa con pared movil (REAL) o
# fija (NULL_SIN_EXPANSION). Vectorizado sobre un eje de
# "replica" = indice de epsilon (una corrida procesa TODAS las
# columnas de epsilon de una semilla+modo a la vez).
# ============================================================
def run_seed_mode(mode: str, seed: int, eps_grid: np.ndarray, a_grid: np.ndarray) -> dict:
    """
    Integra N_EPS replicas en paralelo (una por columna de eps_grid), cada una con
    N_PART particulas en 3D, desde t_g=0 hasta t_g_max=ln(a_grid[-1])/H_EXP,
    muestreando en los checkpoints t_g(a_grid) (mismo metodo markoviano de una sola
    trayectoria que CF2/F3-2: no se re-simula desde cero por punto de `a`).

    mode="REAL": pared en Lh(t_g) = 0.5*L0*a(t_g), a(t_g)=exp(H_EXP*t_g).
    mode="NULL_SIN_EXPANSION": pared fija en Lh=0.5*L0 SIEMPRE (Vw=0), a_grid se usa
    solo como ETIQUETA de checkpoint (no hay ningun a(t) fisico en esta rama).
    """
    n_eps = len(eps_grid)
    rng = np.random.default_rng(seed)

    pos = (rng.random((n_eps, N_PART, 3)) - 0.5) * L0
    vel = rng.normal(0.0, V0, size=(n_eps, N_PART, 3))
    eps_col = eps_grid[:, None, None]  # (n_eps,1,1) broadcast sobre (n_eps,N_PART,3)

    dtg = DTG
    dt_sub = dtg / N_SUB
    sqrt_dt_sub = np.sqrt(dt_sub)

    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    # ledger de energia (T6): KE(t) - KE(0) + W_pared_acumulado - E_ruido_acumulado ~ 0
    KE0 = 0.5 * np.sum(vel**2, axis=(1, 2))  # (n_eps,)
    w_pared_acum = np.zeros(n_eps)
    e_ruido_acum = np.zeros(n_eps)
    n_colisiones = np.zeros(n_eps, dtype=np.int64)

    checkpoints = []  # lista de dicts con arrays (n_eps,)
    next_ckpt_idx = 0

    def record(tg_now, a_now_label):
        KE_now = 0.5 * np.sum(vel**2, axis=(1, 2))
        ledger_dev = KE_now - KE0 + w_pared_acum - e_ruido_acum
        ledger_rel = np.abs(ledger_dev) / np.maximum(KE0, 1e-300)
        checkpoints.append({
            "a": float(a_now_label),
            "tg": float(tg_now),
            "T_all": (np.mean(vel**2, axis=(1, 2))).copy(),           # (n_eps,)
            "T_x": (np.mean(vel[:, :, 0] ** 2, axis=1)).copy(),
            "T_y": (np.mean(vel[:, :, 1] ** 2, axis=1)).copy(),
            "T_z": (np.mean(vel[:, :, 2] ** 2, axis=1)).copy(),
            "n_colisiones_acum": n_colisiones.copy(),
            "ledger_rel_dev": ledger_rel.copy(),
        })

    if tg_targets[0] <= 1e-15:
        record(0.0, a_grid[0])
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        if mode == "NULL_SIN_EXPANSION":
            Lh = 0.5 * L0
            Vw = 0.0
        else:  # REAL
            a_now = float(np.exp(H_EXP * tg))
            Lh = 0.5 * L0 * a_now
            Vw = 0.5 * L0 * H_EXP * a_now

        for _ in range(N_SUB):
            pos = pos + vel * dt_sub

            for _try in range(MAX_COLLISION_ITERS):
                hi = pos > Lh
                lo = pos < -Lh
                if not (hi.any() or lo.any()):
                    break
                ke_before = 0.5 * np.sum(np.where(hi | lo, vel**2, 0.0), axis=(1, 2))

                vel = np.where(hi, 2.0 * Vw - vel, vel)
                pos = np.where(hi, 2.0 * Lh - pos, pos)
                vel = np.where(lo, -2.0 * Vw - vel, vel)
                pos = np.where(lo, -2.0 * Lh - pos, pos)

                ke_after = 0.5 * np.sum(np.where(hi | lo, vel**2, 0.0), axis=(1, 2))
                # energia extraida POR la pared DEL gas en esta reflexion
                w_pared_acum += (ke_before - ke_after)
                n_colisiones += (hi.sum(axis=(1, 2)) + lo.sum(axis=(1, 2)))

            if eps_col.max() > 0:
                ke_before_noise = 0.5 * np.sum(vel**2, axis=(1, 2))
                vel = vel + eps_col * sqrt_dt_sub * rng.standard_normal(vel.shape)
                ke_after_noise = 0.5 * np.sum(vel**2, axis=(1, 2))
                e_ruido_acum += (ke_after_noise - ke_before_noise)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, a_grid[next_ckpt_idx])
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        record(tg_targets[next_ckpt_idx], a_grid[next_ckpt_idx])
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    T_all = np.array([c["T_all"] for c in checkpoints])   # (n_ckpt, n_eps)
    T_x = np.array([c["T_x"] for c in checkpoints])
    T_y = np.array([c["T_y"] for c in checkpoints])
    T_z = np.array([c["T_z"] for c in checkpoints])
    n_coll = np.array([c["n_colisiones_acum"] for c in checkpoints])
    ledger = np.array([c["ledger_rel_dev"] for c in checkpoints])

    return {
        "mode": mode,
        "seed": seed,
        "a_grid": a_vals.tolist(),
        "eps_grid": eps_grid.tolist(),
        "T_all": T_all.tolist(),      # [n_ckpt][n_eps]
        "T_x": T_x.tolist(),
        "T_y": T_y.tolist(),
        "T_z": T_z.tolist(),
        "n_colisiones_final": n_coll[-1].tolist(),
        "ledger_rel_dev_max": float(np.max(ledger)),
    }


# ============================================================
# Ajustes y evaluacion
# ============================================================
def loglog_fit(a_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float, float]:
    x = np.log(a_vals)
    y = np.log(np.clip(y_vals, 1e-300, None))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), _res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


def monotonic_nonincreasing(vals: np.ndarray, tol: float = MONO_TOL) -> bool:
    for i in range(len(vals) - 1):
        if vals[i + 1] > vals[i] * (1.0 + tol) + tol:
            return False
    return True


def evaluate_column(a_vals: np.ndarray, y_real: np.ndarray, y_null: np.ndarray) -> dict:
    mono_real = monotonic_nonincreasing(y_real)
    mono_null = monotonic_nonincreasing(y_null)
    slope_real, _ic_r, r2_real = loglog_fit(a_vals, y_real)
    slope_null, _ic_n, r2_null = loglog_fit(a_vals, y_null)
    slope_diff = abs(slope_null - slope_real)

    # diagnostico segmentado (NO criterio de PASS, protocolo seccion 5)
    n = len(a_vals)
    k = n // 2
    slope_early, _i1, r2_early = loglog_fit(a_vals[:k], y_real[:k])
    slope_late, _i2, r2_late = loglog_fit(a_vals[k:], y_real[k:])

    cond_mono = mono_real
    cond_r2 = r2_real >= R2_MIN
    cond_null_bites = (not mono_null) or (slope_diff >= SLOPE_DIFF_MIN)
    punto_pass = bool(cond_mono and cond_r2 and cond_null_bites)

    return {
        "mono_REAL": bool(mono_real),
        "mono_NULL": bool(mono_null),
        "slope_REAL_global": slope_real,
        "R2_REAL_global": r2_real,
        "slope_NULL_global": slope_null,
        "R2_NULL_global": r2_null,
        "slope_diff_abs": float(slope_diff),
        "slope_REAL_fase_temprana_diagnostico": slope_early,
        "R2_REAL_fase_temprana_diagnostico": r2_early,
        "slope_REAL_fase_tardia_diagnostico": slope_late,
        "R2_REAL_fase_tardia_diagnostico": r2_late,
        "cond_mono_real": bool(cond_mono),
        "cond_r2_real_min": bool(cond_r2),
        "cond_null_muerde": bool(cond_null_bites),
        "punto_pass": punto_pass,
    }


def run_production(seeds: list[int], eps_grid: np.ndarray, a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()

    per_seed = {}
    n_pass_T_all = 0
    n_pass_T_x = 0
    n_null_bites = 0
    n_agree = 0
    n_total = 0
    ledger_max_dev_global = 0.0

    rate_by_eps_T_all_num = {f"{e:.6e}": 0 for e in eps_grid}
    rate_by_eps_T_all_den = {f"{e:.6e}": 0 for e in eps_grid}

    all_slopes_T_all_global = []
    all_slopes_T_all_early = []
    all_slopes_T_all_late = []
    all_slopes_T_x_global = []
    all_slopes_NULL_global = []

    for seed in seeds:
        real = run_seed_mode("REAL", seed, eps_grid, a_grid)
        null = run_seed_mode("NULL_SIN_EXPANSION", seed, eps_grid, a_grid)

        ledger_max_dev_global = max(
            ledger_max_dev_global, real["ledger_rel_dev_max"], null["ledger_rel_dev_max"]
        )

        a_vals = np.array(real["a_grid"])
        T_all_real = np.array(real["T_all"])   # (n_ckpt, n_eps)
        T_all_null = np.array(null["T_all"])
        T_x_real = np.array(real["T_x"])
        T_x_null = np.array(null["T_x"])

        per_seed[str(seed)] = {"eps_columns": {}}

        for j, eps in enumerate(eps_grid):
            eps_key = f"{eps:.6e}"
            ev_all = evaluate_column(a_vals, T_all_real[:, j], T_all_null[:, j])
            ev_x = evaluate_column(a_vals, T_x_real[:, j], T_x_null[:, j])

            per_seed[str(seed)]["eps_columns"][eps_key] = {
                "T_all_REAL": T_all_real[:, j].tolist(),
                "T_all_NULL": T_all_null[:, j].tolist(),
                "T_x_REAL": T_x_real[:, j].tolist(),
                "T_x_NULL": T_x_null[:, j].tolist(),
                "evaluation_T_all": ev_all,
                "evaluation_T_x": ev_x,
            }

            n_total += 1
            if ev_all["punto_pass"]:
                n_pass_T_all += 1
                rate_by_eps_T_all_num[eps_key] += 1
            rate_by_eps_T_all_den[eps_key] += 1
            if ev_x["punto_pass"]:
                n_pass_T_x += 1
            if ev_all["cond_null_muerde"]:
                n_null_bites += 1
            if ev_all["punto_pass"] == ev_x["punto_pass"]:
                n_agree += 1

            all_slopes_T_all_global.append(ev_all["slope_REAL_global"])
            all_slopes_T_all_early.append(ev_all["slope_REAL_fase_temprana_diagnostico"])
            all_slopes_T_all_late.append(ev_all["slope_REAL_fase_tardia_diagnostico"])
            all_slopes_T_x_global.append(ev_x["slope_REAL_global"])
            all_slopes_NULL_global.append(ev_all["slope_NULL_global"])

        # colisiones para el registro (diagnostico de si REAL realmente colisiona con la
        # pared y NULL tambien colisiona -pared fija- pero sin moverse nunca)
        per_seed[str(seed)]["n_colisiones_final_REAL_por_eps"] = real["n_colisiones_final"]
        per_seed[str(seed)]["n_colisiones_final_NULL_por_eps"] = null["n_colisiones_final"]
        per_seed[str(seed)]["ledger_rel_dev_max_REAL"] = real["ledger_rel_dev_max"]
        per_seed[str(seed)]["ledger_rel_dev_max_NULL"] = null["ledger_rel_dev_max"]

    rate_T_all = n_pass_T_all / n_total if n_total else 0.0
    rate_T_x = n_pass_T_x / n_total if n_total else 0.0
    rate_null_bites = n_null_bites / n_total if n_total else 0.0
    rate_agree = n_agree / n_total if n_total else 0.0

    verdict = (
        "E5_4_2_PASS"
        if (rate_T_all >= PASS_RATE_MIN and rate_null_bites >= PASS_RATE_MIN)
        else "E5_4_2_FAIL"
    )

    curva_robustez_P_eps = {
        k: (rate_by_eps_T_all_num[k] / rate_by_eps_T_all_den[k] if rate_by_eps_T_all_den[k] else 0.0)
        for k in rate_by_eps_T_all_num
    }

    def stats(arr):
        a = np.array(arr)
        return {
            "media": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "n": int(len(a)),
        }

    # ------------------------------------------------------------
    # Veredicto explícito sobre el defecto F3-2, calculado sobre
    # los propios resultados de ESTE motor (protocolo sección 6:
    # "no basta con afirmarlo, se muestra el número").
    # ------------------------------------------------------------
    n_global_stats = stats(all_slopes_T_all_global)
    n_null_stats = stats(all_slopes_NULL_global)
    defecto_f3_2 = {
        "descripcion": (
            "F3-2 definia T_energy = grad_energy_comov / a^2 en AMBAS ramas; su ajuste "
            "asintotico REAL convergia a slope=-1.999999... (el exponente exacto de la "
            "propia division por a^2, no una medida fisica), y en (sigma=1e-4, seed=7) "
            "slope_NULL=-2.1876 vs slope_REAL=-2.1225 (diff=0.065) - el NULL casi no se "
            "distinguia del REAL. Este motor mide T=mean(v^2) SIN dividir por ninguna "
            "potencia de a en ninguna rama; el NULL usa pared fisicamente fija (a=1 "
            "siempre), no la misma trayectoria con densidad fija."
        ),
        "media_slope_REAL_global_T_all": n_global_stats["media"],
        "media_slope_NULL_global_T_all": n_null_stats["media"],
        "separacion_REAL_vs_NULL": abs(n_global_stats["media"] - n_null_stats["media"]),
        "evita_division_a_n_en_observable": True,
        "NULL_es_pared_fisicamente_fija_no_reinterpretado": True,
    }

    payload = {
        "experimento": "E5.4-2 exponente de enfriamiento emergente (T∝a^-n)",
        "tag": tag,
        "sello_heredado_CF2_F3_2": {"H_EXP": H_EXP, "DTG": DTG},
        "parametros_propios_E5_4_2": {
            "L0": L0, "V0": V0, "N_SUB": N_SUB, "N_PART": N_PART,
            "MAX_COLLISION_ITERS": MAX_COLLISION_ITERS,
            "modelo": "gas ideal de N_PART particulas libres en caja isotropa 3D; "
                      "colision elastica con pared movil (REAL, Lh=0.5*L0*a(t)) o fija "
                      "(NULL_SIN_EXPANSION, Lh=0.5*L0 siempre); T=mean(v^2) SIN dividir "
                      "por ninguna potencia de a.",
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "eps_grid": eps_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_eps": len(eps_grid),
            "n_total_combinaciones": n_total,
        },
        "criterio_preregistrado": {
            "MONO_TOL": MONO_TOL, "R2_MIN": R2_MIN,
            "SLOPE_DIFF_MIN": SLOPE_DIFF_MIN, "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "punto_pass = mono_REAL AND (R2_REAL_global>=R2_MIN) AND "
                "(NOT mono_NULL OR abs(slope_NULL_global-slope_REAL_global)>=SLOPE_DIFF_MIN); "
                "veredicto principal basado en T_all; PASS si rate_T_all>=PASS_RATE_MIN Y "
                "rate_null_muerde>=PASS_RATE_MIN"
            ),
        },
        "resultados_por_semilla": per_seed,
        "curva_robustez_P_epsilon_T_all": curva_robustez_P_eps,
        "resumen": {
            "n_total_combinaciones": n_total,
            "n_pass_T_all": n_pass_T_all,
            "n_pass_T_x": n_pass_T_x,
            "rate_T_all": rate_T_all,
            "rate_T_x": rate_T_x,
            "rate_NULL_muerde": rate_null_bites,
            "rate_acuerdo_T_all_vs_T_x": rate_agree,
            "exponente_n_T_all_global": {k: -v if k == "media" else v for k, v in n_global_stats.items()} if False else n_global_stats,
            "exponente_n_T_all_fase_temprana_diagnostico": stats(all_slopes_T_all_early),
            "exponente_n_T_all_fase_tardia_diagnostico": stats(all_slopes_T_all_late),
            "exponente_n_T_x_global": stats(all_slopes_T_x_global),
            "slope_NULL_global": n_null_stats,
            "ledger_energia_max_dev_relativa_toda_la_corrida": ledger_max_dev_global,
            "verdict": verdict,
        },
        "veredicto_defecto_F3_2": defecto_f3_2,
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
        seeds = SEEDS_STANDARD[:2]
        eps_grid = EPS_GRID[::3]
        a_grid = np.geomspace(1.0, 1.0e3, 8)
        tag = "smoke"
    else:
        seeds = SEEDS_12
        eps_grid = EPS_GRID
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== E5.4-2 exponente de enfriamiento — modo={args.mode} ===")
    print(f"seeds(n={len(seeds)})={seeds}")
    print(f"eps_grid={eps_grid.tolist()}")
    print(f"a_grid: {a_grid[0]:.3e} .. {a_grid[-1]:.3e} ({len(a_grid)} pts)")

    payload = run_production(seeds, eps_grid, a_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    r = payload["resumen"]
    print(f"n_total_combinaciones={r['n_total_combinaciones']}")
    print(f"rate_T_all={r['rate_T_all']:.3f}  rate_T_x={r['rate_T_x']:.3f}")
    print(f"rate_NULL_muerde={r['rate_NULL_muerde']:.3f}")
    print(f"rate_acuerdo_T_all_vs_T_x={r['rate_acuerdo_T_all_vs_T_x']:.3f}")
    e = r["exponente_n_T_all_global"]
    print(f"slope_REAL_global T_all: media={e['media']:.4f} std={e['std']:.4f} min={e['min']:.4f} max={e['max']:.4f} (n={e['n']})")
    ea = r["exponente_n_T_all_fase_temprana_diagnostico"]
    print(f"slope_REAL fase temprana (diagnostico): media={ea['media']:.4f} std={ea['std']:.4f}")
    el = r["exponente_n_T_all_fase_tardia_diagnostico"]
    print(f"slope_REAL fase tardia (diagnostico): media={el['media']:.4f} std={el['std']:.4f}")
    nn = r["slope_NULL_global"]
    print(f"slope_NULL_global: media={nn['media']:.4f} std={nn['std']:.4f}")
    print(f"ledger max dev relativa en toda la corrida: {r['ledger_energia_max_dev_relativa_toda_la_corrida']:.3e}")
    print(f"\nverdict={r['verdict']}  (umbrales: R2_MIN={R2_MIN}, SLOPE_DIFF_MIN={SLOPE_DIFF_MIN}, PASS_RATE_MIN={PASS_RATE_MIN})")

    print("\ncurva_robustez P(epsilon) [T_all]:")
    for k, v in payload["curva_robustez_P_epsilon_T_all"].items():
        print(f"  eps={k}  P={v:.3f}")

    print("\nveredicto_defecto_F3_2:")
    print(json.dumps(payload["veredicto_defecto_F3_2"], indent=2, ensure_ascii=False))

    out_json = OUT_DIR / f"E5_4_2_exponente_enfriamiento_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
