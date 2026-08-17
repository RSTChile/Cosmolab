#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F4_4_densidad_critica_motor.py — BATERÍA_FUNDAMENTOS, ENFOQUE 4, experimento F4-4

"Densidad crítica de congelamiento: ¿hay un ρ bajo el cual la diferencia se bloquea?"

Pre-registro (leer PRIMERO, no se toca tras ver resultados — T3):
  PROTOCOLO_F4-4_PREREGISTRO.md (mismo directorio, mtime ANTERIOR a este archivo).

Extiende el mecanismo de persistencia r=H/D de
  /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py
(campo continuo en anillo, difusión solo por aristas vivas, expansión = corte
Bernoulli de aristas) añadiendo un eje de densidad ρ que escala la tasa de
difusión — mismo enlace físico D∝ρ/ρ0 ya usado en
  /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py

NINGÚN archivo de esos dos se edita ni se importa con efectos secundarios; las
funciones relevantes se REESCRIBEN aquí (generalizadas con `rate`/`sigma_ruido`)
para no depender de imports frágiles entre carpetas de otros experimentos.

Punto de diseño crítico (ver §2 del pre-registro): el número de pasos ("reloj
físico") se calibra UNA VEZ por N, a ρ=ρ0 de referencia, y se REUSA igual para
TODOS los ρ del barrido. Si se recalibrara por ρ, el efecto de congelamiento
por dilución se cancelaría por construcción.

Este script NO se auto-adjudica "existe/no existe ρ_c" a nivel de batería.
Entrega la curva P(ρ) completa por celda (r, N, σ_ruido) con dispersión real
entre semillas; la adjudicación final es de CS.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "F4_4_densidad_critica"

# ============================================================
# Sello físico heredado (sin retocar) de cs074_rcruz.py / CF2_estiramiento_motor.py
# ============================================================
RATE0 = 0.5     # tasa de difusión de referencia (== cs074_rcruz.paso_difusion, factor 0.5)
RHO0 = 1.0      # densidad de referencia (== CF2_estiramiento_motor.RHO0)
EPS = 1e-3      # amplitud de la mancha sembrada; valor de referencia heredado
                # de la calibración de producción de cs074_rcruz.py (no elegido ad hoc)
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F4-4_PREREGISTRO.md, sección 3)
# ============================================================
RHO_GRID = np.geomspace(1e-6, 1.0, 15).tolist()     # ρ/ρ0, 15 puntos, 6 décadas
R_TARGETS = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]   # 7 puntos, cruza r=1
N_LIST = [200, 400]                                  # verificación cruzada (b): estabilidad en N
SIGMA_RUIDO_LIST = [0.0, 1e-4, 1e-3, 1e-2]           # verificación cruzada (b): estabilidad en ruido dinámico
SEEDS = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803]  # 12 semillas
SEEDS_CALIB = 8  # semillas para calibración de D(ρ) y del reloj físico (instrumento, no producción)


# ============================================================
# Física base (generalización de cs074_rcruz.py con parámetro `rate` y ruido dinámico)
# ============================================================
def campo_inicial(N: int, eps: float, rng: np.random.Generator):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert, x


def paso_difusion(phi: np.ndarray, activo: np.ndarray, rate: float) -> np.ndarray:
    """Difusión solo por aristas vivas, tasa `rate` (== 0.5*rho/rho0 en la corrida real).
    Idéntica en estructura a cs074_rcruz.paso_difusion, generalizada con `rate`
    en vez del 0.5 fijo (rate=0.5 reproduce EXACTAMENTE cs074_rcruz a rho=rho0)."""
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + rate * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo: np.ndarray, H: float, rng: np.random.Generator) -> np.ndarray:
    """Idéntica a cs074_rcruz.paso_expansion: corte Bernoulli por arista viva,
    esperanza de fracción cortada/paso = H (válido también para H*N < 1)."""
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:] = False
        return activo
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def medir_D(N: int, eps: float, seed: int, rate: float) -> float:
    """Fracción de contraste borrada en UN paso de difusión pura (H=0), a la
    tasa `rate` dada. D se MIDE (T1), no se impone; con rate=RATE0*(rho/RHO0)
    esto da D(rho) empírico."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo, rate)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def persistencia(phi: np.ndarray, contraste0: float):
    """P = autocorrelación_lag1(phi) * varianza_normalizada — idéntico a
    cs074_rcruz.persistencia. Devuelve también v_solo (observable secundario
    de diagnóstico: solo el factor de varianza, sin autocorrelación)."""
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0, 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v), v


def medir_pasos_lavado(N: int, eps: float, semillas: int, rate: float,
                        P_thr: float = P_LAVADO, max_steps: int = 200_000,
                        check_every: int = 50) -> dict:
    """Calibra el reloj físico UNA VEZ, a rate=RATE0 (rho=rho0), sin ruido
    dinámico (calibración = condiciones de referencia). Este número de pasos
    se REUSA igual para todos los rho del barrido (ver docstring del módulo)."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(10_000 + s)
        phi, _ = campo_inicial(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion(phi, activo, rate)
            if t % check_every == 0:
                p, _ = persistencia(phi, c0)
                if p < P_thr:
                    t_hit = t
                    break
        if t_hit is None:
            t_hit = max_steps
        tiempos.append(t_hit)
    med = int(np.median(tiempos))
    pasos = int(np.ceil(med * MARGEN_LAVADO))
    return {
        "tiempos": tiempos,
        "mediana": med,
        "pasos": pasos,
        "P_thr": P_thr,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def corrida(N: int, eps: float, rate: float, H: float, pasos: int, seed: int,
            sigma_ruido: float, null: bool = False) -> dict:
    """Una corrida completa: difusión a tasa `rate` + expansión H + ruido
    dinámico sigma_ruido en CADA paso (T7 — perturbación de la dinámica, no
    solo de la semilla). NULL = permutar phi al final (barajado del acople,
    operacionalización idéntica a cs074_rcruz — ver pre-registro §4)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    contraste0 = float(phi.std())
    for _ in range(pasos):
        if sigma_ruido > 0.0:
            phi = phi + rng.normal(0.0, sigma_ruido, size=phi.shape)
        phi = paso_difusion(phi, activo, rate)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    P, v_solo = persistencia(phi, contraste0)
    frac_exp = 1.0 - float(activo.mean())
    return {"P": P, "v_solo": v_solo, "frac_exp": frac_exp,
            "std_ratio": float(phi.std() / contraste0) if contraste0 > 0 else 0.0}


# ============================================================
# Worker de celda (una combinación N, rho_ratio, r_target, sigma_ruido)
# ============================================================
def cell_worker(task: dict) -> dict:
    N = task["N"]
    rho_ratio = task["rho_ratio"]
    r_tgt = task["r_target"]
    sigma = task["sigma_ruido"]
    D = task["D"]
    pasos = task["pasos"]
    rate = RATE0 * rho_ratio

    H = float(min(r_tgt * D, 1.0)) if D > 0 else (1.0 if r_tgt > 0 else 0.0)
    r_eff = (H / D) if D > 0 else float("inf")

    Preal, Pnull, vreal, vnull = [], [], [], []
    for s in SEEDS:
        rr = corrida(N, EPS, rate, H, pasos, seed=s, sigma_ruido=sigma, null=False)
        nn = corrida(N, EPS, rate, H, pasos, seed=s, sigma_ruido=sigma, null=True)
        Preal.append(rr["P"]); Pnull.append(nn["P"])
        vreal.append(rr["v_solo"]); vnull.append(nn["v_solo"])

    Preal = np.array(Preal); Pnull = np.array(Pnull)
    sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
    sd = max(sd, 1.0 / max(len(Preal), 1))
    z = float((Preal.mean() - Pnull.mean()) / sd)

    return {
        "N": N,
        "rho_ratio": rho_ratio,
        "r_target": r_tgt,
        "sigma_ruido": sigma,
        "D": D,
        "H": H,
        "r_eff": r_eff,
        "pasos": pasos,
        "n_seeds": len(SEEDS),
        "P_real_mean": float(Preal.mean()),
        "P_real_std": float(Preal.std()),
        "P_null_mean": float(Pnull.mean()),
        "P_null_std": float(Pnull.std()),
        "v_solo_real_mean": float(np.mean(vreal)),
        "v_solo_null_mean": float(np.mean(vnull)),
        "z": z,
    }


# ============================================================
# Derivación de rho_c por celda (r, N, sigma) — post-proceso de la curva,
# NO un gate (T5): reporta la curva entera igual, esto es solo una lectura
# derivada por interpolación en log(rho).
# ============================================================
def derivar_rho_c(filas_celda: list[dict]) -> dict:
    filas_ord = sorted(filas_celda, key=lambda f: f["rho_ratio"])  # ascendente
    rhos = np.array([f["rho_ratio"] for f in filas_ord])
    Ps = np.array([f["P_real_mean"] for f in filas_ord])
    piso = float(Ps[-1])   # rho=1 (rho0), más lavado (si el barrido cubre rho0)
    techo = float(Ps[0])   # rho mínimo del barrido, más congelado
    if techo - piso < 1e-9:
        return {"existe": False, "rho_c": None, "motivo": "curva plana (techo≈piso)",
                "piso_P": piso, "techo_P": techo}
    mid = (piso + techo) / 2.0
    # recorrer de rho alto a rho bajo (índices descendentes) buscando el primer
    # cruce hacia arriba de la mediana
    cross_idx = None
    for i in range(len(rhos) - 1, 0, -1):
        p_hi_rho, p_lo_rho = Ps[i], Ps[i - 1]  # rhos[i] > rhos[i-1]
        if (p_hi_rho < mid) and (p_lo_rho >= mid):
            cross_idx = i
            break
    if cross_idx is None:
        return {"existe": False, "rho_c": None,
                "motivo": "no se detecta cruce del punto medio en el rango barrido",
                "piso_P": piso, "techo_P": techo, "mid_P": mid}
    x0, x1 = np.log(rhos[cross_idx - 1]), np.log(rhos[cross_idx])
    y0, y1 = Ps[cross_idx - 1], Ps[cross_idx]
    t = (mid - y0) / (y1 - y0) if (y1 - y0) != 0 else 0.5
    log_rho_c = x0 + t * (x1 - x0)
    rho_c = float(np.exp(log_rho_c))
    return {"existe": True, "rho_c": rho_c, "piso_P": piso, "techo_P": techo, "mid_P": mid,
            "bracket_rho": [float(rhos[cross_idx - 1]), float(rhos[cross_idx])]}


# ============================================================
# Orquestación
# ============================================================
def build_tasks(N_list, rho_grid, r_targets, sigma_list, pasos_por_N, D_map):
    tasks = []
    for N in N_list:
        pasos = pasos_por_N[N]
        for rho_ratio in rho_grid:
            D = D_map[(N, rho_ratio)]
            for r_tgt in r_targets:
                for sigma in sigma_list:
                    tasks.append({
                        "N": N, "rho_ratio": rho_ratio, "r_target": r_tgt,
                        "sigma_ruido": sigma, "D": D, "pasos": pasos,
                    })
    return tasks


def run_production(N_list, rho_grid, r_targets, sigma_list, tag: str, max_workers: int) -> dict:
    t0 = time.time()

    print(f"[calibracion] calibrando reloj físico y D(rho) por N (referencia rho=rho0, sin ruido)...")
    pasos_por_N = {}
    calib_por_N = {}
    D_map = {}
    for N in N_list:
        cal = medir_pasos_lavado(N, EPS, SEEDS_CALIB, RATE0)
        pasos_por_N[N] = cal["pasos"]
        calib_por_N[N] = cal
        print(f"  N={N}: mediana_lavado={cal['mediana']} pasos_fijo={cal['pasos']} "
              f"lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}")
        for rho_ratio in rho_grid:
            rate = RATE0 * rho_ratio
            Dvals = [medir_D(N, EPS, s, rate) for s in range(SEEDS_CALIB)]
            D_map[(N, rho_ratio)] = float(np.mean(Dvals))
        print(f"  N={N}: D(rho) medido en {len(rho_grid)} puntos de rho "
              f"(rho_min={rho_grid[0]:.3e}->D={D_map[(N, rho_grid[0])]:.6g}, "
              f"rho_max={rho_grid[-1]:.3e}->D={D_map[(N, rho_grid[-1])]:.6g})")

    tasks = build_tasks(N_list, rho_grid, r_targets, sigma_list, pasos_por_N, D_map)
    print(f"[barrido] {len(tasks)} celdas x {len(SEEDS)} semillas x 2 ramas (REAL/NULL) "
          f"= {len(tasks) * len(SEEDS) * 2} corridas de campo. max_workers={max_workers}")

    filas = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(cell_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            filas.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(f"  [{done}/{len(tasks)}] celdas completas, {elapsed:.1f}s transcurridos", flush=True)

    # ---- control r=0 a rho=rho0 (referencia) debe lavar ----
    ctrl_rows = [f for f in filas if f["r_target"] == 0.0 and abs(f["rho_ratio"] - 1.0) < 1e-9
                 and f["sigma_ruido"] == 0.0]
    control_r0_ok = bool(ctrl_rows) and all(f["P_real_mean"] < 0.15 for f in ctrl_rows)
    control_r0_detail = [{"N": f["N"], "P_real_mean": f["P_real_mean"]} for f in ctrl_rows]

    # ---- derivar rho_c por celda (r, N, sigma) ----
    celdas = {}
    for f in filas:
        key = (f["r_target"], f["N"], f["sigma_ruido"])
        celdas.setdefault(key, []).append(f)
    rho_c_por_celda = []
    for (r_tgt, N, sigma), fs in sorted(celdas.items()):
        d = derivar_rho_c(fs)
        d.update({"r_target": r_tgt, "N": N, "sigma_ruido": sigma})
        rho_c_por_celda.append(d)

    # ---- estabilidad de rho_c entre N y sigma, a r fijo (solo donde existe) ----
    estabilidad_por_r = {}
    for r_tgt in r_targets:
        vals = [d["rho_c"] for d in rho_c_por_celda if d["r_target"] == r_tgt and d["existe"]]
        n_celdas_r = sum(1 for d in rho_c_por_celda if d["r_target"] == r_tgt)
        if len(vals) >= 2:
            log_vals = np.log10(vals)
            estabilidad_por_r[str(r_tgt)] = {
                "n_con_rho_c": len(vals), "n_celdas_totales": n_celdas_r,
                "rho_c_min": float(min(vals)), "rho_c_max": float(max(vals)),
                "rango_decadas": float(log_vals.max() - log_vals.min()),
                "estable_dentro_1_decada": bool((log_vals.max() - log_vals.min()) <= 1.0),
            }
        else:
            estabilidad_por_r[str(r_tgt)] = {
                "n_con_rho_c": len(vals), "n_celdas_totales": n_celdas_r,
                "rho_c_min": None, "rho_c_max": None,
                "rango_decadas": None, "estable_dentro_1_decada": None,
            }

    payload = {
        "experimento": "F4-4 densidad crítica de congelamiento",
        "tag": tag,
        "sello": {"RATE0": RATE0, "RHO0": RHO0, "EPS": EPS, "P_LAVADO": P_LAVADO,
                   "MARGEN_LAVADO": MARGEN_LAVADO},
        "barrido": {
            "rho_grid": rho_grid, "r_targets": r_targets, "N_list": N_list,
            "sigma_ruido_list": sigma_list, "seeds": SEEDS, "n_seeds": len(SEEDS),
            "n_celdas": len(tasks), "n_corridas_campo": len(tasks) * len(SEEDS) * 2,
        },
        "calibracion_reloj_por_N": calib_por_N,
        "pasos_por_N": pasos_por_N,
        "control_r0_lava": control_r0_ok,
        "control_r0_detail": control_r0_detail,
        "filas": filas,
        "rho_c_por_celda": rho_c_por_celda,
        "estabilidad_rho_c_por_r": estabilidad_por_r,
        "pre_inscrito": {
            "lectura_A": "existe rho_c: transición P alto (bajo rho) / P bajo (alto rho), "
                         "estable <=1 decada al variar N y sigma_ruido",
            "lectura_B": "no existe rho_c: curva plana o dominada por dispersión de semilla",
            "ambas_son_hallazgo": True,
        },
        "runtime_seconds": time.time() - t0,
        "generated_at_unix": time.time(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion"], default="produccion", nargs="?")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        rho_grid = np.geomspace(1e-6, 1.0, 5).tolist()
        r_targets = [0.0, 1.0, 10.0]
        N_list = [200]
        sigma_list = [0.0, 1e-3]
        tag = "smoke"
    else:
        rho_grid = RHO_GRID
        r_targets = R_TARGETS
        N_list = N_LIST
        sigma_list = SIGMA_RUIDO_LIST
        tag = "produccion"

    import os
    max_workers = args.workers or min(16, os.cpu_count() or 4)

    print(f"=== F4-4 densidad crítica — modo={args.mode} workers={max_workers} ===")
    payload = run_production(N_list, rho_grid, r_targets, sigma_list, tag, max_workers)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    print(f"control_r0_lava={payload['control_r0_lava']}  detail={payload['control_r0_detail']}")
    for d in payload["rho_c_por_celda"]:
        print(f"  r={d['r_target']:<6} N={d['N']:<4} sigma={d['sigma_ruido']:<8} "
              f"existe_rho_c={d['existe']}  rho_c={d.get('rho_c')}")
    print("\nestabilidad rho_c por r (entre N y sigma_ruido):")
    for r_str, info in payload["estabilidad_rho_c_por_r"].items():
        print(f"  r={r_str}: {info}")

    out_json = OUT_DIR / f"F4_4_resultado_{args.mode}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON -> {out_json}")
    print(f"[elapsed] {payload['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
