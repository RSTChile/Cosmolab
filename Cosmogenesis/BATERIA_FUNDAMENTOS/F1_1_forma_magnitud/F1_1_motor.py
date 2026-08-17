#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1_1_motor.py — F1-1: "Persistencia por autocorrelación de forma contra NULL barajado"
========================================================================================

Experimento F1-1 de la BATERÍA DE FUNDAMENTOS (Enfoque 1). Ver especificación
congelada en:
  BATERIA_FUNDAMENTOS/F1_1_forma_magnitud/PROTOCOLO_F1-1_PREREGISTRO.md
y la adenda de ejecución (motor optimizado, reducción documentada para N=800/1600):
  BATERIA_FUNDAMENTOS/F1_1_forma_magnitud/PROTOCOLO_F1-1_ADENDA_EJECUCION.md

Este motor NO reimplementa la física ni el observable con una fórmula distinta:
los IMPORTA sin modificación desde `cs074_rcruz.py` (código base, archivo ajeno,
no se toca):
  - campo_inicial()      → siembra ε en fondo=1 con perturbación multi-Fourier
  - paso_difusion()      → difusión vectorizada por aristas vivas
  - paso_expansion()     → corte Bernoulli de aristas (expansión)
  - evolucionar()        → corre la dinámica; null=True permuta φ al final
  - medir_D()            → D medido del propio campo (1 paso, H=0)
  - medir_pasos_lavado() → pasos calibrados por lavado medido (no impuesto)
  - persistencia()       → OBSERVABLE = corr(φ,roll(φ,1)) · var(φ)/var(φ₀)
  - corrida()            → una corrida completa (REAL o NULL), devuelve dict con "P"

MOTOR BATCHED (producción): el barrido literal (una corrida() por cada punto
ε×r×N×semilla×{REAL,NULL}) escala mal a N grande (pasos ~ N², medido). Se
implementó un motor vectorizado (`paso_difusion_batch`, `persistencia_batch`,
`trayectoria_regimen`) que reproduce EXACTAMENTE (validado, diff ≤ 1e-15) los
mismos números que llamar `base.corrida()` uno por uno, explotando dos
identidades matemáticas (no aproximaciones):
  1. REAL y NULL comparten la MISMA trayectoria (evolucionar(null=True) con
     una semilla dada = evolucionar(null=False) + una permutación final).
  2. Para ε y semilla fijos, base.corrida() usa un rng FRESCO con la MISMA
     semilla para cada r → el consumo de aleatoriedad es idéntico entre
     valores de r que caen en el mismo "régimen" de paso_expansion (H≤0,
     H≥1, o 0<H<1 — paso_expansion() de cs074_rcruz.py NO consume ningún
     random() en los dos primeros regímenes, solo en el tercero). Se separa
     el barrido en esos 3 regímenes y se vectoriza cada uno correctamente.
Ver validación reproducible en `resultados/F1_1_validacion_batch*.py` y el
benchmark en `resultados/F1_1_benchmark_costo_real.py` (ambos ejecutados y
verificados PASA por CC antes de usarse en producción).

Uso:
  python3 F1_1_motor.py smoke
  python3 F1_1_motor.py produccion --N 200
  python3 F1_1_motor.py produccion --N 400
  python3 F1_1_motor.py produccion --N 800
  python3 F1_1_motor.py produccion --N 1600
  python3 F1_1_motor.py resumen        # combina los 4 JSON de producción + evalúa PASS
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COSMOGENESIS_DIR = HERE.parent.parent  # .../Cosmogenesis
OUT_DIR = HERE / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Importar cs074_rcruz.py SIN MODIFICARLO (por ruta explícita, robusto a cwd) ---
_CS074_PATH = COSMOGENESIS_DIR / "cs074_rcruz.py"
if not _CS074_PATH.exists():
    raise SystemExit(f"[F1-1] ERROR: no se encuentra el código base en {_CS074_PATH}")
_spec = importlib.util.spec_from_file_location("cs074_rcruz_base", str(_CS074_PATH))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # type: ignore

SEED_BASE = 1000


def log(msg: str, log_path: Path | None = None):
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Grid pre-registrado (PROTOCOLO_F1-1_PREREGISTRO.md, sección 4) — congelado
# ---------------------------------------------------------------------------
def eps_grid_full():
    return [0.0] + [float(v) for v in np.logspace(-12, 0, 12)]


def r_grid_full():
    return [
        0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 3.0,
        5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
    ]


def r_grid_reducido_1600():
    """15 puntos (mínimo pre-registrado), fino cerca de r≈1. Ver ADENDA_EJECUCION §3."""
    return [0.0, 0.1, 0.3, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0]


def eps_grid_smoke():
    return [0.0, 1e-9, 1e-3, 1.0]


def r_grid_smoke():
    return [0.0, 0.3, 0.9, 1.0, 1.1, 2.0, 10.0, 100.0]


# N -> (eps_list, r_list, semillas). Ver PROTOCOLO_F1-1_ADENDA_EJECUCION.md §3
# para la justificación de la reducción en N=800 (semillas 12->8) y N=1600
# (semillas 12->4, r 34->15pts). N=200/400 corren el grid COMPLETO.
def config_produccion(N):
    eps_list = eps_grid_full()
    if N in (200, 400):
        return eps_list, r_grid_full(), 12
    elif N == 800:
        return eps_list, r_grid_full(), 8
    elif N == 1600:
        return eps_list, r_grid_reducido_1600(), 4
    else:
        raise SystemExit(f"N={N} no está en el grid pre-registrado {N_LIST_FULL}")


N_LIST_FULL = [200, 400, 800, 1600]
SEMILLAS_SMOKE = 4


# ---------------------------------------------------------------------------
# MOTOR BATCHED — funciones vectorizadas, matemáticamente idénticas a
# cs074_rcruz.py (validado en resultados/F1_1_validacion_batch3_*.py).
# ---------------------------------------------------------------------------
def paso_difusion_batch(phi, activo):
    """Idéntica a base.paso_difusion() pero con eje batch adicional (axis=-1
    sigue siendo el eje espacial N; la fila extra es el batch de r's)."""
    left = np.roll(phi, 1, axis=-1)
    right = np.roll(phi, -1, axis=-1)
    e_left = np.roll(activo, 1, axis=-1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def persistencia_batch(phi, contraste0_arr):
    """Idéntica a base.persistencia() vectorizada por filas."""
    a = phi
    b = np.roll(phi, 1, axis=-1)
    a_mean = a.mean(axis=-1, keepdims=True)
    b_mean = b.mean(axis=-1, keepdims=True)
    cov = ((a - a_mean) * (b - b_mean)).mean(axis=-1)
    sa = a.std(axis=-1)
    sb = b.std(axis=-1)
    denom = sa * sb
    c = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    v = phi.var(axis=-1) / (contraste0_arr ** 2)
    ok = (contraste0_arr > 0) & (phi.std(axis=-1) > 1e-12)
    return np.where(ok, c * v, 0.0)


def H_de_r(r, D):
    if D > 0:
        return float(min(r * D, 1.0))
    return 0.0 if r == 0 else 1.0


def trayectoria_regimen(N, eps, seed, pasos, r_list, D):
    """Para (N, eps, seed) fijos, devuelve {r: (P_real, P_null)} para TODO
    r_list, EXACTAMENTE igual a llamar base.corrida(N,eps,H(r),pasos,seed,
    null=False/True) por cada r por separado (validado, ver
    resultados/F1_1_validacion_batch3_3regimenes_PASA.py)."""
    r_zero = [r for r in r_list if H_de_r(r, D) <= 0.0]
    r_full = [r for r in r_list if H_de_r(r, D) >= 1.0]
    r_mid = [r for r in r_list if 0.0 < H_de_r(r, D) < 1.0]
    results = {}

    if r_zero:
        rng_z = np.random.default_rng(seed)
        phi0_z, _ = base.campo_inicial(N, eps, rng_z)
        activo_z = np.ones(N, dtype=bool)
        phi_z = phi0_z.copy()
        c0_z = float(phi0_z.std())
        for _ in range(pasos):
            phi_z = base.paso_difusion(phi_z, activo_z)
        P_real_z = base.persistencia(phi_z, c0_z)
        idx_z = rng_z.permutation(N)
        P_null_z = base.persistencia(phi_z[idx_z], c0_z)
        for r in r_zero:
            results[r] = (P_real_z, P_null_z)

    if r_full:
        rng_f = np.random.default_rng(seed)
        phi0_f, _ = base.campo_inicial(N, eps, rng_f)
        activo_init = np.ones(N, dtype=bool)
        phi_f = base.paso_difusion(phi0_f, activo_init)  # 1 solo paso, luego congela
        c0_f = float(phi0_f.std())
        P_real_f = base.persistencia(phi_f, c0_f)
        idx_f = rng_f.permutation(N)
        P_null_f = base.persistencia(phi_f[idx_f], c0_f)
        for r in r_full:
            results[r] = (P_real_f, P_null_f)

    if r_mid:
        rng_m = np.random.default_rng(seed)
        phi0_m, _ = base.campo_inicial(N, eps, rng_m)
        R = len(r_mid)
        activo_m = np.ones((R, N), dtype=bool)
        phi_m = np.tile(phi0_m, (R, 1))
        c0_m = float(phi0_m.std())
        H_arr = np.array([H_de_r(r, D) for r in r_mid]).reshape(R, 1)
        for _ in range(pasos):
            phi_m = paso_difusion_batch(phi_m, activo_m)
            u = rng_m.random(N)
            cortar = activo_m & (u[None, :] < H_arr)
            activo_m = activo_m & ~cortar
        c0_arr = np.full(R, c0_m)
        P_real_m = persistencia_batch(phi_m, c0_arr)
        idx_m = rng_m.permutation(N)
        P_null_m = persistencia_batch(phi_m[:, idx_m], c0_arr)
        for i, r in enumerate(r_mid):
            results[r] = (float(P_real_m[i]), float(P_null_m[i]))

    return results


# ---------------------------------------------------------------------------
# Barrido de producción para un N dado (usa trayectoria_regimen, batched)
# ---------------------------------------------------------------------------
def barrido_N(N, eps_list, r_list, semillas, log_path, eps_cal=1e-3, semillas_cal=4):
    t0 = time.time()
    cal = base.medir_pasos_lavado(N, eps_cal, max(semillas_cal, 4))
    pasos = cal["pasos"]
    log(f"[N={N}] calibracion lavado: eps_cal={eps_cal} pasos={pasos} "
        f"mediana={cal['mediana']} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        log_path)

    # filas indexadas por (eps, r) acumulando P_real/P_null por semilla
    acumulado = {}  # (eps, r) -> {"Preal": [...], "Pnull": [...]}
    for r in r_list:
        for eps in eps_list:
            acumulado[(eps, r)] = {"Preal": [], "Pnull": []}

    n_eps = len(eps_list)
    for ei, eps in enumerate(eps_list):
        D = float(np.mean([base.medir_D(N, eps, s) for s in range(semillas)]))
        for s in range(semillas):
            seed = SEED_BASE + s
            res = trayectoria_regimen(N, eps, seed, pasos, r_list, D)
            for r, (Preal, Pnull) in res.items():
                acumulado[(eps, r)]["Preal"].append(Preal)
                acumulado[(eps, r)]["Pnull"].append(Pnull)
        elapsed = time.time() - t0
        log(f"[N={N}] eps {ei+1}/{n_eps} (eps={eps:.3e}, D={D:.6e}) completado. "
            f"t={elapsed:.1f}s ({elapsed/60:.1f}min)", log_path)

    filas = []
    for r in r_list:
        for eps in eps_list:
            Preal = np.array(acumulado[(eps, r)]["Preal"], dtype=float)
            Pnull = np.array(acumulado[(eps, r)]["Pnull"], dtype=float)
            sd = float(np.sqrt((Preal.var() + Pnull.var()) / 2.0))
            sd = max(sd, 1.0 / max(len(Preal), 1))
            z = float((Preal.mean() - Pnull.mean()) / sd)
            filas.append({
                "N": N,
                "eps": eps,
                "r_target": r,
                "pasos": pasos,
                "semillas": semillas,
                "P_real_mean": float(Preal.mean()),
                "P_real_std": float(Preal.std()),
                "P_null_mean": float(Pnull.mean()),
                "P_null_std": float(Pnull.std()),
                "P_real_seeds": [round(float(x), 6) for x in Preal],
                "P_null_seeds": [round(float(x), 6) for x in Pnull],
                "z": round(z, 4),
            })

    elapsed = time.time() - t0
    return filas, cal, pasos, elapsed


def evaluar_criterio(filas):
    """Evaluación MECÁNICA del criterio de PASS pre-registrado (sección 7 del
    protocolo). NO es el veredicto de lectura final — eso lo da CS con la curva."""
    rows_eps0 = [f for f in filas if f["eps"] == 0.0]
    frac_eps0_ok = float(np.mean([f["P_real_mean"] < 0.05 for f in rows_eps0])) if rows_eps0 else None
    control_eps0_pass = (frac_eps0_ok is not None) and (frac_eps0_ok >= 0.95)

    rows_r0 = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    mean_P_r0 = float(np.mean([f["P_real_mean"] for f in rows_r0])) if rows_r0 else None
    control_r0_pass = (mean_P_r0 is not None) and (mean_P_r0 < 0.15)

    rows_null = [f for f in filas if f["r_target"] >= 10.0 and f["eps"] > 1e-6]
    frac_null_cae = float(np.mean([f["z"] >= 3.0 for f in rows_null])) if rows_null else None
    null_cae_pass = (frac_null_cae is not None) and (frac_null_cae >= 0.5)

    if null_cae_pass and control_r0_pass and control_eps0_pass:
        veredicto = "PASS_mecanico"
    else:
        veredicto = "FAIL_o_NEGATIVO_mecanico"

    return {
        "frac_eps0_P_bajo_0.05": frac_eps0_ok,
        "control_eps0_pass": control_eps0_pass,
        "mean_P_real_r0_eps_gt0": mean_P_r0,
        "control_r0_pass": control_r0_pass,
        "frac_null_cae_z_ge_3_en_r_ge_10_eps_gt_1e-6": frac_null_cae,
        "null_cae_pass": null_cae_pass,
        "veredicto_mecanico_PASS_FAIL": veredicto,
        "nota": "Evaluación mecánica del gate pre-registrado. El veredicto de "
                "LECTURA final (qué significa) lo da CS con la curva cruda; CC "
                "no adjudica.",
    }


def run_smoke():
    log_path = OUT_DIR / "F1_1_log_ejecucion.txt"
    log("=== F1-1 SMOKE: inicio (motor batched) ===", log_path)
    t0 = time.time()
    N = 100
    eps_list = eps_grid_smoke()
    r_list = r_grid_smoke()
    semillas = SEMILLAS_SMOKE
    filas, cal, pasos, elapsed = barrido_N(N, eps_list, r_list, semillas, log_path,
                                            eps_cal=1e-3, semillas_cal=semillas)
    criterio = evaluar_criterio(filas)
    result = {
        "experimento": "F1-1",
        "modo": "smoke",
        "motor": "batched",
        "N": N,
        "eps_list": eps_list,
        "r_list": r_list,
        "semillas": semillas,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "filas": filas,
        "criterio_pass_mecanico": criterio,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / "F1_1_smoke_resultado.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-1 SMOKE: fin. elapsed={elapsed:.1f}s -> {out} ===", log_path)
    log(f"[SMOKE] criterio: {criterio}", log_path)
    log(f"=== F1-1 SMOKE total wall time: {time.time()-t0:.1f}s ===", log_path)


def run_produccion(N):
    if N not in N_LIST_FULL:
        raise SystemExit(f"N={N} no está en el grid pre-registrado {N_LIST_FULL}")
    log_path = OUT_DIR / "F1_1_log_ejecucion.txt"
    eps_list, r_list, semillas = config_produccion(N)
    reducido = " (REDUCIDO, ver ADENDA_EJECUCION)" if (semillas < 12 or len(r_list) < 34) else " (grid COMPLETO)"
    log(f"=== F1-1 PRODUCCION N={N}{reducido}: inicio. "
        f"eps={len(eps_list)}pts r={len(r_list)}pts semillas={semillas} ===", log_path)
    filas, cal, pasos, elapsed = barrido_N(N, eps_list, r_list, semillas, log_path,
                                            eps_cal=1e-3, semillas_cal=semillas)
    criterio = evaluar_criterio(filas)
    result = {
        "experimento": "F1-1",
        "modo": "produccion",
        "motor": "batched",
        "N": N,
        "eps_list": eps_list,
        "r_list": r_list,
        "semillas": semillas,
        "grid_reducido_vs_preregistro": (semillas < 12 or len(r_list) < 34),
        "seed_base": SEED_BASE,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "n_filas": len(filas),
        "filas": filas,
        "criterio_pass_mecanico": criterio,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / f"F1_1_produccion_N{N}_resultado.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-1 PRODUCCION N={N}: fin. elapsed={elapsed:.1f}s ({elapsed/60:.1f}min) -> {out} ===",
        log_path)
    log(f"[N={N}] criterio: {criterio}", log_path)


def run_resumen():
    log_path = OUT_DIR / "F1_1_log_ejecucion.txt"
    log("=== F1-1 RESUMEN: combinando producción de los 4 N ===", log_path)
    todas_filas = []
    por_N = {}
    faltantes = []
    for N in N_LIST_FULL:
        p = OUT_DIR / f"F1_1_produccion_N{N}_resultado.json"
        if not p.exists():
            faltantes.append(N)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        todas_filas.extend(data["filas"])
        por_N[N] = {
            "pasos": data["pasos"],
            "semillas": data["semillas"],
            "n_r_puntos": len(data["r_list"]),
            "grid_reducido_vs_preregistro": data.get("grid_reducido_vs_preregistro", False),
            "elapsed_s": data["elapsed_s"],
            "criterio_pass_mecanico": data["criterio_pass_mecanico"],
            "calibracion_lavado": data["calibracion_lavado"],
        }
    if faltantes:
        log(f"[RESUMEN] AVISO: faltan resultados de N={faltantes}, resumen parcial", log_path)

    criterio_global = evaluar_criterio(todas_filas) if todas_filas else None

    # r* aproximado por N: r_target donde z cruza 3.0 por primera vez (a eps=1e-3, referencia)
    r_estrella_por_N = {}
    for N in por_N:
        filas_N = [f for f in todas_filas if f["N"] == N and abs(f["eps"] - 1e-3) < 1e-15]
        filas_N.sort(key=lambda f: f["r_target"])
        r_star = None
        for f in filas_N:
            if f["z"] >= 3.0:
                r_star = f["r_target"]
                break
        r_estrella_por_N[N] = r_star

    resumen = {
        "experimento": "F1-1",
        "modo": "resumen",
        "N_incluidos": list(por_N.keys()),
        "N_faltantes": faltantes,
        "n_filas_totales": len(todas_filas),
        "por_N": por_N,
        "r_estrella_aprox_por_N_en_eps_1e-3": r_estrella_por_N,
        "criterio_pass_mecanico_global": criterio_global,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    out = OUT_DIR / "F1_1_produccion_resumen.json"
    out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"=== F1-1 RESUMEN: fin -> {out} ===", log_path)
    log(f"[RESUMEN] criterio global: {criterio_global}", log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["smoke", "produccion", "resumen"])
    ap.add_argument("--N", type=int, default=None)
    args = ap.parse_args()

    if args.modo == "smoke":
        run_smoke()
    elif args.modo == "produccion":
        if args.N is None:
            for N in N_LIST_FULL:
                run_produccion(N)
        else:
            run_produccion(args.N)
    elif args.modo == "resumen":
        run_resumen()


if __name__ == "__main__":
    main()
