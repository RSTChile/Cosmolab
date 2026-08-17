#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_6_4_motor.py — Sensibilidad de X a la definición de equilibrio de referencia
=================================================================================

Implementa lo pre-registrado en PROTOCOLO_E5.6-4_PREREGISTRO.md (leer ese archivo
primero; este motor NO se ejecuta antes de que el protocolo esté congelado en disco).

Reusa (importa, NO copia ni edita) la física de cs074_rcruz.py:
  campo_inicial, paso_difusion, paso_expansion, medir_D, medir_pasos_lavado.

Pregunta: la exergía X se mide como desviación de un "equilibrio" de referencia.
¿Cambia el veredicto de persistencia (persiste/no bajo r=H/D) según CUÁL referencia se
use? Se implementan 3 definiciones (protocolo sección 3):
  (A) REF_GLOBAL   — media global fija de phi en t=0, constante en el tiempo.
  (B) REF_LOCAL    — media móvil espacial (ventana circular W=21), recalculada sobre
                      el propio estado que se mide (t=0 y t=final por separado).
  (C) REF_DINÁMICA — media móvil exponencial en el TIEMPO, arranca igual que (A) y
                      sigue al campo con retraso (alpha = 20/pasos_fijo) durante TODA
                      la simulación.

Las 3 X se calculan sobre la MISMA trayectoria física simulada (mismo phi(t)) — solo
cambia la vara con la que se mide "cuánto se desvió del equilibrio". Esto aísla el
efecto de la definición de referencia del efecto de la física (que es idéntica).

*** RE-CORRIDO 2026-07-25 (ARREGLO 3, ver ADENDA en PROTOCOLO_E5.6-4_PREREGISTRO.md) ***
Se agrega una CUARTA medición en paralelo: Xh, la definición canónica de exergía del
proyecto (`exergia_X` de `BATERIA_ENFOQUE5/_observables_homologadas.py`, mean-square-
deviation respecto a phi_eq=1 fijo, NO reimplementada a mano). Se calcula sobre phi_f y
phi_null crudos (sin restar ninguna referencia -- phi_eq=1 ya es la referencia implícita
de la fórmula), con el MISMO NULL y el MISMO criterio de veredicto que las 3 referencias
existentes, que NO se tocan. Ver la adenda del protocolo para la predicción pre-registrada
(Xh es invariante a permutación => el NULL por-permutación de este diseño no tiene poder
para esta definición -- se predijo y se reporta tal cual, no se oculta).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # .../Cosmogenesis
sys.path.insert(0, str(ROOT))

from cs074_rcruz import (  # noqa: E402
    campo_inicial,
    paso_difusion,
    paso_expansion,
    medir_D,
    medir_pasos_lavado,
    P_LAVADO,
    MARGEN_LAVADO,
)
from BATERIA_ENFOQUE5._observables_homologadas import exergia_X as exergia_X_canonica  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes congeladas en el pre-registro (protocolo secciones 3, 5, 6)
# ---------------------------------------------------------------------------
N = 200
SEMILLAS = 12
EPS_LIST = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0]
R_TARGETS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
W_LOCAL = 21  # round(N/10)=20 forzado a impar mas cercano
Z_THR = 2.0
NULL_SEED_OFFSET = 500_000
REF_TYPES = ["global", "local", "dinamica"]


def suavizado_circular(v: np.ndarray, W: int) -> np.ndarray:
    """Media movil circular de ventana W, kernel uniforme, envoltura de anillo."""
    N_ = v.size
    half = W // 2
    acc = np.zeros(N_, dtype=float)
    for k in range(-half, W - half):
        acc += np.roll(v, -k)
    return acc / W


def exergia(d_f: np.ndarray, d0: np.ndarray) -> float:
    """X_ref = c * v ; c=coherencia espacial de la desviacion final, v=varianza
    retenida relativa a la desviacion inicial. Guardian: Var(d0)~0 -> X=0."""
    var0 = float(np.var(d0))
    if var0 <= 1e-18:
        return 0.0, 0.0
    if float(np.std(d_f)) <= 1e-12:
        return 0.0, 0.0
    c = np.corrcoef(d_f, np.roll(d_f, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(np.var(d_f) / var0)
    return float(c * v), v


def simular(N_: int, eps: float, H: float, pasos: int, seed: int, alpha_dyn: float):
    """UNA trayectoria fisica (identica a paso_difusion/paso_expansion de la base),
    rastreando ademas ref_dinamica paso a paso. Devuelve phi0, phi_final, ref_dyn_final."""
    rng = np.random.default_rng(seed)
    phi0, _x = campo_inicial(N_, eps, rng)
    activo = np.ones(N_, dtype=bool)
    phi = phi0.copy()
    ref_dyn = np.full(N_, float(phi0.mean()))
    suma0 = float(phi0.sum())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
        ref_dyn = (1.0 - alpha_dyn) * ref_dyn + alpha_dyn * phi
    deriva_E = abs(float(phi.sum()) - suma0) / (abs(suma0) if suma0 != 0 else 1.0)
    return phi0, phi, ref_dyn, deriva_E, activo


def celda(eps: float, r_tgt: float, D: float, pasos: int, alpha_dyn: float, seed: int):
    H = float(min(r_tgt * D, 1.0)) if D > 0 else (0.0 if r_tgt == 0 else 1.0)
    phi0, phi_f, ref_dyn_f, deriva_E, _activo = simular(N, eps, H, pasos, seed, alpha_dyn)

    # NULL: permutacion determinista derivada de seed+offset, aplicada a phi_f (protocolo 4)
    rng_null = np.random.default_rng(seed + NULL_SEED_OFFSET)
    phi_null = rng_null.permutation(phi_f)

    ref_g0 = np.full(N, float(phi0.mean()))  # ref_global, fija, = mean(phi0)

    out = {}
    # --- (A) GLOBAL ---
    d0 = phi0 - ref_g0
    d_real = phi_f - ref_g0
    d_null = phi_null - ref_g0
    X_real, v_real = exergia(d_real, d0)
    X_null, v_null = exergia(d_null, d0)
    out["global"] = (X_real, X_null, v_real, v_null)

    # --- (B) LOCAL (ventana movil, recalculada sobre el estado que se mide) ---
    ref_l0 = suavizado_circular(phi0, W_LOCAL)
    d0 = phi0 - ref_l0
    ref_lf_real = suavizado_circular(phi_f, W_LOCAL)
    ref_lf_null = suavizado_circular(phi_null, W_LOCAL)
    d_real = phi_f - ref_lf_real
    d_null = phi_null - ref_lf_null
    X_real, v_real = exergia(d_real, d0)
    X_null, v_null = exergia(d_null, d0)
    out["local"] = (X_real, X_null, v_real, v_null)

    # --- (C) DINAMICA (ref_dyn viene de la trayectoria REAL; NULL solo baraja phi_f) ---
    d0 = phi0 - ref_g0  # ref_dyn(0) = mean(phi0) = ref_g0, por construccion
    d_real = phi_f - ref_dyn_f
    d_null = phi_null - ref_dyn_f
    X_real, v_real = exergia(d_real, d0)
    X_null, v_null = exergia(d_null, d0)
    out["dinamica"] = (X_real, X_null, v_real, v_null)

    # --- (D) CANONICA (ARREGLO 3, ver ADENDA protocolo) ---
    # Xh(phi) = (1/N)*sum((phi_i-1)^2), sobre phi CRUDO, sin restar ninguna referencia
    # local/dinamica -- phi_eq=1 ya es la referencia fija implicita en la formula.
    # Mismo NULL (phi_null = permutacion de phi_f) que las 3 referencias de arriba.
    Xh_real = exergia_X_canonica(phi_f)
    Xh_null = exergia_X_canonica(phi_null)
    out["canonica"] = (Xh_real, Xh_null)

    return out, H, deriva_E, phi0, phi_f, phi_null


def barrido():
    t0 = time.time()
    cal = medir_pasos_lavado(N, 1e-3, SEMILLAS)
    pasos_fijo = cal["pasos"]
    alpha_dyn = 20.0 / pasos_fijo
    print(
        f"[calibracion] N={N} eps=1e-3 mediana_lavado={cal['mediana']} pasos_fijo={pasos_fijo} "
        f"alpha_dyn={alpha_dyn:.6g} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr, flush=True,
    )

    D_por_eps = {}
    for eps in EPS_LIST:
        if eps <= 0:
            D_por_eps[eps] = 0.0
        else:
            D_por_eps[eps] = float(np.mean([medir_D(N, eps, s) for s in range(SEMILLAS)]))
        print(f"[D] eps={eps:g} D={D_por_eps[eps]:.6g}", file=sys.stderr, flush=True)

    filas = []
    n_total = len(EPS_LIST) * len(R_TARGETS)
    n_done = 0
    for eps in EPS_LIST:
        D = D_por_eps[eps]
        for r_tgt in R_TARGETS:
            acc = {rt: {"X_real": [], "X_null": [], "v_real": [], "v_null": []} for rt in REF_TYPES}
            acc_canonica = {"X_real": [], "X_null": []}
            derivas = []
            H_used = None
            detalle_por_semilla = []  # ARREGLO 3: sum/sumsq de phi0,phi_f,phi_null por semilla
            detalle_repr = None       # arrays completos de la semilla representativa (s=0)
            for s in range(SEMILLAS):
                seed = 2000 + s
                out, H, deriva_E, phi0, phi_f, phi_null = celda(
                    eps, r_tgt, D, pasos_fijo, alpha_dyn, seed=seed
                )
                H_used = H
                derivas.append(deriva_E)
                for rt in REF_TYPES:
                    X_real, X_null, v_real, v_null = out[rt]
                    acc[rt]["X_real"].append(X_real)
                    acc[rt]["X_null"].append(X_null)
                    acc[rt]["v_real"].append(v_real)
                    acc[rt]["v_null"].append(v_null)
                Xh_real, Xh_null = out["canonica"]
                acc_canonica["X_real"].append(Xh_real)
                acc_canonica["X_null"].append(Xh_null)

                detalle_por_semilla.append({
                    "seed": seed,
                    "sum_phi0": float(phi0.sum()), "sumsq_phi0": float(np.sum(phi0 ** 2)),
                    "sum_phi_f": float(phi_f.sum()), "sumsq_phi_f": float(np.sum(phi_f ** 2)),
                    "sum_phi_null": float(phi_null.sum()), "sumsq_phi_null": float(np.sum(phi_null ** 2)),
                })
                if s == 0:
                    detalle_repr = {
                        "seed": seed,
                        "phi0": [round(float(x), 6) for x in phi0],
                        "phi_f": [round(float(x), 6) for x in phi_f],
                        "phi_null": [round(float(x), 6) for x in phi_null],
                    }

            fila = {
                "eps": eps, "r_target": r_tgt, "D": D, "H": H_used, "pasos": pasos_fijo,
                "deriva_E_mean": float(np.mean(derivas)), "deriva_E_max": float(np.max(derivas)),
            }
            for rt in REF_TYPES:
                Xr = np.array(acc[rt]["X_real"]); Xn = np.array(acc[rt]["X_null"])
                vr = np.array(acc[rt]["v_real"]); vn = np.array(acc[rt]["v_null"])
                sd = np.sqrt((Xr.var() + Xn.var()) / 2.0)
                sd = max(float(sd), 1.0 / SEMILLAS)
                z = float((Xr.mean() - Xn.mean()) / sd)
                veredicto = "persiste" if (z > Z_THR and Xr.mean() > Xn.mean()) else "no_persiste"
                fila[f"X_real_{rt}"] = float(Xr.mean())
                fila[f"X_real_{rt}_std"] = float(Xr.std())
                fila[f"X_real_{rt}_por_semilla"] = [round(float(x), 6) for x in Xr]
                fila[f"X_null_{rt}"] = float(Xn.mean())
                fila[f"X_null_{rt}_std"] = float(Xn.std())
                fila[f"var_ratio_real_{rt}"] = float(vr.mean())
                fila[f"var_ratio_null_{rt}"] = float(vn.mean())
                fila[f"z_{rt}"] = round(z, 4)
                fila[f"veredicto_{rt}"] = veredicto
            veredictos = {fila[f"veredicto_{rt}"] for rt in REF_TYPES}
            fila["invariante"] = (len(veredictos) == 1)

            # --- (D) CANONICA: mismo criterio de veredicto, calculado en paralelo ---
            Xhr = np.array(acc_canonica["X_real"]); Xhn = np.array(acc_canonica["X_null"])
            sd_h = np.sqrt((Xhr.var() + Xhn.var()) / 2.0)
            sd_h = max(float(sd_h), 1.0 / SEMILLAS)
            z_h = float((Xhr.mean() - Xhn.mean()) / sd_h)
            veredicto_h = "persiste" if (z_h > Z_THR and Xhr.mean() > Xhn.mean()) else "no_persiste"
            fila["X_real_canonica"] = float(Xhr.mean())
            fila["X_real_canonica_std"] = float(Xhr.std())
            fila["X_real_canonica_por_semilla"] = [round(float(x), 6) for x in Xhr]
            fila["X_null_canonica"] = float(Xhn.mean())
            fila["X_null_canonica_std"] = float(Xhn.std())
            fila["z_canonica"] = round(z_h, 4)
            fila["veredicto_canonica"] = veredicto_h
            fila["coincide_canonica_global"] = (veredicto_h == fila["veredicto_global"])

            fila["detalle_phi"] = {
                "por_semilla": detalle_por_semilla,
                "semilla_representativa": detalle_repr,
            }

            filas.append(fila)
            n_done += 1
            if n_done % 10 == 0 or n_done == n_total:
                print(f"[progreso] {n_done}/{n_total} eps={eps:g} r={r_tgt:g} t={time.time()-t0:.1f}s",
                      file=sys.stderr, flush=True)

    return filas, pasos_fijo, alpha_dyn, cal, D_por_eps, time.time() - t0


def analizar_invariancia(filas):
    no_triviales = [f for f in filas if f["eps"] > 0]
    n_inv = sum(1 for f in no_triviales if f["invariante"])
    n_tot = len(no_triviales)
    divergencias = []
    for f in no_triviales:
        if not f["invariante"]:
            divergencias.append({
                "eps": f["eps"], "r_target": f["r_target"],
                "veredicto_global": f["veredicto_global"],
                "veredicto_local": f["veredicto_local"],
                "veredicto_dinamica": f["veredicto_dinamica"],
                "z_global": f["z_global"], "z_local": f["z_local"], "z_dinamica": f["z_dinamica"],
            })
    frac_inv = n_inv / n_tot if n_tot else float("nan")

    # --- ARREGLO 3: coincidencia veredicto_canonica vs veredicto_global (celda a celda) ---
    n_coincide = sum(1 for f in no_triviales if f["coincide_canonica_global"])
    frac_coincide = n_coincide / n_tot if n_tot else float("nan")
    divergencias_canonica_global = []
    for f in no_triviales:
        if not f["coincide_canonica_global"]:
            divergencias_canonica_global.append({
                "eps": f["eps"], "r_target": f["r_target"],
                "veredicto_global": f["veredicto_global"],
                "veredicto_canonica": f["veredicto_canonica"],
                "z_global": f["z_global"], "z_canonica": f["z_canonica"],
                "X_real_canonica": f["X_real_canonica"], "X_null_canonica": f["X_null_canonica"],
            })

    return {
        "n_celdas_no_triviales": n_tot,
        "n_invariantes": n_inv,
        "fraccion_invariante": frac_inv,
        "PASS_90pct": frac_inv >= 0.90,
        "divergencias": divergencias,
        "canonica_vs_global": {
            "descripcion": "coincidencia celda-por-celda entre veredicto_canonica "
                            "(Xh=(1/N)sum((phi-1)^2), ARREGLO 3) y veredicto_global "
                            "(REF_GLOBAL, c*v con ref=mean(phi0) fija -- la unica de las "
                            "3 referencias originales que reprodujo la fisica esperada)",
            "n_coincide": n_coincide,
            "n_total": n_tot,
            "fraccion_coincide": frac_coincide,
            "divergencias": divergencias_canonica_global,
        },
    }


def main():
    filas, pasos_fijo, alpha_dyn, cal, D_por_eps, elapsed = barrido()
    inv = analizar_invariancia(filas)

    resultado = {
        "experimento": "E5.6-4 sensibilidad de X a la definicion de referencia",
        "protocolo": "PROTOCOLO_E5.6-4_PREREGISTRO.md",
        "N": N, "semillas": SEMILLAS,
        "eps_list": EPS_LIST, "r_targets": R_TARGETS,
        "W_local": W_LOCAL, "Z_THR": Z_THR, "alpha_dyn": alpha_dyn,
        "pasos_fijo": pasos_fijo, "calibracion_lavado": cal,
        "D_por_eps": D_por_eps,
        "ARREGLO_3_definicion_canonica": {
            "descripcion": "Xh(phi) = (1/N)*sum((phi_i-1)^2), importada de "
                            "BATERIA_ENFOQUE5/_observables_homologadas.py::exergia_X, "
                            "calculada en paralelo a las 3 referencias originales "
                            "(global/local/dinamica) sobre phi_f y phi_null crudos, "
                            "mismo NULL y mismo criterio de veredicto. Ver ADENDA en "
                            "PROTOCOLO_E5.6-4_PREREGISTRO.md para la predicción "
                            "pre-registrada (Xh es invariante a permutación => el NULL "
                            "por-permutación de este diseño no tiene poder para "
                            "distinguir real de null bajo esta definición).",
            "campos_por_fila": ["X_real_canonica", "X_null_canonica", "z_canonica",
                                 "veredicto_canonica", "coincide_canonica_global"],
        },
        "filas": filas,
        "elapsed_s": elapsed,
    }
    out_json = HERE / "E5_6_4_resultado_crudo.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    out_inv = HERE / "E5_6_4_invariancia.json"
    out_inv.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[archivo crudo] {out_json}", file=sys.stderr)
    print(f"[archivo invariancia] {out_inv}", file=sys.stderr)
    print(f"[invariancia] fraccion={inv['fraccion_invariante']:.4f} "
          f"({inv['n_invariantes']}/{inv['n_celdas_no_triviales']}) PASS_90pct={inv['PASS_90pct']}",
          file=sys.stderr)
    cvg = inv["canonica_vs_global"]
    print(f"[canonica_vs_global] fraccion={cvg['fraccion_coincide']:.4f} "
          f"({cvg['n_coincide']}/{cvg['n_total']})", file=sys.stderr)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
