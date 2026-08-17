#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.6-2 — "Exergía como energía libre: ¿se comporta como una energía libre real?"
==================================================================================

Motor propio del agente E5.6-2 (batería Enfoque 5, Tema 6). Ver el pre-registro
`PROTOCOLO_E5.6-2_PREREGISTRO.md` (congelado ANTES de escribir este archivo) para las
definiciones exactas de X, E, T_efectiva, S_ent, el barrido, el umbral de PASS y las
trampas evitadas — se implementan aquí tal cual quedaron congeladas.

--- Nota de implementación (post-preregistro, no cambia ninguna definición) ---------
La calibración (medir_D, medir_pasos_lavado) usa las funciones ESCALARES originales de
`cs074_rcruz.py` sin editar, importadas directamente. Para la evolución del campo se usa
una reimplementación VECTORIZADA que corre las 16 semillas de una celda (eps, r) EN
PARALELO como columnas de un array (N, S) en vez de 16 llamadas Python separadas —
puramente una optimización de velocidad (de ~4h estimadas a minutos), NO un cambio de
física. Se validó ANTES de usarla que `campo_inicial_batch`, `paso_difusion_batch` y
`paso_expansion_batch` son BIT-IDÉNTICAS a `cs074_rcruz.campo_inicial/paso_difusion/
paso_expansion` para una sola columna (S=1), incluida una evolución completa de 200
pasos con expansión activa (máx diferencia = 0.0, activo idéntico). Ver verificación en
el reporte final del agente.
--------------------------------------------------------------------------------------

Resumen de las cuatro cantidades (independientes entre sí, ver §4 del pre-registro):
  X_final       = (1/N) Σ (φ_i − 1)²                  — exergía, referencia FIJA φ_eq=1
  E_final       = (1/N) Σ φ_i²                          — energía total (segundo momento crudo)
  S_ent_final   = Shannon de p_i = φ_i² / Σφ_j²         — entropía de densidad de energía
  T_efectiva    = fracción de std borrada por UNA sonda de difusión adicional sobre el
                  estado final (φ_final, activo_final) — mide el acoplamiento térmico
                  remanente tras los cortes de expansión. NO usa la fórmula de X ni de S_ent.

Comparación (el corazón del experimento):
  F_pred_final = E_final − T_efectiva · S_ent_final
  PASS si X_final ≈ F_pred_final (correlación > 0.9 y error relativo mediano < 0.20,
  agregado por celda (r,ε) sobre ≥16 semillas — ver §7 del pre-registro).

Segundo método de verificación (espacio-T, §6.2 del pre-registro):
  T_implied = (E_final − X_final) / S_ent_final  (espejo algebraico del mismo test)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis

# --- Importar la base cs074_rcruz.py SIN editarla (usada para calibración: D, lavado) ---
spec = importlib.util.spec_from_file_location("cs074_rcruz", ROOT / "cs074_rcruz.py")
cs074 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs074)

medir_D = cs074.medir_D
medir_pasos_lavado = cs074.medir_pasos_lavado

# --- Constantes declaradas ANTES de correr (pre-registro §5/§7) -------------------
N = 200
SEMILLAS = 16
NOISE_REL = 0.02  # misma constante que E5.1-1, declarada, no ajustada después
EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0]
R_TARGETS = [0.0] + list(np.logspace(-3, 3, 25))
EPS_CALIBRACION_LAVADO = 1e-2
P_THR_LAVADO = 0.05
MARGEN_LAVADO = 1.15
SEED_BASE = 20_000

UMBRAL_CORR = 0.9
UMBRAL_ERR_REL = 0.20
S_ENT_MIN_VALIDO = 1e-6  # celdas con S_ent medio por debajo se excluyen del agregado (degeneradas)


# --- Reimplementaciones vectorizadas (S columnas = S semillas en paralelo) --------
# Validadas BIT-IDÉNTICAS a cs074_rcruz.{campo_inicial,paso_difusion,paso_expansion}
# para S=1 antes de usarse (ver nota de implementación arriba).

def campo_inicial_batch(N_, eps, rng, S):
    x = np.linspace(0.0, 1.0, N_, endpoint=False)
    fondo = np.ones((N_, S))
    if eps <= 0.0:
        return fondo, x
    pert = np.zeros((N_, S))
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi, size=S)
        pert += np.sin(2 * np.pi * m * x[:, None] + fase[None, :]) / m
    pert -= pert.mean(axis=0, keepdims=True)
    std = pert.std(axis=0, keepdims=True)
    pert = np.divide(pert, std, out=np.zeros_like(pert), where=std > 0)
    return fondo + eps * pert, x


def paso_difusion_batch(phi, activo):
    left = np.roll(phi, 1, axis=0)
    right = np.roll(phi, -1, axis=0)
    e_left = np.roll(activo, 1, axis=0)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo, H, rng):
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


def evolucionar_batch(phi, activo, H, pasos, rng, eps, noise_rel):
    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        activo = paso_expansion_batch(activo, H, rng)
        if eps > 0.0 and noise_rel > 0.0:
            phi = phi + rng.normal(0.0, noise_rel * eps, size=phi.shape)
    return phi, activo


def medir_finales_batch(phi, activo):
    """X, E, S_ent, T_efectiva por columna (fórmulas independientes, pre-registro §4)."""
    X = np.mean((phi - 1.0) ** 2, axis=0)
    E = np.mean(phi ** 2, axis=0)

    sq = phi ** 2
    total = sq.sum(axis=0)
    S_ent = np.zeros(phi.shape[1])
    valido = total > 0
    if np.any(valido):
        p = np.divide(sq, total, out=np.zeros_like(sq), where=valido[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p, out=np.zeros_like(p), where=p > 0), 0.0)
        S_ent = -terms.sum(axis=0)

    c0 = phi.std(axis=0)
    phi_prueba = paso_difusion_batch(phi, activo)
    c1 = phi_prueba.std(axis=0)
    T_efectiva = np.zeros_like(c0)
    m = c0 > 0
    T_efectiva[m] = np.maximum(0.0, (c0[m] - c1[m]) / c0[m])

    return X, E, S_ent, T_efectiva


def agregar(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"media": float("nan"), "mediana": float("nan"), "std": float("nan"), "n": 0}
    return {
        "media": float(arr.mean()),
        "mediana": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def main():
    t0 = time.time()

    print(f"[calibracion] midiendo pasos de lavado en eps={EPS_CALIBRACION_LAVADO} ...", file=sys.stderr, flush=True)
    cal = medir_pasos_lavado(N, EPS_CALIBRACION_LAVADO, max(SEMILLAS, 8), P_thr=P_THR_LAVADO)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps={EPS_CALIBRACION_LAVADO} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr,
        flush=True,
    )

    ss_base = np.random.SeedSequence(SEED_BASE)

    filas = []
    meta_por_eps = []
    n_celdas = len(EPS_LIST) * len(R_TARGETS)
    contador = 0
    t_ultimo_reporte = time.time()

    for eps in EPS_LIST:
        Ds = [medir_D(N, eps, 90_000 + s) for s in range(8)]
        D = float(np.mean(Ds))
        meta_por_eps.append({"eps": eps, "D": D, "D_std": float(np.std(Ds))})

        for r_tgt in R_TARGETS:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            child_ss = ss_base.spawn(1)[0]
            rng = np.random.default_rng(child_ss)

            phi, _ = campo_inicial_batch(N, eps, rng, SEMILLAS)
            activo = np.ones((N, SEMILLAS), dtype=bool)
            sigma_ini = phi.sum(axis=0)

            phi_f, activo_f = evolucionar_batch(phi, activo, H, pasos, rng, eps, NOISE_REL)
            sigma_fin = phi_f.sum(axis=0)
            deriva_E1 = np.abs(sigma_fin - sigma_ini) / (np.abs(sigma_ini) + 1e-12)

            X, E, S_ent, T_eff = medir_finales_batch(phi_f, activo_f)
            F_pred = E - T_eff * S_ent
            T_implied = np.full(SEMILLAS, np.nan)
            m = S_ent > 1e-9
            T_implied[m] = (E[m] - X[m]) / S_ent[m]
            frac_exp = 1.0 - activo_f.mean(axis=0)

            fila = {
                "eps": eps,
                "r_target": r_tgt,
                "H": H,
                "D": D,
                "r_eff": r_eff,
                "pasos": pasos,
                "X": agregar(X),
                "E": agregar(E),
                "S_ent": agregar(S_ent),
                "T_efectiva": agregar(T_eff),
                "F_pred": agregar(F_pred),
                "T_implied": agregar(T_implied),
                "deriva_E1": agregar(deriva_E1),
                "frac_exp": agregar(frac_exp),
            }
            filas.append(fila)
            contador += 1

            if time.time() - t_ultimo_reporte > 15:
                print(
                    f"[progreso] {contador}/{n_celdas} celdas "
                    f"({100*contador/n_celdas:.1f}%) elapsed={time.time()-t0:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                t_ultimo_reporte = time.time()

    # --- Agregado global para el veredicto §7 (por celda, medias sobre semillas) ---
    X_cell, F_cell = [], []
    for f in filas:
        if f["eps"] <= 0.0:
            continue
        if not np.isfinite(f["S_ent"]["media"]) or f["S_ent"]["media"] < S_ENT_MIN_VALIDO:
            continue
        X_cell.append(f["X"]["media"])
        F_cell.append(f["F_pred"]["media"])

    X_cell = np.array(X_cell)
    F_cell = np.array(F_cell)
    if X_cell.size >= 2 and np.std(X_cell) > 0 and np.std(F_cell) > 0:
        corr_global = float(np.corrcoef(X_cell, F_cell)[0, 1])
    else:
        corr_global = float("nan")

    err_rel = np.abs(X_cell - F_cell) / (np.abs(X_cell) + np.abs(F_cell) + 1e-9)
    err_rel_mediana = float(np.median(err_rel)) if err_rel.size else float("nan")
    err_rel_media = float(np.mean(err_rel)) if err_rel.size else float("nan")

    if np.isnan(corr_global):
        veredicto = "INDETERMINADO (celdas insuficientes tras excluir degeneradas)"
    elif corr_global > UMBRAL_CORR and err_rel_mediana < UMBRAL_ERR_REL:
        veredicto = "PASS (coherencia fuerte)"
    elif corr_global > UMBRAL_CORR:
        veredicto = "COHERENCIA PARCIAL (forma coincide, escala no cumple tolerancia)"
    else:
        veredicto = "NEGATIVO (no se comporta como energía libre bajo estas definiciones)"

    # --- Segundo método: T_implied vs T_efectiva (espacio-T, §6.2) ---
    Ti_cell, Te_cell = [], []
    for f in filas:
        if f["eps"] <= 0.0:
            continue
        if not np.isfinite(f["S_ent"]["media"]) or f["S_ent"]["media"] < S_ENT_MIN_VALIDO:
            continue
        if np.isfinite(f["T_implied"]["media"]):
            Ti_cell.append(f["T_implied"]["media"])
            Te_cell.append(f["T_efectiva"]["media"])
    Ti_cell = np.array(Ti_cell)
    Te_cell = np.array(Te_cell)
    if Ti_cell.size >= 2 and np.std(Ti_cell) > 0 and np.std(Te_cell) > 0:
        corr_T = float(np.corrcoef(Ti_cell, Te_cell)[0, 1])
    else:
        corr_T = float("nan")

    result = {
        "experimento": "E5.6-2",
        "titulo": "Exergia como energia libre: X vs E - T*S_ent",
        "N": N,
        "semillas": SEMILLAS,
        "noise_rel": NOISE_REL,
        "eps_list": EPS_LIST,
        "r_targets": R_TARGETS,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "meta_por_eps": meta_por_eps,
        "filas": filas,
        "veredicto": {
            "umbral_corr": UMBRAL_CORR,
            "umbral_err_rel": UMBRAL_ERR_REL,
            "corr_X_vs_Fpred_global": corr_global,
            "err_rel_mediana": err_rel_mediana,
            "err_rel_media": err_rel_media,
            "n_celdas_incluidas": int(X_cell.size),
            "n_celdas_totales": len(filas),
            "corr_Timplied_vs_Tefectiva": corr_T,
            "veredicto": veredicto,
        },
        "elapsed_s": time.time() - t0,
    }

    out_json = HERE / "E5_6_2_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[veredicto] {veredicto}", file=sys.stderr)
    print(
        f"[corr_global={corr_global:.4f} err_rel_mediana={err_rel_mediana:.4f} "
        f"corr_T={corr_T:.4f} celdas={X_cell.size}/{len(filas)}]",
        file=sys.stderr,
    )
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
