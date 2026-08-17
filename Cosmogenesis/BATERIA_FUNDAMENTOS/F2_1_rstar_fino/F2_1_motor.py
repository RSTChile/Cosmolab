#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2-1 — Umbral de congelamiento: barrido fino de r cruzando 1, con r* resuelto
==============================================================================

Experimento paralelo (prefijo F2_1_) de la BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md.
Sucesor directo de cs074_rcruz.py (producción/robustez400 ya corridos y adjudicados
en ADJUDICACION_CF-1_sello_CS.md): ese grid tenía un solo punto entre r=0 y r=0.1.
Este script resuelve el hueco con un grid MUY fino (0.01 .. 3) y añade la
perturbación DINÁMICA (ruido por paso) que el protocolo pre-registrado exige.

NO MODIFICA cs074_rcruz.py — solo importa sus funciones núcleo (observable, difusión,
expansión, medición de D y de pasos_lavado). Ver PROTOCOLO_F2-1_PREREGISTRO.md
(congelado ANTES de correr este motor) para el diseño completo, criterios de PASS y
las desviaciones de cómputo pre-declaradas.

Uso:
    python3 F2_1_motor.py <N> [--sub-grid]

    <N>          200 | 400 | 800 | 1600
    --sub-grid   usa el sub-grid de 15 puntos (reservado para N=1600 por costo, ver
                 protocolo seccion 3). Si se omite, usa el grid fino completo (33 pts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COSMO_ROOT = HERE.parent.parent  # Cosmogenesis/
sys.path.insert(0, str(COSMO_ROOT))

from cs074_rcruz import (  # noqa: E402
    campo_inicial,
    paso_difusion,
    paso_expansion,
    persistencia,
    medir_D,
    medir_pasos_lavado,
    detectar_cuantizacion,
    temperatura_fisica,
    T_SING,
    P_LAVADO,
    MARGEN_LAVADO,
)

EPS_PRINCIPAL = 1e-3
EPS_SECUNDARIO = 1e-2  # solo chequeo barato N=200, sec.4 del protocolo
N_SEMILLAS = 16
SIGMA_RUIDO_DINAMICO = 0.01  # 1% de la std instantanea del campo, cada paso


def grid_fino():
    """r in [0.01, 3], log-espaciado + densificacion en [0.6, 1.5]. 33 puntos."""
    base = np.geomspace(0.01, 3.0, 25)
    denso = np.linspace(0.6, 1.5, 10)
    r = np.unique(np.round(np.concatenate([base, denso]), 6))
    return [0.0] + list(r)  # r=0 control primero, luego los 33 finos


def sub_grid_1600():
    """15 puntos: 1 de cada 2 del grid fino de 33 + extremos, mismo rango/densidad relativa."""
    fino = grid_fino()
    finos = fino[1:]  # sin el 0.0
    sub = finos[::2]
    if finos[-1] not in sub:
        sub = list(sub) + [finos[-1]]
    sub = sorted(set(sub))
    return [0.0] + sub


def paso_ruido_dinamico(phi, sigma_rel, rng):
    """Perturbacion DINAMICA (T7): ruido gaussiano aditivo EN CADA PASO, no solo en
    la condicion inicial. Amplitud relativa a la std instantanea del campo."""
    if sigma_rel <= 0:
        return phi
    s = float(phi.std())
    if s <= 0:
        return phi
    return phi + rng.normal(0.0, sigma_rel * s, phi.shape)


def evolucionar_f21(phi, activo, H, pasos, rng, sigma_rel, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        phi = paso_ruido_dinamico(phi, sigma_rel, rng)
        activo = paso_expansion(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def corrida_f21(N, eps, H, pasos, seed, sigma_rel, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar_f21(phi, activo, H, pasos, rng, sigma_rel, null=null)
    P = persistencia(phi, c0)
    frac_exp = 1.0 - float(activo.mean())
    T_fin = temperatura_fisica(frac_exp)
    return {"P": P, "frac_exp": frac_exp, "T_fin_K": T_fin,
            "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0}


def barrido(N, eps, r_grid, semillas, sigma_rel, pasos_fijo, D):
    filas = []
    for r_tgt in r_grid:
        H = float(min(r_tgt * D, 1.0)) if D > 0 else (1.0 if r_tgt > 0 else 0.0)
        r_eff = (H / D) if D > 0 else (float("inf") if r_tgt > 0 else 0.0)
        Preal, Pnull, srr, srn, fracs = [], [], [], [], []
        for s in range(semillas):
            rr = corrida_f21(N, eps, H, pasos_fijo, seed=2000 + s, sigma_rel=sigma_rel, null=False)
            nn = corrida_f21(N, eps, H, pasos_fijo, seed=2000 + s, sigma_rel=sigma_rel, null=True)
            Preal.append(rr["P"]); Pnull.append(nn["P"])
            srr.append(rr["std_ratio"]); srn.append(nn["std_ratio"])
            fracs.append(rr["frac_exp"])
        Preal = np.array(Preal); Pnull = np.array(Pnull)
        sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
        sd = max(sd, 1.0 / max(len(Preal), 1))
        z = float((Preal.mean() - Pnull.mean()) / sd)
        filas.append({
            "r_target": r_tgt, "H": H, "r_eff": r_eff,
            "P_real_media": float(Preal.mean()), "P_real_std": float(Preal.std()),
            "P_real_semillas": [float(x) for x in Preal],
            "P_null_media": float(Pnull.mean()), "P_null_std": float(Pnull.std()),
            "P_null_semillas": [float(x) for x in Pnull],
            "z": round(z, 4),
            "std_ratio_real_media": float(np.mean(srr)),
            "std_ratio_null_media": float(np.mean(srn)),
            "frac_exp_media": float(np.mean(fracs)),
        })
    return filas


def estimar_rstar(filas):
    """Todas las metricas del protocolo sec.8 -- ninguna se elige a posteriori."""
    rs = np.array([f["r_eff"] if f["r_target"] > 0 else 0.0 for f in filas])
    r_tgt = np.array([f["r_target"] for f in filas])
    P = np.array([f["P_real_media"] for f in filas])
    # excluir el punto de control r=0 para la curva fina (indice 0)
    r_fin = r_tgt[1:]
    P_fin = P[1:]
    logr = np.log10(r_fin)

    def interp_umbral(thr):
        if P_fin[0] >= thr:
            return float(r_fin[0])
        for i in range(1, len(P_fin)):
            if P_fin[i] >= thr:
                x0, x1 = logr[i - 1], logr[i]
                y0, y1 = P_fin[i - 1], P_fin[i]
                if y1 == y0:
                    return float(10 ** x1)
                frac = (thr - y0) / (y1 - y0)
                return float(10 ** (x0 + frac * (x1 - x0)))
        return None  # nunca cruza

    P_r0 = float(P[0])
    P_max = float(P_fin.max())
    half = P_r0 + 0.5 * (P_max - P_r0)
    r_half = interp_umbral(half)
    r_p2 = interp_umbral(0.2)
    r_p5 = interp_umbral(0.5)
    r_p8 = interp_umbral(0.8)

    dP = np.diff(P_fin) / np.diff(logr)
    idx_max_pend = int(np.argmax(dP)) if len(dP) else None
    r_pendiente_max = float(10 ** ((logr[idx_max_pend] + logr[idx_max_pend + 1]) / 2)) if idx_max_pend is not None else None

    return {
        "P_r0_control": P_r0,
        "P_max_grid": P_max,
        "r_half_rise": r_half,
        "r_P_gt_0.2": r_p2,
        "r_P_gt_0.5": r_p5,
        "r_P_gt_0.8": r_p8,
        "r_pendiente_maxima": r_pendiente_max,
        "pendiente_maxima_valor": float(dP[idx_max_pend]) if idx_max_pend is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int, choices=[200, 400, 800, 1600])
    ap.add_argument("--sub-grid", action="store_true")
    ap.add_argument("--eps", type=float, default=EPS_PRINCIPAL)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    N = args.N
    eps = args.eps
    usar_sub = args.sub_grid or (N == 1600 and args.eps == EPS_PRINCIPAL)
    r_grid = sub_grid_1600() if usar_sub else grid_fino()

    t0 = time.time()
    t0_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[F2-1] inicio N={N} eps={eps} sub_grid={usar_sub} n_r={len(r_grid)} "
          f"t0={t0_iso}", file=sys.stderr, flush=True)

    D = float(np.mean([medir_D(N, eps, s) for s in range(N_SEMILLAS)]))
    cal = medir_pasos_lavado(N, eps, N_SEMILLAS, P_thr=P_LAVADO)
    pasos_fijo = cal["pasos"]
    print(f"[F2-1] D={D:.6e} pasos_lavado_mediana={cal['mediana']} pasos_fijo={pasos_fijo} "
          f"lavo_todas={cal['lavo_todas']}", file=sys.stderr, flush=True)

    filas = barrido(N, eps, r_grid, N_SEMILLAS, SIGMA_RUIDO_DINAMICO, pasos_fijo, D)
    rstar = estimar_rstar(filas)

    control_rows = [f for f in filas if f["r_target"] == 0.0]
    control_ok = bool(control_rows and control_rows[0]["P_real_media"] < 0.15)

    t1 = time.time()
    t1_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    result = {
        "experimento": "F2-1_rstar_fino",
        "N": N,
        "eps": eps,
        "sub_grid": usar_sub,
        "n_r_puntos": len(r_grid),
        "r_grid": r_grid,
        "semillas": N_SEMILLAS,
        "sigma_ruido_dinamico": SIGMA_RUIDO_DINAMICO,
        "D_medido": D,
        "calibracion_lavado": cal,
        "pasos_fijo": pasos_fijo,
        "control_r0_lava": control_ok,
        "control_r0_P": control_rows[0]["P_real_media"] if control_rows else None,
        "filas": filas,
        "rstar_metricas": rstar,
        "t0_utc": t0_iso,
        "t1_utc": t1_iso,
        "elapsed_s": t1 - t0,
    }

    tag = args.tag or (f"eps{eps:g}".replace(".", "p"))
    out_json = HERE / f"F2_1_N{N}_{tag}_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[F2-1] fin N={N} elapsed={result['elapsed_s']:.1f}s control_r0_lava={control_ok} "
          f"archivo={out_json}", file=sys.stderr, flush=True)
    print(f"[F2-1] rstar_metricas={json.dumps(rstar)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
