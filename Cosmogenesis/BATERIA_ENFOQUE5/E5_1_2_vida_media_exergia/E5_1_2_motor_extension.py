#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_1_2_motor_extension.py — cierre de TIER2 (recortado), TIER3 (recortado) y TIER4 de
E5.1-2, tras descubrir en producción que el costo por punto escala ~N^2.6-3 (τ~1/D~N²,
costo~N·τ). TIER1 (curva primaria, 16 puntos × 16 semillas, SIN censura) ya terminó
completo y no se toca aquí — ver E5_1_2_resultado_tier1_PARCIAL.json.

DESVIACIÓN DECLARADA (post-preregistro, por costo de cómputo, no por resultado):
  - El motor original (E5_1_2_motor_vida_media.py) fue matado (SIGTERM, limpio) durante
    TIER2 en el ancla N=1493 (eps=1e-9 ya había corrido, 214s por sí solo). Extrapolando
    el costo medido en TIER1 (N=3000 con 16 semillas tardó 2664.7s), el ancla N=3000 de
    TIER2 (8 eps no triviales × 8 semillas) habría costado ~150-190 min ELLA SOLA, y
    TIER3 con N=6000 (6 semillas) otro ~90 min. Total proyectado para completar el diseño
    original: varias horas más. Se trunca aquí por presupuesto de tiempo de sesión, NO
    porque los resultados parciales sugirieran nada en particular (la decisión de recortar
    se tomó ANTES de mirar si la tendencia era "buena" o "mala" — el criterio fue 100%
    costo computacional, verificable con los timestamps del log).
  - TIER2 final: anclas N={16,130,524} completas (8 eps c/u, 8 semillas) — ya estaban
    100% corridas antes de matar el proceso, se reusan tal cual (E5_1_2_resultado_tier2_PARCIAL.json).
    N=1493 queda con UN solo punto (eps=1e-9) como dato suelto, marcado explícitamente.
    N=3000 queda SIN correr en tier2 — reportado como ausente, no simulado ni estimado.
  - TIER3 recortado: en vez de N∈{4500,6000} (6 semillas), se corre solo N=4000 (4
    semillas) — ya medido en la calibración previa a costo conocido (~53s/semilla).
  - TIER4: sin cambios de diseño, pero limitado a anclas N∈{16,130,524} (se excluye
    N=3000 por costo; ver misma razón que arriba).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(HERE))

from cs074_rcruz import medir_D  # noqa: E402
from E5_1_2_motor_vida_media import correr_tau, estimar_presupuesto, medir_D_prom, log  # noqa: E402

OUT_DIR = HERE


def _guardar(nombre, data):
    path = OUT_DIR / f"E5_1_2_resultado_{nombre}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def tier3_recortado():
    log("=== TIER 3 (recortado): extension D — solo N=4000, 4 semillas ===")
    N = 4000
    eps = 1e-3
    semillas = 4
    D = medir_D_prom(N, eps, semillas=4)
    max_steps, check_every = estimar_presupuesto(D)
    t0 = time.time()
    taus, censuras = [], []
    for s in range(semillas):
        r = correr_tau(N, eps, seed=40_000 + s, max_steps=max_steps, check_every=check_every)
        taus.append(r["tau"])
        censuras.append(r["censurado"])
        log(f"    N={N} semilla={s} tau={r['tau']} censurado={r['censurado']}")
    dt = time.time() - t0
    fila = {
        "N": N, "D": D, "eps": eps, "semillas": semillas,
        "max_steps": max_steps, "check_every": check_every,
        "tau_media": float(np.mean(taus)), "tau_std": float(np.std(taus)),
        "taus_todas": [int(x) for x in taus],
        "n_censurados": int(sum(censuras)), "wall_s": dt,
        "nota": "TIER3 recortado por costo; N=4500 y N=6000 originalmente disenados NO se corrieron",
    }
    log(f"  N={N:>6d} D={D:.4e} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} wall={dt:.1f}s")
    resultado = [fila]
    _guardar("tier3_final", resultado)
    return resultado


def tier4_completo():
    log("=== TIER 4: perturbacion dinamica — anclas N={16,130,524}, 2 niveles ruido, 8 semillas ===")
    N_anclas = [16, 130, 524]
    eps = 1e-3
    fracs = [0.01, 0.1]
    semillas = 8
    filas = []
    for N in N_anclas:
        D = medir_D_prom(N, eps, semillas=4)
        max_steps, check_every = estimar_presupuesto(D, margen=4.0)
        for frac in fracs:
            sigma = frac * eps
            t0 = time.time()
            taus, censuras = [], []
            for s in range(semillas):
                r = correr_tau(N, eps, seed=50_000 + s, max_steps=max_steps, check_every=check_every, sigma_ruido=sigma)
                taus.append(r["tau"])
                censuras.append(r["censurado"])
            dt = time.time() - t0
            fila = {
                "N": N, "D": D, "eps": eps, "sigma_ruido": sigma, "frac_ruido": frac,
                "semillas": semillas, "max_steps": max_steps, "check_every": check_every,
                "tau_media": float(np.mean(taus)), "tau_std": float(np.std(taus)),
                "taus_todas": [int(x) for x in taus],
                "n_censurados": int(sum(censuras)), "wall_s": dt,
            }
            filas.append(fila)
            log(f"  N={N:>6d} D={D:.4e} frac_ruido={frac} tau_med={fila['tau_media']:.1f}±{fila['tau_std']:.1f} wall={dt:.1f}s")
            _guardar("tier4_final", filas)
    return filas


def main():
    t0 = time.time()
    log("###### INICIO extension E5.1-2 (tier3 recortado + tier4) ######")
    tier3 = tier3_recortado()
    tier4 = tier4_completo()
    log(f"###### FIN extension. elapsed={time.time()-t0:.1f}s ######")


if __name__ == "__main__":
    main()
