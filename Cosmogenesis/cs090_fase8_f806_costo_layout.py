#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
cs090_fase8_f806_costo_layout.py — F8-06: ¿cuánto cuesta CADA layout a N=4000, y por lo tanto cuál usar?
=========================================================================================================

POR QUÉ EXISTE (a nivel módulo)
--------------------------------
La regla dura de la Fase VIII dice: **ambos brazos de un par deben usar el mismo layout y el mismo θ**,
porque el sesgo que introduce Barnes-Hut con θ=0.3 (+0.0025 a +0.0071 de fracción de masa) es MAYOR que
los residuales que esta línea persigue. La consigna de F8-06 agrega: **medí el costo de los dos y elegí,
documentando la decisión.**

Este script mide **una evaluación de la fuerza de repulsión** a N=4000 con cada método, sobre la misma
nube de posiciones, y multiplica por las 100 iteraciones que usa el protocolo. Mide la parte cara y
única que distingue a los dos layouts: el resto del bucle (atracción sobre aristas, enfriamiento,
frontera reflectante) es **idéntico** en ambos — `cs090_layout_barnes_hut.layout_barnes_hut` importa
esas piezas del módulo congelado.

QUÉ SE DECIDIÓ CON ESTO, Y POR QUÉ NO GANÓ EL MÁS RÁPIDO
---------------------------------------------------------
Barnes-Hut con θ=0.3 salió ~2× más rápido que la suma exacta N². Aun así **se eligió la suma N²**,
por una razón que no es de costo sino de comparabilidad: los puntos de N=2000 de F7-03 y los de N=4000
de O3-A están medidos con `layout_resortes` (N²), y la pregunta de F8-06 es justamente comparar N=4000
contra N=2000. Un punto viejo (N²) no se puede comparar contra uno nuevo (Barnes-Hut). Pagar 2× de
tiempo compra la única comparación que la tarea pide.

USO:  python3.9 cs090_fase8_f806_costo_layout.py [N]        (default N=4000)
Sale `cs090_fase8_f806_costo_layout.csv`.
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

from cs090_layout_barnes_hut import repulsion_exacta, repulsion_barnes_hut   # sólo import
from cs090_fase5b_phantom_adaptador import LADO_FIJO                          # sólo import

ITERS_PROTOCOLO = 100
RUTA_CSV = f"{HERE}/cs090_fase8_f806_costo_layout.csv"


def medir(N=4000, semilla=1):
    lado = float(LADO_FIJO)
    rng = np.random.default_rng(semilla)
    pos = rng.uniform(0.0, lado, size=(N, 3))
    k_fr = (lado ** 3 / N) ** (1.0 / 3.0)

    filas = []
    t0 = time.time(); f_ex = repulsion_exacta(pos, k_fr); t_ex = time.time() - t0
    norma = float(np.mean(np.linalg.norm(f_ex, axis=1)))
    filas.append(dict(N=N, metodo="layout_resortes_N2_exacto", theta="", t_eval_s=round(t_ex, 3),
                      t_100_iters_s=round(t_ex * ITERS_PROTOCOLO, 1), err_rel_medio=0.0,
                      nota="la suma partícula-a-partícula del protocolo original; es la referencia"))
    for th in (0.3, 0.5):
        t0 = time.time(); f_bh = repulsion_barnes_hut(pos, k_fr, lado, th); t_bh = time.time() - t0
        err = float(np.mean(np.linalg.norm(f_bh - f_ex, axis=1)) / norma)
        filas.append(dict(N=N, metodo="barnes_hut", theta=th, t_eval_s=round(t_bh, 3),
                          t_100_iters_s=round(t_bh * ITERS_PROTOCOLO, 1),
                          err_rel_medio=round(err, 6),
                          nota=("theta=0.3 es el unico validado (INFRA); theta>=0.5 no paso la validacion"
                                if th == 0.3 else "theta>=0.5 NO validado -- se mide solo como referencia")))
    with open(RUTA_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f"  N={f['N']} {f['metodo']:<26} theta={str(f['theta']):<4} "
              f"1 eval={f['t_eval_s']:>7.3f}s  x100={f['t_100_iters_s']:>7.1f}s  "
              f"err_rel={f['err_rel_medio']:.4g}")
    print(f"[csv] {RUTA_CSV.split('/')[-1]}")
    print("DECISION: se usa layout_resortes (N2) en LOS DOS BRAZOS — no por costo, por comparabilidad "
          "con los puntos de N=2000 (F7-03) y de N=4000 (O3-A), que son N2.")
    return filas


if __name__ == "__main__":
    medir(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
