#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase8_f802_umbral_ic.py — F8-02, CONTROL: ¿la manipulación le está REGALANDO sumideros a Phantom?

POR QUÉ EXISTE
--------------
Si apretar los grumos sube la masa acretada, hay dos explicaciones muy distintas:

  (a) el gas apretado **colapsa mejor** — el pico inicial cambia la dinámica posterior;
  (b) el gas apretado **ya nace por encima del umbral de sumidero**, y entonces no hicimos un
      experimento sobre la gravedad: le pusimos los sumideros ya prendidos en la mano.

`FASE8_F805_f703_solver_independiente_CS.md` midió el umbral real: `rho_crit_cgs = 1000` en unidades de
código, contra una densidad media de 18800/97.5929³ = 0.02023 → **49.442 veces la densidad media**
(en F8-05 aparece como "49.453×", con el punto como separador de miles). Y comprobó que en las
condiciones iniciales de F7-03 **no había ni una sola partícula por encima de ese umbral**.

Este script hace exactamente esa cuenta sobre las 60 condiciones iniciales de F8-02, con **el mismo
estimador de densidad que usó F8-05** (`cs090_fase8_f805_umbrales.py`, línea 63):

        rho_i = k · m_particula / ((4/3)·π·r_k³)          con k = 8

SALIDA
------
  cs090_fase8_f802_umbral_ic.csv — una fila por condición inicial: densidad máxima en unidades de la
  media, nº de partículas por encima del umbral de Phantom y qué fracción de masa representan.

No corre Phantom. No modifica nada. No declara cierre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AQUI = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
RUTA_IC = f"{AQUI}/cs090_fase8_f802_ic_transformadas.csv"
SALIDA = f"{AQUI}/cs090_fase8_f802_umbral_ic.csv"

K_VECINO = 8
RHO_CRIT = 1000.0                       # el `rho_crit_cgs` del cosmog.in de toda la línea CS073
LADO = 97.5929                          # lado de la nube (idéntico en las 60 IC, ver CSV)
MASA_TOTAL = 18800.0
RHO_MEDIA = MASA_TOTAL / LADO ** 3


def main():
    D = pd.read_csv(RUTA_IC)
    print(f"densidad media = {RHO_MEDIA:.5f}; umbral de Phantom = {RHO_CRIT} "
          f"= {RHO_CRIT/RHO_MEDIA:,.0f}× la media")
    filas = []
    for _, r in D.iterrows():
        pos = np.loadtxt(f"{r.carpeta}/cosmogenesis_ic.txt", skiprows=2)[:, :3]
        dist, _ = cKDTree(pos).query(pos, k=K_VECINO + 1)
        rho = K_VECINO * r.masa_particula / ((4.0 / 3.0) * np.pi * dist[:, K_VECINO] ** 3)
        sobre = rho > RHO_CRIT
        filas.append(dict(rule_id=r.rule_id, seed=r.seed, nivel=r.nivel, a_pico=r.a_pico,
                          pico_logrado=r.pico_logrado,
                          rho_max_en_medias=float(rho.max() / RHO_MEDIA),
                          n_particulas_sobre_umbral=int(sobre.sum()),
                          frac_masa_sobre_umbral=float(sobre.mean())))
    F = pd.DataFrame(filas)
    F.to_csv(SALIDA, index=False)
    print(F.groupby("nivel").agg(
        rho_max_mediana=("rho_max_en_medias", "median"), rho_max_max=("rho_max_en_medias", "max"),
        n_sobre_mediana=("n_particulas_sobre_umbral", "median"),
        n_sobre_max=("n_particulas_sobre_umbral", "max"),
        ic_con_alguna=("n_particulas_sobre_umbral", lambda s: int((s > 0).sum()))).round(1).to_string())
    print(f"\nESCRITO: {SALIDA}")
    return F


if __name__ == "__main__":
    main()
