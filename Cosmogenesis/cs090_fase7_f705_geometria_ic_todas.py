#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F7-05 · PASO 2 — Geometría INICIAL medida sobre TODAS las condiciones iniciales
===============================================================================

POR QUÉ EXISTE
--------------
La hipótesis a poner a prueba es una cadena: `densidad → geometría → masa`. Para
estimarla hace falta el eslabón del medio (la geometría) medido en cada corrida,
no sólo en las 26 de O3-A. Resulta que **ya está en el disco**: cada carpeta de
Phantom guarda su `cosmogenesis_ic.txt`, la nube de puntos tal como nació, antes
de integrar un solo paso. Medirla es re-análisis puro: no se corre Phantom, no se
genera ningún grafo, no se regenera ninguna regla.

Analogía: es pesar el bollo crudo. El bollo crudo está guardado en la heladera de
cada experimento; nadie lo había pesado más que en un par de casos.

QUÉ MIDE (la misma vara que O4-A y O3-A, para hablar el mismo idioma)
----------------------------------------------------------------------
- `fof_masa` importado TAL CUAL de `cs090_fase6_o4a_observable_comun.py`:
  friends-of-friends con longitud de enlace en unidades de la separación media
  (b = 0.20 / 0.30 / 0.50) y umbral de grupo en MASA física fija.
- Además, dos descriptores locales de apelotonamiento que no dependen de umbral:
  · `knn8_cv`  — coeficiente de variación de la densidad estimada con 8 vecinos
                 (cuán desparejo está repartido el gas).
  · `knn8_p90` — razón entre el percentil 90 y la mediana de esa densidad
                 (cuán alto llega el pico respecto de lo típico).

SALIDA
------
  cs090_fase7_f705_geometria_ic_todas.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from cs090_fase6_o4a_observable_comun import fof_masa  # misma vara, sin tocar

RAIZ = "/Users/alexis/phantom_cs073"
SALIDA = os.path.join(AQUI, "cs090_fase7_f705_geometria_ic_todas.csv")
GRILLA_B = (0.20, 0.30, 0.50)
MASA_MIN = 47.0          # 5 partículas de N=2000 — criterio en masa física (O3-A)


def leer_ic(p):
    """Devuelve (posiciones, masa_por_particula, cabecera). Formato cosmogenesis_ic v2."""
    with open(p) as fh:
        cab = fh.readline()
        linea2 = fh.readline().split()
    npart = int(float(linea2[0]))
    m_part = float(linea2[1])
    d = np.loadtxt(p, skiprows=2)
    return d[:, :3], m_part, cab, npart


def descriptores_knn(pos, k=8):
    arbol = cKDTree(pos)
    dist, _ = arbol.query(pos, k=k + 1)
    r_k = dist[:, k]
    rho = k / (r_k ** 3 + 1e-300)          # densidad local ~ k / r_k^3
    med = np.median(rho)
    return float(rho.std() / rho.mean()), float(np.percentile(rho, 90) / med)


def main():
    # las baterías tienen 2 o 3 niveles (p.ej. o3a_resolucion/N4000/<regla>/)
    archivos = sorted(set(glob.glob(os.path.join(RAIZ, "*", "*", "cosmogenesis_ic.txt")) +
                          glob.glob(os.path.join(RAIZ, "*", "*", "*", "cosmogenesis_ic.txt"))))
    print(f"IC encontradas: {len(archivos)}")
    filas = []
    for i, p in enumerate(archivos, 1):
        carpeta = os.path.dirname(p)
        try:
            pos, m_part, cab, npart = leer_ic(p)
        except Exception as e:
            print(f"  [ERROR] {carpeta}: {e}")
            continue
        masas = np.full(len(pos), m_part)
        # lado real de la nube (las IC no son periódicas; se usa la extensión)
        lado = float(np.max(pos.max(0) - pos.min(0)))
        sep = lado / len(pos) ** (1.0 / 3.0)
        fila = {
            "carpeta": carpeta,
            "bateria": os.path.basename(os.path.dirname(carpeta)),
            "nombre_carpeta": os.path.basename(carpeta),
            "npart_ic": len(pos),
            "masa_particula": m_part,
            "lado_nube": lado,
            "sep_media": sep,
        }
        for b in GRILLA_B:
            f, ng = fof_masa(pos, masas, b * sep, MASA_MIN)
            fila[f"geoIC_fof_b{b:.2f}"] = f
            fila[f"geoIC_ngrupos_b{b:.2f}"] = ng
        cv, p90 = descriptores_knn(pos)
        fila["geoIC_knn8_cv"] = cv
        fila["geoIC_knn8_p90_med"] = p90
        # nº de aristas anotado en la cabecera del propio IC (verificación cruzada)
        if "n_aristas=" in cab:
            try:
                fila["n_aristas_cabecera"] = int(cab.split("n_aristas=")[1].split()[0])
            except Exception:
                pass
        filas.append(fila)
        if i % 40 == 0:
            print(f"  {i}/{len(archivos)}")
    D = pd.DataFrame(filas)
    D.to_csv(SALIDA, index=False)
    print(f"\nESCRITO: {SALIDA}  ({len(D)} filas)")
    print(D.groupby("bateria").size().to_string())


if __name__ == "__main__":
    main()
