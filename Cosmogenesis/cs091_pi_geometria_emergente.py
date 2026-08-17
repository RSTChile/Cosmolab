#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
CS091-C — LA CELDA QUE FALTABA: ¿hay algún sustrato EMERGENTE con geometría contra el cual medir π?
====================================================================================================

QUÉ ES ESTO (a nivel módulo)
-----------------------------
El nodo del 16-jul midió π sobre TRES retículas construidas a mano y UN mundo-pequeño. La celda que
falta es la interesante: un sustrato donde la geometría EMERGIÓ del proceso, no donde se la puso a
mano. Mostrar que una retícula cuadrada tiene π=2,0 es casi definicional; lo que valdría es que un
tejido nacido de reglas locales tuviera también π constante.

En el corpus, el candidato más fuerte es el brazo `local` de CS066 (geometrogénesis por costo de
no-localidad, estilo Quantum Graphity), del que `cs066conf_exponentes.md` dice: "locALMENTE 3D pero
GLOBALMENTE compacto — NO es 3-manifold métrico limpio". Este script lo REGENERA (no hay grafos de
CS066 guardados en disco) y le mide π(r), junto a su propio barajado y a sus dos calibradores.

ADEMÁS — Y ES IMPORTANTE — EL CALIBRADOR 3D
--------------------------------------------
Se mide π(r) sobre una retícula CÚBICA construida a mano. Es la prueba de que el criterio del nodo
("π constante = hay geometría") NO mide "hay geometría": mide "hay geometría **de dimensión 2**".
En 3D la frontera de la bola crece como r², así que π(r)=|S(r)|/2r crece LINEAL con r — sin ningún
atajo, sin ningún mundo-pequeño, en un espacio perfectamente geométrico. Analogía: la cáscara de una
naranja crece con el cuadrado del radio; dividirla por el diámetro sigue dando un número que crece.

SALIDA
------
- `pi_contingente_rerun_emergente.csv` : π(r) de los sustratos emergentes y del calibrador 3D
  (se anexa además al CSV principal de curvas)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
sys.path.insert(0, str(AQUI))
os.environ.setdefault("CS066_N", "2500")
os.environ.setdefault("CS066_STEPS", "20")

from cs091_pi_contingente_rerun import (curva_pi, fuentes_gigante, baraja_rapida,
                                        componente_gigante, R_MAX, SEMILLA)
from cs091_pi_analisis import constancia


def reticula_cubica(L):
    """Retícula cúbica L×L×L, 6 vecinos. Frontera teórica: |S(r)| = 4r²+2 → π(r) ≈ 2r (¡NO constante!)."""
    idx = lambda x, y, z: (x * L + y) * L + z
    adj = [set() for _ in range(L ** 3)]
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx(x, y, z)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx < L and ny < L and nz < L:
                        j = idx(nx, ny, nz)
                        adj[i].add(j); adj[j].add(i)
    return adj, idx(L // 2, L // 2, L // 2)


def main():
    filas = []

    def registrar(nombre, adj, fuentes):
        N = len(adj); E = sum(len(s) for s in adj) // 2
        f = curva_pi(adj, fuentes, R_MAX)
        cv, b = constancia(f)
        print(f"  {nombre:34s} N={N:6d} E={E:6d} CV={cv:5.3f} b={b:4.2f}  π(r)= "
              + " ".join(f"{x['pi_r']:.2f}" for x in f[:8]))
        for x in f:
            x.update(sustrato=nombre, N=N, E=E, cv_pi_r2a5=round(cv, 4), b_medio=round(b, 4))
            filas.append(x)

    print("[CALIBRADOR 3D — geometría perfecta, π NO constante]")
    adj, c = reticula_cubica(31)
    registrar("reticula_cubica_3D_a_mano", adj, [c])

    print("\n[SUSTRATO EMERGENTE — brazo `local` de CS066 (geometrogénesis), regenerado]")
    try:
        import cs066_localidad_geometrogenesis as C66
        import cs064_sistema_completo as C64
        RNG = C66.RNG
        for k_local in (5, 6):
            for arm in ("local", "sin_local"):
                rng = RNG(66000 + k_local)
                cat = C64._cataloga(C66.N_NODOS, rng)
                a, V, D, G = C66.proceso066(C66.N_NODOS, cat, arm, k_local, RNG(770 + k_local))
                nom = f"CS066_{arm}_k{k_local}_EMERGENTE"
                registrar(nom, a, fuentes_gigante(a, 200))
                if arm == "local":
                    bar, _ = baraja_rapida(a, factor=20, semilla=SEMILLA + k_local)
                    registrar(f"CS066_local_k{k_local}_BARAJADO", bar, fuentes_gigante(bar, 200))
    except Exception as e:
        print(f"  NO se pudo regenerar CS066: {type(e).__name__}: {e}")

    if filas:
        cols = ["sustrato", "N", "E", "r", "S_r", "B_r", "pi_r", "pi_mediana", "pi_q1", "pi_q3",
                "b_r", "n_fuentes", "cv_pi_r2a5", "b_medio"]
        with open(AQUI / "pi_contingente_rerun_emergente.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for x in filas:
                w.writerow({c: x[c] for c in cols})
        print(f"\nEscrito pi_contingente_rerun_emergente.csv ({len(filas)} filas)")


if __name__ == "__main__":
    main()
