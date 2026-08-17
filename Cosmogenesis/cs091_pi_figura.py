#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
CS091-D — LA FIGURA. Tres paneles, leídos de los CSV ya escritos (no recalcula nada).

Panel A — DONDE HAY GEOMETRÍA: las tres retículas 2D dan π(r) PLANA (2,0 / 3,0 / 1,5) y la retícula
          CÚBICA 3D — geometría perfecta — da π(r) que CRECE LINEAL. O sea: "π constante" no detecta
          geometría, detecta geometría **de dimensión 2**.
Panel B — EL CONTROL QUE DECIDE: grafo real del corpus vs su barajado (grados preservados) vs ER.
Panel C — EL SUSTRATO EMERGENTE (CS066 `local`, geometrogénesis) vs su barajado y vs `sin_local`.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")


def leer(nombre):
    d = defaultdict(lambda: ([], []))
    with open(AQUI / nombre) as f:
        for fila in csv.DictReader(f):
            r = int(fila["r"]); pi = float(fila["pi_r"])
            # la clave lleva el N: dos corridas distintas pueden compartir etiqueta de regla
            # (p.ej. `A2-B0-C2-batch3-r0` existe a N=2000 y a N=4000) y se pisarian entre si.
            k = f'{fila["sustrato"]}@N{fila["N"]}'
            d[k][0].append(r); d[k][1].append(pi)
    return d


C = leer("pi_contingente_rerun_curvas.csv")
Cem = leer("pi_contingente_rerun_emergente.csv")
TODO = {**C, **Cem}

def BUSCA(prefijo):
    """Devuelve la clave completa (`sustrato@N...`) que corresponde a ese sustrato."""
    for k in TODO:
        if k.split("@N")[0] == prefijo:
            return k
    return prefijo

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6))

# ---------------- Panel A ----------------
serie_A = [
    (BUSCA("reticula_cuadrada"),         "#1f77b4", "-",  "retícula cuadrada 2D — π = 2,00 PLANA"),
    (BUSCA("reticula_triangular"),       "#2ca02c", "-",  "retícula triangular 2D — π = 3,00 PLANA"),
    (BUSCA("reticula_hexagonal"),        "#9467bd", "-",  "retícula hexagonal 2D — π = 1,50 PLANA"),
    (BUSCA("reticula_cubica_3D_a_mano"), "#e377c2", "-.", "retícula CÚBICA 3D — π CRECE (≈2r)"),
]
for k, col, ls, lab in serie_A:
    r, y = TODO[k]
    ax[0].plot(r, y, ls, color=col, lw=2.2, marker="o", ms=3.5, label=lab)
ax[0].axhline(np.pi, color="grey", lw=1, ls=":")
ax[0].text(7.2, np.pi * 1.05, "π = 3,1416", fontsize=8, color="grey")
ax[0].set_title("A · Donde SÍ hay geometría\n(3 filas del nodo reproducidas + el calibrador 3D)", fontsize=10)
ax[0].legend(fontsize=8, loc="upper left")

# ---------------- Panel B ----------------
real = [k for k in C if k.startswith("REAL:") and k.endswith("@N2000")][0]
bar = [k for k in C if k.startswith("BARAJADO:") and k.endswith("@N2000")][0]
er = [k for k in C if k.startswith("ER:") and k.endswith("@N2000")][0]
for k, col, ls, lab in ((real, "#000000", "-", "REAL — corpus A2-B0-C2 (N=2000)"),
                        (bar, "#ff7f0e", "--", "NULL barajado (grados preservados)"),
                        (er, "#17becf", ":", "NULL Erdős–Rényi (mismo N, mismo E)"),
                        (BUSCA("mundo_pequeno_k6_p0.1"), "#d62728", "-", "mundo-pequeño sintético (WS)")):
    r, y = TODO[k]
    ax[1].plot(r, y, ls, color=col, lw=2.2, marker="o", ms=3.5, label=lab)
ax[1].set_yscale("log")
ax[1].set_title("B · EL CONTROL QUE DECIDE\nel barajado estalla IGUAL que el real", fontsize=10)
ax[1].legend(fontsize=8, loc="lower right")

# ---------------- Panel C ----------------
for k, col, ls, lab in ((BUSCA("CS066_local_k5_EMERGENTE"),    "#000000", "-",  "CS066 `local` k=5 — EMERGENTE"),
                        (BUSCA("CS066_local_k5_BARAJADO"),     "#ff7f0e", "--", "su barajado (grados preservados)"),
                        (BUSCA("CS066_sin_local_k5_EMERGENTE"), "#8c564b", ":",  "CS066 `sin_local` (blob)"),
                        (BUSCA("reticula_cuadrada"),           "#1f77b4", "-",  "retícula cuadrada (referencia plana)")):
    if k in TODO:
        r, y = TODO[k]
        ax[2].plot(r, y, ls, color=col, lw=2.2, marker="o", ms=3.5, label=lab)
ax[2].set_yscale("symlog", linthresh=1)
ax[2].set_title("C · El sustrato EMERGENTE del corpus\n(geometrogénesis CS066): tampoco es plana", fontsize=10)
ax[2].legend(fontsize=8, loc="upper right")

for a in ax:
    a.set_xlabel("radio r (saltos geodésicos)")
    a.set_ylabel("π_emergente(r) = |S(r)| / 2r")
    a.grid(alpha=0.3)

fig.suptitle("π CONTINGENTE — re-corrida con controles (13-ago-2026). "
             "π(r)=|S(r)|/2r es una re-escritura del perfil de crecimiento de bolas.", fontsize=12)
fig.tight_layout()
fig.savefig(AQUI / "pi_contingente_rerun_curvas.png", dpi=150)
print("Figura escrita: pi_contingente_rerun_curvas.png")
