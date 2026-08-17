#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase6_o3f_figura.py — figura resumen de O3-F (B_τ, branching de futuros)

Tres paneles, uno por pregunta:
  (a) La diferencia pareada de B_τ (III − I) bajo los seis umbrales, y al lado la misma
      diferencia con el numerador APAGADO (H := constante). Si las dos columnas son
      iguales, B_τ no estaba midiendo entropía.
  (b) La diferencia pareada de la ENTROPÍA sola (rarefacción, N igualado): el numerador
      de B_τ, sin dividir por nada.
  (c) B_τ contra la fracción de masa ya acretada, un punto por brazo: muestra si el
      observable es una reescritura de "cuánto colapsó".
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent
pares = [p for p in csv.DictReader(open(RAIZ / "cs090_fase6_o3f_btau_pares.csv", newline=""))
         if p["estado"] == "valido"]
umbrales = list(dict.fromkeys(p["umbral"] for p in pares))


def arr(sub, col):
    return np.array([float(s[col]) for s in sub], dtype=float)


fig, ejes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# --- (a) B_τ real vs postizo -------------------------------------------------
ax = ejes[0]
datos_real, datos_postizo = [], []
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u]
    hI, hIII = arr(sub, "Hrar_v3d_std_I"), arr(sub, "Hrar_v3d_std_III")
    mI, mIII = arr(sub, "masa_difusa_I"), arr(sub, "masa_difusa_III")
    hc = np.mean(np.concatenate([hI, hIII]))
    esc = 1e5
    datos_real.append((hIII / mIII - hI / mI) * esc)
    datos_postizo.append((hc / mIII - hc / mI) * esc)
pos = np.arange(len(umbrales))
b1 = ax.boxplot(datos_real, positions=pos - 0.18, widths=0.3, patch_artist=True,
                showfliers=False)
b2 = ax.boxplot(datos_postizo, positions=pos + 0.18, widths=0.3, patch_artist=True,
                showfliers=False)
for caja in b1["boxes"]:
    caja.set_facecolor("#3d6fb4"); caja.set_alpha(.85)
for caja in b2["boxes"]:
    caja.set_facecolor("#c96f2e"); caja.set_alpha(.85)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xticks(pos); ax.set_xticklabels(umbrales, rotation=30, ha="right", fontsize=8)
ax.set_ylabel(r"$\Delta B_\tau$ (III − I)  $\times 10^{5}$")
ax.set_title("(a) $B_\\tau$ real (azul) vs con el numerador\ncongelado, H:=cte (naranja)")
ax.legend([b1["boxes"][0], b2["boxes"][0]], ["B_τ real", "B_τ postizo (H constante)"],
          fontsize=8, loc="upper right")

# --- (b) entropía sola -------------------------------------------------------
ax = ejes[1]
datos = []
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u]
    datos.append(arr(sub, "Hrar_v3d_std_III") - arr(sub, "Hrar_v3d_std_I"))
b = ax.boxplot(datos, positions=pos, widths=0.5, patch_artist=True, showfliers=False)
for caja in b["boxes"]:
    caja.set_facecolor("#4b9560"); caja.set_alpha(.85)
for i, d in enumerate(datos):
    ax.scatter(np.full(len(d), i) + np.random.uniform(-.13, .13, len(d)), d,
               s=7, color="k", alpha=.35, zorder=3)
ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xticks(pos); ax.set_xticklabels(umbrales, rotation=30, ha="right", fontsize=8)
ax.set_ylabel(r"$\Delta H$ (III − I), bits")
ax.set_title("(b) entropía de velocidades SOLA\n(rarefacción a N igual, sin dividir)")

# --- (c) B_τ vs masa acretada ------------------------------------------------
ax = ejes[2]
sub = [p for p in pares if p["umbral"] == "A_P90_abs"]
for rol, color, marca in (("I", "#3d6fb4", "o"), ("III", "#c0392b", "^")):
    x = arr(sub, f"frac_masa_{rol}")
    y = arr(sub, f"Btau_Hrar_v3d_std_{rol}") * 1e4
    ax.scatter(x, y, s=34, c=color, marker=marca, alpha=.8, label=f"Clase {rol}")
ax.set_xlabel("fracción de masa ya acretada en sumideros")
ax.set_ylabel(r"$B_\tau \times 10^{4}$  (umbral A_P90_abs)")
ax.set_title("(c) $B_\\tau$ es casi una función de\ncuánto colapsó  (Spearman ≈ 0.84)")
ax.legend(fontsize=9)

fig.suptitle("O3-F — $B_\\tau$ = H(v del gas difuso) / |Ω_gas| sobre los 37 pares válidos de Fase V-B",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(RAIZ / "cs090_fase6_o3f_btau.png", dpi=140)
print("escrito cs090_fase6_o3f_btau.png")
