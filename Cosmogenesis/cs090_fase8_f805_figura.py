#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase8_f805_figura.py — el dibujo de F8-05 (4 paneles)

Panel A: los tres números — `solap` menos `disj` en Phantom, en el motor independiente y
         en las condiciones iniciales sin integrar, grafo por grafo.
Panel B: el tamaño del efecto de un motor contra el del otro, par por par.
Panel C: la vara importa — el mismo estado final de Phantom medido de dos maneras.
Panel D: el efecto crece con el número de triángulos disponibles, en los dos motores.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
p = pd.read_csv(os.path.join(AQUI, "cs090_fase8_f805_pares.csv")).sort_values("n_triangulos")
vc = pd.read_csv(os.path.join(AQUI, "cs090_fase8_f805_vara_comun.csv"))

GRANO = 0.0005   # 1 partícula de fracción de masa a N=2000
fig, ax = plt.subplots(2, 3, figsize=(21, 10))

# --- A: los tres números, grafo por grafo -----------------------------------
a = ax[0, 0]
x = np.arange(len(p))
a.bar(x - 0.27, p["frac_masa__d"] / GRANO, 0.27, label="Phantom (masa en sumideros)", color="#1f77b4")
a.bar(x, p["fin_knn8_frac_rho_mayor_1000x__d"] / GRANO, 0.27,
      label="motor independiente (densidad >1000×, t=0.5)", color="#d62728")
a.bar(x + 0.27, p["ini_knn8_frac_rho_mayor_100x__d"] / GRANO, 0.27,
      label="IC SIN integrar (densidad >100×, t=0)", color="#7f7f7f")
a.axhline(0, color="k", lw=0.8)
a.axhline(1, color="k", lw=0.6, ls=":")
a.set_xticks(x)
a.set_xticklabels([f"{r.split('-')[-1]}\nT*={t}" for r, t in zip(p.rule_id, p.n_triangulos)], fontsize=7)
a.set_ylabel("`solap` − `disj`  (en partículas; grano = 1)")
a.set_title("A · el mismo contraste medido de tres maneras")
a.legend(fontsize=8)

# --- B: efecto contra efecto -------------------------------------------------
b = ax[0, 1]
b.scatter(p["frac_masa__d"] / GRANO, p["fin_knn8_frac_rho_mayor_100x__d"] / GRANO,
          s=60, c="#d62728", label="densidad >100× (ρ=+0.83)")
b.scatter(p["frac_masa__d"] / GRANO, p["fin_fof_ell1.0_nmin10__d"] / GRANO,
          s=60, c="#2ca02c", marker="s", label="FoF ell=1.0, ≥10 (ρ=+0.69)")
b.scatter(p["frac_masa__d"] / GRANO, p["fin_fof_ell1.0_nmin5__d"] / GRANO,
          s=60, c="#9467bd", marker="^", label="FoF ell=1.0, ≥5 — el pre-declarado")
b.axhline(0, color="k", lw=0.8)
b.axvline(0, color="k", lw=0.8)
b.set_xlabel("Δ de Phantom (partículas)")
b.set_ylabel("Δ del motor independiente (partículas)")
b.set_title("B · tamaño del efecto de un motor contra el del otro")
b.legend(fontsize=8)

# --- C: la vara importa ------------------------------------------------------
c = ax[1, 0]
pares_vc = vc.pivot_table(index="rule_id", columns="brazo",
                          values=["ph_fofcomun_ell1.0", "ph_fofcomun_ell0.3"])
d_fof10 = (pares_vc[("ph_fofcomun_ell1.0", "solap")] - pares_vc[("ph_fofcomun_ell1.0", "disj")]) / GRANO
d_fof03 = (pares_vc[("ph_fofcomun_ell0.3", "solap")] - pares_vc[("ph_fofcomun_ell0.3", "disj")]) / GRANO
orden = p.rule_id.tolist()
d_fof10 = d_fof10.reindex(orden)
d_fof03 = d_fof03.reindex(orden)
c.bar(x - 0.27, p["frac_masa__d"] / GRANO, 0.27, color="#1f77b4",
      label="Phantom, vara = masa en sumideros (12/12)")
c.bar(x, d_fof03.values, 0.27, color="#ff7f0e",
      label="Phantom, vara = FoF ell=0.3 (12/12)")
c.bar(x + 0.27, d_fof10.values, 0.27, color="#8c564b",
      label="Phantom, vara = FoF ell=1.0 (2/12) ← misma corrida, se da vuelta")
c.axhline(0, color="k", lw=0.8)
c.set_xticks(x)
c.set_xticklabels([r.split("-")[-1] for r in p.rule_id], fontsize=7)
c.set_ylabel("`solap` − `disj` (partículas)")
c.set_title("C · el MISMO estado final de Phantom, medido con tres varas")
c.legend(fontsize=8)

# --- D: el efecto crece con T* ----------------------------------------------
dd = ax[1, 1]
dd.scatter(p.n_triangulos, p["frac_masa__d"] / GRANO, s=60, c="#1f77b4", label="Phantom (ρ=+0.82)")
dd.scatter(p.n_triangulos, p["fin_knn8_frac_rho_mayor_100x__d"] / GRANO, s=60, c="#d62728",
           marker="s", label="motor independiente (ρ=+0.78)")
dd.scatter(p.n_triangulos, p["ini_knn8_frac_rho_mayor_100x__d"] / GRANO, s=60, c="#7f7f7f",
           marker="^", label="IC sin integrar (ρ=+0.82)")
dd.set_xscale("log")
dd.axhline(0, color="k", lw=0.8)
dd.set_xlabel("T* — triángulos disponibles para repartir")
dd.set_ylabel("`solap` − `disj` (partículas)")
dd.set_title("D · cuanto más hay para repartir, más importa dónde se pone")
dd.legend(fontsize=8)

# --- E: la escalera de umbrales de densidad ---------------------------------
e = ax[0, 2]
u = pd.read_csv(os.path.join(AQUI, "cs090_fase8_f805_umbrales.csv"))
e.plot(u.umbral_x_media, u.ini_delta / GRANO, "o-", color="#7f7f7f",
       label="IC sin integrar (t=0)")
e.plot(u.umbral_x_media, u.fin_delta / GRANO, "s-", color="#d62728",
       label="motor independiente (t=0.5)")
e.axhline(GRANO / GRANO, color="k", lw=0.6, ls=":")
e.axhline(0, color="k", lw=0.8)
e.axhline(0.01433 / GRANO, color="#1f77b4", lw=1.6, ls="--", label="Phantom (+28.7 part.)")
e.axvline(49453, color="k", lw=1.2, ls="-.")
e.text(49453, e.get_ylim()[1] * 0.55, " umbral real de\n sumidero de Phantom",
       fontsize=8, ha="right")
e.set_xscale("log")
e.set_xlabel("umbral de densidad, en múltiplos de la densidad media")
e.set_ylabel("`solap` − `disj` (partículas)")
e.set_title("E · a la densidad que Phantom exige, las IC no tienen NADA")
e.legend(fontsize=8, loc="lower left")

# --- F: cuántos signos aporta cada etapa por umbral --------------------------
f = ax[1, 2]
w = 0.4
xi = np.arange(len(u))
f.bar(xi - w / 2, u.ini_signos, w, color="#7f7f7f", label="IC sin integrar")
f.bar(xi + w / 2, u.fin_signos, w, color="#d62728", label="motor independiente")
f.axhline(6, color="k", lw=0.8, ls=":")
f.axhline(12, color="#1f77b4", lw=1.6, ls="--", label="Phantom: 12/12")
f.set_xticks(xi)
f.set_xticklabels([f"{int(v):,}×" for v in u.umbral_x_media], rotation=45, fontsize=8)
f.set_ylabel("grafos con `solap` > `disj` (de 12)")
f.set_xlabel("umbral de densidad")
f.set_title("F · concordancia de signo, umbral por umbral")
f.legend(fontsize=8)

fig.suptitle("F8-05 · F7-03 en un integrador independiente de Phantom (12 grafos, brazos `solap` y `disj`)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
salida = os.path.join(AQUI, "cs090_fase8_f805_comparacion.png")
fig.savefig(salida, dpi=140)
print("escrito", salida)
