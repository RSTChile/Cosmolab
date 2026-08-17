#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODELO NULO DEL GRANO (exceso de concentracion) en Z_n
=======================================================
Pregunta de fondo (sesion kappa_Delta):
  El exceso de concentracion ~0.016 a N~2000 decae como 1/sqrt(N).
  ¿Es eso firma estructural de kappa_Delta (derivacion) o solo
  fluctuacion combinatoria de tamano finito (calibracion trivial)?

Este script NO simula el sistema dinamico de Cosmogenesis (no se dispone de
su codigo). Calcula la LINEA BASE SIN DINAMICA: N nodos, cada uno con fase
uniforme en Z_n. Es el "como se ve la ausencia de estructura".

Un sistema dinamico tiene firma de kappa_Delta si y solo si su grano se APARTA
de esta linea base, sea en:
  (a) el exponente del decaimiento en N   (nulo => exactamente -1/2)
  (b) el prefactor / dependencia en n      (nulo => sqrt((n-1)/n))

Resultado analitico exacto (definicion L2):
  counts ~ Multinomial(N, 1/n).  f_k = c_k/N.
  E[ sum_k (f_k - 1/n)^2 ] = (n-1)/(n N)
  => RMS_L2 = sqrt((n-1)/(n N))    [decae 1/sqrt(N), crece con n]
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(20260630)
OUT = Path("/mnt/user-data/outputs"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Definiciones de "exceso de concentracion" (reporto tres; no se cual es la tuya)
#   excess_max = max_k f_k - 1/n                      (exceso del modo)
#   excess_L2  = sqrt( sum_k (f_k - 1/n)^2 )          (norma L2 de la desviacion)
#   excess_H   = (log n - H)/log n                    (deficit de entropia norm.)
# ---------------------------------------------------------------------------
def measure(counts):
    N = counts.sum(axis=1, keepdims=True)
    f = counts / N
    n = counts.shape[1]
    u = 1.0 / n
    excess_max = f.max(axis=1) - u
    excess_L2  = np.sqrt(((f - u) ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(f > 0, f, 1.0)
        H = -(f * np.log(p)).sum(axis=1)
    excess_H = (np.log(n) - H) / np.log(n)
    return excess_max, excess_L2, excess_H

def null_sample(n, N, trials):
    return rng.multinomial(N, [1.0 / n] * n, size=trials)

# ---------------------------------------------------------------------------
# Barrido
# ---------------------------------------------------------------------------
ns = [4, 6, 8, 12, 16]
Ns = [250, 500, 1000, 2000, 4000, 8000]
TRIALS = 6000

rows = []
for n in ns:
    for N in Ns:
        c = null_sample(n, N, TRIALS)
        em, el2, eh = measure(c)
        rows.append(dict(
            n=n, N=N, trials=TRIALS,
            excess_max=em.mean(),  excess_max_sd=em.std(),
            excess_L2=el2.mean(),  excess_L2_sd=el2.std(),
            excess_H=eh.mean(),
            L2_analitico=np.sqrt((n - 1) / (n * N)),
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT / "grain_null_sweep.csv", index=False)

# ---------------------------------------------------------------------------
# Ajustes: exponente de decaimiento en N (pendiente log-log) por n y definicion
# ---------------------------------------------------------------------------
def loglog_fit(x, y):
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, inter = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope, np.exp(inter)

fit_rows = []
for n in ns:
    sub = df[df.n == n].sort_values("N")
    for col in ["excess_max", "excess_L2", "excess_H"]:
        s, pref = loglog_fit(sub.N.values, sub[col].values)
        fit_rows.append(dict(n=n, definicion=col, exponente=s, prefactor=pref))
fitdf = pd.DataFrame(fit_rows)
fitdf.to_csv(OUT / "grain_null_fits.csv", index=False)

# Prefactor L2 esperado vs medido: si excess_L2 = sqrt((n-1)/n) * N^(-1/2),
# el prefactor del ajuste (a exponente ~ -1/2) debe ser sqrt((n-1)/n).
pref_check = []
for n in ns:
    medido = fitdf[(fitdf.n == n) & (fitdf.definicion == "excess_L2")].prefactor.values[0]
    teorico = np.sqrt((n - 1) / n)
    pref_check.append(dict(n=n, prefactor_medido=medido, prefactor_teorico=teorico,
                           error_rel=abs(medido - teorico) / teorico))
prefdf = pd.DataFrame(pref_check)

# ---------------------------------------------------------------------------
# Predicciones en el punto reportado (n=8, N=2000) para identificar definicion
# ---------------------------------------------------------------------------
c = null_sample(8, 2000, 40000)
em, el2, eh = measure(c)
punto = dict(
    excess_max=em.mean(),
    excess_L2=el2.mean(),
    excess_L2_analitico=np.sqrt(7 / (8 * 2000)),
    excess_H=eh.mean(),
)

# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel A: exceso L2 vs N (log-log) con curva analitica
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ns)))
for n, col in zip(ns, colors):
    sub = df[df.n == n].sort_values("N")
    ax[0].plot(sub.N, sub.excess_L2, "o", color=col, label=f"n={n} (MC)")
    Ngrid = np.array(Ns, float)
    ax[0].plot(Ngrid, np.sqrt((n - 1) / (n * Ngrid)), "-", color=col, alpha=0.6)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("N (numero de nodos)")
ax[0].set_ylabel("exceso L2  =  sqrt( Σ (f_k − 1/n)² )")
ax[0].set_title("Linea base nula: exceso L2 vs N\n(puntos=Monte Carlo, lineas=sqrt((n−1)/(nN)))")
ax[0].grid(True, which="both", alpha=0.25)
ax[0].legend(fontsize=8)

# Panel B: exceso a N=2000 vs n, las tres definiciones + referencia 0.016
sub2000 = df[df.N == 2000].sort_values("n")
ax[1].plot(sub2000.n, sub2000.excess_L2,  "o-", label="exceso L2")
ax[1].plot(sub2000.n, sub2000.excess_max, "s-", label="exceso max")
ax[1].plot(sub2000.n, sub2000.excess_H,   "^-", label="deficit entropia")
ax[1].axhline(0.016, color="crimson", ls="--", lw=1.5, label="0.016 reportado")
ax[1].axvline(8, color="gray", ls=":", lw=1)
ax[1].set_xlabel("n (estados de fase, Z_n)")
ax[1].set_ylabel("exceso a N=2000")
ax[1].set_title("Dependencia en n a N=2000 fijo\n(¿donde cae 0.016?)")
ax[1].grid(True, alpha=0.25)
ax[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "grain_null_model.png", dpi=140)

# ---------------------------------------------------------------------------
# Salida en consola con marcadores
# ---------------------------------------------------------------------------
print("=" * 70)
print("MODELO NULO DEL GRANO — RESULTADOS")
print("=" * 70)
print("\n[1] Exponente de decaimiento en N (nulo predice exactamente -0.5):")
for n in ns:
    s = fitdf[(fitdf.n == n) & (fitdf.definicion == "excess_L2")].exponente.values[0]
    mark = "✅" if abs(s + 0.5) < 0.02 else "❌"
    print(f"    {mark} n={n:2d}: exponente L2 = {s:+.4f}")

print("\n[2] Prefactor L2 medido vs teorico sqrt((n-1)/n):")
for _, r in prefdf.iterrows():
    mark = "✅" if r.error_rel < 0.02 else "❌"
    print(f"    {mark} n={int(r.n):2d}: medido={r.prefactor_medido:.4f}  "
          f"teorico={r.prefactor_teorico:.4f}  err={r.error_rel*100:.2f}%")

print("\n[3] Punto reportado (n=8, N=2000) — ¿que definicion da ~0.016?:")
for k, v in punto.items():
    cerca = "✅" if abs(v - 0.016) < 0.0015 else ("·" if abs(v - 0.016) < 0.005 else "⊘")
    print(f"    {cerca} {k:22s} = {v:.5f}")

print("\n[4] L2 analitico exacto en (n=8,N=2000) = sqrt(7/16000) =",
      f"{np.sqrt(7/(8*2000)):.5f}")
print("\nArchivos: grain_null_sweep.csv, grain_null_fits.csv, grain_null_model.png")
