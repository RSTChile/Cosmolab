#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase6_o3f_control_normalizacion.py — PASO 3 de O3-F: ¿B_τ mide algo, o mide 1/masa?
==========================================================================================

La duda que esta tarea me pidió explícitamente resolver
-------------------------------------------------------
B_τ = H(v_gas) / |Ω_gas|. La Clase III acreta más, así que le queda MENOS gas: su
denominador es más chico *por construcción*. Si B_τ sale mayor en III, hay que
descartar que sea pura aritmética de la división.

Analogía: si mido "libros por estante" y a un estante le saco estantes, el cociente
sube sin que aparezca un solo libro nuevo. La pregunta es si aparecieron libros.

Cuatro pruebas
--------------
1. **Descomposición log**: log B_τ = log H - log M. Cuánta de la varianza (y cuánto de
   la diferencia pareada III-I) aporta cada término.
2. **B_τ postizo**: reemplazo H por su media global (constante) y recalculo el test
   pareado. Si el patrón de signos sobrevive con el numerador APAGADO, el observable
   no estaba midiendo entropía: estaba midiendo 1/masa.
3. **ANCOVA / control por N**: regresión de H sobre el nº de partículas difusas más
   una variable indicadora de clase. ¿Queda efecto de clase una vez descontado N?
4. **Permutación pareada de etiquetas**: barajo cuál brazo es I y cuál es III dentro
   de cada par (10.000 veces) y comparo la mediana observada de delta contra la nula.
   Es el test no paramétrico más limpio para un diseño pareado como éste.

Además reporto la sensibilidad a usar los 40 pares en vez de los 37 válidos.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260811)
N_PERM = 10000

pares = list(csv.DictReader(open(RAIZ / "cs090_fase6_o3f_btau_pares.csv", newline="")))
umbrales = sorted({p["umbral"] for p in pares}, key=lambda u: [q["umbral"] for q in pares].index(u))
umbrales = list(dict.fromkeys(p["umbral"] for p in pares))

salida = []


def arr(sub, col):
    return np.array([float(s[col]) for s in sub], dtype=float)


print("=" * 104)
print("1) DESCOMPOSICIÓN  log B_τ = log H − log M   (delta pareado III − I, entropía rarefacción v3d_std)")
print("=" * 104)
print(f"{'umbral':<14}{'d_logBtau':>11}{'d_logH':>11}{'d_logM':>11}{'aporte_H%':>11}{'aporte_M%':>11}")
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u and p["estado"] == "valido"]
    H_I, H_III = arr(sub, "Hrar_v3d_std_I"), arr(sub, "Hrar_v3d_std_III")
    M_I, M_III = arr(sub, "masa_difusa_I"), arr(sub, "masa_difusa_III")
    dH = np.log(H_III) - np.log(H_I)
    dM = np.log(M_III) - np.log(M_I)
    dB = dH - dM
    tot = np.abs(np.mean(dH)) + np.abs(np.mean(dM))
    ap_H = 100 * abs(np.mean(dH)) / tot if tot else float("nan")
    print(f"{u:<14}{np.mean(dB):>11.3e}{np.mean(dH):>11.3e}{np.mean(dM):>11.3e}"
          f"{ap_H:>11.1f}{100-ap_H:>11.1f}")
    salida.append(dict(prueba="descomposicion_log", umbral=u,
                       d_logBtau=np.mean(dB), d_logH=np.mean(dH), d_logM=np.mean(dM),
                       aporte_H_pct=ap_H, aporte_M_pct=100 - ap_H))

print()
print("=" * 104)
print("2) B_τ POSTIZO: numerador congelado en su media global (H := cte). Si el patrón sobrevive,")
print("   el observable era 1/masa disfrazado.")
print("=" * 104)
print(f"{'umbral':<14}{'Btau real III>I':>17}{'p_signos':>10}{'p_wilcox':>10}"
      f"{'Btau postizo III>I':>20}{'p_signos':>10}{'p_wilcox':>10}")
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u and p["estado"] == "valido"]
    H_I, H_III = arr(sub, "Hrar_v3d_std_I"), arr(sub, "Hrar_v3d_std_III")
    M_I, M_III = arr(sub, "masa_difusa_I"), arr(sub, "masa_difusa_III")
    H_cte = np.mean(np.concatenate([H_I, H_III]))
    filas = []
    for etiqueta, hI, hIII in (("real", H_I, H_III), ("postizo", H_cte, H_cte)):
        d = hIII / M_III - hI / M_I
        d = d[np.isfinite(d)]
        npos, nneg = int((d > 0).sum()), int((d < 0).sum())
        ps = stats.binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else float("nan")
        try:
            pw = stats.wilcoxon(d).pvalue
        except ValueError:
            pw = float("nan")
        filas.append((npos, len(d), ps, pw))
        salida.append(dict(prueba=f"btau_{etiqueta}", umbral=u, n_III_mayor=npos,
                           n_pares=len(d), p_signos=ps, p_wilcoxon=pw))
    (a, na, pa, wa), (b, nb, pb, wb) = filas
    print(f"{u:<14}{a:>10}/{na:<6}{pa:>10.4g}{wa:>10.4g}"
          f"{b:>13}/{nb:<6}{pb:>10.4g}{wb:>10.4g}")

print()
print("=" * 104)
print("3) ANCOVA: H ~ N_difuso + clase.  ¿Queda efecto de clase después de descontar el tamaño?")
print("=" * 104)
print(f"{'umbral':<14}{'coef_N':>12}{'coef_clase(III)':>17}{'p_clase':>10}{'R2':>8}")
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u and p["estado"] == "valido"]
    n = np.concatenate([arr(sub, "n_difuso_I"), arr(sub, "n_difuso_III")])
    H = np.concatenate([arr(sub, "Hrar_v3d_std_I"), arr(sub, "Hrar_v3d_std_III")])
    clase = np.concatenate([np.zeros(len(sub)), np.ones(len(sub))])
    if np.std(n) == 0:  # umbral de N fijo: la regresión sobre N no está definida
        X = np.c_[np.ones(len(H)), clase]
        nombres_n = float("nan")
    else:
        X = np.c_[np.ones(len(H)), n, clase]
    beta, *_ = np.linalg.lstsq(X, H, rcond=None)
    resid = H - X @ beta
    gl = len(H) - X.shape[1]
    s2 = resid @ resid / gl
    cov = s2 * np.linalg.pinv(X.T @ X)
    ee = np.sqrt(np.diag(cov))
    i_clase = X.shape[1] - 1
    t = beta[i_clase] / ee[i_clase]
    p = 2 * stats.t.sf(abs(t), gl)
    r2 = 1 - (resid @ resid) / ((H - H.mean()) @ (H - H.mean()))
    coef_n = beta[1] if X.shape[1] == 3 else float("nan")
    print(f"{u:<14}{coef_n:>12.3e}{beta[i_clase]:>17.3e}{p:>10.4g}{r2:>8.3f}")
    salida.append(dict(prueba="ancova_H_por_N_y_clase", umbral=u, coef_N=coef_n,
                       coef_clase=beta[i_clase], p_clase=p, R2=r2))

print()
print("=" * 104)
print("4) PERMUTACIÓN PAREADA de la etiqueta de clase (10.000 barajadas dentro de cada par)")
print("=" * 104)
metricas = ["Btau_Hrar_v3d_std", "Hrar_v3d_std", "sigma_v", "n_fil_c1.5", "masa_difusa"]
print(f"{'umbral':<14}{'metrica':<20}{'delta_obs(mediana)':>20}{'p_permutacion':>15}")
for u in umbrales:
    sub = [p for p in pares if p["umbral"] == u and p["estado"] == "valido"]
    for m in metricas:
        a, b = arr(sub, f"{m}_I"), arr(sub, f"{m}_III")
        d = b - a
        if not np.any(np.isfinite(d)) or np.allclose(d, 0):
            continue
        obs = np.median(d)
        signos = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        nulos = np.median(signos * d, axis=1)
        p = (np.sum(np.abs(nulos) >= abs(obs)) + 1) / (N_PERM + 1)
        print(f"{u:<14}{m:<20}{obs:>20.5g}{p:>15.4g}")
        salida.append(dict(prueba="permutacion_pareada", umbral=u, metrica=m,
                           delta_mediana=obs, p_permutacion=p))

print()
print("=" * 104)
print("5) SENSIBILIDAD: 37 pares válidos vs los 40 crudos (B_τ, entropía rarefacción v3d_std)")
print("=" * 104)
print(f"{'umbral':<14}{'37 válidos III>I':>18}{'p_wilcox':>10}{'40 crudos III>I':>18}{'p_wilcox':>10}")
for u in umbrales:
    linea = [u]
    for etiqueta, filtro in (("37", lambda p: p["estado"] == "valido"), ("40", lambda p: True)):
        sub = [p for p in pares if p["umbral"] == u and filtro(p)]
        d = arr(sub, "Btau_Hrar_v3d_std_III") - arr(sub, "Btau_Hrar_v3d_std_I")
        d = d[np.isfinite(d)]
        npos = int((d > 0).sum())
        try:
            pw = stats.wilcoxon(d).pvalue
        except ValueError:
            pw = float("nan")
        linea += [f"{npos}/{len(d)}", f"{pw:.4g}"]
        salida.append(dict(prueba=f"sensibilidad_{etiqueta}pares", umbral=u,
                           n_III_mayor=npos, n_pares=len(d), p_wilcoxon=pw))
    print(f"{linea[0]:<14}{linea[1]:>18}{linea[2]:>10}{linea[3]:>18}{linea[4]:>10}")

campos = sorted({k for f in salida for k in f})
with open(RAIZ / "cs090_fase6_o3f_control_normalizacion.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=campos)
    w.writeheader()
    w.writerows(salida)
print("\nescrito cs090_fase6_o3f_control_normalizacion.csv")
