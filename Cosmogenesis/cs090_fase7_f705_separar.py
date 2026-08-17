#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F7-05 · PASO 4 — Separar lo inseparable: tres maneras de sortear la colinealidad
=================================================================================

POR QUÉ HACE FALTA ESTE PASO
-----------------------------
El paso 3 encontró algo que decide cómo hay que leer todo lo demás: la geometría
inicial de la nube (cuánta masa nace ya en grumos) y la densidad del grafo (cuántas
aristas tiene) correlacionan a **r = −0.995**. No son dos variables: son casi la
misma cosa medida dos veces. En ese régimen, una regresión múltiple devuelve
coeficientes enormes y de signo invertido (VIF > 100) que **no significan nada**.

Analogía: querer decidir si el peso de una persona lo explica mejor su altura en
centímetros o su altura en pulgadas. El programa da un número; el número es basura.

Entonces se prueban tres caminos que NO dependen de desenredar lo inseparable:

  A. **Ejes ortogonales (PCA de 2 variables).** En vez de "densidad" y "geometría"
     por separado, se usan "lo que las dos comparten" (eje 1) y "en qué se
     diferencian" (eje 2). El eje 2 es literalmente *geometría que no es densidad*,
     y se puede leer sin colinealidad.

  B. **Estratos de densidad igualada.** Se cortan bandas angostas de grado medio y,
     dentro de cada banda (donde la densidad casi no varía), se mira si la
     geometría / la pendiente / el clustering todavía ordenan la masa.

  C. **El gemelo de densidad EXACTAMENTE idéntica (O3-B).** El rewiring de O3-B
     conserva los grados nodo por nodo: original y gemelo tienen el MISMO número de
     aristas y la MISMA distribución de grados. Cualquier diferencia entre ellos es
     estructura pura. Es el único lugar del corpus donde el clustering está medido
     junto con la masa, y el único donde la densidad está fijada por construcción.

SALIDAS
-------
  cs090_fase7_f705_separar.log
  cs090_fase7_f705_ejes_ortogonales.csv
  cs090_fase7_f705_estratos_densidad.csv
  cs090_fase7_f705_o3b_pareado.csv
  cs090_fase7_f705_caminos.png
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(AQUI, "cs090_fase7_f705_dataset_unificado.csv"))
LOG = os.path.join(AQUI, "cs090_fase7_f705_separar.log")
RNG = np.random.default_rng(705042026)
_L = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _L.append(s)


for c in ["grado_medio", "frac_masa", "pendiente", "clustering", "transitividad",
          "n_triangulos", "geoIC_fof_b0.20", "geoIC_fof_b0.30", "geoIC_knn8_cv",
          "geoIC_knn8_p90_med", "n_aristas", "kcap"]:
    D[c] = pd.to_numeric(D[c], errors="coerce")

D["dens"] = D["grado_medio"]
D["geo"] = D["geoIC_fof_b0.30"]
D["masa"] = D["frac_masa"]
D["pend"] = D["pendiente"]
D["clus"] = D["clustering"]


def z(v):
    v = np.asarray(v, float)
    s = v.std(ddof=1)
    return (v - v.mean()) / (s if s > 0 else 1.0)


def ols(y, X):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    gl = n - k
    s2 = resid @ resid / gl
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return beta, se, p, r2, resid


log("=" * 78)
log("F7-05 · PASO 4 — separar densidad de geometría sin coeficientes basura")
log("=" * 78)

# ===========================================================================
# A. Ejes ortogonales: "lo compartido" y "la geometría que no es densidad"
# ===========================================================================
log("\n" + "=" * 78)
log("A. EJES ORTOGONALES (PCA de densidad + geometría)")
log("=" * 78)
s = D.dropna(subset=["dens", "geo", "masa"]).copy()
X = np.column_stack([z(s.dens.values), z(s.geo.values)])
# PC1 = eje común (densidad y geometría van juntas), PC2 = discrepancia
U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
pc = (X - X.mean(0)) @ Vt.T
s["eje_comun"] = pc[:, 0]
s["eje_geo_no_dens"] = pc[:, 1]
var = (S ** 2) / (S ** 2).sum()
log(f"n={len(s)}   varianza explicada: eje común {var[0]:.3%}, "
    f"eje 'geometría-que-no-es-densidad' {var[1]:.3%}")
log(f"cargas eje común:  dens {Vt[0,0]:+.3f}  geo {Vt[0,1]:+.3f}")
log(f"cargas eje 2:      dens {Vt[1,0]:+.3f}  geo {Vt[1,1]:+.3f}")
log("  (el eje 2 tiene cargas de igual signo: es 'geometría MÁS agrupada de lo que")
log("   su nº de aristas haría esperar', es decir el residuo estructural.)")

y = z(s.masa.values)
Xm = np.column_stack([np.ones(len(s)), z(s.eje_comun.values), z(s.eje_geo_no_dens.values)])
beta, se, p, r2, _ = ols(y, Xm)
log(f"\nmasa ~ eje_comun + eje_geo_no_dens    R²={r2:.3f}   (VIF = 1.0 por construcción)")
log(f"   eje_comun         beta={beta[1]:+.3f} ± {se[1]:.3f}   p={p[1]:.3g}")
log(f"   eje_geo_no_dens   beta={beta[2]:+.3f} ± {se[2]:.3f}   p={p[2]:.3g}")
r_e2, p_e2 = stats.spearmanr(s.masa, s.eje_geo_no_dens)
log(f"   rho(masa, eje_geo_no_dens) = {r_e2:+.3f} (p={p_e2:.3g})")

# con dummies de experimento
from itertools import islice
niv = sorted(s.exp.unique())[1:]
Dex = np.column_stack([(s.exp == n).astype(float).values for n in niv])
Xm2 = np.column_stack([np.ones(len(s)), z(s.eje_comun.values),
                       z(s.eje_geo_no_dens.values), Dex])
b2, se2, p2, r22, _ = ols(y, Xm2)
log(f"\nmismo modelo + experimento de origen   R²={r22:.3f}")
log(f"   eje_comun         beta={b2[1]:+.3f} ± {se2[1]:.3f}   p={p2[1]:.3g}")
log(f"   eje_geo_no_dens   beta={b2[2]:+.3f} ± {se2[2]:.3f}   p={p2[2]:.3g}")

ejes = s[["exp", "rule_id", "seed", "brazo", "dens", "geo", "masa", "pend",
          "eje_comun", "eje_geo_no_dens"]].copy()
ejes.to_csv(os.path.join(AQUI, "cs090_fase7_f705_ejes_ortogonales.csv"), index=False)

# ¿y la pendiente, en el mismo marco ortogonal?
s3 = s.dropna(subset=["pend"])
Xm3 = np.column_stack([np.ones(len(s3)), z(s3.eje_comun.values),
                       z(s3.eje_geo_no_dens.values), z(s3.pend.values)])
b3, se3, p3, r23, _ = ols(z(s3.masa.values), Xm3)
log(f"\nmasa ~ eje_comun + eje_geo_no_dens + pendiente   n={len(s3)}  R²={r23:.3f}")
for i, nm in enumerate(["eje_comun", "eje_geo_no_dens", "pendiente"], start=1):
    log(f"   {nm:18s} beta={b3[i]:+.3f} ± {se3[i]:.3f}   p={p3[i]:.3g}")

# ===========================================================================
# B. Estratos de densidad igualada
# ===========================================================================
log("\n" + "=" * 78)
log("B. ESTRATOS DE DENSIDAD IGUALADA")
log("=" * 78)
log("Dentro de cada banda angosta de grado medio la densidad casi no varía.")
log("Si algo más ordena la masa ahí adentro, ese algo NO es densidad.")

filas = []
for nb in (4, 6, 8):
    s = D.dropna(subset=["dens", "masa"]).copy()
    s["banda"] = pd.qcut(s.dens, nb, labels=False, duplicates="drop")
    log(f"\n--- {nb} bandas ---")
    for v in ["geo", "geo_cv", "pend", "kcap"]:
        col = {"geo": "geo", "geo_cv": "geoIC_knn8_cv", "pend": "pend", "kcap": "kcap"}[v]
        rs, ns = [], []
        for b, g in s.groupby("banda"):
            gg = g.dropna(subset=[col])
            if len(gg) < 8 or gg[col].std() == 0:
                continue
            r, _ = stats.spearmanr(gg.masa, gg[col])
            if np.isfinite(r):
                rs.append(r); ns.append(len(gg))
        if not rs:
            continue
        rs = np.array(rs); ns = np.array(ns)
        # combinación por z de Fisher ponderada por n-3
        zf = np.arctanh(np.clip(rs, -0.999, 0.999))
        w = ns - 3
        zc = (zf * w).sum() / np.sqrt((w).sum())
        pc_ = 2 * stats.norm.sf(abs(zc))
        log(f"   {v:7s}: rho por banda = [{', '.join(f'{x:+.2f}' for x in rs)}]"
            f"   n=[{','.join(str(x) for x in ns)}]")
        log(f"            combinado (Fisher) z={zc:+.2f}  p={pc_:.3g}   "
            f"rho medio={np.tanh(zf.mean()):+.3f}")
        filas.append(dict(n_bandas=nb, variable=v, rho_medio=float(np.tanh(zf.mean())),
                          z_fisher=zc, p=pc_, n_total=int(ns.sum()),
                          rhos=";".join(f"{x:.3f}" for x in rs)))
log("\n--- El control fino: DENTRO de cada banda, ¿la geometría sigue ordenando la")
log("    masa una vez que se le descuenta la densidad residual de la propia banda? ---")
for var, col in [("geo", "geo"), ("geo_cv", "geoIC_knn8_cv"),
                 ("pico_local", "geoIC_knn8_p90_med"), ("pend", "pend")]:
    for nb in (4, 6, 8):
        s = D.dropna(subset=["dens", "masa", col]).copy()
        s["banda"] = pd.qcut(s.dens, nb, labels=False, duplicates="drop")
        rs, ns = [], []
        for bnd, g in s.groupby("banda"):
            if len(g) < 10:
                continue
            # residualiza masa y la variable contra la densidad que aún varía adentro
            Zc = np.column_stack([np.ones(len(g)), g.dens.values])
            ry = g.masa.values - Zc @ np.linalg.lstsq(Zc, g.masa.values, rcond=None)[0]
            rx = g[col].values - Zc @ np.linalg.lstsq(Zc, g[col].values, rcond=None)[0]
            if rx.std() == 0:
                continue
            r, _ = stats.pearsonr(ry, rx)
            rs.append(r); ns.append(len(g))
        if rs:
            rs = np.array(rs); ns = np.array(ns)
            zf = np.arctanh(np.clip(rs, -0.999, 0.999)); w = ns - 4
            zc = (zf * w).sum() / np.sqrt(w.sum())
            log(f"   {var:11s} {nb} bandas: r parcial(masa, {var} | dens) = "
                f"[{', '.join(f'{x:+.2f}' for x in rs)}]  ->  z={zc:+.2f}, "
                f"p={2*stats.norm.sf(abs(zc)):.3g}")
            filas.append(dict(n_bandas=nb,
                              variable=f"{var}|dens (parcial dentro de banda)",
                              rho_medio=float(np.tanh(zf.mean())), z_fisher=zc,
                              p=2 * stats.norm.sf(abs(zc)), n_total=int(ns.sum()),
                              rhos=";".join(f"{x:.3f}" for x in rs)))
pd.DataFrame(filas).to_csv(os.path.join(AQUI, "cs090_fase7_f705_estratos_densidad.csv"),
                           index=False)

# ===========================================================================
# C. El gemelo de densidad idéntica (O3-B): el único test limpio del clustering
# ===========================================================================
log("\n" + "=" * 78)
log("C. O3-B — GEMELOS CON LA MISMA DENSIDAD EXACTA (n=12 pares)")
log("=" * 78)
b = D[D.exp == "O3B_rewiring"].copy()
piv = b.pivot_table(index=["rule_id", "seed"], columns="brazo",
                    values=["masa", "dens", "geo", "geoIC_knn8_cv", "geoIC_fof_b0.20",
                            "geoIC_knn8_p90_med", "pend", "clus", "transitividad",
                            "n_triangulos", "n_aristas"])
piv.columns = [f"{a}_{c}" for a, c in piv.columns]
piv = piv.reset_index()
log(f"pares completos: {len(piv)}")
log("verificación de que la densidad está fijada por construcción: "
    f"max |dens_orig - dens_rewire| = "
    f"{(piv.dens_orig - piv.dens_rewire).abs().max():.3e}")
log("verificación nº de aristas idéntico: "
    f"{int((piv.n_aristas_orig == piv.n_aristas_rewire).sum())}/{len(piv)}")

dcols = ["masa", "geo", "geoIC_fof_b0.20", "geoIC_knn8_cv", "geoIC_knn8_p90_med",
         "pend", "clus", "transitividad", "n_triangulos"]
for c in dcols:
    piv[f"d_{c}"] = piv[f"{c}_orig"] - piv[f"{c}_rewire"]

log("\nDiferencias original − gemelo rebarajado (mediana, signos, Wilcoxon):")
o3b_filas = []
for c in dcols:
    v = piv[f"d_{c}"].dropna()
    if len(v) < 5:
        continue
    pos = int((v > 0).sum()); neg = int((v < 0).sum())
    ps = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
    try:
        pw = stats.wilcoxon(v).pvalue
    except Exception:
        pw = np.nan
    log(f"   d_{c:20s} mediana={v.median():+.5f}  gana_orig={pos}/{pos+neg}"
        f"  p_signos={ps:.4f}  p_wilcoxon={pw:.4f}")
    o3b_filas.append(dict(variable=c, n=len(v), mediana=v.median(), media=v.mean(),
                          gana_orig=pos, gana_rewire=neg, p_signos=ps, p_wilcoxon=pw))

log("\n¿Qué diferencia predice quién gana masa? (correlación entre las Δ, n=12)")
for c in ["geo", "geoIC_fof_b0.20", "geoIC_knn8_cv", "geoIC_knn8_p90_med",
          "pend", "clus", "transitividad", "n_triangulos"]:
    v = piv[["d_masa", f"d_{c}"]].dropna()
    if len(v) < 6:
        continue
    r, p = stats.spearmanr(v.d_masa, v[f"d_{c}"])
    rp, pp = stats.pearsonr(v.d_masa, v[f"d_{c}"])
    log(f"   rho(d_masa, d_{c:20s}) = {r:+.3f} (p={p:.3g})    r={rp:+.3f} (p={pp:.3g})")
    o3b_filas.append(dict(variable=f"d_masa~d_{c}", n=len(v), mediana=np.nan,
                          media=np.nan, gana_orig=np.nan, gana_rewire=np.nan,
                          p_signos=p, p_wilcoxon=pp, rho=r, pearson=rp))

# mediación pareada: Δclustering → Δgeometría → Δmasa (densidad ya fijada)
v = piv[["d_masa", "d_clus", "d_geo"]].dropna()
if len(v) >= 8:
    x, m, yy = z(v.d_clus.values), z(v.d_geo.values), z(v.d_masa.values)
    a = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), m, rcond=None)[0][1]
    sol = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, m]), yy, rcond=None)[0]
    cp, bb = sol[1], sol[2]
    c_tot = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), yy, rcond=None)[0][1]
    boots = []
    for _ in range(5000):
        i = RNG.choice(len(v), len(v), replace=True)
        try:
            xa, ma, ya = x[i], m[i], yy[i]
            aa = np.linalg.lstsq(np.column_stack([np.ones(len(i)), xa]), ma, rcond=None)[0][1]
            s2 = np.linalg.lstsq(np.column_stack([np.ones(len(i)), xa, ma]), ya,
                                 rcond=None)[0]
            boots.append(aa * s2[2])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    log(f"\nMediación PAREADA con densidad fijada (n={len(v)}):")
    log(f"   Δclustering → Δgeometría  a = {a:+.3f}")
    log(f"   Δgeometría → Δmasa | Δclus  b = {bb:+.3f}")
    log(f"   Δclustering → Δmasa total c = {c_tot:+.3f}   directo c' = {cp:+.3f}")
    log(f"   indirecto a·b = {a*bb:+.4f}   IC95% [{lo:+.4f}, {hi:+.4f}]"
        + ("  (excluye 0)" if lo * hi > 0 else "  (INCLUYE 0)"))
    o3b_filas.append(dict(variable="mediacion_pareada_clus->geo->masa", n=len(v),
                          mediana=a * bb, media=np.nan, gana_orig=np.nan,
                          gana_rewire=np.nan, p_signos=np.nan, p_wilcoxon=np.nan,
                          rho=lo, pearson=hi))
# La misma mediación pareada, pero con la GEOMETRÍA LOCAL como eslabón del medio.
# El FoF b=0.30 mide "cuánta masa cae en grumos grandes" (una escala global); el
# cociente p90/mediana de la densidad de 8 vecinos mide "cuán alto llega el pico
# local respecto de lo típico" — que es la escala donde vive un triángulo.
v = piv[["d_masa", "d_clus", "d_geoIC_knn8_p90_med"]].dropna()
if len(v) >= 8:
    x, m, yy = (z(v.d_clus.values), z(v.d_geoIC_knn8_p90_med.values), z(v.d_masa.values))
    a = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), m, rcond=None)[0][1]
    sol = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, m]), yy, rcond=None)[0]
    cp, bb = sol[1], sol[2]
    c_tot = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), yy, rcond=None)[0][1]
    boots = []
    for _ in range(5000):
        i = RNG.choice(len(v), len(v), replace=True)
        try:
            xa, ma, ya = x[i], m[i], yy[i]
            aa = np.linalg.lstsq(np.column_stack([np.ones(len(i)), xa]), ma, rcond=None)[0][1]
            s2 = np.linalg.lstsq(np.column_stack([np.ones(len(i)), xa, ma]), ya, rcond=None)[0]
            boots.append(aa * s2[2])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    r_am, p_am = stats.spearmanr(v.d_clus, v.d_geoIC_knn8_p90_med)
    log(f"\nMediación PAREADA con GEOMETRÍA LOCAL como eslabón (n={len(v)}):")
    log(f"   Δclustering → Δpico local (p90/mediana)  a = {a:+.3f}   "
        f"[rho={r_am:+.3f}, p={p_am:.3g}]")
    log(f"   Δpico local → Δmasa | Δclus              b = {bb:+.3f}")
    log(f"   Δclustering → Δmasa   total c = {c_tot:+.3f}   directo c' = {cp:+.3f}")
    log(f"   indirecto a·b = {a*bb:+.4f}   IC95% [{lo:+.4f}, {hi:+.4f}]"
        + ("  (excluye 0)" if lo * hi > 0 else "  (INCLUYE 0)"))
    log(f"   proporción mediada = {a*bb/c_tot:.1%}" if abs(c_tot) > 1e-9 else "")
    o3b_filas.append(dict(variable="mediacion_pareada_clus->picolocal->masa", n=len(v),
                          mediana=a * bb, media=c_tot, gana_orig=np.nan,
                          gana_rewire=np.nan, p_signos=np.nan, p_wilcoxon=np.nan,
                          rho=lo, pearson=hi))

pd.DataFrame(o3b_filas).to_csv(os.path.join(AQUI, "cs090_fase7_f705_o3b_pareado.csv"),
                               index=False)

# ===========================================================================
# D. La hipótesis, puesta a prueba en una sola frase numérica
# ===========================================================================
log("\n" + "=" * 78)
log("D. LA HIPÓTESIS DE F7-05, PUNTO POR PUNTO")
log("=" * 78)
s = D.dropna(subset=["dens", "geo", "masa"])
r_dg, _ = stats.pearsonr(s.dens, s.geo)
log(f"1) 'densidad → geometría → masa (efecto grande)':")
log(f"   densidad y geometría inicial correlacionan r={r_dg:+.3f} sobre n={len(s)}.")
log(f"   El eslabón a (densidad→geometría) es prácticamente una identidad, así que")
log(f"   la cadena NO se puede estimar como mediación: los dos eslabones son la")
log(f"   misma variable. Lo que sí se puede afirmar es el bloque conjunto:")
b_, se_, p_, r2_, _ = ols(z(s.masa.values),
                          np.column_stack([np.ones(len(s)), z(s.dens.values)]))
log(f"   masa ~ densidad sola: R²={r2_:.3f}, beta={b_[1]:+.3f} (p={p_[1]:.3g})")

log(f"\n2) 'clustering → geometría local → +5% residual':")
if len(piv):
    dm = piv.d_masa.dropna()
    log(f"   El residual existe y se reproduce: el original le gana al gemelo de")
    log(f"   grados idénticos en {int((dm>0).sum())}/{len(dm)} pares, "
        f"mediana Δ={dm.median():+.5f} "
        f"({dm.median()/piv[['masa_orig','masa_rewire']].mean().mean()*100:+.1f}% relativo).")
    vv = piv[["d_masa", "d_clus"]].dropna()
    r, p = stats.spearmanr(vv.d_masa, vv.d_clus)
    log(f"   Δclustering ordena quién gana: rho={r:+.3f} (p={p:.3g}, n={len(vv)}).")
    vv2 = piv[["d_masa", "d_geo"]].dropna()
    r2b, p2b = stats.spearmanr(vv2.d_masa, vv2.d_geo)
    log(f"   Δgeometría inicial ordena quién gana: rho={r2b:+.3f} (p={p2b:.3g}).")

# ===========================================================================
# E. Figura: el diagrama de caminos con los números que sobreviven
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
axes = axes.ravel()

ax = axes[0]
sp = D.dropna(subset=["dens", "geo", "masa"])
sc = ax.scatter(sp.dens, sp.masa, c=sp.geo, cmap="viridis", s=26,
                edgecolor="k", linewidth=0.25)
ax.set_xlabel("densidad del grafo (grado medio 2E/N)")
ax.set_ylabel("fracción de masa en sumideros")
ax.set_title(f"Densidad vs masa (n={len(sp)})\n"
             f"rho = {stats.spearmanr(sp.dens, sp.masa)[0]:+.3f}", fontsize=10)
plt.colorbar(sc, ax=ax, label="geometría inicial (FoF b=0.30)")
ax.grid(alpha=0.25)

ax = axes[1]
ax.scatter(sp.dens, sp.geo, s=26, c="#2b6cb0", edgecolor="k", linewidth=0.25)
ax.set_xlabel("densidad del grafo (grado medio)")
ax.set_ylabel("geometría inicial (FoF b=0.30)")
ax.set_title(f"El problema: densidad y geometría son casi la misma variable\n"
             f"r = {r_dg:+.4f}  (VIF > 100 en cualquier modelo que use las dos)",
             fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
if len(piv):
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    vv = piv[["d_masa", "d_clus"]].dropna()
    ax.scatter(vv.d_clus, vv.d_masa, s=70, c="#c05621", edgecolor="k")
    for _, rw in piv.iterrows():
        if np.isfinite(rw.get("d_clus", np.nan)):
            ax.annotate(str(rw.rule_id).replace("A2-B0-C2-", ""),
                        (rw.d_clus, rw.d_masa), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
    r, p = stats.spearmanr(vv.d_clus, vv.d_masa)
    ax.set_xlabel("Δ clustering (original − gemelo rebarajado)")
    ax.set_ylabel("Δ fracción de masa")
    ax.set_title(f"O3-B: densidad EXACTAMENTE igual (n={len(vv)} pares)\n"
                 f"rho = {r:+.3f} (p={p:.3g})", fontsize=10)
    ax.grid(alpha=0.25)

# 4º panel: el único camino que sobrevive en TODOS los subconjuntos
ax = axes[3]
sq = D.dropna(subset=["dens", "masa", "geoIC_knn8_p90_med"])
Zc = np.column_stack([np.ones(len(sq)), sq.dens.values])
ry = sq.masa.values - Zc @ np.linalg.lstsq(Zc, sq.masa.values, rcond=None)[0]
rx = sq.geoIC_knn8_p90_med.values - Zc @ np.linalg.lstsq(
    Zc, sq.geoIC_knn8_p90_med.values, rcond=None)[0]
for e, g in sq.groupby("exp"):
    mk = sq.exp.values == e
    ax.scatter(rx[mk], ry[mk], s=24, label=e, alpha=0.85, edgecolor="k", linewidth=0.2)
rr, pp_ = stats.pearsonr(ry, rx)
ax.set_xlabel("pico local del gas inicial (p90/mediana), descontada la densidad")
ax.set_ylabel("fracción de masa, descontada la densidad")
ax.set_title(f"Lo único que sobrevive en TODOS los subconjuntos\n"
             f"r parcial = {rr:+.3f} (n={len(sq)}); dentro de cada experimento "
             f"+0.64 a +0.90", fontsize=10)
ax.legend(fontsize=6.5, loc="best")
ax.grid(alpha=0.25)

fig.suptitle("F7-05 · qué camino sobrevive al condicionar — reanálisis de 254 corridas de Phantom",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
png = os.path.join(AQUI, "cs090_fase7_f705_caminos.png")
fig.savefig(png, dpi=140)
log(f"\nFigura -> {png}")

with open(LOG, "w") as fh:
    fh.write("\n".join(_L) + "\n")
log(f"log -> {LOG}")

# ===========================================================================
# F. Control de robustez: ¿los caminos que sobreviven a igualar densidad
#    sobreviven también a NO mezclar resoluciones ni diseños?
# ===========================================================================
log("\n" + "=" * 78)
log("F. ROBUSTEZ: cada camino, descontando densidad, sin mezclar resoluciones")
log("=" * 78)
log("Mezclar N=2000 con N=4000 puede fabricar correlaciones que no existen dentro")
log("de ninguna de las dos (paradoja de Simpson). Se repite el mismo cálculo por")
log("separado.")

rob = []
for var, col in [("geo_FoF", "geo"), ("CV_local", "geoIC_knn8_cv"),
                 ("pico_local", "geoIC_knn8_p90_med"), ("pendiente", "pend")]:
    sub_all = D.dropna(subset=["dens", "masa", col]).copy()
    log(f"\n  --- {var} ---")
    for etq, sb in [("todo (mezclando N)", sub_all),
                    ("sólo N=2000", sub_all[sub_all.N_nodos == 2000]),
                    ("sin controles ER", sub_all[sub_all.brazo != "control_ER"])]:
        if len(sb) < 12:
            continue
        Zc = np.column_stack([np.ones(len(sb)), sb.dens.values])
        ry = sb.masa.values - Zc @ np.linalg.lstsq(Zc, sb.masa.values, rcond=None)[0]
        rx = sb[col].values - Zc @ np.linalg.lstsq(Zc, sb[col].values, rcond=None)[0]
        r, p = stats.pearsonr(ry, rx)
        log(f"     [{etq:18s}] n={len(sb):3d}  r parcial(masa, {var} | dens) = "
            f"{r:+.3f} (p={p:.3g})")
        rob.append(dict(variable=var, subconjunto=etq, n=len(sb), r_parcial=r, p=p))
    for e, g in sub_all.groupby("exp"):
        if len(g) < 12:
            continue
        Zc = np.column_stack([np.ones(len(g)), g.dens.values])
        ry = g.masa.values - Zc @ np.linalg.lstsq(Zc, g.masa.values, rcond=None)[0]
        rx = g[col].values - Zc @ np.linalg.lstsq(Zc, g[col].values, rcond=None)[0]
        r, p = stats.pearsonr(ry, rx)
        log(f"        dentro de {e:18s} n={len(g):3d}  r={r:+.3f} (p={p:.3g})")
        rob.append(dict(variable=var, subconjunto=f"dentro de {e}", n=len(g),
                        r_parcial=r, p=p))
pd.DataFrame(rob).to_csv(os.path.join(AQUI, "cs090_fase7_f705_robustez.csv"), index=False)

with open(LOG, "w") as fh:
    fh.write("\n".join(_L) + "\n")
