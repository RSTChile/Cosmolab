#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F7-05 · PASO 3 — ¿Qué camino sobrevive cuando los demás quedan condicionados?
=============================================================================

LA PREGUNTA (en simple)
-----------------------
Tenemos cuatro candidatos a "lo que realmente mueve la masa acretada":

  1. DENSIDAD      — cuántas aristas tiene el grafo (cuán tupido está tejido).
  2. CLUSTERING    — cuántos triángulos cierra (cuán "en grupitos" está tejido).
  3. GEOMETRÍA     — cuán apelotonada nace la nube de gas ANTES de integrar nada.
  4. PENDIENTE     — el observable geométrico con el que veníamos trabajando.

Los cuatro están enredados entre sí. La pregunta NO es "cuál explica más R²"
(eso lo gana casi siempre el más colineal), sino **cuál sigue en pie cuando a los
otros se les descuenta su parte**.

Analogía: cuatro personas empujan el mismo carro. Preguntar "quién empuja más"
es inútil si están apretadas hombro con hombro. Lo que se hace acá es sacar a
tres del carro por vez y ver si el que queda todavía lo mueve.

QUÉ HACE ESTE SCRIPT
--------------------
A. Pega la geometría inicial (medida sobre las IC guardadas en disco) al dataset
   unificado y lo vuelve a escribir completo.
B. Matriz de correlaciones (Pearson y Spearman) con su n real por par.
C. Correlaciones PARCIALES y SEMIPARCIALES de cada camino con la masa,
   condicionando a los demás y al experimento de origen.
D. Regresión múltiple con VIF, declarando explícitamente qué coeficientes quedan
   ininterpretables por colinealidad.
E. Mediación en cadena  densidad → geometría → masa  con efecto indirecto por
   bootstrap (y el eslabón del clustering donde hay datos para medirlo).
F. Control de SIMPSON: la misma pendiente estimada globalmente y dentro de cada
   experimento por separado, más el modelo con efectos fijos de experimento.
G. Bloques homogéneos: cada diseño analizado solo, y comparado con el conjunto.

NO corre Phantom, no genera grafos, no toca ningún archivo previo.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(AQUI, "cs090_fase7_f705_dataset_unificado.csv")
GEO = os.path.join(AQUI, "cs090_fase7_f705_geometria_ic_todas.csv")
LOG = os.path.join(AQUI, "cs090_fase7_f705_mediacion.log")
OUT_CORR = os.path.join(AQUI, "cs090_fase7_f705_correlaciones.csv")
OUT_MOD = os.path.join(AQUI, "cs090_fase7_f705_modelos.csv")
OUT_MED = os.path.join(AQUI, "cs090_fase7_f705_mediacion.csv")

RNG = np.random.default_rng(70502026)
_L = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _L.append(s)


# ===========================================================================
# HERRAMIENTAS ESTADÍSTICAS (implementadas a mano: no hay statsmodels en venv)
# ===========================================================================
def ols(y, X):
    """Mínimos cuadrados con intercepto ya incluido en X. Devuelve dict."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    gl = n - k
    s2 = resid @ resid / gl
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    sst = ((y - y.mean()) ** 2).sum()
    sse = resid @ resid
    r2 = 1 - sse / sst
    r2a = 1 - (1 - r2) * (n - 1) / gl
    return dict(beta=beta, se=se, t=t, p=p, r2=r2, r2a=r2a, resid=resid, n=n, gl=gl)


def vif(X_sin_int):
    """Factor de inflación de la varianza de cada columna (sin la constante)."""
    out = []
    for j in range(X_sin_int.shape[1]):
        y = X_sin_int[:, j]
        otras = np.delete(X_sin_int, j, axis=1)
        Z = np.column_stack([np.ones(len(y)), otras])
        r2 = ols(y, Z)["r2"]
        out.append(np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2))
    return np.array(out)


def residualizar(y, Z):
    """Saca de `y` todo lo que Z pueda explicar linealmente."""
    Z = np.column_stack([np.ones(len(y)), Z]) if Z.size else np.ones((len(y), 1))
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ b


def parcial(y, x, Z, metodo="pearson"):
    """Correlación PARCIAL: residualiza y y x contra Z, y correlaciona."""
    if metodo == "spearman":
        y = stats.rankdata(y); x = stats.rankdata(x)
        Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])]) if Z.size else Z
    ry = residualizar(y, Z)
    rx = residualizar(x, Z)
    r, p = stats.pearsonr(ry, rx)
    return r, p


def semiparcial(y, x, Z, metodo="pearson"):
    """Correlación SEMIPARCIAL: sólo a x se le descuenta Z. Mide el aporte
    ÚNICO de x a la variación bruta de y."""
    if metodo == "spearman":
        y = stats.rankdata(y); x = stats.rankdata(x)
        Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])]) if Z.size else Z
    rx = residualizar(x, Z)
    r, p = stats.pearsonr(y, rx)
    return r, p


def dummies(serie):
    """Codificación 0/1 de una variable categórica, dejando fuera el primer nivel."""
    niveles = sorted(pd.unique(serie.astype(str)))
    if len(niveles) <= 1:
        return np.zeros((len(serie), 0)), []
    cols = [(serie.astype(str) == n).astype(float).values for n in niveles[1:]]
    return np.column_stack(cols), niveles[1:]


def z(v):
    v = np.asarray(v, float)
    s = v.std(ddof=1)
    return (v - v.mean()) / (s if s > 0 else 1.0)


# ===========================================================================
# A. Dataset con geometría inicial
# ===========================================================================
log("=" * 78)
log("F7-05 · Mediación nueva — reanálisis puro sobre datos existentes")
log("=" * 78)

D = pd.read_csv(DATASET)
# idempotencia: si el dataset ya trae la geometría de una corrida anterior, se
# quita antes de volver a pegarla (evita columnas duplicadas con sufijo _x/_y)
D = D.drop(columns=[c for c in D.columns
                    if c.startswith("geoIC_") or c in ("bateria_raiz", "base",
                                                       "masa_particula", "lado_nube",
                                                       "n_aristas_cabecera")],
           errors="ignore")
G = pd.read_csv(GEO)
G["base"] = G.carpeta.str.split("/").str[-1]
G["bateria_raiz"] = G.carpeta.str.replace("/Users/alexis/phantom_cs073/", "", regex=False
                                          ).str.split("/").str[0]
D["base"] = D.carpeta.astype(str).str.split("/").str[-1]
m = D.exp == "O3A_N4000"
D.loc[m, "base"] = D.loc[m, "rule_id"] + "_" + D.loc[m, "clase"].astype(str)

cols_geo = [c for c in G.columns if c.startswith("geoIC_")] + \
           ["npart_ic", "masa_particula", "lado_nube", "n_aristas_cabecera", "bateria_raiz"]
G2 = G[["base", "npart_ic"] + [c for c in cols_geo if c != "npart_ic"]].copy()
D = D.merge(G2, left_on=["base", "N_nodos"], right_on=["base", "npart_ic"], how="left")
log(f"\nGeometría inicial pegada por (carpeta, npart): "
    f"{int(D['geoIC_fof_b0.30'].notna().sum())} / {len(D)} filas")

# verificación cruzada: el nº de aristas de la cabecera del IC contra el del CSV
v = D.dropna(subset=["n_aristas_cabecera", "n_aristas"])
if len(v):
    coincide = int((v.n_aristas_cabecera.astype(int) == v.n_aristas.astype(int)).sum())
    log(f"Verificación n_aristas (cabecera del IC vs CSV del experimento): "
        f"{coincide}/{len(v)} coinciden exactamente")
    mal = v[v.n_aristas_cabecera.astype(int) != v.n_aristas.astype(int)]
    if len(mal):
        log("  discrepancias:", mal[["exp", "rule_id", "n_aristas", "n_aristas_cabecera"]]
            .head(10).to_string())

D = D.drop(columns=["npart_ic"])
D.to_csv(DATASET, index=False)
log(f"Dataset unificado re-escrito con geometría: {DATASET} ({len(D)} x {len(D.columns)})")

# ---------------------------------------------------------------------------
# Variables de trabajo (todas CONTINUAS; ninguna clase como endpoint)
# ---------------------------------------------------------------------------
# Toda columna que se vaya a usar como número se fuerza a número: si alguna
# fuente coló un booleano o un texto, queda NaN en vez de romper el análisis.
NUMERICAS = ["seed", "N_nodos", "K", "kcap", "J", "noise", "meandeg", "n_aristas",
             "grado_medio", "clustering", "transitividad", "n_triangulos", "holon",
             "giant", "diam", "pendiente", "frac_masa", "masa_acretada", "kappa_v",
             "n_sumideros", "t_primer_sumidero", "densidad"] + \
            [c for c in D.columns if c.startswith("geoIC_")]
for c in NUMERICAS:
    if c in D.columns:
        D[c] = pd.to_numeric(D[c], errors="coerce")

D["dens"] = D["grado_medio"]                 # densidad = grado medio (2E/N)
D["geo"] = D["geoIC_fof_b0.30"]              # geometría inicial: masa en grumos FoF
D["geo_cv"] = D["geoIC_knn8_cv"]             # geometría inicial: desparejo local
D["masa"] = D["frac_masa"]                   # DESENLACE
D["pend"] = D["pendiente"]
D["clus"] = D["clustering"]

log("\n--- Composición del dataset ---")
log(D.groupby(["exp", "diseno"]).size().to_string())
log("\nN_nodos: " + str(D.N_nodos.value_counts().to_dict()))
log("brazos: " + str(D.brazo.value_counts().to_dict()))

# ===========================================================================
# B. Correlaciones crudas
# ===========================================================================
log("\n" + "=" * 78)
log("B. CORRELACIONES CRUDAS con la masa acretada (frac_masa)")
log("=" * 78)

VARS = {
    "dens": "densidad (grado medio 2E/N)",
    "n_aristas": "nº de aristas",
    "geo": "geometría inicial (FoF b=0.30, masa en grumos)",
    "geo_cv": "geometría inicial (CV de densidad local k=8)",
    "geoIC_fof_b0.20": "geometría inicial (FoF b=0.20)",
    "geoIC_knn8_p90_med": "geometría inicial (p90/mediana densidad local)",
    "pend": "pendiente corregida",
    "clus": "clustering local medio",
    "holon": "holonomía del grafo final",
    "diam": "diámetro corregido",
    "giant": "fracción en componente gigante",
    "kcap": "kcap (tope de capacidad)",
    "K": "K",
}
filas = []
for v, etq in VARS.items():
    s = D[["masa", v]].dropna()
    if len(s) < 8:
        continue
    rp, pp = stats.pearsonr(s.masa, s[v])
    rs, ps = stats.spearmanr(s.masa, s[v])
    log(f"  {etq:52s} n={len(s):4d}  r={rp:+.3f} (p={pp:.2e})  rho={rs:+.3f} (p={ps:.2e})")
    filas.append(dict(bloque="crudo_global", variable=v, etiqueta=etq, n=len(s),
                      pearson=rp, p_pearson=pp, spearman=rs, p_spearman=ps))

log("\n--- Colinealidad entre los caminos (Pearson / Spearman, n por par) ---")
pares = [("dens", "pend"), ("dens", "geo"), ("dens", "geo_cv"), ("dens", "clus"),
         ("pend", "geo"), ("pend", "clus"), ("geo", "clus"), ("dens", "kcap"),
         ("geo", "geo_cv"), ("dens", "holon"), ("pend", "holon")]
for a, b in pares:
    s = D[[a, b]].dropna()
    if len(s) < 8:
        log(f"  {a:6s} vs {b:8s}  n={len(s):4d}  (insuficiente)")
        continue
    rp, pp = stats.pearsonr(s[a], s[b])
    rs, ps = stats.spearmanr(s[a], s[b])
    log(f"  {a:6s} vs {b:8s}  n={len(s):4d}  r={rp:+.3f}  rho={rs:+.3f}")
    filas.append(dict(bloque="colinealidad", variable=f"{a}|{b}", etiqueta="", n=len(s),
                      pearson=rp, p_pearson=pp, spearman=rs, p_spearman=ps))

# ===========================================================================
# C. Parciales y semiparciales — el corazón de la tarea
# ===========================================================================
log("\n" + "=" * 78)
log("C. ¿QUÉ CAMINO SOBREVIVE AL CONDICIONAR A LOS DEMÁS?")
log("=" * 78)
log("Parcial  = correlación entre lo que queda de la masa y lo que queda del camino,")
log("           después de sacarle a los dos todo lo que explican los otros caminos.")
log("Semiparc = cuánto de la masa BRUTA explica sólo ese camino, ya limpio de los otros.")


def bloque_parciales(sub, etiqueta, caminos, controles_extra=None, con_exp=True):
    sub = sub.copy()
    necesarias = ["masa"] + caminos + (controles_extra or [])
    sub = sub.dropna(subset=necesarias)
    if len(sub) < len(caminos) + 6:
        log(f"\n[{etiqueta}] n={len(sub)} — insuficiente, se omite")
        return []
    log(f"\n[{etiqueta}]  n={len(sub)}   caminos: {caminos}"
        + (f"  + control {controles_extra}" if controles_extra else "")
        + ("  + experimento de origen" if con_exp else ""))
    Dex, nivs = dummies(sub["exp"]) if con_exp else (np.zeros((len(sub), 0)), [])
    out = []
    y = sub["masa"].values.astype(float)
    for c in caminos:
        otros = [k for k in caminos if k != c] + (controles_extra or [])
        Z = np.column_stack([sub[k].values.astype(float) for k in otros] +
                            ([Dex] if Dex.size else []))
        x = sub[c].values.astype(float)
        rp, pp = parcial(y, x, Z)
        rs, ps = parcial(y, x, Z, metodo="spearman")
        sp, spp = semiparcial(y, x, Z)
        log(f"   {c:8s} parcial r={rp:+.3f} (p={pp:.3g})   rho_parcial={rs:+.3f} (p={ps:.3g})"
            f"   semiparcial r={sp:+.3f} (p={spp:.3g})")
        out.append(dict(bloque=etiqueta, variable=c, n=len(sub),
                        parcial_r=rp, parcial_p=pp, parcial_rho=rs, parcial_rho_p=ps,
                        semiparcial_r=sp, semiparcial_p=spp,
                        controles=",".join(otros) + ("+exp" if con_exp else "")))
    return out


par_filas = []
par_filas += bloque_parciales(D, "GLOBAL dens|geo", ["dens", "geo"], con_exp=True)
par_filas += bloque_parciales(D, "GLOBAL dens|geo|pend", ["dens", "geo", "pend"], con_exp=True)
par_filas += bloque_parciales(D, "GLOBAL dens|geo|geo_cv|pend",
                              ["dens", "geo", "geo_cv", "pend"], con_exp=True)
par_filas += bloque_parciales(D, "GLOBAL sin ajuste por experimento",
                              ["dens", "geo", "pend"], con_exp=False)
par_filas += bloque_parciales(D[D.N_nodos == 2000], "SÓLO N=2000",
                              ["dens", "geo", "pend"], con_exp=True)
par_filas += bloque_parciales(D[D.brazo != "control_ER"], "SIN controles Erdős-Rényi",
                              ["dens", "geo", "pend"], con_exp=True)
# clustering: sólo donde fue medido
par_filas += bloque_parciales(D[D.clus.notna()], "O3B (único con clustering)",
                              ["dens", "geo", "pend", "clus"], con_exp=False)
par_filas += bloque_parciales(D[D.clus.notna()], "O3B clustering | dens+geo",
                              ["dens", "geo", "clus"], con_exp=False)

# ===========================================================================
# D. Regresión múltiple con VIF
# ===========================================================================
log("\n" + "=" * 78)
log("D. REGRESIÓN MÚLTIPLE (variables tipificadas) + VIF")
log("=" * 78)
log("Regla de lectura declarada de antemano (como en O3-D):")
log("  VIF < 5   -> el coeficiente se puede leer;")
log("  5-10      -> se lee con reserva;")
log("  > 10      -> el coeficiente es ININTERPRETABLE individualmente (se informa, no se usa).")

mod_filas = []


def modelo(sub, etiqueta, preds, con_exp=True):
    sub = sub.dropna(subset=["masa"] + preds).copy()
    if len(sub) < len(preds) + 8:
        log(f"\n[{etiqueta}] n={len(sub)} — insuficiente")
        return
    y = z(sub["masa"].values)
    Xc = np.column_stack([z(sub[p].values) for p in preds])
    nombres = list(preds)
    if con_exp:
        Dex, nivs = dummies(sub["exp"])
        if Dex.size:
            Xc = np.column_stack([Xc, Dex])
            nombres += [f"exp={n}" for n in nivs]
    X = np.column_stack([np.ones(len(y)), Xc])
    r = ols(y, X)
    vs = vif(Xc)
    log(f"\n[{etiqueta}]  n={r['n']}  R²={r['r2']:.3f}  R²aj={r['r2a']:.3f}")
    for i, nm in enumerate(nombres):
        b = r["beta"][i + 1]; se = r["se"][i + 1]; p = r["p"][i + 1]; vv = vs[i]
        flag = "OK " if vv < 5 else ("res" if vv < 10 else "XX ")
        log(f"   {flag} {nm:22s} beta={b:+.3f} ± {se:.3f}   p={p:.3g}   VIF={vv:.1f}"
            + ("   <-- ININTERPRETABLE por colinealidad" if vv >= 10 else ""))
        mod_filas.append(dict(modelo=etiqueta, termino=nm, n=r["n"], r2=r["r2"],
                              beta=b, se=se, p=p, vif=vv,
                              interpretable=("no" if vv >= 10 else ("reserva" if vv >= 5 else "si"))))


modelo(D, "M1 masa ~ dens", ["dens"])
modelo(D, "M2 masa ~ dens + geo", ["dens", "geo"])
modelo(D, "M3 masa ~ dens + geo + pend", ["dens", "geo", "pend"])
modelo(D, "M4 masa ~ dens + geo + geo_cv + pend", ["dens", "geo", "geo_cv", "pend"])
modelo(D, "M5 masa ~ dens + geo + pend + kcap", ["dens", "geo", "pend", "kcap"])
modelo(D, "M6 (sin dummies de experimento)", ["dens", "geo", "pend"], con_exp=False)
modelo(D[D.clus.notna()], "M7 O3B: masa ~ dens + geo + clus", ["dens", "geo", "clus"],
       con_exp=False)
modelo(D[D.clus.notna()], "M8 O3B: masa ~ dens + geo + pend + clus",
       ["dens", "geo", "pend", "clus"], con_exp=False)

# ===========================================================================
# E. Mediación en cadena con bootstrap
# ===========================================================================
log("\n" + "=" * 78)
log("E. MEDIACIÓN EN CADENA:  densidad → geometría inicial → masa")
log("=" * 78)
log("En simple: ¿la densidad llega a la masa PORQUE cambia cómo de apelotonada")
log("nace la nube, o llega por otro lado? Se estima el trozo que viaja por la")
log("geometría (indirecto) y el trozo que no (directo), con intervalos por bootstrap.")

med_filas = []


def mediacion(sub, etiqueta, X, M, Y="masa", extra=None, B=5000, estratos=None):
    cols = [X, M, Y] + (extra or [])
    s = sub.dropna(subset=cols).copy()
    if len(s) < 15:
        log(f"\n[{etiqueta}] n={len(s)} — insuficiente")
        return

    def estimar(d):
        x = z(d[X].values); mm = z(d[M].values); y = z(d[Y].values)
        C = [z(d[e].values) for e in (extra or [])]
        # a: X -> M
        Xa = np.column_stack([np.ones(len(d)), x] + C)
        a = np.linalg.lstsq(Xa, mm, rcond=None)[0][1]
        # b y c': X + M -> Y
        Xb = np.column_stack([np.ones(len(d)), x, mm] + C)
        sol = np.linalg.lstsq(Xb, y, rcond=None)[0]
        cp = sol[1]; b = sol[2]
        # c total
        Xc = np.column_stack([np.ones(len(d)), x] + C)
        c = np.linalg.lstsq(Xc, y, rcond=None)[0][1]
        return a, b, c, cp, a * b

    a, b, c, cp, ind = estimar(s)
    boots = []
    idx = np.arange(len(s))
    for _ in range(B):
        if estratos is not None:
            sel = []
            for _, g in s.groupby(estratos):
                sel.append(RNG.choice(g.index.values, size=len(g), replace=True))
            bs = s.loc[np.concatenate(sel)]
        else:
            bs = s.iloc[RNG.choice(idx, size=len(idx), replace=True)]
        try:
            boots.append(estimar(bs)[4])
        except Exception:
            pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    prop = ind / c if abs(c) > 1e-9 else np.nan
    log(f"\n[{etiqueta}]  n={len(s)}  (X={X}, M={M}, Y={Y}"
        + (f", control={extra}" if extra else "") + ")")
    log(f"   a  (X→M)          = {a:+.3f}")
    log(f"   b  (M→Y | X)      = {b:+.3f}")
    log(f"   c  (X→Y total)    = {c:+.3f}")
    log(f"   c' (X→Y directo)  = {cp:+.3f}")
    log(f"   indirecto a·b     = {ind:+.4f}   IC95% bootstrap [{lo:+.4f}, {hi:+.4f}]"
        + ("   (excluye 0)" if lo * hi > 0 else "   (INCLUYE 0)"))
    log(f"   proporción mediada = {prop:.1%}")
    med_filas.append(dict(bloque=etiqueta, X=X, M=M, Y=Y, n=len(s), a=a, b=b, c=c,
                          c_prima=cp, indirecto=ind, ic95_bajo=lo, ic95_alto=hi,
                          proporcion_mediada=prop,
                          control=",".join(extra or []),
                          excluye_cero=bool(lo * hi > 0)))


mediacion(D, "densidad → geometría → masa (global)", "dens", "geo")
mediacion(D, "densidad → geometría → masa (global, ctrl N)", "dens", "geo", extra=["N_nodos"])
mediacion(D[D.N_nodos == 2000], "densidad → geometría → masa (N=2000)", "dens", "geo")
mediacion(D[D.brazo != "control_ER"], "densidad → geometría → masa (sin ER)", "dens", "geo")
mediacion(D, "densidad → CV local → masa (global)", "dens", "geo_cv")
mediacion(D, "densidad → pendiente → masa (el camino viejo)", "dens", "pend")
mediacion(D, "pendiente → geometría → masa", "pend", "geo")
mediacion(D, "densidad → geometría → masa | descontando pendiente", "dens", "geo",
          extra=["pend"])
mediacion(D[D.clus.notna()], "O3B: clustering → geometría → masa | descontando densidad",
          "clus", "geo", extra=["dens"], B=3000)
mediacion(D[D.clus.notna()], "O3B: densidad → geometría → masa", "dens", "geo", B=3000)

# cadena de 3 eslabones donde se puede: dens → clus → geo → masa (sólo O3B)
sub = D[D.clus.notna()].dropna(subset=["dens", "clus", "geo", "masa"])
if len(sub) >= 15:
    log("\n[O3B] cadena de tres eslabones  densidad → clustering → geometría → masa")
    x, m1, m2, y = z(sub.dens.values), z(sub.clus.values), z(sub.geo.values), z(sub.masa.values)
    a1 = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), m1, rcond=None)[0][1]
    a2 = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, m1]), m2, rcond=None)[0][2]
    b3 = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x, m1, m2]), y, rcond=None)[0][3]
    log(f"   dens→clus a1={a1:+.3f} · clus→geo|dens a2={a2:+.3f} · geo→masa|dens,clus b={b3:+.3f}")
    log(f"   indirecto en serie a1·a2·b = {a1*a2*b3:+.4f}")
    med_filas.append(dict(bloque="O3B cadena 3 eslabones", X="dens", M="clus->geo", Y="masa",
                          n=len(sub), a=a1, b=b3, c=np.nan, c_prima=a2,
                          indirecto=a1 * a2 * b3, ic95_bajo=np.nan, ic95_alto=np.nan,
                          proporcion_mediada=np.nan, control="", excluye_cero=False))

# ===========================================================================
# F. Control de SIMPSON: global vs dentro de cada experimento
# ===========================================================================
log("\n" + "=" * 78)
log("F. ¿ARTEFACTO DE SIMPSON? — la misma relación, global y dentro de cada diseño")
log("=" * 78)
log("Si la relación global apunta a un lado y las de adentro de cada experimento")
log("al otro, el conjunto está mintiendo. Se compara una por una.")

sim_filas = []
for v in ["dens", "geo", "pend"]:
    s = D.dropna(subset=["masa", v])
    rg, pg = stats.spearmanr(s.masa, s[v])
    log(f"\n  {v}: rho GLOBAL = {rg:+.3f} (n={len(s)}, p={pg:.2g})")
    for e, g in s.groupby("exp"):
        if len(g) < 8:
            log(f"      {e:18s} n={len(g):3d}  (pocos)")
            continue
        r, p = stats.spearmanr(g.masa, g[v])
        log(f"      {e:18s} n={len(g):3d}  rho={r:+.3f}  p={p:.3g}"
            + ("   <-- SIGNO OPUESTO al global" if r * rg < 0 else ""))
        sim_filas.append(dict(bloque="simpson", variable=v, exp=e, n=len(g),
                              rho=r, p=p, rho_global=rg, signo_opuesto=bool(r * rg < 0)))
    # efectos fijos: se le resta a cada variable la media de su experimento
    s2 = s.copy()
    s2["_y"] = s2.masa - s2.groupby("exp").masa.transform("mean")
    s2["_x"] = s2[v] - s2.groupby("exp")[v].transform("mean")
    rw, pw = stats.pearsonr(s2._y, s2._x)
    log(f"      >> DENTRO de experimento (efectos fijos): r={rw:+.3f} (p={pw:.2g})")
    sim_filas.append(dict(bloque="efectos_fijos", variable=v, exp="TODOS", n=len(s2),
                          rho=rw, p=pw, rho_global=rg, signo_opuesto=bool(rw * rg < 0)))

# ===========================================================================
# G. Bloques homogéneos, uno por uno
# ===========================================================================
log("\n" + "=" * 78)
log("G. CADA DISEÑO POR SEPARADO (subconjuntos homogéneos)")
log("=" * 78)
hom_filas = []
for e, g in D.groupby("exp"):
    if len(g) < 10:
        continue
    log(f"\n[{e}]  n={len(g)}  diseño={g.diseno.iloc[0]}  N={sorted(g.N_nodos.unique())}")
    for v in ["dens", "geo", "geo_cv", "pend", "clus"]:
        s = g.dropna(subset=["masa", v])
        if len(s) < 8 or s[v].std() == 0:
            continue
        r, p = stats.spearmanr(s.masa, s[v])
        log(f"    rho(masa, {v:7s}) = {r:+.3f}  (n={len(s)}, p={p:.3g})")
        hom_filas.append(dict(exp=e, variable=v, n=len(s), rho=r, p=p))
    # dentro del bloque: geo condicionada a densidad y viceversa
    s = g.dropna(subset=["masa", "dens", "geo"])
    if len(s) >= 12:
        r1, p1 = parcial(s.masa.values, s.geo.values, s[["dens"]].values)
        r2, p2 = parcial(s.masa.values, s.dens.values, s[["geo"]].values)
        log(f"    parcial geo|dens = {r1:+.3f} (p={p1:.3g})   "
            f"parcial dens|geo = {r2:+.3f} (p={p2:.3g})")
        hom_filas.append(dict(exp=e, variable="geo|dens", n=len(s), rho=r1, p=p1))
        hom_filas.append(dict(exp=e, variable="dens|geo", n=len(s), rho=r2, p=p2))

# ===========================================================================
# Escritura
# ===========================================================================
pd.DataFrame(filas + [dict(bloque=r.pop("bloque"), **r) for r in []]).to_csv(OUT_CORR, index=False)
pd.DataFrame(par_filas).to_csv(os.path.join(AQUI, "cs090_fase7_f705_parciales.csv"), index=False)
pd.DataFrame(mod_filas).to_csv(OUT_MOD, index=False)
pd.DataFrame(med_filas).to_csv(OUT_MED, index=False)
pd.DataFrame(sim_filas).to_csv(os.path.join(AQUI, "cs090_fase7_f705_simpson.csv"), index=False)
pd.DataFrame(hom_filas).to_csv(os.path.join(AQUI, "cs090_fase7_f705_homogeneos.csv"), index=False)

with open(LOG, "w") as fh:
    fh.write("\n".join(_L) + "\n")
log(f"\nlog -> {LOG}")
