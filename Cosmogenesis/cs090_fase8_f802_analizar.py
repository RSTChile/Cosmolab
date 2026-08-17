#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase8_f802_analizar.py — FASE VIII F8-02: ¿el pico local de densidad inicial CAUSA más masa
acretada, o sólo la acompaña?

QUÉ LEE Y CÓMO
---------------
Las 60 corridas de `/Users/alexis/phantom_cs073/bateria_fase8_f802_pico/` (12 condiciones iniciales
base × 5 niveles de pico), fabricadas por `cs090_fase8_f802_pico.py` y corridas por
`cs090_fase8_f802_correr.py`.

**Observable principal, declarado ANTES de correr nada:** la *fracción de masa en sumideros*, es decir
el criterio de densidad real de Phantom (`rho_crit_cgs=1000` = 49.453 × la densidad media de estas
cajas). **No se usa FoF laxo**: `FASE8_F805_f703_solver_independiente_CS.md` mostró que el FoF a
ell=1.0 puede **invertir el signo** sobre los mismísimos dumps.

Secundarios (también declarados): el pico logrado en la IC, el tiempo al primer sumidero y κ_V.

VERIFICACIÓN CRUZADA OBLIGATORIA contra el `meta_regla.json` de CADA carpeta, antes de que ninguna
corrida entre en la estadística (lección del bug de colisión de nombres de Fase V-B):
  - la tarea declarada es FASE8_F802_pico_local_manipulado,
  - el (rule_id, seed, nivel) declarado coincide con el nombre de la carpeta,
  - la carpeta declarada dentro del meta es la carpeta donde está el meta,
  - los cinco niveles de una IC declaran **el mismo sello sha256 de grafo** y el mismo nº de aristas
    (ésa es la afirmación central del diseño: el grafo NO cambió),
  - masa total = 18800 y N = 2000 en los cinco.
La unión con el CSV de condiciones iniciales es por **(rule_id, seed, nivel)**, nunca por rule_id solo.

QUÉ ESTADÍSTICA SE HACE Y POR QUÉ
----------------------------------
Diseño **pareado con tratamientos ORDENADOS**: cada IC es su propio control a través de sus 5 niveles.
  1. **Page (L) para alternativas ordenadas** — el test específico para "¿hay tendencia monótona a
     través de niveles ordenados?" en bloques. Es el test que corresponde a la pregunta.
  2. **Friedman** — ¿hay ALGUNA diferencia entre niveles? (sin suponer orden).
  3. **Wilcoxon pareado y test de signos** de cada nivel contra el nivel de identidad (L1) y del
     extremo alto contra el bajo (L4 − L0).
  4. **Spearman dentro de cada IC** entre el pico LOGRADO y la masa (12 coeficientes independientes),
     más la versión agrupada centrando ambas variables por IC (efectos fijos de IC).
  5. Todo tamaño de efecto se reporta **en partículas**: 1 partícula = 0.0005 de fracción de masa a
     N=2000, y el **piso práctico de un pareado es ~5 partículas = 0.0025** (medido en F8-01). Por
     debajo de eso se dice "por debajo del piso", nunca "nulo".
  6. **Control del eje-1**: se reporta cuánto se movió la geometría GLOBAL (masa en grumos FoF b=0.30,
     que F7-05 mostró que es la densidad disfrazada, r=−0.9945 con el grado medio). Si el pico se
     movió mucho y el eje-1 no, la intervención hizo lo que decía hacer.
  7. **Reproducibilidad**: L1 sale de un archivo byte a byte idéntico al original, así que su
     `frac_masa` tiene que coincidir con la histórica de Fase V-B. Es una prueba del tubo entero.

SALIDAS
-------
  cs090_fase8_f802_crudo.csv        una fila por corrida (60): IC + Phantom
  cs090_fase8_f802_por_ic.csv       una fila por IC base: pendiente, Spearman, Δ extremos
  cs090_fase8_f802_estadistica.csv  Page, Friedman, Wilcoxon, signos, correlaciones
  cs090_fase8_f802_pico.png         los cuatro paneles

No declara cierre ni veredicto. No modifica nada existente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AQUI = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, AQUI)

from cs090_fase5b_analizar import analizar_carpeta          # sólo import, script congelado

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f802_pico")
RUTA_IC = f"{AQUI}/cs090_fase8_f802_ic_transformadas.csv"
RUTA_CRUDO = f"{AQUI}/cs090_fase8_f802_crudo.csv"
RUTA_POR_IC = f"{AQUI}/cs090_fase8_f802_por_ic.csv"
RUTA_ESTAD = f"{AQUI}/cs090_fase8_f802_estadistica.csv"
RUTA_PNG = f"{AQUI}/cs090_fase8_f802_pico.png"

NIVELES = ["L0", "L1", "L2", "L3", "L4"]
NIVEL_BASE = "L1"                       # el nivel de identidad (a = 0)
GRANO = 0.0005                          # 1 partícula de 2000 (grano del instrumento)
PISO_PAREADO = 5 * GRANO                # piso práctico de un pareado, medido en F8-01: ~5 partículas


# =============================================================================================
# 1) PAGE (L): el test de tendencia monótona para bloques con tratamientos ORDENADOS
# =============================================================================================
def page_l(M):
    """`M` es (n_bloques × k_tratamientos) con las columnas YA en el orden hipotetizado (creciente).
    Devuelve L, z y p (una cola, H1: la mediana crece con el orden de las columnas).

    Cómo funciona, en simple: dentro de cada bloque se ordenan los k valores de peor a mejor y se les
    pone puesto (1..k). Si la hipótesis de tendencia es cierta, los puestos altos caen sistemáticamente
    en las últimas columnas. L pesa cada columna por su posición y mide justamente eso."""
    M = np.asarray(M, dtype=float)
    n, k = M.shape
    rangos = np.apply_along_axis(stats.rankdata, 1, M)      # rangos dentro de cada bloque
    R = rangos.sum(axis=0)
    L = float(np.sum(np.arange(1, k + 1) * R))
    esperado = n * k * (k + 1) ** 2 / 4.0
    varianza = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    z = (L - esperado) / np.sqrt(varianza)
    return L, float(z), float(stats.norm.sf(z))


# =============================================================================================
# 2) LECTURA Y VERIFICACIÓN CRUZADA
# =============================================================================================
def cargar_corridas():
    filas, problemas = [], []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp = carpeta / "meta_regla.json"
        if not mp.exists():
            problemas.append(f"{carpeta.name}: sin meta_regla.json")
            continue
        meta = json.loads(mp.read_text())
        # --- verificación cruzada contra el nombre de la carpeta ---
        esperado = f"{meta['rule_id']}_s{meta['seed']}_f802_{meta['nivel']}"
        if meta.get("tarea") != "FASE8_F802_pico_local_manipulado":
            problemas.append(f"{carpeta.name}: tarea declarada {meta.get('tarea')!r}")
            continue
        if carpeta.name != esperado:
            problemas.append(f"{carpeta.name}: el meta declara {esperado}")
            continue
        if Path(meta.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta apunta a otra carpeta")
            continue
        completa = (carpeta / "cosmog_00500").exists()
        if not completa:
            # NO se descarta en silencio: se lee el último dump que haya y se marca la fila. Phantom
            # puede abortar por su propio guardián de conservación (ver §"lo que no se pudo medir");
            # ocultar esa corrida sería fabricar un resultado más limpio del que hay.
            problemas.append(f"{carpeta.name}: sin cosmog_00500 (corrida incompleta, se marca)")

        r = analizar_carpeta(carpeta)
        filas.append(dict(
            completa=completa, dump_final=r.get("n_dump_final"),
            rule_id=meta["rule_id"], seed=int(meta["seed"]), nivel=meta["nivel"],
            a_pico=float(meta["a_pico"]), clase=meta.get("clase"),
            grafo_sha256=meta["grafo_sha256"], grafo_n_aristas=meta["grafo_n_aristas"],
            masa_total_ic=meta["masa_total_ic"], N=meta["N"],
            frac_masa=r["fraccion_masa_en_sumideros"],
            masa_sumideros=r["masa_sumideros_final"], masa_total_final=r["masa_total_final"],
            n_sumideros=r.get("n_sumideros"), t_primer_sumidero=r.get("t_primer_sumidero"),
            kappa_v=r.get("kappa_v_agregado"), n_gas_inicial=r.get("n_gas_inicial"),
            carpeta=carpeta.name,
        ))
    return pd.DataFrame(filas), problemas


def verificar_bloques(D):
    """Las cinco corridas de una IC tienen que compartir grafo (sello), nº de aristas, masa y N."""
    lineas = []
    for (rid, seed), g in D.groupby(["rule_id", "seed"]):
        lineas.append(dict(
            rule_id=rid, seed=seed, n_niveles=len(g),
            niveles=",".join(sorted(g.nivel)),
            sellos_distintos=g.grafo_sha256.nunique(),
            aristas_distintas=g.grafo_n_aristas.nunique(),
            masa_total_ic=g.masa_total_ic.nunique(), N=g.N.nunique(),
        ))
    V = pd.DataFrame(lineas)
    assert (V.sellos_distintos == 1).all(), "hay IC cuyos niveles NO comparten el mismo grafo"
    assert (V.aristas_distintas == 1).all(), "hay IC cuyos niveles NO comparten el nº de aristas"
    assert (V.masa_total_ic == 1).all() and (V.N == 1).all(), "cambió la masa total o N entre niveles"
    return V


# =============================================================================================
# 3) ANÁLISIS
# =============================================================================================
def main():
    D_ic = pd.read_csv(RUTA_IC)
    D_ph, problemas = cargar_corridas()
    print(f"[f802] corridas leídas y verificadas: {len(D_ph)}   problemas: {len(problemas)}")
    for p in problemas:
        print("   ! " + p)
    V = verificar_bloques(D_ph)
    print(f"[f802] bloques (IC base): {len(V)}; los 5 niveles de cada uno comparten grafo, "
          f"aristas, masa y N: OK")

    D_todo = D_ic.merge(D_ph, on=["rule_id", "seed", "nivel"], how="inner",
                        suffixes=("", "_ph"))                # unión por (rule_id, seed, nivel)
    assert len(D_todo) == len(D_ph), "la unión IC↔Phantom perdió o duplicó filas"
    D_todo["particulas_en_sumideros"] = D_todo.frac_masa * D_todo.N
    D_todo = D_todo.sort_values(["rule_id", "a_pico"]).reset_index(drop=True)
    D_todo.to_csv(RUTA_CRUDO, index=False)                   # el CSV crudo lleva TODO, completo o no
    print(f"[f802] CSV crudo: {RUTA_CRUDO} ({len(D_todo)} filas, "
          f"{int(D_todo.completa.sum())} completas a tmax=0.500)")

    # --- la estadística de bloques exige bloques COMPLETOS: se listan los que se caen y por qué ---
    completos = D_todo.groupby("rule_id").completa.transform("all")
    incompletos = sorted(set(D_todo.loc[~completos, "rule_id"]))
    if incompletos:
        print(f"[f802] IC excluidas de la estadística de bloques por tener algún nivel incompleto: "
              f"{incompletos}  (quedan en el CSV crudo, con su dump final marcado)")
    D = D_todo[completos].copy()

    for rid in incompletos:                                  # se muestran igual, no se esconden
        g = D_todo[D_todo.rule_id == rid].sort_values("a_pico")
        print("     " + rid + ": " + "  ".join(
            f"{r.nivel}={r.frac_masa:.4f}{'' if r.completa else f'*({r.dump_final})'}"
            for r in g.itertuples()))

    # ---- reproducibilidad del nivel de identidad (archivo byte a byte igual al original) ----
    base = D_todo[D_todo.nivel == NIVEL_BASE]
    dif_hist = (base.frac_masa - base.frac_masa_historica).abs()
    print(f"[f802] L1 (identidad) vs. la corrida histórica de Fase V-B: "
          f"máx |Δ| = {dif_hist.max():.6f}  ({(dif_hist < 1e-9).sum()}/{len(base)} idénticas)")

    # ---- matrices bloque × nivel ----
    piv = {c: D.pivot(index="rule_id", columns="nivel", values=c)[NIVELES] for c in
           ["frac_masa", "pico_logrado", "t_primer_sumidero", "kappa_v", "n_sumideros",
            "fof030_logrado", "cv_logrado"]}
    M = piv["frac_masa"].values
    P = piv["pico_logrado"].values
    n_ic = M.shape[0]

    filas_est = []

    def anotar(prueba, variable, **kw):
        filas_est.append(dict(prueba=prueba, variable=variable, **kw))

    # ---- 1) el pico se movió monótonamente con el nivel? (control de la intervención) ----
    L, z, p = page_l(P)
    mono_pico = int(sum(np.all(np.diff(P[i]) > 0) for i in range(n_ic)))
    anotar("Page L (tendencia ordenada)", "pico logrado", L=L, z=z, p=p,
           detalle=f"{mono_pico}/{n_ic} IC con pico estrictamente creciente en los 5 niveles")
    print(f"\n[control de la intervención] pico logrado: Page L z={z:.2f} p={p:.2e}; "
          f"estrictamente creciente en {mono_pico}/{n_ic} IC")

    # ---- 2) OBSERVABLE PRINCIPAL: fracción de masa en sumideros ----
    L, z, p = page_l(M)
    mono_masa = int(sum(np.all(np.diff(M[i]) > 0) for i in range(n_ic)))
    anotar("Page L (tendencia ordenada)", "frac_masa", L=L, z=z, p=p,
           detalle=f"{mono_masa}/{n_ic} IC con masa estrictamente creciente en los 5 niveles")
    fr = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    anotar("Friedman (cualquier diferencia)", "frac_masa", L=float(fr.statistic), z=np.nan,
           p=float(fr.pvalue), detalle=f"{n_ic} bloques × {M.shape[1]} niveles")
    print(f"[PRINCIPAL] frac_masa: Page L z={z:.2f} p={p:.2e}; estrictamente creciente en "
          f"{mono_masa}/{n_ic} IC; Friedman χ²={fr.statistic:.1f} p={fr.pvalue:.2e}")

    # ---- 3) contrastes pareados contra el nivel de identidad, y extremo vs extremo ----
    j_base = NIVELES.index(NIVEL_BASE)
    print("\n[contrastes pareados]  (1 partícula = 0.0005; piso práctico ~5 partículas = 0.0025)")
    for nombre, ja, jb in ([(f"{n} - {NIVEL_BASE}", NIVELES.index(n), j_base)
                            for n in NIVELES if n != NIVEL_BASE] + [("L4 - L0", 4, 0)]):
        d = M[:, ja] - M[:, jb]
        signos = int((d > 0).sum())
        w = stats.wilcoxon(d) if np.any(d != 0) else None
        binom = stats.binomtest(signos, n_ic, 0.5).pvalue
        anotar("contraste pareado", f"frac_masa {nombre}", delta_mediano=float(np.median(d)),
               delta_medio=float(d.mean()), particulas=float(d.mean() / GRANO),
               signos=f"{signos}/{n_ic}", p_wilcoxon=(float(w.pvalue) if w else np.nan),
               p_signos=float(binom), supera_piso=bool(abs(np.median(d)) > PISO_PAREADO))
        print(f"  {nombre:>9}: Δmediano={np.median(d):+.5f} ({np.median(d)/GRANO:+.1f} part.)  "
              f"Δmedio={d.mean():+.5f} ({d.mean()/GRANO:+.1f} part.)  signos {signos}/{n_ic}  "
              f"Wilcoxon p={w.pvalue if w else float('nan'):.2e}  "
              f"{'SUPERA el piso de 5 part.' if abs(np.median(d)) > PISO_PAREADO else 'por debajo del piso'}")

    # ---- 4) Spearman dentro de cada IC (12 coeficientes) + agrupado con efectos fijos de IC ----
    por_ic = []
    for i, rid in enumerate(piv["frac_masa"].index):
        rho, pr = stats.spearmanr(P[i], M[i])
        pend = np.polyfit(np.log10(P[i]), M[i], 1)[0]
        por_ic.append(dict(rule_id=rid,
                           pico_L0=P[i, 0], pico_base=P[i, j_base], pico_L4=P[i, -1],
                           pico_razon_extremos=P[i, -1] / P[i, 0],
                           masa_L0=M[i, 0], masa_base=M[i, j_base], masa_L4=M[i, -1],
                           delta_L4_menos_L0=M[i, -1] - M[i, 0],
                           particulas_L4_menos_L0=(M[i, -1] - M[i, 0]) / GRANO,
                           spearman_pico_masa=rho, p_spearman=pr,
                           pendiente_masa_por_dex_pico=pend,
                           monotona=bool(np.all(np.diff(M[i]) > 0)),
                           t1_L0=piv["t_primer_sumidero"].values[i, 0],
                           t1_L4=piv["t_primer_sumidero"].values[i, -1],
                           kappa_v_L0=piv["kappa_v"].values[i, 0],
                           kappa_v_L4=piv["kappa_v"].values[i, -1]))
    PI = pd.DataFrame(por_ic)
    PI.to_csv(RUTA_POR_IC, index=False)
    n_pos = int((PI.spearman_pico_masa > 0).sum())
    print(f"\n[dentro de cada IC] Spearman(pico logrado, frac_masa): mediana={PI.spearman_pico_masa.median():+.3f}  "
          f"{n_pos}/{n_ic} positivos  (rango {PI.spearman_pico_masa.min():+.2f} a {PI.spearman_pico_masa.max():+.2f})")
    anotar("Spearman dentro de IC", "pico logrado vs frac_masa",
           delta_mediano=float(PI.spearman_pico_masa.median()), signos=f"{n_pos}/{n_ic}",
           p_signos=float(stats.binomtest(n_pos, n_ic, 0.5).pvalue),
           detalle=f"rango {PI.spearman_pico_masa.min():+.2f} a {PI.spearman_pico_masa.max():+.2f}")

    # agrupado, centrando por IC (efectos fijos): saca todo lo que distingue una IC de otra
    for var, etiqueta in [("pico_logrado", "pico local (p90/mediana)"),
                          ("fof030_logrado", "geometría GLOBAL FoF b=0.30 (eje-1)"),
                          ("cv_logrado", "CV de densidad local")]:
        x = D.groupby("rule_id")[var].transform(lambda s: s - s.mean())
        y = D.groupby("rule_id")["frac_masa"].transform(lambda s: s - s.mean())
        rho, pr = stats.spearmanr(x, y)
        r, pp = stats.pearsonr(x, y)
        anotar("agrupado con efectos fijos de IC", etiqueta, delta_mediano=float(rho), p=float(pr),
               detalle=f"Pearson r={r:+.3f} (p={pp:.1e}), n={len(D)}")
        print(f"  agrupado (centrado por IC): {etiqueta:<38} ρ={rho:+.3f} (p={pr:.1e})  r={r:+.3f}")

    # ---- 5) secundarios ----
    print("\n[secundarios]")
    for var, etiq in [("t_primer_sumidero", "tiempo al primer sumidero"),
                      ("kappa_v", "κ_V agregado"), ("n_sumideros", "nº de sumideros"),
                      ("fof030_logrado", "geometría global FoF b=0.30 (eje-1)")]:
        Mv = piv[var].values.astype(float)
        L2, z2, p2 = page_l(Mv)
        d = Mv[:, -1] - Mv[:, j_base]
        anotar("Page L (tendencia ordenada)", var, L=L2, z=z2, p=p2,
               delta_mediano=float(np.median(d)), signos=f"{int((d>0).sum())}/{n_ic}")
        print(f"  {etiq:<34} Page z={z2:+.2f} p={p2:.2e}   Δmediano(L4−{NIVEL_BASE})={np.median(d):+.5g}")

    # ---- 6) cuánto se movió cada eje (control de la intervención) ----
    razon_pico = D.groupby("rule_id").apply(
        lambda g: g.pico_logrado.max() / g.pico_logrado.min(), include_groups=False)
    razon_fof = D.groupby("rule_id").apply(
        lambda g: g.fof030_logrado.max() / g.fof030_logrado.min(), include_groups=False)
    print(f"\n[qué se movió] pico local: ×{razon_pico.min():.2f} a ×{razon_pico.max():.2f} dentro de una IC; "
          f"geometría global FoF: ×{razon_fof.min():.4f} a ×{razon_fof.max():.4f}")
    anotar("recorrido dentro de IC", "pico local / geometría global",
           detalle=f"pico ×{razon_pico.min():.2f}-×{razon_pico.max():.2f} vs "
                   f"FoF global ×{razon_fof.min():.4f}-×{razon_fof.max():.4f}")

    pd.DataFrame(filas_est).to_csv(RUTA_ESTAD, index=False)
    print(f"\n[f802] estadística -> {RUTA_ESTAD}")
    figura(D, piv, PI)
    return D, PI


# =============================================================================================
# 4) FIGURA
# =============================================================================================
def figura(D, piv, PI):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ruleids = list(piv["frac_masa"].index)
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(ruleids)))

    a = ax[0, 0]
    for i, rid in enumerate(ruleids):
        a.plot(range(5), piv["pico_logrado"].values[i], "o-", color=colores[i], lw=1.2, ms=4)
    a.set_yscale("log")
    a.set_xticks(range(5))
    a.set_xticklabels([f"{n}\na={v:+.2f}" for n, v in zip(NIVELES, [-0.35, 0, .2, .35, .5])], fontsize=8)
    a.set_ylabel("pico local logrado  (p90/mediana de la densidad a 8 vecinos)")
    a.set_title("A · lo que se manipuló: el pico local del gas inicial\n(cada línea es una condición inicial)",
                fontsize=10)
    a.grid(alpha=.3)

    a = ax[0, 1]
    for i, rid in enumerate(ruleids):
        a.plot(piv["pico_logrado"].values[i], piv["frac_masa"].values[i] * 2000, "o-",
               color=colores[i], lw=1.2, ms=4)
    a.set_xscale("log")
    a.set_xlabel("pico local logrado en la condición inicial")
    a.set_ylabel("masa en sumideros  (en partículas de 2000)")
    a.set_title("B · el observable principal contra el pico\n(criterio de densidad de Phantom, rho_crit=1000)",
                fontsize=10)
    a.grid(alpha=.3)

    a = ax[1, 0]
    Mb = piv["frac_masa"].values
    j = NIVELES.index(NIVEL_BASE)
    d = (Mb - Mb[:, [j]]) / GRANO
    a.axhspan(-5, 5, color="0.85", zorder=0)
    a.axhline(0, color="k", lw=.8)
    for i in range(len(ruleids)):
        a.plot(range(5), d[i], "o-", color=colores[i], lw=1, ms=3, alpha=.7)
    a.plot(range(5), np.median(d, axis=0), "s-", color="crimson", lw=2.5, ms=7,
           label="mediana de las 12")
    a.set_xticks(range(5))
    a.set_xticklabels(NIVELES)
    a.set_ylabel(f"masa − masa del nivel identidad  (en partículas)")
    a.set_title("C · efecto pareado contra el nivel identidad (a=0)\nbanda gris = piso práctico de ±5 partículas (F8-01)",
                fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    a = ax[1, 1]
    t = piv["t_primer_sumidero"].values.astype(float)
    for i in range(len(ruleids)):
        a.plot(piv["pico_logrado"].values[i], t[i], "o-", color=colores[i], lw=1.2, ms=4)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("pico local logrado")
    a.set_ylabel("tiempo al primer sumidero")
    a.set_title("D · secundario: cuánto tarda en encenderse el primer sumidero", fontsize=10)
    a.grid(alpha=.3)

    fig.suptitle("F8-02 · el pico local de densidad inicial, manipulado a propósito "
                 "(12 condiciones iniciales × 5 niveles, mismo grafo, misma masa, misma caja)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RUTA_PNG, dpi=140)
    print(f"[f802] figura -> {RUTA_PNG}")


if __name__ == "__main__":
    main()
