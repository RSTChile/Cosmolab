#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase8_f806_analizar.py — FASE VIII F8-06: ¿el +13.8 % de F7-03 se mantiene, crece o se diluye
                               al pasar de N=2000 a N=4000?
=======================================================================================================

QUÉ HACE (a nivel módulo)
-------------------------
Lee los dumps de Phantom de la batería `bateria_fase8_f806_n4000` (brazos `solap` y `disj`, N=4000),
los verifica contra el `meta_regla.json` de cada carpeta, y contesta tres preguntas, en este orden:

  1. **¿Cuánto vale el efecto a N=4000?** Diseño pareado por grafo (cada grafo es su propio control:
     mismo N, mismas aristas, mismos grados nodo por nodo, mismo nº de triángulos; lo único distinto es
     dónde están puestos). Se reporta el Δ en fracción de masa, **en partículas** y **en unidades de σ**.
  2. **¿El mismo grafo que ganaba a N=2000 sigue ganando a N=4000?** Unión por `(rule_id, seed)` con
     `cs090_fase7_f703_phantom_crudo.csv` — nunca por `rule_id` solo (bug de colisión de nombres,
     FASE6_O3B §2.1) — y comparación par por par: concordancia de signo y Spearman de los Δ.
  3. **¿El efecto sigue creciendo con T\\*?** F7-03 midió ρ = +0.818 entre el Δ y el nº de triángulos
     disponibles para repartir. A N=4000 el T\\* alcanzable cambia, y eso es un confound que hay que
     mirar de frente, no esconder.

LAS TRES VARAS CONTRA LAS QUE SE REPORTA EL EFECTO
--------------------------------------------------
  * **1 partícula** = 1/4000 = **0.00025** de fracción de masa (a N=2000 era 0.0005). Es la
    CUANTIZACIÓN del observable, no su ruido.
  * **σ de F8-04, extrapolada** = **0.00265** al nº de sumideros que se espera a N=4000 (~29), por el
    ajuste σ ∝ (nº de sumideros)^1.24 (R = 0.92, 13 puntos, un solo grafo).
  * **σ medida acá**, si `cs090_fase8_f806_sigma.csv` existe (réplicas de redondeo a N=4000, mismo
    método de perturbación de `lado` en ULPs que usó F8-04). Es la vara buena; la extrapolada es el
    respaldo.

VERIFICACIÓN CRUZADA, ANTES DE QUE NINGUNA CORRIDA ENTRE EN LA ESTADÍSTICA
--------------------------------------------------------------------------
  - `meta_regla.json` declara la tarea de F7-03 (el generador es literalmente ese) y `meta_f806.json`
    declara la tarea F8-06 con `N=4000`;
  - brazo y `(rule_id, seed)` del meta coinciden con el nombre de la carpeta;
  - la carpeta declarada DENTRO del meta es la carpeta donde está el meta;
  - `grados_identicos_al_original = true`;
  - los dos brazos de un grafo tienen el mismo nº de aristas, la misma `seed_layout`, el mismo nº de
    triángulos y — específico de esta fase — **el mismo layout y el mismo θ**;
  - el sello sha256 del grafo guardado se recalcula al leerlo (lo hace `cargar_grafo`);
  - la corrida arrancó con **4000** partículas de gas (chequeo anti-IC-truncado);
  - existe el dump final `cosmog_00500` (a N=8000 esto NO se cumpliría: 0 de 14; ver F8-04).

Reusa `cs090_fase5b_analizar.analizar_carpeta` TAL CUAL (sólo import): la misma extracción de métricas
de toda la línea, con la **fracción de masa en sumideros** como observable — declarado de antemano,
porque F8-05 mostró que un FoF laxo puede invertir el signo sobre las mismísimas corridas.

Corre con `./venv/bin/python` (necesita `sarracen` para leer los volcados). No declara cierre.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

from cs090_fase5b_analizar import analizar_carpeta       # sólo import, script congelado
import cs090_fase8_f800_grafos as G8                     # sólo import, persistencia de grafos

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f806_n4000")
RUTA_CRUDO = f"{HERE}/cs090_fase8_f806_phantom_crudo.csv"
RUTA_POR_GRAFO = f"{HERE}/cs090_fase8_f806_por_grafo.csv"
RUTA_ESTAD = f"{HERE}/cs090_fase8_f806_estadistica.csv"
RUTA_PAREADO = f"{HERE}/cs090_fase8_f806_pareado_2000_vs_4000.csv"
RUTA_PNG = f"{HERE}/cs090_fase8_f806_n4000.png"
RUTA_SIGMA = f"{HERE}/cs090_fase8_f806_sigma.csv"
RUTA_F703 = f"{HERE}/cs090_fase7_f703_phantom_crudo.csv"

BRAZOS = ("solap", "disj")
N_PART = 4000
GRANO_PARTICULA = 1.0 / N_PART            # 1 partícula de 4000 = 0.00025 de fracción de masa
GRANO_PARTICULA_2000 = 1.0 / 2000         # la vara con la que se reportó F7-03
SIGMA_F804_EXTRAP = 0.00265               # σ ∝ n_sumideros^1.24 evaluada en 29 sumideros (F8-04 §2.1)


def _f(d, k, default=float("nan")):
    v = d.get(k)
    if v in (None, "", "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# =============================================================================================
# 1) VERIFICACIÓN CRUZADA + lectura de los dumps
# =============================================================================================
def recolectar():
    metas, problemas = {}, []
    for carpeta in sorted(c for c in BASE.iterdir() if c.is_dir()):
        mp, mp6 = carpeta / "meta_regla.json", carpeta / "meta_f806.json"
        if not mp.exists() or not mp6.exists():
            problemas.append(f"{carpeta.name}: falta meta_regla.json o meta_f806.json")
            continue
        m, m6 = json.loads(mp.read_text()), json.loads(mp6.read_text())
        if m.get("tarea") != "FASE7_F703_organizacion_triangulos":
            problemas.append(f"{carpeta.name}: tarea del generador = {m.get('tarea')}")
            continue
        if m6.get("tarea") != "FASE8_F806_f703_a_N4000" or int(m6.get("N", 0)) != N_PART:
            problemas.append(f"{carpeta.name}: meta_f806 = {m6.get('tarea')} N={m6.get('N')}")
            continue
        if Path(m.get("carpeta", "")).name != carpeta.name:
            problemas.append(f"{carpeta.name}: el meta declara carpeta={m.get('carpeta')}")
            continue
        if carpeta.name != f"{m['rule_id']}_s{m['seed']}_f703_{m['brazo']}":
            problemas.append(f"{carpeta.name}: el meta corresponde a otra carpeta")
            continue
        if m["brazo"] != m6["brazo"]:
            problemas.append(f"{carpeta.name}: brazo discrepa entre metas")
            continue
        if not m.get("grados_identicos_al_original", False):
            problemas.append(f"{carpeta.name}: el meta NO declara grados idénticos al original")
            continue
        # sello del grafo guardado: se recalcula al cargar (cargar_grafo levanta si no coincide)
        rg = carpeta / ("grafo_final" + G8.SUFIJO)
        if rg.exists():
            adj_g, n_g, meta_g = G8.cargar_grafo(rg)
            if meta_g["sha256"] != m.get("grafo_sha256"):
                problemas.append(f"{carpeta.name}: sha256 del grafo != el anotado en meta_regla.json")
                continue
            m6["grafo_n_aristas_verificado"] = meta_g["E"]
        else:
            problemas.append(f"{carpeta.name}: sin grafo guardado")
            continue
        if not (carpeta / "cosmog_00500").exists():
            problemas.append(f"{carpeta.name}: sin dump final cosmog_00500 (¿todavía corriendo?)")
            continue
        metas[carpeta.name] = (carpeta, m, m6)
    return metas, problemas


def main():
    metas, problemas = recolectar()
    grupos = defaultdict(dict)
    for nombre, (carpeta, m, m6) in metas.items():
        grupos[(m["rule_id"], m["seed"])][m["brazo"]] = (carpeta, m, m6)

    filas, completos = [], []
    for (rid, seed), br in sorted(grupos.items()):
        faltan = [b for b in BRAZOS if b not in br]
        if faltan:
            problemas.append(f"{rid} s{seed}: faltan brazos {faltan} — no entra")
            continue
        aristas = {br[b][1]["n_aristas_grafo_final"] for b in br}
        layouts = {br[b][1]["seed_layout"] for b in br}
        tris = {br[b][1]["n_triangulos"] for b in br}
        motores = {(br[b][2]["layout"], str(br[b][2]["theta"])) for b in br}
        if len(aristas) != 1 or len(layouts) != 1:
            problemas.append(f"{rid} s{seed}: aristas={aristas} seed_layout={layouts} no uniformes")
            continue
        if len(motores) != 1:
            problemas.append(f"{rid} s{seed}: LAYOUTS MEZCLADOS entre brazos: {motores} — NO entra")
            continue
        if len(tris) != 1:
            problemas.append(f"{rid} s{seed}: triángulos NO idénticos {sorted(tris)} "
                             f"(dif {max(tris)-min(tris)}) — entra y se reporta")
        completos.append((rid, seed))
        for b in BRAZOS:
            carpeta, m, m6 = br[b]
            a = analizar_carpeta(carpeta)
            if a.get("n_gas_inicial") not in (None, N_PART):
                problemas.append(f"{carpeta.name}: n_gas_inicial={a['n_gas_inicial']} (¿IC truncado?)")
            filas.append(dict(
                rule_id=rid, seed=seed, lote=m.get("lote"), brazo=b, N=N_PART,
                layout=m6["layout"], theta=m6["theta"], seed_layout=m6["seed_layout"],
                n_aristas=m["n_aristas_grafo_final"], n_triangulos=int(m["n_triangulos"]),
                T_objetivo=m.get("T_objetivo"), dif_max_triangulos=m.get("dif_max_triangulos"),
                clustering_local=float(m["clustering_local"]),
                transitividad=float(m["transitividad"]), gigante=int(m["gigante"]),
                frac_aristas_multi_tri=_f(m, "frac_aristas_multi_tri"),
                gini_tri_nodo=_f(m, "gini_tri_nodo"),
                pendiente_corr=_f(m, "pendiente_corregida"),
                frac_masa=a["fraccion_masa_en_sumideros"], kappa_v=a["kappa_v_agregado"],
                n_sumideros=a["n_sumideros"], t_primer_sumidero=a["t_primer_sumidero"],
                masa_acretada=a["masa_acretada_total"], dump_final=a.get("n_dump_final"),
                t_layout_ic_s=m6.get("t_layout_ic_s"), grafo_sha256=m6.get("grafo_sha256"),
                carpeta=carpeta.name))

    print(f"[f806] {len(completos)} grafos con los 2 brazos; {len(filas)} corridas; "
          f"{len(problemas)} avisos")
    for pr in problemas:
        print(f"   !! {pr}")
    if not filas:
        print("[f806] nada que analizar todavía")
        return

    with open(RUTA_CRUDO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"[csv] {Path(RUTA_CRUDO).name} ({len(filas)} filas)")
    if not completos:
        return

    # ---------------- σ: la medida acá si existe, si no la extrapolada de F8-04 ----------------
    sigma_med, sigma_nota = None, ""
    if Path(RUTA_SIGMA).exists():
        vals = [float(r["frac_masa"]) for r in csv.DictReader(open(RUTA_SIGMA))
                if r.get("frac_masa") not in (None, "", "nan")]
        if len(vals) >= 3:
            sigma_med = float(np.std(vals, ddof=1))
            sigma_nota = (f"σ MEDIDA a N=4000 con {len(vals)} réplicas de redondeo "
                          f"(media {np.mean(vals):.5f}, rango {max(vals)-min(vals):.5f})")
    sigma_usada = sigma_med if sigma_med else SIGMA_F804_EXTRAP
    etiqueta_sigma = "medida" if sigma_med else "extrapolada de F8-04"

    # ---------------- por grafo ----------------
    por_grafo = []
    for (rid, seed) in completos:
        sub = {f["brazo"]: f for f in filas if f["rule_id"] == rid and f["seed"] == seed}
        d = sub["solap"]["frac_masa"] - sub["disj"]["frac_masa"]
        por_grafo.append(dict(
            rule_id=rid, seed=seed, lote=sub["solap"]["lote"], n_aristas=sub["solap"]["n_aristas"],
            T_estrella=sub["solap"]["n_triangulos"],
            dif_triangulos=abs(sub["solap"]["n_triangulos"] - sub["disj"]["n_triangulos"]),
            masa_solap=sub["solap"]["frac_masa"], masa_disj=sub["disj"]["frac_masa"],
            d_solap_disj=float(d), d_particulas=float(d / GRANO_PARTICULA),
            d_sigmas=float(d / sigma_usada),
            pct=float(100.0 * d / sub["disj"]["frac_masa"]),
            sumideros_solap=sub["solap"]["n_sumideros"], sumideros_disj=sub["disj"]["n_sumideros"],
            kappa_solap=sub["solap"]["kappa_v"], kappa_disj=sub["disj"]["kappa_v"],
            t1_solap=sub["solap"]["t_primer_sumidero"], t1_disj=sub["disj"]["t_primer_sumidero"],
            multi_solap=sub["solap"]["frac_aristas_multi_tri"],
            multi_disj=sub["disj"]["frac_aristas_multi_tri"],
            gigante_solap=sub["solap"]["gigante"], gigante_disj=sub["disj"]["gigante"]))
    with open(RUTA_POR_GRAFO, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(por_grafo[0].keys()))
        w.writeheader()
        w.writerows(por_grafo)
    print(f"[csv] {Path(RUTA_POR_GRAFO).name} ({len(por_grafo)} grafos)")

    d = np.array([g["d_solap_disj"] for g in por_grafo])
    n = len(d)
    print(f"\n--- fracción de masa, N=4000, un grafo por fila (σ {etiqueta_sigma} = {sigma_usada:.5f}) ---")
    print(f"   {'regla':<16}{'T*':>6}{'solap':>9}{'disj':>9}{'Δ':>10}{'part.':>8}{'σ':>7}{'%':>8}"
          f"{'sumid.':>9}")
    for g in por_grafo:
        print(f"   {g['rule_id'].replace('A2-B0-C2-',''):<16}{g['T_estrella']:>6}"
              f"{g['masa_solap']:>9.4f}{g['masa_disj']:>9.4f}{g['d_solap_disj']:>+10.5f}"
              f"{g['d_particulas']:>+8.1f}{g['d_sigmas']:>+7.1f}{g['pct']:>+8.1f}"
              f"{g['sumideros_solap']:>5}/{g['sumideros_disj']:<3}")
    print(f"   {'MEDIA':<16}{'':>6}{np.mean([g['masa_solap'] for g in por_grafo]):>9.4f}"
          f"{np.mean([g['masa_disj'] for g in por_grafo]):>9.4f}{d.mean():>+10.5f}"
          f"{d.mean()/GRANO_PARTICULA:>+8.1f}{d.mean()/sigma_usada:>+7.1f}"
          f"{np.mean([g['pct'] for g in por_grafo]):>+8.1f}")

    # ---------------- estadística ----------------
    resumen = []

    def anota(prueba, est, p, detalle):
        resumen.append(dict(prueba=prueba, estadistico=float(est), p=float(p), n=n, detalle=detalle))

    sg = int((d > 0).sum())
    w = float(stats.wilcoxon(d, alternative="two-sided").pvalue) if n >= 5 else float("nan")
    pb = float(stats.binomtest(sg, n, 0.5, alternative="two-sided").pvalue)
    anota("solap − disj (N=4000), fracción de masa", d.mean(), w,
          f"Δ={d.mean():+.5f} ({d.mean()/GRANO_PARTICULA:+.1f} part. de 4000, "
          f"{d.mean()/sigma_usada:+.2f} σ), signos {sg}/{n}, binomial p={pb:.4g}, "
          f"% medio {np.mean([g['pct'] for g in por_grafo]):+.1f}")
    t = stats.ttest_rel([g["masa_solap"] for g in por_grafo], [g["masa_disj"] for g in por_grafo])
    anota("t pareado solap vs disj", float(t.statistic), float(t.pvalue),
          f"sd(Δ)={d.std(ddof=1):.5f}, ee={d.std(ddof=1)/np.sqrt(n):.5f}")

    for campo, etiq in (("sumideros", "nº de sumideros"), ("kappa", "κ_V agregado"),
                        ("t1", "t del primer sumidero")):
        a = np.array([g[f"{campo}_solap"] for g in por_grafo], dtype=float)
        b = np.array([g[f"{campo}_disj"] for g in por_grafo], dtype=float)
        dd = a - b
        try:
            pw = float(stats.wilcoxon(dd, alternative="two-sided").pvalue)
        except Exception:
            pw = float("nan")
        anota(f"{etiq} (solap − disj)", float(dd.mean()), pw,
              f"signos {(dd>0).sum()}/{n}, medias {a.mean():.4f} vs {b.mean():.4f}")

    T = np.array([g["T_estrella"] for g in por_grafo], dtype=float)
    if np.std(T) > 0:
        rho, p = stats.spearmanr(T, d)
        anota("Spearman Δ vs T* (¿el efecto crece con los triángulos disponibles?)", rho, p,
              f"T* de {int(T.min())} a {int(T.max())} (F7-03 a N=2000: ρ=+0.818, p=0.0011)")

    # ---------------- N=2000 contra N=4000, par por par ----------------
    pareado = []
    if Path(RUTA_F703).exists():
        prev = defaultdict(dict)
        for r in csv.DictReader(open(RUTA_F703)):
            prev[(r["rule_id"], int(r["seed"]))][r["brazo"]] = r
        for g in por_grafo:
            k = (g["rule_id"], g["seed"])
            if "solap" not in prev.get(k, {}) or "disj" not in prev[k]:
                continue
            d2 = float(prev[k]["solap"]["frac_masa"]) - float(prev[k]["disj"]["frac_masa"])
            pareado.append(dict(
                rule_id=g["rule_id"], seed=g["seed"], lote=g["lote"],
                T_2000=int(prev[k]["solap"]["n_triangulos"]), T_4000=g["T_estrella"],
                masa_solap_2000=float(prev[k]["solap"]["frac_masa"]),
                masa_disj_2000=float(prev[k]["disj"]["frac_masa"]),
                masa_solap_4000=g["masa_solap"], masa_disj_4000=g["masa_disj"],
                d_2000=d2, d_2000_particulas=d2 / GRANO_PARTICULA_2000,
                d_4000=g["d_solap_disj"], d_4000_particulas=g["d_particulas"],
                pct_2000=100.0 * d2 / float(prev[k]["disj"]["frac_masa"]), pct_4000=g["pct"],
                sumideros_2000=int(float(prev[k]["solap"]["n_sumideros"])),
                sumideros_4000=g["sumideros_solap"],
                mismo_signo=bool((d2 > 0) == (g["d_solap_disj"] > 0))))
        if pareado:
            with open(RUTA_PAREADO, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(pareado[0].keys()))
                wr.writeheader()
                wr.writerows(pareado)
            a2 = np.array([p["d_2000"] for p in pareado])
            a4 = np.array([p["d_4000"] for p in pareado])
            ig = sum(p["mismo_signo"] for p in pareado)
            rho, pv = stats.spearmanr(a2, a4)
            rp, pp = stats.pearsonr(a2, a4)
            anota("concordancia de signo del Δ entre N=2000 y N=4000", ig, float("nan"),
                  f"{ig}/{len(pareado)} grafos con el mismo signo; "
                  f"positivos a N=2000: {(a2>0).sum()}/{len(a2)}, a N=4000: {(a4>0).sum()}/{len(a4)}")
            anota("Spearman Δ(N=2000) vs Δ(N=4000)", rho, pv,
                  f"Pearson={rp:+.3f} (p={pp:.3g}); media Δ 2000={a2.mean():+.5f} "
                  f"({a2.mean()/GRANO_PARTICULA_2000:+.1f} part.), 4000={a4.mean():+.5f} "
                  f"({a4.mean()/GRANO_PARTICULA:+.1f} part.)")
            print(f"[csv] {Path(RUTA_PAREADO).name} ({len(pareado)} grafos en ambas resoluciones)")

    if sigma_nota:
        anota("σ del instrumento a N=4000", sigma_usada, float("nan"), sigma_nota)
    else:
        anota("σ del instrumento a N=4000 (extrapolada)", sigma_usada, float("nan"),
              "σ ∝ n_sumideros^1.24 de F8-04 evaluada en 29 sumideros; no se midió acá")

    with open(RUTA_ESTAD, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        wr.writeheader()
        wr.writerows(resumen)
    print(f"\n[csv] {Path(RUTA_ESTAD).name}")
    for r in resumen:
        print(f"   {r['prueba']:<58} est={r['estadistico']:<+11.5f} p={r['p']:<10.4g} ({r['detalle']})")

    graficar(por_grafo, pareado, sigma_usada, etiqueta_sigma)
    return filas, por_grafo, resumen, pareado


# =============================================================================================
def graficar(por_grafo, pareado, sigma, etiqueta_sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2))
    col = plt.cm.viridis(np.linspace(0, 0.9, len(por_grafo)))

    ax = axs[0]
    for k, g in enumerate(por_grafo):
        ax.plot([0, 1], [g["masa_solap"], g["masa_disj"]], "o-", color=col[k], lw=1.2, ms=5,
                alpha=0.85, label=f"{g['rule_id'].replace('A2-B0-C2-','')} (T*={g['T_estrella']})")
    ax.plot([0, 1], [np.mean([g["masa_solap"] for g in por_grafo]),
                     np.mean([g["masa_disj"] for g in por_grafo])],
            "s-", color="crimson", lw=2.8, ms=9, label="media")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["solap", "disj"])
    ax.set_ylabel("fracción de masa en sumideros")
    ax.set_title("N=4000 · mismos grados nodo por nodo\ny mismo nº de triángulos", fontsize=10)
    ax.legend(fontsize=6.4, ncol=2); ax.grid(alpha=0.25)

    ax = axs[1]
    d = np.array([g["d_solap_disj"] for g in por_grafo])
    orden = np.argsort(d)
    ax.barh(range(len(d)), d[orden] / sigma, color=["#2a7fbf" if v > 0 else "#c04040" for v in d[orden]])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([por_grafo[i]["rule_id"].replace("A2-B0-C2-", "") for i in orden], fontsize=7)
    ax.axvline(0, color="k", lw=0.9)
    ax.axvline(d.mean() / sigma, color="crimson", lw=2, ls="--",
               label=f"media = {d.mean()/sigma:+.1f} σ")
    ax.axvspan(-1, 1, color="0.75", alpha=0.5, label=f"±1 σ ({etiqueta_sigma})")
    ax.set_xlabel("Δ (solap − disj)  [unidades de σ]")
    ax.set_title(f"El efecto contra el grano del instrumento\nσ = {sigma:.5f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")

    ax = axs[2]
    if pareado:
        x = np.array([p["d_2000"] for p in pareado])
        y = np.array([p["d_4000"] for p in pareado])
        ax.scatter(x, y, c=col[:len(x)], s=55, zorder=3)
        for p, xi, yi in zip(pareado, x, y):
            ax.annotate(p["rule_id"].replace("A2-B0-C2-", ""), (xi, yi), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        lim = max(abs(np.r_[x, y]).max() * 1.15, 1e-4)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], "0.6", lw=1, ls=":")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δ (solap − disj) a N=2000  [fracción de masa]")
        ax.set_ylabel("Δ (solap − disj) a N=4000")
        rho, pv = stats.spearmanr(x, y)
        ig = sum(p["mismo_signo"] for p in pareado)
        ax.set_title(f"¿El mismo grafo sigue ganando?\nmismo signo {ig}/{len(pareado)} · "
                     f"Spearman ρ={rho:+.2f} (p={pv:.3g})", fontsize=10)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "sin datos de N=2000", ha="center", va="center")

    fig.suptitle("F8-06 · F7-03 a N=4000 — mismo layout O(N²) en los dos brazos", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RUTA_PNG, dpi=140)
    print(f"[png] {Path(RUTA_PNG).name}")


if __name__ == "__main__":
    main()
