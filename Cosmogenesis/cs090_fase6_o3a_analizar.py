"""
cs090_fase6_o3a_analizar.py — FASE VI, tarea O3-A: análisis de la convergencia del efecto con la
resolución (11-ago-2026). Lee el CSV crudo que arma
`cs090_fase6_o3a_convergencia_resolucion.py tabla` (una fila por regla y por N) y produce:

  1. `cs090_fase6_o3a_pares_por_N.csv` — una fila por (par, N) con fracción de masa de la Clase I, de la
     Clase III, su diferencia Δmasa = III − I, lo mismo para κ_V, y los conteos de sumideros.
  2. Las tres respuestas que pidió el equipo, impresas y guardadas en
     `cs090_fase6_o3a_resumen_estadistico.csv`:
       (a) ¿en cuántos pares se MANTIENE EL SIGNO de Δmasa al subir N?
       (b) ¿la MAGNITUD MEDIA de Δ crece, se mantiene o decrece con N?
       (c) ¿la CORRELACIÓN par-a-par entre N=2000 y N=4000 es alta (el que ganaba sigue ganando)?
  3. `cs090_fase6_o3a_delta_vs_N.png` — el gráfico Δ vs N.

NO se exige que la fracción ABSOLUTA sea igual entre resoluciones (a más partículas la misma masa se
fragmenta en más sumideros, eso ya se sabe desde `FASE5B_investigacion_8sumideros_y_escala_CS.md`); lo
que se mira es el SIGNO y el ORDEN RELATIVO. No se declara cierre ni veredicto: se tabula y se grafica.

DECISIONES DE DISEÑO DEL GRÁFICO (por qué se ve como se ve)
-----------------------------------------------------------
Son 12 pares: pintar 12 colores distintos sería inventar identidad donde no la hay y el ojo no podría
seguir ninguna línea. En vez de eso, las líneas individuales van finas y se colorean por POLARIDAD
(paleta divergente de dos tonos + gris neutro): azul = el par donde ganaba la Clase III a N=2000,
naranja = el par donde ganaba la Clase I (Δ<0), gris = empate práctico. Encima va UNA línea gruesa con
la media entre pares, que es la serie que de verdad contesta la pregunta. Eje x en escala log2 porque
las resoluciones se duplican (2000 → 4000 → 8000). Línea de cero marcada, porque el signo de Δ es
exactamente lo que se está juzgando. Un solo eje y por panel (nunca doble eje).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
CSV_CRUDO = HERE / "cs090_fase6_o3a_resolucion_crudo.csv"
CSV_PARES = HERE / "cs090_fase6_o3a_pares_por_N.csv"
CSV_RESUMEN = HERE / "cs090_fase6_o3a_resumen_estadistico.csv"
PNG = HERE / "cs090_fase6_o3a_delta_vs_N.png"
SEL_JSON = HERE / "cs090_fase6_o3a_pares_seleccionados.json"

# paleta: divergente de dos polos + gris neutro (regla: nunca arcoíris, nunca hue en el punto medio)
C_POS, C_NEG, C_CERO, C_MEDIA = "#1f6feb", "#d9730d", "#9aa0a6", "#111418"
TOL_CERO = 1e-9


def cargar():
    import json
    sel = json.loads(SEL_JSON.read_text())
    d = pd.read_csv(CSV_CRUDO)
    filas = []
    for s in sel:
        for N in sorted(d.N.unique()):
            gI = d[(d.rule_id == s["rid_I"]) & (d.N == N)]
            gIII = d[(d.rule_id == s["rid_III"]) & (d.N == N)]
            if gI.empty or gIII.empty:
                continue
            I, III = gI.iloc[0], gIII.iloc[0]
            if pd.isna(I.fraccion_masa_en_sumideros) or pd.isna(III.fraccion_masa_en_sumideros):
                continue
            filas.append(dict(
                par=s["par"], N=N, rid_I=s["rid_I"], rid_III=s["rid_III"], K=s["K"], kcap=s["kcap"],
                fm_I=I.fraccion_masa_en_sumideros, fm_III=III.fraccion_masa_en_sumideros,
                d_fraccion_masa=III.fraccion_masa_en_sumideros - I.fraccion_masa_en_sumideros,
                kv_I=I.kappa_v_agregado, kv_III=III.kappa_v_agregado,
                d_kappa_v=III.kappa_v_agregado - I.kappa_v_agregado,
                nsink_I=I.n_sumideros, nsink_III=III.n_sumideros,
                abortado_I=I.abortado_por_timeout, abortado_III=III.abortado_por_timeout,
                dump_I=I.n_dump_final, dump_III=III.n_dump_final,
                aristas_I=I.n_aristas_grafo_final, aristas_III=III.n_aristas_grafo_final,
                diam_I=I.diam_grafo_final_OFICIAL, diam_III=III.diam_grafo_final_OFICIAL))
    return pd.DataFrame(filas)


def signo(x):
    return 0 if abs(x) <= TOL_CERO else (1 if x > 0 else -1)


def resumen(pares: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    base = pares[pares.N == 2000].set_index("par")
    out = []
    for N in sorted(pares.N.unique()):
        g = pares[pares.N == N].set_index("par")
        comunes = [p for p in g.index if p in base.index]
        dfm = g.loc[comunes, "d_fraccion_masa"].astype(float)
        dkv = g.loc[comunes, "d_kappa_v"].astype(float)
        b_fm = base.loc[comunes, "d_fraccion_masa"].astype(float)
        b_kv = base.loc[comunes, "d_kappa_v"].astype(float)

        signos_iguales = int(sum(signo(a) == signo(b) and signo(b) != 0
                                 for a, b in zip(dfm, b_fm)))
        n_pos = int((dfm > 0).sum())
        # test de signos (binomial de dos colas) sobre "III > I"
        p_signos = float(stats.binomtest(n_pos, len(dfm), 0.5).pvalue) if len(dfm) else np.nan
        try:
            p_wilcox = float(stats.wilcoxon(dfm).pvalue) if len(dfm) >= 6 else np.nan
        except Exception:
            p_wilcox = np.nan
        fila = dict(
            N=N, n_pares=len(dfm),
            dfm_media=dfm.mean(), dfm_mediana=dfm.median(), dfm_sem=dfm.sem(),
            dfm_media_abs=dfm.abs().mean(),
            fm_media_I=g.loc[comunes, "fm_I"].astype(float).mean(),
            fm_media_III=g.loc[comunes, "fm_III"].astype(float).mean(),
            dfm_media_relativa=(dfm / g.loc[comunes, "fm_I"].astype(float)).mean(),
            n_pares_III_mayor=n_pos, p_test_signos=p_signos, p_wilcoxon=p_wilcox,
            n_signo_igual_a_N2000=signos_iguales,
            pearson_vs_N2000=float(np.corrcoef(dfm, b_fm)[0, 1]) if len(dfm) > 2 else np.nan,
            spearman_vs_N2000=float(stats.spearmanr(dfm, b_fm).statistic) if len(dfm) > 2 else np.nan,
            dkv_media=dkv.mean(), dkv_mediana=dkv.median(),
            n_pares_kv_III_mayor=int((dkv > 0).sum()),
            spearman_kv_vs_N2000=float(stats.spearmanr(dkv, b_kv).statistic) if len(dkv) > 2 else np.nan,
            nsink_medio=pd.concat([g.loc[comunes, "nsink_I"], g.loc[comunes, "nsink_III"]]).astype(float).mean(),
        )
        out.append(fila)
    return pd.DataFrame(out)


def grafico(pares: pd.DataFrame, res: pd.DataFrame) -> None:
    base = pares[pares.N == 2000].set_index("par")["d_fraccion_masa"].to_dict()
    Ns = sorted(pares.N.unique())

    # panel del medio: Δ RELATIVO. La fracción absoluta sube sola con la resolución (a más partículas
    # la misma masa se resuelve en más sitios de colapso), así que un Δ absoluto más grande a N=4000
    # podría ser sólo "todo creció". Δ/fracción(Clase I) descuenta esa marea que sube para los dos.
    pares = pares.copy()
    pares["d_relativo"] = pares.d_fraccion_masa.astype(float) / pares.fm_I.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, col, titulo, ylab in (
            (axes[0], "d_fraccion_masa", "Δ fracción de masa en sumideros  (Clase III − Clase I)",
             "Δ fracción de masa"),
            (axes[1], "d_relativo", "Δ RELATIVO  (Clase III − Clase I) / Clase I",
             "Δ relativo (adimensional)"),
            (axes[2], "d_kappa_v", "Δ κ_V agregado  (Clase III − Clase I)", "Δ κ_V")):
        ax.axhline(0.0, color="#3c4043", lw=1.0, zorder=1)
        for par, g in pares.groupby("par"):
            g = g.sort_values("N")
            s = signo(base.get(par, 0.0))
            c = C_POS if s > 0 else (C_NEG if s < 0 else C_CERO)
            ax.plot(g.N, g[col].astype(float), "-o", color=c, lw=1.2, ms=4.5, alpha=0.55, zorder=2)
        col_media = {"d_fraccion_masa": "dfm_media", "d_relativo": "dfm_media_relativa",
                     "d_kappa_v": "dkv_media"}[col]
        m = res.set_index("N")[col_media]
        ax.plot(m.index, m.values, "-o", color=C_MEDIA, lw=2.6, ms=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.6, label="media entre pares")
        for N, v in m.items():
            ax.annotate(f"{v:+.4f}" if col == "d_fraccion_masa" else f"{v:+.3f}",
                        (N, v), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=C_MEDIA)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("N (partículas SPH; masa física total fija = 18800)")
        ax.set_ylabel(ylab)
        ax.set_title(titulo, fontsize=10.5, color="#202124")
        ax.grid(True, axis="y", color="#e8eaed", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#bdc1c6")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_POS, lw=1.6, marker="o", ms=4.5,
                      label="par con Δ>0 a N=2000 (ganaba Clase III)"),
               Line2D([], [], color=C_NEG, lw=1.6, marker="o", ms=4.5,
                      label="par con Δ<0 a N=2000 (ganaba Clase I)"),
               Line2D([], [], color=C_MEDIA, lw=2.6, marker="o", ms=8, label="media entre pares")]
    axes[0].legend(handles=handles, loc="best", frameon=False, fontsize=8.5)
    fig.suptitle("O3-A · ¿el efecto Clase III > Clase I sobrevive al subir la resolución?",
                 fontsize=12.5, color="#202124")
    fig.savefig(PNG, dpi=170)
    print(f"[png] {PNG}")


def restringir_a_pares_completos(pares: pd.DataFrame) -> pd.DataFrame:
    """Se queda SÓLO con los pares que tienen dato en TODAS las resoluciones presentes. Si un par tiene
    N=2000 pero le falta N=4000 (por ejemplo porque el presupuesto de tiempo se agotó antes de generar
    su layout), incluirlo haría que la media a N=2000 se calcule sobre más pares que la media a N=4000 y
    las dos columnas dejarían de ser comparables. Se informa cuáles quedaron afuera."""
    Ns = set(pares.N.unique())
    completos = [p for p, g in pares.groupby("par") if set(g.N) == Ns]
    fuera = sorted(set(pares.par) - set(completos))
    if fuera:
        print(f"[aviso] {len(fuera)} par(es) sin dato en todas las resoluciones, excluidos del resumen "
              f"(sí quedan en el CSV crudo): {fuera}")
    return pares[pares.par.isin(completos)].copy()


if __name__ == "__main__":
    pares = cargar()
    pares.to_csv(CSV_PARES, index=False)
    print(f"[pares] {len(pares)} filas (par×N) -> {CSV_PARES}")
    pd.set_option("display.width", 220)
    print(pares[["par", "N", "fm_I", "fm_III", "d_fraccion_masa", "d_kappa_v",
                 "nsink_I", "nsink_III"]].to_string(index=False))
    comp = restringir_a_pares_completos(pares)
    res = resumen(comp)
    res.to_csv(CSV_RESUMEN, index=False)
    print("\n[resumen por N -- sólo pares con dato en TODAS las resoluciones]")
    print(res.to_string(index=False))
    grafico(comp, res)
