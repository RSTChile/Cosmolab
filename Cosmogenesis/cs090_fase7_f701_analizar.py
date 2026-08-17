"""
cs090_fase7_f701_analizar.py — F7-01: estadística del factorial ortogonal kcap x M
====================================================================================

QUIÉN SOY: leo el CSV crudo que dejó `cs090_fase7_f701_factorial.py analizar` y produzco los números
que el criterio congelado de la tarea pide, más la figura de la superficie.

Los DOS NÚMEROS QUE DECIDEN (declarados antes de correr, no elegidos después):
  * R² de M sobre la masa A kcap FIJO  -> ¿mover sólo la densidad, con el tope de vecinos quieto,
                                          mueve la masa acretada?
  * R² de kcap sobre la masa A M FIJO  -> ¿queda algo del tope de vecinos cuando la densidad ya está
                                          igualada?
Si el segundo se va a cero, `kcap` opera únicamente vía densidad. Si queda un residual reproducible,
`kcap` tiene una segunda vía estructural. No se declara cierre: se reportan los números.

Se repite el mismo par de preguntas sobre los DOS ENDPOINTS geométricos congelados de la línea
(pendiente continua y clustering), porque si `kcap` no mueve la geometría a M igualado, tampoco tiene
por dónde mover la masa.

Ningún archivo existente se modifica. Sin commits.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
CRUDO = _HERE / "cs090_fase7_f701_crudo.csv"
PNG = _HERE / "cs090_fase7_f701_superficie.png"
RESUMEN = _HERE / "cs090_fase7_f701_resumen_celdas.csv"


def ols(X, y):
    """Mínimos cuadrados con intercepto. Devuelve (beta, R2, t, p, gl)."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    n, k = X.shape
    gl = n - k
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    s2 = sse / gl if gl > 0 else np.nan
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    return beta, r2, t, p, gl


def r2_simple(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0:
        return float("nan"), float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return r ** 2, r


def bloque(df, endpoint, etiqueta, out):
    """Los dos R² del criterio congelado, para un endpoint cualquiera."""
    out(f"\n{'='*94}\nENDPOINT: {etiqueta}  ({endpoint})\n{'='*94}")
    kcaps = sorted(df.kcap.unique())
    Ms = sorted(df.M_objetivo.unique())

    out("\n-- (1) A kcap FIJO: efecto de M (la densidad) --")
    r2s_M = []
    for k in kcaps:
        d = df[df.kcap == k]
        r2, r = r2_simple(d.n_aristas_grafo_final, d[endpoint])
        b, _, t, p, gl = ols([d.n_aristas_grafo_final], d[endpoint])
        r2s_M.append(r2)
        out(f"   kcap={k} (n={len(d)}): R²(M) = {r2:.4f}  r = {r:+.3f}  "
            f"pendiente = {b[1]*1000:+.5f} por 1000 aristas  t({gl}) = {t[1]:+.2f}  p = {p[1]:.4g}")
    out(f"   -> R² MEDIO de M sobre {etiqueta} a kcap fijo: {np.nanmean(r2s_M):.4f}")

    out("\n-- (2) A M FIJO: efecto de kcap (el tope de vecinos) --")
    r2s_k = []
    for M in Ms:
        d = df[df.M_objetivo == M]
        r2, r = r2_simple(d.kcap, d[endpoint])
        g6 = d[d.kcap == 6][endpoint].values; g7 = d[d.kcap == 7][endpoint].values
        try:
            U, pmw = stats.mannwhitneyu(g6, g7, alternative="two-sided")
        except ValueError:
            U, pmw = np.nan, np.nan
        tt, ptt = stats.ttest_ind(g6, g7, equal_var=False)
        r2s_k.append(r2)
        out(f"   M≈{M} (n={len(d)}): R²(kcap) = {r2:.4f}  media kcap6 = {g6.mean():.4f} "
            f"vs kcap7 = {g7.mean():.4f}  Δ = {g7.mean()-g6.mean():+.4f}  "
            f"t = {tt:+.2f} p = {ptt:.3f}  MWU p = {pmw:.3f}")
    out(f"   -> R² MEDIO de kcap sobre {etiqueta} a M fijo: {np.nanmean(r2s_k):.4f}")

    out("\n-- (3) Modelos anidados sobre las 24 corridas --")
    for nombre, cols in [("~ M", ["n_aristas_grafo_final"]),
                         ("~ kcap", ["kcap"]),
                         ("~ M + kcap", ["n_aristas_grafo_final", "kcap"]),
                         ("~ M+kcap+meandeg", ["n_aristas_grafo_final", "kcap", "meandeg"])]:
        b, r2, t, p, gl = ols([df[c] for c in cols], df[endpoint])
        det = "  ".join(f"{c} β={b[i+1]:+.6g} (t={t[i+1]:+.2f} p={p[i+1]:.3g})" for i, c in enumerate(cols))
        out(f"   {etiqueta} {nombre:<12} R² = {r2:.4f}   {det}")
    # residual de kcap una vez sacada M (correlación parcial)
    resid_y = df[endpoint] - np.poly1d(np.polyfit(df.n_aristas_grafo_final, df[endpoint], 1))(df.n_aristas_grafo_final)
    resid_k = df.kcap - np.poly1d(np.polyfit(df.n_aristas_grafo_final, df.kcap, 1))(df.n_aristas_grafo_final)
    rp = np.corrcoef(resid_y, resid_k)[0, 1]
    n = len(df); tp = rp * np.sqrt((n - 3) / max(1e-12, 1 - rp ** 2))
    out(f"   correlación parcial r(kcap, {etiqueta} | M) = {rp:+.3f}  "
        f"(t({n-3}) = {tp:+.2f}, p = {2*stats.t.sf(abs(tp), n-3):.4g})")
    out(f"   residuo medio de kcap6 = {resid_y[df.kcap==6].mean():+.5f} · "
        f"kcap7 = {resid_y[df.kcap==7].mean():+.5f}")
    # confound residual DECLARADO: a M igualado, `meandeg` (grado medio del ER de partida) NO queda
    # igualado — no se puede fijar M, meandeg y kcap a la vez con este generador. Se mide su peso.
    resid_md = df.meandeg - np.poly1d(np.polyfit(df.n_aristas_grafo_final, df.meandeg, 1))(df.n_aristas_grafo_final)
    rmd = np.corrcoef(resid_y, resid_md)[0, 1]
    out(f"   [confound declarado] r(meandeg, {etiqueta} | M) = {rmd:+.3f} · "
        f"r(kcap, meandeg | M) = {np.corrcoef(resid_k, resid_md)[0,1]:+.3f}")
    return np.nanmean(r2s_M), np.nanmean(r2s_k)


def main():
    df = pd.read_csv(CRUDO)
    df = df[df.kcap.notna()].copy()
    df["kcap"] = df.kcap.astype(int)
    # M objetivo: la celda de diseño a la que pertenece cada regla (el M real está en n_aristas)
    df["M_objetivo"] = (df.n_aristas_grafo_final / 200).round() * 200
    df["M_objetivo"] = df.M_objetivo.astype(int)

    lineas = []
    def out(s=""):
        print(s); lineas.append(str(s))

    out("F7-01 — FACTORIAL ORTOGONAL kcap x M (número de aristas)")
    out(f"n corridas = {len(df)}   ·   igualación de M = "
        f"{sorted(set(df.igualacion_M.dropna()))} (100 % por selección, 0 celdas por poda)")

    out("\n>> LO PRIMERO: ¿se rompió la colinealidad?")
    r_kM = np.corrcoef(df.kcap, df.n_aristas_grafo_final)[0, 1]
    r_kG = np.corrcoef(df.kcap, df.grado_medio_grafo_final)[0, 1]
    vif_k = 1 / (1 - r_kM ** 2)
    out(f"   r(kcap, M) = {r_kM:+.4f}   [en O3-D era r(kcap, grado medio) = +0,984]")
    out(f"   r(kcap, grado medio) = {r_kG:+.4f}   ·   VIF(kcap) = {vif_k:.2f}   "
        f"[en O3-D VIF(kcap) = 32,5 y VIF(grado medio) = 47,8]")
    out(f"   M por kcap: " + " · ".join(
        f"kcap{k}: media {df[df.kcap==k].n_aristas_grafo_final.mean():.1f} "
        f"[{df[df.kcap==k].n_aristas_grafo_final.min()}–{df[df.kcap==k].n_aristas_grafo_final.max()}]"
        for k in sorted(df.kcap.unique())))

    out("\n>> TABLA POR CELDA")
    g = df.groupby(["kcap", "M_objetivo"]).agg(
        n=("fraccion_masa_en_sumideros", "size"),
        M_real=("n_aristas_grafo_final", "mean"),
        grado_medio=("grado_medio_grafo_final", "mean"),
        meandeg_ER=("meandeg", "mean"),
        clustering=("clustering_medio", "mean"),
        pendiente=("pendiente_corregida", "mean"),
        frac_masa=("fraccion_masa_en_sumideros", "mean"),
        frac_masa_ee=("fraccion_masa_en_sumideros", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        kappa_v=("kappa_v_agregado", "mean"),
        n_sinks=("n_sumideros", "mean"),
        t_1er_sink=("t_primer_sumidero", "mean"),
    ).reset_index()
    out(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    g.to_csv(RESUMEN, index=False)

    r2M_masa, r2k_masa = bloque(df, "fraccion_masa_en_sumideros", "fracción de masa", out)
    bloque(df, "clustering_medio", "clustering", out)
    bloque(df, "pendiente_corregida", "pendiente corregida", out)
    bloque(df, "kappa_v_agregado", "κ_V agregado", out)

    out(f"\n{'='*94}\nLOS DOS NÚMEROS DEL CRITERIO CONGELADO (sobre la fracción de masa)")
    out(f"   R² de M sobre la masa, a kcap fijo : {r2M_masa:.4f}")
    out(f"   R² de kcap sobre la masa, a M fijo : {r2k_masa:.4f}")
    out("=" * 94)

    # ---------------- figura ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    colores = {6: "#1f77b4", 7: "#d62728"}
    paneles = [("fraccion_masa_en_sumideros", "fracción de masa en sumideros"),
               ("clustering_medio", "clustering medio del grafo"),
               ("pendiente_corregida", "pendiente corregida (diám. gigante)"),
               ("kappa_v_agregado", "κ_V agregado")]
    for ax, (col, lab) in zip(axes.ravel(), paneles):
        for k in sorted(df.kcap.unique()):
            d = df[df.kcap == k].sort_values("n_aristas_grafo_final")
            ax.scatter(d.n_aristas_grafo_final, d[col], s=46, color=colores[k],
                       label=f"kcap = {k}", edgecolor="k", linewidth=.4, zorder=3)
            if len(d) > 2:
                cf = np.polyfit(d.n_aristas_grafo_final, d[col], 1)
                xs = np.linspace(d.n_aristas_grafo_final.min(), d.n_aristas_grafo_final.max(), 20)
                ax.plot(xs, np.poly1d(cf)(xs), color=colores[k], lw=1.6, alpha=.75, zorder=2)
        ax.set_xlabel("M — aristas del grafo final")
        ax.set_ylabel(lab)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("F7-01 — factorial ortogonal: mismo M, distinto kcap (r(kcap,M) = "
                 f"{r_kM:+.3f}, VIF = {vif_k:.2f})\n"
                 "si las dos nubes se superponen a M igual, el tope de vecinos actuaba sólo vía densidad",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PNG, dpi=145)
    out(f"\n[figura] -> {PNG}")

    (_HERE / "cs090_fase7_f701_analisis.log").write_text("\n".join(lineas))


if __name__ == "__main__":
    main()
