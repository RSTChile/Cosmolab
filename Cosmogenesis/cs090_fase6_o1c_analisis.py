"""
CS090 — FASE VI / O1-C: ANÁLISIS ESTADÍSTICO DE LA MUESTRA GRANDE DE A0
====================================================================================================
QUIÉN SOY
---------
Lee `cs090_fase6_o1c_a0_resumen.csv` (que produce `cs090_fase6_o1c_cierre_a0.py`) y contesta LA
pregunta central de la tarea O1-C con tests formales:

    Las reglas que el método VIEJO llama "Clase II — mundo-pequeño congelado",
    ¿se distinguen de las "Clase I" cuando se las mira con las métricas NATIVAS del campo
    (ξ(r) y dominios por adyacencia física, sin ningún grafo derivado)?

Con la muestra de 2 reglas Clase II que había antes no se podía testear nada. Acá se usa la muestra
grande y se aplican, en este orden:

  1. **Kolmogorov-Smirnov de dos muestras** (el test que pidió el analista): ¿son distinguibles las
     dos distribuciones nativas COMPLETAS? Es sensible a cualquier diferencia (posición, dispersión,
     forma), no sólo al promedio.
  2. **Mann-Whitney U**: ¿tiende un grupo a dar valores más altos que el otro? (test de rangos, no
     supone normalidad ni varianzas iguales).
  3. **Permutación** (20.000 remuestreos de las etiquetas): p-valor que no depende de ninguna
     aproximación asintótica. Importa porque varias métricas nativas tienen MUCHOS empates
     (ξ(r) toma pocos valores discretos) y con empates el KS asintótico se vuelve conservador.
  4. **Cliff's delta**: tamaño del efecto (no sólo si hay diferencia, sino cuánta). |δ|<0.147 se lee
     como "insignificante", 0.147-0.33 chico, 0.33-0.474 mediano, >0.474 grande (convención de Romano
     et al., citada tal cual; el umbral no se inventa acá).
  5. **Holm-Bonferroni** sobre la familia de métricas nativas: se testean varias métricas, así que se
     reporta también el p ajustado (no se pesca el mínimo y se lo presenta como si fuera único).
  6. **Análisis de potencia por simulación**: si NO se encuentra diferencia, ¿qué diferencia SÍ se
     habría encontrado? Se remuestrea de la distribución empírica de Clase I, se le suma un
     desplazamiento a un grupo del tamaño del de Clase II, y se mide con qué frecuencia el test lo
     detecta. Sin esto, "p>0.05" no dice nada: podría ser falta de potencia. CON esto se puede decir
     "un efecto de tamaño X o mayor se habría visto el 80% de las veces".

Además:
  - **Versión continua (sin dicotomía)**: correlación de Spearman entre la pendiente vieja (variable
    continua, antes de que el umbral 0.35 la corte en clases) y cada métrica nativa. Si el "Clase II"
    de A0 midiera algo real del campo, la pendiente vieja debería correlacionar con lo nativo aunque
    el corte sea arbitrario.
  - **Control NULL** (punto 4 de la tarea): Wilcoxon pareado REAL vs. NULL (campo barajado) sobre la
    misma muestra nueva, para confirmar que las métricas nativas SIGUEN distinguiendo estructura real
    de ruido acá — si no distinguieran, cualquier "no hay diferencia I vs II" sería trivial.
  - **Todo el análisis se repite con la clase del método viejo CORREGIDO** (diámetro medido en la
    componente gigante, `cs090_diam_corregido`), por si el bug de diámetro moviera las etiquetas.

Salidas: `cs090_fase6_o1c_tests.csv` (todos los tests en una tabla) y `cs090_fase6_o1c_distribuciones.png`.
No declara cierre ni veredicto: reporta números.
"""
from __future__ import annotations

import csv
import sys

import numpy as np
from scipy import stats

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"

# métricas nativas que se comparan entre grupos (todas leen el campo, ninguna usa grafo derivado)
METRICAS_NATIVAS = [
    ("corr_slope_nativo", "pendiente de ξ(r) vs escala (log-log)"),
    ("dom_slope_nativo", "pendiente del dominio local vs escala (log-log)"),
    ("giant_frac_b1", "fracción del anillo en el dominio mayor (b=1)"),
    ("corr_len_b1", "longitud de correlación ξ a resolución nativa (b=1)"),
    ("n_dominios_b1", "número de dominios de fase (b=1)"),
]

COLOR_I = "#2a78d6"      # categorical slot 1 (azul)
COLOR_II = "#eb6834"     # categorical slot 2 (naranja)
COLOR_NULL = "#8a8a85"   # gris neutro: el control, nunca compite con las dos series
SURFACE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"


# ============================================================================================
# 1) CARGA
# ============================================================================================
def cargar(ruta=f"{_HERE}/cs090_fase6_o1c_a0_resumen.csv"):
    with open(ruta) as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k, v in list(f.items()):
            if k in ("rule_id", "clase_vieja", "clase_vieja_corr"):
                continue
            if v in ("True", "False"):
                f[k] = (v == "True")
                continue
            try:
                f[k] = float(v)
            except (TypeError, ValueError):
                pass
    return filas


# ============================================================================================
# 2) TESTS (cada uno devuelve un dict de una línea, para armar la tabla)
# ============================================================================================
def cliffs_delta(a, b):
    """Cliff's delta = P(a>b) - P(a<b). Va de -1 a +1; 0 = las dos muestras se superponen del todo."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    may = sum((x > b).sum() for x in a)
    men = sum((x < b).sum() for x in a)
    return (may - men) / (len(a) * len(b))


def p_permutacion(a, b, n_perm=20000, semilla=12345):
    """p-valor de dos colas por permutación de etiquetas sobre la DIFERENCIA DE MEDIAS. No supone
    ninguna distribución ni corrección de empates: baraja las etiquetas n_perm veces y cuenta cuántas
    barajadas dan una diferencia tan extrema como la observada."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    todos = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(semilla)
    cuenta = 0
    for _ in range(n_perm):
        rng.shuffle(todos)
        if abs(todos[:na].mean() - todos[na:].mean()) >= obs - 1e-12:
            cuenta += 1
    return (cuenta + 1) / (n_perm + 1)


def comparar_grupos(g1, g2, etq1, etq2, columna, descripcion):
    a = np.array([f[columna] for f in g1], float)
    b = np.array([f[columna] for f in g2], float)
    ks = stats.ks_2samp(a, b)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(
        metrica=columna, descripcion=descripcion,
        n_1=len(a), n_2=len(b), etq_1=etq1, etq_2=etq2,
        media_1=round(float(a.mean()), 4), media_2=round(float(b.mean()), 4),
        mediana_1=round(float(np.median(a)), 4), mediana_2=round(float(np.median(b)), 4),
        min_1=round(float(a.min()), 4), max_1=round(float(a.max()), 4),
        min_2=round(float(b.min()), 4), max_2=round(float(b.max()), 4),
        KS_D=round(float(ks.statistic), 4), KS_p=float(ks.pvalue),
        MW_U=float(mw.statistic), MW_p=float(mw.pvalue),
        perm_p=p_permutacion(a, b),
        cliffs_delta=round(cliffs_delta(a, b), 4),
    )


def holm(ps):
    """Holm-Bonferroni: devuelve los p ajustados en el mismo orden en que entraron."""
    m = len(ps)
    orden = np.argsort(ps)
    ajust = np.empty(m)
    prev = 0.0
    for rango, i in enumerate(orden):
        val = min(1.0, (m - rango) * ps[i])
        prev = max(prev, val)
        ajust[i] = prev
    return ajust


def spearman_parcial(x, y, z):
    """Spearman parcial: correlación de rangos entre x e y DESPUÉS de sacarle a cada uno lo que
    explica z (por regresión lineal sobre los rangos). Hace falta porque las métricas nativas de
    dominio usan el MISMO umbral `sim_thr_frac*K` que el grafo de medición del método viejo: si los
    dos se mueven juntos, podría ser por el parámetro compartido y no por estructura del campo.
    Devuelve (rho_parcial, p) con el p por la t de Student con n-3 grados de libertad."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.column_stack([rz, np.ones_like(rz)])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    ex, ey = resid(rx), resid(ry)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = len(x)
    if n <= 3 or abs(r) >= 1:
        return r, float("nan")
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def potencia_por_simulacion(muestra_base, n1, n2, deltas, n_sim=2000, alfa=0.05, semilla=777):
    """¿Qué tan grande tendría que ser la diferencia para que estos tests la vieran con esta n?
    Receta: se remuestrea CON reemplazo de la distribución empírica observada (bootstrap), se le suma
    `delta` al grupo chico, y se cuenta la fracción de simulaciones en que KS y Mann-Whitney dan
    p<alfa. Devuelve lista de (delta, potencia_KS, potencia_MW).

    LIMITACIÓN DECLARADA: esto sólo tiene sentido para métricas de valores razonablemente CONTINUOS.
    Para una métrica cuasi-discreta (ξ(r) toma un puñado de valores fijos) sumar cualquier Δ crea
    valores que NO existen en la muestra base, y el test los separa trivialmente: la curva de potencia
    daría ~95% incluso para Δ minúsculos, lo cual no informa nada. Por eso `analizar_grupo` corre la
    potencia sólo sobre las métricas con muchos valores distintos, y lo documenta."""
    rng = np.random.default_rng(semilla)
    base = np.asarray(muestra_base, float)
    out = []
    for d in deltas:
        ok_ks = ok_mw = 0
        for _ in range(n_sim):
            a = rng.choice(base, n1, replace=True)
            b = rng.choice(base, n2, replace=True) + d
            if stats.ks_2samp(a, b).pvalue < alfa:
                ok_ks += 1
            if stats.mannwhitneyu(a, b, alternative="two-sided").pvalue < alfa:
                ok_mw += 1
        out.append((d, ok_ks / n_sim, ok_mw / n_sim))
    return out


# ============================================================================================
# 3) GRÁFICO
# ============================================================================================
def graficar(filas, col_clase, ruta_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gI = [f for f in filas if f[col_clase] == "I"]
    gII = [f for f in filas if f[col_clase] == "II"]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#d7d6d2")
        ax.tick_params(colors=TINTA2, labelsize=9)
        ax.grid(True, color="#eceae6", linewidth=0.8)
        ax.set_axisbelow(True)

    # (a) la variable del método viejo, con la banda de umbral que define Clase II
    ax = axes[0, 0]
    pend = np.array([f["pendiente_vieja"] for f in filas])
    ax.hist(pend, bins=40, color="#9ec5f4", edgecolor=SURFACE, linewidth=0.6)
    ax.axvspan(0.35, 0.45, color=COLOR_II, alpha=0.16, lw=0)
    ax.axvline(0.35, color=COLOR_II, lw=2)
    ax.set_title("a) Método viejo: pendiente log(diám)–log(cajas)\nla banda naranja es el corte "
                 "que define 'Clase II'", fontsize=10.5, color=TINTA, loc="left")
    ax.set_xlabel("pendiente (método viejo)", fontsize=9.5, color=TINTA2)
    ax.set_ylabel("nº de reglas", fontsize=9.5, color=TINTA2)

    # (b)-(c) ECDF de las dos métricas nativas principales, Clase I vs Clase II
    for ax, col, titulo in ((axes[0, 1], "dom_slope_nativo", "b) Nativo: pendiente de dominio local"),
                            (axes[1, 0], "corr_slope_nativo", "c) Nativo: pendiente de ξ(r)")):
        for g, color, etq in ((gI, COLOR_I, f"Clase I (n={len(gI)})"),
                              (gII, COLOR_II, f"Clase II (n={len(gII)})")):
            v = np.sort([f[col] for f in g])
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(np.concatenate([v, v[-1:]]), np.concatenate([y, [1.0]]),
                    where="post", color=color, lw=2, label=etq)
        ax.set_title(titulo + "\n(si las curvas se pisan, los dos grupos son el mismo campo)",
                     fontsize=10.5, color=TINTA, loc="left")
        ax.set_xlabel(col, fontsize=9.5, color=TINTA2)
        ax.set_ylabel("fracción acumulada de reglas", fontsize=9.5, color=TINTA2)
        leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
        for t in leg.get_texts():
            t.set_color(TINTA2)

    # (d) control NULL: la métrica nativa sobre campo real vs. campo barajado
    ax = axes[1, 1]
    real = np.array([f["dom_slope_nativo"] for f in filas])
    nulo = np.array([f["dom_slope_null"] for f in filas])
    bins = np.linspace(min(real.min(), nulo.min()), max(real.max(), nulo.max()), 40)
    ax.hist(nulo, bins=bins, color=COLOR_NULL, alpha=0.85, edgecolor=SURFACE, linewidth=0.6,
            label=f"NULL barajado (n={len(nulo)})")
    ax.hist(real, bins=bins, color=COLOR_I, alpha=0.85, edgecolor=SURFACE, linewidth=0.6,
            label=f"campo REAL (n={len(real)})")
    ax.set_title("d) Control: ¿las métricas nativas ven algo?\nreal vs. el mismo campo con las "
                 "posiciones barajadas", fontsize=10.5, color=TINTA, loc="left")
    ax.set_xlabel("dom_slope_nativo", fontsize=9.5, color=TINTA2)
    ax.set_ylabel("nº de reglas", fontsize=9.5, color=TINTA2)
    leg = ax.legend(frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(TINTA2)

    # (e) el confound: las dos clases viven en tramos distintos del MISMO umbral compartido
    ax = axes[2, 0]
    for g, color, etq in ((gI, COLOR_I, "Clase I"), (gII, COLOR_II, "Clase II")):
        x = [f["sim_thr_frac"] * f["K"] for f in g]
        y = [f["giant_frac_b1"] for f in g]
        ax.scatter(x, y, s=22, color=color, alpha=0.7, linewidths=0.5, edgecolors=SURFACE, label=etq)
    ax.set_title("e) El confound: umbral compartido thr = sim_thr_frac·K\nlas dos clases son tramos "
                 "distintos de la MISMA curva", fontsize=10.5, color=TINTA, loc="left")
    ax.set_xlabel("thr = sim_thr_frac · K  (lo usan los DOS métodos)", fontsize=9.5, color=TINTA2)
    ax.set_ylabel("giant_frac_b1 (métrica nativa)", fontsize=9.5, color=TINTA2)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(TINTA2)

    # (f) la cadena causal del método viejo: umbral -> densidad del grafo derivado -> diámetro entero
    ax = axes[2, 1]
    bordes = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.8]
    centros, tasas, etiquetas = [], [], []
    for a, b in zip(bordes, bordes[1:]):
        g = [f for f in filas if a <= f["sim_thr_frac"] * f["K"] < b]
        if not g:
            continue
        centros.append(len(centros))
        tasas.append(100.0 * sum(1 for f in g if f[col_clase] == "II") / len(g))
        etiquetas.append(f"{a:.1f}–{b:.1f}\n(n={len(g)})")
    barras = ax.bar(centros, tasas, color=COLOR_II, width=0.62)
    for x, v in zip(centros, tasas):
        ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9, color=TINTA2)
    ax.set_xticks(centros); ax.set_xticklabels(etiquetas, fontsize=8.5)
    ax.set_title("f) Qué predice ser 'Clase II': el umbral de la regla\n(a umbral alto, el grafo "
                 "derivado es denso y NUNCA sale Clase II)", fontsize=10.5, color=TINTA, loc="left")
    ax.set_xlabel("banda de thr = sim_thr_frac · K", fontsize=9.5, color=TINTA2)
    ax.set_ylabel("% de reglas clasificadas Clase II", fontsize=9.5, color=TINTA2)

    fig.suptitle("O1-C — A0: lo que el método viejo separa en clases, ¿se separa también en el campo?",
                 fontsize=13, color=TINTA, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(ruta_png, dpi=150, facecolor=SURFACE)
    print(f"  -> {ruta_png}")


# ============================================================================================
# 4) MAIN
# ============================================================================================
def analizar_grupo(filas, col_clase, etiqueta_metodo, salida_tests):
    gI = [f for f in filas if f[col_clase] == "I"]
    gII = [f for f in filas if f[col_clase] == "II"]
    otros = [f for f in filas if f[col_clase] not in ("I", "II")]
    print(f"\n{'='*100}\nGRUPOS SEGÚN {etiqueta_metodo}: Clase I n={len(gI)}, Clase II n={len(gII)}, "
          f"otras clases n={len(otros)}")
    if otros:
        cl = {}
        for f in otros:
            cl[f[col_clase]] = cl.get(f[col_clase], 0) + 1
        print(f"  (otras: {cl} — se excluyen de la comparación I vs II, no se fuerzan a un grupo)")
    if len(gII) < 3:
        print("  !! muy pocas Clase II para testear con esta muestra")
        return
    print(f"{'='*100}")

    resultados = []
    for col, desc in METRICAS_NATIVAS:
        resultados.append(comparar_grupos(gI, gII, "Clase I", "Clase II", col, desc))
    p_ks_aj = holm([r["KS_p"] for r in resultados])
    p_mw_aj = holm([r["MW_p"] for r in resultados])
    for r, a, b in zip(resultados, p_ks_aj, p_mw_aj):
        r["KS_p_holm"] = float(a); r["MW_p_holm"] = float(b)
        r["metodo_clase"] = etiqueta_metodo

    # ---- ¿los dos grupos difieren en los PARÁMETROS de la regla? (confound compartido) ----
    # El umbral `sim_thr_frac*K` lo usan LAS DOS mediciones: el grafo derivado del método viejo y los
    # dominios nativos. Si Clase II tuviera sistemáticamente otro umbral, ambas métricas se moverían
    # juntas por el parámetro, no por estructura del campo. Se chequea explícitamente.
    print("\nParámetros de la regla por grupo (chequeo de confound; el umbral thr=sim_thr_frac*K lo "
          "comparten los DOS métodos):")
    for col in ("K", "J", "noise", "sim_thr_frac", "n_aristas_b1"):
        a = np.array([f[col] for f in gI], float); b = np.array([f[col] for f in gII], float)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {col:<14} I={a.mean():9.4f}  II={b.mean():9.4f}   MW p={mw.pvalue:.4f}")
        salida_tests.append(dict(metodo_clase=etiqueta_metodo, metrica=col,
                                 descripcion="parámetro de la regla, Clase I vs Clase II (confound)",
                                 n_1=len(a), n_2=len(b), etq_1="Clase I", etq_2="Clase II",
                                 media_1=round(float(a.mean()), 4), media_2=round(float(b.mean()), 4),
                                 MW_U=float(mw.statistic), MW_p=float(mw.pvalue)))
    thr_I = np.array([f["sim_thr_frac"] * f["K"] for f in gI])
    thr_II = np.array([f["sim_thr_frac"] * f["K"] for f in gII])
    mw = stats.mannwhitneyu(thr_I, thr_II, alternative="two-sided")
    print(f"  {'thr=sim_thr*K':<14} I={thr_I.mean():9.4f}  II={thr_II.mean():9.4f}   "
          f"MW p={mw.pvalue:.4f}")
    salida_tests.append(dict(metodo_clase=etiqueta_metodo, metrica="thr_sim_thr_frac_x_K",
                             descripcion="umbral compartido por los dos métodos, I vs II",
                             n_1=len(thr_I), n_2=len(thr_II), etq_1="Clase I", etq_2="Clase II",
                             media_1=round(float(thr_I.mean()), 4),
                             media_2=round(float(thr_II.mean()), 4),
                             MW_U=float(mw.statistic), MW_p=float(mw.pvalue)))

    print(f"\n{'métrica nativa':<22} {'media I':>9} {'media II':>9} {'KS D':>7} {'KS p':>9} "
          f"{'MW p':>9} {'perm p':>9} {'δ Cliff':>8} {'KS p Holm':>10}")
    print("-" * 100)
    for r in resultados:
        print(f"{r['metrica']:<22} {r['media_1']:>9.4f} {r['media_2']:>9.4f} {r['KS_D']:>7.3f} "
              f"{r['KS_p']:>9.4f} {r['MW_p']:>9.4f} {r['perm_p']:>9.4f} {r['cliffs_delta']:>8.3f} "
              f"{r['KS_p_holm']:>10.4f}")
    salida_tests += resultados

    # ---- versión continua: ¿la pendiente vieja (sin dicotomizar) correlaciona con lo nativo? ----
    col_pend = "pendiente_vieja" if col_clase == "clase_vieja" else "pendiente_vieja_corr"
    pend = np.array([f[col_pend] for f in filas], float)
    thr_todas = np.array([f["sim_thr_frac"] * f["K"] for f in filas], float)
    print(f"\nVersión continua (sin dicotomía) — Spearman entre {col_pend} y cada métrica nativa, "
          f"sobre las {len(filas)} reglas; 'parcial' = descontando el umbral compartido "
          f"thr=sim_thr_frac*K:")
    for col, desc in METRICAS_NATIVAS:
        v = np.array([f[col] for f in filas], float)
        rho, p = stats.spearmanr(pend, v)
        rho_p, p_p = spearman_parcial(pend, v, thr_todas)
        print(f"  {col:<22} rho={rho:+.3f} (p={p:.4f})   parcial(thr)={rho_p:+.3f} (p={p_p:.4f})")
        salida_tests.append(dict(metodo_clase=etiqueta_metodo, metrica=col,
                                 descripcion="Spearman pendiente_vieja vs métrica nativa (continuo)",
                                 n_1=len(filas), n_2=0, etq_1="rho", etq_2="rho_parcial_thr",
                                 KS_D=round(float(rho), 4), KS_p=float(p),
                                 MW_U=round(float(rho_p), 4), MW_p=float(p_p)))

    # ---- comparación EMPAREJADA por umbral (el confound compartido) ----
    # Para cada regla Clase II se busca la regla Clase I con el umbral thr más parecido (sin
    # reemplazo). Así los dos grupos quedan con el mismo umbral y cualquier diferencia nativa que
    # sobreviva ya no puede explicarse por el parámetro compartido. Es el mismo espíritu que el
    # emparejamiento por densidad que usan los NULL del proyecto.
    libres = list(range(len(gI)))
    thr_I_todos = [f["sim_thr_frac"] * f["K"] for f in gI]
    pares = []
    for f2 in gII:
        t2 = f2["sim_thr_frac"] * f2["K"]
        if not libres:
            break
        j = min(libres, key=lambda i: abs(thr_I_todos[i] - t2))
        libres.remove(j)
        pares.append((gI[j], f2))
    d_thr = [abs(a["sim_thr_frac"] * a["K"] - b["sim_thr_frac"] * b["K"]) for a, b in pares]
    print(f"\nComparación EMPAREJADA por umbral thr (n={len(pares)} pares; |Δthr| medio="
          f"{np.mean(d_thr):.4f}, máx={np.max(d_thr):.4f}):")
    for col, _ in METRICAS_NATIVAS:
        a = np.array([x[col] for x, _ in pares], float)
        b = np.array([y[col] for _, y in pares], float)
        if np.allclose(a, b):
            continue
        w = stats.wilcoxon(a, b)
        print(f"  {col:<22} Clase I={a.mean():9.4f}  Clase II={b.mean():9.4f}   "
              f"Wilcoxon pareado p={w.pvalue:.4f}")
        salida_tests.append(dict(metodo_clase=etiqueta_metodo, metrica=col,
                                 descripcion="Wilcoxon pareado I vs II, emparejados por umbral thr",
                                 n_1=len(pares), n_2=len(pares), etq_1="Clase I (emparejada)",
                                 etq_2="Clase II", media_1=round(float(a.mean()), 4),
                                 media_2=round(float(b.mean()), 4),
                                 MW_U=float(w.statistic), MW_p=float(w.pvalue)))

    # ---- potencia: ¿qué diferencia SÍ se habría detectado con estas n? ----
    # sólo sobre métricas con suficientes valores distintos (ver limitación en potencia_por_simulacion)
    continuas = [c for c, _ in METRICAS_NATIVAS
                 if len(set(round(f[c], 6) for f in filas)) > 0.5 * len(filas)]
    print(f"\nPotencia por simulación (bootstrap de la distribución observada, n1={len(gI)}, "
          f"n2={len(gII)}, alfa=0.05). Métricas cuasi-discretas excluidas "
          f"(valores distintos < 50% de n): se corren {continuas}")
    for col in continuas:
        base = np.array([f[col] for f in filas], float)
        sd = base.std()
        deltas = [round(x * sd, 4) for x in (0.25, 0.5, 0.75, 1.0, 1.5)]
        pot = potencia_por_simulacion(base, len(gI), len(gII), deltas, n_sim=800)
        txt = "  ".join(f"Δ={d:g} ({d/sd:.2f}σ): KS {pk*100:.0f}% / MW {pm*100:.0f}%"
                        for d, pk, pm in pot)
        print(f"  {col} (σ={sd:.3f}):\n     {txt}")
        for d, pk, pm in pot:
            salida_tests.append(dict(metodo_clase=etiqueta_metodo, metrica=col,
                                     descripcion=f"potencia simulada para desplazamiento Δ={d} "
                                                 f"({d/sd:.2f} sigma)",
                                     n_1=len(gI), n_2=len(gII), etq_1="potencia_KS",
                                     etq_2="potencia_MW", KS_D=round(pk, 4), MW_U=round(pm, 4)))


def analizar_tuplas_diametro(filas, ruta=f"{_HERE}/cs090_fase6_o1c_a0_viejo_raw.csv"):
    """El diámetro de un grafo es un ENTERO CHICO (acá 2-6). La pendiente del método viejo sale de
    ajustar una recta a log(diám) contra log(n_cajas) sobre 5 escalas, o sea a 5 enteros chicos: hay
    poquísimas combinaciones posibles, y la pendiente sólo puede tomar unos pocos valores. Esta
    función tabula qué combinación de diámetros (b=1,2,4,8,16) tiene cada regla y cómo se reparten las
    clases entre esas combinaciones. Si "Clase II" resulta ser simplemente "las reglas cuya tupla de
    enteros es tal", entonces el corte 0.35 no está separando dos regímenes del campo sino dos
    resultados de redondeo del contador de saltos del BFS."""
    with open(ruta) as fh:
        crudo = list(csv.DictReader(fh))
    por_regla = {}
    for r in crudo:
        por_regla.setdefault(r["rule_id"], {})[int(r["escala_b"])] = int(float(r["diam_real_orig"]))
    clase_de = {f["rule_id"]: f["clase_vieja"] for f in filas}
    pend_de = {f["rule_id"]: f["pendiente_vieja"] for f in filas}
    conteo = {}
    for rid, dd in por_regla.items():
        if rid not in clase_de:
            continue
        tupla = tuple(dd[b] for b in (1, 2, 4, 8, 16))
        e = conteo.setdefault(tupla, {"I": 0, "II": 0, "otras": 0, "pend": []})
        c = clase_de[rid]
        e[c if c in ("I", "II") else "otras"] += 1
        e["pend"].append(pend_de[rid])
    print(f"\n{'='*100}\nDE DÓNDE SALE LA PENDIENTE VIEJA: la tupla de diámetros ENTEROS por escala\n{'='*100}")
    print(f"{'diám (b=1,2,4,8,16)':<24} {'pendiente':>10} {'n Clase I':>10} {'n Clase II':>11} {'total':>7}")
    print("-" * 100)
    orden = sorted(conteo.items(), key=lambda kv: -(kv[1]["I"] + kv[1]["II"] + kv[1]["otras"]))
    for tupla, e in orden[:14]:
        tot = e["I"] + e["II"] + e["otras"]
        print(f"{str(tupla):<24} {np.mean(e['pend']):>10.3f} {e['I']:>10} {e['II']:>11} {tot:>7}")
    if len(orden) > 14:
        print(f"... y {len(orden)-14} combinaciones más con menos reglas cada una")
    n_tuplas = len(orden)
    n_puras = sum(1 for _, e in orden if (e["I"] == 0) != (e["II"] == 0))
    print(f"\n  combinaciones de diámetros distintas: {n_tuplas}; de ellas, {n_puras} son 'puras' "
          f"(todas sus reglas caen en la MISMA clase)")
    return conteo


def main():
    filas = cargar()
    print(f"Cargadas {len(filas)} reglas A0-B0-C0 de cs090_fase6_o1c_a0_resumen.csv")

    # -------- control NULL primero: si las métricas nativas no ven nada, nada más tiene sentido ------
    print(f"\n{'='*100}\nCONTROL NULL (punto 4 de la tarea): campo REAL vs. el MISMO campo barajado\n{'='*100}")
    tests = []
    for col_real, col_null in (("dom_slope_nativo", "dom_slope_null"),
                               ("corr_slope_nativo", "corr_slope_null"),
                               ("giant_frac_b1", "giant_frac_null_b1"),
                               ("n_dominios_b1", "n_dominios_null_b1")):
        a = np.array([f[col_real] for f in filas], float)
        b = np.array([f[col_null] for f in filas], float)
        difs = a - b
        if np.allclose(difs, 0):
            w_p = 1.0; w_stat = 0.0
        else:
            w = stats.wilcoxon(a, b)
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        n_a_favor = int(np.sum(a > b))
        print(f"  {col_real:<22} REAL media={a.mean():8.4f} [{a.min():.3f},{a.max():.3f}]  |  "
              f"NULL media={b.mean():8.4f} [{b.min():.3f},{b.max():.3f}]  |  "
              f"REAL>NULL en {n_a_favor}/{len(a)}  Wilcoxon p={w_p:.3g}")
        tests.append(dict(metodo_clase="control NULL", metrica=col_real,
                          descripcion="Wilcoxon pareado campo REAL vs campo barajado",
                          n_1=len(a), n_2=len(b), etq_1="REAL", etq_2="NULL",
                          media_1=round(float(a.mean()), 4), media_2=round(float(b.mean()), 4),
                          KS_D=w_stat, KS_p=w_p, cliffs_delta=round(cliffs_delta(a, b), 4)))

    analizar_tuplas_diametro(filas)

    # -------- ¿DE QUÉ depende la pendiente vieja? (mecanismo, no sólo "sí/no hay diferencia") ------
    # Si la pendiente del método viejo estuviera midiendo el CAMPO, debería moverse con las métricas
    # nativas. Si estuviera midiendo el GRAFO DERIVADO que el propio método fabrica, debería moverse
    # con lo que fija la densidad de ese grafo: el umbral de similitud y el número de aristas.
    print(f"\n{'='*100}\nMECANISMO: ¿con qué correlaciona la pendiente del método viejo?\n{'='*100}")
    pend = np.array([f["pendiente_vieja"] for f in filas], float)
    print(f"{'variable':<26} {'qué es':<46} {'Spearman rho':>13} {'p':>10}")
    print("-" * 100)
    variables = [
        ("sim_thr_frac", "parámetro: umbral de similitud (fracción de K)"),
        ("K", "parámetro: tamaño del alfabeto de fase"),
        ("J", "parámetro: constante de acople de la dinámica"),
        ("noise", "parámetro: ruido por sweep"),
        ("n_aristas_b1", "aristas del GRAFO DERIVADO de medición"),
        ("dom_slope_nativo", "métrica NATIVA del campo (dominios)"),
        ("giant_frac_b1", "métrica NATIVA del campo (dominio mayor)"),
        ("corr_slope_nativo", "métrica NATIVA del campo (ξ(r))"),
    ]
    thr_todas = np.array([f["sim_thr_frac"] * f["K"] for f in filas], float)
    for col, que_es in variables:
        v = np.array([f[col] for f in filas], float)
        rho, p = stats.spearmanr(pend, v)
        print(f"{col:<26} {que_es:<46} {rho:>+13.3f} {p:>10.2e}")
        tests.append(dict(metodo_clase="mecanismo", metrica=col, descripcion=que_es,
                          n_1=len(filas), etq_1="Spearman vs pendiente_vieja",
                          KS_D=round(float(rho), 4), KS_p=float(p)))
    rho, p = stats.spearmanr(pend, thr_todas)
    print(f"{'thr = sim_thr_frac*K':<26} {'umbral absoluto (lo usan LOS DOS métodos)':<46} "
          f"{rho:>+13.3f} {p:>10.2e}")
    tests.append(dict(metodo_clase="mecanismo", metrica="thr_sim_thr_frac_x_K",
                      descripcion="umbral absoluto, Spearman vs pendiente_vieja",
                      n_1=len(filas), etq_1="Spearman vs pendiente_vieja",
                      KS_D=round(float(rho), 4), KS_p=float(p)))

    # -------- cuántos valores distintos toma cada métrica (para leer bien los KS con empates) ------
    print(f"\nValores DISTINTOS por métrica nativa (con empates masivos el KS asintótico se vuelve "
          f"conservador; por eso también se reporta el p por permutación):")
    for col, _ in METRICAS_NATIVAS:
        vals = sorted(set(round(f[col], 6) for f in filas))
        muestra = vals[:6] if len(vals) > 6 else vals
        print(f"  {col:<22} {len(vals):>4} valores distintos sobre {len(filas)} reglas "
              f"(los más bajos: {muestra})")

    # -------- comparación central, con las dos versiones del método viejo --------
    analizar_grupo(filas, "clase_vieja", "método viejo HISTÓRICO (cs055._diam)", tests)
    iguales = all(f["clase_vieja"] == f["clase_vieja_corr"] for f in filas)
    if iguales:
        print(f"\n{'='*100}\nMétodo viejo CORREGIDO (diam_gigante): asigna EXACTAMENTE las mismas "
              f"clases que el histórico en las {len(filas)} reglas\n"
              f"-> el análisis no se repite porque sería fila por fila idéntico (ver sección "
              f"'BUG DE DIÁMETRO' abajo para el porqué).\n{'='*100}")
    else:
        analizar_grupo(filas, "clase_vieja_corr", "método viejo CORREGIDO (diam_gigante)", tests)

    # -------- ¿el bug de diámetro afectó a esta línea? --------
    n_desc = sum(1 for f in filas if f["algun_descarrile"])
    n_cambio = sum(1 for f in filas if f["clase_vieja"] != f["clase_vieja_corr"])
    d_pend = np.array([abs(f["pendiente_vieja"] - f["pendiente_vieja_corr"]) for f in filas])
    print(f"\n{'='*100}\nBUG DE DIÁMETRO en esta línea (grafos de medición derivados de A0)\n{'='*100}")
    print(f"  reglas con alguna escala descarrilada (componente medida <10% de la gigante): "
          f"{n_desc}/{len(filas)}")
    print(f"  reglas cuya CLASE cambia al usar el diámetro corregido: {n_cambio}/{len(filas)}")
    print(f"  |Δ pendiente| entre las dos versiones: media={d_pend.mean():.5f} máx={d_pend.max():.5f}")
    tests.append(dict(metodo_clase="bug diámetro", metrica="descarrilamiento",
                      descripcion="reglas con alguna escala midiendo un fragmento en vez de la gigante",
                      n_1=n_desc, n_2=len(filas)))
    tests.append(dict(metodo_clase="bug diámetro", metrica="cambio_de_clase",
                      descripcion="reglas cuya clase cambia con el diámetro corregido",
                      n_1=n_cambio, n_2=len(filas)))

    campos = []
    for t in tests:
        for k in t:
            if k not in campos:
                campos.append(k)
    with open(f"{_HERE}/cs090_fase6_o1c_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for t in tests:
            w.writerow({k: t.get(k, "") for k in campos})
    print(f"\n  -> cs090_fase6_o1c_tests.csv ({len(tests)} filas)")
    graficar(filas, "clase_vieja", f"{_HERE}/cs090_fase6_o1c_distribuciones.png")


if __name__ == "__main__":
    sys.exit(main())
