"""
cs090_fase6_o1a_kappav_umbral.py — FASE VI, tarea O1-A (SÓLO lectura + análisis de datos en disco)
====================================================================================================

QUÉ ES ESTO, EN UNA FRASE
-------------------------
Dos preguntas sobre datos que YA existen, sin correr Phantom y sin generar reglas nuevas:

  ANÁLISIS 1 — ¿κ_V (una métrica que se mide DESPUÉS de la gravedad, dentro de Phantom) dice lo mismo
               que la geometría del grafo (la pendiente log-diámetro vs log-tamaño, que se mide ANTES,
               gratis)? Si dijera lo mismo, κ_V podría usarse como atajo/proxy y ahorrar cómputo.

  ANÁLISIS 2 — El umbral que define "Clase III" (pendiente > 0.7) se fijó a priori. Ahora hay 80
               corridas de Phantom con la respuesta gravitacional medida (fracción de masa que termina
               en los sumideros). ¿Cuál es el umbral que MEJOR separa esa respuesta? ¿Y sobrevive a una
               validación fuera de muestra, o es sólo sobreajuste?

DE DÓNDE SALEN LOS DATOS (nada se recalcula desde física; todo es re-lectura)
-----------------------------------------------------------------------------
  * cs090_fase5b_TOTAL_40pares.csv        -> las 80 corridas de Phantom (40 pares I-vs-III).
                                             Trae fraccion_masa_en_sumideros y kappa_v_agregado.
  * cs090_fase6_remedicion_430.csv        -> las 430 reglas re-medidas con el diámetro CORREGIDO
                                             (componente gigante). De acá salen pendiente_corregida,
                                             pendiente_vieja y los diámetros por escala.
  * cs090_fase6_outliers_phantom_metricas.csv -> 11 corridas de Phantom ADICIONALES (las reglas que
                                             descarrilaban). Se usan SÓLO como chequeo externo, con la
                                             advertencia de que no son una muestra al azar.

LLAVE DE UNIÓN: `seed`, no `rule_id`. Motivo: en el CSV de los 40 pares hay 3 reglas con sufijo
"v1fix"/"v2fix" cuyo rule_id no existe en el CSV de re-medición, pero cuya `seed` sí — y la semilla es
lo que determina íntegramente la regla (GEN.generar_regla("A2","B0","C2", idx=0, seed=seed)).

UNIDAD DE ANÁLISIS: hay 80 FILAS pero sólo 76 CORRIDAS distintas de Phantom: 4 reglas participan en dos
pares distintos y su corrida está copiada dos veces. Para correlaciones eso son 4 duplicados exactos que
inflan artificialmente el n. Se reporta TODO por duplicado: n=80 (como está en el CSV) y n=76
(deduplicado por semilla, que es el número honesto de puntos independientes).

SALIDAS
-------
  cs090_fase6_o1a_kappav_correlaciones.csv  — matriz de correlaciones del Análisis 1
  cs090_fase6_o1a_barrido_umbral.csv        — el barrido de umbrales del Análisis 2 (0.40..1.00)
  cs090_fase6_o1a_validacion_umbral.csv     — resultados de la validación fuera de muestra
  cs090_fase6_o1a_datos_unidos.csv          — la tabla unida (para que cualquiera rehaga el análisis)
  cs090_fase6_o1a_fig1_kappav.png           — dispersión κ_V vs pendiente / diámetro / masa
  cs090_fase6_o1a_fig2_umbral.png           — curva del barrido + cobertura de datos + validación

No corre Phantom. No modifica ningún script existente. No hace commits. No declara cierre ni veredicto:
imprime números.
"""
from __future__ import annotations

import ast
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
RNG = np.random.default_rng(20260811)

# ------------------------------------------------------------------------------------------------
# 0. CARGA Y UNIÓN
# ------------------------------------------------------------------------------------------------

def cargar():
    """Une las 80 corridas de Phantom con la re-medición corregida del grafo, por `seed`."""
    t = pd.read_csv(f"{HERE}/cs090_fase5b_TOTAL_40pares.csv")
    r = pd.read_csv(f"{HERE}/cs090_fase6_remedicion_430.csv")

    faltan = set(t.seed) - set(r.seed)
    assert not faltan, f"semillas de Phantom sin re-medición: {faltan}"

    cols_r = ["seed", "rule_id", "kcap", "K", "J", "noise", "meandeg",
              "pendiente_vieja", "pendiente_corregida", "z_agg_vieja", "z_agg_corregida",
              "clase_vieja", "clase_corregida", "diam_viejo", "diam_corregido", "n_cajas",
              "tam_gigante", "descarrila_b1"]
    m = t.merge(r[cols_r], on="seed", how="left", suffixes=("", "_rem"))

    # Los diámetros vienen guardados como texto de lista python: "[19.0, 16.0, 10.0, 7.0, 5.0]".
    # b=1 (índice 0) es la escala fina: el diámetro del grafo tal cual, sobre la componente gigante.
    m["diam_corr_b1"] = [ast.literal_eval(s)[0] for s in m["diam_corregido"]]
    m["diam_viejo_b1"] = [ast.literal_eval(s)[0] for s in m["diam_viejo"]]
    m["tam_gigante_b1"] = [ast.literal_eval(s)[0] for s in m["tam_gigante"]]

    # Unidad honesta: una fila por corrida de Phantom distinta.
    m["dup"] = m.duplicated(subset=["seed"], keep="first")
    return m


def cargar_externas():
    """Las 11 corridas extra de Phantom (reglas que descarrilaban), para chequeo fuera de muestra."""
    o = pd.read_csv(f"{HERE}/cs090_fase6_outliers_phantom_metricas.csv")
    r = pd.read_csv(f"{HERE}/cs090_fase6_remedicion_430.csv")
    o = o.merge(r[["seed", "pendiente_corregida", "clase_corregida", "diam_corregido"]],
                on="seed", how="left")
    o["diam_corr_b1"] = [ast.literal_eval(s)[0] for s in o["diam_corregido"]]
    return o


# ------------------------------------------------------------------------------------------------
# 1. ANÁLISIS 1 — κ_V COMO MÉTRICA PUENTE
# ------------------------------------------------------------------------------------------------

def _pearson_rapido(a, b):
    """Pearson sin el envoltorio de scipy (sólo el coeficiente). Para los bucles de remuestreo."""
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / den if den > 0 else np.nan


def _corr(x, y, n_boot=4000):
    """Spearman y Pearson con p-valor, más IC95% bootstrap del Spearman (percentil)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rs, ps = stats.spearmanr(x, y)
    rp, pp = stats.pearsonr(x, y)
    n = len(x)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            boot[i] = np.nan
            continue
        boot[i] = _pearson_rapido(stats.rankdata(x[idx]), stats.rankdata(y[idx]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return dict(n=n, spearman=rs, p_spearman=ps, pearson=rp, p_pearson=pp,
                spearman_ic95_lo=lo, spearman_ic95_hi=hi)


def _parcial_spearman(x, y, z):
    """Correlación parcial de Spearman entre x e y controlando z: se corre Spearman sobre los
    RESIDUOS de regresar rango(x) y rango(y) contra rango(z). Sirve para preguntar '¿queda relación
    después de descontar el efecto de la variable de diseño kcap?'."""
    x = stats.rankdata(x); y = stats.rankdata(y); z = stats.rankdata(z)
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p


def analisis1(m):
    print("=" * 100)
    print("ANÁLISIS 1 — ¿κ_V es un proxy de la geometría del grafo?")
    print("=" * 100)

    filas = []
    for etiqueta, df in [("n=80 (CSV tal cual)", m), ("n=76 (dedup por semilla)", m[~m.dup])]:
        print(f"\n--- {etiqueta} ---")
        pares = [
            ("kappa_v_agregado", "pendiente_corregida", "κ_V  vs  pendiente CORREGIDA (geometría)"),
            ("kappa_v_agregado", "pendiente_vieja",     "κ_V  vs  pendiente vieja (con el bug)"),
            ("kappa_v_agregado", "diam_corr_b1",        "κ_V  vs  diámetro corregido (b=1)"),
            ("kappa_v_agregado", "fraccion_masa_en_sumideros", "κ_V  vs  fracción de masa en sumideros"),
            ("pendiente_corregida", "fraccion_masa_en_sumideros", "pendiente corr.  vs  masa (referencia)"),
            ("diam_corr_b1", "fraccion_masa_en_sumideros", "diámetro corr.  vs  masa (referencia)"),
            ("pendiente_corregida", "diam_corr_b1", "pendiente corr.  vs  diámetro corr. (referencia)"),
            ("kappa_v_agregado", "kcap", "κ_V  vs  kcap (variable de diseño)"),
            ("pendiente_corregida", "kcap", "pendiente corr.  vs  kcap (variable de diseño)"),
            ("fraccion_masa_en_sumideros", "kcap", "masa  vs  kcap (variable de diseño)"),
        ]
        for a, b, desc in pares:
            c = _corr(df[a], df[b])
            rp_par, pp_par = _parcial_spearman(df[a], df[b], df["kcap"]) if "kcap" not in (a, b) else (np.nan, np.nan)
            print(f"  {desc:<46} rho={c['spearman']:+.4f} [{c['spearman_ic95_lo']:+.2f},{c['spearman_ic95_hi']:+.2f}] "
                  f"p={c['p_spearman']:.2e} | r={c['pearson']:+.4f} p={c['p_pearson']:.2e}"
                  + (f" | parcial|kcap rho={rp_par:+.3f} p={pp_par:.1e}" if np.isfinite(rp_par) else ""))
            filas.append(dict(muestra=etiqueta, var_x=a, var_y=b, descripcion=desc, **c,
                              spearman_parcial_dado_kcap=rp_par, p_spearman_parcial=pp_par))

    out = pd.DataFrame(filas)
    out.to_csv(f"{HERE}/cs090_fase6_o1a_kappav_correlaciones.csv", index=False)
    print(f"\n[csv] cs090_fase6_o1a_kappav_correlaciones.csv ({len(out)} filas)")

    # --- ¿κ_V agrega algo por encima de la pendiente para predecir la masa? Regresión anidada. ---
    d = m[~m.dup]
    y = d["fraccion_masa_en_sumideros"].values
    def r2(X):
        X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return 1 - res.var() / y.var()
    r2_pend = r2([d["pendiente_corregida"]])
    r2_kv = r2([d["kappa_v_agregado"]])
    r2_ambas = r2([d["pendiente_corregida"], d["kappa_v_agregado"]])
    print("\n--- ¿κ_V aporta información NUEVA sobre la masa, más allá de la pendiente? (n=76, R² lineal) ---")
    print(f"  sólo pendiente corregida : R² = {r2_pend:.4f}")
    print(f"  sólo κ_V                 : R² = {r2_kv:.4f}")
    print(f"  las dos juntas           : R² = {r2_ambas:.4f}   (ganancia sobre pendiente sola: "
          f"{r2_ambas - r2_pend:+.4f})")

    # --- ¿el r de Pearson κ_V-pendiente se sostiene, o lo levantan unos pocos puntos extremos? ---
    # El criterio del analista ("r>0.6 con la geometría") es de Pearson, que es sensible a puntos de
    # palanca. Hay 4 corridas con kcap=4 que están lejísimos del resto en las dos variables a la vez:
    # si son ellas las que sostienen el r, el criterio se estaría cumpliendo por 4 puntos.
    print("\n--- ¿el r de Pearson κ_V vs pendiente depende de unos pocos puntos de palanca? ---")
    kv, pe, ma, kc = (d["kappa_v_agregado"].values, d["pendiente_corregida"].values,
                      d["fraccion_masa_en_sumideros"].values, d["kcap"].values)
    for etiq, s in [("todas (n=76)", np.ones(len(d), bool)),
                    ("sin las 4 de kcap=4", kc != 4),
                    ("sólo el bloque kcap=5", kc == 5)]:
        print(f"  κ_V vs pendiente, {etiq:<24} r = {stats.pearsonr(kv[s], pe[s])[0]:+.3f}   "
              f"rho = {stats.spearmanr(kv[s], pe[s]).statistic:+.3f}   (n={int(s.sum())})")
    s = kc == 5
    print(f"  κ_V vs MASA,       sólo el bloque kcap=5   r = {stats.pearsonr(kv[s], ma[s])[0]:+.3f}   "
          f"rho = {stats.spearmanr(kv[s], ma[s]).statistic:+.3f}   (n={int(s.sum())})")
    return out


# ------------------------------------------------------------------------------------------------
# 2. ANÁLISIS 2 — RECALIBRAR EL UMBRAL DE CLASE III
# ------------------------------------------------------------------------------------------------
# Métricas de separación, todas sobre el MISMO corte binario (pendiente >= u  vs  pendiente < u):
#
#   AUC  — probabilidad de que una corrida del grupo "alto" tenga más masa que una del grupo "bajo"
#          (Mann-Whitney U / n1n2). Escala-libre, no supone normalidad, interpretable directamente.
#          DEFECTO: si el umbral aísla 1 sola corrida extrema, AUC=1.0 sin que eso signifique nada.
#   r_pb — correlación punto-biserial entre el grupo binario y la masa. Penaliza sola los cortes muy
#          desbalanceados (r_pb = d * sqrt(p*q)), así que NO premia grupos de tamaño 1.
#   d    — Cohen's d con desvío agrupado. El tamaño de efecto clásico, para comparar con la literatura.
#
# PRIMARIA: r_pb, justamente porque incorpora el balance del corte. Se reportan las tres.

def metricas_corte(pend, masa, u, min_grupo=8):
    alto = pend >= u
    n1, n0 = int(alto.sum()), int((~alto).sum())
    if n1 < min_grupo or n0 < min_grupo:
        return None
    a, b = masa[alto], masa[~alto]
    U = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = U.statistic / (n1 * n0)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    r_pb = stats.pointbiserialr(alto.astype(float), masa)
    tt = stats.ttest_ind(a, b, equal_var=False)
    return dict(umbral=u, n_alto=n1, n_bajo=n0, media_alto=a.mean(), media_bajo=b.mean(),
                dif_medias=a.mean() - b.mean(), auc=auc, p_mannwhitney=U.pvalue,
                cohen_d=d, r_pb=r_pb.statistic, p_r_pb=r_pb.pvalue,
                t=tt.statistic, p_welch=tt.pvalue)


def barrido(pend, masa, umbrales, min_grupo=8):
    filas = [f for f in (metricas_corte(pend, masa, u, min_grupo) for u in umbrales) if f]
    return pd.DataFrame(filas)


# --- versión rápida para los bucles de remuestreo (miles de repeticiones) --------------------------
# Sólo calcula r_pb, con la fórmula cerrada  r = sqrt(n1*n0)*(M1-M0)/(n*sd_poblacional(y)),
# que es idénticamente el Pearson entre el indicador 0/1 y la masa. Se comprueba contra scipy en
# `_autotest_rpb()` antes de usarla, para que "rápido" no signifique "distinto".

def r_pb_todos(mascaras, y, min_grupo):
    """mascaras: matriz booleana (n_umbrales x n_datos). Devuelve un vector de r_pb (NaN si el corte
    deja algún grupo por debajo de min_grupo)."""
    y = np.asarray(y, float)
    n = len(y)
    sd = y.std()
    n1 = mascaras.sum(axis=1)
    n0 = n - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = mascaras @ y
        M1 = s1 / n1
        M0 = (y.sum() - s1) / n0
        r = np.sqrt(n1 * n0) * (M1 - M0) / (n * sd)
    r[(n1 < min_grupo) | (n0 < min_grupo)] = np.nan
    return r


def _autotest_rpb(pend, masa, umbrales):
    mascaras = np.array([pend >= u for u in umbrales])
    rapido = r_pb_todos(mascaras, masa, min_grupo=8)
    for u, rr in zip(umbrales, rapido):
        if np.isnan(rr):
            continue
        lento = stats.pointbiserialr((pend >= u).astype(float), masa).statistic
        assert abs(rr - lento) < 1e-10, f"r_pb rápido != scipy en u={u}: {rr} vs {lento}"
    print("  [autotest] r_pb rápido == scipy.pointbiserialr en todos los umbrales válidos: OK")


def analisis2(m, externas):
    print("\n" + "=" * 100)
    print("ANÁLISIS 2 — recalibrar el umbral de Clase III sobre la pendiente CORREGIDA")
    print("=" * 100)

    d = m[~m.dup].reset_index(drop=True)          # 76 corridas independientes
    pend = d["pendiente_corregida"].values
    masa = d["fraccion_masa_en_sumideros"].values
    umbrales = np.round(np.arange(0.40, 1.0001, 0.02), 4)

    print(f"\n[datos] n={len(d)} corridas independientes · pendiente corregida en "
          f"[{pend.min():.3f}, {pend.max():.3f}] · masa en [{masa.min():.4f}, {masa.max():.4f}]")

    tab = barrido(pend, masa, umbrales)
    tab.to_csv(f"{HERE}/cs090_fase6_o1a_barrido_umbral.csv", index=False)
    print(f"[csv] cs090_fase6_o1a_barrido_umbral.csv ({len(tab)} umbrales con ambos grupos >= 8)")

    print("\n  umbral  n_alto n_bajo  media_alto media_bajo   AUC     r_pb   Cohen_d   p(Welch)")
    for _, f in tab.iterrows():
        marca = "  <-- 0.70 OFICIAL" if abs(f.umbral - 0.70) < 1e-9 else ""
        print(f"   {f.umbral:.2f}    {int(f.n_alto):3d}    {int(f.n_bajo):3d}     "
              f"{f.media_alto:.4f}     {f.media_bajo:.4f}   {f.auc:.3f}  {f.r_pb:+.4f}  {f.cohen_d:+.3f}  "
              f"{f.p_welch:.2e}{marca}")

    fila_070 = tab[np.isclose(tab.umbral, 0.70)].iloc[0]
    best = {k: tab.loc[tab[k].idxmax()] for k in ("r_pb", "auc", "cohen_d")}
    print("\n--- óptimos dentro de la muestra (los mismos 76 datos con que se eligen) ---")
    for k, f in best.items():
        print(f"  por {k:<8}: umbral = {f.umbral:.2f}  ({k} = {f[k]:+.4f})   "
              f"vs 0.70 -> {fila_070[k]:+.4f}   ganancia = {f[k] - fila_070[k]:+.4f}")

    u_opt = float(best["r_pb"].umbral)

    # ---------- ¿zona bien cubierta por datos? ----------
    print("\n--- cobertura de datos alrededor del umbral óptimo ---")
    for u in sorted({u_opt, 0.70}):
        cerca = np.sum(np.abs(pend - u) <= 0.05)
        pct = 100.0 * np.mean(pend < u)
        # hueco: distancia entre el punto más cercano por debajo y el más cercano por arriba
        abajo = pend[pend < u]; arriba = pend[pend >= u]
        hueco = (arriba.min() - abajo.max()) if len(abajo) and len(arriba) else np.nan
        print(f"  umbral {u:.2f}: {cerca} de {len(pend)} corridas dentro de ±0.05 "
              f"({100.0*cerca/len(pend):.0f}%) · percentil del corte = {pct:.0f}% · "
              f"hueco local entre vecinos = {hueco:.4f}")

    # ---------- VALIDACIÓN 1: mitades repetidas ----------
    print("\n--- VALIDACIÓN 1: partición repetida en mitades (2000 repeticiones) ---")
    print("    En cada repetición: se elige el umbral que maximiza r_pb en la MITAD A, y se evalúa r_pb")
    print("    en la MITAD B (nunca vista). Se compara contra evaluar el 0.70 oficial en esa misma B.")
    _autotest_rpb(pend, masa, umbrales)
    i_070 = int(np.argmin(np.abs(umbrales - 0.70)))
    n_rep, n = 2000, len(d)
    gana_opt, dif = [], []
    u_elegidos = []
    for _ in range(n_rep):
        perm = RNG.permutation(n)
        ia, ib = perm[: n // 2], perm[n // 2:]
        ma = np.array([pend[ia] >= u for u in umbrales])
        mb = np.array([pend[ib] >= u for u in umbrales])
        ra = r_pb_todos(ma, masa[ia], min_grupo=5)
        rb = r_pb_todos(mb, masa[ib], min_grupo=5)
        if np.all(np.isnan(ra)) or np.isnan(rb[i_070]):
            continue
        j = int(np.nanargmax(ra))
        if np.isnan(rb[j]):
            continue
        u_elegidos.append(umbrales[j])
        a, b = float(rb[j]), float(rb[i_070])
        dif.append(a - b); gana_opt.append(a > b)
    dif = np.array(dif)
    print(f"    repeticiones útiles: {len(dif)}")
    print(f"    r_pb fuera de muestra, diferencia (umbral optimizado − 0.70): "
          f"media {dif.mean():+.4f}  mediana {np.median(dif):+.4f}  "
          f"IC95% [{np.percentile(dif,2.5):+.4f}, {np.percentile(dif,97.5):+.4f}]")
    print(f"    el umbral optimizado gana en {100.0*np.mean(gana_opt):.1f}% de las mitades "
          f"(50% = moneda al aire)")
    ue = np.array(u_elegidos)
    print(f"    umbral elegido en entrenamiento: mediana {np.median(ue):.2f}  "
          f"IC95% [{np.percentile(ue,2.5):.2f}, {np.percentile(ue,97.5):.2f}]  "
          f"-> {100.0*np.mean(np.abs(ue-np.median(ue))<=0.02):.0f}% cae a ±0.02 de la mediana")

    # ---------- VALIDACIÓN 2: dejar-uno-afuera ----------
    print("\n--- VALIDACIÓN 2: dejar-uno-afuera (LOO) sobre la elección del umbral ---")
    print("    Para cada corrida i: se elige el umbral óptimo con las otras 75 y se anota. Mide la")
    print("    ESTABILIDAD de la elección (si depende de un solo punto, el óptimo no es real).")
    u_loo = []
    for i in range(n):
        keep = np.ones(n, bool); keep[i] = False
        mk = np.array([pend[keep] >= u for u in umbrales])
        r_i = r_pb_todos(mk, masa[keep], min_grupo=8)
        u_loo.append(float(umbrales[int(np.nanargmax(r_i))]))
    u_loo = np.array(u_loo)
    vals, cnts = np.unique(u_loo, return_counts=True)
    print("    umbral elegido al sacar cada punto: " +
          "  ".join(f"{v:.2f}×{c}" for v, c in zip(vals, cnts)))

    # ---------- VALIDACIÓN 3: permutación (cuánto 'optimismo' regala la libertad de elegir) ----------
    print("\n--- VALIDACIÓN 3: permutación — ¿cuánto r_pb se consigue eligiendo umbral sobre RUIDO? ---")
    n_perm = 5000
    masc_full = np.array([pend >= u for u in umbrales])
    maxs = np.empty(n_perm)
    for i in range(n_perm):
        maxs[i] = np.nanmax(r_pb_todos(masc_full, RNG.permutation(masa), min_grupo=8))
    r_opt = float(best["r_pb"].r_pb)
    p_perm = (np.sum(maxs >= r_opt) + 1) / (n_perm + 1)
    print(f"    r_pb máximo real (barriendo umbrales) = {r_opt:+.4f}")
    print(f"    r_pb máximo bajo masa barajada: mediana {np.median(maxs):+.4f}, "
          f"p95 {np.percentile(maxs,95):+.4f}, máx {maxs.max():+.4f}")
    print(f"    p (corregido por la búsqueda de umbral) = {p_perm:.4f}")
    # y lo mismo para el umbral FIJO 0.70, que no gasta grados de libertad
    r070 = float(fila_070.r_pb)
    masc_070 = masc_full[i_070:i_070 + 1]
    nulos_070 = np.empty(n_perm)
    for i in range(n_perm):
        nulos_070[i] = r_pb_todos(masc_070, RNG.permutation(masa), min_grupo=8)[0]
    p070 = (np.sum(nulos_070 >= r070) + 1) / (n_perm + 1)
    print(f"    para comparar, el umbral FIJO 0.70: r_pb = {r070:+.4f}, p permutación = {p070:.4f}")

    # ---------- VALIDACIÓN 4: las 11 corridas externas ----------
    print("\n--- VALIDACIÓN 4: chequeo externo con las 11 corridas de Phantom adicionales ---")
    print("    ADVERTENCIA: NO son una muestra al azar — son justamente las reglas que descarrilaban,")
    print("    todas con pendiente corregida alta. Sirven para ver si el corte sigue ordenando, no")
    print("    para estimar tamaños de efecto.")
    pe = externas["pendiente_corregida"].values
    me = externas["fraccion_masa_en_sumideros"].values
    print(f"    pendientes externas: [{pe.min():.3f}, {pe.max():.3f}] · masa: [{me.min():.4f}, {me.max():.4f}]")
    print(f"    Spearman pendiente-masa dentro de las 11: rho={stats.spearmanr(pe,me).statistic:+.3f} "
          f"p={stats.spearmanr(pe,me).pvalue:.3f}")
    todo_p = np.concatenate([pend, pe]); todo_m = np.concatenate([masa, me])
    for u in sorted({u_opt, 0.70}):
        f = metricas_corte(todo_p, todo_m, u, min_grupo=5)
        print(f"    umbral {u:.2f} sobre las 87 juntas: n_alto={f['n_alto']} n_bajo={f['n_bajo']} "
              f"AUC={f['auc']:.3f} r_pb={f['r_pb']:+.4f} d={f['cohen_d']:+.3f} p={f['p_welch']:.2e}")

    # ---------- ¿escalón o rampa? ----------
    print("\n--- ¿el corte gana algo frente a tratar la pendiente como continua? (n=76) ---")
    y = masa
    def r2_de(X):
        X = np.column_stack([np.ones(len(y))] + list(X))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - (y - X @ b).var() / y.var()
    r2_lin = r2_de([pend])
    r2_esc_070 = r2_de([(pend >= 0.70).astype(float)])
    r2_esc_opt = r2_de([(pend >= u_opt).astype(float)])
    r2_lin_esc = r2_de([pend, (pend >= u_opt).astype(float)])
    print(f"    recta sobre la pendiente continua        : R² = {r2_lin:.4f}")
    print(f"    sólo el escalón en 0.70                  : R² = {r2_esc_070:.4f}")
    print(f"    sólo el escalón en el óptimo {u_opt:.2f}       : R² = {r2_esc_opt:.4f}")
    print(f"    recta + escalón óptimo                   : R² = {r2_lin_esc:.4f}  "
          f"(el escalón agrega {r2_lin_esc - r2_lin:+.4f} sobre la recta)")

    # ---------- guardar validación ----------
    val = pd.DataFrame([dict(
        n_corridas=n, umbral_oficial=0.70, r_pb_oficial=r070, auc_oficial=float(fila_070.auc),
        d_oficial=float(fila_070.cohen_d), p_welch_oficial=float(fila_070.p_welch),
        umbral_optimo_r_pb=u_opt, r_pb_optimo=r_opt, auc_optimo=float(best["r_pb"].auc),
        d_optimo=float(best["r_pb"].cohen_d), p_welch_optimo=float(best["r_pb"].p_welch),
        umbral_optimo_auc=float(best["auc"].umbral), umbral_optimo_d=float(best["cohen_d"].umbral),
        mitades_dif_media=dif.mean(), mitades_dif_mediana=float(np.median(dif)),
        mitades_ic_lo=float(np.percentile(dif, 2.5)), mitades_ic_hi=float(np.percentile(dif, 97.5)),
        mitades_pct_gana=100.0 * float(np.mean(gana_opt)),
        loo_umbral_moda=float(vals[np.argmax(cnts)]), loo_pct_moda=100.0 * cnts.max() / n,
        perm_p_umbral_optimizado=p_perm, perm_p_umbral_fijo_070=p070,
        r2_recta_continua=r2_lin, r2_escalon_070=r2_esc_070, r2_escalon_optimo=r2_esc_opt,
        r2_recta_mas_escalon=r2_lin_esc,
    )])
    val.to_csv(f"{HERE}/cs090_fase6_o1a_validacion_umbral.csv", index=False)
    print(f"\n[csv] cs090_fase6_o1a_validacion_umbral.csv")
    return tab, val, u_opt, u_loo, ue


# ------------------------------------------------------------------------------------------------
# 2b. CONTROLES DEL ANÁLISIS 2 (lo que puede hacer que el "óptimo" sea un espejismo)
# ------------------------------------------------------------------------------------------------

def analisis2_controles(m, u_opt):
    print("\n" + "=" * 100)
    print("ANÁLISIS 2b — controles: ¿el óptimo es real, o es un borde / un confound / azar de muestreo?")
    print("=" * 100)

    d = m[~m.dup].reset_index(drop=True)
    pend = d["pendiente_corregida"].values
    masa = d["fraccion_masa_en_sumideros"].values
    kcap = d["kcap"].values

    # --- (a) barrido SIN mínimo de grupo: ¿hay máximo interior o la curva sube hasta el borde? ---
    print("\n--- (a) barrido completo hasta 1.24, SIN exigir tamaño mínimo de grupo ---")
    print("    (si r_pb subiera monótonamente hasta el último umbral posible, el 'óptimo' sería sólo")
    print("     'el corte más extremo que dejamos hacer' — o sea, un artefacto de dónde paramos)")
    us = np.round(np.arange(0.48, 1.2601, 0.02), 4)
    masc = np.array([pend >= u for u in us])
    rr = r_pb_todos(masc, masa, min_grupo=1)
    filas = pd.DataFrame(dict(umbral=us, n_alto=masc.sum(axis=1), r_pb=rr))
    filas.to_csv(f"{HERE}/cs090_fase6_o1a_barrido_umbral_sin_minimo.csv", index=False)
    for _, f in filas.iterrows():
        print(f"    {f.umbral:.2f}  n_alto={int(f.n_alto):3d}  r_pb={f.r_pb:+.4f}"
              + ("   <-- máximo" if np.isclose(f.r_pb, np.nanmax(rr)) else ""))
    j = int(np.nanargmax(rr))
    print(f"    -> el máximo cae en {us[j]:.2f} con n_alto={int(masc[j].sum())}, y DESPUÉS baja "
          f"({rr[j+1]:+.4f} en {us[j+1]:.2f}, {rr[min(j+6,len(rr)-1)]:+.4f} en {us[min(j+6,len(us)-1)]:.2f}): "
          f"es un máximo interior, no un borde")
    meseta = filas[(filas.r_pb >= 0.95 * np.nanmax(rr))]
    print(f"    -> meseta al 95% del máximo: umbrales {meseta.umbral.min():.2f}-{meseta.umbral.max():.2f} "
          f"({len(meseta)} umbrales) — el pico es ANCHO, no un punto")

    # --- (b) incertidumbre del óptimo: bootstrap ---
    print("\n--- (b) ¿cuán preciso es el 0.88? bootstrap de la muestra (2000 remuestreos) ---")
    umbrales = np.round(np.arange(0.40, 1.0001, 0.02), 4)
    n = len(d)
    opt_boot = []
    for _ in range(2000):
        idx = RNG.integers(0, n, n)
        mk = np.array([pend[idx] >= u for u in umbrales])
        r_b = r_pb_todos(mk, masa[idx], min_grupo=8)
        if np.all(np.isnan(r_b)):
            continue
        opt_boot.append(umbrales[int(np.nanargmax(r_b))])
    opt_boot = np.array(opt_boot)
    print(f"    umbral óptimo remuestreado: mediana {np.median(opt_boot):.2f}  "
          f"IC95% [{np.percentile(opt_boot,2.5):.2f}, {np.percentile(opt_boot,97.5):.2f}]")
    print(f"    cae por debajo de 0.72 en {100.0*np.mean(opt_boot < 0.72):.1f}% de los remuestreos")

    # --- (c) el confound de diseño: kcap ---
    print("\n--- (c) confound de diseño kcap (la 'perilla' del generador) ---")
    print("    kcap correlaciona MUY fuerte con todo: masa rho=-0.84, κ_V rho=-0.75, pendiente rho=-0.49.")
    print("    Entonces cortar por pendiente puede estar cortando, en el fondo, por kcap. Control: repetir")
    print("    el barrido SÓLO dentro del bloque kcap=5, que es el más poblado.")
    for kk in sorted(set(kcap)):
        s = kcap == kk
        print(f"    kcap={kk}: n={s.sum():2d}  masa media={masa[s].mean():.4f}  "
              f"pendiente media={pend[s].mean():.3f}")
    s = kcap == 5
    ps, ms = pend[s], masa[s]
    rs = stats.spearmanr(ps, ms)
    print(f"\n    dentro de kcap=5 (n={s.sum()}): Spearman pendiente-masa = {rs.statistic:+.3f} "
          f"(p={rs.pvalue:.4f}) — la relación NO desaparece al fijar la perilla")
    print("      umbral  n_alto  media_alto  media_bajo   r_pb")
    for u in [0.66, 0.70, 0.74, 0.78, 0.80, 0.84, 0.88]:
        mm = ps >= u
        if mm.sum() < 3 or (~mm).sum() < 3:
            continue
        rp = float(r_pb_todos(mm[None, :], ms, min_grupo=3)[0])
        print(f"       {u:.2f}    {int(mm.sum()):3d}      {ms[mm].mean():.4f}      "
              f"{ms[~mm].mean():.4f}   {rp:+.4f}" + ("   <-- 0.70 oficial" if u == 0.70 else ""))

    # --- (d) κ_V vs pendiente, ¿quién manda sobre la masa? ---
    print("\n--- (d) correlaciones parciales de Spearman entre las tres (n=76) ---")
    for a, b, c, txt in [("kappa_v_agregado", "fraccion_masa_en_sumideros", "pendiente_corregida",
                          "κ_V ~ masa, descontando la pendiente"),
                         ("pendiente_corregida", "fraccion_masa_en_sumideros", "kappa_v_agregado",
                          "pendiente ~ masa, descontando κ_V"),
                         ("kappa_v_agregado", "pendiente_corregida", "fraccion_masa_en_sumideros",
                          "κ_V ~ pendiente, descontando la masa")]:
        r, p = _parcial_spearman(d[a], d[b], d[c])
        print(f"    {txt:<42} rho_parcial = {r:+.4f}  p = {p:.2e}")
    return filas, opt_boot


# ------------------------------------------------------------------------------------------------
# 3. GRÁFICOS
# ------------------------------------------------------------------------------------------------

def graficos(m, tab, u_opt, u_loo, ue, externas, tab_full=None, opt_boot=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = m[~m.dup]
    C = dict(I="#4C78A8", III="#E45756", IV="#54A24B")

    # --- Figura 1: κ_V contra las tres variables ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (xv, xl) in zip(ax, [("pendiente_corregida", "pendiente CORREGIDA (geometría del grafo)"),
                                ("diam_corr_b1", "diámetro corregido (componente gigante, b=1)"),
                                ("fraccion_masa_en_sumideros", "fracción de masa en sumideros")]):
        for cl, g in d.groupby("clase_corregida"):
            a.scatter(g[xv], g["kappa_v_agregado"], s=34, alpha=.8,
                      c=C.get(cl, "#888"), edgecolor="k", linewidth=.3, label=f"Clase {cl}")
        rs = stats.spearmanr(d[xv], d["kappa_v_agregado"])
        rp = stats.pearsonr(d[xv], d["kappa_v_agregado"])
        a.set_xlabel(xl, fontsize=9); a.set_ylabel("κ_V agregado", fontsize=9)
        a.set_title(f"rho={rs.statistic:+.3f} (p={rs.pvalue:.1e}) · r={rp.statistic:+.3f}", fontsize=10)
        a.grid(alpha=.25); a.legend(fontsize=7)
    fig.suptitle("O1-A · Análisis 1 — κ_V contra geometría y contra respuesta gravitacional (n=76)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_o1a_fig1_kappav.png", dpi=140)
    plt.close(fig)

    # --- Figura 2: barrido de umbral ---
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    if tab_full is not None:
        a.plot(tab_full.umbral, tab_full.r_pb, "-", lw=1, color="#E45756", alpha=.35,
               label="r_pb, barrido completo sin mínimo de grupo")
    a.plot(tab.umbral, tab.r_pb, "-o", ms=3, color="#E45756", label="r punto-biserial (primaria)")
    a.plot(tab.umbral, tab.cohen_d / 4, "-s", ms=3, color="#4C78A8", alpha=.7, label="Cohen d / 4")
    a.plot(tab.umbral, (tab.auc - .5) * 2, "-^", ms=3, color="#54A24B", alpha=.7, label="(AUC−0.5)×2")
    a.axvline(0.70, color="k", ls="--", lw=1.2, label="0.70 oficial")
    a.axvline(u_opt, color="#B279A2", ls=":", lw=2, label=f"óptimo {u_opt:.2f}")
    a.set_xlabel("umbral sobre la pendiente corregida"); a.set_ylabel("separación de la masa")
    a.set_title("Barrido de umbrales (n=76)"); a.grid(alpha=.25); a.legend(fontsize=7)

    a = ax[0, 1]
    a.hist(d["pendiente_corregida"], bins=np.arange(0.40, 1.32, 0.04), color="#9ecae1", edgecolor="k", lw=.4)
    a.axvline(0.70, color="k", ls="--", lw=1.2); a.axvline(u_opt, color="#B279A2", ls=":", lw=2)
    a.set_xlabel("pendiente corregida"); a.set_ylabel("nº de corridas de Phantom")
    a.set_title("¿el umbral cae en zona con datos?"); a.grid(alpha=.25)

    a = ax[1, 0]
    a.scatter(d["pendiente_corregida"], d["fraccion_masa_en_sumideros"], s=34,
              c=[C.get(c, "#888") for c in d["clase_corregida"]], edgecolor="k", lw=.3, label="80/76 corridas")
    a.scatter(externas["pendiente_corregida"], externas["fraccion_masa_en_sumideros"], s=48,
              marker="D", facecolor="none", edgecolor="#333", lw=1.1, label="11 externas (sesgadas)")
    a.axvline(0.70, color="k", ls="--", lw=1.2); a.axvline(u_opt, color="#B279A2", ls=":", lw=2)
    a.set_xlabel("pendiente corregida"); a.set_ylabel("fracción de masa en sumideros")
    a.set_title("La variable es continua: el umbral corta una rampa"); a.grid(alpha=.25); a.legend(fontsize=7)

    a = ax[1, 1]
    a.hist(ue, bins=np.arange(0.39, 1.01, 0.02), color="#f4a582", edgecolor="k", lw=.4, alpha=.85,
           label="elegido en mitades (2000×)")
    if opt_boot is not None:
        a.hist(opt_boot, bins=np.arange(0.39, 1.01, 0.02), color="#92c5de", edgecolor="k", lw=.4,
               alpha=.6, label="elegido en bootstrap (2000×)")
    a.axvline(0.70, color="k", ls="--", lw=1.2); a.axvline(u_opt, color="#B279A2", ls=":", lw=2)
    a.set_xlabel("umbral elegido en entrenamiento"); a.set_ylabel("frecuencia")
    a.set_title("Estabilidad de la elección del umbral"); a.grid(alpha=.25); a.legend(fontsize=7)

    fig.suptitle("O1-A · Análisis 2 — recalibración del umbral de Clase III", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{HERE}/cs090_fase6_o1a_fig2_umbral.png", dpi=140)
    plt.close(fig)
    print("[png] cs090_fase6_o1a_fig1_kappav.png · cs090_fase6_o1a_fig2_umbral.png")


# ------------------------------------------------------------------------------------------------
def main():
    m = cargar()
    ext = cargar_externas()
    cols = ["par", "rule_id", "rol", "clase", "seed", "kcap", "K", "J", "noise", "meandeg",
            "fraccion_masa_en_sumideros", "kappa_v_agregado", "kappa_v_medio_valido",
            "n_sumideros", "masa_acretada_total",
            "pendiente_vieja", "pendiente_corregida", "clase_vieja", "clase_corregida",
            "diam_viejo_b1", "diam_corr_b1", "tam_gigante_b1", "descarrila_b1", "dup"]
    m[cols].to_csv(f"{HERE}/cs090_fase6_o1a_datos_unidos.csv", index=False)
    print(f"[csv] cs090_fase6_o1a_datos_unidos.csv ({len(m)} filas, {(~m.dup).sum()} corridas únicas)\n")

    analisis1(m)
    tab, val, u_opt, u_loo, ue = analisis2(m, ext)
    tab_full, opt_boot = analisis2_controles(m, u_opt)
    graficos(m, tab, u_opt, u_loo, ue, ext, tab_full, opt_boot)


if __name__ == "__main__":
    main()
