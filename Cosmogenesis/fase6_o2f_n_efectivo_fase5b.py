#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 6 · O2-F — N efectivo real de los 40 pares de Fase V-B
===========================================================

QUÉ HACE ESTE SCRIPT (en una frase)
-----------------------------------
Toma los 40 pares ya corridos de Fase V-B (`cs090_fase5b_TOTAL_40pares.csv`), reconstruye de
qué "lote de semillas" y de qué familia de parámetros viene cada regla, mide cuánto se parecen
entre sí los pares que comparten lote/parámetro (la ICC), traduce ese parecido a un "N efectivo"
(deff) y vuelve a calcular la significancia del resultado principal (III acreta más masa que I)
de tres maneras distintas: cruda, ajustada por deff, y con tests que respetan el agrupamiento.

POR QUÉ IMPORTA (analogía)
--------------------------
Si mido la altura de 40 personas pero resulta que son 4 familias de 10 hermanos, no tengo 40
medidas independientes: los hermanos se parecen. Mi "N efectivo" es menor que 40. La ICC mide
cuánto se parecen los hermanos; el deff traduce ese parecido a "cuántas medidas realmente
independientes tengo". Esta tarea pregunta: ¿los 40 pares son 40 personas sueltas o son
hermanos disfrazados?

ESTRUCTURA DEL SCRIPT
---------------------
 §1  Carga y reconstrucción de la estructura de agrupamiento de las 80 filas / 40 pares.
 §2  Construcción de las diferencias pareadas Δ (la unidad que alimenta signos y Wilcoxon).
 §3  ICC y deff por cada unidad de agrupamiento candidata (lote, kcap, K, tanda, componente
     de reglas compartidas), sobre Δ y también sobre los 80 valores individuales.
 §4  Recálculo de la significancia:
       (a) crudo (como está publicado)
       (b) inflando SE / bajando gl según deff
       (c) tests que respetan el agrupamiento: sign-flip por conglomerado entero,
           bootstrap de conglomerados, test sobre las medias de conglomerado,
           y permutación estratificada dentro de conglomerado.
 §5  El chequeo pedido explícitamente: ¿kcap, que está balanceado DENTRO del par, es fuente de
     pseudorreplicación para las DIFERENCIAS? Se mide, no se asume.
 §6  Volcado de CSVs y resumen por consola.

NO MODIFICA NINGÚN ARCHIVO EXISTENTE. Sólo lee el CSV de entrada y escribe CSVs nuevos.
"""

import csv
import math
import os
from collections import defaultdict, Counter

import numpy as np
from scipy import stats

BASE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
ENTRADA = os.path.join(BASE, "cs090_fase5b_TOTAL_40pares.csv")
SAL_ESTRUCTURA = os.path.join(BASE, "FASE6_O2F_estructura_agrupamiento.csv")
SAL_RESULTADOS = os.path.join(BASE, "FASE6_O2F_resultados_neff.csv")

RNG = np.random.default_rng(20260811)
N_PERM = 20000
N_BOOT = 20000

# Los tres pares que la corrección de diámetro (FASE6_adopcion_diam_corregido_CS.md) dejó sin
# contraste válido: la regla que hacía de "I" se re-etiquetó y dejó de ser Clase I.
REGLAS_REETIQUETADAS = {
    "A2-B0-C2-batch3-r100",   # I -> III  (contraste roto: III vs III)
    "A2-B0-C2-batch4-r51",    # I -> III  (contraste roto: III vs III)
    "A2-B0-C2-batch3-r143",   # I -> IV   (contraste invertido: IV vs III)
}

# Lotes de semillas conocidos de la línea Fase V-B: cada tanda generó candidatas con un
# seed_base distinto y las semillas individuales salen de seed_base + intento*97 + 1.
SEED_BASES = [271828, 371828, 471828, 571828]
ANCHO_LOTE = 30000  # holgura: el rango real de una tanda es << 30000


# ---------------------------------------------------------------------------------------
# §1 — Carga y reconstrucción de la estructura de agrupamiento
# ---------------------------------------------------------------------------------------

def lote_de_seed(seed):
    """Devuelve el seed_base (lote de generación) al que pertenece una semilla individual."""
    s = int(seed)
    for b in SEED_BASES:
        if b < s <= b + ANCHO_LOTE:
            return b
    return -1


def cargar():
    with open(ENTRADA, newline="", encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for r in filas:
        r["seed"] = int(r["seed"])
        r["K"] = int(r["K"])
        r["kcap"] = int(r["kcap"])
        r["frac"] = float(r["fraccion_masa_en_sumideros"])
        r["masa"] = float(r["masa_acretada_total"])
        r["kv"] = float(r["kappa_v_agregado"]) if r["kappa_v_agregado"] not in ("", "nan") else float("nan")
        r["lote"] = lote_de_seed(r["seed"])
    return filas


# ---------------------------------------------------------------------------------------
# §2 — Diferencias pareadas
# ---------------------------------------------------------------------------------------

def construir_pares(filas):
    """Un registro por par, con Δ = valor(III) − valor(I) y las etiquetas de agrupamiento."""
    porpar = defaultdict(dict)
    for r in filas:
        porpar[r["par"]][r["rol"]] = r

    pares = []
    for nombre, d in porpar.items():
        assert set(d) == {"I", "III"}, (nombre, list(d))
        a, b = d["I"], d["III"]
        pares.append({
            "par": nombre,
            "rule_I": a["rule_id"], "rule_III": b["rule_id"],
            "seed_I": a["seed"], "seed_III": b["seed"],
            "lote_I": a["lote"], "lote_III": b["lote"],
            # el lote del par: si ambos coinciden es ése; si no, se marca la mezcla
            "lote_par": a["lote"] if a["lote"] == b["lote"] else -2,
            "kcap_I": a["kcap"], "kcap_III": b["kcap"],
            "kcap_par": a["kcap"] if a["kcap"] == b["kcap"] else -2,
            "K_I": a["K"], "K_III": b["K"],
            "K_par": a["K"] if a["K"] == b["K"] else -2,
            "tanda": a["origen_tarea"],
            "frac_I": a["frac"], "frac_III": b["frac"],
            "d_frac": b["frac"] - a["frac"],
            "masa_I": a["masa"], "masa_III": b["masa"],
            "d_masa": b["masa"] - a["masa"],
            "kv_I": a["kv"], "kv_III": b["kv"],
            "d_kv": b["kv"] - a["kv"],
            "valido_diam": (a["rule_id"] not in REGLAS_REETIQUETADAS
                            and b["rule_id"] not in REGLAS_REETIQUETADAS),
        })
    pares.sort(key=lambda p: p["par"])
    return pares


def componentes_por_regla_compartida(pares):
    """
    Cuatro reglas se reutilizaron en dos pares distintos (76 corridas para 80 filas). Dos pares
    que comparten una corrida NO son independientes entre sí. Esto agrupa los pares en
    componentes conexas del grafo "comparten al menos una regla".
    Analogía: si dos encuestas comparten la mitad de los encuestados, no son dos encuestas.
    """
    padre = {p["par"]: p["par"] for p in pares}

    def find(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            padre[ra] = rb

    por_regla = defaultdict(list)
    for p in pares:
        por_regla[p["rule_I"]].append(p["par"])
        por_regla[p["rule_III"]].append(p["par"])
    for regla, lista in por_regla.items():
        for otro in lista[1:]:
            union(lista[0], otro)

    for p in pares:
        p["componente_regla"] = find(p["par"])
    return pares


# ---------------------------------------------------------------------------------------
# §3 — ICC (one-way random effects, diseño desbalanceado) y deff
# ---------------------------------------------------------------------------------------

def icc_una_via(valores, grupos):
    """
    ICC de un ANOVA de una vía con efectos aleatorios, para conglomerados de tamaño desigual.

    ICC = (MSB − MSW) / (MSB + (n0 − 1)·MSW)
    con n0 = (N − Σ n_g² / N) / (G − 1)   (tamaño de conglomerado 'efectivo' de Snedecor)

    Interpretación simple: qué fracción de la variación total se explica por "a qué grupo
    pertenece la medida" en vez de por diferencias dentro del grupo. 0 = los grupos no importan.
    """
    v = np.asarray(valores, dtype=float)
    g = np.asarray(grupos)
    niveles = np.unique(g)
    G, N = len(niveles), len(v)
    if G < 2 or N <= G:
        return dict(icc=float("nan"), deff=float("nan"), n_eff=float("nan"),
                    G=G, N=N, n0=float("nan"), m_medio=float("nan"),
                    MSB=float("nan"), MSW=float("nan"), F=float("nan"), p_F=float("nan"))
    gran = v.mean()
    ssb = ssw = 0.0
    ns = []
    for lv in niveles:
        sel = v[g == lv]
        ns.append(len(sel))
        ssb += len(sel) * (sel.mean() - gran) ** 2
        ssw += ((sel - sel.mean()) ** 2).sum()
    ns = np.array(ns, dtype=float)
    msb = ssb / (G - 1)
    msw = ssw / (N - G)
    n0 = (N - (ns ** 2).sum() / N) / (G - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) != 0 else 0.0
    icc_trunc = max(icc, 0.0)          # ICC negativa = "menos parecido que el azar" -> se trunca a 0
    m_medio = N / G
    deff = 1.0 + (m_medio - 1.0) * icc_trunc
    F = msb / msw if msw > 0 else float("inf")
    p_F = float(stats.f.sf(F, G - 1, N - G)) if msw > 0 else 0.0
    return dict(icc=icc, icc_trunc=icc_trunc, deff=deff, n_eff=N / deff, G=G, N=N,
                n0=n0, m_medio=m_medio, MSB=msb, MSW=msw, F=F, p_F=p_F)


def _resumen_grupos(v, codigos, G):
    """n_g, suma y suma de cuadrados por grupo — la materia prima de SSB/SSW."""
    n = np.bincount(codigos, minlength=G).astype(float)
    S = np.bincount(codigos, weights=v, minlength=G)
    Q = np.bincount(codigos, weights=v * v, minlength=G)
    return n, S, Q


def _icc_desde_resumen(n, S, Q):
    """ICC de una vía a partir de (n_g, ΣX_g, ΣX²_g). Versión vectorizable del cálculo."""
    ok = n > 0
    n, S, Q = n[ok], S[ok], Q[ok]
    G = len(n)
    N = n.sum()
    if G < 2 or N <= G:
        return float("nan")
    ssb = (S ** 2 / n).sum() - S.sum() ** 2 / N
    ssw = Q.sum() - (S ** 2 / n).sum()
    msb = ssb / (G - 1)
    msw = ssw / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den != 0 else 0.0


def icc_permutacion(valores, grupos, n_perm=4000):
    """p-valor de permutación para la ICC: se barajan las etiquetas de grupo."""
    v = np.asarray(valores, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G = len(niveles)
    obs = _icc_desde_resumen(*_resumen_grupos(v, cod, G))
    cnt = 0
    c = cod.copy()
    for _ in range(n_perm):
        RNG.shuffle(c)
        if _icc_desde_resumen(*_resumen_grupos(v, c, G)) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), obs


def icc_boot_ic(valores, grupos, n_boot=4000):
    """IC 95 % de la ICC por bootstrap de conglomerados enteros (vectorizado por resúmenes)."""
    v = np.asarray(valores, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G = len(niveles)
    n_g, S_g, Q_g = _resumen_grupos(v, cod, G)
    out = []
    for _ in range(n_boot):
        el = RNG.integers(0, G, size=G)
        # al remuestrear grupos enteros, cada copia sigue siendo un grupo aparte
        r = _icc_desde_resumen(n_g[el], S_g[el], Q_g[el])
        if np.isfinite(r):
            out.append(r)
    if not out:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(out, [2.5, 97.5]))


# ---------------------------------------------------------------------------------------
# §4 — Tests
# ---------------------------------------------------------------------------------------

def test_signos(d):
    """Test de signos bilateral: ¿cuántas Δ son > 0?"""
    d = np.asarray(d, float)
    d = d[d != 0]
    k = int((d > 0).sum())
    n = len(d)
    p = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return k, n, p


def wilcoxon(d):
    r = stats.wilcoxon(np.asarray(d, float), alternative="two-sided", zero_method="wilcox")
    return float(r.statistic), float(r.pvalue)


def z_de_p(p):
    """z bilateral equivalente a un p (para poder deflactar el z por sqrt(deff))."""
    p = min(max(p, 1e-300), 1.0)
    return float(stats.norm.isf(p / 2.0))


def p_de_z(z):
    return float(2 * stats.norm.sf(abs(z)))


def t_una_muestra(d, deff=1.0, gl=None):
    """t de una muestra sobre Δ, con la opción de inflar el SE por sqrt(deff) y bajar los gl."""
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    se_aj = se * math.sqrt(deff)
    gl_ef = (n / deff - 1) if gl is None else gl
    t = m / se_aj
    p = float(2 * stats.t.sf(abs(t), max(gl_ef, 1)))
    return dict(media=m, se=se, se_aj=se_aj, t=t, gl=gl_ef, p=p)


def signflip_por_conglomerado(d, grupos, exhaustivo_hasta=16):
    """
    Randomización que respeta el agrupamiento: en vez de invertir el signo de cada Δ por
    separado (que asume 40 monedas independientes), se invierte el signo de TODAS las Δ de un
    conglomerado a la vez (G monedas). Es el test más conservador posible bajo la hipótesis de
    que el conglomerado entero es la unidad independiente.
    Con G pequeño el p mínimo alcanzable es 2/2^G — se reporta explícitamente.
    """
    d = np.asarray(d, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G = len(niveles)
    N = len(d)
    # suma de Δ dentro de cada conglomerado: con signos por conglomerado, la media global
    # permutada es simplemente (1/N)·Σ_g s_g·S_g  →  un producto punto, nada de bucles internos
    S = np.bincount(cod, weights=d, minlength=G)
    obs = abs(d.mean())
    if G <= exhaustivo_hasta:
        total = 2 ** G
        bits = np.arange(total)[:, None]
        signos = 1.0 - 2.0 * ((bits >> np.arange(G)[None, :]) & 1)
        medias = signos @ S / N
        cnt = int((np.abs(medias) >= obs - 1e-15).sum())
        return cnt / total, total, 2.0 / total
    signos = RNG.choice([-1.0, 1.0], size=(N_PERM, G))
    medias = signos @ S / N
    cnt = int((np.abs(medias) >= obs - 1e-15).sum())
    return (cnt + 1) / (N_PERM + 1), N_PERM, 1.0 / N_PERM


def signflip_dentro_de_conglomerado(d, grupos, n_perm=N_PERM):
    """
    Variante intermedia: se invierten signos individualmente pero el estadístico se centra
    dentro de cada conglomerado (media de medias de conglomerado). Mantiene la independencia
    par-a-par dentro del grupo pero no deja que un grupo grande domine.
    """
    d = np.asarray(d, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G = len(niveles)
    n_g = np.bincount(cod, minlength=G).astype(float)
    # media-de-medias = Σ_i w_i·d_i con w_i = 1/(G·n_{g(i)}): otra vez un producto punto
    w = 1.0 / (G * n_g[cod])
    obs = abs(float(w @ d))
    eps = RNG.choice([-1.0, 1.0], size=(n_perm, len(d)))
    stats_perm = np.abs((eps * d) @ w)
    cnt = int((stats_perm >= obs - 1e-15).sum())
    return (cnt + 1) / (n_perm + 1)


def bootstrap_conglomerados(d, grupos, n_boot=N_BOOT):
    """
    Bootstrap de conglomerados: se remuestrean grupos ENTEROS con reemplazo. Devuelve IC 95 %
    de la media de Δ y un p bilateral por inversión del IC (fracción de réplicas ≤ 0, x2).
    """
    d = np.asarray(d, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G = len(niveles)
    n_g = np.bincount(cod, minlength=G).astype(float)
    S_g = np.bincount(cod, weights=d, minlength=G)
    el = RNG.integers(0, G, size=(n_boot, G))
    medias = S_g[el].sum(axis=1) / n_g[el].sum(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    p = 2 * min((medias <= 0).mean(), (medias >= 0).mean())
    p = max(p, 1.0 / n_boot)
    return lo, hi, p, medias.mean()


def se_robusto_por_conglomerado(d, grupos):
    """
    Error estándar robusto a conglomerados (sándwich CR1) para la media de Δ, con gl = G−1.
    Analogía: en vez de contar 40 votos sueltos, cuenta cuántos "bloques" votan distinto entre sí.
    Devuelve además el deff EMPÍRICO = (SE_robusto / SE_ingenuo)², que es la medida directa de
    cuánta información se pierde por el agrupamiento — sin pasar por la ICC.
    """
    d = np.asarray(d, float)
    niveles, cod = np.unique(np.asarray(grupos), return_inverse=True)
    G, N = len(niveles), len(d)
    m = d.mean()
    resid = d - m
    sg = np.bincount(cod, weights=resid, minlength=G)     # suma de residuos por conglomerado
    correc = (G / (G - 1)) * ((N - 1) / max(N - 1, 1))     # corrección CR1 de muestra chica
    var = correc * (sg ** 2).sum() / (N ** 2)
    se_cr = math.sqrt(var)
    se_naive = d.std(ddof=1) / math.sqrt(N)
    t = m / se_cr if se_cr > 0 else float("inf")
    p = float(2 * stats.t.sf(abs(t), G - 1))
    return dict(media=m, se_naive=se_naive, se_cr=se_cr, t=t, gl=G - 1, p=p,
                deff_empirico=(se_cr / se_naive) ** 2 if se_naive > 0 else float("nan"),
                N_eff_empirico=N / ((se_cr / se_naive) ** 2) if se_naive > 0 else float("nan"))


def test_medias_de_conglomerado(d, grupos):
    """
    El test más brutal: colapsar cada conglomerado a UNA media y hacer un t de una muestra con
    G observaciones. Equivale a decir "sólo tengo G datos independientes".
    """
    d = np.asarray(d, float)
    g = np.asarray(grupos)
    niveles = np.unique(g)
    medias = np.array([d[g == lv].mean() for lv in niveles])
    if len(medias) < 2:
        return dict(G=len(medias), media=float(medias.mean()), t=float("nan"), p=float("nan"))
    t, p = stats.ttest_1samp(medias, 0.0)
    return dict(G=len(medias), media=float(medias.mean()), t=float(t), p=float(p),
                medias=[float(x) for x in medias], niveles=[str(x) for x in niveles])


# ---------------------------------------------------------------------------------------
# §6 — Ejecución
# ---------------------------------------------------------------------------------------

def bloque_agrupamiento(nombre, d, grupos, resultados, subconjunto):
    """Calcula ICC/deff/N_eff + los tests que respetan ese agrupamiento y lo guarda."""
    r = icc_una_via(d, grupos)
    ic_lo, ic_hi = icc_boot_ic(d, grupos)
    p_perm_icc, _ = icc_permutacion(d, grupos, n_perm=4000)

    k, n, p_signos = test_signos(d)
    W, p_wil = wilcoxon(d)

    # (b) ajuste por deff
    n_eff = r["n_eff"]
    k_eff = int(round(k / n * n_eff))
    n_eff_i = max(int(round(n_eff)), 2)
    k_eff = min(max(k_eff, 0), n_eff_i)
    p_signos_eff = float(stats.binomtest(k_eff, n_eff_i, 0.5, alternative="two-sided").pvalue)
    p_wil_eff = p_de_z(z_de_p(p_wil) / math.sqrt(max(r["deff"], 1.0)))
    tt = t_una_muestra(d, deff=max(r["deff"], 1.0))
    tt_G = t_una_muestra(d, deff=max(r["deff"], 1.0), gl=r["G"] - 1)

    # (c) tests que respetan el agrupamiento
    p_flip, n_flip, p_min_flip = signflip_por_conglomerado(d, grupos)
    p_flip_dentro = signflip_dentro_de_conglomerado(d, grupos)
    blo, bhi, p_boot, bmedia = bootstrap_conglomerados(d, grupos)
    tmc = test_medias_de_conglomerado(d, grupos)
    cr = se_robusto_por_conglomerado(d, grupos)

    resultados.append(dict(
        subconjunto=subconjunto, unidad_agrupamiento=nombre,
        G=r["G"], N=r["N"], m_medio=round(r["m_medio"], 3), n0=round(r["n0"], 3),
        ICC=round(r["icc"], 4), ICC_IC95_lo=round(ic_lo, 4), ICC_IC95_hi=round(ic_hi, 4),
        p_perm_ICC=round(p_perm_icc, 4),
        F_anova=round(r["F"], 3), p_F=round(r["p_F"], 4),
        deff=round(r["deff"], 4), N_eff=round(r["n_eff"], 2),
        signos_k=k, signos_n=n, p_signos_crudo=p_signos,
        p_signos_Neff=p_signos_eff, signos_k_eff=k_eff, signos_n_eff=n_eff_i,
        W=W, p_wilcoxon_crudo=p_wil, p_wilcoxon_Neff=p_wil_eff,
        t_deff=round(tt["t"], 3), gl_deff=round(tt["gl"], 2), p_t_deff=tt["p"],
        p_t_glG=tt_G["p"],
        p_signflip_conglomerado=p_flip, n_flips=n_flip, p_min_alcanzable_flip=p_min_flip,
        p_signflip_dentro=p_flip_dentro,
        boot_media=round(bmedia, 6), boot_IC95_lo=round(blo, 6), boot_IC95_hi=round(bhi, 6),
        p_boot_conglomerados=p_boot,
        t_medias_conglom=round(tmc["t"], 3) if np.isfinite(tmc["t"]) else "",
        p_medias_conglom=tmc["p"],
        se_naive=round(cr["se_naive"], 6), se_cluster_robusto=round(cr["se_cr"], 6),
        deff_empirico_CR=round(cr["deff_empirico"], 4),
        N_eff_empirico_CR=round(cr["N_eff_empirico"], 2),
        t_CR=round(cr["t"], 3), gl_CR=cr["gl"], p_CR=cr["p"],
    ))
    return r


def main():
    filas = cargar()
    pares = componentes_por_regla_compartida(construir_pares(filas))

    print("=" * 90)
    print("§1 — ESTRUCTURA DE AGRUPAMIENTO REAL DE LOS 80 DATOS")
    print("=" * 90)
    print(f"filas: {len(filas)} | pares: {len(pares)} | seeds distintos: {len(set(f['seed'] for f in filas))} "
          f"| rule_id distintos: {len(set(f['rule_id'] for f in filas))}")
    print("lotes (seed_base) a nivel de FILA :", dict(sorted(Counter(f['lote'] for f in filas).items())))
    print("lotes (seed_base) a nivel de PAR  :", dict(sorted(Counter(p['lote_par'] for p in pares).items())))
    print("kcap a nivel de FILA              :", dict(sorted(Counter(f['kcap'] for f in filas).items())))
    print("kcap a nivel de PAR               :", dict(sorted(Counter(p['kcap_par'] for p in pares).items())))
    print("K a nivel de PAR                  :", dict(sorted(Counter(p['K_par'] for p in pares).items())))
    print("tanda a nivel de PAR              :", dict(Counter(p['tanda'] for p in pares).items())
          if False else dict(Counter(p['tanda'] for p in pares)))
    print("pares con kcap idéntico en ambos roles:",
          sum(1 for p in pares if p['kcap_par'] != -2), "/", len(pares))
    print("pares con K idéntico en ambos roles   :",
          sum(1 for p in pares if p['K_par'] != -2), "/", len(pares))
    print("pares con lote idéntico en ambos roles:",
          sum(1 for p in pares if p['lote_par'] != -2), "/", len(pares))
    comp = Counter(p['componente_regla'] for p in pares)
    print(f"componentes por reglas compartidas: {len(comp)} (tamaños {sorted(comp.values(), reverse=True)[:6]}...)")
    print("prefijos de rule_id:", dict(Counter(f['rule_id'].rsplit('-r', 1)[0] for f in filas)))

    # volcado de la estructura
    campos = ["par", "rule_I", "rule_III", "seed_I", "seed_III", "lote_I", "lote_III", "lote_par",
              "kcap_I", "kcap_III", "kcap_par", "K_I", "K_III", "K_par", "tanda",
              "componente_regla", "frac_I", "frac_III", "d_frac", "masa_I", "masa_III", "d_masa",
              "kv_I", "kv_III", "d_kv", "valido_diam"]
    with open(SAL_ESTRUCTURA, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for p in pares:
            w.writerow({c: p[c] for c in campos})
    print("→", SAL_ESTRUCTURA)

    resultados = []

    for subconjunto, sel in [("40_pares_original", pares),
                             ("37_pares_validos_diam", [p for p in pares if p["valido_diam"]])]:
        d = np.array([p["d_frac"] for p in sel], float)
        print("\n" + "=" * 90)
        print(f"SUBCONJUNTO: {subconjunto}  (n = {len(sel)} pares)")
        print("=" * 90)
        k, n, p_s = test_signos(d)
        W, p_w = wilcoxon(d)
        print(f"CRUDO — signos: {k}/{n} (p = {p_s:.6g}) | Wilcoxon W = {W:.0f} (p = {p_w:.3g}) "
              f"| media Δ = {d.mean():.5f} | mediana Δ = {np.median(d):.5f}")

        grupos = {
            "lote_seed_base": [p["lote_par"] for p in sel],
            "kcap_del_par": [p["kcap_par"] for p in sel],
            "K_del_par": [p["K_par"] for p in sel],
            "tanda_generacion": [p["tanda"] for p in sel],
            "componente_reglas_compartidas": [p["componente_regla"] for p in sel],
            "lote_x_kcap": [f"{p['lote_par']}|{p['kcap_par']}" for p in sel],
            # variantes sin conglomerado artificial: los pocos pares con lote/kcap distinto entre
            # sus dos reglas se asignan al valor de la regla Clase III, en vez de formar un grupo
            # "mixto" propio (que sería un conglomerado de tamaño 1-2 y distorsiona la ICC)
            "lote_seed_base_sin_mixto": [p["lote_III"] for p in sel],
            "kcap_del_par_sin_mixto": [p["kcap_III"] for p in sel],
        }
        for nombre, g in grupos.items():
            r = bloque_agrupamiento(nombre, d, g, resultados, subconjunto)
            fila = resultados[-1]
            print(f"\n  [{nombre}]  G={r['G']}  m̄={r['m_medio']:.2f}")
            print(f"    ICC = {r['icc']:+.4f}  IC95 [{fila['ICC_IC95_lo']}, {fila['ICC_IC95_hi']}]  "
                  f"p_perm = {fila['p_perm_ICC']}  | ANOVA F = {fila['F_anova']}, p = {fila['p_F']}")
            print(f"    deff = {r['deff']:.3f}   →   N_eff = {r['n_eff']:.1f} de {r['N']}")
            print(f"    signos ajustado : {fila['signos_k_eff']}/{fila['signos_n_eff']}  p = {fila['p_signos_Neff']:.4g}")
            print(f"    Wilcoxon ajust. : p = {fila['p_wilcoxon_Neff']:.4g}")
            print(f"    t con SE×√deff  : p = {fila['p_t_deff']:.4g} (gl = {fila['gl_deff']})   "
                  f"| gl = G−1: p = {fila['p_t_glG']:.4g}")
            print(f"    sign-flip por conglomerado ENTERO : p = {fila['p_signflip_conglomerado']:.4g} "
                  f"(mínimo alcanzable {fila['p_min_alcanzable_flip']:.4g}, {fila['n_flips']} flips)")
            print(f"    sign-flip centrado dentro de grupo: p = {fila['p_signflip_dentro']:.4g}")
            print(f"    bootstrap de conglomerados        : media {fila['boot_media']:.5f} "
                  f"IC95 [{fila['boot_IC95_lo']:.5f}, {fila['boot_IC95_hi']:.5f}]  p = {fila['p_boot_conglomerados']:.4g}")
            print(f"    t sobre las {r['G']} medias de conglomerado : p = {fila['p_medias_conglom']:.4g}")
            print(f"    SE cluster-robusto (CR1, gl={fila['gl_CR']}) : SE {fila['se_naive']:.6f} → "
                  f"{fila['se_cluster_robusto']:.6f}  deff_emp = {fila['deff_empirico_CR']}  "
                  f"N_eff_emp = {fila['N_eff_empirico_CR']}  p = {fila['p_CR']:.4g}")

    # -----------------------------------------------------------------------------------
    # §5 — ¿kcap contamina las DIFERENCIAS o sólo los NIVELES?
    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("§5 — kcap BALANCEADO DENTRO DEL PAR: ¿contamina la diferencia o sólo el nivel?")
    print("=" * 90)
    niveles = np.array([f["frac"] for f in filas], float)
    g_kcap_fila = [f["kcap"] for f in filas]
    g_lote_fila = [f["lote"] for f in filas]
    r_niv_kcap = icc_una_via(niveles, g_kcap_fila)
    r_niv_lote = icc_una_via(niveles, g_lote_fila)
    d40 = np.array([p["d_frac"] for p in pares], float)
    r_dif_kcap = icc_una_via(d40, [p["kcap_par"] for p in pares])
    r_dif_lote = icc_una_via(d40, [p["lote_par"] for p in pares])

    def eta2(valores, grupos):
        v = np.asarray(valores, float); g = np.asarray(grupos)
        gran = v.mean()
        ssb = sum(len(v[g == lv]) * (v[g == lv].mean() - gran) ** 2 for lv in np.unique(g))
        sst = ((v - gran) ** 2).sum()
        return ssb / sst if sst > 0 else float("nan")

    print(f"NIVELES (80 fracciones individuales):")
    print(f"  por kcap : ICC = {r_niv_kcap['icc']:+.4f}  η² = {eta2(niveles, g_kcap_fila):.4f}  "
          f"F = {r_niv_kcap['F']:.2f}  p = {r_niv_kcap['p_F']:.4g}")
    print(f"  por lote : ICC = {r_niv_lote['icc']:+.4f}  η² = {eta2(niveles, g_lote_fila):.4f}  "
          f"F = {r_niv_lote['F']:.2f}  p = {r_niv_lote['p_F']:.4g}")
    print(f"DIFERENCIAS pareadas (40 Δ):")
    print(f"  por kcap : ICC = {r_dif_kcap['icc']:+.4f}  η² = {eta2(d40, [p['kcap_par'] for p in pares]):.4f}  "
          f"F = {r_dif_kcap['F']:.2f}  p = {r_dif_kcap['p_F']:.4g}")
    print(f"  por lote : ICC = {r_dif_lote['icc']:+.4f}  η² = {eta2(d40, [p['lote_par'] for p in pares]):.4f}  "
          f"F = {r_dif_lote['F']:.2f}  p = {r_dif_lote['p_F']:.4g}")

    # medias de Δ y proporción de aciertos por nivel de kcap y de lote
    print("\n  Δ media y aciertos por kcap (pares):")
    for lv in sorted(set(p["kcap_par"] for p in pares)):
        sel = [p for p in pares if p["kcap_par"] == lv]
        dd = np.array([p["d_frac"] for p in sel])
        print(f"    kcap={lv:>3}  n={len(sel):>2}  Δ media={dd.mean():+.5f}  aciertos={(dd>0).sum()}/{len(dd)}")
    print("  Δ media y aciertos por lote (pares):")
    for lv in sorted(set(p["lote_par"] for p in pares)):
        sel = [p for p in pares if p["lote_par"] == lv]
        dd = np.array([p["d_frac"] for p in sel])
        print(f"    lote={lv:>7}  n={len(sel):>2}  Δ media={dd.mean():+.5f}  aciertos={(dd>0).sum()}/{len(dd)}")

    # -----------------------------------------------------------------------------------
    # §5ter — replicación LOTE POR LOTE y combinación de lotes independientes
    # -----------------------------------------------------------------------------------
    # La pregunta honesta cuando hay conglomerados no es sólo "¿cuánto pierdo de N?", sino
    # "¿el efecto aparece por separado en cada conglomerado?". Si sí, la dependencia interna
    # deja de ser una amenaza: son réplicas, no ecos. Cada lote (seed_base) es un sorteo
    # independiente de familias de reglas, así que combinar los lotes es legítimo.
    print("\n" + "=" * 90)
    print("§5ter — ¿SE REPLICA EN CADA LOTE POR SEPARADO? (y combinación de lotes)")
    print("=" * 90)
    for sub_nombre, sel in [("40 pares", pares),
                            ("37 válidos", [p for p in pares if p["valido_diam"]])]:
        ps, zs, pesos, lineas = [], [], [], []
        for lv in sorted(set(p["lote_par"] for p in sel)):
            if lv == -2:
                continue
            dd = np.array([p["d_frac"] for p in sel if p["lote_par"] == lv], float)
            k, n, p_s = test_signos(dd)
            ps.append(p_s)
            # z con signo, para Stouffer (positivo si el efecto va en la dirección esperada)
            z = z_de_p(p_s) * (1 if dd.mean() > 0 else -1)
            zs.append(z)
            pesos.append(math.sqrt(len(dd)))
            lineas.append(f"    lote {lv}: n={n:>2}  signos {k}/{n}  p={p_s:.4g}  "
                          f"Δ media={dd.mean():+.5f}  Wilcoxon p="
                          f"{(wilcoxon(dd)[1] if len(dd) > 1 else float('nan')):.4g}")
        print(f"  [{sub_nombre}]")
        for l in lineas:
            print(l)
        chi2 = -2 * sum(math.log(max(p, 1e-300)) for p in ps)
        p_fisher = float(stats.chi2.sf(chi2, 2 * len(ps)))
        z_st = sum(w * z for w, z in zip(pesos, zs)) / math.sqrt(sum(w * w for w in pesos))
        p_st = p_de_z(z_st)
        print(f"    → lotes con Δ media > 0: {sum(1 for z in zs if z > 0)}/{len(zs)}")
        print(f"    → Fisher (combina los {len(ps)} lotes): χ² = {chi2:.3f}, p = {p_fisher:.4g}")
        print(f"    → Stouffer ponderado por √n: z = {z_st:.3f}, p = {p_st:.4g}")
        resultados.append(dict(subconjunto=f"{sub_nombre}_combinacion_de_lotes",
                               unidad_agrupamiento="lote_seed_base", G=len(ps), N=len(sel),
                               p_fisher_lotes=p_fisher, z_stouffer=round(z_st, 3),
                               p_stouffer_lotes=p_st))

    # -----------------------------------------------------------------------------------
    # §5bis — contraste con la hipótesis de O2-B (que era un cálculo a mano, sin datos)
    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("§5bis — LA HIPÓTESIS DE O2-B ('2 lotes, m≈40, deff≈2.3') CONTRA LA ESTRUCTURA REAL")
    print("=" * 90)
    icc_real = icc_una_via(d40, [p["lote_par"] for p in pares])["icc"]
    icc_o2b = 0.033
    for etiqueta, G_h, m_h, icc_h in [
        ("hipótesis O2-B: 2 lotes de 40 reglas, ICC=0.033", 2, 40, icc_o2b),
        ("hipótesis O2-B con m en PARES (2 lotes x 20 pares)", 2, 20, icc_o2b),
        ("real: 4 lotes de pares, ICC de O2-B", 4, 40 / 4, icc_o2b),
        ("real: 4 lotes de pares, ICC medida acá", 4, 40 / 4, max(icc_real, 0.0)),
    ]:
        deff_h = 1 + (m_h - 1) * icc_h
        print(f"  {etiqueta:<52} G={G_h:>2}  m={m_h:>5.1f}  ICC={icc_h:.4f}  "
              f"deff={deff_h:.3f}  N_eff={40 / deff_h:.1f}")

    # -----------------------------------------------------------------------------------
    # secundarios: masa acretada total y kappa_v (para tener el cuadro completo)
    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("OBSERVABLES SECUNDARIOS (40 pares) — sólo cifra cruda y sign-flip por lote")
    print("=" * 90)
    for etiqueta, clave in [("masa_acretada_total", "d_masa"), ("kappa_v_agregado", "d_kv")]:
        dd = np.array([p[clave] for p in pares], float)
        ok = np.isfinite(dd)
        dd = dd[ok]
        gl = [p["lote_par"] for p, m in zip(pares, ok) if m]
        k, n, ps = test_signos(dd)
        W, pw = wilcoxon(dd)
        r = icc_una_via(dd, gl)
        pf, nf, pmin = signflip_por_conglomerado(dd, gl)
        print(f"  {etiqueta:>22}: signos {k}/{n} p={ps:.4g} | Wilcoxon p={pw:.4g} | "
              f"ICC_lote={r['icc']:+.4f} deff={r['deff']:.3f} N_eff={r['n_eff']:.1f} | "
              f"sign-flip x lote p={pf:.4g} (mín {pmin:.4g})")
        resultados.append(dict(subconjunto=f"40_pares_{etiqueta}", unidad_agrupamiento="lote_seed_base",
                               G=r["G"], N=r["N"], ICC=round(r["icc"], 4), deff=round(r["deff"], 4),
                               N_eff=round(r["n_eff"], 2), signos_k=k, signos_n=n,
                               p_signos_crudo=ps, W=W, p_wilcoxon_crudo=pw,
                               p_signflip_conglomerado=pf, p_min_alcanzable_flip=pmin))

    # volcado de resultados
    campos_r = []
    for r in resultados:
        for c in r:
            if c not in campos_r:
                campos_r.append(c)
    with open(SAL_RESULTADOS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos_r)
        w.writeheader()
        for r in resultados:
            w.writerow(r)
    print("\n→", SAL_RESULTADOS)


if __name__ == "__main__":
    main()
