#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase8_f805_analizar.py — F8-05: la comparación entre motores, par por par
================================================================================

QUÉ CONTESTA
------------
1. ¿El ordenamiento `solap` > `disj` de F7-03 (12/12 en Phantom) se repite en el motor
   independiente? Cuántos de los 12 pares.
2. ¿Cuánto correlaciona el observable de Phantom con el del motor independiente a través
   de las 24 corridas?
3. ¿Qué tamaño tiene el efecto en el motor independiente, comparado con el +13.8% de
   Phantom?
4. **El control que exige O4-A**: el mismo observable medido sobre las CONDICIONES
   INICIALES, sin integrar un solo paso. Si las IC ya separan `solap` de `disj` con la
   misma fuerza, lo que se está midiendo es geometría de partida, no dinámica.

Además, opcionalmente (`--vara-comun`), aplica la MISMA vara a las dos mesas: reconstruye
el estado final de Phantom como nube de masas puntuales (gas vivo + sumideros con la masa
que acretaron) y le corre el mismo friends-of-friends, con criterio de tamaño en MASA.
Eso cierra la objeción "el desacuerdo puede venir de usar dos reglas distintas".

Analogía de por qué el punto 4 importa: si dos jueces distintos coinciden en qué torta
subió más, pero resulta que una masa ya venía más inflada ANTES de entrar al horno,
entonces los jueces coinciden sobre la masa cruda, no sobre el horno.

USO
---
    ./venv/bin/python cs090_fase8_f805_analizar.py [--vara-comun]
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
BATERIA = "/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion"
M_PART = 9.4
ELL_PRINCIPAL = 1.0
NMIN_PRINCIPAL = 5
COL_PRINCIPAL = f"fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}"


def cargar():
    """Une las 24 corridas del motor propio con las 24 corridas de Phantom de F7-03.

    La unión es por (rule_id, seed, brazo) — NUNCA por rule_id solo: hay reglas distintas
    con el mismo rule_id en lotes distintos (bug documentado en FASE6_O3B §2.1).
    """
    nb = pd.read_csv(os.path.join(AQUI, "cs090_fase8_f805_corridas_nbody.csv"))
    ph = pd.read_csv(os.path.join(AQUI, "cs090_fase7_f703_phantom_crudo.csv"))
    ph = ph[ph.brazo.isin(["solap", "disj"])].copy()
    assert len(nb) == 24 and len(ph) == 24, (len(nb), len(ph))
    d = nb.merge(ph[["rule_id", "seed", "brazo", "frac_masa", "kappa_v", "n_sumideros",
                     "t_primer_sumidero", "n_triangulos", "n_aristas", "clustering_local",
                     "frac_aristas_multi_tri", "gini_tri_nodo", "tri_por_arista_media",
                     "frac_aristas_en_triangulo", "gigante"]],
                 on=["rule_id", "seed", "brazo"], how="inner",
                 validate="one_to_one")
    assert len(d) == 24, f"la unión perdió filas: {len(d)}"
    return d


def tabla_pares(d, columnas):
    """Una fila por grafo con el valor de `solap`, el de `disj` y su diferencia,
    para cada observable pedido."""
    filas = []
    for (rid, seed), g in d.groupby(["rule_id", "seed"], sort=True):
        s = g[g.brazo == "solap"].iloc[0]
        j = g[g.brazo == "disj"].iloc[0]
        fila = dict(rule_id=rid, seed=seed)
        if "n_triangulos" in g.columns:
            fila["n_triangulos"] = int(s.n_triangulos)
        for c in columnas:
            fila[f"{c}__solap"] = float(s[c])
            fila[f"{c}__disj"] = float(j[c])
            fila[f"{c}__d"] = float(s[c]) - float(j[c])
        filas.append(fila)
    return pd.DataFrame(filas)


def resumen_efecto(pares, col, etiqueta):
    """Signos, media, % relativo, Wilcoxon y t pareado para un observable."""
    d = pares[f"{col}__d"].to_numpy()
    base = pares[f"{col}__disj"].to_numpy()
    n = len(d)
    k = int((d > 0).sum())
    med = float(d.mean())
    rel = 100.0 * med / float(base.mean())
    try:
        w = stats.wilcoxon(d).pvalue
    except ValueError:
        w = np.nan
    t = stats.ttest_1samp(d, 0.0).pvalue
    binom = stats.binomtest(k, n, 0.5).pvalue
    return dict(observable=etiqueta, columna=col, n=n, signos_positivos=k,
                delta_medio=med, delta_mediano=float(np.median(d)),
                media_solap=float(pares[f"{col}__solap"].mean()),
                media_disj=float(base.mean()), pct_relativo=rel,
                p_wilcoxon=w, p_t_pareado=t, p_binomial_signos=binom)


def parcial(x, y, z):
    """Correlación parcial de Pearson entre x e y descontando z (residuos de la
    regresión lineal de cada uno sobre z)."""
    x, y, z = map(np.asarray, (x, y, z))
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p


def main(vara_comun=False):
    d = cargar()
    pd.set_option("display.width", 220)

    cols_nb_fin = ([c for c in d.columns if c.startswith("fin_fof_")]
                   + [c for c in d.columns if c.startswith("fin_knn8_")])
    cols_nb_ini = ([c for c in d.columns if c.startswith("ini_fof_")]
                   + [c for c in d.columns if c.startswith("ini_knn8_")])
    cols_ngr_fin = [c for c in d.columns if c.startswith("fin_ngrupos_")]

    print("=" * 90)
    print("SALUD NUMÉRICA DE LAS 24 CORRIDAS")
    print("=" * 90)
    print(f"deriva de energía   : {d.deriva_energia_rel.min():.3e} .. {d.deriva_energia_rel.max():.3e}")
    print(f"error rel. momento  : max {d.error_rel_momento.max():.3e}")
    print(f"virial final        : {d.virial_final.min():.3f} .. {d.virial_final.max():.3f}")
    print(f"md5 de IC distintos : {d.md5_ic.nunique()} de 24")
    print(f"segundos por corrida: {d.segundos.min():.0f} .. {d.segundos.max():.0f}")

    # ---------------------------------------------------------------------
    # 1. LA COMPARACIÓN CENTRAL: par por par
    # ---------------------------------------------------------------------
    principal_fin = "fin_" + COL_PRINCIPAL
    principal_ini = "ini_" + COL_PRINCIPAL
    pares = tabla_pares(d, ["frac_masa", principal_fin, principal_ini,
                            "kappa_v", "n_sumideros"] + cols_nb_fin + cols_nb_ini)
    pares["coincide_fin"] = np.sign(pares[f"{principal_fin}__d"]) == np.sign(pares["frac_masa__d"])
    pares["coincide_ini"] = np.sign(pares[f"{principal_ini}__d"]) == np.sign(pares["frac_masa__d"])

    print("\n" + "=" * 90)
    print(f"1. ORDEN `solap` vs `disj` — PHANTOM contra MOTOR INDEPENDIENTE  ({COL_PRINCIPAL})")
    print("=" * 90)
    vista = pares[["rule_id", "seed", "n_triangulos",
                   "frac_masa__solap", "frac_masa__disj", "frac_masa__d",
                   f"{principal_fin}__solap", f"{principal_fin}__disj", f"{principal_fin}__d",
                   f"{principal_ini}__d", "coincide_fin", "coincide_ini"]].copy()
    vista.columns = ["rule_id", "seed", "T*", "ph_solap", "ph_disj", "ph_d",
                     "nb_solap", "nb_disj", "nb_d", "ic_d", "coincide_fin", "coincide_ini"]
    print(vista.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    k_fin, k_ini, n = int(pares.coincide_fin.sum()), int(pares.coincide_ini.sum()), len(pares)
    print(f"\n  concordancia de orden Phantom ↔ motor independiente (t=0.5): {k_fin}/{n}  "
          f"binomial p={stats.binomtest(k_fin, n, 0.5).pvalue:.4f}")
    print(f"  concordancia de orden Phantom ↔ IC sin integrar   (t=0  ): {k_ini}/{n}  "
          f"binomial p={stats.binomtest(k_ini, n, 0.5).pvalue:.4f}")

    # concordancia en TODAS las definiciones de "región densa"
    print("\n  -- la misma cuenta en cada definición del observable --")
    filas_rob = []
    for c in cols_nb_fin:
        base = c[4:]
        pr = tabla_pares(d, ["frac_masa", c, "ini_" + base])
        kf = int((np.sign(pr[f"{c}__d"]) == np.sign(pr["frac_masa__d"])).sum())
        ki = int((np.sign(pr[f"ini_{base}__d"]) == np.sign(pr["frac_masa__d"])).sum())
        sf = int((pr[f"{c}__d"] > 0).sum())
        si = int((pr[f"ini_{base}__d"] > 0).sum())
        filas_rob.append(dict(observable=base, coincide_fin=kf, coincide_ini=ki,
                              signos_pos_fin=sf, signos_pos_ini=si,
                              d_medio_fin=pr[f"{c}__d"].mean(),
                              d_medio_ini=pr[f"ini_{base}__d"].mean()))
    rob = pd.DataFrame(filas_rob)
    # correlación DENTRO de grafo para cada definición (la variación entre brazos, que
    # es lo único que la intervención movió) — el equivalente "continuo" de la
    # concordancia de signos, sin binarizar
    dcen = d.copy()
    for c in ["frac_masa"] + cols_nb_fin + cols_nb_ini:
        dcen[c + "_c"] = dcen[c] - dcen.groupby(["rule_id", "seed"])[c].transform("mean")
    rr = []
    for _, f in rob.iterrows():
        base = f["observable"]
        r_fin = stats.pearsonr(dcen["frac_masa_c"], dcen["fin_" + base + "_c"])[0]
        r_ini = stats.pearsonr(dcen["frac_masa_c"], dcen["ini_" + base + "_c"])[0]
        rr.append((r_fin, r_ini))
    rob["r_dentro_grafo_fin"] = [x[0] for x in rr]
    rob["r_dentro_grafo_ini"] = [x[1] for x in rr]
    print(rob.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rob.to_csv(os.path.join(AQUI, "cs090_fase8_f805_robustez_grilla.csv"), index=False)

    # ¿cambia el NÚMERO de grumos, o sólo su tamaño? (en Phantom no cambia: 8.08 vs 8.08)
    print("\n  -- nº de grupos FoF en el motor independiente (Phantom: 8.08 vs 8.08 sumideros) --")
    prg = tabla_pares(d, cols_ngr_fin)
    for c in cols_ngr_fin:
        print(f"     {c[4:]:22s} solap {prg[c+'__solap'].mean():6.2f}  "
              f"disj {prg[c+'__disj'].mean():6.2f}  Δ={prg[c+'__d'].mean():+6.2f}  "
              f"signos {int((prg[c+'__d']>0).sum())}/12")

    # ---------------------------------------------------------------------
    # 2. CORRELACIÓN ENTRE MOTORES A TRAVÉS DE LAS 24 CORRIDAS
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("2. CORRELACIÓN ENTRE EL OBSERVABLE DE PHANTOM Y EL PROPIO (24 corridas)")
    print("=" * 90)
    filas_corr = []
    for c in cols_nb_fin + cols_nb_ini:
        rp, pp = stats.pearsonr(d.frac_masa, d[c])
        rs, ps = stats.spearmanr(d.frac_masa, d[c])
        filas_corr.append(dict(observable=c, pearson=rp, p_pearson=pp,
                               spearman=rs, p_spearman=ps))
    corr = pd.DataFrame(filas_corr).sort_values("pearson", ascending=False)
    print(corr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    corr.to_csv(os.path.join(AQUI, "cs090_fase8_f805_correlaciones.csv"), index=False)

    # la parcial que hace la pregunta de O4-A §5.3
    rp_fin, _ = stats.pearsonr(d.frac_masa, d[principal_fin])
    rp_ini, _ = stats.pearsonr(d.frac_masa, d[principal_ini])
    rp_nbini, _ = stats.pearsonr(d[principal_fin], d[principal_ini])
    r_par, p_par = parcial(d.frac_masa, d[principal_fin], d[principal_ini])
    print(f"\n  Phantom ↔ motor independiente (t=0.5) : Pearson {rp_fin:+.3f}")
    print(f"  Phantom ↔ IC sin integrar     (t=0  ) : Pearson {rp_ini:+.3f}")
    print(f"  motor independiente ↔ IC              : Pearson {rp_nbini:+.3f}")
    print(f"  PARCIAL Phantom ↔ motor, descontando la IC: {r_par:+.3f} (p={p_par:.3f})")
    print("     ^ si esto es ~0, lo que comparten los motores ya estaba en el punto de partida")

    # correlación DENTRO de grafo (centrando cada valor en la media de su grafo),
    # que es la variación que la intervención realmente movió
    dc = d.copy()
    for c in ["frac_masa", principal_fin, principal_ini]:
        dc[c + "_c"] = dc[c] - dc.groupby(["rule_id", "seed"])[c].transform("mean")
    rc_fin, pc_fin = stats.pearsonr(dc["frac_masa_c"], dc[principal_fin + "_c"])
    rc_ini, pc_ini = stats.pearsonr(dc["frac_masa_c"], dc[principal_ini + "_c"])
    print(f"\n  centrado DENTRO de cada grafo (sólo la variación entre brazos, n=24):")
    print(f"     Phantom ↔ motor (t=0.5): {rc_fin:+.3f} (p={pc_fin:.4f})")
    print(f"     Phantom ↔ IC   (t=0  ): {rc_ini:+.3f} (p={pc_ini:.4f})")

    # ---------------------------------------------------------------------
    # 3. TAMAÑO DEL EFECTO: LOS TRES NÚMEROS
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. TAMAÑO DEL EFECTO `solap` − `disj`: IC SIN INTEGRAR / MOTOR PROPIO / PHANTOM")
    print("=" * 90)
    res = [resumen_efecto(pares, "frac_masa", "PHANTOM (fracción de masa en sumideros)"),
           resumen_efecto(pares, principal_fin, "MOTOR INDEPENDIENTE (t=0.5, FoF ell=1.0 n>=5)"),
           resumen_efecto(pares, principal_ini, "IC SIN INTEGRAR (t=0, FoF ell=1.0 n>=5)")]
    for c in cols_nb_fin:
        if c != principal_fin:
            res.append(resumen_efecto(pares, c, f"motor indep. — {c[4:]}"))
    for c in cols_nb_ini:
        if c != principal_ini:
            res.append(resumen_efecto(pares, c, f"IC t=0 — {c[4:]}"))
    res.append(resumen_efecto(pares, "kappa_v", "PHANTOM κ_V"))
    ef = pd.DataFrame(res)
    print(ef.head(3).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n  -- todas las definiciones --")
    print(ef.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    ef.to_csv(os.path.join(AQUI, "cs090_fase8_f805_efectos.csv"), index=False)

    # el efecto en grano de partículas (1 partícula = 0.0005 de fracción de masa)
    print("\n  en partículas (1 partícula = 0.0005 de fracción de masa a N=2000):")
    for r in res[:3]:
        print(f"    {r['observable']:52s} Δ={r['delta_medio']:+.5f} "
              f"= {r['delta_medio']/0.0005:+.1f} partículas "
              f"({r['pct_relativo']:+.1f}%), signos {r['signos_positivos']}/12, "
              f"Wilcoxon p={r['p_wilcoxon']:.2e}")

    pares.to_csv(os.path.join(AQUI, "cs090_fase8_f805_pares.csv"), index=False)
    d.to_csv(os.path.join(AQUI, "cs090_fase8_f805_unido.csv"), index=False)

    # ---------------------------------------------------------------------
    # 4. ¿LA DINÁMICA AGREGA ALGO? (final menos inicial)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. LO QUE AGREGÓ LA GRAVEDAD: observable final MENOS inicial")
    print("=" * 90)
    d["incremento"] = d[principal_fin] - d[principal_ini]
    pr_inc = tabla_pares(d, ["frac_masa", "incremento"])
    k_inc = int((np.sign(pr_inc["incremento__d"]) == np.sign(pr_inc["frac_masa__d"])).sum())
    r_inc, p_inc = stats.pearsonr(d.frac_masa, d.incremento)
    print(f"  Δ(solap−disj) del INCREMENTO: media {pr_inc['incremento__d'].mean():+.5f}, "
          f"signos {int((pr_inc['incremento__d']>0).sum())}/12, "
          f"orden coincidente con Phantom {k_inc}/12")
    print(f"  Pearson(frac_masa de Phantom, incremento del motor propio) = {r_inc:+.3f} (p={p_inc:.3f})")

    if vara_comun:
        vara_comun_phantom(d, pares)


# ---------------------------------------------------------------------------
# LA MISMA VARA EN LAS DOS MESAS (opcional, requiere sarracen)
# ---------------------------------------------------------------------------
def fof_masa(pos, masas, ell, masa_min):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(pos)
    arbol = cKDTree(pos)
    prs = arbol.query_pairs(r=ell, output_type="ndarray")
    if len(prs) == 0:
        etiq, ncomp = np.arange(n), n
    else:
        g = coo_matrix((np.ones(len(prs)), (prs[:, 0], prs[:, 1])), shape=(n, n))
        ncomp, etiq = connected_components(g, directed=False)
    mg = np.bincount(etiq, weights=masas, minlength=ncomp)
    grandes = mg >= masa_min
    return float(mg[grandes].sum() / masas.sum()), int(grandes.sum())


def vara_comun_phantom(d, pares):
    """Reconstruye el estado final de Phantom (gas vivo + sumideros con su masa acretada)
    y le aplica EL MISMO friends-of-friends que al motor propio, con criterio de tamaño
    en MASA (>= 5 × 9.4 = 47), porque un sumidero es un punto que ya vale por decenas."""
    import sarracen
    print("\n" + "=" * 90)
    print("5. LA MISMA VARA EN LAS DOS MESAS (FoF sobre el estado final de Phantom)")
    print("=" * 90)
    masa_min = 5 * M_PART
    filas = []
    for _, f in d.iterrows():
        carpeta = os.path.dirname(f.ruta_ic)
        r = sarracen.read_phantom(os.path.join(carpeta, "cosmog_00500"))
        gas = r[0]
        pos = gas[["x", "y", "z"]].to_numpy()
        masas = np.full(len(pos), M_PART)
        if len(r) > 1 and len(r[1]) > 0:
            s = r[1]
            pos = np.vstack([pos, s[["x", "y", "z"]].to_numpy()])
            masas = np.concatenate([masas, s["m"].to_numpy()])
        fila = dict(rule_id=f.rule_id, seed=f.seed, brazo=f.brazo,
                    n_puntos=len(masas), masa_total=float(masas.sum()))
        for ell in (0.3, 0.45, 0.6, 1.0, 2.0):
            v, ng = fof_masa(pos, masas, ell, masa_min)
            fila[f"ph_fofcomun_ell{ell}"] = v
            fila[f"nb_fofcomun_ell{ell}"] = float(f[f"fin_fof_ell{ell}_nmin5"])
            fila[f"ic_fofcomun_ell{ell}"] = float(f[f"ini_fof_ell{ell}_nmin5"])
        filas.append(fila)
    vc = pd.DataFrame(filas)
    vc.to_csv(os.path.join(AQUI, "cs090_fase8_f805_vara_comun.csv"), index=False)
    for ell in (0.3, 0.45, 0.6, 1.0, 2.0):
        cph, cnb, cic = (f"ph_fofcomun_ell{ell}", f"nb_fofcomun_ell{ell}", f"ic_fofcomun_ell{ell}")
        pr = tabla_pares(vc, [cph, cnb, cic])
        k = int((np.sign(pr[f"{cnb}__d"]) == np.sign(pr[f"{cph}__d"])).sum())
        ki = int((np.sign(pr[f"{cic}__d"]) == np.sign(pr[f"{cph}__d"])).sum())
        rp, _ = stats.pearsonr(vc[cph], vc[cnb])
        print(f"  ell={ell:<5} Phantom(vara común) media solap {pr[f'{cph}__solap'].mean():.4f} "
              f"vs disj {pr[f'{cph}__disj'].mean():.4f} | Δ={pr[f'{cph}__d'].mean():+.5f} "
              f"signos {int((pr[f'{cph}__d']>0).sum())}/12 | Pearson ph↔nb {rp:+.3f} | "
              f"orden ph↔nb {k}/12, ph↔IC {ki}/12")


if __name__ == "__main__":
    main(vara_comun="--vara-comun" in sys.argv)
