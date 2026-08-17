#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase6_o4a_analizar.py — compara el motor independiente contra Phantom
===========================================================================

Pregunta central: para cada par (una regla Clase III y una Clase I), ¿el motor de
gravedad pura pone al mismo ganador que Phantom?

Tres lecturas, no una:
  * 'fin'   — observable al final de la integración (t=0.5). Es la réplica propiamente
              dicha.
  * 'ini'   — el MISMO observable medido sobre las condiciones iniciales, SIN correr
              nada. Es el control barato y necesario: si el orden ya está en la
              geometría de partida, entonces "los dos motores coinciden" no dice nada
              sobre la dinámica, sólo dice que ambos heredan el mismo punto de partida.
  * 'delta' — lo que la dinámica AGREGÓ (final menos inicial). Es la parte del
              observable que sí depende de haber integrado.

Salidas:
  `cs090_fase6_o4a_comparacion_pares.csv`  (una fila por par, observable principal)
  `cs090_fase6_o4a_robustez_grilla.csv`    (la cuenta repetida en toda la grilla)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from cs090_fase6_o4a_correr import PARES, ELL_PRINCIPAL, NMIN_PRINCIPAL, GRILLA_ELL, GRILLA_NMIN

AQUI = os.path.dirname(os.path.abspath(__file__))
COL_PRINCIPAL = f"fof_ell{ELL_PRINCIPAL}_nmin{NMIN_PRINCIPAL}"


def cargar():
    nb = pd.read_csv(os.path.join(AQUI, "cs090_fase6_o4a_corridas_nbody.csv"))
    ph = pd.read_csv(os.path.join(AQUI, "cs090_fase5b_TOTAL_40pares.csv"))
    frac_ph = ph.groupby("rule_id")["fraccion_masa_en_sumideros"].first().to_dict()
    return nb.set_index("regla"), frac_ph


def valor(nb, regla, col, modo):
    if modo == "fin":
        return float(nb.loc[regla, "fin_" + col])
    if modo == "ini":
        return float(nb.loc[regla, "ini_" + col])
    return float(nb.loc[regla, "fin_" + col]) - float(nb.loc[regla, "ini_" + col])


def tabla_pares(nb, frac_ph, col, modo="fin"):
    filas = []
    for par, rI, rIII, dph, estrato in PARES:
        fI_ph, fIII_ph = frac_ph[rI], frac_ph[rIII]
        fI_nb, fIII_nb = valor(nb, rI, col, modo), valor(nb, rIII, col, modo)
        d_ph, d_nb = fIII_ph - fI_ph, fIII_nb - fI_nb
        filas.append(dict(
            par=par, estrato=estrato, regla_I=rI, regla_III=rIII,
            phantom_frac_I=fI_ph, phantom_frac_III=fIII_ph,
            phantom_dIII_menos_I=round(d_ph, 6),
            nbody_obs_I=round(fI_nb, 6), nbody_obs_III=round(fIII_nb, 6),
            nbody_dIII_menos_I=round(d_nb, 6),
            gana_en_phantom=("III" if d_ph > 0 else "I" if d_ph < 0 else "empate"),
            gana_en_nbody=("III" if d_nb > 0 else "I" if d_nb < 0 else "empate"),
            orden_coincide=bool(np.sign(d_ph) == np.sign(d_nb) and d_ph != 0),
        ))
    return pd.DataFrame(filas)


def resumen(tp, etiqueta):
    lin = [f"[{etiqueta}]"]
    for est in ("fuerte", "empate"):
        s = tp[tp.estrato == est]
        k, n = int(s.orden_coincide.sum()), len(s)
        lin.append(f"  estrato {est:7s}: {k}/{n}  p={stats.binomtest(k, n, 0.5).pvalue:.4f}")
    k, n = int(tp.orden_coincide.sum()), len(tp)
    lin.append(f"  TOTAL            : {k}/{n}  p={stats.binomtest(k, n, 0.5).pvalue:.4f}")
    x, y = [], []
    for _, r in tp.iterrows():
        x += [r.phantom_frac_I, r.phantom_frac_III]
        y += [r.nbody_obs_I, r.nbody_obs_III]
    sr = stats.spearmanr(x, y)
    pr = stats.pearsonr(np.array(x), np.array(y))
    lin.append(f"  20 corridas : Pearson r={pr.statistic:+.3f} p={pr.pvalue:.4f} | "
               f"Spearman rho={sr.statistic:+.3f} p={sr.pvalue:.4f}")
    dsr = stats.spearmanr(tp.phantom_dIII_menos_I.values, tp.nbody_dIII_menos_I.values)
    dpr = stats.pearsonr(tp.phantom_dIII_menos_I.values, tp.nbody_dIII_menos_I.values)
    lin.append(f"  10 deltas   : Pearson r={dpr.statistic:+.3f} p={dpr.pvalue:.4f} | "
               f"Spearman rho={dsr.statistic:+.3f} p={dsr.pvalue:.4f}")
    return "\n".join(lin)


def main():
    nb, frac_ph = cargar()

    print("=" * 104)
    print(f"OBSERVABLE PRINCIPAL: {COL_PRINCIPAL}  (eps=0.6, t_final=0.5, dt={nb.dt.iloc[0]})")
    print("=" * 104)
    tp = tabla_pares(nb, frac_ph, COL_PRINCIPAL, "fin")
    tp.to_csv(os.path.join(AQUI, "cs090_fase6_o4a_comparacion_pares.csv"), index=False)
    print(tp.drop(columns=["regla_I", "regla_III"]).to_string(index=False))

    print()
    for modo, etiq in (("fin", "FINAL t=0.5 — la réplica"),
                       ("ini", "INICIAL t=0 — control: geometría pura, sin dinámica"),
                       ("delta", "DELTA fin-ini — lo que agregó la dinámica")):
        print(resumen(tabla_pares(nb, frac_ph, COL_PRINCIPAL, modo), etiq))
        print()

    print("--- robustez: misma cuenta con otras definiciones de 'región densa' ---")
    cols = [f"fof_ell{e}_nmin{m}" for e in GRILLA_ELL for m in GRILLA_NMIN]
    cols += [c[4:] for c in nb.columns if c.startswith("fin_knn8")]
    rob = []
    for c in cols:
        fila = dict(observable=c, media_fin=round(float(nb["fin_" + c].mean()), 4),
                    media_ini=round(float(nb["ini_" + c].mean()), 4))
        for modo in ("fin", "ini", "delta"):
            t = tabla_pares(nb, frac_ph, c, modo)
            fila[f"coinc_fuerte_{modo}"] = int(t[t.estrato == "fuerte"].orden_coincide.sum())
            fila[f"coinc_empate_{modo}"] = int(t[t.estrato == "empate"].orden_coincide.sum())
            fila[f"coinc_total_{modo}"] = int(t.orden_coincide.sum())
        t = tabla_pares(nb, frac_ph, c, "fin")
        x, y = [], []
        for _, r in t.iterrows():
            x += [r.phantom_frac_I, r.phantom_frac_III]
            y += [r.nbody_obs_I, r.nbody_obs_III]
        fila["spearman20_fin"] = round(float(stats.spearmanr(x, y).statistic), 3)
        rob.append(fila)
    rdf = pd.DataFrame(rob)
    rdf.to_csv(os.path.join(AQUI, "cs090_fase6_o4a_robustez_grilla.csv"), index=False)
    print(rdf.to_string(index=False))

    print("\n--- salud numérica de las corridas ---")
    print(nb[["dt", "nsteps", "deriva_energia_rel", "virial_final", "segundos"]]
          .describe().loc[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
