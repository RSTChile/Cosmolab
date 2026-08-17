#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase6_o4a_observable_comun.py — el MISMO observable medido en los dos motores
===================================================================================

POR QUÉ HACE FALTA ESTO
-----------------------
La comparación principal de O4-A enfrenta dos cantidades que NO son la misma:
"fracción de masa en sumideros" (Phantom) contra "fracción de masa apelotonada"
(el motor de gravedad pura). Está bien para comparar ORDEN, pero deja abierta la
objeción: ¿y si el desacuerdo (o el acuerdo) viene sólo de que son dos varas distintas?

Este script cierra esa rendija: aplica LA MISMA vara — el mismo friends-of-friends,
con la misma longitud de enlace y el mismo criterio de tamaño — al estado final de
Phantom. Para eso reconstruye el estado final de Phantom como una nube de masas
puntuales: las partículas de gas que quedaron vivas (masa 9.4 c/u) MÁS los sumideros,
cada uno con la masa que acretó.

Criterio de tamaño en MASA (no en número de miembros), porque un sumidero es un solo
punto que ya vale por decenas de partículas: un grupo cuenta si su masa total supera
`masa_min` (por omisión 5 × 9.4 = 47, el equivalente a 5 partículas).

Analogía: antes comparábamos "cuántos terrones quedaron" medidos con dos reglas
distintas; acá usamos la misma regla en las dos mesas.

Uso:
    ./venv/bin/python cs090_fase6_o4a_observable_comun.py
"""

import os
import numpy as np
import pandas as pd
import sarracen
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import cs090_fase6_o4a_nbody as nb
from cs090_fase6_o4a_correr import PARES, ruta_ic, GRILLA_ELL, T_FINAL, DT, EPS_PRINCIPAL

AQUI = os.path.dirname(os.path.abspath(__file__))
M_PART = 9.4
N_MIN_EQUIV = 5                       # tamaño mínimo en "partículas equivalentes"
MASA_MIN = N_MIN_EQUIV * M_PART


def fof_masa(pos, masas, ell, masa_min):
    """Friends-of-friends con criterio de tamaño en MASA. Devuelve la fracción de la
    masa total que quedó en grupos cuya masa supera `masa_min`."""
    n = len(pos)
    arbol = cKDTree(pos)
    pares = arbol.query_pairs(r=ell, output_type="ndarray")
    if len(pares) == 0:
        etiq = np.arange(n)
        ncomp = n
    else:
        g = coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])), shape=(n, n))
        ncomp, etiq = connected_components(g, directed=False)
    masa_grupo = np.bincount(etiq, weights=masas, minlength=ncomp)
    grandes = masa_grupo >= masa_min
    return float(masa_grupo[grandes].sum() / masas.sum()), int(grandes.sum())


def estado_final_phantom(carpeta):
    """Nube de masas puntuales equivalente al estado final de Phantom (t=0.5)."""
    r = sarracen.read_phantom(os.path.join(carpeta, "cosmog_00500"))
    gas = r[0]
    pos = gas[["x", "y", "z"]].to_numpy()
    masas = np.full(len(pos), M_PART)
    if len(r) > 1 and len(r[1]) > 0:
        s = r[1]
        pos = np.vstack([pos, s[["x", "y", "z"]].to_numpy()])
        masas = np.concatenate([masas, s["m"].to_numpy()])
    return pos, masas


def main():
    corridas = pd.read_csv(os.path.join(AQUI, "cs090_fase6_o4a_corridas_nbody.csv")).set_index("regla")
    filas = []
    vistos = set()
    for _, rI, rIII, _, estrato in PARES:
        for regla, rol in ((rI, "I"), (rIII, "III")):
            if regla in vistos:
                continue
            vistos.add(regla)
            carpeta = os.path.dirname(ruta_ic(regla))
            pph, mph = estado_final_phantom(carpeta)
            # estado final del motor propio: se re-integra (barato: ya está validado)
            pos0, vel0, m_part, _ = nb.leer_ic_cosmogenesis(ruta_ic(regla))
            fila = dict(regla=regla, rol=rol, estrato=estrato,
                        masa_total_phantom=float(mph.sum()), n_puntos_phantom=len(mph))
            for ell in GRILLA_ELL:
                f, ng = fof_masa(pph, mph, ell, MASA_MIN)
                fila[f"phantom_fofcomun_ell{ell}"] = round(f, 6)
                fila[f"phantom_ngrupos_ell{ell}"] = ng
                # el mismo FoF sobre las condiciones iniciales (control de geometría)
                f0, _ = fof_masa(pos0, np.full(len(pos0), m_part), ell, MASA_MIN)
                fila[f"ic_fofcomun_ell{ell}"] = round(f0, 6)
                # y el del motor propio (ya calculado con criterio en nº de miembros,
                # que para partículas de masa igual es exactamente el mismo criterio)
                fila[f"nbody_fofcomun_ell{ell}"] = float(
                    corridas.loc[regla, f"fin_fof_ell{ell}_nmin{N_MIN_EQUIV}"])
            filas.append(fila)
            print(f"  {regla:24s} {rol:3s} "
                  + "  ".join(f"ell{e}: ph={fila[f'phantom_fofcomun_ell{e}']:.3f} "
                              f"nb={fila[f'nbody_fofcomun_ell{e}']:.3f}" for e in GRILLA_ELL),
                  flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(AQUI, "cs090_fase6_o4a_observable_comun.csv"), index=False)

    # --- concordancia de orden por par con la vara común ---
    from scipy import stats
    idx = df.set_index("regla")
    print("\n=== ORDEN POR PAR CON LA MISMA VARA (FoF de masa, masa_min=47) ===")
    for ell in GRILLA_ELL:
        res = []
        for par, rI, rIII, dph, estrato in PARES:
            dph_c = idx.loc[rIII, f"phantom_fofcomun_ell{ell}"] - idx.loc[rI, f"phantom_fofcomun_ell{ell}"]
            dnb = idx.loc[rIII, f"nbody_fofcomun_ell{ell}"] - idx.loc[rI, f"nbody_fofcomun_ell{ell}"]
            dic = idx.loc[rIII, f"ic_fofcomun_ell{ell}"] - idx.loc[rI, f"ic_fofcomun_ell{ell}"]
            res.append((par, estrato, dph, dph_c, dnb, dic))
        r = pd.DataFrame(res, columns=["par", "estrato", "d_sumideros_phantom",
                                       "d_fofcomun_phantom", "d_fofcomun_nbody", "d_fofcomun_IC"])
        r["coincide_ph_vs_nb"] = np.sign(r.d_fofcomun_phantom) == np.sign(r.d_fofcomun_nbody)
        r["coincide_sumideros_vs_nb"] = np.sign(r.d_sumideros_phantom) == np.sign(r.d_fofcomun_nbody)
        r["coincide_IC_vs_sumideros"] = np.sign(r.d_fofcomun_IC) == np.sign(r.d_sumideros_phantom)
        print(f"\n-- ell = {ell} --")
        print(r.to_string(index=False))
        for c in ("coincide_ph_vs_nb", "coincide_sumideros_vs_nb", "coincide_IC_vs_sumideros"):
            k, n = int(r[c].sum()), len(r)
            kf = int(r[r.estrato == "fuerte"][c].sum())
            ke = int(r[r.estrato == "empate"][c].sum())
            print(f"   {c:28s}: total {k}/{n} (fuerte {kf}/5, empate {ke}/5)  "
                  f"p={stats.binomtest(k, n, 0.5).pvalue:.4f}")
        r.to_csv(os.path.join(AQUI, f"cs090_fase6_o4a_ordenes_ell{ell}.csv"), index=False)


if __name__ == "__main__":
    main()
