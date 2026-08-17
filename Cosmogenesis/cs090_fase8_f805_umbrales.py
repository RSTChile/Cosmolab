#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_fase8_f805_umbrales.py — ¿cambia la respuesta según cuán denso exijamos que sea?
=======================================================================================

POR QUÉ HACE FALTA
------------------
Phantom no cuenta "masa apelotonada": cuenta masa que cruzó un umbral de densidad muy
alto. En el `cosmog.in` de esta línea el umbral es `rho_crit_cgs = 1000` en unidades de
código, y la densidad media de estas cajas es 18800/97.6³ = 0.0202. O sea que
**el umbral de Phantom está unas 49.500 veces por encima de la densidad media**.

La primera tanda de F8-05 midió el análogo sólo hasta 1000×. Este script barre la
escalera completa —de 10× a 50.000×— sobre las MISMAS 24 corridas, para contestar dos
cosas:

  1. ¿el contraste `solap` − `disj` crece o se apaga a medida que el umbral se acerca
     al de Phantom?
  2. ¿cuánto de ese contraste ya estaba en las condiciones iniciales, y cuánto lo pone
     la gravedad? (la pregunta de O4-A, ahora resuelta umbral por umbral)

Analogía: no es lo mismo preguntar "¿cuánta harina quedó en montoncitos?" que "¿cuánta
harina quedó tan apretada que ya es masa?". Phantom pregunta lo segundo. Este barrido
recorre todos los grados intermedios de esa pregunta.

Se re-integra (el integrador es determinista: los estados finales salen idénticos a los
de `cs090_fase8_f805_correr.py`); no se toca ningún script existente.

USO
---
    ./venv/bin/python cs090_fase8_f805_umbrales.py [n_procesos]
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats

import cs090_fase6_o4a_nbody as nb
from cs090_fase8_f805_correr import (inventario_f703, T_FINAL, DT, EPS_PRINCIPAL, BRAZOS)

AQUI = os.path.dirname(os.path.abspath(__file__))
LADO = 97.6
M_PART = 9.4
RHO_MEDIA = 2000 * M_PART / LADO ** 3          # 0.02022 en unidades de código
RHO_CRIT_PHANTOM = 1000.0                      # el de `cosmog.in`
MULT_PHANTOM = RHO_CRIT_PHANTOM / RHO_MEDIA    # ≈ 49453

ESCALERA = [10, 30, 100, 300, 1000, 3000, 10000, 30000, MULT_PHANTOM]
K_VECINO = 8


def fracciones(pos):
    """Fracción de masa con densidad local (8.º vecino) por encima de cada umbral."""
    from scipy.spatial import cKDTree
    arbol = cKDTree(pos)
    dist, _ = arbol.query(pos, k=K_VECINO + 1)
    rk = np.maximum(dist[:, K_VECINO], 1e-12)
    rho = K_VECINO * M_PART / ((4.0 / 3.0) * np.pi * rk ** 3)
    return {f"m{int(round(m))}": float((rho > m * RHO_MEDIA).mean()) for m in ESCALERA}


def correr_una(args):
    rule_id, seed, brazo, ruta_ic = args
    t0 = time.time()
    pos0, vel0, m_part, _ = nb.leer_ic_cosmogenesis(ruta_ic)
    ini = fracciones(pos0)
    posf, _, diag = nb.integrar_leapfrog(pos0, vel0, m_part, EPS_PRINCIPAL, T_FINAL, DT)
    fin = fracciones(posf)
    fila = dict(rule_id=rule_id, seed=seed, brazo=brazo,
                deriva_energia_rel=diag["deriva_rel"], segundos=round(time.time() - t0, 1))
    fila.update({"ini_" + k: v for k, v in ini.items()})
    fila.update({"fin_" + k: v for k, v in fin.items()})
    print(f"  [ok] {rule_id:24s} {brazo:6s} ({fila['segundos']}s)", flush=True)
    return fila


def main(nproc=6):
    print(f"densidad media = {RHO_MEDIA:.5f}; umbral de Phantom = {RHO_CRIT_PHANTOM} "
          f"= {MULT_PHANTOM:.0f}× la media", flush=True)
    grafos = inventario_f703()
    tareas = [(rid, seed, b, d[b]["ic"]) for (rid, seed), d in sorted(grafos.items())
              for b in BRAZOS]
    with Pool(nproc) as p:
        filas = p.map(correr_una, tareas)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(AQUI, "cs090_fase8_f805_umbrales_crudo.csv"), index=False)

    ph = pd.read_csv(os.path.join(AQUI, "cs090_fase7_f703_phantom_crudo.csv"))
    ph = ph[ph.brazo.isin(list(BRAZOS))]
    df = df.merge(ph[["rule_id", "seed", "brazo", "frac_masa"]],
                  on=["rule_id", "seed", "brazo"], validate="one_to_one")

    print("\n" + "=" * 116)
    print("ESCALERA DE UMBRALES — `solap` − `disj`, en las condiciones iniciales y "
          "después de integrar")
    print("=" * 116)
    res = []
    for m in ESCALERA:
        et = f"m{int(round(m))}"
        r = {}
        for etapa in ("ini", "fin"):
            c = f"{etapa}_{et}"
            piv = df.pivot_table(index=["rule_id", "seed"], columns="brazo", values=c)
            d = (piv["solap"] - piv["disj"]).to_numpy()
            base = piv["disj"].to_numpy()
            r[etapa] = dict(delta=d.mean(), signos=int((d > 0).sum()),
                            pct=100 * d.mean() / base.mean() if base.mean() > 0 else np.nan,
                            nivel=piv[["solap", "disj"]].to_numpy().mean(),
                            p=stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan,
                            corr_ph=stats.pearsonr(df.frac_masa, df[c])[0])
        # ¿el orden coincide par a par con Phantom?
        pivp = df.pivot_table(index=["rule_id", "seed"], columns="brazo", values="frac_masa")
        dph = (pivp["solap"] - pivp["disj"]).to_numpy()
        pivf = df.pivot_table(index=["rule_id", "seed"], columns="brazo", values=f"fin_{et}")
        dfin = (pivf["solap"] - pivf["disj"]).to_numpy()
        pivi = df.pivot_table(index=["rule_id", "seed"], columns="brazo", values=f"ini_{et}")
        dini = (pivi["solap"] - pivi["disj"]).to_numpy()
        res.append(dict(umbral_x_media=int(round(m)),
                        masa_sobre_umbral_media=r["fin"]["nivel"],
                        ini_delta=r["ini"]["delta"], ini_signos=r["ini"]["signos"],
                        ini_pct=r["ini"]["pct"], ini_p=r["ini"]["p"],
                        fin_delta=r["fin"]["delta"], fin_signos=r["fin"]["signos"],
                        fin_pct=r["fin"]["pct"], fin_p=r["fin"]["p"],
                        aporte_dinamica=r["fin"]["delta"] - r["ini"]["delta"],
                        orden_fin_vs_phantom=int((np.sign(dfin) == np.sign(dph)).sum()),
                        orden_ini_vs_phantom=int((np.sign(dini) == np.sign(dph)).sum()),
                        pearson_fin_vs_phantom=r["fin"]["corr_ph"],
                        pearson_ini_vs_phantom=r["ini"]["corr_ph"]))
    tabla = pd.DataFrame(res)
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tabla.to_csv(os.path.join(AQUI, "cs090_fase8_f805_umbrales.csv"), index=False)
    print(f"\nreferencia — PHANTOM: Δ=+0.01433 (+12.6%), 12/12, "
          f"nivel medio de masa en sumideros 0.1209")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
