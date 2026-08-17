"""
cs090_fase6_remedir_fase3.py — ¿el bug del diámetro toca a FASE III (renormalización, cs080)?
==============================================================================================

POR QUÉ HAY QUE PREGUNTARLO
---------------------------
`cs080_renormalizacion.py` mide el diámetro con `C7._diam` — que es la MISMA función congelada de
cs055 que trae el bug (arranca el doble-BFS en el nodo no aislado de índice más bajo del grafo entero;
si ése cayó en un fragmento suelto, mide el fragmento). Y la conclusión central de
`FASE3_renormalizacion_resultado_CS.md` es una PENDIENTE (diám vs nº de cajas, 0.35-0.45, indistinguible
entre REAL / barajado / ER) — exactamente el tipo de número que el bug distorsiona.

QUÉ HACE ESTE SCRIPT
--------------------
Reconstruye los mismos sustratos de Fase III (3 semillas × 3 brazos: local / local_barajado / er_null,
N=8000, reusando `cs080_renormalizacion.construir_sustrato` sin tocarlo), los coarse-granea a las mismas
6 escalas (b=1,2,4,8,16,32) con `cajas_bfs`/`grafo_grueso`, y en CADA escala mide el diámetro de las dos
maneras: la histórica y la corregida (componente gigante). Reporta, escala por escala, el tamaño de la
componente donde cayó la medición histórica frente al de la gigante — que es el diagnóstico que dice si
el bug muerde o no — y la pendiente log(diám) vs log(N_cajas) con las dos varas.

LIMITACIÓN DECLARADA (no se disimula)
--------------------------------------
`cs080_renormalizacion.py` deriva varias de sus semillas de rng con `hash(arm)` (p.ej.
`RNG(seed*137 + hash(arm) % 9973 + 5)`). El hash de strings en Python está aleatorizado por proceso
salvo que se fije `PYTHONHASHSEED`, así que la corrida histórica de Fase III **no es reproducible bit a
bit** sin conocer ese valor (que no quedó registrado). Este script fija `PYTHONHASHSEED=0` para ser
reproducible de acá en adelante, y por eso sus grafos son del MISMO tipo pero no la misma realización
que los del informe de Fase III. Eso no afecta a la pregunta que se está haciendo — "¿los grafos de
Fase III se fragmentan al punto de descarrilar la medición?" es una propiedad del tipo de sustrato, no
de una realización particular — pero sí impide comparar los diámetros número a número con el CSV
histórico, y por eso no se hace esa comparación.

No corre Phantom. No toca ningún script congelado. No declara veredicto.
"""
from __future__ import annotations
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)   # re-arranca con hash determinista

import csv
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs080_renormalizacion as CS80
import cs090_diam_corregido as DC
from cs090_fase5_clasificador import _pendiente_loglog

SEMILLAS = (80100, 80200, 80300)
BRAZOS = ("local", "local_barajado", "er_null")
ESCALAS = (1, 2, 4, 8, 16, 32)
N = CS80.N_NODOS


def main():
    filas, t0 = [], time.time()
    for seed in SEMILLAS:
        for arm in BRAZOS:
            adj, V = CS80.construir_sustrato(N, seed, arm)
            n_cajas_l, d_o, d_c = [], [], []
            for b in ESCALAS:
                if b == 1:
                    adj_g, nc = adj, N
                else:
                    rng_b = np.random.default_rng(seed * 733 + b * 31 + hash(arm) % 4999)
                    asign, nc = CS80.cajas_bfs(adj, N, b, rng_b)
                    adj_g = CS80.grafo_grueso(adj, N, asign, nc)
                dg = DC.diagnostico(adj_g, nc)
                n_cajas_l.append(nc); d_o.append(dg["diam_orig"]); d_c.append(dg["diam_corr"])
                filas.append(dict(seed=seed, arm=arm, b=b, n_cajas=nc,
                                  diam_viejo=dg["diam_orig"], diam_corregido=dg["diam_corr"],
                                  tam_comp_medida=dg["tam_comp_medida"], tam_gigante=dg["tam_gigante"],
                                  n_componentes=dg["n_componentes"], n_aislados=dg["n_aislados"],
                                  descarrila=dg["descarrila"]))
                print(f"  {arm:<15} seed={seed} b={b:<3} n_cajas={nc:<5} "
                      f"diam viejo={dg['diam_orig']:.0f} corregido={dg['diam_corr']:.0f}  "
                      f"comp_medida={dg['tam_comp_medida']:<6} gigante={dg['tam_gigante']:<6} "
                      f"comps={dg['n_componentes']:<4} aislados={dg['n_aislados']:<5}"
                      f"{'  <-- DESCARRILA' if dg['descarrila'] else ''}", flush=True)
            pv = _pendiente_loglog(n_cajas_l, d_o)
            pc = _pendiente_loglog(n_cajas_l, d_c)
            print(f"  >>> {arm:<15} seed={seed}  pendiente VIEJA={pv:+.4f}  CORREGIDA={pc:+.4f}  "
                  f"(diferencia {pc-pv:+.4f})\n", flush=True)

    with open(f"{HERE}/cs090_fase6_remedicion_fase3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for d in filas:
            w.writerow(d)
    n_desc = sum(1 for d in filas if d["descarrila"])
    n_dif = sum(1 for d in filas if d["diam_viejo"] != d["diam_corregido"])
    print(f"[csv] cs090_fase6_remedicion_fase3.csv ({len(filas)} filas, {time.time()-t0:.0f}s)")
    print(f"[resultado] escalas donde la medición vieja DESCARRILA: {n_desc}/{len(filas)}")
    print(f"[resultado] escalas donde viejo != corregido:           {n_dif}/{len(filas)}")


if __name__ == "__main__":
    main()
