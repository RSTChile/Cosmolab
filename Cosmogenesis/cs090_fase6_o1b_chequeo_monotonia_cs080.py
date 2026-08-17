"""
cs090_fase6_o1b_chequeo_monotonia_cs080.py — las 2 no-monotonías del CSV histórico de cs080: ¿bug o azar?
==========================================================================================================

DE DÓNDE SALE ESTA PREGUNTA (tarea O1-B, Parte 1, seguimiento)
---------------------------------------------------------------
El detector barato aplicado al CSV histórico `cs080_renormalizacion.csv` (Fase III Exp.1) encontró
**2 series de 9** en las que el diámetro NO decrece de forma monótona al agrupar: en el brazo
`local_barajado`, semillas 80100 y 80200, el diámetro pasa de **3 en b=8 a 4 en b=16**.

Por qué importa: contraer cajas CONEXAS no puede alargar un camino *respecto del grafo original*, así que
`diám(b=16) > diám(b=8)` podría ser la huella del bug de `_diam` (que en una de las dos escalas midió un
fragmento suelto en vez de la componente gigante). Pero hay una explicación alternativa, inocente: las
cajas de b=8 y las de b=16 **no están anidadas** — `cajas_bfs` sortea semillas de caja al azar y hace un
recubrimiento nuevo, independiente, en cada escala. Dos recubrimientos distintos del mismo grafo pueden
dar diámetros gruesos que difieran en ±1 sin que nada esté roto.

CÓMO SE DISTINGUE UNA COSA DE LA OTRA
--------------------------------------
Se construye UNA vez el sustrato del caso señalado (arm=`local_barajado`, semilla 80100, N=8000, con
`cs080_renormalizacion.construir_sustrato` sin tocarlo) y se repite el recubrimiento en b=8 y b=16
**muchas veces**, cambiando sólo la semilla del sorteo de cajas. Para cada réplica se guarda:

  - el diámetro medido con la vara vieja (`cs055._diam`) y con la corregida (componente gigante),
  - el tamaño de la componente donde cayó la vieja frente al de la gigante (el diagnóstico del bug),
  - si la pareja (b=8, b=16) es no-monótona.

Lectura del resultado:
  - si aparecen no-monotonías con `descarrila=False` y `viejo == corregido` -> es **azar de recubrimiento**,
    no el bug;
  - si las no-monotonías vienen siempre acompañadas de `descarrila=True` -> es **el bug**.

ADVERTENCIA DE REPRODUCIBILIDAD (la misma de `cs090_fase6_remedir_fase3.py`)
-----------------------------------------------------------------------------
cs080 deriva semillas con `hash(arm)`, aleatorizado por proceso salvo `PYTHONHASHSEED` fijo (no quedó
registrado el de la corrida histórica). Acá se fija `PYTHONHASHSEED=0`: el sustrato es del mismo tipo,
no la misma realización. Por eso la pregunta que se hace es estadística ("¿este tipo de recubrimiento
produce ±1 de no-monotonía sin descarrilar?") y no "reproducí exactamente esas dos filas".

No toca ningún script congelado. No corre Phantom. No declara cierre ni veredicto.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("CS080_N", "8000")     # el histórico corrió con N=8000 (n_cajas=8000 en b=1)
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import csv
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs080_renormalizacion as C80
import cs090_diam_corregido as DC

RNG = np.random.default_rng

SEED_SUSTRATO = 80100
ARM = "local_barajado"
ESCALAS = (8, 16)
N_REPLICAS = 40                 # recubrimientos independientes por escala
OUT = os.path.join(HERE, "cs090_fase6_o1b_monotonia_cs080.csv")


def main():
    N = C80.N_NODOS
    print("=" * 100, flush=True)
    print(f"O1-B — ¿las 2 no-monotonías del CSV histórico de cs080 son el bug o azar de recubrimiento?")
    print(f"sustrato: arm={ARM} seed={SEED_SUSTRATO} N={N} · {N_REPLICAS} recubrimientos por escala {ESCALAS}",
          flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    adj, V = C80.construir_sustrato(N, SEED_SUSTRATO, ARM)
    print(f"  sustrato construido ({time.time()-t0:.0f}s, aristas={sum(len(a) for a in adj)//2})", flush=True)

    filas = []
    por_escala = {b: [] for b in ESCALAS}
    for b in ESCALAS:
        for rep in range(N_REPLICAS):
            rng_b = RNG(1_000_000 + b * 1013 + rep)      # semilla de recubrimiento, la única que varía
            asign, n_cajas = C80.cajas_bfs(adj, N, b, rng_b)
            adj_g = C80.grafo_grueso(adj, N, asign, n_cajas)
            d = DC.diagnostico(adj_g, n_cajas)
            fila = dict(b=b, replica=rep, n_cajas=n_cajas,
                        diam_viejo=d["diam_orig"], diam_corregido=d["diam_corr"],
                        tam_comp_medida=d["tam_comp_medida"], tam_gigante=d["tam_gigante"],
                        n_componentes=d["n_componentes"], n_aislados=d["n_aislados"],
                        descarrila=d["descarrila"])
            filas.append(fila)
            por_escala[b].append(fila)

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        for fila in filas:
            wr.writerow(fila)

    print()
    for b in ESCALAS:
        ds = [f["diam_viejo"] for f in por_escala[b]]
        print(f"  b={b:<3} diám viejo: min={min(ds):.0f} max={max(ds):.0f} media={np.mean(ds):.2f}  "
              f"valores={sorted(set(int(x) for x in ds))}   "
              f"descarrilamientos={sum(1 for f in por_escala[b] if f['descarrila'])}/{len(ds)}   "
              f"viejo!=corregido={sum(1 for f in por_escala[b] if f['diam_viejo'] != f['diam_corregido'])}/{len(ds)}",
              flush=True)

    # pares (b=8 rep_i, b=16 rep_j): ¿con qué frecuencia el grueso da MÁS que el fino?
    d8 = [f["diam_viejo"] for f in por_escala[8]]
    d16 = [f["diam_viejo"] for f in por_escala[16]]
    pares = [(x, y) for x in d8 for y in d16]
    n_no_mono = sum(1 for x, y in pares if y > x)
    print(f"\n  pares (b=8, b=16) cruzados: {len(pares)}   con diám(16) > diám(8): {n_no_mono} "
          f"({100*n_no_mono/len(pares):.1f}%)", flush=True)
    print(f"  descarrilamientos TOTALES en las {len(filas)} réplicas: "
          f"{sum(1 for f in filas if f['descarrila'])}", flush=True)
    print(f"\n[csv] {OUT} ({len(filas)} filas, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
