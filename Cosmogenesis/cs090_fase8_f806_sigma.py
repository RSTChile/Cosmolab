#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
cs090_fase8_f806_sigma.py — F8-06, pieza auxiliar: **medir el grano (σ) del instrumento a N=4000**
====================================================================================================

POR QUÉ HACE FALTA (a nivel módulo)
------------------------------------
El efecto que esta tarea persigue se tiene que reportar **en unidades de σ**, no sólo en valor
absoluto: a N=4000 nacen ~29 sumideros contra 8 a N=2000, y `FASE8_F804_grano_n8000_CS.md` midió que
**el grano no lo pone N, lo ponen los sumideros** (σ ∝ n_sumideros^1.24, R=0.92). De ese ajuste sale
σ ≈ 0.00265 a N=4000 — pero **es una extrapolación desde un solo grafo**. Este script la reemplaza por
una medición directa.

EL MÉTODO, QUE ES EL DE F8-04 (no se inventa uno nuevo)
--------------------------------------------------------
Se pesa **el mismo objeto** muchas veces cambiando **nada**: se mueve el último bit de un número y se
mira cuánto baila el resultado. Concretamente, `lado` —el lado de la caja del layout— se corre
**k ULPs** con `np.nextafter` (k = 0…N_REPLICAS-1). El ULP de `lado = 2000^(1/3) = 12.599210498948731`
es 1.78e-15 absoluto = **1.4e-16 relativo**: exactamente el orden del control de redondeo que la
validación de infraestructura usó como vara.

Por qué esa inyección es legítima y no una perturbación física disfrazada: el layout
Fruchterman-Reingold es **covariante de escala** (`pos ~ lado`, `k_FR ~ lado`, repulsión, atracción y
paso todos ~ `lado`), así que multiplicar `lado` por (1+ε) da, **en aritmética exacta, el mismo layout
multiplicado por (1+ε)** — la misma configuración salvo un reescalado de 1e-16, que en una fracción de
masa es rigurosamente nada. Todo lo que aparezca por encima de eso es **redondeo amplificado por las
100 iteraciones**, que es justo lo que se quiere medir. Y `lado_fijo` es un argumento de
`generar_ic_masa_fija_desde_grafo`: no hay que tocar una línea de código congelado.

QUÉ SE MANTIENE IGUAL AL RESTO DE F8-06
-----------------------------------------
El **mismo grafo** (se carga de `grafo_final.grafo.gz`, con verificación de sello sha256), el **mismo
layout O(N²)** `layout_resortes` sin Barnes-Hut, la misma `seed_layout=12345`, la misma masa total
18800, la misma dilatación, la misma turbulencia, y el mismo protocolo de Phantom. Lo único que cambia
entre réplicas es el último bit de `lado`.

*(La réplica k=0 NO es un "original privilegiado": F8-04 §1.4b mostró que el layout no es reproducible
ni consigo mismo entre procesos. Las réplicas son k tiradas equivalentes de la misma distribución, que
es exactamente lo que una σ necesita.)*

USO
---
    python3.9 cs090_fase8_f806_sigma.py ic <carpeta_origen> <k>     # una réplica (paralelizable)
    ./venv/bin/python cs090_fase8_f806_sigma.py medir               # lee los dumps -> CSV

Sale `cs090_fase8_f806_sigma.csv`, que `cs090_fase8_f806_analizar.py` recoge solo si existe.
No declara cierre ni veredicto.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase8_f800_grafos as G8                              # sólo import
from cs090_fase5b_phantom_adaptador import (                      # sólo import
    generar_ic_masa_fija_desde_grafo, LADO_FIJO)

BASE_REPLICAS = Path("/Users/alexis/phantom_cs073/bateria_fase8_f806_n4000_sigma")
N_PART = 4000
SEED_LAYOUT = 12345
RUTA_CSV = f"{HERE}/cs090_fase8_f806_sigma.csv"


def lado_perturbado(k: int) -> float:
    """`LADO_FIJO` movido k ULPs hacia arriba. k=0 devuelve el valor exacto de la serie."""
    lado = float(LADO_FIJO)
    for _ in range(int(k)):
        lado = float(np.nextafter(lado, np.inf))
    return lado


def generar_replica(carpeta_origen: str, k: int):
    origen = Path(carpeta_origen)
    adj, n, meta_g = G8.cargar_grafo(origen / ("grafo_final" + G8.SUFIJO))   # verifica el sello
    assert n == N_PART, f"el grafo guardado tiene N={n}, se esperaba {N_PART}"
    meta_orig = json.loads((origen / "meta_regla.json").read_text())

    destino = BASE_REPLICAS / f"{origen.name}_rep{k:02d}"
    destino.mkdir(parents=True, exist_ok=True)
    lado = lado_perturbado(k)
    t0 = time.time()
    info = generar_ic_masa_fija_desde_grafo(adj, N=N_PART, seed_layout=SEED_LAYOUT,
                                            ruta_salida=str(destino / "cosmogenesis_ic.txt"),
                                            lado_fijo=lado)
    t_ic = round(time.time() - t0, 2)
    with open(destino / "meta_f806_sigma.json", "w") as f:
        json.dump(dict(tarea="FASE8_F806_sigma_a_N4000", origen=str(origen), replica=int(k),
                       N=N_PART, lado=repr(lado), lado_nominal=repr(float(LADO_FIJO)),
                       ulps=int(k), seed_layout=SEED_LAYOUT,
                       layout="layout_resortes_N2_exacto", theta=None,
                       rule_id=meta_orig["rule_id"], seed=meta_orig["seed"],
                       brazo=meta_orig["brazo"], grafo_sha256=meta_g["sha256"],
                       n_aristas=meta_g["E"], t_layout_ic_s=t_ic, masa_total=info["masa_total"]), f,
                  indent=2)
    print(f"[sigma] {destino.name}: lado={lado!r} (k={k} ULP) ic en {t_ic}s", flush=True)
    return destino


def medir():
    from cs090_fase5b_analizar import analizar_carpeta            # sólo import (necesita el venv)
    filas = []
    for carpeta in sorted(c for c in BASE_REPLICAS.iterdir() if c.is_dir()):
        mp = carpeta / "meta_f806_sigma.json"
        if not mp.exists() or not (carpeta / "cosmog_00500").exists():
            print(f"   !! {carpeta.name}: sin meta o sin dump final — se salta")
            continue
        m = json.loads(mp.read_text())
        a = analizar_carpeta(carpeta)
        if a.get("n_gas_inicial") not in (None, N_PART):
            print(f"   !! {carpeta.name}: n_gas_inicial={a['n_gas_inicial']}")
        filas.append(dict(carpeta=carpeta.name, rule_id=m["rule_id"], seed=m["seed"],
                          brazo=m["brazo"], replica=m["replica"], ulps=m["ulps"], lado=m["lado"],
                          frac_masa=a["fraccion_masa_en_sumideros"], n_sumideros=a["n_sumideros"],
                          kappa_v=a["kappa_v_agregado"],
                          t_primer_sumidero=a["t_primer_sumidero"],
                          dump_final=a.get("n_dump_final")))
    if not filas:
        print("[sigma] sin réplicas medibles")
        return
    with open(RUTA_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    v = np.array([f["frac_masa"] for f in filas], dtype=float)
    ns = [f["n_sumideros"] for f in filas]
    print(f"[sigma] n={len(v)} réplicas · media={v.mean():.5f} · σ={v.std(ddof=1):.5f} "
          f"({v.std(ddof=1)*N_PART:.1f} partículas) · rango={v.max()-v.min():.5f} · "
          f"sumideros {min(ns)}-{max(ns)}")
    print(f"[csv] {os.path.basename(RUTA_CSV)}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "medir":
        medir()
    elif len(sys.argv) >= 4 and sys.argv[1] == "ic":
        generar_replica(sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
