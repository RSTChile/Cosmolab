#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_f804_costo_por_dump.py — F8-04 parte 2: ¿cuánto cuesta REALMENTE una corrida a N=8000?
=============================================================================================

POR QUÉ ASÍ, Y NO CON UN CRONÓMETRO SUELTO
-------------------------------------------
`INFRA_layout_barnes_hut_CS.md` §8.1 dejó pendiente "cuánto cuesta una corrida COMPLETA a N=8000", y la
demo sólo dejó un número agregado: 1504 s **sin llegar a tmax**. Un solo número agregado no distingue
entre "es caro, linealmente" y "el costo por unidad de tiempo simulado está DIVERGIENDO". La diferencia
decide si la extrapolación tiene sentido.

Phantom escribe un volcado por cada `dtmax` de tiempo simulado (`cosmog_00000` … `cosmog_00500` para
tmax=0.500 con dtmax=0.001). La **fecha de modificación de cada volcado** es, entonces, un cronómetro
gratuito y ya grabado: la diferencia de mtime entre dumps consecutivos es el **tiempo de pared por unidad
de tiempo simulado**. Esa curva es lo que hay que mirar, y existe para TODA corrida ya hecha del proyecto
(N=2000, N=4000, N=8000), sin volver a correr nada.

LIMITACIÓN, DICHA ANTES: el mtime mide PARED, no CPU, así que incluye la contención de la máquina. Se la
usa para (a) comparar la FORMA de la curva entre resoluciones, que es machine-independiente, y (b) dar
cotas superiores de costo. Los tiempos absolutos de corridas hechas con la máquina cargada son cotas
superiores, no medidas de máquina libre.

USO
    ./venv/bin/python cs090_f804_costo_por_dump.py            # escanea las carpetas de referencia
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
from leer_volcado_phantom import listar_dumps   # congelado: sólo import

CSV_SALIDA = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_f804_costo_por_dump.csv")

# (etiqueta, N, patrón glob de carpetas de corrida)
FUENTES = [
    ("N2000_fase5b", 2000, "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/*"),
    ("N4000_o3a", 4000, "/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion/N4000/*"),
    ("N8000_demo_infra", 8000, "/Users/alexis/phantom_cs073/infra_layout_bh_demo_n8000/N8000/*"),
    ("N8000_f804", 8000, "/Users/alexis/phantom_cs073/f804_grano_n8000/N8000/*"),
]
MAX_POR_FUENTE = 8   # con 8 corridas por resolución alcanza para ver la forma; no hace falta barrer todo


def curva(carpeta: Path):
    dumps = listar_dumps(carpeta)
    # OJO: en las baterías viejas de N=2000 y N=4000 los volcados INTERMEDIOS fueron borrados para
    # liberar disco (quedan sólo `cosmog_00000` y `cosmog_00500`). Con dos puntos no hay curva, pero sí
    # hay el dato que importa para la comparación: el tiempo de pared TOTAL de una corrida completa.
    if len(dumps) < 2:
        return None
    idx = np.array([int(p.stem[7:]) for p in dumps])
    mt = np.array([p.stat().st_mtime for p in dumps])
    mt = mt - mt[0]
    return idx, mt


def main():
    filas = []
    for etiqueta, N, patron in FUENTES:
        import glob
        carpetas = sorted(Path(p) for p in glob.glob(patron) if Path(p).is_dir())[:MAX_POR_FUENTE]
        for c in carpetas:
            r = curva(c)
            if r is None:
                continue
            idx, mt = r
            d = np.diff(mt)
            completa = int(idx[-1]) >= 500
            # ritmo en tramos de 50 dumps
            tramos = {}
            for a in range(0, 501, 50):
                sel = (idx[1:] >= a) & (idx[1:] < a + 50)
                if sel.sum() >= 2:
                    tramos[a] = float(d[sel].mean())
            filas.append(dict(
                fuente=etiqueta, N=N, corrida=c.name, ultimo_dump=int(idx[-1]),
                completa_hasta_tmax=completa, n_dumps=len(idx),
                pared_total_s=round(float(mt[-1]), 1),
                s_por_dump_medio=round(float(mt[-1] / max(idx[-1], 1)), 3),
                s_por_dump_primeros50=round(tramos.get(0, float("nan")), 3),
                s_por_dump_ultimos10=round(float(d[-10:].mean()), 3),
                **{f"s_dump_{a:03d}_{a+50:03d}": round(v, 3) for a, v in tramos.items()}))
            print(f"[{etiqueta}] {c.name}: ultimo={idx[-1]} completa={completa} "
                  f"pared={mt[-1]:.0f}s  s/dump primeros50={tramos.get(0, float('nan')):.2f} "
                  f"ultimos10={d[-10:].mean():.2f}", flush=True)
    if not filas:
        print("sin datos")
        return
    campos = []
    for f in filas:
        for k in f:
            if k not in campos:
                campos.append(k)
    with open(CSV_SALIDA, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[CSV] {len(filas)} corridas -> {CSV_SALIDA}")


if __name__ == "__main__":
    main()
