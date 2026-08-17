#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs090_fase6_o3f_extraer_gas.py — PASO 1 de O3-F: extracción cruda del gas
==========================================================================

Qué hago (código autodescriptivo)
---------------------------------
Leo, de cada corrida de Phantom de la Fase V-B (los 40 pares Clase I vs Clase III de
A2-B0-C2), DOS volcados:
  * `cosmog_00000` — condición inicial (t=0), que uso sólo como ANCLA de densidad:
    me da la escala de densidad "sin colapsar" propia de esa corrida.
  * `cosmog_00500` — estado final (t=0.5 en unidades de código), donde vive el
    observable.

De cada volcado guardo, en un caché `.npz` por corrida, los arreglos crudos que el
paso 2 necesita: posiciones (x,y,z), velocidades (vx,vy,vz), densidad SPH (rho),
longitud de suavizado (h), más la metadata (masa por partícula, tiempo, nº de
sumideros y masa acretada).

Por qué en dos pasos
--------------------
Leer 152 volcados binarios es lo caro (~2-3 min). El análisis de B_τ se va a repetir
muchas veces (varios umbrales, varias definiciones de entropía, remuestreos), y no
quiero pagar la lectura binaria cada vez. Además, congelar el crudo en disco hace que
el paso 2 sea auditable y reproducible sin tocar Phantom.

No modifico NADA existente: reuso `leer_volcado_phantom.py` (congelado) tal cual, y
escribo sólo archivos nuevos con prefijo `cs090_fase6_o3f_`.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leer_volcado_phantom import leer_dump  # noqa: E402

RAIZ_PROYECTO = Path(__file__).resolve().parent
RAIZ_PHANTOM = Path("/Users/alexis/phantom_cs073")

# Las cinco tandas donde pueden vivir las corridas de los 40 pares de Fase V-B.
BATERIAS = [
    "bateria_fase5b_a2b0c2_piloto",
    "bateria_fase5b_a2b0c2_escala_v2",
    "bateria_fase5b_a2b0c2_escala_v3",
    "bateria_fase5b_a2b0c2_escala_v4",
    "bateria_fase6_outliers_negativos",
]

CSV_PARES = RAIZ_PROYECTO / "cs090_fase5b_TOTAL_40pares.csv"
DIR_CACHE = RAIZ_PROYECTO / "cs090_fase6_o3f_cache"


def resolver_carpeta(rule_id: str, rol: str) -> Path:
    """Encuentro la carpeta de corrida de una regla+rol. El CSV de los 40 pares trae
    la columna `carpeta` a veces absoluta y a veces relativa, así que no me fío de
    ella: busco `<rule_id>_<rol>` en las cinco tandas conocidas."""
    nombre = f"{rule_id}_{rol}"
    candidatas = [RAIZ_PHANTOM / b / nombre for b in BATERIAS]
    existentes = [c for c in candidatas if c.is_dir()]
    if not existentes:
        raise FileNotFoundError(f"no encuentro carpeta de corrida para {nombre}")
    if len(existentes) > 1:
        raise RuntimeError(f"nombre ambiguo {nombre}: {existentes}")
    return existentes[0]


def extraer_una(carpeta: Path, destino: Path) -> dict:
    """Leo IC y volcado final de una corrida y dejo el crudo en un .npz."""
    d_ini = carpeta / "cosmog_00000"
    d_fin = carpeta / "cosmog_00500"
    if not d_ini.is_file() or not d_fin.is_file():
        raise FileNotFoundError(f"faltan volcados en {carpeta}")

    gas0, _ = leer_dump(d_ini)
    gasF, sinks = leer_dump(d_fin)

    datos = dict(
        x=gasF["x"].to_numpy(np.float64),
        y=gasF["y"].to_numpy(np.float64),
        z=gasF["z"].to_numpy(np.float64),
        vx=gasF["vx"].to_numpy(np.float64),
        vy=gasF["vy"].to_numpy(np.float64),
        vz=gasF["vz"].to_numpy(np.float64),
        rho=gasF["rho"].to_numpy(np.float64),
        h=gasF["h"].to_numpy(np.float64),
        rho0=gas0["rho"].to_numpy(np.float64),
    )
    meta = dict(
        carpeta=str(carpeta),
        n_gas_final=len(gasF),
        n_gas_inicial=len(gas0),
        masa_por_particula=float(gasF.params["massoftype"]),
        tiempo_final=float(gasF.params["time"]),
        n_sumideros=0 if sinks is None else int(len(sinks)),
        masa_sumideros=0.0 if sinks is None else float(np.sum(sinks["m"].to_numpy())),
        rho0_mediana=float(np.median(datos["rho0"])),
    )
    np.savez_compressed(destino, **datos, **{f"meta_{k}": v for k, v in meta.items()})
    return meta


def main() -> None:
    DIR_CACHE.mkdir(exist_ok=True)
    with open(CSV_PARES, newline="") as fh:
        filas = list(csv.DictReader(fh))

    corridas = {}  # nombre_corrida -> carpeta
    for f in filas:
        nombre = f"{f['rule_id']}_{f['rol']}"
        corridas[nombre] = resolver_carpeta(f["rule_id"], f["rol"])
    print(f"{len(filas)} filas del CSV -> {len(corridas)} corridas únicas a leer")

    resumen = []
    t0 = time.time()
    for i, (nombre, carpeta) in enumerate(sorted(corridas.items()), 1):
        destino = DIR_CACHE / f"{nombre}.npz"
        if destino.is_file():
            print(f"[{i}/{len(corridas)}] {nombre}: ya en caché")
            continue
        meta = extraer_una(carpeta, destino)
        resumen.append(dict(corrida=nombre, **meta))
        print(
            f"[{i}/{len(corridas)}] {nombre}: gas {meta['n_gas_inicial']}->"
            f"{meta['n_gas_final']}, {meta['n_sumideros']} sumideros, "
            f"m_sink={meta['masa_sumideros']:.1f} ({time.time()-t0:.0f}s)"
        )

    if resumen:
        salida = RAIZ_PROYECTO / "cs090_fase6_o3f_corridas_leidas.csv"
        with open(salida, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
            w.writeheader()
            w.writerows(resumen)
        print(f"escrito {salida}")
    print(f"listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
