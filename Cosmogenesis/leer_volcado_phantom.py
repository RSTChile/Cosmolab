#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leer_volcado_phantom.py — Lector de volcados binarios de Phantom para Cosmogénesis
====================================================================================

Quién soy / qué hago (código autodescriptivo):
  Resuelve el bloqueo de infraestructura identificado en
  DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md (nodo C-N4, sección 7) y en el
  roadmap multi-IA del 5-ago-2026 (frente #11): hasta ahora Cosmogénesis no tenía forma
  de leer, en Python, las posiciones y densidades de TODAS las partículas de gas de una
  corrida de Phantom (formato binario Fortran "sphNG/Phantom") — sólo los `.sink`
  (sumideros) eran legibles como texto plano.

  Los volcados completos son los archivos SIN extensión `.sink`, tipo `cosmog_00030`,
  dentro de cada carpeta de corrida (p.ej. /Users/alexis/phantom_cs073/bateria_n2000/
  ic_real/, ic_null1/ .. ic_null8/). El identificador binario embebido confirma el
  formato: "FT:Phantom:2026.0.1: (hydro+grav)".

Qué método usé:
  La librería `sarracen` (paquete Python de la propia comunidad de Phantom/SPH,
  https://github.com/ttricco/sarracen) sabe parsear ese binario directamente —
  NO hace falta compilar utilidades de Fortran (phantom2gadget, phantom2hdf5, etc.).
  Se instaló en el venv del proyecto (venv/). El único tropiezo fue que `pip install
  sarracen` intentaba compilar `llvmlite` desde fuente (sin wheel para Python 3.13) por
  cómo resuelve pip las versiones de `numba`; se resolvió instalando primero
  `numba`+`llvmlite` más recientes con --only-binary=:all: y luego sarracen sin problema.

Qué devuelve:
  leer_dump(path) -> (gas: SarracenDataFrame, sinks: SarracenDataFrame | None)
    `sarracen.read_phantom` devuelve UN solo DataFrame si el volcado no tiene todavía
    partículas sumidero (típico al inicio de la corrida), o una LISTA [gas, sinks] una
    vez que nacieron sumideros (columnas de sinks: x,y,z,m,h,maccreted,...). Esta
    función normaliza ambos casos: siempre devuelve la tupla (gas, sinks_o_None).
    Columnas mínimas garantizadas en `gas`: x, y, z, vx, vy, vz, h (longitud de
    suavizado), rho (densidad, calculada vía calc_density() con la masa por partícula
    del propio dump — `massoftype` en gas.params). `gas.params` trae metadata (tiempo,
    masa por partícula, nparttot, etc.) del propio volcado.

Cómo se usa:
    from leer_volcado_phantom import leer_dump, listar_dumps
    gas, sinks = leer_dump("/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog_00030")
    print(gas[["x", "y", "z", "rho"]].describe())
    if sinks is not None:
        print(f"{len(sinks)} sumideros, masa total={sinks['m'].sum():.3g}")

    for p in listar_dumps("/Users/alexis/phantom_cs073/bateria_n2000/ic_null3"):
        sdf = leer_dump(p)
        ...

Verificación de éxito mínimo (corrida como smoke test al ejecutar este archivo
directamente): lee un dump de ic_real y uno de ic_null1, calcula densidad, confirma
2000 partículas, sin NaN, y rangos de densidad físicamente sensatos (no artefacto).

Qué falta para escalar esto (no resuelto acá, a propósito — este script es sólo la
herramienta, no un experimento):
  - Leer TODOS los pasos temporales de una corrida (no sólo un dump) para análisis de
    evolución temporal → usar listar_dumps() + un loop, cuidando memoria si son muchos.
  - Para corridas grandes (N8550, test_massiva) verificar tiempo de lectura por dump
    antes de asumir que escalar a un barrido completo es barato.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import sarracen

warnings.filterwarnings("ignore", category=UserWarning, module="sarracen")


def leer_dump(path: str | Path):
    """Lee un volcado binario de Phantom. Devuelve (gas, sinks) — sinks es None si
    el volcado todavía no tiene partículas sumidero (sarracen sólo devuelve lista
    [gas, sinks] una vez que existen)."""
    resultado = sarracen.read_phantom(str(path))
    if isinstance(resultado, list):
        gas, sinks = resultado[0], resultado[1]
    else:
        gas, sinks = resultado, None
    gas.calc_density()
    return gas, sinks


def listar_dumps(carpeta: str | Path) -> list[Path]:
    """Devuelve, ordenados, los paths de los volcados binarios completos de una
    carpeta de corrida de Phantom (excluye .sink, .ev, .in y cualquier otro
    archivo auxiliar)."""
    carpeta = Path(carpeta)
    dumps = [
        p
        for p in carpeta.iterdir()
        if p.is_file()
        and p.name.startswith("cosmog_")
        and p.suffix == ""
        and p.stem[7:].isdigit()
    ]
    return sorted(dumps, key=lambda p: p.stem)


def _smoke_test() -> None:
    base = Path("/Users/alexis/phantom_cs073/bateria_n2000")
    casos = [base / "ic_real", base / "ic_null1"]
    for carpeta in casos:
        dumps = listar_dumps(carpeta)
        assert dumps, f"no se encontraron volcados en {carpeta}"
        objetivo = dumps[len(dumps) // 2]  # uno intermedio, no el t=0
        gas, sinks = leer_dump(objetivo)
        n = len(gas)
        nan = gas[["x", "y", "z", "rho"]].isna().any().any()
        rho = gas["rho"]
        extra = f", {len(sinks)} sumideros" if sinks is not None else ""
        print(
            f"{carpeta.name}: {objetivo.name} -> "
            f"{n} partículas gas{extra}, NaN={bool(nan)}, "
            f"rho[min={rho.min():.4g}, mediana={rho.median():.4g}, max={rho.max():.4g}]"
        )
        assert not nan, "se encontraron NaN — el volcado no es confiable"
        assert (rho > 0).all(), "densidad no positiva — el volcado no es confiable"
    print("OK — lector de volcados Phantom verificado sobre ic_real e ic_null1.")


if __name__ == "__main__":
    _smoke_test()
