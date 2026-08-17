"""
cs090_fase6_o3a_verificar_ic.py — FASE VI, tarea O3-A: chequeo de integridad de las condiciones
iniciales generadas a resolución alta, ANTES de correr Phantom sobre ellas (11-ago-2026).

POR QUÉ EXISTE ESTE ARCHIVO
---------------------------
Los layouts a N=4000 se generaron con varios procesos en paralelo (el paso caro, `layout_resortes`, es
O(N²) y de un solo hilo). En un par de casos la misma regla quedó encolada dos veces — una en la cola
principal y otra lanzada a mano para completar su par antes — así que dos procesos pudieron escribir el
MISMO archivo `cosmogenesis_ic.txt` al mismo tiempo. El contenido que escriben es idéntico (todo el
pipeline es determinista: misma semilla de layout, misma semilla de turbulencia, mismo grafo), pero dos
escrituras entrelazadas pueden dejar un archivo truncado o con una línea partida al medio, y eso
Phantom lo leería como basura silenciosa.

Este script verifica, para cada carpeta:
  - que la cabecera declare el N correcto y la masa por partícula correcta (masa_total 18800 / N);
  - que el archivo tenga EXACTAMENTE N+2 líneas (2 de cabecera + una por partícula);
  - que cada línea de partícula tenga 7 números parseables (x,y,z,vx,vy,vz,h) y ninguno NaN/inf;
  - que la masa total reconstruida dé 18800 (la masa física fija de toda la jerarquía CS073).

Las que fallen se listan para regenerarlas; no se corre Phantom sobre ellas. No modifica nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion")
MASA_TOTAL_OBJETIVO = 18800.0


def verificar(carpeta: Path) -> tuple[bool, str]:
    ic = carpeta / "cosmogenesis_ic.txt"
    if not ic.exists():
        return False, "no existe cosmogenesis_ic.txt"
    lineas = ic.read_text().splitlines()
    if len(lineas) < 3:
        return False, f"sólo {len(lineas)} líneas"
    try:
        n_decl, masa_part, _hfact, _polyk = lineas[1].split()
        n_decl, masa_part = int(n_decl), float(masa_part)
    except Exception as e:
        return False, f"segunda línea ilegible: {e}"
    if len(lineas) != n_decl + 2:
        return False, f"líneas={len(lineas)}, se esperaban {n_decl + 2} (archivo truncado o duplicado)"
    try:
        datos = np.array([[float(x) for x in ln.split()] for ln in lineas[2:]])
    except Exception as e:
        return False, f"línea de partícula ilegible: {e}"
    if datos.shape != (n_decl, 7):
        return False, f"forma {datos.shape}, se esperaba ({n_decl}, 7)"
    if not np.isfinite(datos).all():
        return False, "hay NaN o inf"
    masa_total = masa_part * n_decl
    if abs(masa_total - MASA_TOTAL_OBJETIVO) > 1e-6:
        return False, f"masa total {masa_total} != {MASA_TOTAL_OBJETIVO}"
    return True, f"ok N={n_decl} masa_particula={masa_part} masa_total={masa_total}"


if __name__ == "__main__":
    malas = []
    for carpeta in sorted(BASE.glob("N*/*")):
        if not carpeta.is_dir():
            continue
        ok, msg = verificar(carpeta)
        print(f"[{'OK ' if ok else 'MAL'}] {carpeta.parent.name}/{carpeta.name}: {msg}")
        if not ok:
            malas.append(carpeta)
    print(f"\n{len(malas)} condición(es) inicial(es) con problema")
    sys.exit(1 if malas else 0)
