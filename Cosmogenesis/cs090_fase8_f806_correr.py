#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
cs090_fase8_f806_correr.py — FASE VIII F8-06: corre Phantom sobre los brazos `solap` / `disj` a N=4000
=======================================================================================================

QUÉ HACE (a nivel módulo)
-------------------------
Pasa cada condición inicial de `cs090_fase8_f806_n4000.py` por EXACTAMENTE el mismo tubo que usó F7-03
a N=2000 — porque si el tubo cambiara, la comparación entre resoluciones no diría nada de la
resolución:

  1. `phantomsetup_cosmogenesis_backup cosmog` sobre el ASCII ya escrito,
  2. se reescribe el bloque de sumideros + `tmax`/`dtmax` al protocolo estándar de toda la jerarquía
     CS073 / Fase V-B / VI / VII: `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`, `h_acc=0.3`,
     `f_acc=0.8`, `tmax=0.500`, `dtmax=0.001`,
  3. `phantom_cosmogenesis_backup cosmog.in`, midiendo el tiempo de pared real.

**El paso 2 no se reescribe acá: se IMPORTA `editar_cosmog_in` de `cs090_fase7_f703_correr.py`**, que es
el archivo que corrió los puntos de N=2000 que este experimento quiere comparar. Así el bloque de
sumideros es literalmente el mismo texto, con el mismo `assert` que aborta si el binario de
`phantomsetup` cambió y el bloque por defecto ya no coincide.

DIFERENCIAS CON EL RUNNER DE N=2000, TODAS DECLARADAS
------------------------------------------------------
  * `BASE` apunta a `bateria_fase8_f806_n4000` (prefijo `f806`, jamás usado antes en el proyecto);
  * la condición inicial tiene **4002 líneas** (cabecera + parámetros + 4000 partículas), no 2002;
  * el tope de pared por corrida sube de 20 a 30 min. A N=4000 las 8 corridas de O3-A tardaron 18-57 s,
    así que el tope no debería tocarse nunca; está sólo como salvaguarda contra el colapso del paso de
    tiempo que F8-04 documentó a N=8000.

No recomputa una carpeta que ya tenga `cosmog_00500`. No declara cierre ni veredicto. Sin commits.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs090_fase7_f703_correr import editar_cosmog_in, PHANTOM, PHANTOMSETUP   # sólo import

BASE = Path("/Users/alexis/phantom_cs073/bateria_fase8_f806_n4000")
N_PARTICULAS = 4000
LINEAS_IC_ESPERADAS = N_PARTICULAS + 2
TIEMPO_LIMITE_UNA_CORRIDA_S = 30 * 60


def correr_una(carpeta: Path) -> dict:
    ic = carpeta / "cosmogenesis_ic.txt"
    assert ic.exists(), f"falta {ic} -- correr cs090_fase8_f806_n4000.py primero"
    if (carpeta / "cosmog_00500").exists():
        return dict(carpeta=str(carpeta), exit_setup=0, t_setup=0.0, exit_run=0, t_run=0.0,
                    ya_corrida=True)

    t0 = time.time()
    with open(carpeta / "setup.log", "w") as f:
        r_setup = subprocess.run([PHANTOMSETUP, "cosmog"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                 stdout=f, stderr=subprocess.STDOUT)
    t_setup = time.time() - t0
    assert r_setup.returncode == 0, f"phantomsetup falló en {carpeta} (ver setup.log)"

    editar_cosmog_in(carpeta / "cosmog.in")

    t1 = time.time()
    with open(carpeta / "run.log", "w") as f:
        r_run = subprocess.run([PHANTOM, "cosmog.in"], cwd=carpeta, stdin=subprocess.DEVNULL,
                               stdout=f, stderr=subprocess.STDOUT,
                               timeout=TIEMPO_LIMITE_UNA_CORRIDA_S)
    t_run = time.time() - t1

    return dict(carpeta=str(carpeta), exit_setup=r_setup.returncode, t_setup=round(t_setup, 2),
                exit_run=r_run.returncode, t_run=round(t_run, 2))


def main(carpetas):
    t_inicio = time.time()
    for carpeta in carpetas:
        print(f"[{carpeta.name}] setup + run...", flush=True)
        info = correr_una(carpeta)
        if info.get("ya_corrida"):
            print(f"[{carpeta.name}] ya tenía cosmog_00500 -- se salta", flush=True)
            continue
        print(f"[{carpeta.name}] exit_setup={info['exit_setup']} t_setup={info['t_setup']}s  "
              f"exit_run={info['exit_run']} t_run={info['t_run']}s", flush=True)
        if info["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-20:]
            print("  AVISO: exit_run != 0. Tail run.log:\n  " + "\n  ".join(tail), flush=True)
    print(f"\n[TOTAL] tiempo script={time.time()-t_inicio:.1f}s", flush=True)


if __name__ == "__main__":
    # uso: python3.9 cs090_fase8_f806_correr.py [patron_glob]
    # Misma salvaguarda que F7-02/F7-03: que el archivo de condición inicial EXISTA no significa que
    # esté COMPLETO — el generador tarda minutos en escribirlo y este runner corre en paralelo. Se exige
    # (a) `meta_regla.json`, que el generador escribe DESPUÉS del IC, y (b) las 4002 líneas exactas.
    patron = sys.argv[1] if len(sys.argv) > 1 else "*"
    carpetas = sorted(c for c in BASE.glob(patron)
                      if c.is_dir() and (c / "cosmogenesis_ic.txt").exists()
                      and (c / "meta_regla.json").exists()
                      and len((c / "cosmogenesis_ic.txt").read_text().splitlines())
                      == LINEAS_IC_ESPERADAS)
    print(f"patron={patron}: {len(carpetas)} carpetas -> {[c.name for c in carpetas]}")
    main(carpetas)
