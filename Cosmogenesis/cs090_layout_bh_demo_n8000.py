#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cs090_layout_bh_demo_n8000.py — demostración end-to-end: grafo → layout Barnes-Hut → IC → Phantom, a N=8000
============================================================================================================

POR QUÉ EXISTE
--------------
`FASE6_O3A_convergencia_resolucion_CS.md` §8 dejó N=8000 fuera de la serie por costo: el layout
Fruchterman-Reingold O(N²) pedía ~73 min por grafo y dos intentos con la máquina libre se abortaron a los
17 min sin terminar un solo layout. Este script prueba que, con el layout O(N log N) de
`cs090_layout_barnes_hut.py`, la CADENA COMPLETA a N=8000 corre: grafo relacional → layout → condición
inicial de masa fija → phantomsetup → Phantom → métricas.

QUÉ SE MANTIENE IDÉNTICO AL PROTOCOLO DE LA SERIE
--------------------------------------------------
Todo lo que no sea el layout: masa total fija 18800 (no escala con N), lado de caja fijo 2000^(1/3),
`Expansion` de 60 pasos, turbulencia Mach=3 semilla 42, hfact/polyk de `fase1_traducir_a_phantom`,
100 iteraciones de layout, semilla de layout 12345, 14 sweeps del motor relacional, y el bloque de
Phantom `icreate_sinks=1 / rho_crit_cgs=1000 / r_crit=0.600 / h_acc=0.300 / f_acc=0.800 / tmax=0.500 /
dtmax=0.001`. El bloque de `cosmog.in` se reescribe con `editar_cosmog_in`, IMPORTADA de
`cs090_fase5b_correr.py` (congelado) — no se reimplementa ni se edita a ciegas.

Lo único que cambia respecto de `generar_ic_masa_fija_desde_grafo` es la llamada al layout. La receta de
la IC se replica aquí (misma dilatación, mismo campo de velocidad, mismo formato de archivo, mismo
encabezado) en vez de modificar el adaptador congelado.

DIFERENCIA CON `cs090_fase5b_correr.correr_una`: el límite de tiempo. Aquel tiene 20 min fijos, pensados
para N=2000. Acá el límite es un parámetro (por defecto 35 min) porque a N=8000 el costo de Phantom es
justamente lo que se quiere MEDIR, y Phantom escribe dumps intermedios: si se agota el límite, se analiza
el último dump disponible y se reporta como corrida incompleta, no se descarta.

USO
    ./venv/bin/python cs090_layout_bh_demo_n8000.py <rule_id> <seed> <clase> [N] [theta] [limite_s]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs090_layout_barnes_hut import layout_barnes_hut
from cs090_fase5b_phantom_adaptador import (reconstruir_regla_a2b0c2, LADO_FIJO, MASA_TOTAL_OBJETIVO,
                                            TURB_SEED, CS_SONORA, N_REFERENCIA_LADO)
from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from fase1_traducir_a_phantom import HFACT, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from cs090_fase5b_correr import editar_cosmog_in, PHANTOMSETUP, PHANTOM   # congelado: sólo import
from cs090_fase5b_analizar import analizar_carpeta                        # congelado: sólo import
import cs090_diam_corregido as DIAMC

BASE = Path("/Users/alexis/phantom_cs073/infra_layout_bh_demo_n8000")
SEED_LAYOUT = 12345
N_SWEEPS = 14
ITERS_LAYOUT = 100


def escribir_ic_bh(adj_lista, N, ruta_salida, theta, seed_layout=SEED_LAYOUT,
                   iters_layout=ITERS_LAYOUT, n_pasos_expansion=60):
    """Misma receta que `generar_ic_masa_fija_desde_grafo`, con `layout_barnes_hut` en lugar de
    `layout_resortes`. Devuelve el mismo dict de información más el tiempo del layout."""
    adj = {i: adj_lista[i] for i in range(N)}
    n_aristas = sum(len(v) for v in adj_lista) // 2

    diag = {}
    t0 = time.time()
    pos = layout_barnes_hut(adj, N, lado=LADO_FIJO, iters=iters_layout, seed=seed_layout,
                            theta=theta, diagnostico=diag)
    t_layout = time.time() - t0

    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    vel = vel_gen(pos, adj, np.ones(N))
    h_guess = np.full(N, HFACT)
    masa_particula = MASA_TOTAL_OBJETIVO / N

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 INFRA layout Barnes-Hut theta={theta} -- lado_caja_fijo="
                f"{LADO_FIJO:.6f} (=N_referencia={N_REFERENCIA_LADO}^(1/3), NO depende de n) "
                f"masa_total_objetivo={MASA_TOTAL_OBJETIVO:.6g} (NO escala con n) -- "
                f"npart={N} n_aristas={n_aristas} masa_particula={masa_particula:.17g} hfact={HFACT} "
                f"polyk={POLYK:.17g} seed_layout={seed_layout} con_turbulencia=True\n")
        f.write(f"{N} {masa_particula:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(N):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=str(ruta_salida), n=N, masa_particula=masa_particula, lado=LADO_FIJO,
                a_final=a_final, n_aristas=n_aristas, seed_layout=seed_layout, theta=theta,
                t_layout_s=round(t_layout, 1), prof_octree=diag.get("prof"),
                interacciones_por_iter=diag.get("interacciones"))


def correr_phantom(carpeta: Path, limite_s: int) -> dict:
    """Igual que `cs090_fase5b_correr.correr_una` pero con el límite de tiempo como parámetro."""
    t0 = time.time()
    with open(carpeta / "setup.log", "w") as f:
        r = subprocess.run([PHANTOMSETUP, "cosmog"], cwd=carpeta, stdin=subprocess.DEVNULL,
                           stdout=f, stderr=subprocess.STDOUT)
    t_setup = time.time() - t0
    assert r.returncode == 0, f"phantomsetup falló en {carpeta} (ver setup.log)"
    editar_cosmog_in(carpeta / "cosmog.in")
    t1 = time.time()
    timeout = False
    try:
        with open(carpeta / "run.log", "w") as f:
            rr = subprocess.run([PHANTOM, "cosmog.in"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT, timeout=limite_s)
        exit_run = rr.returncode
    except subprocess.TimeoutExpired:
        exit_run, timeout = None, True
    return dict(exit_setup=r.returncode, t_setup_s=round(t_setup, 1),
                exit_run=exit_run, t_run_s=round(time.time() - t1, 1), timeout=timeout)


def main(rule_id, seed, clase, N=8000, theta=0.5, limite_s=35 * 60):
    carpeta = BASE / f"N{N}" / f"{rule_id}_{clase}"
    carpeta.mkdir(parents=True, exist_ok=True)
    t_ini = time.time()
    ic = carpeta / "cosmogenesis_ic.txt"
    meta_p = carpeta / "meta_regla.json"

    if ic.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
    else:
        t0 = time.time()
        p, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=N_SWEEPS)
        t_grafo = time.time() - t0
        info = escribir_ic_bh(m["adj_final"], N, ic, theta)
        # OJO: `info` ya trae `theta` (lo devuelve `escribir_ic_bh`), así que NO se repite acá — pasarlo
        # dos veces revienta con "dict() got multiple values for keyword argument 'theta'".
        meta = dict(rule_id=rule_id, clase=clase, seed=seed, N=N,
                    layout="barnes_hut", t_grafo_s=round(t_grafo, 1),
                    K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                    n_aristas_grafo_final=m["n_aristas"],
                    grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                    diam_grafo_final_OFICIAL=DIAMC.diam_gigante(m["adj_final"], N),
                    holon_grafo_final=m["holonomia"], **info)
        meta_p.write_text(json.dumps(meta, indent=2, default=str))
        print(f"[IC] {rule_id} N={N} grafo={meta['t_grafo_s']}s layout_BH={meta['t_layout_s']}s",
              flush=True)

    info_run = correr_phantom(carpeta, limite_s)
    meta.update(info_run)
    try:
        meta.update({f"metrica_{k}": v for k, v in analizar_carpeta(carpeta).items()})
    except Exception as e:
        meta["error_analisis"] = f"{type(e).__name__}: {e}"
    meta["t_total_s"] = round(time.time() - t_ini, 1)
    (carpeta / "resultado_demo.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[OK] {rule_id} clase={clase} N={N} t_layout={meta.get('t_layout_s')}s "
          f"t_phantom={meta.get('t_run_s')}s timeout={meta.get('timeout')} "
          f"frac={meta.get('metrica_fraccion_masa_en_sumideros')} "
          f"nsink={meta.get('metrica_n_sumideros')} dump={meta.get('metrica_n_dump_final')} "
          f"t_total={meta['t_total_s']}s", flush=True)
    return meta


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2],
         N=int(a[3]) if len(a) > 3 else 8000,
         theta=float(a[4]) if len(a) > 4 else 0.5,
         limite_s=int(a[5]) if len(a) > 5 else 35 * 60)
