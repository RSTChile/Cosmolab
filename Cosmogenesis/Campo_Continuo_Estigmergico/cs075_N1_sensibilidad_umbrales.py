#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_N1_sensibilidad_umbrales.py -- N1 de INSTRUCCION_CS075_v4_NULL_PARA_CC.md.

Pregunta: el borde de v3 (entre amp=1,028 y 1,323) esta ANCLADO en MIN_PERSISTENCIA=5 y
FACTOR_ATOMOS=2, ambos "DECLARADOS, no anclados" en el codigo (ver docstring de
cs075_23_sobre_fisica.py, hay_atomos). Si el borde se desplaza mas de un escalon de la
grilla al mover esos valores, el borde es mio (artefacto de calibracion), no del modelo.

Parametrizacion SIN editar cs075_23_sobre_fisica.py: MIN_PERSISTENCIA y FACTOR_ATOMOS son
constantes de MODULO (no atributos de instancia), leidas por nombre global dentro de
Proceso23SobreFisica._hitos() en cada llamada (LOAD_GLOBAL tardio de Python). Verificado
en disco: parchear `cs075_23_sobre_fisica.MIN_PERSISTENCIA = X` ANTES de construir el
proceso cambia el valor que _hitos() usa, sin tocar el archivo. Cada worker (spawn) hace
su propio parche antes de correr, para no pisar otros procesos.

9 combinaciones (MIN_PERSISTENCIA en 3/5/10 x FACTOR_ATOMOS en 1/2/4) x 6 amplitudes del
tramo del borde de v3 (indices 10-15 de la grilla, 0,621 a 2,190) x 8 semillas = 432
configuraciones.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N = 16
DT = 1e-3
T_TOTAL = 5.0
K_ENFRIAMIENTO = 50.0

# tramo del borde: mismos 6 puntos que v3 (indices 10-15 de geomspace(0.05,6.0,20))
_GRID_V3 = np.geomspace(0.05, 6.0, 20)
AMPLITUDES = _GRID_V3[10:16].tolist()
SEMILLAS = [12345, 1, 2, 3, 4, 5, 6, 7]

MIN_PERSISTENCIA_VALORES = [3, 5, 10]
FACTOR_ATOMOS_VALORES = [1, 2, 4]


def correr_config(args):
    amp, seed, min_persist, factor_atomos = args
    import cs075_23_sobre_fisica as m

    # parche de modulo -- ver docstring. Se hace ANTES de construir_23() para que
    # Proceso23SobreFisica.__init__ y toda la corrida vean el valor parcheado.
    m.MIN_PERSISTENCIA = min_persist
    m.FACTOR_ATOMOS = factor_atomos

    t0 = time.time()
    proceso, agentes = m.construir_23(N=N, dt=DT, seed=seed, amp_asimetria=amp,
                                       k_enfriamiento=K_ENFRIAMIENTO)
    for k in range(int(T_TOTAL / DT)):
        proceso.paso(agentes)
    elapsed = time.time() - t0

    informes = [ag.informe() for ag in agentes]
    n_despiertos = sum(1 for i in informes if i["paso_despertar"] is not None)

    return dict(amp_asimetria=amp, seed=seed, min_persistencia=min_persist,
                factor_atomos=factor_atomos, n_despiertos=n_despiertos,
                tiempo_config_s=elapsed)


def main():
    t0 = time.time()
    trabajos = [(amp, seed, mp_, fa_)
                for mp_ in MIN_PERSISTENCIA_VALORES
                for fa_ in FACTOR_ATOMOS_VALORES
                for amp in AMPLITUDES
                for seed in SEMILLAS]
    assert len(trabajos) == 432, f"esperaba 432, hay {len(trabajos)}"

    n_workers = min(8, os.cpu_count() or 1)
    print(f"[N1] {len(trabajos)} configuraciones, {n_workers} workers", flush=True)

    with mp.Pool(processes=n_workers) as pool:
        resultados = pool.map(correr_config, trabajos)

    elapsed_total = time.time() - t0

    # --- tabla de 9 filas: por combinacion, amplitud del borde (cruce de 0,5) ---
    tabla = []
    for min_p in MIN_PERSISTENCIA_VALORES:
        for fa in FACTOR_ATOMOS_VALORES:
            fila_combo = [r for r in resultados
                          if r["min_persistencia"] == min_p and r["factor_atomos"] == fa]
            fracciones = []
            for amp in AMPLITUDES:
                sub = [r for r in fila_combo if abs(r["amp_asimetria"] - amp) < 1e-9]
                n23 = sum(1 for r in sub if r["n_despiertos"] == 23)
                fracciones.append(n23 / len(sub))

            borde_entre = None
            for i in range(len(fracciones) - 1):
                if fracciones[i] < 0.5 <= fracciones[i + 1]:
                    borde_entre = (AMPLITUDES[i], AMPLITUDES[i + 1])
                    break
            siempre_completo = all(f == 1.0 for f in fracciones)
            nunca_completo = all(f == 0.0 for f in fracciones)

            tabla.append(dict(
                min_persistencia=min_p, factor_atomos=fa,
                fracciones_23_23=fracciones,
                borde_entre=borde_entre,
                siempre_completo=siempre_completo,
                nunca_completo=nunca_completo,
            ))

    # --- veredicto pre-inscrito (v4 §2) ---
    borde_real_v3 = (AMPLITUDES[2], AMPLITUDES[3])  # (1.028341, 1.323024)
    dentro_del_mismo_escalon = [
        f for f in tabla
        if f["borde_entre"] is not None and abs(f["borde_entre"][0] - borde_real_v3[0]) < 1e-6
    ]
    n_dentro = len(dentro_del_mismo_escalon)
    n_desplazado = sum(1 for f in tabla
                        if f["borde_entre"] is not None
                        and abs(f["borde_entre"][0] - borde_real_v3[0]) >= 1e-6)
    n_sin_borde = sum(1 for f in tabla if f["borde_entre"] is None)

    if n_dentro >= 5:
        veredicto = "EL BORDE ES DEL MODELO -- se mantiene en el mismo escalon en la mayoria de las 9 combinaciones."
    elif n_desplazado > n_dentro:
        veredicto = "EL BORDE ES MIO -- se desplaza mas de un escalon en la mayoria de las 9 combinaciones. El resultado de v3 es artefacto de calibracion."
    else:
        veredicto = "RESULTADO MIXTO -- ver tabla completa, no hay mayoria clara en ninguna direccion."

    salida = dict(
        N=N, dt=DT, T_total=T_TOTAL, amplitudes=AMPLITUDES, semillas=SEMILLAS,
        min_persistencia_valores=MIN_PERSISTENCIA_VALORES,
        factor_atomos_valores=FACTOR_ATOMOS_VALORES,
        n_configuraciones=len(trabajos),
        resultados=resultados,
        tabla_9_filas=tabla,
        borde_real_v3_default=borde_real_v3,
        n_combinaciones_dentro_del_escalon=n_dentro,
        n_combinaciones_desplazadas=n_desplazado,
        n_combinaciones_sin_borde=n_sin_borde,
        veredicto=veredicto,
        elapsed_total_s=elapsed_total,
        n_workers=n_workers,
    )

    out = HERE / "cs075_resultado_N1_sensibilidad_umbrales.json"
    out.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n[tabla 9 filas] MIN_PERSISTENCIA | FACTOR_ATOMOS | fracciones 23/23 por amp | borde_entre")
    for f in tabla:
        print(f"  {f['min_persistencia']:2d} | {f['factor_atomos']:2d} | "
              f"{[round(x,2) for x in f['fracciones_23_23']]} | {f['borde_entre']}")
    print(f"\n[veredicto N1] {veredicto}")
    print(f"  dentro del escalon original: {n_dentro}/9  desplazadas: {n_desplazado}/9  sin borde: {n_sin_borde}/9")
    print(f"\n[archivo] {out}")
    print(f"[costo] {elapsed_total:.1f}s total ({n_workers} workers)")


if __name__ == "__main__":
    main()
