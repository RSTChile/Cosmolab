#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_N2_puertas_permutadas.py -- N2 de INSTRUCCION_CS075_v4_NULL_PARA_CC.md.

El control que ataca la tesis: ¿el despertar escalonado de los 23 agentes refleja la
cadena causal del inventario (gravedad necesita sobredensidad, EM necesita nucleos...), o
es inevitable con CUALQUIER asignacion de precondiciones porque la temperatura baja
monotonamente y los hitos se cumplen en secuencia fija de todos modos?

Estructura real, verificada en disco (construir_23(), orden de lista): 6 agentes con
requiere=() (sin tocar), 17 agentes con exactamente un hito, en este orden:
  5_debil, 7_masa, 6_catalogo, 16_ssb                    -> T_bajo_electrodebil (4)
  3_fuerte, 8_aniquilacion, 1_espin, 11_tres_cuerpos,
  13_pauli                                                -> T_bajo_confinamiento (5)
  2_gravedad, 12_localidad                                -> hay_sobredensidad (2)
  4_em                                                     -> hay_nucleos (1)
  14_correlacion, M2_memoria, 17_oscuro                    -> hay_atomos (3)
  18_poda, 15_causal                                       -> hay_red (2)

El NULL permuta esa asignacion (que agente espera cual hito) preservando el histograma
4/5/2/1/3/2 exacto -- la misma dificultad que el real, mismo numero de agentes esperando
cada hito. Se logra sobreescribiendo `.requiere` en la INSTANCIA de cada agente (no en la
clase, no en el archivo): Agente23.condiciones_dadas() lee `self.requiere`, que Python
resuelve por instancia antes que por clase -- verificado en disco.

20 permutaciones distintas (generadas con semillas declaradas, no al azar en tiempo de
corrida), cada una x 8 semillas de fisica x 3 amplitudes (0,799 bajo el borde, 1,028 en
el borde, 1,702 sobre el borde -- mismos valores exactos de la grilla v3) = 480
configuraciones. El brazo REAL en esas 3 amplitudes se reusa del JSON de v3, no se
recorre de nuevo.

Caso degenerado (v4 §3): una permutacion puede intercambiar dos agentes que YA comparten
el mismo hito real, dando una asignacion IDENTICA a la real en la practica. Se detecta y
se marca aparte, no se mezcla con las permutaciones genuinamente distintas.
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
SEMILLAS = [12345, 1, 2, 3, 4, 5, 6, 7]

# mismos 3 valores exactos de la grilla v3 (indices 9, 10, 12 de geomspace(0.05,6.0,20))
# geomspace(0.05,6.0,20): idx11=0.7993, idx12=1.0283, idx14=1.7022 -- verificado contra
# la tabla de v3 (ADENDA 5): 0.799294->17/23 siempre, 1.028341->mixto (3/8 en 23/23),
# 1.702152->23/23 siempre.
_GRID_V3 = np.geomspace(0.05, 6.0, 20)
AMP_BAJO = float(_GRID_V3[11])   # 0.799294
AMP_BORDE = float(_GRID_V3[12])  # 1.028341
AMP_ALTO = float(_GRID_V3[14])   # 1.702152
AMPLITUDES = [AMP_BAJO, AMP_BORDE, AMP_ALTO]

ORDEN_AGENTES_UN_HITO = [
    "5_debil", "7_masa", "6_catalogo", "16_ssb",
    "3_fuerte", "8_aniquilacion", "1_espin", "11_tres_cuerpos", "13_pauli",
    "2_gravedad", "12_localidad",
    "4_em",
    "14_correlacion", "M2_memoria", "17_oscuro",
    "18_poda", "15_causal",
]
HITO_SLOTS_REAL = (
    ["T_bajo_electrodebil"] * 4 + ["T_bajo_confinamiento"] * 5 +
    ["hay_sobredensidad"] * 2 + ["hay_nucleos"] * 1 + ["hay_atomos"] * 3 +
    ["hay_red"] * 2
)
assert len(ORDEN_AGENTES_UN_HITO) == 17 == len(HITO_SLOTS_REAL)

ASIGNACION_REAL = dict(zip(ORDEN_AGENTES_UN_HITO, HITO_SLOTS_REAL))


def generar_permutaciones(n=20, seed_max=200):
    """Genera n asignaciones DISTINTAS entre si (permutaciones del multiset de hitos
    sobre los 17 agentes), con semillas declaradas 1..seed_max, en orden. Marca cada una
    como 'degenerada' si coincide exactamente con la asignacion real."""
    vistas = set()
    permutaciones = []
    seed = 0
    while len(permutaciones) < n and seed < seed_max:
        seed += 1
        idx = np.random.RandomState(seed).permutation(17)
        etiquetas = [HITO_SLOTS_REAL[i] for i in idx]
        asignacion = dict(zip(ORDEN_AGENTES_UN_HITO, etiquetas))
        clave = tuple(etiquetas)  # el orden de agentes es fijo, la tupla de etiquetas identifica la asignacion
        if clave in vistas:
            continue  # duplicado de una permutacion ya aceptada -- se descarta, no cuenta
        vistas.add(clave)
        es_degenerada = (asignacion == ASIGNACION_REAL)
        permutaciones.append(dict(
            permutacion_id=len(permutaciones),
            seed_generador=seed,
            asignacion=asignacion,
            es_degenerada=es_degenerada,
        ))
    return permutaciones


def correr_config(args):
    perm_id, asignacion, amp, seed_fisica = args
    from cs075_23_sobre_fisica import construir_23

    t0 = time.time()
    proceso, agentes = construir_23(N=N, dt=DT, seed=seed_fisica, amp_asimetria=amp,
                                     k_enfriamiento=K_ENFRIAMIENTO)
    for ag in agentes:
        if ag.nombre in asignacion:
            ag.requiere = (asignacion[ag.nombre],)  # override de INSTANCIA, no de clase

    for k in range(int(T_TOTAL / DT)):
        proceso.paso(agentes)
    elapsed = time.time() - t0

    informes = [ag.informe() for ag in agentes]
    n_despiertos = sum(1 for i in informes if i["paso_despertar"] is not None)
    orden_despertar = sorted(
        [dict(nombre=i["nombre"], paso=i["paso_despertar"]) for i in informes
         if i["paso_despertar"] is not None],
        key=lambda d: d["paso"],
    )

    return dict(permutacion_id=perm_id, amp_asimetria=amp, seed_fisica=seed_fisica,
                n_despiertos=n_despiertos, orden_despertar=orden_despertar,
                tiempo_config_s=elapsed)


def cargar_brazo_real_v3():
    """Reusa el JSON de v3 -- no se re-corre el brazo real."""
    v3_path = HERE / "cs075_resultado_barrido_v3.json"
    d = json.loads(v3_path.read_text(encoding="utf-8"))
    por_amp = {}
    for amp in AMPLITUDES:
        fila = [r for r in d["resultados"] if abs(r["amp_asimetria"] - amp) < 1e-6]
        assert len(fila) == 8, f"esperaba 8 semillas para amp={amp} en v3, hay {len(fila)}"
        por_amp[amp] = dict(
            n_despiertos_medio=float(np.mean([r["n_despiertos"] for r in fila])),
            n_23_23=sum(1 for r in fila if r["n_despiertos"] == 23),
            valores=[r["n_despiertos"] for r in fila],
        )
    return por_amp


def main():
    t0 = time.time()
    permutaciones = generar_permutaciones(n=20)
    assert len(permutaciones) == 20, f"solo se generaron {len(permutaciones)} distintas"
    n_degeneradas = sum(1 for p in permutaciones if p["es_degenerada"])
    print(f"[N2] {len(permutaciones)} permutaciones generadas, {n_degeneradas} degeneradas "
          f"(identicas a la real)", flush=True)

    trabajos = [(p["permutacion_id"], p["asignacion"], amp, seed)
                for p in permutaciones for amp in AMPLITUDES for seed in SEMILLAS]
    assert len(trabajos) == 480, f"esperaba 480, hay {len(trabajos)}"

    n_workers = min(8, os.cpu_count() or 1)
    print(f"[N2] {len(trabajos)} configuraciones, {n_workers} workers", flush=True)

    with mp.Pool(processes=n_workers) as pool:
        resultados = pool.map(correr_config, trabajos)

    elapsed_total = time.time() - t0
    brazo_real = cargar_brazo_real_v3()

    # --- tabla de 20 filas ---
    tabla = []
    for p in permutaciones:
        pid = p["permutacion_id"]
        fila_perm = [r for r in resultados if r["permutacion_id"] == pid]
        por_amp = {}
        for amp in AMPLITUDES:
            sub = [r for r in fila_perm if abs(r["amp_asimetria"] - amp) < 1e-9]
            por_amp[amp] = dict(
                n_despiertos_medio=float(np.mean([r["n_despiertos"] for r in sub])),
                n_23_23=sum(1 for r in sub if r["n_despiertos"] == 23),
            )
        tabla.append(dict(
            permutacion_id=pid, seed_generador=p["seed_generador"],
            es_degenerada=p["es_degenerada"],
            asignacion=p["asignacion"],
            por_amplitud={str(a): v for a, v in por_amp.items()},
        ))

    # --- veredicto en amp_borde (donde real=8/8) ---
    no_degeneradas = [f for f in tabla if not f["es_degenerada"]]
    n23_no_degeneradas_en_alto = [f["por_amplitud"][str(AMP_ALTO)]["n_23_23"] for f in no_degeneradas]
    real_n23_en_alto = brazo_real[AMP_ALTO]["n_23_23"]

    n_perm_llegan_23_23_en_alto = sum(1 for n in n23_no_degeneradas_en_alto if n == 8)
    if n_perm_llegan_23_23_en_alto == 0:
        veredicto = ("LA ASIGNACION IMPORTA -- ninguna permutacion genuina alcanza 8/8 en "
                     "23/23 donde el real si (amp_alto). La cascada es del inventario, no "
                     "de la monotonia del enfriamiento. Sostiene la tesis.")
    elif n_perm_llegan_23_23_en_alto == len(no_degeneradas):
        veredicto = ("LA CASCADA ES INEVITABLE -- todas las permutaciones genuinas alcanzan "
                     "8/8 en 23/23 igual que el real. La tesis NO queda refutada, pero SI "
                     "queda sin evidencia con este observable.")
    else:
        veredicto = (f"RESULTADO INTERMEDIO -- {n_perm_llegan_23_23_en_alto}/{len(no_degeneradas)} "
                     f"permutaciones genuinas alcanzan 8/8 en 23/23. Revisar que tienen en "
                     f"comun las que fallan.")

    salida = dict(
        N=N, dt=DT, T_total=T_TOTAL, amplitudes=AMPLITUDES, semillas=SEMILLAS,
        orden_agentes_un_hito=ORDEN_AGENTES_UN_HITO, asignacion_real=ASIGNACION_REAL,
        n_permutaciones=len(permutaciones), n_degeneradas=n_degeneradas,
        n_configuraciones=len(trabajos),
        resultados=resultados,
        tabla_20_filas=tabla,
        brazo_real_v3=brazo_real,
        n_permutaciones_llegan_23_23_en_amp_alto=n_perm_llegan_23_23_en_alto,
        real_llega_23_23_en_amp_alto=real_n23_en_alto,
        veredicto=veredicto,
        elapsed_total_s=elapsed_total,
        n_workers=n_workers,
    )

    out = HERE / "cs075_resultado_N2_puertas_permutadas.json"
    out.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n[brazo real v3] " + ", ".join(
        f"amp={a:.4f}: media={brazo_real[a]['n_despiertos_medio']:.2f} 23/23={brazo_real[a]['n_23_23']}/8"
        for a in AMPLITUDES))
    print(f"\n[tabla 20 filas] perm_id | degenerada | " + " | ".join(f"amp={a:.4f}" for a in AMPLITUDES))
    for f in tabla:
        cols = " | ".join(
            f"media={f['por_amplitud'][str(a)]['n_despiertos_medio']:.2f} "
            f"23/23={f['por_amplitud'][str(a)]['n_23_23']}/8"
            for a in AMPLITUDES)
        print(f"  {f['permutacion_id']:2d} | {'SI' if f['es_degenerada'] else 'no':3s} | {cols}")

    print(f"\n[veredicto N2] {veredicto}")
    print(f"  permutaciones genuinas que llegan 8/8 en amp_alto={AMP_ALTO:.4f}: "
          f"{n_perm_llegan_23_23_en_alto}/{len(no_degeneradas)}  (real: {real_n23_en_alto}/8)")
    print(f"\n[archivo] {out}")
    print(f"[costo] {elapsed_total:.1f}s total ({n_workers} workers)")


if __name__ == "__main__":
    main()
