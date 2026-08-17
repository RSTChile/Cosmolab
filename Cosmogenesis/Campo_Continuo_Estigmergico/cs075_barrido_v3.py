#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_barrido_v3.py -- El barrido de INSTRUCCION_CS075_v3_BARRIDO_PARA_CC.md.

Pregunta: entre amp_asimetria=0,5 (17/23 despiertos) y 2,0 (23/23), el universo pasa de
incompleto a completo (visto en el smoke, ADENDA 4). Este barrido localiza el borde,
revisa si hay techo (banda vs semirrecta) y mide la dispersión entre semillas -- lo mas
importante: si el 23/23 fue suerte de la semilla 12345 o es del modelo.

20 amplitudes log-espaciadas de 0,05 a 6,0 (razon 1,287, verificado: np.geomspace(0.05,
6.0, 20) reproduce sola los 4 tramos que describe la instruccion -- no hace falta
espaciar cada tramo por separado, es una unica grilla log-uniforme).
8 semillas fijas, declaradas (NO generadas al azar en tiempo de corrida).
160 configuraciones = 20 x 8. dt=1e-3, N=16, T_TOTAL=5,0 (5.000 pasos, igual que
cs075_smoke_23_sobre_fisica.py v2, ya validado -- 139x margen sobre el cruce de
confinamiento en el paso 36).

Import-only de cs075_23_sobre_fisica.construir_23 -- no se toca cs075_base_fisica.py,
cs075_arquitectura_agentes.py ni cs075_23_sobre_fisica.py.
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
N_PASOS = int(T_TOTAL / DT)

AMPLITUDES = np.geomspace(0.05, 6.0, 20).tolist()
SEMILLAS = [12345, 1, 2, 3, 4, 5, 6, 7]  # 12345 = la de todo el experimento anterior

NIVEL_DE = {
    "23_campo": 0, "22_qcd": 0, "9_expansion": 0, "10_enfriamiento": 0,
    "M1_semilla": 0, "M3_fase_cuantica": 0,
    "5_debil": 1, "7_masa": 1, "6_catalogo": 1, "16_ssb": 1,
    "3_fuerte": 2, "8_aniquilacion": 2, "1_espin": 2, "11_tres_cuerpos": 2, "13_pauli": 2,
    "2_gravedad": 3, "12_localidad": 3,
    "4_em": 4,
    "14_correlacion": 5, "M2_memoria": 5, "17_oscuro": 5,
    "18_poda": 6, "15_causal": 6,
}


def correr_config(args):
    amp, seed = args
    from cs075_23_sobre_fisica import construir_23  # import dentro del worker (spawn)

    # OJO: proceso._hitos() tiene efecto secundario (actualiza contador_sobredenso, el
    # contador de persistencia para hay_nucleos/hay_atomos/hay_red). proceso.paso() ya
    # la llama UNA vez por paso, internamente. Llamarla de nuevo acá para "leer" el hito
    # duplicaria el contador y corromperia esos tres hitos -- se evita por completo: el
    # paso en que cada hito se cumple se deriva de paso_despertar de los agentes (todos
    # los `requiere` son de UN solo hito, verificado por grep), y expansion_supraluminica
    # (que ningun agente usa como puerta) se lee de estado.es_supraluminico() -- metodo
    # puro, sin efecto secundario, verificado en cs075_base_fisica.py.
    t0 = time.time()
    proceso, agentes = construir_23(N=N, dt=DT, seed=seed, amp_asimetria=amp,
                                     k_enfriamiento=K_ENFRIAMIENTO)
    paso_supraluminico = None
    for k in range(N_PASOS):
        proceso.paso(agentes)
        if paso_supraluminico is None and proceso.estado.es_supraluminico():
            paso_supraluminico = k + 1
    elapsed = time.time() - t0
    hitos_final = proceso._hitos()  # UNA sola llamada extra, al final -- ya no importa
    # (el barrido termina acá; el contador de persistencia no se vuelve a leer)

    informes = [ag.informe() for ag in agentes]
    despiertos = [i for i in informes if i["paso_despertar"] is not None]
    n_despiertos = len(despiertos)
    nivel_maximo = max((NIVEL_DE[i["nombre"]] for i in despiertos), default=-1)

    hitos_por_paso = {"expansion_supraluminica": paso_supraluminico}
    for hito_nombre in ("T_bajo_electrodebil", "T_bajo_confinamiento", "hay_sobredensidad",
                        "hay_nucleos", "hay_atomos", "hay_red"):
        candidatos = [ag.paso_despertar for ag in agentes
                      if ag.requiere == (hito_nombre,) and ag.paso_despertar is not None]
        hitos_por_paso[hito_nombre] = min(candidatos) if candidatos else None

    return dict(
        amp_asimetria=amp, seed=seed,
        n_despiertos=n_despiertos,
        nivel_maximo_alcanzado=nivel_maximo,
        hitos_paso=hitos_por_paso,
        hitos_finales={k: v for k, v in hitos_final.items()
                       if k not in ("n_celdas_sobredensas", "n_celdas_nucleo",
                                    "n_celdas_atomo", "n_regiones_atomo", "T_media")},
        n_celdas_sobredensas=hitos_final["n_celdas_sobredensas"],
        n_celdas_nucleo=hitos_final["n_celdas_nucleo"],
        n_celdas_atomo=hitos_final["n_celdas_atomo"],
        n_regiones_atomo=hitos_final["n_regiones_atomo"],
        estado_final=proceso.estado.estado(),
        informes_agentes=informes,
        ms_por_paso=elapsed / N_PASOS * 1000,
        tiempo_config_s=elapsed,
    )


def main():
    t0 = time.time()
    trabajos = [(amp, seed) for amp in AMPLITUDES for seed in SEMILLAS]
    assert len(trabajos) == 160, f"esperaba 160 configuraciones, hay {len(trabajos)}"

    n_workers = min(8, os.cpu_count() or 1)
    print(f"[barrido] {len(trabajos)} configuraciones, {n_workers} workers", flush=True)

    with mp.Pool(processes=n_workers) as pool:
        resultados = pool.map(correr_config, trabajos)

    elapsed_total = time.time() - t0
    costo_medio_s = sum(r["tiempo_config_s"] for r in resultados) / len(resultados)

    # --- tabla de 20 filas (v3 §6) ---
    tabla = []
    for amp in AMPLITUDES:
        fila = [r for r in resultados if abs(r["amp_asimetria"] - amp) < 1e-12]
        n_desp = [r["n_despiertos"] for r in fila]
        n_23 = sum(1 for r in fila if r["n_despiertos"] == 23)
        tabla.append(dict(
            amp_asimetria=amp,
            media_n_despiertos=float(np.mean(n_desp)),
            min_n_despiertos=int(np.min(n_desp)),
            max_n_despiertos=int(np.max(n_desp)),
            std_n_despiertos=float(np.std(n_desp)),
            semillas_con_23_23=n_23,
            fraccion_23_23=n_23 / len(fila),
            nivel_maximo_alcanzado=max(r["nivel_maximo_alcanzado"] for r in fila),
        ))

    # --- 5.1 el borde ---
    fracciones = [f["fraccion_23_23"] for f in tabla]
    cruza_en = None
    escalon_o_rampa = None
    for i in range(len(fracciones) - 1):
        if fracciones[i] < 0.5 <= fracciones[i + 1]:
            cruza_en = (AMPLITUDES[i], AMPLITUDES[i + 1])
            break
    n_puntos_transicion = sum(1 for f in fracciones if 0.0 < f < 1.0)
    if n_puntos_transicion <= 1:
        escalon_o_rampa = "escalon"
    else:
        escalon_o_rampa = "rampa"

    # --- 5.2 la banda ---
    idx_ultimo_23 = max((i for i, f in enumerate(fracciones) if f == 1.0), default=None)
    hay_techo = (idx_ultimo_23 is not None and idx_ultimo_23 < len(fracciones) - 1
                 and any(f < 1.0 for f in fracciones[idx_ultimo_23 + 1:]))

    # --- 5.3 la semilla ---
    stds = [f["std_n_despiertos"] for f in tabla]
    dispersion_alta_en_todas_partes = all(s > 1.0 for s in stds)  # umbral declarado abajo

    analisis = dict(
        borde=dict(
            cruza_0_5_entre=cruza_en,
            tipo=escalon_o_rampa,
            n_puntos_en_transicion=n_puntos_transicion,
        ),
        banda=dict(
            idx_ultimo_amp_con_23_23=idx_ultimo_23,
            amp_ultimo_23_23=AMPLITUDES[idx_ultimo_23] if idx_ultimo_23 is not None else None,
            hay_techo=hay_techo,
        ),
        semilla=dict(
            std_por_amplitud=stds,
            std_maximo=max(stds),
            std_promedio=float(np.mean(stds)),
            dispersion_alta_en_todas_partes=dispersion_alta_en_todas_partes,
            nota="umbral std>1.0 (sobre una escala 0-23) declarado aqui como 'dispersion "
                 "alta', no anclado en ningun archivo del proyecto -- es una lectura, no "
                 "un test estadistico formal.",
        ),
    )

    salida = dict(
        N=N, dt=DT, T_total=T_TOTAL, k_enfriamiento=K_ENFRIAMIENTO,
        amplitudes=AMPLITUDES, semillas=SEMILLAS,
        n_configuraciones=len(trabajos),
        resultados=resultados,
        tabla_20_filas=tabla,
        analisis=analisis,
        costo=dict(
            elapsed_total_s=elapsed_total,
            n_workers=n_workers,
            costo_medio_por_config_s=costo_medio_s,
            costo_serie_estimado_s=costo_medio_s * len(trabajos),
        ),
    )

    out = HERE / "cs075_resultado_barrido_v3.json"
    out.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\n[tabla] amp | media | min | max | std | #semillas=23/23 | nivel_max")
    for f in tabla:
        print(f"  {f['amp_asimetria']:.4f} | {f['media_n_despiertos']:.2f} | "
              f"{f['min_n_despiertos']:2d} | {f['max_n_despiertos']:2d} | "
              f"{f['std_n_despiertos']:.3f} | {f['semillas_con_23_23']}/8 | "
              f"{f['nivel_maximo_alcanzado']}")

    print(f"\n[5.1 borde] cruza 0,5 entre {analisis['borde']['cruza_0_5_entre']} -- "
          f"tipo: {analisis['borde']['tipo']} "
          f"({analisis['borde']['n_puntos_en_transicion']} puntos en transicion)")
    print(f"[5.2 banda] hay techo: {analisis['banda']['hay_techo']} -- "
          f"ultimo amp con 8/8 semillas en 23/23: {analisis['banda']['amp_ultimo_23_23']}")
    print(f"[5.3 semilla] std maximo: {analisis['semilla']['std_maximo']:.3f}  "
          f"std promedio: {analisis['semilla']['std_promedio']:.3f}  "
          f"dispersion alta en TODAS partes: {analisis['semilla']['dispersion_alta_en_todas_partes']}")

    print(f"\n[archivo] {out}")
    print(f"[costo] {elapsed_total:.1f}s total ({n_workers} workers), "
          f"{costo_medio_s:.3f}s/config promedio, "
          f"~{costo_medio_s * len(trabajos):.1f}s si fuera en serie")


if __name__ == "__main__":
    main()
