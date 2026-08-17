#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_smoke_23_sobre_fisica.py — Paso 3 de INSTRUCCION_CS075_v2_EJECUCION_PARA_CC.md,
con los umbrales corregidos.
=========================================================================================

Historial: la versión anterior de este script reportaba que cruzar T_bajo_confinamiento
tomaba ~21 millones de pasos (~20h/config), porque los umbrales v1 se derivaban de la
razón física 159 GeV/155 MeV. La instrucción v2 corrigió eso: el proyecto ya tenía
T_CONF=0,6 / T_EW=0,9 normalizados en `cs072_motor_23.py` l.42-43. Con los umbrales
corregidos, medido en `Proceso23SobreFisica.__init__`: confinamiento cruza en el
**paso 36** (T_bajo_electrodebil en el paso 5). Ver RESULTADO_..._PARA_CS.md ADENDA 2.

Este script corre las 4 configuraciones de la instrucción §5.1/v2§3 a T_TOTAL=5,0
(5.000 pasos, ~140x margen sobre el cruce de confinamiento) -- de sobra para que los
agentes de Nivel 2 (que requieren T_bajo_confinamiento) tengan oportunidad real de
actuar, y para que la persistencia de sobredensidad (MIN_PERSISTENCIA, FACTOR_ATOMOS)
tenga tiempo de acumularse hacia hay_nucleos/hay_atomos/hay_red si corresponde.
Reporta qué niveles se alcanzan, cuáles agentes quedan dormidos y por qué hito, y el
costo medido (ms/paso, pasos hasta T_CONF, horas equivalentes si se escalara), como
pide v2§3 "Qué entregar".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs075_23_sobre_fisica import construir_23, RATIO_EW_CONF  # noqa: E402

CONFIGS = [0.01, 0.1, 0.5, 2.0]  # instrucción §5.1 / v2§3
T_TOTAL = 5.0
N = 16
DT = 1e-3


def main():
    t0 = time.time()
    resultado = {"configs": {}, "T_total_usado": T_TOTAL, "N": N, "dt": DT}

    for amp in CONFIGS:
        print(f"=== amp_asimetria={amp} ===", file=sys.stderr, flush=True)
        t_config0 = time.time()
        proceso, agentes = construir_23(N=N, dt=DT, seed=12345, amp_asimetria=amp,
                                         k_enfriamiento=50.0)
        paso_conf = None
        n_pasos_objetivo = int(T_TOTAL / DT)
        for k in range(n_pasos_objetivo):
            proceso.paso(agentes)
            if paso_conf is None and proceso._hitos()["T_bajo_confinamiento"]:
                paso_conf = k + 1
        elapsed_config = time.time() - t_config0
        hitos_final = proceso._hitos()

        informes = [ag.informe() for ag in agentes]
        n_dormidos = sum(1 for i in informes if i["paso_despertar"] is None)
        despiertos = sorted([i for i in informes if i["paso_despertar"] is not None],
                            key=lambda i: i["paso_despertar"])

        ms_por_paso = elapsed_config / n_pasos_objetivo * 1000
        print(f"  T_final={proceso.estado.temperatura_media():.4f}  "
              f"hitos={hitos_final}", file=sys.stderr, flush=True)
        print(f"  paso_confinamiento={paso_conf}  ms/paso={ms_por_paso:.4f}  "
              f"tiempo_config={elapsed_config:.1f}s", file=sys.stderr, flush=True)
        print(f"  despiertos: {len(despiertos)}/23  dormidos: {n_dormidos}/23",
              file=sys.stderr, flush=True)
        for i in despiertos:
            print(f"    paso {i['paso_despertar']:>6d}  {i['nombre']}", file=sys.stderr)
        print(f"  DORMIDOS (hito que falta):", file=sys.stderr)
        for i in informes:
            if i["paso_despertar"] is None:
                print(f"    {i['nombre']:20s} requiere {i['requiere']}", file=sys.stderr)

        resultado["configs"][str(amp)] = dict(
            amp_asimetria=amp,
            hitos_finales=hitos_final,
            estado_final=proceso.estado.estado(),
            informes_agentes=informes,
            n_despiertos=len(despiertos),
            n_dormidos=n_dormidos,
            orden_despertar=[dict(nombre=i["nombre"], paso=i["paso_despertar"]) for i in despiertos],
            cronologia=proceso.cronologia,
            paso_confinamiento=paso_conf,
            ms_por_paso=ms_por_paso,
            tiempo_config_s=elapsed_config,
        )

    resultado["elapsed_s"] = time.time() - t0
    resultado["RATIO_EW_CONF"] = RATIO_EW_CONF
    out = HERE / "cs075_resultado_23_sobre_fisica_v2.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}", file=sys.stderr)
    print(f"[elapsed] {resultado['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
