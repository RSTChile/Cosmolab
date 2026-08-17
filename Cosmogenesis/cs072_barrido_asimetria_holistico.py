#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_barrido_asimetria_holistico.py -- cuanto importa la asimetria materia/antimateria,
medido con conteo REAL de bariones e hidrogeno (no un proxy), usando el proceso
holistico ya verificado (cs072_proceso_holistico.py) contra el motor probado.

Parametro que se barre: el EXCESO de materia sobre antimateria. naq (antiquarks) y npos
(positrones) quedan fijos en los valores ya probados (21 y 7). nq = 21+exceso,
ne = 7+round(exceso/3) -- misma proporcion 3:1 que ya tenia la configuracion probada
(30,21,10,7): exceso=9 reproduce EXACTO esa configuracion (nq=30, ne=10), y es el punto
de anclaje/control de este barrido -- tiene que dar 3 bariones, 2 hidrogeno, o algo esta
mal.

exceso=0 es materia=antimateria exacto: la aniquilacion (#8, ya verificada) empareja y
mata todo lo que puede emparejar -- si no queda nada, es la version medida (no supuesta)
de por que la asimetria importa, la misma pregunta S>0 de la que parte todo el proyecto.

homogeneo=False, expansion=True, pasos=300 -- mismos parametros que el caso ya verificado
en las 4 pruebas (P1-P4) de cs072_pruebas_proceso_holistico.py. No se inventa nada nuevo
aca; cero_azar (cs072_motor_23 no usa RNG, no hace falta mas de una corrida por punto).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import corre, cuenta  # noqa: E402
from cs072_proceso_holistico import corre_holistico  # noqa: E402

NAQ_FIJO = 21
NPOS_FIJO = 7
EXCESOS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
PUNTOS_CONTROL_SEQ = {0, 9, 27}  # verificados tambien contra el motor secuencial


def main():
    t0 = time.time()
    filas = []
    for exceso in EXCESOS:
        nq = NAQ_FIJO + exceso
        ne = NPOS_FIJO + round(exceso / 3)
        estado_h = corre_holistico(nq, NAQ_FIJO, ne, NPOS_FIJO, homogeneo=False,
                                    expansion=True, pasos=300)
        c_h = cuenta(estado_h)

        fila = dict(exceso=exceso, nq=nq, naq=NAQ_FIJO, ne=ne, npos=NPOS_FIJO,
                    n_total=nq + NAQ_FIJO + ne + NPOS_FIJO, holistico=c_h)

        if exceso in PUNTOS_CONTROL_SEQ:
            estado_s = corre(nq, NAQ_FIJO, ne, NPOS_FIJO, homogeneo=False,
                              expansion=True, pasos=300)
            c_s = cuenta(estado_s)
            fila["secuencial"] = c_s
            fila["coincide_con_secuencial"] = bool(
                c_h["bariones"] == c_s["bariones"] and c_h["hidrogeno"] == c_s["hidrogeno"]
                and c_h["quarks_sueltos"] == c_s["quarks_sueltos"])

        filas.append(fila)

    elapsed = time.time() - t0

    print(f"{'exceso':>7s} {'nq':>4s} {'naq':>4s} {'ne':>4s} {'npos':>5s} "
          f"{'bariones':>9s} {'protones':>9s} {'hidrogeno':>10s} {'sueltos':>8s}  control")
    for f in filas:
        c = f["holistico"]
        ctrl = ""
        if "coincide_con_secuencial" in f:
            ctrl = "OK (=seq)" if f["coincide_con_secuencial"] else "*** DISTINTO DE SEQ ***"
        print(f"{f['exceso']:>7d} {f['nq']:>4d} {f['naq']:>4d} {f['ne']:>4d} {f['npos']:>5d} "
              f"{c['bariones']:>9d} {c['protones']:>9d} {c['hidrogeno']:>10d} "
              f"{c['quarks_sueltos']:>8d}  {ctrl}")

    control_ok = all(f.get("coincide_con_secuencial", True) for f in filas)
    print(f"\nControles contra el motor secuencial: {'TODOS OK' if control_ok else '*** FALLARON ***'}")
    print(f"tiempo: {elapsed:.2f}s")

    out = HERE / "cs072_resultado_barrido_asimetria_holistico.json"
    out.write_text(json.dumps(dict(filas=filas, elapsed_s=elapsed, control_ok=control_ok),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[archivo] {out}")


if __name__ == "__main__":
    main()
