#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_experimento_completo.py -- el experimento de punta a punta, como lo pidio el
director (30-jul-2026): "1) un agente controla la rugosidad inicial, el resto trabaja
con lo que eso produce. 2) comparamos resultados."

Punto de entrada UNICO: corre_holistico_desde_semilla(eps) de cs072_proceso_holistico.py.
Nada se reimplementa aca -- este script solo BARRE los eps ya validados por CF-1 y
compara. Tres verificaciones antes de confiar en la tabla:
  V1: la integracion (M1_Semilla+A23_Campo -> catalogo) da lo MISMO que el puente
      standalone (cs072_asimetria_desde_CF.py) -- no debe haber divergido al conectarlo.
  V2: para cada eps, el resultado de bariones/hidrogeno coincide con el motor
      SECUENCIAL probado (cs072_motor_23.corre), no solo con el holistico -- la cadena
      completa sigue siendo fiel al motor original, no solo la parte de asimetria.
  V3: E1-E4 (inventario/nadie-madruga/cero-exacto/cero-turnos), YA verificadas en
      cs072_pruebas_proceso_holistico.py, no se re-verifican aca (no cambiaron: esta
      capa nueva vive ANTES del catalogo, el bucle de 23 piezas es el mismo objeto).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import corre, cuenta  # noqa: E402
from cs072_asimetria_desde_CF import generar_asimetria, EPS_LIST_CF1  # noqa: E402
from cs072_proceso_holistico import corre_holistico_desde_semilla  # noqa: E402


def main():
    print("=== V1+V2: integracion vs puente standalone, y vs motor secuencial ===\n")
    print(f"{'eps':>8s} {'nq':>4s} {'naq':>4s} {'ne':>4s} {'npos':>5s}  "
          f"{'hol_bar':>8s} {'hol_H':>6s}  {'seq_bar':>8s} {'seq_H':>6s}  V1  V2")

    filas = []
    v1_ok_total = True
    v2_ok_total = True
    for eps in EPS_LIST_CF1:
        asim_standalone = generar_asimetria(eps)
        estado_hol = corre_holistico_desde_semilla(eps)
        c_hol = cuenta(estado_hol)
        d = estado_hol["diagnostico_semilla"]

        v1_ok = (d["nq"] == asim_standalone["nq"] and d["naq"] == asim_standalone["naq"]
                 and d["ne"] == asim_standalone["ne"] and d["npos"] == asim_standalone["npos"])
        v1_ok_total = v1_ok_total and v1_ok

        estado_seq = corre(d["nq"], d["naq"], d["ne"], d["npos"], homogeneo=False,
                            expansion=True, pasos=300)
        c_seq = cuenta(estado_seq)
        v2_ok = (c_hol["bariones"] == c_seq["bariones"] and c_hol["hidrogeno"] == c_seq["hidrogeno"]
                 and c_hol["quarks_sueltos"] == c_seq["quarks_sueltos"])
        v2_ok_total = v2_ok_total and v2_ok

        print(f"{eps:>8g} {d['nq']:>4d} {d['naq']:>4d} {d['ne']:>4d} {d['npos']:>5d}  "
              f"{c_hol['bariones']:>8d} {c_hol['hidrogeno']:>6d}  "
              f"{c_seq['bariones']:>8d} {c_seq['hidrogeno']:>6d}  "
              f"{'SI' if v1_ok else 'NO'}  {'SI' if v2_ok else 'NO'}")

        filas.append(dict(eps=eps, nq=d["nq"], naq=d["naq"], ne=d["ne"], npos=d["npos"],
                          holistico=c_hol, secuencial=c_seq, v1_ok=bool(v1_ok), v2_ok=bool(v2_ok)))

    print(f"\nV1 (integracion == puente standalone) en las {len(EPS_LIST_CF1)} eps: "
          f"{'PASA' if v1_ok_total else 'FALLA'}")
    print(f"V2 (holistico == motor secuencial) en las {len(EPS_LIST_CF1)} eps: "
          f"{'PASA' if v2_ok_total else 'FALLA'}")

    out = HERE / "cs072_resultado_experimento_completo.json"
    out.write_text(json.dumps(dict(filas=filas, v1_ok_total=bool(v1_ok_total),
                                    v2_ok_total=bool(v2_ok_total)),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
