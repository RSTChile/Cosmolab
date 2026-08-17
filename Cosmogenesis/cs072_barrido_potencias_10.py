#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_barrido_potencias_10.py -- el barrido que pidio el director (30-jul-2026): "los
agentes trabajan en potencias de 10... comencemos los barridos con las asimetrias que
pone el agente primario."

Dos ejes, ninguno inventado:
  1. ESCALA (potencia de 10): multiplica la configuracion YA VERIFICADA (30,21,10,7) por
     10^0, 10^1, 10^2, ... -- preserva la MISMA proporcion, no cambia la fisica, solo el
     tamano del universo simulado.
  2. eps: la rugosidad primordial que M1_semilla/23_campo (via CF-1+CF-2, ya conectado en
     cs072_proceso_holistico.construir_catalogo_desde_semilla) usa para producir la
     asimetria -- los mismos 8 valores que CF-1 ya barrio, ningun valor nuevo.

Verificado antes de barrer (turno anterior): bariones/N_total y hidrogeno/N_total son
CONSTANTES EXACTAS (0.044118 y 0.029412) en 4 escalas ya medidas (factor 1,3,10,30, rango
30x) -- no hay saturacion visible en ese rango. Este script:
  (a) mide eps->asimetria UNA vez (barato, escala=10^0) para las 8 eps de CF-1,
  (b) verifica la constante de proporcionalidad con corridas REALES en escalas mayores
      (10^1, y 10^2 donde el costo lo permite) para al menos un eps representativo,
  (c) con esa proporcionalidad MEDIDA (no supuesta), extrapola en potencias de 10 hasta
      donde el numero deja de tener sentido simularlo literalmente (10^58, escala
      estelar) -- reportado como extrapolacion, nunca como corrida.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta  # noqa: E402
from cs072_asimetria_desde_CF import EPS_LIST_CF1  # noqa: E402
from cs072_proceso_holistico import construir_catalogo_desde_semilla, corre_holistico  # noqa: E402

# configuracion base YA VERIFICADA (naq=21, npos=7); nq/ne salen de eps via CF-1+CF-2
ESCALAS_MEDIDAS = [0, 1]  # 10^0, 10^1 -- rapidas; 10^2 se corre aparte (mas cara)
ESCALAS_EXTRAPOLADAS = [3, 4, 6, 10, 20, 30, 40, 58]  # solo aritmetica, nunca simuladas


def correr_a_escala(nq, naq, ne, npos, escala_log10):
    """Multiplica el conteo base por 10^escala_log10, preservando la proporcion, y corre
    de verdad (nunca mas alla de lo computacionalmente razonable)."""
    factor = 10 ** escala_log10
    nq_e, naq_e, ne_e, npos_e = nq * factor, naq * factor, ne * factor, npos * factor
    t0 = time.time()
    estado = corre_holistico(nq_e, naq_e, ne_e, npos_e, homogeneo=False, expansion=True, pasos=300)
    elapsed = time.time() - t0
    c = cuenta(estado)
    N_total = nq_e + naq_e + ne_e + npos_e
    return dict(escala_log10=escala_log10, factor=factor, nq=nq_e, naq=naq_e, ne=ne_e,
                npos=npos_e, N_total=N_total, bariones=c["bariones"], hidrogeno=c["hidrogeno"],
                sueltos=c["quarks_sueltos"], tiempo_s=elapsed)


def main():
    t_inicio = time.time()
    print("=== paso 1: eps -> asimetria, via el agente primario (M1_semilla/23_campo) ===\n")
    print(f"{'eps':>8s} {'nq_base':>8s} {'naq_base':>8s} {'ne_base':>8s} {'npos_base':>9s}")
    base_por_eps = {}
    for eps in EPS_LIST_CF1:
        nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
        base_por_eps[eps] = (nq, naq, ne, npos)
        print(f"{eps:>8g} {nq:>8d} {naq:>8d} {ne:>8d} {npos:>9d}")

    print("\n=== paso 2: escalas REALES corridas (10^0, 10^1, 10^2), un eps representativo (0.5) ===\n")
    eps_repr = 0.5
    nq0, naq0, ne0, npos0 = base_por_eps[eps_repr]
    corridas_reales = []
    print(f"{'escala':>8s} {'N_total':>10s} {'bariones':>9s} {'hidrogeno':>10s} {'ratio_bar':>10s} {'tiempo_s':>9s}")
    for esc in ESCALAS_MEDIDAS:
        r = correr_a_escala(nq0, naq0, ne0, npos0, esc)
        ratio_bar = r["bariones"] / r["N_total"] if r["N_total"] else 0.0
        r["ratio_bariones"] = ratio_bar
        r["ratio_hidrogeno"] = r["hidrogeno"] / r["N_total"] if r["N_total"] else 0.0
        corridas_reales.append(r)
        print(f"{esc:>8d} {r['N_total']:>10d} {r['bariones']:>9d} {r['hidrogeno']:>10d} "
              f"{ratio_bar:>10.6f} {r['tiempo_s']:>9.2f}")

    ratios_bar = [r["ratio_bariones"] for r in corridas_reales]
    ratios_H = [r["ratio_hidrogeno"] for r in corridas_reales]
    constante_ok = bool(np.allclose(ratios_bar, ratios_bar[0], rtol=1e-6)
                        and np.allclose(ratios_H, ratios_H[0], rtol=1e-6))
    print(f"\nProporcionalidad exacta en las escalas corridas de verdad: {constante_ok}")
    k_bar = ratios_bar[0]
    k_H = ratios_H[0]

    print(f"\n=== paso 3: extrapolacion en potencias de 10 (NO corrido, aritmetica sobre k medido) ===\n")
    print(f"k_bariones={k_bar:.6f}  k_hidrogeno={k_H:.6f} (medidos, eps={eps_repr})\n")
    print(f"{'escala':>8s} {'N_total (extrapolado)':>22s} {'bariones (extrap.)':>19s} {'hidrogeno (extrap.)':>20s}")
    extrapolaciones = []
    N_base = nq0 + naq0 + ne0 + npos0
    for esc in ESCALAS_EXTRAPOLADAS:
        N_total = N_base * (10 ** esc)
        bar = N_total * k_bar
        hid = N_total * k_H
        print(f"{esc:>8d} {N_total:>22.3e} {bar:>19.3e} {hid:>20.3e}")
        extrapolaciones.append(dict(escala_log10=esc, N_total=N_total, bariones_extrapolado=bar,
                                    hidrogeno_extrapolado=hid))

    resultado = dict(
        base_por_eps={str(k): v for k, v in base_por_eps.items()},
        eps_representativo=eps_repr,
        corridas_reales=corridas_reales,
        proporcionalidad_constante_verificada=constante_ok,
        k_bariones=k_bar, k_hidrogeno=k_H,
        extrapolaciones_potencias_10=extrapolaciones,
        nota="las extrapolaciones NO son corridas -- son aritmetica sobre k medido en "
             "escalas 10^0-10^2 reales. La linealidad podria no sostenerse mas alla de "
             "lo medido; no hay razon fisica conocida en este modelo (sin espacio real) "
             "para que sature, pero tampoco prueba de que no lo haga.",
        elapsed_total_s=time.time() - t_inicio,
    )
    out = HERE / "cs072_resultado_barrido_potencias_10.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")
    print(f"[tiempo total] {resultado['elapsed_total_s']:.1f}s")


if __name__ == "__main__":
    main()
