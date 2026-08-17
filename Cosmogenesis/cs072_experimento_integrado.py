#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_experimento_integrado.py -- todas las piezas de la sesion de hoy, juntas en un
solo experimento. Pedido del director (30-jul-2026): "usemos la malla prestada y
aprendamos que sucede con eso... es un experimento".

LAS PIEZAS, cada una ya verificada por separado, ahora combinadas:
  1. Asimetria real (CF-1+CF-2, no elegida a mano) -- construir_catalogo_desde_semilla(eps)
  2. Posicion comovil (malla 16^3 PRESTADA de EstadoFisico, determinista por indice-
     scrambled) -- declarada como supuesto, no derivada (el muro de los 40 experimentos
     de Topologia sigue sin resolver, no se pretende lo contrario)
  3. 3_fuerte con corte duro (RADIO=3) sobre distancia FISICA que crece con la expansion
     (misma ley exponencial ya sellada de CF-2, con la normalizacion de tiempo correcta
     -- tg=paso/(pasos_totales-1))
  4. 4_em retrasada por temperatura (T_umbral barrible, hasta el piso real ~0.0158)
  5. 12_localidad+M2_memoria ACOTADA (tanh+escala, el arreglo que evita la explosion
     numerica que EM-siempre-activa tapaba sin que nadie lo supiera)

NO TOCA cs072_motor_23.py, cs072_proceso_holistico.py, cs075_base_fisica.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_STRONG, R_EM, T_CONF  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import Agente23, Aporte, construir_catalogo_desde_semilla  # noqa: E402
from cs072_asimetria_desde_CF import H_EXP  # noqa: E402 -- CF-2 sellado (H_EXP=3.0)


def posicion_comovil(N_particulas, N_grid=16):
    capacidad = N_grid ** 3
    idx = (np.arange(N_particulas) * 97) % capacidad
    pos = np.array(np.unravel_index(idx, (N_grid, N_grid, N_grid))).T
    diffs = pos[:, None, :] - pos[None, :, :]
    D_comovil = np.linalg.norm(diffs, axis=-1)
    return pos, D_comovil


class A3_Fuerte_integrado(Agente23):
    """#3 fuerte, corte duro sobre distancia fisica creciente (comovil * a(paso)),
    a(paso)=exp(H_EXP*tg). CORREGIDO (bug propio, encontrado por el director): tg NO
    se normaliza contra pasos_totales de ESTA corrida (eso acopla sin querer "cuanto
    dura la simulacion" con "que tan rapido se expande el universo" -- alargar la
    corrida para darle tiempo a EM diluia sin querer la expansion de fuerte). Se
    normaliza contra una referencia FIJA (PASOS_REF=300, la escala original en la que
    se verifico y sello a_final=exp(H_EXP) -- misma referencia siempre, corra lo que
    corra la simulacion real)."""
    numero, nombre, fase = 3, "3_fuerte_integrado", "enlace"
    RADIO = 3
    PASOS_REF = 300  # referencia FIJA del reloj de expansion, no el largo de esta corrida

    def __init__(self, D_comovil, pasos_totales):
        super().__init__()
        self.D_comovil = D_comovil
        self.paso_actual = 0

    def aporte(self, e, apagar):
        tg = self.paso_actual / max(self.PASOS_REF - 1, 1)
        a = float(np.exp(H_EXP * tg))
        self.paso_actual += 1
        if "3_fuerte" not in apagar and e.T_ef < T_CONF:
            D_fisica = self.D_comovil * a
            mask_cerca = D_fisica <= self.RADIO
            dB = R_STRONG * (e.cd & e.me & mask_cerca).astype(float)
            return Aporte(dB=dB)
        return Aporte()


class A4_EM_integrado(Agente23):
    """#4 EM, retrasada por temperatura (T_umbral barrible)."""
    numero, nombre, fase = 4, "4_em_integrado", "enlace"

    def __init__(self, T_umbral):
        super().__init__()
        self.T_umbral = T_umbral

    def aporte(self, e, apagar):
        if "4_em" not in apagar and e.T_ef < self.T_umbral:
            return Aporte(dB=R_EM * e.co.astype(float))
        return Aporte()


class A12_LocalidadM2Memoria_acotada(Agente23):
    """#12+M2 acotada (tanh+escala) -- evita la explosion numerica, verificado antes."""
    numero, nombre, fase = 12, "12_localidad_M2_memoria_acotada", "enlace"

    def aporte(self, e, apagar):
        if "M2_memoria" not in apagar:
            b0 = max(float(e.B.mean()), 1e-12)
            persist = e.B > b0
            if "12_localidad" not in apagar:
                dB = 0.05 * persist * np.tanh(e.B / b0) * b0
                return Aporte(dB=dB)
        return Aporte()


def correr_integrado(eps, T_umbral_em=0.9, radio_fuerte=3, pasos=2000):
    """T_umbral_em=0.9 (>T_EW, siempre activa) reproduce el comportamiento original;
    bajarlo activa el retraso real."""
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
    N = nq + naq + ne + npos
    pos, D_comovil = posicion_comovil(N)

    ag_fuerte = A3_Fuerte_integrado(D_comovil, pasos)
    ag_fuerte.RADIO = radio_fuerte
    reemplazos = {
        "3_fuerte": ag_fuerte,
        "4_em": A4_EM_integrado(T_umbral_em),
        "12_localidad_M2_memoria": A12_LocalidadM2Memoria_acotada(),
    }
    nueva = [reemplazos.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva if p.fase == "enlace"]
    estado = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=pasos)
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
    c = cuenta(estado)
    return dict(eps=eps, N=N, T_umbral_em=T_umbral_em, radio_fuerte=radio_fuerte, pasos=pasos,
                bariones=c["bariones"], hidrogeno=c["hidrogeno"], sueltos=c["quarks_sueltos"],
                B_max=float(estado["B"].max()))


def main():
    t0 = time.time()

    print("=== control: sin restricciones espaciales/temporales (EM siempre activa), verifica baseline ===")
    r0 = correr_integrado(0.5, T_umbral_em=0.9, radio_fuerte=999, pasos=300)
    print(f"  {r0}")

    print("\n=== barrido: eps x radio_fuerte, EM retrasada al piso real (0.0158), 2000 pasos ===")
    filas = []
    print(f"{'eps':>6s} {'radio':>6s} {'bariones':>9s} {'hidrogeno':>10s} {'B_max':>9s}")
    for eps in [0.2, 0.5, 0.8, 1.0]:
        for radio in [2, 3, 5, 8]:
            r = correr_integrado(eps, T_umbral_em=0.0158, radio_fuerte=radio, pasos=2000)
            print(f"{eps:>6g} {radio:>6d} {r['bariones']:>9d} {r['hidrogeno']:>10d} {r['B_max']:>9.4f}", flush=True)
            filas.append(r)

    out = HERE / "cs072_resultado_experimento_integrado.json"
    out.write_text(json.dumps(dict(control=r0, barrido=filas, elapsed_s=time.time() - t0),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")
    print(f"[tiempo] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
