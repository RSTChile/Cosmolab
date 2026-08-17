#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_espacio_expandiendose.py -- corrige el error de cs072_posicion_en_espacio.py /
cs072_fuerte_em_con_distancia.py / cs072_bateria_hipotesis_espacio.py: esos archivos
ponian a las particulas en posiciones FIJAS, dispersas desde el principio, en una malla
ya extendida -- lo opuesto al modelo estandar (director, 30-jul-2026): "el Big Bang no
arrojo materia hacia afuera de un sitio, sino que estiro el espacio mismo... ocurrio en
todas partes a la vez".

MODELO CORREGIDO:
  - POSICION COMOVIL: fija, determinista (misma asignacion por indice-scrambled que ya
    se uso, cs072_posicion_en_espacio.asignar_posiciones con idx*97) -- es una ETIQUETA,
    no una distancia real. Dos particulas "comovilmente separadas" pueden estar juntas
    o lejos en el espacio FISICO segun cuanto se haya expandido el universo.
  - DISTANCIA FISICA(paso) = distancia_comovil * a(paso). a(paso) usa LA MISMA ley
    exponencial ya sellada de CF-2 (a=exp(H_EXP*t), H_EXP=3.0, DT=0.25 -- cs072_
    asimetria_desde_CF.py, sin constantes nuevas), evaluada paso a paso dentro del
    bucle de 300 pasos de cs072_proceso_holistico -- no fija de una vez.
  - A t=0 (paso 0), a=1 -- no es literalmente CERO (CF-2 no modela el instante exacto
    de la singularidad, arranca en a=1 por diseño de su propio protocolo sellado), pero
    crece rapido: los primeros pasos tienen distancia fisica chica (particulas
    efectivamente juntas), y se separan a medida que avanza la simulacion -- la
    direccion correcta, aunque no un t=0 literal de singularidad.

NO TOCA ningun archivo existente. 3_fuerte con corte duro (RADIO=3, el que SI mostro
efecto real en la bateria de hipotesis) ahora usa esta distancia que crece con el paso,
en vez de una distancia fija.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_STRONG, T_CONF  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import Agente23, Aporte, construir_catalogo_desde_semilla  # noqa: E402
from cs072_posicion_en_espacio import asignar_posiciones, _trios_con_indices  # noqa: E402
from cs072_asimetria_desde_CF import H_EXP, DT_CF2  # noqa: E402 -- CF-2 sellado, reusado


def posicion_comovil_scramble(N_particulas, N_grid=16):
    capacidad = N_grid ** 3
    idx = (np.arange(N_particulas) * 97) % capacidad
    pos = np.array(np.unravel_index(idx, (N_grid, N_grid, N_grid))).T
    diffs = pos[:, None, :] - pos[None, :, :]
    D_comovil = np.linalg.norm(diffs, axis=-1)
    return pos, D_comovil


class A3_Fuerte_expansion(Agente23):
    """#3 fuerte, corte duro (RADIO=3, verbatim del hallazgo real en la bateria de
    hipotesis), pero sobre distancia FISICA que crece paso a paso: D_fisica(paso) =
    D_comovil * a(paso). CORREGIDO (bug propio, encontrado antes de correr nada):
    a(paso)=exp(H_EXP*paso*DT_CF2) explotaba sin sentido (a~10^97 a los 300 pasos) --
    confundi el DT=0.25 de CF-2 (que ahi se usa para una sub-difusion interna, nada que
    ver con el tiempo de expansion) con su verdadero reloj. CF-2 (cf2_estiramiento_
    densidad.py l.109) usa tg = paso/(PASOS_totales-1), NORMALIZADO de 0 a 1 sobre TODO
    el barrido -- asi a_final=exp(H_EXP) da exactamente 20.1 para H_EXP=3.0, el mismo
    numero YA sellado en RESUMEN_CF2_crudo.md. Se usa esa misma convencion aca, sobre
    los 300 pasos propios de este bucle (no los 400 de CF2 -- normalizado, no importa
    el total exacto)."""
    numero, nombre, fase = 3, "3_fuerte_expansion", "enlace"
    RADIO = 3
    PASOS_TOTALES = 300

    def __init__(self, D_comovil):
        super().__init__()
        self.D_comovil = D_comovil
        self.paso_actual = 0

    def a_de(self, paso):
        tg = paso / max(self.PASOS_TOTALES - 1, 1)
        return float(np.exp(H_EXP * tg))

    def aporte(self, e, apagar):
        a = self.a_de(self.paso_actual)
        self.paso_actual += 1
        if "3_fuerte" not in apagar and e.T_ef < T_CONF:
            D_fisica = self.D_comovil * a
            mask_cerca = D_fisica <= self.RADIO
            dB = R_STRONG * (e.cd & e.me & mask_cerca).astype(float)
            return Aporte(dB=dB)
        return Aporte()


def correr_variante(nq, naq, ne, npos, piezas_reemplazadas):
    nueva_lista = [piezas_reemplazadas.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva_lista if p.fase == "enlace"]
    estado = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]
    return estado


def main():
    print(f"a(paso) con H_EXP={H_EXP}, DT_CF2={DT_CF2} (CF-2 sellado):")
    for paso in [0, 10, 20, 30, 50, 100, 150, 200, 250, 299]:
        print(f"  paso={paso:>4d}  a={np.exp(H_EXP*paso*DT_CF2):.4e}")

    print(f"\n{'eps':>6s} {'N':>5s} {'bar_base':>9s} {'bar_expansion':>14s} {'H_base':>7s} {'H_expansion':>12s} cambio")
    filas = []
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
        N = nq + naq + ne + npos
        pos, D_comovil = posicion_comovil_scramble(N)

        estado_base = correr_variante(nq, naq, ne, npos, {})
        c_base = cuenta(estado_base)

        estado_exp = correr_variante(nq, naq, ne, npos, {"3_fuerte": A3_Fuerte_expansion(D_comovil)})
        c_exp = cuenta(estado_exp)

        cambio = c_base["bariones"] != c_exp["bariones"]
        print(f"{eps:>6g} {N:>5d} {c_base['bariones']:>9d} {c_exp['bariones']:>14d} "
              f"{c_base['hidrogeno']:>7d} {c_exp['hidrogeno']:>12d} {cambio}", flush=True)
        filas.append(dict(eps=eps, N=N, bariones_base=c_base["bariones"],
                          bariones_expansion=c_exp["bariones"], hidrogeno_base=c_base["hidrogeno"],
                          hidrogeno_expansion=c_exp["hidrogeno"], cambio=bool(cambio)))

    out = HERE / "cs072_resultado_espacio_expandiendose.json"
    out.write_text(json.dumps(dict(H_EXP=H_EXP, DT_CF2=DT_CF2, filas=filas),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
