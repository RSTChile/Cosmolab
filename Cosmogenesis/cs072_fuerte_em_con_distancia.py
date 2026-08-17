#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_fuerte_em_con_distancia.py -- extiende cs072_gravedad_con_distancia.py: ahora
3_fuerte y 4_em (las dos fuerzas que de verdad deciden que se liga, ~47x y ~3257x mas
fuertes que gravedad -- medido) tambien dependen de la distancia real entre particulas.
Pedido del director (30-jul-2026): "prueba con ambas... aunque en este caso se trata de
dimension cuantica, no espacial". Es decir: esto NO pretende resolver el espacio
macroscopico (el muro de los 40 experimentos de Topologia sigue en pie, sin tocar) --
es la escala cuantica DENTRO de un hadron/atomo (confinamiento, ligadura EM), un paso
mas modesto y honesto.

MISMA forma que ya se probo en gravedad (cuadrado inverso), por consistencia -- no se
inventa una ley distinta para cada fuerza. Simplificacion declarada: el confinamiento
real (QCD) en realidad CRECE con la distancia (al reves de 1/d^2) a escalas mayores;
aca se usa la MISMA forma que gravedad/EM por consistencia metodologica, no porque sea
fisicamente exacta para el confinamiento -- se reporta como lo que es.

NO TOCA cs072_motor_23.py, cs072_proceso_holistico.py, cs075_base_fisica.py ni
cs072_gravedad_con_distancia.py (los reusa).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_STRONG, R_EM  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import Agente23, Aporte, construir_catalogo_desde_semilla  # noqa: E402
from cs072_posicion_en_espacio import asignar_posiciones, _trios_con_indices  # noqa: E402
from cs072_gravedad_con_distancia import A2_Gravedad_con_distancia  # noqa: E402


class A3_Fuerte_con_distancia(Agente23):
    """#3 fuerte -- MISMA formula que corre() l.130-131, dividida por distancia^2.
    Simplificacion declarada: el confinamiento real NO decae con la distancia como
    gravedad/EM (al reves, crece) -- se usa 1/d^2 por consistencia metodologica con
    las otras dos pruebas de esta sesion, no porque sea la fisica exacta del
    confinamiento. Reportado, no escondido."""
    numero, nombre, fase = 3, "3_fuerte_con_distancia", "enlace"

    def __init__(self, D):
        super().__init__()
        self.D2 = np.maximum(D ** 2, 1.0)

    def aporte(self, e, apagar):
        from cs072_motor_23 import T_CONF
        if "3_fuerte" not in apagar and e.T_ef < T_CONF:
            dB = R_STRONG * (e.cd & e.me).astype(float) / self.D2
            return Aporte(dB=dB)
        return Aporte()


class A4_EM_con_distancia(Agente23):
    """#4 EM -- MISMA formula que corre() l.132-134, dividida por distancia^2. Esta SI
    es la forma fisicamente correcta (Coulomb es cuadrado inverso de verdad)."""
    numero, nombre, fase = 4, "4_em_con_distancia", "enlace"

    def __init__(self, D):
        super().__init__()
        self.D2 = np.maximum(D ** 2, 1.0)

    def aporte(self, e, apagar):
        if "4_em" not in apagar:
            dB = R_EM * e.co.astype(float) / self.D2
            return Aporte(dB=dB)
        return Aporte()


def construir_posicion_y_D(N_particulas, N_grid=16):
    pos, hay_wrap = asignar_posiciones(N_particulas, N_grid=N_grid)
    diffs = pos[:, None, :] - pos[None, :, :]
    D = np.linalg.norm(diffs, axis=-1)
    return D, pos, hay_wrap


def parchar(piezas_con_distancia, D):
    """piezas_con_distancia: subconjunto de {'2_gravedad','3_fuerte','4_em'} a
    reemplazar por su version con distancia. El resto queda igual que siempre."""
    reemplazos = {}
    if "2_gravedad" in piezas_con_distancia:
        reemplazos["2_gravedad"] = A2_Gravedad_con_distancia(D)
    if "3_fuerte" in piezas_con_distancia:
        reemplazos["3_fuerte"] = A3_Fuerte_con_distancia(D)
    if "4_em" in piezas_con_distancia:
        reemplazos["4_em"] = A4_EM_con_distancia(D)

    nueva_lista = [reemplazos.get(p.nombre, p) for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva_lista if p.fase == "enlace"]


def restaurar():
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]


def correr_variante(nq, naq, ne, npos, D, piezas_con_distancia):
    if piezas_con_distancia:
        parchar(piezas_con_distancia, D)
    else:
        restaurar()
    estado = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)
    restaurar()
    c = cuenta(estado)
    bar, _, hid = _trios_con_indices(estado)
    return c, [sorted(t) for t in bar]


def main():
    eps = 0.5
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)
    N_particulas = nq + naq + ne + npos
    D, pos, hay_wrap = construir_posicion_y_D(N_particulas)

    variantes = [
        ("ninguna (baseline ya verificado)", set()),
        ("solo gravedad", {"2_gravedad"}),
        ("solo fuerte", {"3_fuerte"}),
        ("solo EM", {"4_em"}),
        ("fuerte + EM", {"3_fuerte", "4_em"}),
        ("las tres", {"2_gravedad", "3_fuerte", "4_em"}),
    ]

    print(f"{'variante':>35s} {'bariones':>9s} {'hidrogeno':>10s} {'sueltos':>8s}  mismos_indices_que_baseline")
    filas = []
    bar_baseline = None
    for nombre, piezas in variantes:
        c, bar = correr_variante(nq, naq, ne, npos, D, piezas)
        if bar_baseline is None:
            bar_baseline = bar
        mismos = (bar == bar_baseline)
        print(f"{nombre:>35s} {c['bariones']:>9d} {c['hidrogeno']:>10d} {c['quarks_sueltos']:>8d}  {mismos}")
        filas.append(dict(variante=nombre, piezas_con_distancia=sorted(piezas), conteo=c,
                          bariones_indices=bar, mismos_indices_que_baseline=bool(mismos)))

    out = HERE / "cs072_resultado_fuerte_em_con_distancia.json"
    out.write_text(json.dumps(dict(eps=eps, N_particulas=N_particulas, filas=filas),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
