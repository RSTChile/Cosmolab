#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_gravedad_con_distancia.py -- hace que 2_gravedad dependa de la distancia REAL
entre particulas (posicion determinista, cs072_posicion_en_espacio.asignar_posiciones),
en vez de ser puramente masa sin espacio. Pedido del director (30-jul-2026), paso 2
tras confirmar que pintar posicion DESPUES de formar el enlace no servia de nada (la
cercania medida era un artefacto del orden, cazado y descartado).

NO TOCA cs072_motor_23.py, cs072_proceso_holistico.py ni cs075_base_fisica.py. La unica
pieza nueva es A2_Gravedad_con_distancia, que reemplaza a 2_gravedad en la lista de
agentes via el mismo parche de modulo ya usado y verificado en esta sesion (Python
resuelve nombres de modulo en tiempo de llamada -- parchear ANTES de llamar
corre_holistico() cambia lo que ve, sin editar el archivo).

FORMULA: identica a corre() l.137 (R_GRAV * outer(masa_ef,masa_ef) / mean(masa_ef)^2 *
0.1), con UN termino nuevo: dividido por distancia^2 (ley de cuadrado inverso, la forma
newtoniana estandar -- no una eleccion arbitraria). Ninguna otra constante se toca ni se
reescala para "que de mejor" -- si el resultado cambia poco o mucho, se reporta tal cual.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_motor_23 import cuenta, R_GRAV  # noqa: E402
import cs072_proceso_holistico as ph  # noqa: E402
from cs072_proceso_holistico import Agente23, Aporte, construir_catalogo_desde_semilla  # noqa: E402
from cs072_posicion_en_espacio import asignar_posiciones, _trios_con_indices  # noqa: E402


class A2_Gravedad_con_distancia(Agente23):
    """#2 gravedad -- MISMA formula que corre() l.135-137, con un termino nuevo:
    dividida por distancia^2 (cuadrado inverso, forma newtoniana estandar). Posicion
    fija, determinista (asignar_posiciones, mismo criterio 'invariante al indice, cero
    azar' que ya usa _catalogo() para color). Ninguna otra constante se toca."""
    numero, nombre, fase = 2, "2_gravedad_con_distancia", "enlace"

    def __init__(self, D):
        super().__init__()
        self.D2 = np.maximum(D ** 2, 1.0)  # piso=1 (celdas vecinas), evita /0 en diagonal

    def aporte(self, e, apagar):
        if "2_gravedad" not in apagar:
            dB = (R_GRAV * np.outer(e.masa_ef, e.masa_ef)
                  / max(float(e.masa_ef.mean()) ** 2, 1e-300) * 0.1 / self.D2)
            return Aporte(dB=dB)
        return Aporte()


def parchar_gravedad_con_distancia(N_particulas, N_grid=16):
    """Reemplaza 2_gravedad en la lista de agentes de cs072_proceso_holistico por la
    version con distancia. Devuelve la matriz de distancia usada (para inspeccion)."""
    pos, hay_wrap = asignar_posiciones(N_particulas, N_grid=N_grid)
    diffs = pos[:, None, :] - pos[None, :, :]
    D = np.linalg.norm(diffs, axis=-1)

    nueva_pieza = A2_Gravedad_con_distancia(D)
    nueva_lista = [nueva_pieza if p.nombre == "2_gravedad" else p for p in ph.PIEZAS_23]
    ph._AGENTES_CON_APORTE = [p for p in nueva_lista if p.fase == "enlace"]
    return D, pos, hay_wrap


def restaurar_gravedad_original():
    """Deshace el parche -- vuelve a la lista original de cs072_proceso_holistico."""
    ph._AGENTES_CON_APORTE = [p for p in ph.PIEZAS_23 if p.fase == "enlace"]


def main():
    eps = 0.5
    nq, naq, ne, npos, diag = construir_catalogo_desde_semilla(eps)

    print("=== SIN distancia (gravedad original, ya verificada) ===")
    restaurar_gravedad_original()
    estado_sin = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)
    c_sin = cuenta(estado_sin)
    bar_sin, _, hid_sin = _trios_con_indices(estado_sin)
    print(f"  bariones={c_sin['bariones']} hidrogeno={c_sin['hidrogeno']} "
          f"sueltos={c_sin['quarks_sueltos']}")
    print(f"  indices bariones: {[list(t) for t in bar_sin]}")

    print("\n=== CON distancia real (2_gravedad reemplazada) ===")
    N_particulas = nq + naq + ne + npos
    D, pos, hay_wrap = parchar_gravedad_con_distancia(N_particulas)
    estado_con = ph.corre_holistico(nq, naq, ne, npos, homogeneo=False, expansion=True, pasos=300)
    c_con = cuenta(estado_con)
    bar_con, _, hid_con = _trios_con_indices(estado_con)
    print(f"  bariones={c_con['bariones']} hidrogeno={c_con['hidrogeno']} "
          f"sueltos={c_con['quarks_sueltos']}")
    print(f"  indices bariones: {[list(t) for t in bar_con]}")
    restaurar_gravedad_original()  # deja el modulo limpio para cualquier uso posterior

    cambio = dict(
        bariones_igual=(c_sin["bariones"] == c_con["bariones"]),
        hidrogeno_igual=(c_sin["hidrogeno"] == c_con["hidrogeno"]),
        mismos_indices_bariones=([sorted(t) for t in bar_sin] == [sorted(t) for t in bar_con]),
    )
    print(f"\n=== comparacion ===")
    print(f"  mismo conteo de bariones: {cambio['bariones_igual']}")
    print(f"  mismo conteo de hidrogeno: {cambio['hidrogeno_igual']}")
    print(f"  MISMAS particulas exactas en cada barion: {cambio['mismos_indices_bariones']}")

    resultado = dict(eps=eps, N_particulas=N_particulas,
                     sin_distancia=dict(conteo=c_sin, bariones_indices=[list(t) for t in bar_sin]),
                     con_distancia=dict(conteo=c_con, bariones_indices=[list(t) for t in bar_con]),
                     cambio=cambio)
    out = HERE / "cs072_resultado_gravedad_con_distancia.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[archivo] {out}")


if __name__ == "__main__":
    main()
