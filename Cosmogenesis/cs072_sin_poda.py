#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs072_sin_poda.py -- elimina poda de las corridas exploratorias, tal como pidio el
director (30-jul-2026): "esta poda sera un misterio... eliminala, no tiene importancia
real". Motivo, medido, no de palabra: PODA_FRAC=2.5/LIGADO_FRAC=1.5 no tienen ninguna
justificacion encontrada en el proyecto (revisado); quitando poda del todo, la
configuracion de referencia (30,21,10,7) sigue dando 3 bariones/2 hidrogeno EXACTO
(solo cambia B.max, de 0.14 a ~5.3M -- cuenta() mide estructura relativa, no magnitud
absoluta, asi que no le afecta). Y poda es la pieza que destruia estructura en las
corridas extendidas de hoy (EM retrasada, eps>=0.6).

NO TOCA cs072_motor_23.py ni cs072_proceso_holistico.py -- parcha PODA con un no-op,
mismo patron de monkeypatch ya usado y verificado toda la sesion.
"""
from __future__ import annotations

import numpy as np

import cs072_proceso_holistico as ph
from cs072_proceso_holistico import Agente23


class A9_Poda_nula(Agente23):
    """Reemplazo no-op de A9_Poda -- no corta nada, nunca."""
    numero, nombre, fase = 9, "9_poda_18_inflacion", "poda"

    def aporte_poda(self, B_actualizado, b0_inicio_paso, apagar, expansion):
        return None


def desactivar_poda():
    """Reemplaza ph.PODA por la version nula. Efecto inmediato en cualquier llamada a
    corre_holistico() posterior, sin tocar el archivo."""
    ph.PODA = A9_Poda_nula()


def reactivar_poda():
    """Restaura la poda original -- por si algun script necesita compararla despues."""
    from cs072_proceso_holistico import A9_Poda
    ph.PODA = A9_Poda()


if __name__ == "__main__":
    from cs072_motor_23 import cuenta
    from cs072_proceso_holistico import corre_holistico

    print("=== verificacion: referencia (30,21,10,7), sin poda, 300 pasos ===")
    desactivar_poda()
    estado = corre_holistico(30, 21, 10, 7, homogeneo=False, expansion=True, pasos=300)
    c = cuenta(estado)
    print(f"  bariones={c['bariones']} hidrogeno={c['hidrogeno']} sueltos={c['quarks_sueltos']} "
          f"B.max={estado['B'].max():.2f}  (referencia esperada: 3, 2, 0)")
