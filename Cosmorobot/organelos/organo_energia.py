#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_energia — el robot SIENTE su propia batería.

PEDIDO DE ALEXIS (2026-07-10): un propioceptor de energía — no para que el
robot se detenga por batería baja (eso no es su trabajo; si algún día hace
falta un corte de seguridad, va aparte), sino para que el organismo SEPA
cuánta energía tiene y esa sensación entre en su bienestar, igual que
cualquier otra sensación interna. Qué hace con ese saber (explorar más o
menos) es cosa de la deliberación/propiocepción, no de este organelo.

PROCEDENCIA: `organo_propiocepcion.py` ya esperaba un campo `energia`
(0-1) desde el principio — venía en 0 por honestidad porque ningún
organelo lo calculaba todavía (ver su docstring: "el resto (energía, LF,
H...) quedan en 0 hasta que existan organelos que los calculen"). Este
organelo es ESO: lee el voltaje real de la batería del NXT
(`get_battery_level()`, milivolts) y lo normaliza a 0-1.

CALIBRACIÓN (honestidad, LF): los límites MIN/MAX son a priori del
diseñador basados en lo observado en esta sesión (2026-07-10) — batería
"mala" que causó fallos de sensores ~6979mV, batería fresca que anduvo
bien ~8211mV. Sin fase de calibración propia todavía (mismo estado que
UMBRAL_CRITICO_CM y las escalas de CambioTotal).
"""
from __future__ import annotations

import time


class OrganoEnergia:
    BATERIA_MIN_MV = 6000.0   # a priori: por debajo de esto, energía se satura en 0
    BATERIA_MAX_MV = 8300.0   # a priori: batería fresca observada en campo, energía satura en 1

    def __init__(self, conexion) -> None:
        self.conexion = conexion
        self._ultimo_ts = 0.0
        self._activo = False

    def arrancar(self) -> bool:
        try:
            self.conexion.brick.get_battery_level()
            self._activo = True
            return True
        except Exception as e:
            print(f"[OrganoEnergia] no se pudo leer la batería: {e}")
            return False

    @property
    def vivo(self) -> bool:
        return self._activo and (time.time() - self._ultimo_ts) < 5.0

    def leer(self) -> float | None:
        if not self._activo:
            return None
        try:
            mv = self.conexion.brick.get_battery_level()
            self._ultimo_ts = time.time()
            return float(mv)
        except Exception as e:
            print(f"[OrganoEnergia] error de lectura: {e}")
            return None

    def inyectar(self, fila: dict) -> None:
        mv = self.leer()
        if mv is None:
            fila.setdefault("bateria_mv", None)
            fila.setdefault("energia", 0.0)
            fila["energia_vivo"] = 0
            return
        rango = self.BATERIA_MAX_MV - self.BATERIA_MIN_MV
        energia = 0.0 if rango <= 0 else (mv - self.BATERIA_MIN_MV) / rango
        energia = 0.0 if energia < 0.0 else 1.0 if energia > 1.0 else energia
        fila["bateria_mv"] = mv
        fila["energia"] = round(energia, 4)
        fila["energia_vivo"] = 1

    def cerrar(self) -> None:
        self._activo = False
