#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_eopd — sensor EOPD HiTechnic (puerto 2). ESTADO: PRESENTE.

Directo (sin multiplexor). nxt-python trae la clase nativa
`nxt.sensor.hitechnic.EOPD` (BaseAnalogSensor) — verificado en el código
fuente instalado, no en la ayuda del fabricante. Dos modos:
`set_range_long()` (más sensible, ~4x, puede saturar con blancos) o
`set_range_short()` (evita saturación).

BUG REAL ENCONTRADO Y CORREGIDO (2026-07-10): `arrancar()` nunca llamaba a
NINGUNO de los dos (`set_range_long`/`set_range_short`), y ni
`nxt.sensor.Sensor.__init__` ni `BaseAnalogSensor` fijan un modo por
defecto — el puerto quedaba SIN CONFIGURAR. Eso hacía que
`get_valid_input_values().raw_value` leyera 0 siempre, y como
`get_raw_value() = 1023 - raw_value`, el resultado era 1023 (saturado al
máximo) en TODO ciclo, sin importar la distancia real — confirmado en
campo: el mismo valor exacto (`eopd_raw=7.81631749350903`, que es
`250/sqrt(1023)`) en corridas separadas por minutos/horas. Se probaron
ambos modos en campo (2026-07-10): `set_range_short()` (LIGHT_INACTIVE)
saturó al extremo OPUESTO (raw_value pegado a 1023, division por cero,
cae al valor de respaldo 250.0 fijo — sigue sin servir). `set_range_long()`
(LIGHT_ACTIVE, activa el emisor propio del sensor) SÍ dio lecturas que
varían ciclo a ciclo (125.0 / 144.34, es decir raw≈3-4) — es el modo
correcto para este sensor/geometría. `arrancar()` ahora llama a
`set_range_long()`.

`get_scaled_value()` (= get_sample) da un valor que escala linealmente con
la distancia — es lo que se usa como "distancia" de este sensor. NO es cm;
es una unidad propia del EOPD (mayor = más cerca, según fórmula del
fabricante: SCALE_CONSTANT/sqrt(raw)).
"""
from __future__ import annotations

import time


class OrganoEOPD:
    def __init__(self, conexion, numero_puerto: int) -> None:
        self.conexion = conexion
        self.numero_puerto = numero_puerto
        self.sensor = None
        self._ultimo_ts = 0.0

    def arrancar(self) -> bool:
        try:
            import nxt.sensor.hitechnic as ht
            self.sensor = self.conexion.get_sensor_puerto(self.numero_puerto, ht.EOPD)
            self.sensor.set_range_long()
            return True
        except Exception as e:
            print(f"[OrganoEOPD] no se pudo abrir puerto {self.numero_puerto}: {e}")
            return False

    @property
    def vivo(self) -> bool:
        return self.sensor is not None and (time.time() - self._ultimo_ts) < 2.0

    def leer(self) -> float | None:
        if self.sensor is None:
            return None
        try:
            valor = self.sensor.get_scaled_value()
            self._ultimo_ts = time.time()
            return float(valor)
        except Exception as e:
            print(f"[OrganoEOPD] error de lectura: {e}")
            return None

    def inyectar(self, fila: dict) -> None:
        v = self.leer()
        fila["eopd_raw"] = v
        fila["eopd_vivo"] = 1 if (v is not None) else 0

    def cerrar(self) -> None:
        self.sensor = None
