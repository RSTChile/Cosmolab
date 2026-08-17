#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_color — sensor Color HiTechnic (puerto 3). ESTADO: PRESENTE.

Directo (sin multiplexor).

BUG REAL #1 ENCONTRADO Y CORREGIDO (2026-07-10): nxt-python solo trae
`nxt.sensor.hitechnic.Colorv2`, para el modelo NUEVO (product_id "ColorPD").
El sensor físico de este robot se identifica como product_id **"Color"**
(el modelo v1/original) — confirmado con `get_sensor_info()`. Los dos
modelos tienen mapas de registros distintos.

BUG REAL #2 ENCONTRADO Y CORREGIDO (2026-07-10, el que de verdad causaba
"siempre (0,0,0)"): el RAW de este sensor SÍ son 3 enteros de 16 bits
(0-1023, 10 bits reales) empezando en 0x46 — pero en **big-endian**
(byte alto primero), Y no son consecutivos de a 2 en 2 como un array
plano: Red=0x46, Green=**0x48**, Blue=**0x4A**. Mi formato original
`"3H"` (sin prefijo `>`) usaba el orden de bytes NATIVO (little-endian en
x86), que para estos valores pequeños siempre da 0 o basura — de ahí la
lectura fantasma. Fuente que lo resolvió: driver real y probado de
**leJOS** (`lejos.nxt.addon.ColorSensor`, Java), que arma cada raw como
`(buf[0]<<8)|buf[1]` — es decir big-endian explícito — leyendo 0x46/0x48/0x4A
por separado. Confirmado en campo: con el fix, los tres canales pasaron
de (0,0,0) fijo a valores reales que varían ciclo a ciclo (ej. r=14-15,
g=1-2, b=1 en luz ambiente).

Registros del v1 (todos confirmados contra leJOS, fuente de mayor
confianza que el resumen del driver ev3dev usado antes):
    0x41  modo/calibración (write: 0x42=calibrar negro, 0x43=calibrar blanco)
    0x42  color activo, 1 byte, índice 0-17
    0x43  rojo 8 bits (0-255)
    0x44  verde 8 bits (0-255)
    0x45  azul 8 bits (0-255)
    0x46  rojo RAW, 2 bytes big-endian (0-1023)
    0x48  verde RAW, 2 bytes big-endian (0-1023)
    0x4A  azul RAW, 2 bytes big-endian (0-1023)
    0x4C  índice de color de 6 bits (R=bits5-4, G=bits3-2, B=bits1-0)
    0x4D/0x4E/0x4F  R/G/B normalizados, 8 bits cada uno

DECISIÓN DE DISEÑO (de la sesión teórica original, 2026-07-09, sigue
vigente): el índice de color nominal (registro COLOR, 1-17 sin orden) NO
sirve para diferencias/CambioTotal — no es una magnitud ordenada, restar
dos índices no significa nada. Por eso este organelo lee el registro RAW
(RGB crudo 10 bits) y expone la LUMINOSIDAD TOTAL (r+g+b) como el valor
comparable.
"""
from __future__ import annotations

import time

from nxt.sensor.digital import BaseDigitalSensor


class ColorV1(BaseDigitalSensor):
    """Acceso de bajo nivel al HiTechnic Color v1 (product_id "Color").
    No confundir con `nxt.sensor.hitechnic.Colorv2` (modelo distinto,
    mapa de registros distinto — ver docstring del módulo)."""

    I2C_ADDRESS = BaseDigitalSensor.I2C_ADDRESS.copy()
    I2C_ADDRESS.update({
        "color_num": (0x42, "B"),
        "raw_red": (0x46, ">H"),
        "raw_green": (0x48, ">H"),
        "raw_blue": (0x4A, ">H"),
    })

    def __init__(self, brick, port):
        super().__init__(brick, port, check_compatible=False)


class OrganoColor:
    def __init__(self, conexion, numero_puerto: int) -> None:
        self.conexion = conexion
        self.numero_puerto = numero_puerto
        self.sensor = None
        self._ultimo_ts = 0.0

    def arrancar(self) -> bool:
        try:
            self.sensor = self.conexion.get_sensor_puerto(self.numero_puerto, ColorV1)
            return True
        except Exception as e:
            print(f"[OrganoColor] no se pudo abrir puerto {self.numero_puerto}: {e}")
            return False

    @property
    def vivo(self) -> bool:
        return self.sensor is not None and (time.time() - self._ultimo_ts) < 2.0

    def leer(self) -> dict | None:
        if self.sensor is None:
            return None
        try:
            r = self.sensor.read_value("raw_red")[0]
            g = self.sensor.read_value("raw_green")[0]
            b = self.sensor.read_value("raw_blue")[0]
            self._ultimo_ts = time.time()
            return {"r": r, "g": g, "b": b, "luminosidad": r + g + b}
        except Exception as e:
            print(f"[OrganoColor] error de lectura: {e}")
            return None

    def inyectar(self, fila: dict) -> None:
        d = self.leer()
        if d is not None:
            fila["color_r"] = d["r"]
            fila["color_g"] = d["g"]
            fila["color_b"] = d["b"]
            fila["color_luminosidad"] = d["luminosidad"]
            fila["color_vivo"] = 1
        else:
            fila.setdefault("color_r", None)
            fila.setdefault("color_g", None)
            fila.setdefault("color_b", None)
            fila.setdefault("color_luminosidad", None)
            fila["color_vivo"] = 0

    def cerrar(self) -> None:
        self.sensor = None
