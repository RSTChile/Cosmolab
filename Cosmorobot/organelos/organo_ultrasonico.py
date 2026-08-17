#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_ultrasonico — sensor de distancia (puerto 1).

PATRÓN: sigue la convención de ANIMA para lectores de hardware
(Célula_Madre/organelos/VST_LectorSensores.py — patrón, no contenido: ese
archivo lee GPS/luz de un ATmega, específico del organismo E-Planta; aquí
NO hay GPS, es un órgano propio del NXT):
    .arrancar()   -> abre la lectura
    .vivo         -> property, True si hay lectura fresca (watchdog)
    .inyectar(fila) -> escribe el dato crudo en el diccionario de estado compartido
    .cerrar()      -> libera el sensor

A diferencia de los lectores de ANIMA (que usan un hilo daemon porque leen
puertos serie/TCP en background), aquí la lectura es SÍNCRONA y barata (una
sola llamada USB por ciclo), así que no hace falta hilo: `inyectar()` lee y
escribe en el mismo golpe. Se mantiene la MISMA interfaz para que, si más
adelante se vuelve asíncrono, el resto del código no cambie.
"""
from __future__ import annotations

import time


class OrganoUltrasonico:
    def __init__(self, conexion, numero_puerto: int) -> None:
        self.conexion = conexion
        self.numero_puerto = numero_puerto
        self.sensor = None
        self._ultima_lectura_cm: float | None = None
        self._ultimo_ts = 0.0

    def arrancar(self) -> bool:
        try:
            import nxt.sensor
            import nxt.sensor.generic
            self.sensor = self.conexion.get_sensor_puerto(
                self.numero_puerto, nxt.sensor.generic.Ultrasonic
            )
            return True
        except Exception as e:
            print(f"[OrganoUltrasonico] no se pudo abrir puerto {self.numero_puerto}: {e}")
            return False

    def cerrar(self) -> None:
        self.sensor = None

    @property
    def vivo(self) -> bool:
        return self.sensor is not None and (time.time() - self._ultimo_ts) < 2.0

    def leer_cm(self) -> float | None:
        """Lectura directa (0-255 cm; None si el sensor no está listo o fuera
        de rango). No escribe en ninguna fila; solo devuelve el valor."""
        if self.sensor is None:
            return None
        try:
            valor = self.sensor.get_sample()
            self._ultima_lectura_cm = float(valor)
            self._ultimo_ts = time.time()
            return self._ultima_lectura_cm
        except Exception as e:
            print(f"[OrganoUltrasonico] error de lectura: {e}")
            return None

    def inyectar(self, fila: dict) -> None:
        """Escribe la lectura cruda en el diccionario de estado compartido
        (mismo nombre de método que los lectores de ANIMA, por convención)."""
        cm = self.leer_cm()
        fila["ultra_cm"] = cm
        fila["ultra_vivo"] = 1 if (cm is not None) else 0
