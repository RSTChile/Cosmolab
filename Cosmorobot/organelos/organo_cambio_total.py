#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_cambio_total — CambioTotal: la diferencia agregada del
sistema completo (todos los sensores), normalizada.

DISEÑO ORIGINAL (sesión con GPT, retomado 2026-07-10): "el robot no
responde a sensores; responde a configuraciones de estado". CambioTotal es
esa configuración de estado resumida en un número: cuánto cambió el mundo
percibido, en total, desde el ciclo anterior. Se usa para modular —no
determinar— la selección de acción (más CambioTotal → más peso a
exploración/rotación; poco CambioTotal sostenido → conteo para disparar
exploración).

CORRECCIÓN DE DISEÑO (ver config/escalas_sensores.py): cada |Δ| se divide
por la escala característica del sensor ANTES de sumar. Sin esto, el
sensor de mayor rango numérico domina la suma y CambioTotal deja de medir
lo que dice medir.

Este organelo es CONSUMIDOR puro: no lee hardware directamente, solo
recibe la `fila` ya poblada por los demás organelos de sensor (ultrasónico,
EOPD, color, SMUX) y calcula la diferencia contra la fila anterior.
"""
from __future__ import annotations

from config import escalas_sensores as E


def _delta_lineal(nuevo, anterior) -> float | None:
    if nuevo is None or anterior is None:
        return None
    return abs(float(nuevo) - float(anterior))


def _delta_circular(nuevo, anterior, periodo: float = 360.0) -> float | None:
    """Diferencia angular correcta: min(|Δ|, periodo-|Δ|). Sin esto, un
    giro real de 1° cerca de 0°/360° se mediría como 359°."""
    if nuevo is None or anterior is None:
        return None
    d = abs(float(nuevo) - float(anterior)) % periodo
    return min(d, periodo - d)


class OrganoCambioTotal:
    def __init__(self) -> None:
        self._anterior: dict = {}

    def actualizar(self, fila: dict) -> dict:
        """Lee los valores actuales de `fila` (ya escritos por los
        organelos de sensor), calcula CambioTotal contra el estado anterior
        guardado, escribe el resultado (y los deltas por sensor, para
        diagnóstico) en `fila`, y actualiza su propio estado anterior."""
        act = self._anterior
        deltas: dict[str, float] = {}

        if E.SENSORES_EN_CAMBIO_TOTAL.get("ultra_cm"):
            d = _delta_lineal(fila.get("ultra_cm"), act.get("ultra_cm"))
            if d is not None:
                deltas["d_ultra"] = d / E.ESCALA_ULTRA_CM

        if E.SENSORES_EN_CAMBIO_TOTAL.get("eopd_raw"):
            d = _delta_lineal(fila.get("eopd_raw"), act.get("eopd_raw"))
            if d is not None:
                deltas["d_eopd"] = d / E.ESCALA_EOPD

        if E.SENSORES_EN_CAMBIO_TOTAL.get("color_luminosidad"):
            d = _delta_lineal(fila.get("color_luminosidad"), act.get("color_luminosidad"))
            if d is not None:
                deltas["d_color"] = d / E.ESCALA_COLOR_LUMINOSIDAD

        if E.SENSORES_EN_CAMBIO_TOTAL.get("gyro_raw"):
            d = _delta_lineal(fila.get("gyro_raw"), act.get("gyro_raw"))
            if d is not None:
                deltas["d_gyro"] = d / E.ESCALA_GYRO_RAW

        if E.SENSORES_EN_CAMBIO_TOTAL.get("accel"):
            dx = _delta_lineal(fila.get("accel_x"), act.get("accel_x"))
            dy = _delta_lineal(fila.get("accel_y"), act.get("accel_y"))
            dz = _delta_lineal(fila.get("accel_z"), act.get("accel_z"))
            partes = [d for d in (dx, dy, dz) if d is not None]
            if partes:
                deltas["d_accel"] = (sum(partes) / len(partes)) / E.ESCALA_ACCEL

        if E.SENSORES_EN_CAMBIO_TOTAL.get("compass_deg"):
            d = _delta_circular(fila.get("compass_deg"), act.get("compass_deg"))
            if d is not None:
                deltas["d_compass"] = d / E.ESCALA_COMPASS_DEG

        cambio_total = sum(deltas.values())

        fila["cambio_total"] = round(cambio_total, 4)
        for nombre, valor in deltas.items():
            fila[nombre] = round(valor, 4)

        # actualizar estado anterior para el próximo ciclo
        for clave in ("ultra_cm", "eopd_raw", "color_luminosidad", "gyro_raw",
                      "accel_x", "accel_y", "accel_z", "compass_deg"):
            if fila.get(clave) is not None:
                self._anterior[clave] = fila[clave]

        return {"cambio_total": cambio_total, **deltas}
