#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.escalas_sensores — escalas de normalización para CambioTotal.

PROBLEMA QUE RESUELVE (identificado en el análisis inicial de la sesión,
2026-07-09, sobre el diseño original con GPT): sumar |Δ| crudos de todos
los sensores no funciona — cada uno vive en una escala distinta
(ultrasónico 0-255cm, brújula 0-360° circular, acelerómetro ~±200/g,
etc.). El sensor de mayor rango numérico domina la suma y CambioTotal deja
de medir "cuánto cambió el estado", mide "cuánto cambió el sensor con más
rango". La corrección: normalizar cada |Δ| por una escala característica
ANTES de sumar.

HONESTIDAD (LF): estas escalas son PARÁMETROS DEL DISEÑADOR, no derivados
de una fase de calibración del propio robot (eso queda pendiente, igual
que UMBRAL_CRITICO_CM en pool_acciones.py). Se ponen valores iniciales
razonables y se corrigen con datos reales del datalog, como pidió Alexis
(2026-07-10): construir completo primero, corregir después con datos.
"""
from __future__ import annotations

# Escala = "cuánto cambio en esta unidad se considera significativo".
# CambioTotal = suma de |Δ_sensor| / escala_sensor sobre todos los sensores vivos.
ESCALA_ULTRA_CM = 100.0          # ultrasónico: 0-255cm, ~100cm = cambio grande
ESCALA_EOPD = 50.0                # EOPD: unidad propia (SCALE_CONSTANT/sqrt(raw)) — SIN CALIBRAR
ESCALA_COLOR_LUMINOSIDAD = 500.0  # r+g+b crudo (Colorv2 raw, 16-bit c/u) — SIN CALIBRAR
ESCALA_GYRO_RAW = 200.0           # SMUX canal analógico — SIN CALIBRAR (ver organo_smux.py)
ESCALA_TOUCH_RAW = 200.0          # SMUX canal analógico — SIN CALIBRAR
ESCALA_ACCEL = 50.0               # HiTechnic accel: 200 unidades = 1g (doc. fabricante)
ESCALA_COMPASS_DEG = 30.0         # brújula: 30° = giro notable. Circular (ver nota abajo)

# Qué sensores entran al CambioTotal (permite apagar uno sin tocar el
# organelo — p.ej. si el SMUX aún no está confiable en campo).
SENSORES_EN_CAMBIO_TOTAL = {
    "ultra_cm": True,
    "eopd_raw": True,
    "color_luminosidad": True,
    "gyro_raw": True,
    "accel": True,       # combina accel_x/y/z (ver organo_cambio_total.py)
    "compass_deg": True,
}

# NOTA sobre la brújula: es circular (0°=360°). La diferencia correcta no
# es |nuevo-anterior| directo (eso da 359 para un giro real de 1°), sino
# min(|Δ|, 360-|Δ|). Implementado en organo_cambio_total.py, no aquí.
