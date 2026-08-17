#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.puertos — mapa físico del Robot Cosmosemiótico (CosmoRobot).

Solo DATOS: qué está enchufado dónde. Ningún organelo debe hardcodear un
puerto; todos importan de aquí. Si el cableado físico cambia, se cambia
SOLO este archivo (principio de modularidad de Alexis).

Confirmado por Alexis, sesión 2026-07-09. Cableado reconfirmado 2026-07-10
tras varias pruebas de conector (ver MEMORY.md — el Color de HiTechnic
tuvo contacto intermitente, no un bug de puerto; ahora reconectado firme
en puerto 1):
    Frente:    Color (HiTechnic) -> puerto 1
               Ultrasónico -> puerto 2
               EOPD (HiTechnic) -> puerto 3
               Multiplexor HiTechnic (SMUX) -> puerto 4
    Trasera (vía SMUX, puerto 4):
               canal 1 -> Sensor de Contacto (Touch)
               canal 2 -> Giroscopio (HiTechnic)
               canal 3 -> Acelerómetro (HiTechnic)
    Superior (vía SMUX, puerto 4):
               canal 4 -> Compass (HiTechnic)
    Motores:   A -> lado izquierdo (vista frontal)
               C -> lado derecho (vista frontal)
               B -> LIBRE (reserva estructural, sin exaptar aún — PRE O-N8.19)
"""
from __future__ import annotations

# --- Sensores directos (puerto NXT 1-4) ---
PUERTO_COLOR = 1
PUERTO_ULTRASONICO = 2
PUERTO_EOPD = 3
PUERTO_SMUX = 4

# --- Sensores vía el multiplexor HiTechnic (canales 1-4 del SMUX en puerto 4) ---
SMUX_CANAL_TOUCH = 1
SMUX_CANAL_GYRO = 2
SMUX_CANAL_ACCEL = 3
SMUX_CANAL_COMPASS = 4

# --- Motores ---
MOTOR_IZQUIERDO = "A"
MOTOR_DERECHO = "C"
MOTOR_LIBRE = "B"   # no usado aún; ver PRE (Principio de Reserva Estructural)

# --- Conexión ---
# MAC Bluetooth confirmada del NXT (2026-07-10, vía get_device_info() por USB).
# Ver conexion_nxt/backend_bluetooth_nativo.py — permite operar sin cable.
MAC_BLUETOOTH_NXT = "00:16:53:1C:03:23"
