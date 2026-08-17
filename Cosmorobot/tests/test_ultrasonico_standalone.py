#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_ultrasonico_standalone.py — prueba aislada de hardware.

Solo conecta y lee el ultrasónico en bucle, SIN mover motores. Úsalo primero
para confirmar que el USB/driver Fantom y el sensor responden, antes de
correr main.py (que sí mueve el robot).

Uso:
    py tests/test_ultrasonico_standalone.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import puertos as P
from conexion_nxt.conexion import ConexionNXT
from organelos.organo_ultrasonico import OrganoUltrasonico


def main() -> None:
    conexion = ConexionNXT()
    if not conexion.conectar():
        print("No se pudo conectar. Revisa USB, batería y driver Fantom.")
        return
    sensor = OrganoUltrasonico(conexion, P.PUERTO_ULTRASONICO)
    if not sensor.arrancar():
        print("No se pudo abrir el sensor ultrasónico.")
        conexion.cerrar()
        return
    print("Leyendo distancia (Ctrl+C para salir)...")
    try:
        while True:
            cm = sensor.leer_cm()
            print(f"distancia: {cm} cm" if cm is not None else "sin lectura")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\ndetenido.")
    finally:
        sensor.cerrar()
        conexion.cerrar()


if __name__ == "__main__":
    main()
