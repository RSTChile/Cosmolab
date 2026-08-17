#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_motor_primera_prueba.py — primera prueba física de motores.

Corre el ciclo completo de main.py pero ACOTADO a pocos ciclos: se detiene
solo y libera los motores, sin necesitar Ctrl+C. Pensado para el primer
movimiento real del robot con poco radio de cable (evita dejarlo corriendo
sin límite y tensar el cable repetidamente).

Uso:
    py tests/test_motor_primera_prueba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import main

if __name__ == "__main__":
    print("Prueba acotada: 5 ciclos, luego se detiene sola.")
    main(max_ciclos=5)
