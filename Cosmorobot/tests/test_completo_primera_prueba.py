#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_completo_primera_prueba.py — primera prueba del programa
completo: todos los sensores, CambioTotal, capa reactiva doble, Bluetooth.

Acotada a 20 ciclos: se detiene sola y libera los motores, sin necesitar
Ctrl+C. Robot libre de cable (Bluetooth) — el PC debe permanecer conectado
mientras corre (sin cerebro remoto, el robot no actúa).

Uso:
    py tests/test_completo_primera_prueba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import main

if __name__ == "__main__":
    print("Prueba completa acotada: 20 ciclos por Bluetooth, luego se detiene sola.")
    main(max_ciclos=20, metodo="bluetooth")
