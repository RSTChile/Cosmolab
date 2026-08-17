#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datalog.registrador — registro de sesión a CSV (el "Data Logging" del robot).

Un CSV simple por sesión, con timestamp en el nombre. Cada ciclo del loop
principal escribe una fila. Deliberadamente plano (sin dependencias) para
poder abrirlo en Excel/pandas después del experimento.

DECISIÓN DE DISEÑO (bug real encontrado en la primera corrida completa,
2026-07-10): las filas NO tienen siempre las mismas columnas — p.ej.
`organo_cambio_total` solo agrega las columnas `d_ultra`/`d_eopd`/etc. a
partir del 2º ciclo (el 1º no tiene "anterior" con qué comparar), y las
filas de veto reactivo tienen menos columnas que las normales. Fijar las
columnas con la PRIMERA fila (como hacía la versión anterior) revienta en
cuanto aparece una columna nueva. Por eso ahora se BUFFEREA todo en
memoria y se escribe el CSV recién al cerrar, con la UNIÓN de todas las
columnas que aparecieron en la sesión (orden de primera aparición). Para
sesiones de duración normal (cientos/miles de ciclos) esto no pesa nada.
"""
from __future__ import annotations

import csv
import os
import time


class Registrador:
    def __init__(self, carpeta: str | None = None) -> None:
        self.carpeta = carpeta or os.path.join(os.path.dirname(__file__))
        os.makedirs(self.carpeta, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.ruta = os.path.join(self.carpeta, f"sesion_{ts}.csv")
        self._filas: list[dict] = []
        self._columnas: list[str] = []  # orden de primera aparición
        print(f"[Registrador] escribiendo en {self.ruta}")

    def escribir(self, fila: dict) -> None:
        for clave in fila.keys():
            if clave not in self._columnas:
                self._columnas.append(clave)
        self._filas.append(dict(fila))

    def cerrar(self) -> None:
        try:
            with open(self.ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._columnas, restval="")
                writer.writeheader()
                for fila in self._filas:
                    writer.writerow(fila)
            print(f"[Registrador] {len(self._filas)} filas guardadas en {self.ruta}")
        except Exception as e:
            print(f"[Registrador] error al guardar: {e}")
