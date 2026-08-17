#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nxt.conexion — capa de comunicación de bajo nivel con el ladrillo NXT.

Envuelve `nxt-python` (https://ni.srht.site/nxt-python/) para que el resto
del proyecto (organelos) no importe `nxt.*` directamente ni sepa de puertos
USB/Bluetooth. Si mañana cambia la librería de bajo nivel, solo se toca este
archivo (principio de modularidad).

API real de nxt-python confirmada (2026-07-09):
    import nxt.locator
    with nxt.locator.find() as b:
        sensor = b.get_sensor(nxt.sensor.Port.S1, nxt.sensor.generic.Ultrasonic)
        motor  = b.get_motor(nxt.motor.Port.A)

MODO BLUETOOTH (2026-07-10): el backend Bluetooth de nxt-python necesita
PyBluez, que no compila en este sistema (ver
conexion_nxt/backend_bluetooth_nativo.py para el detalle). Se usa un
backend propio sobre socket.AF_BLUETOOTH nativo de Python en su lugar.
Permite operar SIN el cable USB (radio de movimiento libre).
"""
from __future__ import annotations

from typing import Optional


class ConexionNXT:
    """Un único punto de entrada al ladrillo. `conectar()` debe llamarse una
    vez al arrancar; el resto de organelos reciben la instancia del brick."""

    def __init__(self) -> None:
        self.brick = None
        self._ctx = None
        self._brick_directo = None  # usado en modo bluetooth (sin context manager)

    def conectar(self, metodo: str = "usb", host_bluetooth: Optional[str] = None) -> bool:
        """Conecta al NXT. `metodo`: "usb" (por defecto, requiere cable y el
        driver WinUSB puesto con Zadig) o "bluetooth" (sin cable, requiere
        `host_bluetooth` con la MAC del ladrillo, ej. "00:16:53:1C:03:23").
        Devuelve True si se conectó."""
        if metodo == "bluetooth":
            return self._conectar_bluetooth(host_bluetooth)
        return self._conectar_usb()

    def _conectar_usb(self) -> bool:
        try:
            import nxt.locator
        except ImportError as e:
            print(f"[ConexionNXT] falta la librería nxt-python: {e}")
            print("[ConexionNXT] instalar con: py -m pip install nxt-python")
            return False
        try:
            self._ctx = nxt.locator.find()
            self.brick = self._ctx.__enter__()
            info = self.brick.get_device_info()
            print(f"[ConexionNXT] conectado por USB: {info[0]}")
            return True
        except Exception as e:
            print(f"[ConexionNXT] no se encontró el NXT por USB: {e}")
            print("[ConexionNXT] verificar: USB conectado, ladrillo encendido, driver WinUSB (Zadig) puesto.")
            return False

    def _conectar_bluetooth(self, host_bluetooth: Optional[str]) -> bool:
        if not host_bluetooth:
            print("[ConexionNXT] modo bluetooth requiere host_bluetooth (MAC del NXT)")
            return False
        try:
            from conexion_nxt.backend_bluetooth_nativo import conectar_bluetooth_nativo
            self._brick_directo = conectar_bluetooth_nativo(host_bluetooth)
            self.brick = self._brick_directo
            info = self.brick.get_device_info()
            print(f"[ConexionNXT] conectado por Bluetooth: {info[0]}")
            return True
        except Exception as e:
            print(f"[ConexionNXT] no se pudo conectar por Bluetooth a {host_bluetooth}: {e}")
            print("[ConexionNXT] verificar: NXT encendido, emparejado, Bluetooth de Windows activo.")
            return False

    def cerrar(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
        elif self._brick_directo is not None:
            try:
                self._brick_directo.close()
            except Exception:
                pass
            self._brick_directo = None
        self.brick = None

    def get_sensor_puerto(self, numero_puerto: int, clase_sensor=None):
        """Devuelve el objeto sensor en el puerto NXT `numero_puerto` (1-4).
        `clase_sensor` es opcional (p.ej. nxt.sensor.generic.Ultrasonic); si
        se omite, nxt-python intenta autodetectar."""
        import nxt.sensor
        puerto = {1: nxt.sensor.Port.S1, 2: nxt.sensor.Port.S2,
                  3: nxt.sensor.Port.S3, 4: nxt.sensor.Port.S4}[numero_puerto]
        if clase_sensor is not None:
            return self.brick.get_sensor(puerto, clase_sensor)
        return self.brick.get_sensor(puerto)

    def get_motor_puerto(self, letra_puerto: str):
        """Devuelve el objeto motor en el puerto NXT 'A', 'B' o 'C'."""
        import nxt.motor
        puerto = {"A": nxt.motor.Port.A, "B": nxt.motor.Port.B, "C": nxt.motor.Port.C}[letra_puerto]
        return self.brick.get_motor(puerto)
