#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conexion_nxt.backend_bluetooth_nativo — Bluetooth SIN PyBluez.

PROBLEMA: el backend Bluetooth propio de nxt-python (`nxt/backend/bluetooth.py`)
necesita el paquete `bluetooth` (PyBluez). PyBluez está abandonado y no
compila en este sistema (Python 3.13, Windows, sin Visual Studio C++):
falla con "use_2to3 is invalid" (el paquete original) y los forks
mantenidos (`pybluez2`, `PyBluez-updated`) también fallan al compilar o
tienen bugs de empaquetado (verificado 2026-07-10).

SOLUCIÓN: Windows expone Bluetooth RFCOMM directo en el módulo estándar
`socket` de Python (`socket.AF_BLUETOOTH` + `socket.BTPROTO_RFCOMM`),
verificado disponible en este sistema — no necesita ninguna librería
externa. Esta clase implementa el mismo contrato mínimo que
`nxt.brick.Brick` espera de su objeto "sock" (ver brick.py: usa
`.bsize`, `.close()`, `.send()`, `.recv()`), igual que
`nxt/backend/bluetooth.py:BluetoothSock`, pero sobre socket nativo.

Uso:
    from conexion_nxt.backend_bluetooth_nativo import conectar_bluetooth_nativo
    brick = conectar_bluetooth_nativo("00:16:53:1C:03:23")
"""
from __future__ import annotations

import socket
import struct


class SocketBluetoothNativo:
    """Mismo contrato que nxt.backend.bluetooth.BluetoothSock, pero usando
    socket.AF_BLUETOOTH nativo de Python en vez de PyBluez."""

    bsize = 118  # mismo tamaño de bloque que el backend oficial
    type = "bluetooth"

    #: Canal RFCOMM estándar del NXT (mismo valor que nxt.backend.bluetooth.PORT)
    PUERTO_RFCOMM = 1

    def __init__(self, host: str) -> None:
        self._host = host
        self._sock: socket.socket | None = None

    def __str__(self) -> str:
        return f"Bluetooth nativo ({self._host})"

    def connect(self):
        """Conecta y devuelve un nxt.brick.Brick ya listo para usar."""
        import nxt.brick

        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        s.connect((self._host, self.PUERTO_RFCOMM))
        self._sock = s
        return nxt.brick.Brick(self)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, data: bytes) -> None:
        """Igual que BluetoothSock.send: antepone el largo del mensaje (2 bytes,
        little-endian) — es el framing que espera el firmware del NXT."""
        data = struct.pack("<H", len(data)) + data
        self._sock.send(data)

    def recv(self) -> bytes:
        """Igual que BluetoothSock.recv: lee 2 bytes de largo, luego el mensaje
        completo (con reintentos, un socket TCP/RFCOMM puede entregar en
        varios pedazos)."""
        header = self._sock.recv(2)
        (plen,) = struct.unpack("<H", header)
        data = b""
        while len(data) < plen:
            chunk = self._sock.recv(plen - len(data))
            if not chunk:
                break
            data += chunk
        return data


def conectar_bluetooth_nativo(host_mac: str):
    """Conecta directo por MAC de Bluetooth (ej. '00:16:53:1C:03:23') sin
    pasar por nxt.locator (que requeriría el backend 'bluetooth' de
    nxt-python, no disponible aquí). Devuelve un nxt.brick.Brick."""
    return SocketBluetoothNativo(host_mac).connect()
