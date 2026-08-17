#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_motor — el actuador: motores A (izquierdo) + C (derecho).

Implementa el "Volante" (steering, -100..100) con la MISMA semántica que
el bloque Mover de NXT-G (ver ayuda oficial de LEGO, Help_Move.htm):
    < 0  gira hacia el motor izquierdo
    > 0  gira hacia el motor derecho
    extremos ±100 = rotación pura; 0 = recto; intermedios = arco.

DECISIÓN DE DISEÑO (2026-07-10, tras pruebas físicas con Alexis):
Se abandonó `nxt.motor.SynchronizedMotors`. La propia librería la marca
"THIS CODE IS EXPERIMENTAL!!!" en su código fuente (motor.py:409-414), y
en pruebas reales resultó NO CONFIABLE: una vez no produjo ningún
movimiento (con baterías buenas, sin excepción — resultó ser batería baja
en un intento anterior, pero en un segundo intento YA con baterías
nuevas, un comando "adelante + arco amplio derecha" hizo que la rueda
IZQUIERDA girara en REVERSA, sin relación clara con lo pedido).
En cambio, motores INDIVIDUALES con `Motor.run()`/`Motor.turn()` (API
simple, no experimental) se probaron limpios: A→gira izquierda,
C→gira derecha, ambos confirmados visualmente.
Por eso aquí el volante se implementa A MANO, con dos llamadas
independientes a `Motor.run()` por rueda — dirección diferencial
estándar: cada rueda recibe una potencia distinta (incluso de signo
distinto) según el volante, sin depender de ningún modo "sync" del
firmware.
"""
from __future__ import annotations

import time


class OrganoMotor:
    def __init__(self, conexion, puerto_izquierdo: str, puerto_derecho: str) -> None:
        self.conexion = conexion
        self.puerto_izquierdo = puerto_izquierdo
        self.puerto_derecho = puerto_derecho
        self.motor_izq = None
        self.motor_der = None

    def arrancar(self) -> bool:
        try:
            self.motor_izq = self.conexion.get_motor_puerto(self.puerto_izquierdo)
            self.motor_der = self.conexion.get_motor_puerto(self.puerto_derecho)
            return True
        except Exception as e:
            print(f"[OrganoMotor] no se pudo abrir motores: {e}")
            return False

    def mover(self, volante: float, potencia: int, adelante: bool = True,
              duracion_s: float = 0.6) -> None:
        """Movimiento diferencial A+C, bloqueante durante `duracion_s`.
        `volante` en [-100, 100]; `potencia` en [0, 100]; `adelante` decide
        el sentido base (True=adelante, False=atrás).

        Fórmula de dirección diferencial (steering -100..100), estándar,
        NO depende de ninguna clase experimental:
            steering=0    -> ambas ruedas misma potencia (recto)
            steering=+100 -> izquierda +potencia, derecha -potencia (rota
                              sobre el eje hacia la derecha)
            steering=-100 -> izquierda -potencia, derecha +potencia (rota
                              hacia la izquierda)
            intermedios   -> arco (una rueda se frena/invierte gradual)
        PENDIENTE DE VERIFICAR EN CAMPO (con los datos que se van a
        loguear): que el sentido de giro coincida exactamente con lo
        documentado en config/pool_acciones.py. Si sale invertido, el
        arreglo es de una línea (invertir el signo de `v` abajo).
        """
        signo = 1 if adelante else -1
        p = max(0, min(100, abs(potencia))) * signo
        v = max(-100.0, min(100.0, volante)) / 100.0  # normalizado [-1, 1]

        if v >= 0:
            power_izq = p
            power_der = p * (1 - 2 * v)
        else:
            power_izq = p * (1 + 2 * v)
            power_der = p

        power_izq = int(max(-100, min(100, round(power_izq))))
        power_der = int(max(-100, min(100, round(power_der))))

        self.motor_izq.run(power=power_izq, regulated=False)
        self.motor_der.run(power=power_der, regulated=False)
        time.sleep(max(0.0, duracion_s))
        self.motor_izq.idle()
        self.motor_der.idle()

    def detener(self) -> None:
        """Frena ambos motores (mantiene posición, con torque). Dominio
        'detener_activo' — usado por la capa reactiva."""
        try:
            self.motor_izq.brake()
            self.motor_der.brake()
        except Exception as e:
            print(f"[OrganoMotor] error al frenar: {e}")

    def liberar(self) -> None:
        """Suelta ambos motores (sin torque). Dominio 'liberar' — distinto de
        detener: el robot queda disponible para ser movido externamente."""
        try:
            self.motor_izq.idle()
            self.motor_der.idle()
        except Exception as e:
            print(f"[OrganoMotor] error al liberar: {e}")

    # --- movimientos de la capa reactiva (deterministas, sin deliberación) ---
    def retroceder_breve(self, potencia: int, duracion_s: float) -> None:
        self.mover(volante=0, potencia=potencia, adelante=False, duracion_s=duracion_s)

    def girar_breve(self, potencia: int, duracion_s: float, hacia_derecha: bool = True) -> None:
        self.mover(volante=(100 if hacia_derecha else -100), potencia=potencia,
                    adelante=True, duracion_s=duracion_s)

    def cerrar(self) -> None:
        self.liberar()
