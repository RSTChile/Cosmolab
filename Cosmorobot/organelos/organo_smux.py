#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_smux — Multiplexor HiTechnic en puerto 4: Touch/Gyro/Accel/Compass.
ESTADO: PRESENTE (implementado a mano — nxt-python NO trae esta clase).

PROCEDENCIA DEL PROTOCOLO: nxt-python no incluye soporte para el
multiplexor HiTechnic (NSX2020). Se implementó el protocolo I2C a mano,
usando como referencia el driver de kernel GPL de ev3dev (código real,
funcional, no un tutorial ni una suposición):
    github.com/ev3dev/lego-linux-drivers
    sensors/ht_nxt_smux.h, sensors/ht_nxt_smux.c

MAPA DE REGISTROS (verbatim del header, confianza ALTA):
    comando         0x20   (0=halt, 1=detect, 2=run)
    estado          0x21   (bit0=batt, bit1=busy, bit2=halt, bit3=error)
    config canal 1  0x22   } cada bloque = 5 bytes:
    config canal 2  0x27   }   [0]=modo [1]=tipo [2]=cant_i2c
    config canal 3  0x2C   }   [3]=addr_i2c [4]=reg_i2c
    config canal 4  0x31   }
    datos analog.   0x36 / 0x38 / 0x3A / 0x3C  (canal 1/2/3/4, 2 bytes c/u)
    datos I2C       0x40 / 0x50 / 0x60 / 0x70  (canal 1/2/3/4, 16 bytes c/u,
                     espejo de los registros del sensor conectado, empezando
                     en el "reg_i2c" configurado)
    bits de modo:   I2C=bit0 (0x01), 9V_EN=bit1 (0x02)

CANALES DE ESTE ROBOT (config/puertos.py, confirmado por Alexis):
    canal 1 = Touch (contacto trasero)     -> ANALÓGICO
    canal 2 = Gyro HiTechnic (trasero)     -> ANALÓGICO (nxt-python mismo
                                               clasifica Gyro como
                                               BaseAnalogSensor, no I2C,
                                               incluso en puerto directo)
    canal 3 = Accelerometer HiTechnic      -> I2C (addr 0x02, reg 0x42,
                                               6 bytes, mismo formato que
                                               nxt.sensor.hitechnic.Accelerometer)
    canal 4 = Compass HiTechnic (superior) -> I2C (addr 0x02, reg 0x42,
                                               2 bytes: heading + adder,
                                               mismo cálculo que
                                               nxt.sensor.hitechnic.Compass)

CONFIANZA Y LO QUE FALTA VERIFICAR EN CAMPO (honestidad, LF):
    ALTA:  direcciones de todos los registros (verbatim del driver real).
    ALTA:  fórmula de accel/compass (copiada de nxt.sensor.hitechnic, que
           SÍ está probada — solo cambia DÓNDE se lee el byte).
    MEDIA: formato/endianness de los 2 bytes analógicos (Touch, Gyro) —
           se asume little-endian sin verificar; puede estar invertido.
    MEDIA: polaridad del Touch (¿mayor valor = presionado, o menor?) — sin
           verificar; se expone el valor CRUDO además del booleano
           derivado para poder recalibrar con datos reales.
    MEDIA: byte "tipo" (offset 1 del config) — se deja en 0 (no se encontró
           el enum exacto que esperaba el firmware); si el multiplexor
           necesita un valor específico para reconocer el sensor, esto
           podría no leer bien hasta corregirlo.
    Todo esto se deja así a propósito por decisión de Alexis
    (2026-07-10): construir completo, corregir después con datos logueados,
    no seguir probando pieza por pieza.
"""
from __future__ import annotations

import time

from nxt.sensor.digital import BaseDigitalSensor


class SMUX(BaseDigitalSensor):
    """Multiplexor HiTechnic — acceso de bajo nivel a sus registros."""

    I2C_ADDRESS = BaseDigitalSensor.I2C_ADDRESS.copy()
    I2C_ADDRESS.update({
        "command": (0x20, "B"),
        "status": (0x21, "B"),
        "ch1_config": (0x22, "5B"),
        "ch2_config": (0x27, "5B"),
        "ch3_config": (0x2C, "5B"),
        "ch4_config": (0x31, "5B"),
        "ch1_analog": (0x36, "<H"),
        "ch2_analog": (0x38, "<H"),
        "ch3_i2c_accel": (0x60, "3b3B"),  # mismo formato que hitechnic.Accelerometer
        "ch4_i2c_compass": (0x70, "2B"),   # heading_byte, adder_byte
    })

    #: dirección I2C del SMUX (2026-07-10, bug real encontrado): HiTechnic usa
    #: 0x02 para sus sensores comunes (Color, EOPD, etc. — de ahí el default de
    #: BaseDigitalSensor), pero el SMUX es uno de los pocos dispositivos
    #: HiTechnic con dirección propia distinta. Confirmado cruzando la fuente
    #: real del driver de kernel ev3dev (nxt_i2c_sensor_defs.c,
    #: HT_NXT_SENSOR_MUX: default_address 7-bit = 0x08 -> 8-bit = 0x10) con el
    #: mismo patrón ya presente en nxt-python para otros sensores HiTechnic no
    #: estándar (nxt/sensor/hitechnic.py: I2C_DEV = 0x10, comentado "different
    #: from standard 0x02"). Con 0x02 (el default heredado) TODA lectura del
    #: SMUX daba timeout, incluso los registros de identificación — el hardware
    #: nunca estuvo mal, era la dirección de bus.
    I2C_DEV = 0x10

    COMMAND_HALT = 0
    COMMAND_DETECT = 1
    COMMAND_RUN = 2
    CONFIG_I2C = 0x01
    CONFIG_9V_EN = 0x02

    def __init__(self, brick, port):
        super().__init__(brick, port, check_compatible=False)

    def halt(self) -> None:
        self.write_value("command", (self.COMMAND_HALT,))

    def run(self) -> None:
        self.write_value("command", (self.COMMAND_RUN,))

    def _config_analogico(self, canal: int, con_9v: bool) -> None:
        modo = self.CONFIG_9V_EN if con_9v else 0
        self.write_value(f"ch{canal}_config", (modo, 0, 0, 0, 0))

    def _config_i2c(self, canal: int, i2c_addr_7bit: int, i2c_reg: int, cantidad: int) -> None:
        modo = self.CONFIG_I2C | self.CONFIG_9V_EN
        addr8 = (i2c_addr_7bit << 1) & 0xFF
        self.write_value(f"ch{canal}_config", (modo, 0, min(cantidad, 8), addr8, i2c_reg))

    def configurar_canales(self) -> None:
        """Configura los 4 canales para este robot (ver mapeo arriba) y
        arranca el barrido. Debe llamarse una vez al inicio."""
        self.halt()
        time.sleep(0.05)
        self._config_analogico(1, con_9v=False)       # Touch
        self._config_analogico(2, con_9v=True)         # Gyro (activo, necesita 9V)
        self._config_i2c(3, 0x02, 0x42, 6)              # Accelerometer
        self._config_i2c(4, 0x02, 0x42, 2)              # Compass
        time.sleep(0.05)
        self.write_value("command", (self.COMMAND_DETECT,))  # detectar sensores en cada canal
        time.sleep(0.5)
        self.run()
        time.sleep(0.3)  # dar tiempo al primer barrido del multiplexor

    def leer_touch_raw(self) -> int:
        return self.read_value("ch1_analog")[0]

    def leer_gyro_raw(self) -> int:
        return self.read_value("ch2_analog")[0]

    def leer_accel(self):
        xh, yh, zh, xl, yl, zl = self.read_value("ch3_i2c_accel")
        x = xh << 2 | xl
        y = yh << 2 | yl
        z = zh << 2 | zl
        return x, y, z

    def leer_compass_deg(self) -> float:
        two_deg, adder = self.read_value("ch4_i2c_compass")
        return two_deg * 2 + adder


class OrganoSMUX:
    """Envoltorio con la misma convención (arrancar/vivo/inyectar/cerrar)
    que el resto de organelos de sensor."""

    #: umbral crudo para decidir "presionado" — SIN VERIFICAR, ajustar con
    #: datos reales del datalog (ver docstring del módulo).
    TOUCH_UMBRAL_PRESIONADO = 500

    def __init__(self, conexion, numero_puerto: int) -> None:
        self.conexion = conexion
        self.numero_puerto = numero_puerto
        self.smux: SMUX | None = None
        self._ultimo_ts = 0.0

    def arrancar(self) -> bool:
        try:
            self.smux = self.conexion.get_sensor_puerto(self.numero_puerto, SMUX)
            self.smux.configurar_canales()
            return True
        except Exception as e:
            print(f"[OrganoSMUX] no se pudo abrir/configurar puerto {self.numero_puerto}: {e}")
            self.smux = None
            return False

    @property
    def vivo(self) -> bool:
        return self.smux is not None and (time.time() - self._ultimo_ts) < 2.0

    def inyectar(self, fila: dict) -> None:
        if self.smux is None:
            fila.setdefault("touch_raw", None)
            fila.setdefault("touch_presionado", 0)
            fila.setdefault("gyro_raw", None)
            fila.setdefault("accel_x", None)
            fila.setdefault("accel_y", None)
            fila.setdefault("accel_z", None)
            fila.setdefault("compass_deg", None)
            fila["smux_vivo"] = 0
            return
        try:
            touch_raw = self.smux.leer_touch_raw()
            gyro_raw = self.smux.leer_gyro_raw()
            ax, ay, az = self.smux.leer_accel()
            compass = self.smux.leer_compass_deg()
            fila["touch_raw"] = touch_raw
            fila["touch_presionado"] = 1 if touch_raw < self.TOUCH_UMBRAL_PRESIONADO else 0
            fila["gyro_raw"] = gyro_raw
            fila["accel_x"] = ax
            fila["accel_y"] = ay
            fila["accel_z"] = az
            fila["compass_deg"] = compass
            fila["smux_vivo"] = 1
            self._ultimo_ts = time.time()
        except Exception as e:
            print(f"[OrganoSMUX] error de lectura: {e}")
            fila.setdefault("touch_raw", None)
            fila.setdefault("touch_presionado", 0)
            fila.setdefault("gyro_raw", None)
            fila.setdefault("accel_x", None)
            fila.setdefault("accel_y", None)
            fila.setdefault("accel_z", None)
            fila.setdefault("compass_deg", None)
            fila["smux_vivo"] = 0

    def cerrar(self) -> None:
        if self.smux is not None:
            try:
                self.smux.halt()
            except Exception:
                pass
        self.smux = None
