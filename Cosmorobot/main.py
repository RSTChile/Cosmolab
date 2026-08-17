#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — CosmoRobot: el ciclo D -> R -> Acción -> A, completo.

Este archivo SOLO ensambla organelos ya definidos; no contiene lógica de
sensor, motor ni decisión propia (esa vive en cada organelo). Si un organelo
cambia, este archivo no debería necesitar cambiar (principio de modularidad).

VERSIÓN COMPLETA (2026-07-10, por pedido de Alexis: "no probar pieza por
pieza, construir el programa completo con todos los sensores, el diseño
original, y corregir después con los datos"):

FLUJO POR CICLO
----------------
1. D (captura):      leer TODOS los sensores (ultrasónico, EOPD, color,
                      SMUX: touch/gyro/accel/compass) -> fila.
2. CambioTotal:       organo_cambio_total agrega la diferencia normalizada
                      de todos los sensores contra el ciclo anterior (NO
                      solo el ultrasónico) — "el robot no responde a
                      sensores, responde a configuraciones de estado".
3. CAPA REACTIVA:     si hay obstáculo frontal cerca (ultrasónico) O
                      colisión trasera (touch) -> veto determinista
                      (detener + retroceder + girar), SIN pasar por el
                      pool. Deja huella en memoria episódica ('trauma').
4. R -> Acción:       si NO hubo veto, se delibera: pool de Volante
                      modulado por CambioTotal (conflicto D_actual) y por
                      el contador de ciclos sin cambio (fuerza exploración
                      si el mundo lleva mucho tiempo igual).
5. Instanciación:     potencia y duración se sortean dentro de un rango
                      (slack, Principio de Reserva Estructural).
6. Ejecutar:          organo_motor.mover(volante, potencia, duracion).
7. A (actualización): nueva lectura de todos los sensores -> error
                      respecto a la distancia objetivo -> aprender()
                      actualiza la valencia de la opción elegida.
                      Propiocepción registra bienestar.
8. Log:               todo el ciclo (todos los sensores + CambioTotal +
                      decisión) se escribe en datalog/ — es lo que permite
                      corregir DESPUÉS sin observar cada ciclo en vivo.
"""
from __future__ import annotations

import random
import time

from config import puertos as P
from config import pool_acciones as A
from conexion_nxt.conexion import ConexionNXT
from organelos.organo_ultrasonico import OrganoUltrasonico
from organelos.organo_eopd import OrganoEOPD
from organelos.organo_color import OrganoColor
from organelos.organo_smux import OrganoSMUX
from organelos.organo_energia import OrganoEnergia
from organelos.organo_cambio_total import OrganoCambioTotal
from organelos.organo_motor import OrganoMotor
from organelos.organo_deliberacion import OrganoDeliberacion
from organelos.organo_propiocepcion import OrganoPropiocepcion
from datalog.registrador import Registrador


def main(max_ciclos: int | None = None, metodo: str = "bluetooth") -> None:
    """`max_ciclos`: si se da, el loop se detiene solo tras esa cantidad de
    ciclos (y libera los motores) sin necesitar Ctrl+C. `metodo`:
    "bluetooth" (por defecto — sin cable, ver config/puertos.py para la MAC)
    o "usb" (requiere el driver WinUSB puesto con Zadig)."""
    conexion = ConexionNXT()
    conectado = (conexion.conectar(metodo="bluetooth", host_bluetooth=P.MAC_BLUETOOTH_NXT)
                 if metodo == "bluetooth" else conexion.conectar(metodo="usb"))
    if not conectado:
        print("main: no se pudo conectar al NXT.")
        return

    conexion.brick.play_tone(440, 300)  # sonido de inicio (pedido por Alexis, 2026-07-10)

    ultrasonico = OrganoUltrasonico(conexion, P.PUERTO_ULTRASONICO)
    eopd = OrganoEOPD(conexion, P.PUERTO_EOPD)
    color = OrganoColor(conexion, P.PUERTO_COLOR)
    smux = OrganoSMUX(conexion, P.PUERTO_SMUX)
    energia = OrganoEnergia(conexion)
    cambio_total = OrganoCambioTotal()
    motor = OrganoMotor(conexion, P.MOTOR_IZQUIERDO, P.MOTOR_DERECHO)
    mente = OrganoDeliberacion()
    propiocepcion = OrganoPropiocepcion()
    log = Registrador()

    # Cada sensor se arranca por separado; si uno falla, se sigue sin él
    # (inyecta None/vivo=0) en vez de abortar todo el robot por un sensor.
    ultrasonico_ok = ultrasonico.arrancar()
    eopd_ok = eopd.arrancar()
    color_ok = color.arrancar()
    smux_ok = smux.arrancar()
    energia_ok = energia.arrancar()
    motor_ok = motor.arrancar()
    print(f"main: sensores -> ultrasonico={ultrasonico_ok} eopd={eopd_ok} "
          f"color={color_ok} smux={smux_ok} energia={energia_ok} motor={motor_ok}")
    if not ultrasonico_ok or not motor_ok:
        print("main: ultrasónico y motor son mínimos indispensables. Abortando.")
        conexion.cerrar()
        return

    print("main: CosmoRobot en marcha. Ctrl+C para detener.")
    t0 = time.time()
    ultima_decision: float | None = None
    ciclo = 0
    contador_sin_cambio = 0
    contador_vetos_consecutivos = 0

    try:
        while max_ciclos is None or ciclo < max_ciclos:
            ciclo += 1
            fila: dict = {"ciclo": ciclo, "t": round(time.time() - t0, 3)}

            # 1) D — captura de TODOS los sensores
            ultrasonico.inyectar(fila)
            if eopd_ok:
                eopd.inyectar(fila)
            if color_ok:
                color.inyectar(fila)
            if smux_ok:
                smux.inyectar(fila)
            if energia_ok:
                energia.inyectar(fila)
            distancia = fila.get("ultra_cm")
            eopd_val = fila.get("eopd_raw")

            # 2) CambioTotal — diferencia agregada y normalizada de TODO el estado
            cambio_total.actualizar(fila)

            # 3) CAPA REACTIVA (LF=0, determinista, protege — no decide)
            colision_trasera = bool(fila.get("touch_presionado"))
            obstaculo_frontal = distancia is not None and distancia < A.UMBRAL_CRITICO_CM
            # Segundo disparador frontal por EOPD (pedido de Alexis, 2026-07-11):
            # el ultrasónico tiene una zona muerta de corto alcance donde deja de
            # servir justo al quedar pegado a un obstáculo (ver UMBRAL_CRITICO_CM
            # más arriba). El EOPD mide luz reflejada, no eco — no tiene esa zona
            # muerta. Convención confirmada en campo: el valor BAJA cuanto más
            # cerca está el objeto.
            obstaculo_cercano_eopd = eopd_val is not None and eopd_val < A.UMBRAL_EOPD_CERCANO
            if obstaculo_frontal or obstaculo_cercano_eopd or colision_trasera:
                if obstaculo_frontal:
                    motivo = "obstaculo_frontal"
                elif obstaculo_cercano_eopd:
                    motivo = "obstaculo_frontal_eopd"
                else:
                    motivo = "colision_trasera"

                # ESCALADA POR PRESIÓN INTERNA, no por fórmula mecánica (pedido
                # de Alexis, 2026-07-10, tras ver al robot atrapado 63 ciclos
                # reintentando el mismo retroceso/giro débil sin separarse de
                # verdad del obstáculo — datalog sesion_20260710_233516.csv).
                # Cada veto consecutivo sube `presion_desacople`, la MISMA
                # señal que organo_propiocepcion ya usa para calcular malestar
                # (no una señal nueva ad-hoc). La potencia/duración real del
                # retroceso sale de esa presión: el robot "quiere salir" cada
                # vez más, no ejecuta un patrón programado.
                contador_vetos_consecutivos += 1
                presion_desacople = min(1.0, contador_vetos_consecutivos / A.LIMITE_VETOS_PRESION_MAXIMA)
                fila["presion_desacople"] = round(presion_desacople, 3)
                propiocepcion.observar(fila)  # deja que el malestar sienta esta presión también
                potencia_reactiva = round(A.REACTIVA_POTENCIA_BASE + presion_desacople *
                                           (A.REACTIVA_POTENCIA_MAX - A.REACTIVA_POTENCIA_BASE))
                escala_duracion = 1.0 + presion_desacople * (A.REACTIVA_DURACION_ESCALA_MAX - 1.0)

                print(f"[reactiva] {motivo} -> veto (presión de desacople={presion_desacople:.2f}, "
                      f"potencia={potencia_reactiva})")
                motor.detener()
                motor.retroceder_breve(potencia=potencia_reactiva,
                                        duracion_s=A.REACTIVA_RETROCESO_S * escala_duracion)
                motor.girar_breve(potencia=potencia_reactiva,
                                   duracion_s=A.REACTIVA_GIRO_S * escala_duracion,
                                   hacia_derecha=random.choice([True, False]))
                fila["veto_reactivo"] = 1
                fila["veto_motivo"] = motivo
                fila["volante_elegido"] = None
                fila["vetos_consecutivos"] = contador_vetos_consecutivos
                if ultima_decision is not None:
                    mente.registrar_episodio(fila["t"], ultima_decision, resultado="trauma",
                                              contexto=motivo)
                log.escribir(fila)
                print(f"  c{ciclo:03d} t={fila['t']:6.2f}s  VETO({motivo})  "
                      f"ultra={distancia}  eopd={eopd_val}  touch={colision_trasera}")
                continue

            contador_vetos_consecutivos = 0
            fila["veto_reactivo"] = 0

            # 4) R -> Acción — deliberación sobre el pool de Volante,
            # modulada por CambioTotal (diseño original con GPT) + contador
            # de estabilidad (exploración forzada si el mundo no cambia).
            ct = fila.get("cambio_total", 0.0)
            if ct <= A.UMBRAL_CAMBIO_TOTAL_ESTABLE:
                contador_sin_cambio += 1
            else:
                contador_sin_cambio = 0
            if contador_sin_cambio >= A.LIMITE_SIN_CAMBIO:
                D_actual = 1.0  # el mundo no cambia -> el sistema introduce diferencia
            else:
                D_actual = min(1.0, ct / A.CAMBIO_TOTAL_ESCALA_CONFLICTO)
            resultado = mente.decidir(A.POOL_VOLANTE, D_actual, current_sp=ultima_decision)
            volante = resultado.decision
            fila["volante_elegido"] = volante
            fila["D_actual"] = round(D_actual, 3)
            fila["contador_sin_cambio"] = contador_sin_cambio
            fila["tiempo_deliberacion_s"] = round(resultado.tiempo_s, 3)

            # 5) Instanciación (slack / PRE — no siempre el mismo parámetro)
            potencia = random.randint(A.POTENCIA_MIN, A.POTENCIA_MAX)
            duracion = random.uniform(A.DURACION_MIN_S, A.DURACION_MAX_S)
            fila["potencia"] = potencia
            fila["duracion_s"] = round(duracion, 3)

            # 6) Ejecutar
            motor.mover(volante=volante, potencia=potencia, adelante=True, duracion_s=duracion)

            # 7) A — actualización: nueva lectura, error, aprendizaje
            fila_post: dict = {}
            ultrasonico.inyectar(fila_post)
            nueva_distancia = fila_post.get("ultra_cm")
            fila["ultra_cm_post"] = nueva_distancia
            if nueva_distancia is not None:
                error = abs(nueva_distancia - A.DISTANCIA_OBJETIVO_CM)
                costo = duracion * potencia / 100.0
                mente.aprender(volante, error=error, costo_pagado=costo, dt=duracion)
                fila["error_post"] = round(error, 2)

            fila["e_R"] = fila.get("error_post", 0.0)
            propiocepcion.observar(fila)

            log.escribir(fila)
            ultima_decision = volante

            nombre_volante = A.NOMBRES_VOLANTE.get(volante, str(volante))
            print(f"  c{ciclo:03d} t={fila['t']:6.2f}s  "
                  f"ultra={distancia}->{nueva_distancia}cm  CT={ct:.3f}  "
                  f"sinCambio={contador_sin_cambio}  D={D_actual:.2f}  "
                  f"volante={volante:+.0f}({nombre_volante})  pot={potencia} dur={duracion:.2f}s  "
                  f"touch={bool(fila.get('touch_presionado'))}  compass={fila.get('compass_deg')}")

        if max_ciclos is not None:
            print(f"main: {max_ciclos} ciclos completados (límite de prueba). Liberando motores.")

    except KeyboardInterrupt:
        print("\nmain: detenido por el usuario.")
    finally:
        # Cada paso de limpieza en su propio try/except: si uno falla (p.ej.
        # la batería se agota a mitad de la limpieza), los siguientes pasos
        # -sobre todo log.cerrar(), que es cuando recién se escribe el CSV a
        # disco- igual se ejecutan. Antes, una falla temprana acá se comía
        # el datalog completo de la sesión (bug real, 2026-07-10).
        for paso in (
            motor.liberar,
            ultrasonico.cerrar,
            *((eopd.cerrar,) if eopd_ok else ()),
            *((color.cerrar,) if color_ok else ()),
            *((smux.cerrar,) if smux_ok else ()),
            *((energia.cerrar,) if energia_ok else ()),
            motor.cerrar,
        ):
            try:
                paso()
            except Exception as e:
                print(f"main: error en limpieza ({paso.__qualname__}): {e}")
        try:
            conexion.brick.play_tone(220, 500)  # sonido de fin (pedido por Alexis, 2026-07-10)
        except Exception as e:
            print(f"main: error al reproducir sonido de fin: {e}")
        log.cerrar()
        conexion.cerrar()


if __name__ == "__main__":
    main()
