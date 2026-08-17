#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.pool_acciones — el espacio de acción del robot, como PARÁMETRO, no como lista.

Corrección de diseño (sesión 2026-07-09): "acciones" no son discretas
(avanzar/girar/retroceder como casos separados). Son REGIONES de una familia
continua. El NXT-G Bloque Mover ya lo confirma con su propia semántica:

    Volante (Steering)  ∈ [-100, 100]   numérico
        < 0  → gira hacia el motor izquierdo
        > 0  → gira hacia el motor derecho
        extremos (±100) → rotación pura sobre el eje
        centro (0)      → traslación pura, recto
        intermedios     → arco/curva

Es decir: Traslación, Rotación y Curva son la MISMA variable (Volante), no
tres acciones distintas. Este archivo declara el POOL como un muestreo de esa
variable continua — la deliberación (organo_deliberacion) elige un punto del
pool; la instanciación de parámetros (organo_motor) añade slack alrededor.

Dirección (adelante/atrás) es una variable APARTE (lógica, no numérica: ver
ayuda oficial de LEGO, toma "Dirección" = Verdadero/Falso). No se mete en el
pool de Volante: la decide `main.py` a partir del gradiente (¿acercarse o
alejarse del objetivo?).
"""
from __future__ import annotations

# --- Pool de Volante (steering) — muestreo de la familia continua [-100, 100] ---
# 7 puntos: rotación extrema, arco amplio, arco suave, recto, y sus espejos.
# Nombrado aquí solo para lectura humana (el motor de deliberación NO usa los
# nombres, solo los números — el significado no está prescrito).
POOL_VOLANTE = [-100, -60, -25, 0, 25, 60, 100]
NOMBRES_VOLANTE = {  # documentación, no lógica
    -100: "rotación (izq)", -60: "arco amplio (izq)", -25: "arco suave (izq)",
    0: "traslación recta",
    25: "arco suave (der)", 60: "arco amplio (der)", 100: "rotación (der)",
}

# --- Slack de instanciación (Principio de Reserva Estructural, PRE O-N8.19) ---
# La potencia y duración NO son fijas: se sortean dentro de un rango cada vez
# que se ejecuta una acción. Esto es lo que evita que el robot sea "óptimo" y
# por tanto sin excedente para exaptar.
#
# CALIBRADO EN CAMPO (2026-07-11): con potencia mínima en 40, Alexis observó
# que el robot "casi no se mueve por sí mismo" — el compass quedaba igual
# durante decenas de ciclos seguidos pese a ejecutar comandos de motor cada
# ciclo (datalog sesion_20260711_004009.csv). Sospecha confirmada: 40 (y
# menos aún la mitad de eso en un arco suave, donde una rueda recibe la
# mitad de la potencia) probablemente queda por debajo del umbral real
# donde el motor vence la fricción estática del piso/ruedas — el comando se
# ejecuta pero no hay movimiento físico real. Subido el piso a 75. Tope
# subido a 90 (se mantiene POR DEBAJO de REACTIVA_POTENCIA_MAX=95 más abajo,
# para que la respuesta reactiva de escape siga sintiéndose más urgente que
# la exploración normal).
POTENCIA_MIN = 75
POTENCIA_MAX = 90
DURACION_MIN_S = 0.4
DURACION_MAX_S = 1.2

# --- Capa reactiva (LF=0, protección física, determinista, fuera de la deliberación) ---
# Umbral de distancia bajo el cual el robot frena/retrocede SIN pasar por el
# pool.
# CALIBRADO EN CAMPO (2026-07-10): con el parachoques tocando el obstáculo,
# el ultrasónico marca 1 cm (confirmado por Alexis). Con el umbral original
# de 8 cm el robot quedó atascado contra un muro/sillón sin retroceder: cerca
# del obstáculo el ultrasónico entra en su zona muerta de corto alcance y
# empieza a alternar entre lecturas bajas (14-15 cm) y falsos "fuera de
# rango" (255 cm) en vez de bajar limpiamente — visto en el datalog
# sesion_20260710_223550.csv (fila c063: 14.0->255.0). El veto nunca ve un
# valor confiablemente por debajo de 8, así que nunca dispara. Subido a 20 cm
# para que el veto dispare mientras la lectura todavía es confiable, con
# margen suficiente para el momentum de un ciclo completo de movimiento.
#
# SEGUNDA CALIBRACIÓN (2026-07-10, misma noche): con 20cm el robot igual
# quedó "clavado" ~40 ciclos seguidos a 22cm de un obstáculo (datalog
# sesion_20260710_231405.csv, ciclos 60-100): 22 > 20, así que el veto
# nunca disparaba, y la deliberación seguía eligiendo traslación recta sin
# progresar ni retroceder. Subido a 25cm para dar margen por encima de ese
# punto de estancamiento observado.
UMBRAL_CRITICO_CM = 25.0
REACTIVA_RETROCESO_S = 0.4
REACTIVA_GIRO_S = 0.3

# --- Segundo disparador de cercanía: EOPD, no solo el ultrasónico ---
# Pedido de Alexis (2026-07-11): el ultrasónico tiene una zona muerta de
# corto alcance (ver arriba) donde deja de servir justo cuando el robot ya
# está pegado a algo. El EOPD mide luz reflejada, no eco de sonido — no
# tiene esa misma zona muerta, así que puede cubrir el hueco. Convención
# del EOPD (organo_eopd.py, formula del fabricante): el valor BAJA cuanto
# más CERCA está el objeto (confirmado en campo: pelota a 5cm -> ~12.6;
# aire libre/nada cerca -> ~125-144). Umbral a priori (sin fase de
# calibración propia todavía, mismo estado honesto que UMBRAL_CRITICO_CM):
# por debajo de este valor se considera "objeto pegado", dispara el mismo
# veto reactivo que el ultrasónico.
UMBRAL_EOPD_CERCANO = 40.0

# --- Escalada de la respuesta reactiva por PRESIÓN INTERNA, no por contador ---
# Hallazgo de campo (2026-07-10, datalog sesion_20260710_233516.csv, ciclos
# 61-123): con retroceso/giro a potencia fija (50) el robot quedó atrapado
# 63 ciclos seguidos re-disparando el veto a ~22-24cm sin lograr separarse
# de verdad del obstáculo. Pedido de Alexis: la respuesta no debe escalar
# por una fórmula mecánica (contador * X), sino por la MISMA señal de
# malestar/presión de desacople que ya calcula organo_propiocepcion — el
# robot "quiere salir" cada vez más, no ejecuta un patrón programado.
# main.py alimenta `presion_desacople` (0-1) a partir de vetos consecutivos
# y lo pasa por propiocepción; la potencia/duración real del retroceso sale
# de ese malestar resultante, escalando entre estos límites.
REACTIVA_POTENCIA_BASE = 50
REACTIVA_POTENCIA_MAX = 95
REACTIVA_DURACION_ESCALA_MAX = 2.2  # a malestar máximo, retroceso/giro duran hasta 2.2x lo base
LIMITE_VETOS_PRESION_MAXIMA = 6  # ciclos de veto consecutivo para llegar a presión de desacople = 1.0

# --- Umbral de "estabilidad" para exploración cuando el mundo no cambia ---
# AHORA en unidades de CambioTotal normalizado (ver organelos/organo_cambio_total.py
# y config/escalas_sensores.py), no cm — desde que CambioTotal es multi-sensor.
UMBRAL_CAMBIO_TOTAL_ESTABLE = 0.15
LIMITE_SIN_CAMBIO = 5   # ciclos consecutivos sin cambio antes de forzar exploración máxima

# --- Escala del gradiente ---
# distancia objetivo de "crucero": el robot prefiere mantenerse cerca de esto.
DISTANCIA_OBJETIVO_CM = 40.0

# --- Cuánto pesa CambioTotal vs el error de distancia-objetivo al calcular
# D_actual (el "conflicto" que ve organo_deliberacion). Diseño original con
# GPT: "si CambioTotal es alto -> más peso a rotación/exploración". Valor
# inicial del diseñador, a calibrar con datos reales.
CAMBIO_TOTAL_ESCALA_CONFLICTO = 2.0  # cambio_total=2.0 -> D_actual=1.0 (máximo)
