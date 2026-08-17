#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelos.organo_deliberacion — LA MENTE: selección no determinista desde el pool.

PROCEDENCIA
-----------
Puerto limpio (no vendorizado literal) de la lógica de decisión de ANIMA:
    Célula_Madre/campo/VST_Celula_Madre_001.py
        clase ValenciaLocal            (líneas 195-231)
        clase MemoriaEpisodicaV180c    (líneas 236-279)
        clase MemoriaDeTrabajo.deliberar (líneas 284-331)
Se porta la LÓGICA tal cual (mismo algoritmo, mismas constantes de
comportamiento), pero desacoplada de motor/audio/hemisferios — aquí opera
sobre CUALQUIER pool de setpoints escalares (para CosmoRobot: el Volante,
ver config/pool_acciones.py), no solo sobre orientación.

QUÉ HACE (y por qué NO es random.choices)
------------------------------------------
La selección es un ARGMAX MODULADO POR CONFLICTO, no un sorteo uniforme:
    puntaje(opción) = valencia(opción)·peso_valencia
                     + bono_exploración (crece con el conflicto D_actual)
                     + bono_inercia (si opción ≈ setpoint actual)
                     − 100 si la memoria episódica marca esa opción como "trauma"
    decisión = argmax(puntaje)

Esto es más fiel a la teoría que un Random ponderado ingenuo: incorpora
MEMORIA (valencia aprendida por opción) y NEGACIÓN OPERATIVA (el veto de
trauma domina cualquier valencia positiva — la "R_op" de la genealogía LF).
El no-determinismo real viene de que la valencia y la memoria episódica
cambian con la experiencia del robot, no de un dado en cada ciclo.

Si en algún momento se quiere sorteo estocástico puro (softmax sobre los
puntajes en vez de argmax), es un cambio de una línea en `decidir()` — no
hace falta tocar el resto.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# --- constantes de comportamiento (mismos valores que CM001, declaradas aquí
#     para que el organelo sea autocontenido y no dependa del monolito) ---
TRAUMA_COSTO_MULTIPLIER = 5.0
TASA_APRENDIZAJE = 0.001


class ValenciaLocal:
    """Memoria de "cómo me fue" por cada opción del pool (bucket redondeado a 5).

    Aprende con la experiencia: refuerza opciones con error bajo, castiga
    error alto y costo, y castiga fuerte (incondicional) si la opción llevó
    a un evento marcado como trauma.
    """

    def __init__(self, trauma_costo_multiplier: float = TRAUMA_COSTO_MULTIPLIER) -> None:
        self.valencia: dict[float, float] = {}
        self.tasa_aprendizaje = TASA_APRENDIZAJE
        self.historial: dict[float, list] = {}
        self.trauma_costo_multiplier = trauma_costo_multiplier

    @staticmethod
    def _key(setpoint: float) -> float:
        return round(setpoint / 5) * 5 if setpoint != 0 else 0

    def actualizar(self, setpoint: float, error: float, costo_pagado: float, dt: float,
                    reward: float = 0.0, good_threshold: float = 5.0, trauma: bool = False) -> float:
        key = self._key(setpoint)
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []

        costo_efectivo = costo_pagado * self.trauma_costo_multiplier if trauma else costo_pagado

        if trauma:
            self.valencia[key] -= self.tasa_aprendizaje * dt * 150.0
        elif abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2

        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_efectivo * 0.1
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]

    def get_valencia(self, setpoint: float) -> float:
        return self.valencia.get(self._key(setpoint), 0.0)

    def reset(self) -> None:
        self.valencia = {}
        self.historial = {}


class MemoriaEpisodica:
    """Eventos recientes marcados con resultado (p.ej. 'trauma' tras un veto
    reactivo). `recuperar` busca eventos cercanos al setpoint consultado
    dentro de una ventana temporal — es lo que permite el veto episódico."""

    def __init__(self, capacidad_max: int = 50) -> None:
        self.memoria: list[dict] = []
        self.capacidad_max = capacidad_max
        self.tiempo_recuperacion = 0.0

    def registrar(self, t: float, setpoint: float, valencia: float, resultado: str,
                  contexto: str = "normal") -> None:
        self.memoria.append({"t": t, "setpoint": setpoint, "valencia": valencia,
                              "resultado": resultado, "contexto": contexto})
        if len(self.memoria) > self.capacidad_max:
            self.memoria.pop(0)

    def recuperar(self, setpoint_buscado: float, ventana_t: float = 15.0):
        self.tiempo_recuperacion = 0.0
        candidatos = [ev for ev in self.memoria if abs(ev["setpoint"] - setpoint_buscado) < ventana_t]
        if not candidatos:
            return None, 0.0
        self.tiempo_recuperacion = 0.15 + (len(candidatos) * 0.08)
        candidatos.sort(key=lambda x: abs(x["valencia"]), reverse=True)
        return candidatos[0], self.tiempo_recuperacion

    def reset(self) -> None:
        self.memoria = []
        self.tiempo_recuperacion = 0.0


@dataclass
class Deliberacion:
    """Resultado de un ciclo de deliberación (para log/depuración)."""
    decision: float
    puntajes: dict = field(default_factory=dict)
    tiempo_s: float = 0.0
    impacto_recuerdo: float = 0.0


class OrganoDeliberacion:
    """La mente: dado un pool de setpoints y el estado (conflicto, memoria),
    elige uno. Autocontenida — no sabe nada de motores ni sensores del NXT."""

    def __init__(self, steps_por_opcion: int = 125, dt_por_step: float = 0.008) -> None:
        self.valencia = ValenciaLocal()
        self.memoria_episodica = MemoriaEpisodica()
        self.steps_por_opcion = steps_por_opcion
        self.dt_por_step = dt_por_step
        self.ultima: Optional[Deliberacion] = None

    def decidir(self, pool: list[float], D_actual: float, current_sp: Optional[float] = None) -> Deliberacion:
        """Elige una opción del pool. `D_actual` = conflicto/entropía del
        estado (0 = tranquilo, sube = más exploración). `current_sp` = última
        opción ejecutada (da un pequeño bono de inercia, evita zigzag).

        DESVIACIÓN DOCUMENTADA respecto al original de ANIMA (CM001:292-331):
        se evalúa el pool en orden BARAJADO en vez de orden fijo. Motivo:
        `max(dict, key=...)` de Python resuelve empates devolviendo el PRIMER
        máximo encontrado. Con memoria vacía (arranque en frío) todos los
        puntajes parten en 0 -> empate total -> el argmax sería determinista
        y elegiría siempre la primera opción del pool, ciclo tras ciclo. Eso
        contradice la tesis central de la teoría ("cuando el mundo no
        cambia, el sistema introduce diferencia" — no repite por defecto).
        Barajar el orden de evaluación no toca la fórmula, los puntajes ni
        el veto de trauma: solo decide a favor de quién se resuelve un
        empate. Con valencias diferenciadas (tras aprender) esto no tiene
        efecto — el máximo deja de estar empatado."""
        puntajes: dict[float, float] = {}
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        tiempo_base_por_opcion = self.steps_por_opcion * self.dt_por_step
        tiempo_total_busqueda = 0.0
        max_impacto = 0.0

        pool_evaluado = list(pool)
        random.shuffle(pool_evaluado)
        for opcion in pool_evaluado:
            val = self.valencia.get_valencia(opcion)
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            puntaje = val * val_w + explor_bonus + current_bonus

            evento, t_rec = self.memoria_episodica.recuperar(opcion, ventana_t=15.0)
            tiempo_total_busqueda += t_rec
            if evento and evento["resultado"] == "trauma":
                puntaje -= 100.0
                max_impacto = max(max_impacto, 100.0)

            puntajes[opcion] = puntaje

        factor_conflicto = 1.0 + (D_actual * 3.5)
        tiempo_deliberacion = tiempo_base_por_opcion * len(pool) * factor_conflicto + tiempo_total_busqueda
        decision = max(puntajes, key=puntajes.get)

        resultado = Deliberacion(decision=decision, puntajes=puntajes,
                                  tiempo_s=tiempo_deliberacion, impacto_recuerdo=max_impacto)
        self.ultima = resultado
        return resultado

    def aprender(self, setpoint: float, error: float, costo_pagado: float, dt: float,
                 reward: float = 0.0, trauma: bool = False) -> float:
        """Actualiza la valencia de `setpoint` según cómo resultó la acción."""
        return self.valencia.actualizar(setpoint, error, costo_pagado, dt, reward=reward, trauma=trauma)

    def registrar_episodio(self, t: float, setpoint: float, resultado: str, contexto: str = "normal") -> None:
        """Registra un evento en la memoria episódica (p.ej. tras un veto
        reactivo inmediatamente después de elegir `setpoint`: resultado='trauma')."""
        val = self.valencia.get_valencia(setpoint)
        self.memoria_episodica.registrar(t, setpoint, val, resultado, contexto)
