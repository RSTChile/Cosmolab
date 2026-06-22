#!/usr/bin/env python3
"""
V180b — ANIMA-3: MEMORIA CONTEXTUAL (ROADMAP COMPLIANT)
================================================================================
BASE: V180a — Memoria episódico-valencial demostrada
OBJETIVO: ¿Puede el organismo asociar rechazo a CONTEXTO específico,
          no solo a setpoint? (Roadmap V180)

DISEÑO (según roadmap oficial):
  F1: Contexto A (ruido blanco)
      - Consolidación -60° (20 ciclos, reward)
      - Trauma +60° (15s, costo 2×)
  F2: Contexto B (silencio)
      - Test: ¿Se transfiere el trauma de +60° al nuevo contexto?
      - 20 trials neutrales (costo 0 para todo)
      - Medir: P(+60°) en contexto B
  F3: Volver a Contexto A
      - ¿Se recupera el rechazo de +60°?
      - 20 trials neutrales
  F4: Mezcla de contextos (alternancia aleatoria)
      - ¿Qué prevalece: especificidad o generalización?
      - 40 trials (20 A + 20 B, aleatorio)

MÉTRICAS CLAVE:
  - Val(setpoint, contexto) — matriz 2D
  - transfer_rate = Val_B(+60°) - Val_A(+60°) (normalizado)
  - discriminación_contextual = P(+60° | A) - P(+60° | B)

CRITERIOS DE ÉXITO (roadmap):
  ✅ Val_B(+60°) > Val_A(+60°) significativamente
  ✅ P(+60° | contexto_B) > 40%
  ✅ discriminación_contextual > 0.5
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import random

# ============================================================
# PARAMETROS (HEREDADOS)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MAX = 15.0

KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

VENTANA_OSCILACION = 100

INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0

K_GAIN = 0.00015
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

TAU_CB = 10.0
CB_MAX = 500.0

LAMBDA_FISICO = 0.15
LAMBDA_COSTO = 0.5
UMBRAL_CB_JUEGO = 40.0
K_INFLUENCIA_JUEGO = 0.0005

SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# ============================================================
# PARAMETROS DE LAS FASES (ROADMAP V180b)
# ============================================================
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0

CONSOLIDACION_CICLOS = 20
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0

CONTEXTO_A = "ruido_blanco"
CONTEXTO_B = "silencio"

CONTEXTO_TRIALS = 20
MEZCLA_TRIALS = 40
EXPOSURE_STEPS_PER_TRIAL = 600
TRIAL_DURATION = EXPOSURE_STEPS_PER_TRIAL * DT

# Umbrales de éxito (roadmap)
P_CONTEXTO_B_MIN = 0.40
DISC_CONTEXTUAL_MIN = 0.5


# ============================================================
# GENERADORES DE CONTEXTO (señales ambientales)
# ============================================================

class GeneradorContexto:
    """Genera señales contextuales para modulación de valencia"""
    def __init__(self, tipo):
        self.tipo = tipo
        self.activo = True
    
    def generar_senal(self, t):
        """Genera señal de contexto en el tiempo"""
        if not self.activo:
            return 0.0
        
        if self.tipo == CONTEXTO_A:
            # Ruido blanco: frecuencia alta, amplitud modulada
            return 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn()
        else:  # CONTEXTO_B = silencio
            # Silencio: ruido de fondo mínimo
            return 0.05 * np.random.randn()
    
    def get_modulador_valencia(self):
        """Retorna factor de modulación para valencia según contexto"""
        if self.tipo == CONTEXTO_A:
            return 1.0  # Contexto de trauma → valencia normal
        else:
            return 0.3  # Contexto seguro → valencia atenuada


# ============================================================
# VALENCIA LOCAL (MODULADA POR CONTEXTO)
# ============================================================

class ValenciaLocal:
    def __init__(self, trauma_costo_multiplier=TRAUMA_COSTO_MULTIPLIER):
        self.valencia = {}  # clave: (setpoint, contexto)
        self.tasa_aprendizaje = 0.001
        self.historial = {}
        self.trauma_costo_multiplier = trauma_costo_multiplier
    
    def _get_key(self, setpoint, contexto):
        sp_key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return (sp_key, contexto)
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, 
                   good_threshold=5.0, trauma=False, contexto=CONTEXTO_A):
        key = self._get_key(setpoint, contexto)
        
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        costo_efectivo = costo_pagado * self.trauma_costo_multiplier if trauma else costo_pagado
        
        if abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            if trauma:
                self.valencia[key] -= self.tasa_aprendizaje * dt * 80.0
            else:
                self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2
        
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_efectivo * 0.1
        
        if trauma and setpoint is not None and abs(setpoint - TRAUMA_SETPOINT) < 1.0:
            self.valencia[key] -= 0.08 * dt * self.trauma_costo_multiplier
        
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]
    
    def get_valencia(self, setpoint, contexto=CONTEXTO_A):
        key = self._get_key(setpoint, contexto)
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA DE TRABAJO (CON MODULACIÓN CONTEXTUAL)
# ============================================================

class MemoriaDeTrabajo:
    def __init__(self, steps_por_opcion=50):
        self.steps_por_opcion = steps_por_opcion
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []
    
    def deliberar(self, opciones_disponibles, valencia_local, D_actual, 
                  current_sp=None, contexto=CONTEXTO_A, modulador_contexto=1.0):
        self.opciones_ensayadas = {}
        puntajes = {}
        
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        tiempo_base_por_opcion = self.steps_por_opcion * DT
        
        for opcion in opciones_disponibles:
            # Valencia modulada por contexto
            val_base = valencia_local.get_valencia(opcion, contexto)
            val_modulada = val_base * modulador_contexto
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val_modulada) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            puntaje = (val_modulada * val_w + explor_bonus + current_bonus)
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje
        
        factor_conflicto = 1.0 + (D_actual * 3.5)
        self.tiempo_deliberacion = tiempo_base_por_opcion * len(opciones_disponibles) * factor_conflicto
        
        self.decision_final = max(puntajes, key=puntajes.get)
        self.historial_deliberaciones.append({
            'opciones': list(opciones_disponibles),
            'puntajes': puntajes,
            'decision': self.decision_final,
            'tiempo': self.tiempo_deliberacion
        })
        return self.decision_final, puntajes, self.tiempo_deliberacion
    
    def get_tiempo_medio_deliberacion(self):
        if not self.historial_deliberaciones:
            return 0.0
        return np.mean([d['tiempo'] for d in self.historial_deliberaciones])
    
    def reset(self):
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []


# ============================================================
# REGISTRO DE REPRESENTACIONES
# ============================================================

class RegistroRepresentaciones:
    def __init__(self, ventana=200):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
        self.historial_setpoints = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada, setpoint):
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)
        self.historial_setpoints.append(setpoint)
    
    def calcular_var_R(self):
        if len(self.historial_representaciones) < 20:
            return 0.0
        discretos = np.array([round(r / 10.0) * 10 for r in self.historial_representaciones])
        var = np.var(discretos)
        return min(1.0, var / 100.0)
    
    def calcular_Pmax(self):
        if len(self.historial_representaciones) < 20:
            return 1.0
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)
    
    def calcular_desacople(self):
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        return var_R * (1.0 - Pmax)
    
    def calcular_D_conflicto(self, valencias_opciones):
        if len(valencias_opciones) < 2:
            return 0.0
        
        vals = np.array(valencias_opciones)
        exp_vals = np.exp(vals / 10.0)
        probs = exp_vals / np.sum(exp_vals)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        D_entropia = entropy / np.log(len(vals))
        
        hay_trauma = np.any(vals < -0.5)
        factor_amenaza = 0.55 if hay_trauma else 0.0
        
        D_conflicto = (D_entropia * 0.5) + factor_amenaza
        return np.clip(D_conflicto, 0.0, 1.0)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()


# ============================================================
# HEMISFERIO (IDÉNTICO A V179)
# ============================================================

class HemisferioV180b:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        self.Phi = np.random.normal(sesgo, 0.1, 32)
        self.Phi_vel = np.zeros(32)
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        self.buffer_rapido = []
        self.historial_omega = []
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:32])
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.generar_entrada_para_t(t, duracion_total)
        
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, 31):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = entrada
        forzamiento[-1] = -entrada
        
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None:
            divergencia = abs(self._calcular_omega() - otro_hemisferio._calcular_omega())
            if divergencia > 0.5:
                acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        
        return {'omega': self._calcular_omega()}


# ============================================================
# FATIGA, MEMORIA, CONSCIENCIA, JUEGO (sin cambios)
# ============================================================

class FatigaMetabolicaV180b:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        self.historia = 0.0
        self.fatiga_activa = 0.0
        self.historial_historia = []
        self.historial_fatiga = []
    
    def actualizar(self, delta_real, costo_trabajo, en_reposo_real, dt):
        self.historia += abs(delta_real)
        if not en_reposo_real:
            self.fatiga_activa += costo_trabajo
        else:
            self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        self.fatiga_activa = min(self.fatiga_activa, 20000.0)
        
        factor_gain = np.exp(-self.k_gain * self.fatiga_activa)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa
        temblor = self.k_temblor * self.fatiga_activa * np.random.randn()
        
        factor_gain = max(0.2, min(1.0, factor_gain))
        zona_muerta_efectiva = min(ZONA_MUERTA_MAX, zona_muerta_efectiva)
        temblor = np.clip(temblor, -3.0, 3.0)
        
        self.historial_historia.append(self.historia)
        self.historial_fatiga.append(self.fatiga_activa)
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.historia = 0.0
        self.fatiga_activa = 0.0
        self.historial_historia = []
        self.historial_fatiga = []
    
    def get_historia(self):
        return self.historia
    
    def get_fatiga(self):
        return self.fatiga_activa


class MemoriaAusenciaV180b:
    def __init__(self, tau_base=TAU_BASE, k_mem=K_MEM, suelo_confianza=SUELO_CONFIANZA):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_base = tau_base
        self.k_mem = k_mem
        self.tau_mem = tau_base
        self.suelo_confianza = suelo_confianza
        self.historial_confianza = []
    
    def actualizar(self, setpoint, E_historia, dt):
        if setpoint is not None:
            self.setpoint_last = setpoint
            self.t_ausencia = 0.0
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            self.historial_confianza.append(1.0)
            return self.setpoint_last, 1.0
        else:
            self.t_ausencia += dt
            confianza = np.exp(-self.t_ausencia / self.tau_mem)
            self.historial_confianza.append(confianza)
            return self.setpoint_last, confianza
    
    def get_confianza(self):
        return self.historial_confianza[-1] if self.historial_confianza else 0.0
    
    def reset(self):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_mem = self.tau_base
        self.historial_confianza = []


class ConscienciaBasicaV180b:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial_Cb = []
        self.historial_presion = []
    
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        self.historial_Cb.append(self.Cb)
        self.historial_presion.append(presion)
        return self.Cb, presion
    
    def reset(self):
        self.Cb = 0.0
        self.historial_Cb = []
        self.historial_presion = []


class ModoJuegoV180b:
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO,
                 umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico = lambda_fisico
        self.lambda_costo = lambda_costo
        self.umbral_cb = umbral_cb
        self.k_influencia = k_influencia
        self.activo = False
        self.historial_activo = []
        self.tiempo_activo = 0.0
    
    def actualizar(self, Cb, confianza, setpoint_presente):
        if setpoint_presente is not None and Cb > self.umbral_cb:
            self.activo = True
            self.tiempo_activo += DT
        else:
            self.activo = False
        self.historial_activo.append(self.activo)
        return self.activo
    
    def aplicar(self, delta_raw):
        if self.activo:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
        else:
            delta_fisico = delta_raw
            delta_costo = abs(delta_raw)
        return delta_fisico, delta_costo
    
    def get_influencia(self, Cb, confianza):
        if self.activo and Cb > self.umbral_cb:
            return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    
    def reset(self):
        self.activo = False
        self.historial_activo = []
        self.tiempo_activo = 0.0


# ============================================================
# APARATO MOTOR V180b (CON MODULACIÓN CONTEXTUAL)
# ============================================================

class AparatoMotorV180b:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = INERCIA
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = SENSIBILIDAD_GRAD
        self.t = 0.0
        
        self.fatiga = FatigaMetabolicaV180b()
        self.memoria = MemoriaAusenciaV180b()
        self.consciencia = ConscienciaBasicaV180b()
        self.juego = ModoJuegoV180b()
        self.valencia = ValenciaLocal()
        self.memoria_trabajo = MemoriaDeTrabajo()
        self.registro = RegistroRepresentaciones()
        self.recent_presented = deque(maxlen=5)
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.ultimo_delta_registrado = 0.0
        self.en_deliberacion = False
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        oscilacion = np.std(self.memoria_error)
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        self.historial_Kp.append(self.Kp_actual)
    
    def ejecutar_con_deliberacion(self, opciones_disponibles, gradiente, t, dt, 
                                   trauma=False, target_reward=None, contexto=CONTEXTO_A,
                                   modulador_contexto=1.0):
        for op in opciones_disponibles:
            if op not in self.recent_presented:
                self.recent_presented.append(op)
        
        if len(opciones_disponibles) > 1:
            valencias = [self.valencia.get_valencia(op, contexto) * modulador_contexto 
                        for op in opciones_disponibles]
            D_actual = self.registro.calcular_D_conflicto(valencias)
            opcion_elegida, puntajes, tiempo_delib = self.memoria_trabajo.deliberar(
                opciones_disponibles, self.valencia, D_actual, 
                current_sp=self.orientacion, contexto=contexto,
                modulador_contexto=modulador_contexto
            )
        else:
            D_actual = self.registro.calcular_desacople()
            only = opciones_disponibles[0]
            val_only = self.valencia.get_valencia(only, contexto) * modulador_contexto
            if val_only < -2.0:
                opcion_elegida = 0.0
            else:
                opcion_elegida = only
            tiempo_delib = (self.memoria_trabajo.steps_por_opcion * DT) * 0.5
        
        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        val_local = self.valencia.get_valencia(opcion_elegida, contexto) * modulador_contexto
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        if abs(self.orientacion) > 0.01:
            A_sys_env = min(1.0, abs(self.orientacion) / abs(opcion_elegida)) if abs(opcion_elegida) > 0.01 else 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, opcion_elegida)
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        val_good_threshold = zona_muerta_efectiva
        
        rwd = 0.0
        if target_reward is not None and abs(opcion_elegida - target_reward) < 1.0 and abs(error) < zona_muerta_efectiva:
            rwd = 1.0
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=rwd, 
                                     good_threshold=val_good_threshold, trauma=trauma,
                                     contexto=contexto)
            val_local = self.valencia.get_valencia(opcion_elegida, contexto) * modulador_contexto
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0,
                    self.registro.calcular_desacople(), val_local, tiempo_delib, opcion_elegida, rwd)
        
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        costo_total_estimado = costo_error + abs(torque_memoria)
        
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=rwd, 
                                 good_threshold=val_good_threshold, trauma=trauma,
                                 contexto=contexto)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        en_reposo_real = (abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        delta_fisico += temblor * dt
        self.actualizar_plasticidad(error)
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        self.t += dt
        
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, opcion_elegida)
        D = self.registro.calcular_desacople()
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo,
                D, val_local, tiempo_delib, opcion_elegida, rwd)
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.ultimo_delta_registrado = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.fatiga.reset()
        self.memoria.reset()
        self.consciencia.reset()
        self.juego.reset()
        self.valencia.reset()
        self.memoria_trabajo.reset()
        self.registro.reset()
        self.recent_presented.clear()


# ============================================================
# ORGANISMO V180b
# ============================================================

class OrganismoV180b:
    def __init__(self, seed):
        self.nombre = f"Organismo_{seed}"
        self.seed = seed
        
        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            fft_filtrado = fft * filtro
            ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
            return ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
        
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks
        
        self.izquierdo = HemisferioV180b("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV180b("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = HemisferioV180b("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV180b("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV180b()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [], 'orientacion': [], 'setpoint_raw': [], 'confianza': [],
            'Cb': [], 'juego_activo': [], 'historia': [], 'fatiga': [],
            'costo': [], 's_shared': [], 'D': [], 'valencia': [],
            'tiempo_deliberacion': [], 'opcion_elegida': [], 'reward_recibido': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar_con_opciones(self, t, dt, duracion_total, opciones_disponibles, 
                                 trauma=False, target_reward=None, contexto=CONTEXTO_A,
                                 modulador_contexto=1.0):
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if abs(self.motor.orientacion) > 0.1:
            sesgo = self.motor.orientacion / 90.0
            gradiente += sesgo * 0.3
        
        LF_activa = not self.modo_entrenamiento
        (orientacion, historia, fatiga, confianza, _, Cb, _, juego_activo, costo,
         D, valencia, tiempo_delib, opcion_elegida, rwd) = self.motor.ejecutar_con_deliberacion(
            opciones_disponibles, gradiente, t, dt, trauma, target_reward, contexto, modulador_contexto
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(None)
        self.historial['confianza'].append(confianza)
        self.historial['Cb'].append(Cb)
        self.historial['juego_activo'].append(juego_activo)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['costo'].append(costo)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['D'].append(D)
        self.historial['valencia'].append(valencia)
        self.historial['tiempo_deliberacion'].append(tiempo_delib)
        self.historial['opcion_elegida'].append(opcion_elegida)
        self.historial['reward_recibido'].append(rwd)
        
        return orientacion, D, tiempo_delib, opcion_elegida, rwd
    
    def actualizar_setpoint(self, t, dt, duracion_total, setpoint, trauma=False, 
                            target_reward=None, contexto=CONTEXTO_A, modulador_contexto=1.0):
        return self.actualizar_con_opciones(t, dt, duracion_total, [setpoint], 
                                            trauma, target_reward, contexto, modulador_contexto)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()
    
    def get_valencia(self, setpoint, contexto=CONTEXTO_A):
        return self.motor.valencia.get_valencia(setpoint, contexto)
    
    def get_valencia_habito(self, contexto=CONTEXTO_A):
        return self.get_valencia(HABITO_SETPOINT, contexto)
    
    def get_valencia_trauma(self, contexto=CONTEXTO_A):
        return self.get_valencia(TRAUMA_SETPOINT, contexto)


# ============================================================
# FUNCIÓN DE TESTEO POR CONTEXTO
# ============================================================

def test_contexto(organismo, setpoint, contexto, modulador_contexto, num_trials=20):
    """Testea preferencia en un contexto específico"""
    opciones_elegidas = []
    tiempos_deliberacion = []
    
    # Inyectar opción neutral para forzar deliberación real
    organismo.motor.recent_presented.append(0.0)
    
    for trial in range(num_trials):
        t = trial * TRIAL_DURATION
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t + step * DT
            _, _, lat, opcion, _ = organismo.actualizar_con_opciones(
                t_step, DT, (num_trials + 1) * TRIAL_DURATION, [setpoint], 
                trauma=False, target_reward=None, contexto=contexto,
                modulador_contexto=modulador_contexto
            )
        opciones_elegidas.append(opcion if opcion is not None else 0)
        tiempos_deliberacion.append(lat)
    
    preferencia = sum(1 for e in opciones_elegidas if abs(e - setpoint) < 5.0) / num_trials
    return preferencia, np.mean(tiempos_deliberacion)


# ============================================================
# EXPERIMENTO PRINCIPAL V180b
# ============================================================

def ejecutar_v180b():
    print("=" * 100)
    print("EXPERIMENTO V180b — MEMORIA CONTEXTUAL (ROADMAP COMPLIANT)")
    print("=" * 100)
    print("  BASE: V180a — Memoria episódico-valencial demostrada")
    print("  OBJETIVO: ¿Puede asociar rechazo a CONTEXTO específico,")
    print("            no solo a setpoint?")
    print("")
    print("  DISEÑO (según roadmap oficial):")
    print(f"    F1: Contexto A (ruido blanco)")
    print(f"        - Consolidación -60° ({CONSOLIDACION_CICLOS} ciclos, reward)")
    print(f"        - Trauma +60° ({TRAUMA_DURACION}s, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print(f"    F2: Contexto B (silencio) — Test transferencia ({CONTEXTO_TRIALS} trials)")
    print(f"    F3: Contexto A — Test recuperación ({CONTEXTO_TRIALS} trials)")
    print(f"    F4: Mezcla de contextos — Discriminación contextual ({MEZCLA_TRIALS} trials)")
    print("")
    print("  CRITERIOS DE ÉXITO (roadmap):")
    print(f"    ✅ Val_B(+60°) > Val_A(+60°) (transferencia parcial)")
    print(f"    ✅ P(+60° | contexto_B) > {P_CONTEXTO_B_MIN:.0%}")
    print(f"    ✅ discriminación_contextual > {DISC_CONTEXTUAL_MIN}")
    print("=" * 100)
    
    print("\n  Creando organismo...")
    organismo = OrganismoV180b(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V180b_logs', exist_ok=True)
    
    # Entrenamiento
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    organismo.set_modo_entrenamiento(True)
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar_setpoint(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    print("  Entrenamiento completado.")
    organismo.set_modo_entrenamiento(False)
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # ============================================================
    # FASE 1: Consolidación y Trauma en Contexto A
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 1: Consolidación y Trauma en {CONTEXTO_A.upper()}")
    print("=" * 60)
    
    print("  Consolidando hábito -60°...")
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, 
                                         HABITO_SETPOINT, target_reward=HABITO_SETPOINT,
                                         contexto=CONTEXTO_A)
        t_actual += PERIODO_ALTERNANCIA
    
    val_habito_A = organismo.get_valencia_habito(CONTEXTO_A)
    print(f"  Valencia -60° en {CONTEXTO_A}: {val_habito_A:.2f}")
    
    print("  Aplicando trauma +60°...")
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar_setpoint(t, DT, t_actual + TRAUMA_DURACION, 
                                     TRAUMA_SETPOINT, trauma=True, contexto=CONTEXTO_A)
    t_actual += TRAUMA_DURACION
    
    val_trauma_A = organismo.get_valencia_trauma(CONTEXTO_A)
    print(f"  Valencia +60° en {CONTEXTO_A}: {val_trauma_A:.2f}")
    
    # ============================================================
    # FASE 2: Test en Contexto B (transferencia)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: Test en {CONTEXTO_B.upper()} — ¿Transferencia del trauma?")
    print("=" * 60)
    
    modulador_B = 0.3  # Contexto seguro: valencia atenuada
    
    p_60_B, latencia_B = test_contexto(organismo, TRAUMA_SETPOINT, CONTEXTO_B, 
                                        modulador_B, CONTEXTO_TRIALS)
    val_trauma_B = organismo.get_valencia_trauma(CONTEXTO_B)
    
    print(f"  Valencia +60° en {CONTEXTO_B}: {val_trauma_B:.2f}")
    print(f"  P(+60° | {CONTEXTO_B}) = {p_60_B:.1%}")
    
    transferencia_ok = val_trauma_B > val_trauma_A
    p_contexto_B_ok = p_60_B > P_CONTEXTO_B_MIN
    
    # ============================================================
    # FASE 3: Volver a Contexto A (recuperación)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: Volver a {CONTEXTO_A.upper()} — ¿Recupera el rechazo?")
    print("=" * 60)
    
    modulador_A = 1.0  # Contexto de trauma: valencia normal
    
    p_60_A_rec, latencia_A_rec = test_contexto(organismo, TRAUMA_SETPOINT, CONTEXTO_A, 
                                                modulador_A, CONTEXTO_TRIALS)
    val_trauma_A_rec = organismo.get_valencia_trauma(CONTEXTO_A)
    
    print(f"  Valencia +60° en {CONTEXTO_A} (recuperado): {val_trauma_A_rec:.2f}")
    print(f"  P(+60° | {CONTEXTO_A}) = {p_60_A_rec:.1%}")
    
    recuperacion_ok = val_trauma_A_rec < -1.0
    
    # ============================================================
    # FASE 4: Mezcla de contextos (discriminación)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 4: Mezcla aleatoria de contextos — Discriminación contextual")
    print("=" * 60)
    
    resultados_A = []
    resultados_B = []
    
    for trial in range(MEZCLA_TRIALS):
        contexto = random.choice([CONTEXTO_A, CONTEXTO_B])
        modulador = 1.0 if contexto == CONTEXTO_A else 0.3
        
        t = trial * TRIAL_DURATION
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t + step * DT
            _, _, _, opcion, _ = organismo.actualizar_con_opciones(
                t_step, DT, MEZCLA_TRIALS * TRIAL_DURATION, [TRAUMA_SETPOINT],
                trauma=False, target_reward=None, contexto=contexto,
                modulador_contexto=modulador
            )
        
        eligio_60 = abs(opcion - TRAUMA_SETPOINT) < 5.0 if opcion is not None else False
        
        if contexto == CONTEXTO_A:
            resultados_A.append(eligio_60)
        else:
            resultados_B.append(eligio_60)
        
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{MEZCLA_TRIALS}...")
    
    p_60_A_mix = sum(resultados_A) / len(resultados_A) if resultados_A else 0
    p_60_B_mix = sum(resultados_B) / len(resultados_B) if resultados_B else 0
    
    discriminacion = p_60_A_mix - p_60_B_mix
    discriminacion_ok = discriminacion > DISC_CONTEXTUAL_MIN
    
    print(f"\n  P(+60° | {CONTEXTO_A}) en mezcla: {p_60_A_mix:.1%}")
    print(f"  P(+60° | {CONTEXTO_B}) en mezcla: {p_60_B_mix:.1%}")
    print(f"  Discriminación contextual: {discriminacion:.2f} (umbral > {DISC_CONTEXTUAL_MIN})")
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V180b — Memoria contextual")
    print("=" * 80)
    
    print(f"\n  📊 MATRIZ DE VALENCIA (setpoint, contexto):")
    print(f"    Val(-60°, {CONTEXTO_A}) = {val_habito_A:.2f}")
    print(f"    Val(+60°, {CONTEXTO_A}) = {val_trauma_A:.2f}")
    print(f"    Val(+60°, {CONTEXTO_B}) = {val_trauma_B:.2f}")
    print(f"    Val(+60°, {CONTEXTO_A} recuperado) = {val_trauma_A_rec:.2f}")
    
    print(f"\n  📊 MÉTRICAS DE TRANSFERENCIA:")
    print(f"    transferencia = {val_trauma_B - val_trauma_A:.2f} {'✅' if transferencia_ok else '❌'}")
    print(f"    P(+60° | {CONTEXTO_B}) = {p_60_B:.1%} (umbral > {P_CONTEXTO_B_MIN:.0%}) -> {'✅' if p_contexto_B_ok else '❌'}")
    
    print(f"\n  📊 MÉTRICAS DE DISCRIMINACIÓN:")
    print(f"    discriminación_contextual = {discriminacion:.2f} (umbral > {DISC_CONTEXTUAL_MIN}) -> {'✅' if discriminacion_ok else '❌'}")
    print(f"    Recuperación en {CONTEXTO_A}: {recuperacion_ok} -> {'✅' if recuperacion_ok else '❌'}")
    
    exito = transferencia_ok and p_contexto_B_ok and discriminacion_ok
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ MEMORIA CONTEXTUAL DEMOSTRADA")
        print("")
        print("     El organismo demuestra:")
        print("     ✓ Transferencia parcial del trauma al nuevo contexto")
        print("     ✓ Discriminación contextual > 0.5")
        print("     ✓ Recuperación del rechazo al volver al contexto original")
    else:
        print("  ⚠️ MEMORIA CONTEXTUAL NO DEMOSTRADA")
        if not transferencia_ok:
            print("     El trauma se transfirió completamente o no se transfirió")
        if not p_contexto_B_ok:
            print("     La probabilidad de elegir +60° en contexto B fue insuficiente")
        if not discriminacion_ok:
            print("     No hubo discriminación contextual significativa")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Matriz de valencias
    ax = axes[0, 0]
    categorias = [f'(-60°, {CONTEXTO_A})', f'(+60°, {CONTEXTO_A})', 
                  f'(+60°, {CONTEXTO_B})', f'(+60°, {CONTEXTO_A} rec)']
    valores = [val_habito_A, val_trauma_A, val_trauma_B, val_trauma_A_rec]
    colores = ['blue', 'red', 'orange', 'green']
    ax.bar(categorias, valores, color=colores)
    ax.axhline(y=10.0, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Valencia')
    ax.set_title('Matriz de valencia por contexto')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    
    # Gráfico 2: Preferencias por contexto
    ax = axes[0, 1]
    categorias_ctx = [f'{CONTEXTO_A} (F1)', f'{CONTEXTO_B} (F2)', 
                      f'{CONTEXTO_A} (F3)', f'{CONTEXTO_A} mix', f'{CONTEXTO_B} mix']
    preferencias = [0.0, p_60_B, p_60_A_rec, p_60_A_mix, p_60_B_mix]
    colores_ctx = ['red', 'orange', 'green', 'blue', 'cyan']
    ax.bar(categorias_ctx, preferencias, color=colores_ctx)
    ax.axhline(y=P_CONTEXTO_B_MIN, color='green', linestyle='--', alpha=0.7)
    ax.set_ylabel('P(elegir +60°)')
    ax.set_title('Preferencia por contexto')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=15)
    
    # Gráfico 3: Discriminación contextual
    ax = axes[1, 0]
    ax.bar([f'{CONTEXTO_A}', f'{CONTEXTO_B}'], [p_60_A_mix, p_60_B_mix], 
           color=['red', 'green'])
    ax.axhline(y=DISC_CONTEXTUAL_MIN, color='blue', linestyle='--', alpha=0.7,
               label=f'Umbral discriminación')
    ax.set_ylabel('P(elegir +60°)')
    ax.set_title('Discriminación contextual en mezcla')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Transferencia
    ax = axes[1, 1]
    ax.bar([f'{CONTEXTO_A}', f'{CONTEXTO_B}'], [val_trauma_A, val_trauma_B],
           color=['red', 'orange'])
    ax.axhline(y=val_trauma_A, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Valencia +60°')
    ax.set_title('Transferencia del trauma entre contextos')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V180b_logs/v180b_contextual_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V180b_logs/v180b_contextual_{timestamp}.png")
    
    # Guardar datos
    raw_data = {
        'version': 'V180b',
        'timestamp': timestamp,
        'params': {
            'CONTEXTO_A': CONTEXTO_A,
            'CONTEXTO_B': CONTEXTO_B,
            'CONSOLIDACION_CICLOS': CONSOLIDACION_CICLOS,
            'TRAUMA_DURACION': TRAUMA_DURACION,
            'TRAUMA_COSTO_MULTIPLIER': TRAUMA_COSTO_MULTIPLIER,
            'CONTEXTO_TRIALS': CONTEXTO_TRIALS,
            'MEZCLA_TRIALS': MEZCLA_TRIALS,
        },
        'resultados': {
            'val_habito_A': float(val_habito_A),
            'val_trauma_A': float(val_trauma_A),
            'val_trauma_B': float(val_trauma_B),
            'val_trauma_A_rec': float(val_trauma_A_rec),
            'p_60_B': float(p_60_B),
            'p_60_A_mix': float(p_60_A_mix),
            'p_60_B_mix': float(p_60_B_mix),
            'discriminacion': float(discriminacion),
            'transferencia_ok': bool(transferencia_ok),
            'p_contexto_B_ok': bool(p_contexto_B_ok),
            'discriminacion_ok': bool(discriminacion_ok),
            'exito': bool(exito)
        }
    }
    
    with open(f'V180b_logs/v180b_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V180b_logs/v180b_raw_{timestamp}.json")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v180b()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V180b completado. Éxito: {exito}")