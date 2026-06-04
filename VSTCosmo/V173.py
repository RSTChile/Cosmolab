#!/usr/bin/env python3
"""
V173 — ANIMA-2: VALENCIA ASOCIATIVA LOCAL (Precursor del "No")
================================================================================
Objetivo: Demostrar que ANIMA-2 puede asignar valores diferenciales locales
          a representaciones específicas, separando valencia (Val) de Cb.

Diseño según consenso (Qwen, Grok, Meta AI, GPT, Alexis):
  - Valencia es local por representación, no inferida de Cb global
  - Consolidación previa del hábito (Cb(-60°) < 100)
  - Trauma específico suave (costo 2x solo en +60°)
  - Evaluación forzada (no permitir huida del setpoint)

CRITERIOS DE ÉXITO:
  1. Valencia diferencial: Val(+60°) < Val(-60°) y Val(+60°) < Val(0°)
  2. Hábito preservado: Cb(-60°) < 150
  3. Desacople sostenido: D > 0.1 por ≥ 3s
  4. No abstención: tasa acción en opciones seguras > 0.3
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque
import random

# ============================================================
# PARAMETROS BASE
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

K_GAIN = 0.0003
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

TAU_CB = 10.0
CB_MAX = 500.0

LAMBDA_FISICO = 0.1
LAMBDA_COSTO = 1.0
UMBRAL_CB_JUEGO = 35.0
K_INFLUENCIA_JUEGO = 0.00035

RUIDO_SETPOINT_AMP = 5.0
RUIDO_SETPOINT_PERIODO = 10.0

# ============================================================
# CONSTANTES GLOBALES
# ============================================================
PERIODO_ALTERNANCIA = 80.0
SEMILLA_BASE = 44

# ============================================================
# PARAMETROS DE VALENCIA ASOCIATIVA LOCAL
# ============================================================
SETPOINTS_POSIBLES = [-60.0, -30.0, 0.0, 30.0, 60.0]

# FASE 0: Consolidación del hábito (reforzada)
CONSOLIDACION_CICLOS = 100               # 100 ciclos para consolidar
CONSOLIDACION_REWARD = 10.0              # Reward positivo si error < 5°
CONSOLIDACION_OBJETIVO_CB = 100.0        # Cb(-60°) debe bajar de 100
CONSOLIDACION_CICLOS_ESTABLES = 20       # Mantener 20 ciclos estable

# FASE 1: Exploración con baseline plano
EXPLORACION_DURACION = 5 * PERIODO_ALTERNANCIA  # 5 ciclos
EXPLORACION_COSTO_FACTOR = 0.0           # Costo 0 para opciones seguras
EXPLORACION_COSTO_TRAUMA = 0.5           # Costo reducido para +60° en exploración

# FASE 2: Trauma específico suave
TRAUMA_SETPOINT = 60.0
TRAUMA_COSTO_MULTIPLIER = 2.0            # Reducido de 3x a 2x
TRAUMA_DURACION = 15.0                   # 15 segundos
TRAUMA_ANCLAJE_FRECUENCIA = 3.0          # Cada 3s, anclaje seguro
TRAUMA_REWARD_ANCLAJE = 5.0              # Reward en anclaje

# FASE 3: Evaluación forzada (sin permitir huida)
TEST_DURACION_POR_SETPOINT = 8.0         # 8 segundos fijos por setpoint
TEST_UMBRAL_CB_VALENCIA = 50.0           # Diferencia mínima de valencia

# Umbrales de éxito
UMBRAL_VALENCIA_DIFERENCIAL = 50.0       # Val(+60°) < Val(-60°) - 50
UMBRAL_HABITO_SEGURO = 150.0             # Cb(-60°) < 150
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 3.0
UMBRAL_TASA_ACCION = 0.3                 # 30% acciones en opciones seguras

# ============================================================
# PARAMETROS RITUAL
# ============================================================
RITUAL_TAU = 300.0
RITUAL_REPETICION_MIN = 2
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.35
RITUAL_UMBRAL_CB = 28.0
RITUAL_SALIDA_SUAVE = 0.98
RITUAL_PERSISTENCIA_MIN = 5


# ============================================================
# VALENCIA ASOCIATIVA LOCAL (NUEVO)
# ============================================================

class ValenciaAsociativaLocal:
    """
    Implementa Val(Ri) = acoplamiento_exitoso - costo_historico - Cb_asociada
    La valencia es LOCAL por representación, no inferida de Cb global.
    """
    
    def __init__(self, setpoints_posibles, tasa_decaimiento=0.99):
        # Valencia por setpoint (valor asociativo)
        self.valencia = {sp: 0.0 for sp in setpoints_posibles}
        # Historial de Cb asociada por setpoint
        self.cb_asociada = {sp: 0.0 for sp in setpoints_posibles}
        # Éxitos de acoplamiento por setpoint
        self.exitos_acoplamiento = {sp: 0 for sp in setpoints_posibles}
        self.costos_historicos = {sp: 0.0 for sp in setpoints_posibles}
        
        self.tasa_decaimiento = tasa_decaimiento
        self.historial_valencia = {sp: [] for sp in setpoints_posibles}
        
        self.ultimo_setpoint = None
        self.ultimo_error = None
    
    def actualizar(self, setpoint_actual, error, Cb_actual, costo_pagado, dt,
                   reward=0.0, es_trauma=False):
        """
        Actualiza la valencia asociada al setpoint actual.
        """
        # Decaimiento global (olvido gradual)
        for sp in self.valencia:
            self.valencia[sp] *= self.tasa_decaimiento
        
        # Actualizar métricas para el setpoint actual
        key = round(setpoint_actual / 10) * 10 if setpoint_actual != 0 else 0
        
        # Registrar Cb asociada
        self.cb_asociada[key] = self.cb_asociada[key] * 0.99 + Cb_actual * 0.01
        
        # Registrar costo histórico
        if es_trauma:
            self.costos_historicos[key] += costo_pagado * dt
        
        # Si hubo acoplamiento exitoso (error bajo), aumentar valencia
        if abs(error) < 5.0:
            self.exitos_acoplamiento[key] += 1
            # Reward positivo refuerza valencia
            self.valencia[key] += (reward + 1.0) * dt
        else:
            # Error alto reduce valencia
            self.valencia[key] -= abs(error) * dt * 0.01
        
        # Si hay trauma, reducir valencia específicamente
        if es_trauma:
            self.valencia[key] -= costo_pagado * dt * 0.1
        
        # Limitar valencia
        self.valencia[key] = max(-500.0, min(500.0, self.valencia[key]))
        
        # Registrar historial
        self.historial_valencia[key].append(self.valencia[key])
        
        self.ultimo_setpoint = setpoint_actual
        self.ultimo_error = error
        
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def get_cb_asociada(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        return self.cb_asociada.get(key, 0.0)
    
    def calcular_diferencial(self, setpoint_trauma, setpoint_habito, setpoint_neutral=0.0):
        """
        Calcula si hay valencia diferencial significativa.
        """
        val_trauma = self.get_valencia(setpoint_trauma)
        val_habito = self.get_valencia(setpoint_habito)
        val_neutral = self.get_valencia(setpoint_neutral)
        
        # La valencia del trauma debe ser menor que las otras
        trauma_peor_que_habito = val_trauma < val_habito - UMBRAL_VALENCIA_DIFERENCIAL
        trauma_peor_que_neutral = val_trauma < val_neutral - UMBRAL_VALENCIA_DIFERENCIAL
        
        return trauma_peor_que_habito and trauma_peor_que_neutral
    
    def reset(self):
        for sp in self.valencia:
            self.valencia[sp] = 0.0
            self.cb_asociada[sp] = 0.0
            self.exitos_acoplamiento[sp] = 0
            self.costos_historicos[sp] = 0.0
            self.historial_valencia[sp] = []
        self.ultimo_setpoint = None
        self.ultimo_error = None


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV173:
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
# FATIGA METABOLICA
# ============================================================

class FatigaMetabolicaV173:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.historia = 0.0
        self.fatiga_activa = 0.0
    
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
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.historia = 0.0
        self.fatiga_activa = 0.0
    
    def get_historia(self):
        return self.historia
    
    def get_fatiga(self):
        return self.fatiga_activa


# ============================================================
# MEMORIA DE AUSENCIA
# ============================================================

class MemoriaAusenciaV173:
    def __init__(self, tau_base=TAU_BASE, k_mem=K_MEM, suelo_confianza=SUELO_CONFIANZA):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_base = tau_base
        self.k_mem = k_mem
        self.tau_mem = tau_base
        self.suelo_confianza = suelo_confianza
    
    def actualizar(self, setpoint, E_historia, dt):
        if setpoint is not None:
            self.setpoint_last = setpoint
            self.t_ausencia = 0.0
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            return self.setpoint_last, 1.0
        else:
            self.t_ausencia += dt
            confianza = np.exp(-self.t_ausencia / self.tau_mem)
            return self.setpoint_last, confianza
    
    def get_confianza(self):
        return 1.0
    
    def get_tau_mem(self):
        return self.tau_mem
    
    def reset(self):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_mem = self.tau_base


# ============================================================
# CONSCIENCIA BÁSICA (Cb)
# ============================================================

class ConscienciaBasicaV173:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial_presion = []
    
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        self.historial_presion.append(presion)
        return self.Cb
    
    def reset(self):
        self.Cb = 0.0
        self.historial_presion = []


# ============================================================
# MODO JUEGO
# ============================================================

class ModoJuegoV173:
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO,
                 umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico = lambda_fisico
        self.lambda_costo = lambda_costo
        self.umbral_cb = umbral_cb
        self.k_influencia = k_influencia
        self.activo = False
        self.tiempo_activo = 0.0
    
    def actualizar(self, Cb, confianza, setpoint_presente):
        if setpoint_presente is not None and Cb > self.umbral_cb:
            self.activo = True
            self.tiempo_activo += DT
        else:
            self.activo = False
        return self.activo
    
    def aplicar(self, delta_raw, trauma_mode=False, juego_mode=False, reward_mode=False):
        if self.activo or juego_mode:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
            if trauma_mode:
                delta_costo *= TRAUMA_COSTO_MULTIPLIER
            if juego_mode:
                delta_costo *= EXPLORACION_COSTO_FACTOR
            if reward_mode:
                # Reward: costo negativo (beneficio)
                delta_costo = -abs(delta_raw) * CONSOLIDACION_REWARD
        else:
            delta_fisico = delta_raw
            delta_costo = abs(delta_raw)
            if trauma_mode:
                delta_costo *= TRAUMA_COSTO_MULTIPLIER
        
        return delta_fisico, delta_costo
    
    def get_influencia(self, Cb, confianza):
        if self.activo and Cb > self.umbral_cb:
            return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    
    def get_tiempo_activo(self):
        return self.tiempo_activo
    
    def reset(self):
        self.activo = False
        self.tiempo_activo = 0.0


# ============================================================
# RITUAL
# ============================================================

class RitualV173:
    def __init__(self, tau=RITUAL_TAU, repeticion_min=RITUAL_REPETICION_MIN,
                 ritual_gain=RITUAL_GAIN, patron_temporal=RITUAL_PATRON_TEMPORAL,
                 tolerancia=RITUAL_TOLERANCIA, umbral_activacion=RITUAL_UMBRAL_ACTIVACION,
                 umbral_cb=RITUAL_UMBRAL_CB, salida_suave=RITUAL_SALIDA_SUAVE,
                 persistencia_min=RITUAL_PERSISTENCIA_MIN):
        self.tau = tau
        self.repeticion_min = repeticion_min
        self.ritual_gain = ritual_gain
        self.patron_temporal = patron_temporal
        self.tolerancia = tolerancia
        self.umbral_activacion = umbral_activacion
        self.umbral_cb = umbral_cb
        self.salida_suave = salida_suave
        self.persistencia_min = persistencia_min
        
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.tiempo_activo = 0.0
        
        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0
    
    def detectar_cruce_por_cero(self, orientacion):
        cruce = (self.ultima_orientacion < 0 and orientacion >= 0) or \
                (self.ultima_orientacion > 0 and orientacion <= 0)
        self.ultima_orientacion = orientacion
        if cruce:
            self.cruces += 1
            self.ciclos_sin_cruce = 0
            return True
        else:
            self.ciclos_sin_cruce += 1
            return False
    
    def actualizar(self, orientacion, Cb, tiempo_actual, dt):
        es_cruce = self.detectar_cruce_por_cero(orientacion)
        
        if es_cruce and Cb > self.umbral_cb:
            es_patron = False
            for t_prev in self.patron_buffer:
                dt_desde_prev = tiempo_actual - t_prev
                if abs(dt_desde_prev - self.patron_temporal) <= (self.patron_temporal * self.tolerancia):
                    es_patron = True
                    break
            
            if es_patron:
                self.repeticiones_consecutivas += 1
                if self.repeticiones_consecutivas >= self.repeticion_min:
                    incremento = Cb * self.repeticiones_consecutivas / 100.0
                    self.activation += incremento * dt
            else:
                self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 1)
            
            self.patron_buffer.append(tiempo_actual)
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.1)
        
        self.activation *= np.exp(-dt / self.tau)
        self.activation = max(0.0, min(2.0, self.activation))
        
        if self.activation > self.umbral_activacion:
            self.active = True
        elif self.active:
            if self.ciclos_sin_cruce > self.persistencia_min:
                self.active = False
            else:
                self.active = self.active * self.salida_suave
        
        if self.active:
            self.tiempo_activo += dt
        
        return self.active
    
    def modular_correccion(self, delta_raw, correccion_ritual):
        if self.active:
            return delta_raw * (1 - self.activation * 0.3) + correccion_ritual * self.activation
        return delta_raw
    
    def reset(self):
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.tiempo_activo = 0.0
        self.ultima_orientacion = 0.0
        self.cruces = 0
        self.ciclos_sin_cruce = 0


# ============================================================
# REGISTRO DE REPRESENTACIONES PARA DESACOPLE
# ============================================================

class RegistroRepresentaciones:
    def __init__(self, ventana=100, ruido_sigma=5.0):
        self.ventana = ventana
        self.ruido_sigma = ruido_sigma
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada):
        if self.ruido_sigma > 0:
            representacion_ruidosa = representacion + np.random.normal(0, self.ruido_sigma)
        else:
            representacion_ruidosa = representacion
        
        self.historial_representaciones.append(representacion_ruidosa)
        self.historial_acciones.append(accion_ejecutada)
    
    def calcular_var_R(self):
        if len(self.historial_representaciones) < 10:
            return 0.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        _, counts = np.unique(discretos, return_counts=True)
        probs = counts / len(discretos)
        var = -np.sum(probs * np.log(probs + 1e-10))
        return var
    
    def calcular_Pmax(self):
        if len(self.historial_representaciones) < 10:
            return 1.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)
    
    def calcular_desacople(self):
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        var_norm = min(1.0, var_R / 3.0)
        return var_norm * (1.0 - Pmax)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()


# ============================================================
# APARATO MOTOR V173
# ============================================================

class AparatoMotorV173:
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
        
        self.fatiga = FatigaMetabolicaV173()
        self.memoria = MemoriaAusenciaV173()
        self.consciencia = ConscienciaBasicaV173()
        self.juego = ModoJuegoV173()
        self.ritual = RitualV173()
        
        # NUEVO: Valencia Asociativa Local
        self.valencia = ValenciaAsociativaLocal(SETPOINTS_POSIBLES)
        
        self.registro = RegistroRepresentaciones()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT,
               modo_juego=False, trauma=False, reward=False):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, False, 0.0, 0, 0.0, 0.0, 0.0)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, False, 0.0, 0, 0.0, 0.0, 0.0)
        
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), dt)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb = self.consciencia.actualizar(e_R, A_sys_env, dt)
        
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        else:
            juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        accion_ejecutada = abs(self.ultimo_delta) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada)
        
        D = self.registro.calcular_desacople()
        
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, setpoint_objetivo, juego_activo,
                    self.juego.get_tiempo_activo(), ritual_activo, self.ritual.activation,
                    self.ritual.cruces, D, 0.0, 0.0)
        
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        
        if juego_activo and not ritual_activo:
            influencia_juego = self.juego.get_influencia(Cb, confianza)
            if influencia_juego != 0:
                delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        correccion_ritual = 0.0
        if ritual_activo and self.ritual.patron_buffer:
            correccion_ritual = 5.0 * self.ritual.ritual_gain
        
        delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        
        # Calcular costo antes de aplicar (para valencia)
        costo_estimado = abs(delta)
        
        delta_fisico, delta_costo = self.juego.aplicar(delta, trauma, modo_juego, reward)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        # Actualizar Valencia Asociativa Local
        self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                  error, Cb, costo_estimado, dt,
                                  reward=CONSOLIDACION_REWARD if reward else 0.0,
                                  es_trauma=trauma)
        
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        
        delta_fisico += temblor * dt
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += dt
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, setpoint_objetivo, juego_activo,
                self.juego.get_tiempo_activo(), ritual_activo, self.ritual.activation,
                self.ritual.cruces, D, 0.0, 0.0)
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.fatiga.reset()
        self.memoria.reset()
        self.consciencia.reset()
        self.juego.reset()
        self.ritual.reset()
        self.registro.reset()
        self.valencia.reset()


# ============================================================
# SISTEMA V173
# ============================================================

class SistemaV173:
    def __init__(self, nombre, seed=SEMILLA_BASE):
        self.nombre = nombre

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

        self.izquierdo = HemisferioV173("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV173("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV173("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV173("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV173()
        self.modo_entrenamiento = True
        self.fase_actual = 0

    def actualizar(self, t, dt, duracion_total, setpoint_real, 
                   modo_juego=False, trauma=False, reward=False):
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if setpoint_real is not None:
            sesgo = setpoint_real / 90.0
            gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        
        (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces,
         D, _, _) = self.motor.actuar(
            gradiente, LF_activa, True, t, setpoint_real, dt, modo_juego, trauma, reward
        )
        
        return (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()
    
    def get_valencia(self, setpoint):
        return self.motor.valencia.get_valencia(setpoint)
    
    def get_cb_asociada(self, setpoint):
        return self.motor.valencia.get_cb_asociada(setpoint)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def setpoint_normal(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def setpoint_test_fijo(t, setpoint_fijo):
    """Setpoint fijo para evaluación forzada"""
    return setpoint_fijo


def generar_setpoint_con_ruido(t, setpoint_func, **kwargs):
    setpoint_base = setpoint_func(t, **kwargs)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


# ============================================================
# EXPERIMENTO V173 — VALENCIA ASOCIATIVA LOCAL
# ============================================================

def ejecutar_v173():
    print("=" * 100)
    print("EXPERIMENTO V173 — ANIMA-2: VALENCIA ASOCIATIVA LOCAL")
    print("=" * 100)
    print("  Objetivo: Demostrar valencia diferencial local por representación")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    1. Valencia diferencial: Val(+60°) < Val(-60°) y < Val(0°)")
    print(f"    2. Hábito preservado: Cb(-60°) < {UMBRAL_HABITO_SEGURO}")
    print(f"    3. Desacople sostenido: D > {UMBRAL_DESACOPLE} por ≥ {TIEMPO_MINIMO_DESACOPLE}s")
    print(f"    4. No abstención: tasa acción segura > {UMBRAL_TASA_ACCION * 100:.0f}%")
    print("=" * 100)

    organismo = SistemaV173("V173", seed=SEMILLA_BASE)

    print("\n" + "=" * 80)
    print("FASE 0: Consolidación del hábito (reforzada)")
    print("=" * 80)
    print(f"  Duración: {CONSOLIDACION_CICLOS} ciclos")
    print(f"  Reward: +{CONSOLIDACION_REWARD} si error < 5°")
    print(f"  Objetivo: Cb(-60°) < {CONSOLIDACION_OBJETIVO_CB}")
    
    organismo.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    organismo.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # Consolidación con reward
    cb_habito_vals = []
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = -60.0  # Fijar en -60° para consolidación
            (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
             juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
                t, DT, t_actual + PERIODO_ALTERNANCIA, setpoint, reward=True)
            
            if i % 500 == 0:
                cb_habito_vals.append(Cb)
        
        t_actual += PERIODO_ALTERNANCIA
        if (ciclo + 1) % 20 == 0:
            media_cb = np.mean(cb_habito_vals[-100:]) if cb_habito_vals else 0
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, Cb(-60°) media: {media_cb:.1f}")
    
    cb_habito_final = np.mean(cb_habito_vals[-500:]) if cb_habito_vals else 0
    print(f"  Consolidación completada. Cb(-60°) final: {cb_habito_final:.1f}")
    
    if cb_habito_final >= CONSOLIDACION_OBJETIVO_CB:
        print(f"  ⚠️ ADVERTENCIA: Cb(-60°) no bajó del umbral ({CONSOLIDACION_OBJETIVO_CB})")
    
    print("\n" + "=" * 80)
    print("FASE 1: Exploración con baseline plano")
    print("=" * 80)
    print(f"  Duración: {EXPLORACION_DURACION/PERIODO_ALTERNANCIA:.0f} ciclos")
    print(f"  Costo opciones seguras: 0x")
    print(f"  Costo +60°: {EXPLORACION_COSTO_TRAUMA}x")
    
    exploracion_datos = {'setpoints': [], 'Cb': [], 'valencia': []}
    
    for i in range(int(EXPLORACION_DURACION / DT)):
        t = t_actual + i * DT
        # Setpoint aleatorio, pero +60° tiene costo reducido
        setpoint = random.choice(SETPOINTS_POSIBLES)
        es_trauma_soft = (abs(setpoint - TRAUMA_SETPOINT) < 5.0)
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + EXPLORACION_DURACION, setpoint, 
            modo_juego=True, trauma=es_trauma_soft)
        
        exploracion_datos['setpoints'].append(setpoint)
        exploracion_datos['Cb'].append(Cb)
        exploracion_datos['valencia'].append(organismo.get_valencia(setpoint))
    
    t_actual += EXPLORACION_DURACION
    
    print(f"  Cb media durante exploración: {np.mean(exploracion_datos['Cb']):.1f}")
    
    print("\n" + "=" * 80)
    print("FASE 2: Trauma específico suave")
    print("=" * 80)
    print(f"  Setpoint forzado a +{TRAUMA_SETPOINT:.0f}° por {TRAUMA_DURACION}s")
    print(f"  Costo multiplicado por {TRAUMA_COSTO_MULTIPLIER}x")
    print(f"  Anclaje: cada {TRAUMA_ANCLAJE_FRECUENCIA}s, setpoint seguro -60° con reward")
    
    trauma_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 
                    'valencia_trauma': [], 'valencia_habito': []}
    anclaje_timer = 0.0
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        
        if anclaje_timer >= TRAUMA_ANCLAJE_FRECUENCIA:
            setpoint = -60.0
            trauma = False
            reward = True
            anclaje_timer = 0.0
        else:
            setpoint = TRAUMA_SETPOINT
            trauma = True
            reward = False
            anclaje_timer += DT
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + TRAUMA_DURACION, setpoint, trauma=trauma, reward=reward)
        
        trauma_datos['t'].append(t)
        trauma_datos['orient'].append(orient)
        trauma_datos['setpoint'].append(setpoint)
        trauma_datos['Cb'].append(Cb)
        trauma_datos['valencia_trauma'].append(organismo.get_valencia(TRAUMA_SETPOINT))
        trauma_datos['valencia_habito'].append(organismo.get_valencia(-60.0))
    
    t_actual += TRAUMA_DURACION
    
    valencia_trauma_final = organismo.get_valencia(TRAUMA_SETPOINT)
    valencia_habito_final = organismo.get_valencia(-60.0)
    print(f"  Valencia final +60°: {valencia_trauma_final:.1f}")
    print(f"  Valencia final -60°: {valencia_habito_final:.1f}")
    
    print("\n" + "=" * 80)
    print("FASE 3: Evaluación forzada")
    print("=" * 80)
    print(f"  Cada setpoint se presenta fijo por {TEST_DURACION_POR_SETPOINT}s")
    print("  (No se permite cambiar de setpoint hasta completar el tiempo)")
    
    test_datos = {'setpoint': [], 'Cb': [], 'D': [], 'accion': [], 
                  'valencia': {}, 'cb_asociada': {}}
    
    for sp in SETPOINTS_POSIBLES:
        test_datos['valencia'][sp] = []
        test_datos['cb_asociada'][sp] = []
    
    for sp in SETPOINTS_POSIBLES:
        print(f"\n    Evaluando setpoint {sp:+.0f}°...")
        inicio_sp = t_actual
        
        while t_actual - inicio_sp < TEST_DURACION_POR_SETPOINT:
            t = t_actual
            setpoint = sp
            
            (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
             juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
                t, DT, t_actual + TEST_DURACION_POR_SETPOINT, setpoint, trauma=False)
            
            test_datos['setpoint'].append(sp)
            test_datos['Cb'].append(Cb)
            test_datos['D'].append(D)
            test_datos['valencia'][sp].append(organismo.get_valencia(sp))
            test_datos['cb_asociada'][sp].append(organismo.get_cb_asociada(sp))
            
            # Determinar si hubo acción (cambio significativo de orientación)
            if len(test_datos['accion']) > 0:
                delta = abs(orient - test_datos.get('ultima_orient', orient))
                test_datos['accion'].append(delta > 0.5)
            else:
                test_datos['accion'].append(False)
            test_datos['ultima_orient'] = orient
            
            t_actual += DT
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V173 — Valencia asociativa local")
    print("=" * 80)
    
    # Calcular valencia media por setpoint
    valencia_media = {}
    cb_media = {}
    accion_por_setpoint = {}
    
    for sp in SETPOINTS_POSIBLES:
        valencia_media[sp] = np.mean(test_datos['valencia'][sp]) if test_datos['valencia'][sp] else 0
        cb_media[sp] = np.mean(test_datos['Cb']) if test_datos['Cb'] else 0
    
    # Acciones por setpoint (aproximado por segmentos)
    for sp in SETPOINTS_POSIBLES:
        sp_indices = [i for i, s in enumerate(test_datos['setpoint']) if abs(s - sp) < 5.0]
        if sp_indices:
            accion_por_setpoint[sp] = np.mean([test_datos['accion'][i] for i in sp_indices])
        else:
            accion_por_setpoint[sp] = 0.0
    
    # Valencia diferencial
    val_trauma = valencia_media.get(TRAUMA_SETPOINT, 0)
    val_habito = valencia_media.get(-60.0, 0)
    val_neutral = valencia_media.get(0.0, 0)
    
    valencia_diferencial_ok = (val_trauma < val_habito - UMBRAL_VALENCIA_DIFERENCIAL and
                               val_trauma < val_neutral - UMBRAL_VALENCIA_DIFERENCIAL)
    
    # Desacople sostenido
    D_test = np.array(test_datos['D'])
    tiempo_desacople = 0.0
    max_tiempo_desacople = 0.0
    for d in D_test:
        if d > UMBRAL_DESACOPLE:
            tiempo_desacople += DT
            if tiempo_desacople > max_tiempo_desacople:
                max_tiempo_desacople = tiempo_desacople
        else:
            tiempo_desacople = 0.0
    desacople_sostenido = max_tiempo_desacople >= TIEMPO_MINIMO_DESACOPLE
    
    # No abstención
    acciones_seguras = [accion_por_setpoint.get(-60.0, 0), accion_por_setpoint.get(0.0, 0)]
    tasa_accion_segura = np.mean(acciones_seguras)
    
    # Hábito preservado
    cb_habito = cb_media.get(-60.0, 0)
    habito_preservado = cb_habito < UMBRAL_HABITO_SEGURO
    
    print(f"\n  📊 VALENCIA MEDIA POR SETPOINT:")
    for sp in SETPOINTS_POSIBLES:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: Val={valencia_media[sp]:.1f}, Cb={cb_media[sp]:.1f}, P(acción)={accion_por_setpoint[sp]:.3f}{marker}")
    
    print(f"\n  📊 MÉTRICAS CLAVE:")
    print(f"    Valencia diferencial: Val(+60°)={val_trauma:.1f}, Val(-60°)={val_habito:.1f}, Val(0°)={val_neutral:.1f} -> {'✅' if valencia_diferencial_ok else '❌'}")
    print(f"    Hábito preservado: Cb(-60°)={cb_habito:.1f} < {UMBRAL_HABITO_SEGURO} -> {'✅' if habito_preservado else '❌'}")
    print(f"    Desacople sostenido: {max_tiempo_desacople:.2f}s > {TIEMPO_MINIMO_DESACOPLE}s -> {'✅' if desacople_sostenido else '❌'}")
    print(f"    No abstención: tasa acción segura={tasa_accion_segura:.3f} > {UMBRAL_TASA_ACCION} -> {'✅' if tasa_accion_segura > UMBRAL_TASA_ACCION else '❌'}")
    
    exito = (valencia_diferencial_ok and habito_preservado and 
             desacople_sostenido and tasa_accion_segura > UMBRAL_TASA_ACCION)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ VALENCIA ASOCIATIVA LOCAL DEMOSTRADA")
        print("")
        print("     ANIMA-2 demuestra:")
        print("     ✓ Val(+60°) < Val(-60°) y Val(+60°) < Val(0°)")
        print("     ✓ Hábito preservado (Cb baja en -60°)")
        print("     ✓ Desacople sostenido durante la evaluación")
        print("     ✓ El sistema sigue actuando en opciones seguras")
        print("")
        print("  Siguiente paso: V174 — Primer 'No' operativo (negación específica)")
    else:
        print("  ⚠️ VALENCIA ASOCIATIVA LOCAL NO DEMOSTRADA")
        if not valencia_diferencial_ok:
            print("     No se logró valencia diferencial suficiente")
        if not habito_preservado:
            print("     El hábito fue contaminado (Cb alta en -60°)")
        if not desacople_sostenido:
            print("     Desacople insuficiente durante la evaluación")
        if tasa_accion_segura <= UMBRAL_TASA_ACCION:
            print("     El sistema se abstiene en opciones seguras")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Valencia por setpoint
    ax = axes[0, 0]
    sps = list(valencia_media.keys())
    vals = list(valencia_media.values())
    colors = ['red' if sp == TRAUMA_SETPOINT else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), vals, color=colors)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Valencia')
    ax.set_title('FASE 3: Valencia por setpoint')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Cb por setpoint
    ax = axes[0, 1]
    cbs = list(cb_media.values())
    ax.bar(range(len(sps)), cbs, color=colors)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Cb')
    ax.set_title('FASE 3: Cb por setpoint')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Desacople D
    ax = axes[0, 2]
    ax.plot(test_datos['D'], 'purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_DESACOPLE, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(test_datos['D'])), 0, test_datos['D'],
                    where=np.array(test_datos['D']) > UMBRAL_DESACOPLE,
                    color='green', alpha=0.3)
    ax.set_xlabel('Paso')
    ax.set_ylabel('D')
    ax.set_title(f'Desacople (máx {max_tiempo_desacople:.1f}s)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Acciones por setpoint
    ax = axes[1, 0]
    accs = [accion_por_setpoint.get(sp, 0) for sp in sps]
    ax.bar(range(len(sps)), accs, color=colors)
    ax.axhline(y=UMBRAL_TASA_ACCION, color='green', linestyle='--', alpha=0.5, label='Umbral no abstención')
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('P(acción)')
    ax.set_title('FASE 3: Probabilidad de ejecución')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Valencia durante trauma
    ax = axes[1, 1]
    ax.plot(trauma_datos['valencia_trauma'], 'red', linewidth=0.5, label='Valencia +60°')
    ax.plot(trauma_datos['valencia_habito'], 'green', linewidth=0.5, label='Valencia -60°')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Valencia')
    ax.set_title('FASE 2: Evolución de valencia durante trauma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Consolidación de hábito
    ax = axes[1, 2]
    ax.plot(cb_habito_vals, 'blue', linewidth=0.5)
    ax.axhline(y=CONSOLIDACION_OBJETIVO_CB, color='red', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Cb(-60°)')
    ax.set_title('FASE 0: Consolidación del hábito')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V173_logs', exist_ok=True)
    plt.savefig(f'V173_logs/v173_valencia_asociativa_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V173_logs/v173_valencia_asociativa_{timestamp}.png")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v173()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V173 completado. Valencia asociativa local demostrada: {exito}")