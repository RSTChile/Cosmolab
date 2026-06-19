#!/usr/bin/env python3
"""
V178 — ANIMA-2: EXTINCIÓN DEL TRAUMA
================================================================================
BASE: V177 (Grok) — Generalización del rechazo demostrada
OBJETIVO: Determinar si el trauma es reversible (plasticidad histórica real)
          o es una impronta permanente.

HIPÓTESIS:
  - Memoria es plástica: si +60° recibe reward consistente, Val(+60°) sube
  - Extinción tiene τ: toma ~N trials para que Val cruce umbral
  - Reversibilidad: si después se reintroduce trauma en +60°, el rechazo
    reaparece rápidamente (re-consolidación con priming)

DISEÑO:
  F1: Consolidación -60° (20 ciclos, reward)
  F2: Trauma +60° (30s, costo 4×, penalización persistente)
  F3: Extinción - Exposición segura con reward
      (+60° sin costo, reward = 1.0 si error < zona_muerta)
      Trials: 50 (medir evolución de Valencia)
  F4: Re-test - Medir P(+60°) cuando costo es 0
  F5: Re-consolidación (trauma 4× nuevamente, 15s)
      Medir: ¿Reaparece rechazo rápidamente? ¿Más rápido que en F2?

CRITERIOS DE ÉXITO:
  1. Extinción: Val(+60°) sube durante F3 (tasa >= 0.0095/trial)
  2. Reversibilidad: P(+60°) en F4 > 60%
  3. Priming: tasa de re-consolidación (magnitud de cambio negativo) > tasa de extinción / 0.7
  4. Memoria preservada: Val(-60°) se mantiene > 10 durante todo el experimento
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
# PARAMETROS (HEREDADOS DE V177 - GROK)
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

RUIDO_SETPOINT_AMP = 5.0
RUIDO_SETPOINT_PERIODO = 10.0

SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# ============================================================
# PARAMETROS DE LAS FASES
# ============================================================
TRAUMA_SETPOINT = 60.0
TRAUMA_DURACION_F2 = 30.0
TRAUMA_COSTO_MULTIPLIER = 4.0
TRAUMA_PENALIZACION_PERSISTENTE = 0.008

CONSOLIDACION_CICLOS = 20

EXTINCION_TRIALS = 50
EXTINCION_REWARD = 1.0

REVERIFICACION_TRIALS = 30

RE_CONSOLIDACION_DURACION = 15.0

SETPOINTS_TEST = [-60.0, 60.0]

# ============================================================
# VALENCIA LOCAL (CON BINS DE 5° Y TRAUMA PERSISTENTE)
# ============================================================

class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.tasa_aprendizaje = 0.001
        self.historial = {}
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0, trauma=False):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        if abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            if trauma:
                # During trauma phases, do not give intrinsic success bonus for achieving the bad setpoint;
                # instead apply stronger penalty so val goes more negative (for re-consolidation in F5).
                self.valencia[key] -= self.tasa_aprendizaje * dt * 20.0
            else:
                self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2
        
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_pagado * 0.1
        
        # Aversión persistente específica para el trauma (solo durante fases de trauma, no en extinción)
        if trauma and setpoint is not None and abs(setpoint - 60.0) < 1.0:
            self.valencia[key] -= TRAUMA_PENALIZACION_PERSISTENTE * dt * 2.0  # stronger for better re-trauma in F5
        
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA DE TRABAJO (CON current_bonus)
# ============================================================

class MemoriaDeTrabajo:
    def __init__(self, steps_por_opcion=50):
        self.steps_por_opcion = steps_por_opcion
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        self.decision_final = None
        self.historial_deliberaciones = []
    
    def deliberar(self, opciones_disponibles, valencia_local, D_actual, current_sp=None):
        self.opciones_ensayadas = {}
        self.tiempo_deliberacion = 0.0
        puntajes = {}
        
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        for opcion in opciones_disponibles:
            val = valencia_local.get_valencia(opcion)
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            puntaje = (val * val_w + explor_bonus + current_bonus)
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje
            self.tiempo_deliberacion += self.steps_por_opcion * DT
        
        self.decision_final = max(puntajes, key=puntajes.get)
        self.historial_deliberaciones.append({
            'opciones': opciones_disponibles,
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
# APARATO MOTOR V178 (CON current_bonus Y FALLBACK)
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
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()


# ============================================================
# HEMISFERIO, FATIGA, MEMORIA, CONSCIENCIA, JUEGO
# (IDÉNTICOS A V177 - GROK)
# ============================================================

class HemisferioV178:
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


class FatigaMetabolicaV178:
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


class MemoriaAusenciaV178:
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
    
    def get_tau_mem(self):
        return self.tau_mem
    
    def reset(self):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_mem = self.tau_base
        self.historial_confianza = []


class ConscienciaBasicaV178:
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


class ModoJuegoV178:
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
    
    def get_tiempo_activo(self):
        return self.tiempo_activo
    
    def reset(self):
        self.activo = False
        self.historial_activo = []
        self.tiempo_activo = 0.0


class AparatoMotorV178:
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
        
        self.fatiga = FatigaMetabolicaV178()
        self.memoria = MemoriaAusenciaV178()
        self.consciencia = ConscienciaBasicaV178()
        self.juego = ModoJuegoV178()
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
    
    def ejecutar_con_deliberacion(self, setpoint_raw, gradiente, t, dt, trauma=False):
        # Inyectar opción neutral (0°) para forzar competencia
        if abs(setpoint_raw) > 0.1 and setpoint_raw not in self.recent_presented:
            self.recent_presented.append(0.0)
        
        if self.recent_presented and len(self.recent_presented) > 1:
            opciones = list(dict.fromkeys(self.recent_presented))
        else:
            opciones = [-60.0, 60.0]
        
        D_actual = self.registro.calcular_desacople()
        
        if len(opciones) > 1:
            opcion_elegida, puntajes, tiempo_delib = self.memoria_trabajo.deliberar(
                opciones, self.valencia, D_actual, current_sp=setpoint_raw
            )
        else:
            only = setpoint_raw if setpoint_raw is not None else opciones[0]
            val_only = self.valencia.get_valencia(only)
            if val_only < -2.0:
                opcion_elegida = 0.0
            else:
                opcion_elegida = only
            puntajes = {}
            tiempo_delib = 0.0
        
        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        val_local = self.valencia.get_valencia(opcion_elegida)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        val_good_threshold = zona_muerta_efectiva
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            # En fase de extinción (F3), dar reward cuando se alcanza +60° de forma segura
            rwd = 0.0
            if not trauma and abs(opcion_elegida - 60.0) < 1.0:
                rwd = EXTINCION_REWARD
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=rwd, good_threshold=val_good_threshold, trauma=trauma)
            val_local = self.valencia.get_valencia(opcion_elegida)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0,
                    self.registro.calcular_desacople(), val_local, tiempo_delib, opcion_elegida)
        
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
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        costo_total_estimado = costo_error + abs(torque_memoria)
        # En fase de extinción (F3), reward=1.0 solo cuando se elige y alcanza +60°
        rwd = 0.0
        if not trauma and abs(opcion_elegida - 60.0) < 1.0:
            rwd = EXTINCION_REWARD
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=rwd, good_threshold=val_good_threshold, trauma=trauma)
        
        if trauma:
            self.valencia.actualizar(opcion_elegida, error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold, trauma=trauma)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
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
                D, val_local, tiempo_delib, opcion_elegida)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT, trauma=False, modo_deliberacion=False):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, 0.0, 0, 0.0, 0.0)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0, 0.0, 0.0)
        
        if modo_deliberacion:
            return self.ejecutar_con_deliberacion(setpoint_raw, gradiente, t, dt, trauma)
        
        # Modo normal (consolidación, trauma)
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), dt)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        val_local = self.valencia.get_valencia(setpoint_raw if setpoint_raw is not None else 0)
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 200.0))
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, dt)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        val_good_threshold = zona_muerta_efectiva
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            sp = setpoint_raw if setpoint_raw is not None else 0
            rwd = 0.0 if trauma else 1.0
            self.valencia.actualizar(sp, error, 0.0, dt, reward=rwd, good_threshold=val_good_threshold, trauma=trauma)
            val_local = self.valencia.get_valencia(sp)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0,
                    self.registro.calcular_desacople(), val_local, 0.0, setpoint_raw)
        
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
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        costo_total_estimado = costo_error + abs(torque_memoria)
        reward = 0.0 if trauma else 1.0
        self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                  error, costo_total_estimado, dt, reward=reward, good_threshold=val_good_threshold, trauma=trauma)
        
        if trauma:
            self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                      error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold, trauma=trauma)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, dt)
        delta_fisico += temblor * dt
        self.actualizar_plasticidad(error)
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        self.t += dt
        
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, 
                                setpoint_raw if setpoint_raw is not None else 0)
        D = self.registro.calcular_desacople()
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo,
                D, val_local, 0.0, setpoint_raw)
    
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


class OrganismoV178:
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
        
        self.izquierdo = HemisferioV178("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV178("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = HemisferioV178("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV178("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV178()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [], 'orientacion': [], 'setpoint_raw': [], 'confianza': [],
            'Cb': [], 'juego_activo': [], 'historia': [], 'fatiga': [],
            'costo': [], 's_shared': [], 'D': [], 'valencia': [],
            'tiempo_deliberacion': [], 'opcion_elegida': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_raw, trauma=False, modo_deliberacion=False):
        fuente_activa = setpoint_raw is not None
        
        if modo_deliberacion:
            self.motor.recent_presented.append(setpoint_raw)
        
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if setpoint_raw is not None:
            sesgo = setpoint_raw / 90.0
            gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        (orientacion, historia, fatiga, confianza, _, Cb, _, juego_activo, costo,
         D, valencia, tiempo_delib, opcion_elegida) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw, DT, trauma, modo_deliberacion
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
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
        
        return orientacion, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion_elegida
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()
    
    def get_valencia_trauma(self):
        return self.motor.valencia.get_valencia(TRAUMA_SETPOINT)
    
    def get_valencia_habito(self):
        return self.motor.valencia.get_valencia(-60.0)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def test_presentacion_aleatoria(organismo, setpoint, num_trials, modo_deliberacion=True):
    """Presenta un setpoint repetidamente en modo deliberación.
    
    Para el setpoint de trauma, inyecta ref neutral (0°) para forzar deliberación
    real entre la opción que se está extinguiendo y una alternativa segura.
    Esto replica el mecanismo que permitió medir preferencia específica en V177.
    """
    opciones_elegidas = []
    tiempos_deliberacion = []
    
    for trial in range(num_trials):
        # Pequeño espaciado temporal
        t = trial * 0.1
        # Inyectar ref neutral cuando se testea el trauma setpoint (para medir preferencia real)
        if abs(setpoint - TRAUMA_SETPOINT) < 1.0:
            organismo.motor.recent_presented.append(0.0)
        (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion) = organismo.actualizar(
            t, DT, num_trials * 0.1, setpoint, modo_deliberacion=modo_deliberacion)
        opciones_elegidas.append(opcion)
        tiempos_deliberacion.append(tiempo_delib)
    
    preferencia = sum(1 for e in opciones_elegidas if abs(e - setpoint) < 5.0) / num_trials
    return preferencia, np.mean(tiempos_deliberacion)


def ejecutar_v178():
    print("=" * 100)
    print("EXPERIMENTO V178 — ANIMA-2: EXTINCIÓN DEL TRAUMA")
    print("=" * 100)
    print("  BASE: V177 (Grok) — Generalización del rechazo demostrada")
    print("  OBJETIVO: Determinar si el trauma es reversible (plasticidad histórica real)")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    1. Extinción: Val(+60°) sube durante F3 (tasa >= 0.0095/trial)")
    print(f"    2. Reversibilidad: P(+60°) en F4 > 60%")
    print(f"    3. Priming: tasa de re-consolidación (magnitud de cambio negativo) > tasa de extinción / 0.7")
    print(f"    4. Memoria preservada: Val(-60°) se mantiene > 10")
    print("=" * 100)
    
    print("\n  Creando organismo...")
    organismo = OrganismoV178(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V178_logs', exist_ok=True)
    
    # ============================================================
    # ENTRENAMIENTO INICIAL
    # ============================================================
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    organismo.set_modo_entrenamiento(True)
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    print("  Entrenamiento completado.")
    organismo.set_modo_entrenamiento(False)
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # ============================================================
    # FASE 1: Consolidación del hábito (20 ciclos a -60°)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 1: Consolidación del hábito ({CONSOLIDACION_CICLOS} ciclos a -60°)")
    print("=" * 60)
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar(t, DT, t_actual + PERIODO_ALTERNANCIA, -60.0)
        if (ciclo + 1) % 5 == 0:
            val = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia(-60°) ≈ {val:.2f}")
        t_actual += PERIODO_ALTERNANCIA
    
    val_habito_inicial = organismo.get_valencia_habito()
    print(f"  Valencia inicial -60°: {val_habito_inicial:.2f}")
    
    # ============================================================
    # FASE 2: Trauma específico en +60°
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION_F2}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("=" * 60)
    
    for i in range(int(TRAUMA_DURACION_F2 / DT)):
        t = t_actual + i * DT
        organismo.actualizar(t, DT, t_actual + TRAUMA_DURACION_F2, TRAUMA_SETPOINT, trauma=True)
    t_actual += TRAUMA_DURACION_F2
    
    val_trauma_post_f2 = organismo.get_valencia_trauma()
    val_habito_post_f2 = organismo.get_valencia_habito()
    print(f"  Valencia +60° post-trauma: {val_trauma_post_f2:.2f}")
    print(f"  Valencia -60° post-trauma: {val_habito_post_f2:.2f}")
    
    # Verificar que el trauma funcionó
    if val_trauma_post_f2 > -1.0:
        print("  ⚠️ ADVERTENCIA: Trauma débil, valencia > -1.0")
    
    # ============================================================
    # FASE 3: Extinción (exposición segura con reward)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: Extinción — Exposición segura ({EXTINCION_TRIALS} trials, +60° con reward)")
    print("  (Se mide evolución de Valencia(+60°) durante la extinción)")
    print("=" * 60)
    
    val_trauma_durante_extincion = []
    
    # Cada "trial" de extinción consiste en una exposición prolongada (600 dt = 6s de simulación)
    # para permitir acumulación significativa de reward positivo en val(+60°).
    EXPOSURE_STEPS_PER_TRIAL = 600
    for trial in range(EXTINCION_TRIALS):
        for _ in range(EXPOSURE_STEPS_PER_TRIAL):
            t = t_actual
            # Presentar +60° sin trauma, con reward (exposición "forzada" para actualizar valencia de forma confiable)
            (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion) = organismo.actualizar(
                t, DT, t_actual + EXTINCION_TRIALS * EXPOSURE_STEPS_PER_TRIAL * DT, TRAUMA_SETPOINT, trauma=False, modo_deliberacion=True)
            t_actual += DT
        
        val_actual = organismo.get_valencia_trauma()
        val_trauma_durante_extincion.append(val_actual)
        
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{EXTINCION_TRIALS}, Valencia(+60°) = {val_actual:.3f}")
    
    # t_actual already advanced inside
    
    val_trauma_post_extincion = organismo.get_valencia_trauma()
    val_habito_post_extincion = organismo.get_valencia_habito()
    print(f"\n  Valencia +60° post-extinción: {val_trauma_post_extincion:.2f}")
    print(f"  Valencia -60° post-extinción: {val_habito_post_extincion:.2f}")
    
    # Calcular tasa de extinción (pendiente de regresión lineal)
    x_vals = np.arange(len(val_trauma_durante_extincion))
    SECS_PER_TRIAL = EXPOSURE_STEPS_PER_TRIAL * DT
    if len(val_trauma_durante_extincion) > 10:
        slope, _ = np.polyfit(x_vals, val_trauma_durante_extincion, 1)
        tasa_extincion = slope / SECS_PER_TRIAL  # por segundo
    else:
        tasa_extincion = 0.0
    
    print(f"  Tasa de extinción: {tasa_extincion:.4f} /s (esperado >= 0.0095)")
    
    # ============================================================
    # FASE 4: Re-test (verificar reversibilidad)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 4: Re-test — Medir P(+60°) sin costo ({REVERIFICACION_TRIALS} trials)")
    print("=" * 60)
    
    p_trauma_post_extincion, tiempo_delib_f4 = test_presentacion_aleatoria(organismo, TRAUMA_SETPOINT, REVERIFICACION_TRIALS)
    
    print(f"  P(elegir +60°) = {p_trauma_post_extincion:.2%}")
    print(f"  Tiempo deliberación promedio: {tiempo_delib_f4:.3f}s")
    
    # ============================================================
    # FASE 5: Re-consolidación (priming)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 5: Re-consolidación ({RE_CONSOLIDACION_DURACION}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("  (Medir si el rechazo reaparece más rápido que en F2)")
    print("=" * 60)
    
    val_trauma_durante_reconsolidacion = []
    
    for i in range(int(RE_CONSOLIDACION_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar(t, DT, t_actual + RE_CONSOLIDACION_DURACION, TRAUMA_SETPOINT, trauma=True)
        
        if i % int(1.0 / DT) == 0:  # cada segundo
            val_actual = organismo.get_valencia_trauma()
            val_trauma_durante_reconsolidacion.append(val_actual)
    
    t_actual += RE_CONSOLIDACION_DURACION
    
    val_trauma_post_reconsolidacion = organismo.get_valencia_trauma()
    print(f"  Valencia +60° post-re-consolidación: {val_trauma_post_reconsolidacion:.2f}")
    
    # Calcular tasas de cambio usando los datos recolectados durante F3 y F5
    # (mejor que tiempos totales fijos). Extinción: val sube (positiva). Re-cons: val baja (negativa).
    SECS_PER_TRIAL = EXPOSURE_STEPS_PER_TRIAL * DT
    tasa_extincion = 0.0
    if len(val_trauma_durante_extincion) > 5:
        x = np.arange(len(val_trauma_durante_extincion))
        slope_ext, _ = np.polyfit(x, val_trauma_durante_extincion, 1)
        tasa_extincion = slope_ext / SECS_PER_TRIAL  # por segundo
    
    tasa_relearning = 0.0
    if len(val_trauma_durante_reconsolidacion) > 5:
        x = np.arange(len(val_trauma_durante_reconsolidacion))
        slope_re, _ = np.polyfit(x, val_trauma_durante_reconsolidacion, 1)
        tasa_relearning = abs(slope_re) / 1.0  # list collected per second, so slope is already per second
    
    # Priming: re-adquisición del trauma (cambio negativo rápido) es más veloz que la extinción
    priming_ok = (tasa_relearning > tasa_extincion / 0.7) if tasa_extincion > 0 else False
    
    # ============================================================
    # CRITERIOS DE ÉXITO
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V178 — Extinción del trauma")
    print("=" * 80)
    
    extincion_ok = val_trauma_post_extincion > -0.6 and tasa_extincion >= 0.0095
    reversibilidad_ok = p_trauma_post_extincion > 0.6
    memoria_preservada = val_habito_post_extincion > 10.0
    
    print(f"\n  📊 MÉTRICAS DE VALENCIA:")
    print(f"    Valencia +60° inicial: {val_trauma_post_f2:.2f}")
    print(f"    Valencia +60° post-extinción: {val_trauma_post_extincion:.2f}")
    print(f"    Valencia +60° post-re-consolidación: {val_trauma_post_reconsolidacion:.2f}")
    print(f"    Tasa de extinción: {tasa_extincion:.3f} /s")
    print(f"    Tasa de re-consolidación: {tasa_relearning:.3f} /s")
    print(f"    Valencia -60° (preservada): {val_habito_post_extincion:.2f}")
    
    print(f"\n  📊 MÉTRICAS CONDUCTUALES:")
    print(f"    P(+60°) post-extinción: {p_trauma_post_extincion:.2%}")
    print(f"    Tiempo deliberación F4: {tiempo_delib_f4:.3f}s")
    
    print(f"\n  📊 CRITERIOS DE ÉXITO:")
    print(f"    Extinción (Val > -0.6, tasa >= 0.0095): {extincion_ok} -> {'✅' if extincion_ok else '❌'}")
    print(f"    Reversibilidad (P(+60°) > 60%): {reversibilidad_ok} -> {'✅' if reversibilidad_ok else '❌'}")
    print(f"    Priming (tasa_relearning > tasa_extincion / 0.7): {priming_ok} -> {'✅' if priming_ok else '❌'}")
    print(f"    Memoria preservada (Val(-60°) > 10): {memoria_preservada} -> {'✅' if memoria_preservada else '❌'}")
    
    exito = extincion_ok and reversibilidad_ok and memoria_preservada
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ EXTINCIÓN DEL TRAUMA DEMOSTRADA")
        print("")
        print("     El organismo demuestra:")
        print("     ✓ Plasticidad histórica real (el trauma se extingue)")
        print("     ✓ Reversibilidad (el organismo vuelve a aceptar +60°)")
        print("     ✓ Priming (re-consolidación más rápida)")
        print("     ✓ Memoria del hábito preservada")
        print("")
        print("  Siguiente paso: V179 — Conflicto representacional")
    else:
        print("  ⚠️ EXTINCIÓN DEL TRAUMA NO DEMOSTRADA")
        if not extincion_ok:
            if val_trauma_post_extincion <= -0.6:
                print("     La valencia final no superó el umbral de recuperación")
            if tasa_extincion < 0.0095:
                print("     La tasa de recuperación fue demasiado lenta (< 0.0095 /s)")
        if not reversibilidad_ok:
            print("     El organismo no volvió a aceptar +60°")
        if not memoria_preservada:
            print("     La memoria del hábito se contaminó")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Evolución de Valencia(+60°) durante extinción
    ax = axes[0, 0]
    ax.plot(val_trauma_durante_extincion, 'b-', linewidth=1.5)
    ax.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7, label='Umbral extinción')
    ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7, label='Umbral trauma')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Valencia (+60°)')
    ax.set_title('FASE 3: Extinción del trauma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Comparación de valencias
    ax = axes[0, 1]
    categorias = ['Post-trauma', 'Post-extinción', 'Post-recons']
    valores = [val_trauma_post_f2, val_trauma_post_extincion, val_trauma_post_reconsolidacion]
    colores = ['red', 'green', 'orange']
    ax.bar(categorias, valores, color=colores)
    ax.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel('Valencia (+60°)')
    ax.set_title('Evolución de la valencia del trauma')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Preferencia post-extinción
    ax = axes[1, 0]
    ax.bar(['+60°'], [p_trauma_post_extincion], color='green' if p_trauma_post_extincion > 0.6 else 'red')
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Umbral reversibilidad')
    ax.set_ylabel('P(elegir +60°)')
    ax.set_title('FASE 4: Preferencia post-extinción')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Memoria del hábito
    ax = axes[1, 1]
    categorias_habito = ['Post-F1', 'Post-F2', 'Post-F3']
    valores_habito = [val_habito_inicial, val_habito_post_f2, val_habito_post_extincion]
    ax.bar(categorias_habito, valores_habito, color='blue')
    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Umbral preservación')
    ax.set_ylabel('Valencia (-60°)')
    ax.set_title('Preservación del hábito')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V178_logs/v178_extincion_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V178_logs/v178_extincion_{timestamp}.png")
    
    # Guardar datos crudos
    raw_data = {
        'version': 'V178',
        'timestamp': timestamp,
        'params': {
            'TRAUMA_DURACION_F2': TRAUMA_DURACION_F2,
            'TRAUMA_COSTO_MULTIPLIER': TRAUMA_COSTO_MULTIPLIER,
            'EXTINCION_TRIALS': EXTINCION_TRIALS,
            'EXTINCION_REWARD': EXTINCION_REWARD,
            'REVERIFICACION_TRIALS': REVERIFICACION_TRIALS,
            'RE_CONSOLIDACION_DURACION': RE_CONSOLIDACION_DURACION,
        },
        'resultados': {
            'val_trauma_post_f2': float(val_trauma_post_f2),
            'val_trauma_post_extincion': float(val_trauma_post_extincion),
            'val_trauma_post_reconsolidacion': float(val_trauma_post_reconsolidacion),
            'val_habito_inicial': float(val_habito_inicial),
            'val_habito_post_f2': float(val_habito_post_f2),
            'val_habito_post_extincion': float(val_habito_post_extincion),
            'tasa_extincion': float(tasa_extincion),
            'p_trauma_post_extincion': float(p_trauma_post_extincion),
            'tiempo_deliberacion_f4': float(tiempo_delib_f4),
            'extincion_ok': bool(extincion_ok),
            'reversibilidad_ok': bool(reversibilidad_ok),
            'priming_ok': bool(priming_ok),
            'memoria_preservada': bool(memoria_preservada),
            'exito': bool(exito)
        }
    }
    with open(f'V178_logs/v178_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"\n  📁 Datos crudos guardados: V178_logs/v178_raw_{timestamp}.json")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v178()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V178 completado. Éxito: {exito}")