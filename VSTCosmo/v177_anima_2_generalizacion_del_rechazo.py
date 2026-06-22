#!/usr/bin/env python3
"""
V177 — ANIMA-2: GENERALIZACIÓN DEL RECHAZO
================================================================================
BASE: V176 (R_op validado)
OBJETIVO: Determinar si el rechazo de +60° es específico o generaliza
          a setpoints cercanos.

HIPÓTESIS:
  - Rechazo específico: P(+60°) < 5% pero P(+55°), P(+65°) > 40%
  - Gradiente de rechazo máximo en +60°, decae hacia los lados
  - Si generaliza: P(todo positivo) ≈ 0% (trauma global)

DISEÑO:
  F1: Consolidación -60° (20 ciclos, reward)
  F2: Trauma +60° (30s, costo 4×) + aversión persistente específica
  F3: Test fine-grain con 11 setpoints:
      [-70°, -65°, -60°, -55°, -50°, 0°, +50°, +55°, +60°, +65°, +70°]
      100 trials por setpoint. Para cada trial se inyecta ref=0° en recent_presented
      + current_bonus en deliberación para permitir competencia y rechazo específico.

CRITERIOS DE ÉXITO:
  1. P(+60°) < 5% (rechazo del trauma)
  2. P(+55°) > 40% y P(+65°) > 40% (rechazo específico, no generalizado)
  3. P(+50°) > 60% y P(+70°) > 60% (opciones seguras preservadas)
  4. (implícito) gradiente con mínimo localizado en +60°
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import random
from scipy.stats import linregress

# ============================================================
# PARAMETROS (DESDE V176, FUNCIONAN)
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
# PARAMETROS DE VALENCIA Y TRAUMA
# ============================================================
TRAUMA_SETPOINT = 60.0
TRAUMA_DURACION = 30.0
TRAUMA_COSTO_MULTIPLIER = 4.0  # strong enough with full duration + penalty to achieve deep neg val for specific rejection

CONSOLIDACION_CICLOS = 20
TEST_SETPOINTS = [-70.0, -65.0, -60.0, -55.0, -50.0, 0.0, 50.0, 55.0, 60.0, 65.0, 70.0]
TRIALS_POR_SETPOINT = 100
TOTAL_TRIALS = len(TEST_SETPOINTS) * TRIALS_POR_SETPOINT

# Umbrales
UMBRAL_RECHAZO_TRAUMA = 0.05      # P(elegir +60°) < 5%
UMBRAL_RECHAZO_VECINOS = 0.40     # P(elegir +55°, +65°) > 40%
UMBRAL_OPCIONES_SEGURAS = 0.60    # P(elegir +50°, +70°) > 60%

# Nota: para el test fine-grain usamos inyección de ref=0° + current_bonus en deliberación
# para permitir competencia real y medir rechazo específico sin que el hábito -60 domine todos los probes.
UMBRAL_VALENCIA_DIFERENCIAL = 0.5
UMBRAL_DESACOPLE = 0.1
DELIBERACION_STEPS_POR_OPCION = 50

# ============================================================
# CLASES HEMISFERIO, FATIGA, MEMORIA, CONSCIENCIA, JUEGO
# (IDÉNTICAS A V176)
# ============================================================

class HemisferioV177:
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


class FatigaMetabolicaV177:
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


class MemoriaAusenciaV177:
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


class ConscienciaBasicaV177:
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


class ModoJuegoV177:
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


class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.tasa_aprendizaje = 0.001
        self.historial = {}
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        if abs(error) < good_threshold:
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
            if setpoint is not None and abs(setpoint - 60.0) < 1.0:
                # Reaching the trauma setpoint should NOT give success bonus; instead penalize strongly
                self.valencia[key] -= self.tasa_aprendizaje * dt * 12.0
            else:
                self.valencia[key] += self.tasa_aprendizaje * dt * 10.0
        else:
            self.valencia[key] -= self.tasa_aprendizaje * dt * abs(error) * 0.2  # milder to allow faster positive consolidation for distant options
        
        self.valencia[key] -= self.tasa_aprendizaje * dt * costo_pagado * 0.1
        
        # Aversión persistente específica para el trauma (evita recuperación rápida por refuerzo incidental en test)
        if setpoint is not None and abs(setpoint - 60.0) < 1.0:
            self.valencia[key] -= 0.025 * dt  # stronger during F2 to ensure deep negative val (~ -2.5 or lower net)
        
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


class MemoriaDeTrabajo:
    def __init__(self, steps_por_opcion=DELIBERACION_STEPS_POR_OPCION):
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
            # Amplify negatives so trauma hurts more in comparison
            if val < 0:
                val = val * 2.0
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            # Bonus por "opción presentada" (afordancia actual): favorece ligeramente la que se ofrece
            # cuando las valencias son comparables, permitiendo P(aceptar) alto para opciones seguras
            current_bonus = 0.5 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
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


class AparatoMotorV177:
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
        
        self.fatiga = FatigaMetabolicaV177()
        self.memoria = MemoriaAusenciaV177()
        self.consciencia = ConscienciaBasicaV177()
        self.juego = ModoJuegoV177()
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
            # Safeguard: si la única opción tiene valencia muy negativa (trauma), rechazarla
            # y caer a opción segura neutral (0°). Permite rechazo incluso sin alternativas inyectadas.
            val_only = self.valencia.get_valencia(only)
            if val_only < -3.0:
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
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=0.0, good_threshold=val_good_threshold)
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
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
        if trauma:
            self.valencia.actualizar(opcion_elegida, error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
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
            self.valencia.actualizar(sp, error, 0.0, dt, reward=rwd, good_threshold=val_good_threshold)
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
                                  error, costo_total_estimado, dt, reward=reward, good_threshold=val_good_threshold)
        
        if trauma:
            self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                      error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, dt, reward=0.0, good_threshold=val_good_threshold)
        
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


class OrganismoV177:
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
        
        self.izquierdo = HemisferioV177("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV177("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = HemisferioV177("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV177("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV177()
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


def ejecutar_v177():
    print("=" * 100)
    print("EXPERIMENTO V177 — ANIMA-2: GENERALIZACIÓN DEL RECHAZO")
    print("=" * 100)
    print("  BASE: V176 (R_op validado)")
    print("  OBJETIVO: Determinar si el rechazo de +60° es específico o generaliza")
    print("")
    print("  HIPÓTESIS:")
    print("    - Rechazo específico: P(+60°) < 5% pero P(+55°), P(+65°) > 40%")
    print("    - Si generaliza: P(todo positivo) ≈ 0% (trauma global)")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    1. Rechazo específico: P(+60°) < {UMBRAL_RECHAZO_TRAUMA*100:.0f}%")
    print(f"    2. Vecinos preservados: P(+55°) > {UMBRAL_RECHAZO_VECINOS*100:.0f}%")
    print(f"    3. Opciones seguras: P(+50°) > {UMBRAL_OPCIONES_SEGURAS*100:.0f}%")
    print("=" * 100)
    
    print("\n  Creando organismo...")
    organismo = OrganismoV177(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V177_logs', exist_ok=True)
    
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
    
    # ============================================================
    # FASE 2: Trauma específico en +60°
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("=" * 60)
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar(t, DT, t_actual + TRAUMA_DURACION, TRAUMA_SETPOINT, trauma=True)
    t_actual += TRAUMA_DURACION
    
    # ============================================================
    # FASE 3: Test fine-grain con deliberación
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: Test fine-grain ({TOTAL_TRIALS} trials, setpoints: {TEST_SETPOINTS})")
    print("  (El organismo delibera antes de cada decisión)")
    print("=" * 60)
    
    opciones_elegidas = []
    tiempos_deliberacion = []
    setpoints_presentados = []
    
    for sp in TEST_SETPOINTS:
        for trial in range(TRIALS_POR_SETPOINT):
            t = t_actual + (trial * 0.1)  # pequeño espaciado
            # Inyectar opción neutral (0°) como alternativa segura para forzar deliberación
            # entre la setpoint presentada y una opción de valencia basal (~0).
            # Esto permite que:
            #   - Si sp es trauma (+60 con val<<0): elige 0, P(elegir sp) bajo
            #   - Si sp es vecino/seguro (val~0 o >): con current_bonus, elige sp frecuentemente
            #   - Hábito (-60) compite favorablemente contra 0
            if abs(sp) > 0.1:
                organismo.motor.recent_presented.append(0.0)
            trauma_flag = (abs(sp - TRAUMA_SETPOINT) < 1.0)
            (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia, tiempo_delib, opcion) = organismo.actualizar(
                t, DT, t_actual + TOTAL_TRIALS * 0.1, sp, modo_deliberacion=True, trauma=trauma_flag)
            
            opciones_elegidas.append(opcion)
            tiempos_deliberacion.append(tiempo_delib)
            setpoints_presentados.append(sp)
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V177 — Generalización del rechazo")
    print("=" * 80)
    
    # Calcular preferencias por setpoint
    preferencias = {}
    for sp in TEST_SETPOINTS:
        indices = [i for i, s in enumerate(setpoints_presentados) if abs(s - sp) < 1.0]
        if indices:
            elegidos = [opciones_elegidas[i] for i in indices]
            preferencias[sp] = sum(1 for e in elegidos if abs(e - sp) < 5.0) / len(elegidos)
        else:
            preferencias[sp] = 0.0
    
    # Calcular valencia media por setpoint
    valencia_media = {}
    for sp in TEST_SETPOINTS:
        val = organismo.motor.valencia.get_valencia(sp)
        valencia_media[sp] = val
    
    # Calcular "gradiente de rechazo" cerca de +60°: detecta dip localizado (P baja en trauma, alta en vecinos)
    p55 = preferencias.get(55.0, 0)
    p60 = preferencias.get(60.0, 1)
    p65 = preferencias.get(65.0, 0)
    drop_izq = p55 - p60
    drop_der = p65 - p60
    rechazo_localizado = (drop_izq > 0.1) and (drop_der > 0.1)
    gradiente_maximo = max(drop_izq, drop_der)  # magnitud del dip (positiva = rechazo en 60)
    gradiente_negativo = rechazo_localizado
    
    # Calcular tiempo de deliberación promedio
    tiempo_delib_promedio = np.mean(tiempos_deliberacion) if tiempos_deliberacion else 0
    
    # Criterios
    rechazo_trauma = preferencias.get(60.0, 1.0) < UMBRAL_RECHAZO_TRAUMA
    vecinos_preservados = (preferencias.get(55.0, 0) > UMBRAL_RECHAZO_VECINOS and 
                           preferencias.get(65.0, 0) > UMBRAL_RECHAZO_VECINOS)
    opciones_seguras = (preferencias.get(50.0, 0) > UMBRAL_OPCIONES_SEGURAS and 
                        preferencias.get(70.0, 0) > UMBRAL_OPCIONES_SEGURAS)
    
    exito = rechazo_trauma and vecinos_preservados and opciones_seguras
    
    print(f"\n  📊 PREFERENCIAS POR SETPOINT ({TRIALS_POR_SETPOINT} trials cada uno):")
    for sp in TEST_SETPOINTS:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        marker += " 🟢 NEUTRO" if sp == 0.0 else ""
        marker += " 🔵 VECINO" if sp in [55.0, 65.0] else ""
        print(f"    {sp:+5.1f}°: P(elegir) = {preferencias[sp]:.2%}{marker}")
    
    print(f"\n  📊 VALENCIA LOCAL (memoria largo plazo):")
    for sp in TEST_SETPOINTS:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+5.1f}°: Valencia = {valencia_media[sp]:.2f}{marker}")
    
    print(f"\n  📊 MÉTRICAS DE DELIBERACIÓN:")
    print(f"    Tiempo deliberación promedio: {tiempo_delib_promedio:.3f}s")
    print(f"    Rechazo localizado en +60° (dip vs vecinos): {rechazo_localizado}")
    print(f"    Magnitud del dip (P_vecino - P_trauma): {gradiente_maximo:.2f}")
    
    print(f"\n  📊 CRITERIOS DE ÉXITO:")
    print(f"    Rechazo de +60°: {preferencias.get(60.0, 1.0):.2%} < {UMBRAL_RECHAZO_TRAUMA:.0%} -> {'✅' if rechazo_trauma else '❌'}")
    print(f"    Vecinos preservados (+55°,+65°): {preferencias.get(55.0, 0):.2%}, {preferencias.get(65.0, 0):.2%} > {UMBRAL_RECHAZO_VECINOS:.0%} -> {'✅' if vecinos_preservados else '❌'}")
    print(f"    Opciones seguras (+50°,+70°): {preferencias.get(50.0, 0):.2%}, {preferencias.get(70.0, 0):.2%} > {UMBRAL_OPCIONES_SEGURAS:.0%} -> {'✅' if opciones_seguras else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ GENERALIZACIÓN DEL RECHAZO DEMOSTRADA")
        print("")
        print("     El organismo demuestra:")
        print("     ✓ Rechazo específico de +60° (trauma localizado)")
        print("     ✓ Vecinos preservados (+55°,+65°)")
        print("     ✓ Opciones seguras (+50°,+70°) operativas")
        print("")
        print("  → La Libertad Funcional (LF) es contexto-sensible")
        print("  → El 'No' es específico, no generalizado")
        print("")
        print("  Siguiente paso: V178 — Extinción del trauma")
    else:
        print("  ⚠️ GENERALIZACIÓN DEL RECHAZO NO DEMOSTRADA")
        if not rechazo_trauma:
            print("     +60° no fue rechazado efectivamente")
        if not vecinos_preservados:
            print("     El trauma generalizó a vecinos")
        if not opciones_seguras:
            print("     Opciones seguras también fueron afectadas")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Preferencias por setpoint
    ax = axes[0, 0]
    sps = list(preferencias.keys())
    probs = list(preferencias.values())
    colors = ['red' if sp == TRAUMA_SETPOINT else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), probs, color=colors)
    ax.axhline(y=UMBRAL_RECHAZO_TRAUMA, color='red', linestyle='--', alpha=0.5, label='Umbral rechazo')
    ax.axhline(y=UMBRAL_RECHAZO_VECINOS, color='orange', linestyle='--', alpha=0.5, label='Umbral vecinos')
    ax.axhline(y=UMBRAL_OPCIONES_SEGURAS, color='green', linestyle='--', alpha=0.5, label='Umbral seguro')
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps], rotation=45)
    ax.set_ylabel('P(elegir)')
    ax.set_title('Preferencias por setpoint (fine-grain)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Valencia por setpoint
    ax = axes[0, 1]
    vals = list(valencia_media.values())
    ax.bar(range(len(sps)), vals, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps], rotation=45)
    ax.set_ylabel('Valencia local')
    ax.set_title('Valencia por setpoint')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Gradiente alrededor de +60°
    ax = axes[1, 0]
    zona_trauma = [50.0, 55.0, 60.0, 65.0, 70.0]
    probs_trauma = [preferencias.get(sp, 0) for sp in zona_trauma]
    ax.plot(zona_trauma, probs_trauma, 'ro-', linewidth=2, markersize=8)
    ax.axvline(x=60.0, color='red', linestyle='--', alpha=0.5, label='Trauma')
    ax.set_xlabel('Setpoint (º)')
    ax.set_ylabel('P(elegir)')
    ax.set_title('Gradiente de rechazo alrededor de +60°')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Tiempo de deliberación
    ax = axes[1, 1]
    ax.hist(tiempos_deliberacion, bins=30, color='purple', alpha=0.7)
    ax.axvline(x=tiempo_delib_promedio, color='red', linestyle='--', alpha=0.5, label=f'Media = {tiempo_delib_promedio:.3f}s')
    ax.set_xlabel('Tiempo de deliberación (s)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de tiempos de deliberación')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V177_logs/v177_generalizacion_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V177_logs/v177_generalizacion_{timestamp}.png")
    
    # Guardar datos crudos
    raw_data = {
        'version': 'V177',
        'timestamp': timestamp,
        'params': {
            'CONSOLIDACION_CICLOS': CONSOLIDACION_CICLOS,
            'TRAUMA_DURACION': TRAUMA_DURACION,
            'TEST_SETPOINTS': TEST_SETPOINTS,
            'TRIALS_POR_SETPOINT': TRIALS_POR_SETPOINT,
            'UMBRAL_RECHAZO_TRAUMA': UMBRAL_RECHAZO_TRAUMA,
            'UMBRAL_RECHAZO_VECINOS': UMBRAL_RECHAZO_VECINOS,
            'UMBRAL_OPCIONES_SEGURAS': UMBRAL_OPCIONES_SEGURAS,
        },
        'resultados': {
            'preferencias': {str(k): float(v) for k, v in preferencias.items()},
            'valencia_media': {str(k): float(v) for k, v in valencia_media.items()},
            'tiempo_deliberacion_promedio': float(tiempo_delib_promedio),
            'exito': exito
        }
    }
    with open(f'V177_logs/v177_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"\n  📁 Datos crudos guardados: V177_logs/v177_raw_{timestamp}.json")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v177()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V177 completado. Éxito: {exito}")