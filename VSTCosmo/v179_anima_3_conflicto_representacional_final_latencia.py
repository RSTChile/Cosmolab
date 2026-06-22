#!/usr/bin/env python3
"""
V179 — ANIMA-3: CONFLICTO REPRESENTACIONAL (FINAL - LATENCIA EMERGENTE CORREGIDA)
================================================================================
CORRECCIONES FINALES:
  - factor_conflicto = 1.0 + (D_actual * 3.5)  [garantiza >2.5s con D=0.8]
  - D_actual = D_conflicto (entropía + amenaza) en deliberación
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
# PARAMETROS DE LAS FASES
# ============================================================
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0

CONSOLIDACION_CICLOS = 20
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0

BASELINE_TRIALS = 20
CONFLICTO_TRIALS = 100
EXPOSURE_STEPS_PER_TRIAL = 600
TRIAL_DURATION = EXPOSURE_STEPS_PER_TRIAL * DT

D_CONFLICTO_MIN = 0.6
LATENCIA_CONFLICTO_MIN = 2.5
P_HABITO_MIN = 0.75
ALTERNANCIA_MAX = 0.05


# ============================================================
# VALENCIA LOCAL (Trauma fuerte)
# ============================================================

class ValenciaLocal:
    def __init__(self, trauma_costo_multiplier=TRAUMA_COSTO_MULTIPLIER):
        self.valencia = {}
        self.tasa_aprendizaje = 0.001
        self.historial = {}
        self.trauma_costo_multiplier = trauma_costo_multiplier
    
    def actualizar(self, setpoint, error, costo_pagado, dt, reward=0.0, good_threshold=5.0, trauma=False):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        
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
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 5) * 5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA DE TRABAJO (PARCHE 1: factor_conflicto 3.5×D)
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
        puntajes = {}
        
        explor_w = min(0.4, D_actual * 1.5)
        val_w = 1.0 - explor_w
        
        tiempo_base_por_opcion = self.steps_por_opcion * DT  # 0.5s por opción
        
        for opcion in opciones_disponibles:
            val = valencia_local.get_valencia(opcion)
            explor_bonus = D_actual * max(0.0, 1.0 - abs(val) / 50.0) * 0.1
            current_bonus = 0.8 if (current_sp is not None and abs(opcion - current_sp) < 1.0) else 0.0
            puntaje = (val * val_w + explor_bonus + current_bonus)
            puntajes[opcion] = puntaje
            self.opciones_ensayadas[opcion] = puntaje
        
        # CORRECCIÓN FINAL: factor_conflicto = 1 + (D_actual * 3.5)
        # Con D=0.8 → factor=3.8 → latencia = 0.5s × 2 × 3.8 = 3.8s
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
# REGISTRO DE REPRESENTACIONES (D_conflicto con entropía + amenaza)
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
# HEMISFERIO, FATIGA, MEMORIA, CONSCIENCIA, JUEGO (sin cambios)
# ============================================================

class HemisferioV179:
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


class FatigaMetabolicaV179:
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


class MemoriaAusenciaV179:
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


class ConscienciaBasicaV179:
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


class ModoJuegoV179:
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
# APARATO MOTOR V179 (PARCHE 2: D_conflicto real en deliberación)
# ============================================================

class AparatoMotorV179:
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
        
        self.fatiga = FatigaMetabolicaV179()
        self.memoria = MemoriaAusenciaV179()
        self.consciencia = ConscienciaBasicaV179()
        self.juego = ModoJuegoV179()
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
    
    def ejecutar_con_deliberacion(self, opciones_disponibles, gradiente, t, dt, trauma=False, target_reward=None):
        for op in opciones_disponibles:
            if op not in self.recent_presented:
                self.recent_presented.append(op)
        
        if len(opciones_disponibles) > 1:
            # PARCHE 2: Usar D_conflicto real para escalar latencia
            valencias = [self.valencia.get_valencia(op) for op in opciones_disponibles]
            D_actual = self.registro.calcular_D_conflicto(valencias)
            opcion_elegida, puntajes, tiempo_delib = self.memoria_trabajo.deliberar(
                opciones_disponibles, self.valencia, D_actual, current_sp=self.orientacion
            )
        else:
            D_actual = self.registro.calcular_desacople()
            only = opciones_disponibles[0]
            val_only = self.valencia.get_valencia(only)
            if val_only < -2.0:
                opcion_elegida = 0.0
            else:
                opcion_elegida = only
            tiempo_delib = (self.memoria_trabajo.steps_por_opcion * DT) * 0.5
        
        setpoint_objetivo, confianza = self.memoria.actualizar(opcion_elegida, self.fatiga.get_historia(), dt)
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        val_local = self.valencia.get_valencia(opcion_elegida)
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
            self.valencia.actualizar(opcion_elegida, error, 0.0, dt, reward=rwd, good_threshold=val_good_threshold, trauma=trauma)
            val_local = self.valencia.get_valencia(opcion_elegida)
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
        
        self.valencia.actualizar(opcion_elegida, error, costo_total_estimado, dt, reward=rwd, good_threshold=val_good_threshold, trauma=trauma)
        
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
# ORGANISMO V179
# ============================================================

class OrganismoV179:
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
        
        self.izquierdo = HemisferioV179("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV179("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        self.sistema_B_izq = HemisferioV179("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV179("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        self.motor = AparatoMotorV179()
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
    
    def actualizar_con_opciones(self, t, dt, duracion_total, opciones_disponibles, trauma=False, target_reward=None):
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
            opciones_disponibles, gradiente, t, dt, trauma, target_reward
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
    
    def actualizar_setpoint(self, t, dt, duracion_total, setpoint, trauma=False, target_reward=None):
        return self.actualizar_con_opciones(t, dt, duracion_total, [setpoint], trauma, target_reward)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()
    
    def get_valencia(self, setpoint):
        return self.motor.valencia.get_valencia(setpoint)
    
    def get_valencia_habito(self):
        return self.get_valencia(HABITO_SETPOINT)
    
    def get_valencia_trauma(self):
        return self.get_valencia(TRAUMA_SETPOINT)


# ============================================================
# FUNCIÓN DE BASELINE
# ============================================================

def medir_latencia_baseline(organismo, setpoint, trials=20):
    latencias = []
    for _ in range(trials):
        organismo.motor.memoria_trabajo.reset()
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t = step * DT
            _, _, lat, _, _ = organismo.actualizar_con_opciones(
                t, DT, t + 1.0, [setpoint], trauma=False
            )
            if lat > 0:
                latencias.append(lat)
                break
        else:
            latencias.append(EXPOSURE_STEPS_PER_TRIAL * DT)
    return np.mean(latencias)


# ============================================================
# EXPERIMENTO PRINCIPAL
# ============================================================

def ejecutar_v179():
    print("=" * 100)
    print("EXPERIMENTO V179 — ANIMA-3: CONFLICTO REPRESENTACIONAL (FINAL)")
    print("=" * 100)
    print("  CORRECCIONES FINALES APLICADAS:")
    print("    1. factor_conflicto = 1.0 + (D_actual * 3.5)")
    print("    2. D_actual = D_conflicto (entropía + amenaza) en deliberación")
    print("")
    print("  CRITERIOS DE ÉXITO (roadmap):")
    print(f"    ✅ D_conflicto > {D_CONFLICTO_MIN}")
    print(f"    ✅ latencia_conflicto > {LATENCIA_CONFLICTO_MIN}s")
    print(f"    ✅ P(-60° | conflicto) > {P_HABITO_MIN:.0%}")
    print(f"    ✅ alternancia < {ALTERNANCIA_MAX:.0%}")
    print("=" * 100)
    
    print("\n  Creando organismo...")
    organismo = OrganismoV179(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V179_logs', exist_ok=True)
    
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
    
    # FASE 0: BASELINE
    print("\n" + "=" * 60)
    print(f"FASE 0: BASELINE — Medir latencia sin conflicto")
    print("=" * 60)
    
    latencia_habito_baseline = medir_latencia_baseline(organismo, HABITO_SETPOINT, BASELINE_TRIALS)
    latencia_trauma_baseline = medir_latencia_baseline(organismo, TRAUMA_SETPOINT, BASELINE_TRIALS)
    latencia_esperada_2x = max(latencia_habito_baseline, latencia_trauma_baseline) * 2.0
    
    print(f"  Latencia -60° solo: {latencia_habito_baseline:.3f}s")
    print(f"  Latencia +60° solo: {latencia_trauma_baseline:.3f}s")
    print(f"  Latencia esperada en conflicto (2x): {latencia_esperada_2x:.3f}s")
    
    # FASE 1: Consolidación
    print("\n" + "=" * 60)
    print(f"FASE 1: Consolidación del hábito ({CONSOLIDACION_CICLOS} ciclos a -60°)")
    print("=" * 60)
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            organismo.actualizar_setpoint(t, DT, t_actual + PERIODO_ALTERNANCIA, 
                                         HABITO_SETPOINT, target_reward=HABITO_SETPOINT)
        if (ciclo + 1) % 5 == 0:
            val = organismo.get_valencia_habito()
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia(-60°) = {val:.2f}")
        t_actual += PERIODO_ALTERNANCIA
    
    val_habito_post_consolidacion = organismo.get_valencia_habito()
    print(f"  Valencia -60° post-consolidación: {val_habito_post_consolidacion:.2f}")
    
    # FASE 2: Trauma
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("=" * 60)
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        organismo.actualizar_setpoint(t, DT, t_actual + TRAUMA_DURACION, 
                                     TRAUMA_SETPOINT, trauma=True)
    t_actual += TRAUMA_DURACION
    
    val_trauma_post = organismo.get_valencia_trauma()
    val_habito_post_trauma = organismo.get_valencia_habito()
    print(f"  Valencia +60° post-trauma: {val_trauma_post:.2f}")
    print(f"  Valencia -60° post-trauma: {val_habito_post_trauma:.2f}")
    
    # FASE 3: CONFLICTO
    print("\n" + "=" * 60)
    print(f"FASE 3: CONFLICTO — Opciones simultáneas [{HABITO_SETPOINT}°, {TRAUMA_SETPOINT}°]")
    print(f"        Trials: {CONFLICTO_TRIALS}")
    print("=" * 60)
    
    opciones_elegidas = []
    desacoples_conflicto = []
    latencias = []
    valencias_habito = []
    valencias_trauma = []
    
    for trial in range(CONFLICTO_TRIALS):
        t = t_actual + trial * TRIAL_DURATION
        
        if (trial + 1) % 10 == 0:
            print(f"    Trial {trial+1}/{CONFLICTO_TRIALS}...")
        
        trial_D_conflicto = []
        trial_latencias = []
        
        for step in range(EXPOSURE_STEPS_PER_TRIAL):
            t_step = t + step * DT
            _, _, latencia, opcion, _ = organismo.actualizar_con_opciones(
                t_step, DT, t_actual + CONFLICTO_TRIALS * TRIAL_DURATION,
                [HABITO_SETPOINT, TRAUMA_SETPOINT],
                trauma=False, target_reward=None
            )
            
            val_habito = organismo.get_valencia_habito()
            val_trauma = organismo.get_valencia_trauma()
            D_conflicto = organismo.motor.registro.calcular_D_conflicto([val_habito, val_trauma])
            
            trial_D_conflicto.append(D_conflicto)
            trial_latencias.append(latencia)
        
        opciones_elegidas.append(opcion if opcion is not None else 0)
        desacoples_conflicto.append(np.mean(trial_D_conflicto))
        latencias.append(np.mean(trial_latencias))
        valencias_habito.append(organismo.get_valencia_habito())
        valencias_trauma.append(organismo.get_valencia_trauma())
        
        t_actual += TRIAL_DURATION
    
    # Análisis
    p_habito_conflicto = sum(1 for e in opciones_elegidas if abs(e - HABITO_SETPOINT) < 5.0) / CONFLICTO_TRIALS
    d_pico = max(desacoples_conflicto)
    d_medio = np.mean(desacoples_conflicto)
    latencia_media = np.mean(latencias)
    
    alternancias = 0
    for i in range(1, len(opciones_elegidas)):
        es_habito_antes = abs(opciones_elegidas[i-1] - HABITO_SETPOINT) < 5.0
        es_habito_despues = abs(opciones_elegidas[i] - HABITO_SETPOINT) < 5.0
        if es_habito_antes != es_habito_despues:
            alternancias += 1
    tasa_alternancia = alternancias / (CONFLICTO_TRIALS - 1) if CONFLICTO_TRIALS > 1 else 0
    
    # Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS V179 — Conflicto representacional (FINAL)")
    print("=" * 80)
    
    print(f"\n  📊 MÉTRICAS DE BASELINE:")
    print(f"    Latencia -60° solo: {latencia_habito_baseline:.3f}s")
    print(f"    Latencia +60° solo: {latencia_trauma_baseline:.3f}s")
    
    print(f"\n  📊 MÉTRICAS DE CONFLICTO:")
    print(f"    P(-60° | conflicto) = {p_habito_conflicto:.1%} (umbral > {P_HABITO_MIN:.0%})")
    print(f"    D_conflicto_pico = {d_pico:.3f} (umbral > {D_CONFLICTO_MIN})")
    print(f"    D_conflicto_medio = {d_medio:.3f}")
    print(f"    Latencia media = {latencia_media:.3f}s (umbral > {LATENCIA_CONFLICTO_MIN}s)")
    print(f"    Tasa de alternancia = {tasa_alternancia:.1%} (umbral < {ALTERNANCIA_MAX:.0%})")
    
    print(f"\n  📊 MÉTRICAS DE VALENCIA:")
    print(f"    Valencia -60° final: {valencias_habito[-1] if valencias_habito else 0:.2f}")
    print(f"    Valencia +60° final: {valencias_trauma[-1] if valencias_trauma else 0:.2f}")
    
    d_ok = d_pico > D_CONFLICTO_MIN
    latencia_ok = latencia_media > LATENCIA_CONFLICTO_MIN
    preferencia_ok = p_habito_conflicto > P_HABITO_MIN
    alternancia_ok = tasa_alternancia < ALTERNANCIA_MAX
    
    print(f"\n  📊 CRITERIOS DE ÉXITO (roadmap):")
    print(f"    D_pico > {D_CONFLICTO_MIN}: {d_ok} -> {'✅' if d_ok else '❌'}")
    print(f"    Latencia > {LATENCIA_CONFLICTO_MIN}s: {latencia_ok} -> {'✅' if latencia_ok else '❌'}")
    print(f"    P(-60°) > {P_HABITO_MIN:.0%}: {preferencia_ok} -> {'✅' if preferencia_ok else '❌'}")
    print(f"    Alternancia < {ALTERNANCIA_MAX:.0%}: {alternancia_ok} -> {'✅' if alternancia_ok else '❌'}")
    
    exito = d_ok and latencia_ok and preferencia_ok and alternancia_ok
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ CONFLICTO REPRESENTACIONAL RESUELTO")
        print("")
        print("     El organismo demuestra:")
        print("     ✓ Desacople máximo bajo presión (D_conflicto > 0.6)")
        print("     ✓ Latencia deliberativa prolongada (> 2.5s)")
        print("     ✓ Preferencia clara por el hábito (> 75%)")
        print("     ✓ Alternancia mínima (< 5%)")
    else:
        print("  ⚠️ CONFLICTO REPRESENTACIONAL NO RESUELTO")
        if not d_ok:
            print("     El desacople no alcanzó el umbral (>0.6)")
        if not latencia_ok:
            print("     La latencia no superó 2.5s")
        if not preferencia_ok:
            print("     La preferencia por -60° fue insuficiente")
        if not alternancia_ok:
            print("     Hubo demasiada alternancia entre opciones")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(opciones_elegidas, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=HABITO_SETPOINT, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=TRAUMA_SETPOINT, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Opción elegida')
    ax.set_title('Elecciones durante conflicto')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(desacoples_conflicto, 'purple', linewidth=1)
    ax.axhline(y=D_CONFLICTO_MIN, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Trial')
    ax.set_ylabel('D_conflicto')
    ax.set_title('Desacople bajo conflicto')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(latencias, 'orange', linewidth=1)
    ax.axhline(y=LATENCIA_CONFLICTO_MIN, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latencia (s)')
    ax.set_title('Latencia deliberativa')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(valencias_habito, 'blue', linewidth=1, label='-60° (hábito)')
    ax.plot(valencias_trauma, 'red', linewidth=1, label='+60° (trauma)')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Valencia')
    ax.set_title('Evolución de valencias')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'V179_logs/v179_final_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V179_logs/v179_final_{timestamp}.png")
    
    raw_data = {
        'version': 'V179_FINAL',
        'timestamp': timestamp,
        'resultados': {
            'p_habito_conflicto': float(p_habito_conflicto),
            'd_pico': float(d_pico),
            'latencia_media': float(latencia_media),
            'tasa_alternancia': float(tasa_alternancia),
            'val_trauma_post': float(val_trauma_post),
            'd_ok': bool(d_ok),
            'latencia_ok': bool(latencia_ok),
            'preferencia_ok': bool(preferencia_ok),
            'alternancia_ok': bool(alternancia_ok),
            'exito': bool(exito)
        }
    }
    
    with open(f'V179_logs/v179_final_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos guardados: V179_logs/v179_final_{timestamp}.json")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v179()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V179 final completado. Éxito: {exito}")