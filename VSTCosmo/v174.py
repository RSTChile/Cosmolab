#!/usr/bin/env python3
"""
V174 — ANIMA-2: COMPARTIMENTALIZACIÓN DE VALENCIA (CORREGIDO)
================================================================================
Objetivo: Demostrar que ANIMA-2 puede mantener valencia local diferenciada
          incluso cuando Cb_global está elevada (estrés, fatiga, incertidumbre).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque
import random

# ============================================================
# PARAMETROS BASE (DEFINIDOS PRIMERO)
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
# PARAMETROS RITUAL (DEFINIDOS ANTES DE LA CLASE)
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
# PARAMETROS DE COMPARTIMENTALIZACIÓN
# ============================================================
SETPOINTS_POSIBLES = [-60.0, -30.0, 0.0, 30.0, 60.0]
SETPOINTS_TEST = [-60.0, 0.0, 60.0]  # Solo tres para el test

# FASE 0: Consolidación del hábito (Val positiva)
CONSOLIDACION_CICLOS = 100
CONSOLIDACION_REWARD = 10.0
CONSOLIDACION_OBJETIVO_VAL = 10.0    # Val(-60°) debe ser > 10

# FASE 1: Trauma específico suave
TRAUMA_SETPOINT = 60.0
TRAUMA_COSTO_MULTIPLIER = 2.0
TRAUMA_DURACION = 15.0
TRAUMA_ANCLAJE_FRECUENCIA = 3.0
TRAUMA_REWARD_ANCLAJE = 5.0

# FASE 2: Test de compartimentalización
TEST_DURACION = 30.0                 # 30 segundos de libre elección
TEST_CAMBIO_INTERVALO = 10.0         # Cambiar setpoint cada 10s

# Umbrales de éxito
UMBRAL_VAL_POSITIVA = 5.0            # Val(-60°) > 5
UMBRAL_VAL_DIFERENCIAL = 2.0         # Val(-60°) - Val(+60°) > 2
UMBRAL_TASA_ACCION = 0.3
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 2.0


# ============================================================
# VALENCIA LOCAL COMPARTIMENTALIZADA
# ============================================================

class ValenciaCompartimentalizada:
    """
    Valencia local que persiste independientemente de Cb_global.
    Implementa memoria lenta por representación.
    """
    
    def __init__(self, setpoints_posibles, tasa_aprendizaje=0.01, tasa_decaimiento=0.999):
        self.valencia = {sp: 0.0 for sp in setpoints_posibles}
        self.tasa_aprendizaje = tasa_aprendizaje
        self.tasa_decaimiento = tasa_decaimiento
        self.historial_valencia = {sp: [] for sp in setpoints_posibles}
        self.exitos = {sp: 0 for sp in setpoints_posibles}
        self.fracasos = {sp: 0 for sp in setpoints_posibles}
    
    def actualizar(self, setpoint_actual, error, reward=0.0, es_trauma=False, dt=DT):
        """
        Actualiza la valencia local basada en experiencia.
        La valencia es MEMORIA LENTA (no se resetea con Cb_global).
        """
        key = round(setpoint_actual / 10) * 10 if setpoint_actual != 0 else 0
        
        # Decaimiento gradual (olvido lento)
        self.valencia[key] *= self.tasa_decaimiento
        
        # Recompensa por éxito (error bajo)
        if abs(error) < 5.0:
            self.exitos[key] += 1
            self.valencia[key] += reward * self.tasa_aprendizaje * dt
        else:
            self.fracasos[key] += 1
            # Penalización por error
            self.valencia[key] -= abs(error) * self.tasa_aprendizaje * dt * 0.1
        
        # Trauma reduce valencia específicamente
        if es_trauma:
            self.valencia[key] -= TRAUMA_COSTO_MULTIPLIER * self.tasa_aprendizaje * dt
        
        # Limitar rango
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial_valencia[key].append(self.valencia[key])
        
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def get_tasa_exito(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        total = self.exitos[key] + self.fracasos[key]
        if total == 0:
            return 0.5
        return self.exitos[key] / total
    
    def reset(self):
        for sp in self.valencia:
            self.valencia[sp] = 0.0
            self.historial_valencia[sp] = []
            self.exitos[sp] = 0
            self.fracasos[sp] = 0


# ============================================================
# Cb GLOBAL (separada de valencia local)
# ============================================================

class CbGlobal:
    """
    Cb global mide presión de desacople general.
    No debe contaminar la valencia local.
    """
    
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial = []
    
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        self.historial.append(self.Cb)
        return self.Cb
    
    def reset(self):
        self.Cb = 0.0
        self.historial = []
    
    def get(self):
        return self.Cb


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV174:
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

class FatigaMetabolicaV174:
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

class MemoriaAusenciaV174:
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
# MODO JUEGO
# ============================================================

class ModoJuegoV174:
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
    
    def aplicar(self, delta_raw, trauma_mode=False, reward_mode=False):
        if self.activo:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
            if trauma_mode:
                delta_costo *= TRAUMA_COSTO_MULTIPLIER
            if reward_mode:
                delta_costo = -abs(delta_raw) * CONSOLIDACION_REWARD
        else:
            delta_fisico = delta_raw
            delta_costo = abs(delta_raw)
            if trauma_mode:
                delta_costo *= TRAUMA_COSTO_MULTIPLIER
            if reward_mode:
                delta_costo = -abs(delta_raw) * CONSOLIDACION_REWARD
        
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

class RitualV174:
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
        self.historial_setpoints = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada, setpoint):
        if self.ruido_sigma > 0:
            representacion_ruidosa = representacion + np.random.normal(0, self.ruido_sigma)
        else:
            representacion_ruidosa = representacion
        
        self.historial_representaciones.append(representacion_ruidosa)
        self.historial_acciones.append(accion_ejecutada)
        self.historial_setpoints.append(setpoint)
    
    def calcular_probabilidad_eleccion(self, setpoint_value):
        if len(self.historial_setpoints) < 10:
            return 0.5
        
        ocurrencias = []
        for sp, acc in zip(self.historial_setpoints, self.historial_acciones):
            if abs(sp - setpoint_value) < 5.0:
                ocurrencias.append(acc)
        
        if len(ocurrencias) == 0:
            return 0.5
        return np.mean(ocurrencias)
    
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
        self.historial_setpoints.clear()


# ============================================================
# APARATO MOTOR V174
# ============================================================

class AparatoMotorV174:
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
        
        self.fatiga = FatigaMetabolicaV174()
        self.memoria = MemoriaAusenciaV174()
        self.cb_global = CbGlobal()
        self.juego = ModoJuegoV174()
        self.ritual = RitualV174()
        
        # Valencia compartimentalizada (separada de Cb)
        self.valencia = ValenciaCompartimentalizada(SETPOINTS_POSIBLES)
        
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
               modo_trauma=False, modo_reward=False):
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
        
        # Cb GLOBAL (presión de desacople)
        Cb = self.cb_global.actualizar(e_R, A_sys_env, dt)
        
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        else:
            juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # Actualizar valencia LOCAL (separada de Cb)
        reward_val = CONSOLIDACION_REWARD if modo_reward else 0.0
        self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                  error, reward=reward_val, 
                                  es_trauma=modo_trauma, dt=dt)
        
        accion_ejecutada = abs(self.ultimo_delta) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, 
                                setpoint_raw if setpoint_raw is not None else 0)
        
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
        
        delta_fisico, delta_costo = self.juego.aplicar(delta, modo_trauma, modo_reward)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
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
        self.cb_global.reset()
        self.juego.reset()
        self.ritual.reset()
        self.valencia.reset()
        self.registro.reset()


# ============================================================
# SISTEMA V174
# ============================================================

class SistemaV174:
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

        self.izquierdo = HemisferioV174("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV174("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV174("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV174("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV174()
        self.modo_entrenamiento = True

    def actualizar(self, t, dt, duracion_total, setpoint_real, 
                   modo_trauma=False, modo_reward=False):
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
            gradiente, LF_activa, True, t, setpoint_real, dt, modo_trauma, modo_reward
        )
        
        return (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()
    
    def get_valencia(self, setpoint):
        return self.motor.valencia.get_valencia(setpoint)
    
    def get_cb_global(self):
        return self.motor.cb_global.get()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def setpoint_normal(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def setpoint_test_alternante(t, intervalo=TEST_CAMBIO_INTERVALO,
                              posibles=SETPOINTS_TEST):
    fase = int(t / intervalo) % len(posibles)
    return posibles[fase]


def generar_setpoint_con_ruido(t, setpoint_func, **kwargs):
    setpoint_base = setpoint_func(t, **kwargs)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


# ============================================================
# EXPERIMENTO V174 — COMPARTIMENTALIZACIÓN DE VALENCIA
# ============================================================

def ejecutar_v174():
    print("=" * 100)
    print("EXPERIMENTO V174 — ANIMA-2: COMPARTIMENTALIZACIÓN DE VALENCIA")
    print("=" * 100)
    print("  Objetivo: Demostrar que Val_local persiste aunque Cb_global esté alta")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    1. Valencia positiva: Val(-60°) > {UMBRAL_VAL_POSITIVA}")
    print(f"    2. Valencia diferencial: Val(-60°) - Val(+60°) > {UMBRAL_VAL_DIFERENCIAL}")
    print(f"    3. No abstención: P(acción en -60°) > {UMBRAL_TASA_ACCION}")
    print(f"    4. Desacople: D > {UMBRAL_DESACOPLE} por ≥ {TIEMPO_MINIMO_DESACOPLE}s")
    print("=" * 100)

    organismo = SistemaV174("V174", seed=SEMILLA_BASE)

    print("\n" + "=" * 80)
    print("FASE 0: Consolidación del hábito (Val positiva)")
    print("=" * 80)
    print(f"  Duración: {CONSOLIDACION_CICLOS} ciclos")
    print(f"  Reward: +{CONSOLIDACION_REWARD} por movimiento exitoso")
    print(f"  Objetivo: Val(-60°) > {UMBRAL_VAL_POSITIVA}")
    
    organismo.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    organismo.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # Consolidación con reward
    val_habito_vals = []
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = -60.0
            (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
             juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
                t, DT, t_actual + PERIODO_ALTERNANCIA, setpoint, modo_reward=True)
            
            if i % 500 == 0:
                val_habito_vals.append(organismo.get_valencia(-60.0))
        
        t_actual += PERIODO_ALTERNANCIA
        if (ciclo + 1) % 20 == 0:
            val_actual = organismo.get_valencia(-60.0)
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, Val(-60°) = {val_actual:.1f}")
    
    val_habito_final = organismo.get_valencia(-60.0)
    print(f"  Consolidación completada. Val(-60°) final: {val_habito_final:.1f}")
    
    print("\n" + "=" * 80)
    print("FASE 1: Trauma específico suave")
    print("=" * 80)
    print(f"  Setpoint forzado a +{TRAUMA_SETPOINT:.0f}° por {TRAUMA_DURACION}s")
    print(f"  Costo multiplicado por {TRAUMA_COSTO_MULTIPLIER}x")
    print(f"  Anclaje: cada {TRAUMA_ANCLAJE_FRECUENCIA}s, setpoint -60° con reward")
    
    trauma_datos = {'Cb': [], 'valencia_trauma': [], 'valencia_habito': []}
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
            t, DT, t_actual + TRAUMA_DURACION, setpoint, modo_trauma=trauma, modo_reward=reward)
        
        trauma_datos['Cb'].append(Cb)
        trauma_datos['valencia_trauma'].append(organismo.get_valencia(TRAUMA_SETPOINT))
        trauma_datos['valencia_habito'].append(organismo.get_valencia(-60.0))
    
    t_actual += TRAUMA_DURACION
    
    val_trauma_final = organismo.get_valencia(TRAUMA_SETPOINT)
    val_habito_post_trauma = organismo.get_valencia(-60.0)
    print(f"  Valencia +60° final: {val_trauma_final:.1f}")
    print(f"  Valencia -60° post-trauma: {val_habito_post_trauma:.1f}")
    
    print("\n" + "=" * 80)
    print("FASE 2: Test de compartimentalización")
    print("=" * 80)
    print(f"  Setpoints: {SETPOINTS_TEST}")
    print(f"  Duración: {TEST_DURACION}s")
    print("  (Medimos si Val local persiste independientemente de Cb_global)")
    
    test_datos = {'setpoint': [], 'Cb': [], 'D': [], 'accion': [], 
                  'valencia': {}, 'setpoint_presentado': []}
    
    for sp in SETPOINTS_TEST:
        test_datos['valencia'][sp] = []
    
    for i in range(int(TEST_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = setpoint_test_alternante(t, intervalo=TEST_CAMBIO_INTERVALO,
                                             posibles=SETPOINTS_TEST)
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + TEST_DURACION, setpoint)
        
        test_datos['setpoint'].append(setpoint)
        test_datos['Cb'].append(Cb)
        test_datos['D'].append(D)
        test_datos['setpoint_presentado'].append(setpoint)
        
        # Determinar acción ejecutada
        if 'ultima_orient' in test_datos:
            delta = abs(orient - test_datos['ultima_orient'])
            test_datos['accion'].append(delta > 0.5)
        else:
            test_datos['accion'].append(False)
        test_datos['ultima_orient'] = orient
        
        # Registrar valencias
        for sp in SETPOINTS_TEST:
            test_datos['valencia'][sp].append(organismo.get_valencia(sp))
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V174 — Compartimentalización de valencia")
    print("=" * 80)
    
    # Calcular valencia media por setpoint
    valencia_media = {}
    for sp in SETPOINTS_TEST:
        valencia_media[sp] = np.mean(test_datos['valencia'][sp]) if test_datos['valencia'][sp] else 0
    
    # Calcular probabilidad de acción por setpoint
    accion_por_setpoint = {}
    for sp in SETPOINTS_TEST:
        sp_indices = [i for i, s in enumerate(test_datos['setpoint_presentado']) if abs(s - sp) < 5.0]
        if sp_indices:
            accion_por_setpoint[sp] = np.mean([test_datos['accion'][i] for i in sp_indices])
        else:
            accion_por_setpoint[sp] = 0.0
    
    # Valencia diferencial
    val_habito = valencia_media.get(-60.0, 0)
    val_trauma = valencia_media.get(60.0, 0)
    val_diferencial = val_habito - val_trauma
    
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
    
    # Cb global (últimos valores)
    Cb_final = test_datos['Cb'][-1] if test_datos['Cb'] else 0
    Cb_media = np.mean(test_datos['Cb']) if test_datos['Cb'] else 0
    
    print(f"\n  📊 VALENCIA LOCAL POR SETPOINT:")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == 60.0 else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: Val = {valencia_media[sp]:.2f}, P(acción) = {accion_por_setpoint[sp]:.3f}{marker}")
    
    print(f"\n  📊 ESTADO GLOBAL:")
    print(f"    Cb global media: {Cb_media:.1f}")
    print(f"    Cb global final: {Cb_final:.1f}")
    print(f"    Desacople sostenido: {max_tiempo_desacople:.2f}s > {TIEMPO_MINIMO_DESACOPLE}s -> {'✅' if desacople_sostenido else '❌'}")
    
    print(f"\n  📊 MÉTRICAS CLAVE:")
    print(f"    Valencia positiva: Val(-60°) = {val_habito:.2f} > {UMBRAL_VAL_POSITIVA} -> {'✅' if val_habito > UMBRAL_VAL_POSITIVA else '❌'}")
    print(f"    Valencia diferencial: Val(-60°) - Val(+60°) = {val_diferencial:.2f} > {UMBRAL_VAL_DIFERENCIAL} -> {'✅' if val_diferencial > UMBRAL_VAL_DIFERENCIAL else '❌'}")
    print(f"    No abstención: P(acción -60°) = {accion_por_setpoint.get(-60.0, 0):.3f} > {UMBRAL_TASA_ACCION} -> {'✅' if accion_por_setpoint.get(-60.0, 0) > UMBRAL_TASA_ACCION else '❌'}")
    
    exito = (val_habito > UMBRAL_VAL_POSITIVA and
             val_diferencial > UMBRAL_VAL_DIFERENCIAL and
             accion_por_setpoint.get(-60.0, 0) > UMBRAL_TASA_ACCION and
             desacople_sostenido)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ COMPARTIMENTALIZACIÓN DE VALENCIA DEMOSTRADA")
        print("")
        print("     ANIMA-2 demuestra:")
        print("     ✓ Val(-60°) > 0 (el hábito es valorado positivamente)")
        print("     ✓ Val(-60°) > Val(+60°) (trauma reconocido)")
        print("     ✓ El sistema sigue actuando en opción segura")
        print("     ✓ Desacople sostenido durante la evaluación")
        print("")
        print("  Siguiente paso: V175 — Primer 'No' operativo (negación específica)")
    else:
        print("  ⚠️ COMPARTIMENTALIZACIÓN DE VALENCIA NO DEMOSTRADA")
        if val_habito <= UMBRAL_VAL_POSITIVA:
            print("     El hábito no alcanzó valencia positiva")
        if val_diferencial <= UMBRAL_VAL_DIFERENCIAL:
            print("     No hay valencia diferencial suficiente")
        if accion_por_setpoint.get(-60.0, 0) <= UMBRAL_TASA_ACCION:
            print("     El sistema no actúa en opción segura")
        if not desacople_sostenido:
            print("     Desacople insuficiente")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Valencia por setpoint
    ax = axes[0, 0]
    sps = list(valencia_media.keys())
    vals = list(valencia_media.values())
    colors = ['red' if sp == 60.0 else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), vals, color=colors)
    ax.axhline(y=UMBRAL_VAL_POSITIVA, color='green', linestyle='--', alpha=0.5, label=f'Umbral positivo')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Valencia local')
    ax.set_title('Valencia por setpoint (compartimentalizada)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Cb global durante test
    ax = axes[0, 1]
    ax.plot(test_datos['Cb'], 'orange', linewidth=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb global')
    ax.set_title('Cb global (presión de desacople)')
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
    ax.set_title('Probabilidad de ejecución')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Evolución de valencia durante consolidación
    ax = axes[1, 1]
    ax.plot(val_habito_vals, 'green', linewidth=0.5)
    ax.axhline(y=UMBRAL_VAL_POSITIVA, color='blue', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Val(-60°)')
    ax.set_title('FASE 0: Consolidación de valencia positiva')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Valencia durante trauma
    ax = axes[1, 2]
    ax.plot(trauma_datos['valencia_trauma'], 'red', linewidth=0.5, label='Valencia +60°')
    ax.plot(trauma_datos['valencia_habito'], 'green', linewidth=0.5, label='Valencia -60°')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Valencia')
    ax.set_title('FASE 1: Evolución de valencia durante trauma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v174_logs', exist_ok=True)
    plt.savefig(f'v174_logs/v174_compartimentalizacion_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v174_logs/v174_compartimentalizacion_{timestamp}.png")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v174()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V174 completado. Compartimentalización demostrada: {exito}")