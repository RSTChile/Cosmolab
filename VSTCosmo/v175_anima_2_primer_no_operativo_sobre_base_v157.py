#!/usr/bin/env python3
"""
V175 — ANIMA-2: PRIMER "NO" OPERATIVO (Sobre base V157)
================================================================================
Lecciones aprendidas de V158-V174:
  1. No se puede forzar valencia con recompensas externas
  2. Cb debe ser local por representación, no global
  3. El trauma debe ser específico, no contaminar todo el sistema
  4. La evaluación no debe ser forzada (abstención garantizada)
  5. La fatiga saturada mata la plasticidad

Diseño conservador sobre base V157:
  - Se mantiene toda la dinámica original (hemisferios, plasticidad, juego)
  - Se AÑADE: Tracker de valencia local por setpoint
  - Se MODIFICA: Umbral de juego y condiciones de activación
  - NO se fuerza evaluación (el organismo elige libremente)

CRITERIOS DE ÉXITO:
  1. El organismo con memoria aversiva rechaza +60° (P(ejecutar) < 0.2)
  2. El organismo mantiene acción en -60° (P(ejecutar) > 0.4)
  3. Diferencial de valencia detectable (val(-60°) - val(+60°) > umbral)  # trauma penaliza +60
  4. Desacople sostenido (D > 0.1 por >2s)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from collections import deque
import copy

# ============================================================
# PARAMETROS (DESDE V157, FUNCIONAN)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10

# Asimetria forzada
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta base
ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MAX = 15.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Inercia
INERCIA = 0.95
SENSIBILIDAD_GRAD = 10.0

# Fatiga (ajustada para evitar saturación prematura)
K_GAIN = 0.00015                     # Reducido a la mitad para más ventana
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

# Memoria de ausencia
TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2
K_HOLD = 0.001

# Consciencia básica
TAU_CB = 10.0
CB_MAX = 500.0

# Juego enactuado (parámetros ajustados)
LAMBDA_FISICO = 0.15                   # Aumentado para más efecto
LAMBDA_COSTO = 0.5                     # Reducido para asimetría
UMBRAL_CB_JUEGO = 40.0                 # Aumentado para activación más selectiva
K_INFLUENCIA_JUEGO = 0.0005

# Ruido para forzar corrección
RUIDO_SETPOINT_AMP = 5.0
RUIDO_SETPOINT_PERIODO = 10.0

# Semilla base
SEMILLA_BASE = 44

# Período
PERIODO_ALTERNANCIA = 80.0

# ============================================================
# PARAMETROS NUEVOS: VALENCIA LOCAL Y TRAUMA
# ============================================================
TRAUMA_SETPOINT = 60.0
TRAUMA_DURACION = 15.0
TRAUMA_COSTO_MULTIPLIER = 2.0

CONSOLIDACION_CICLOS = 30             # Consolidación de hábito
TEST_DURACION = 30.0                  # Test de elección libre
SETPOINTS_TEST = [-60.0, -30.0, 0.0, 30.0, 60.0]

# Umbrales
UMBRAL_VALENCIA_DIFERENCIAL = 50.0
UMBRAL_RECHAZO = 0.2
UMBRAL_ACCION_SEGURA = 0.4
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 2.0


# ============================================================
# HEMISFERIO (IDÉNTICO A V157)
# ============================================================

class HemisferioV175:
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
# FATIGA METABOLICA (CON K_GAIN REDUCIDO)
# ============================================================

class FatigaMetabolicaV175:
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


# ============================================================
# MEMORIA DE AUSENCIA (IDÉNTICO A V157)
# ============================================================

class MemoriaAusenciaV175:
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


# ============================================================
# CONSCIENCIA BÁSICA (IDÉNTICO A V157)
# ============================================================

class ConscienciaBasicaV175:
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


# ============================================================
# MODO JUEGO (CON PARÁMETROS AJUSTADOS)
# ============================================================

class ModoJuegoV175:
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


# ============================================================
# VALENCIA LOCAL (NUEVO - MEMORIA LENTA POR SETPOINT)
# ============================================================

class ValenciaLocal:
    """
    Memoria de valencia por representación (setpoint).
    Se actualiza lentamente y persiste independientemente de Cb global.
    """
    
    def __init__(self):
        self.valencia = {}  # dict: setpoint_key -> valencia acumulada
        self.tasa_aprendizaje = 0.001
        self.historial = {}
    
    def actualizar(self, setpoint, error, costo_pagado, dt, good_threshold=5.0):
        """
        Actualiza valencia local basada en experiencia.
        El umbral "good" ahora puede provenir del organismo (e.g. su zona_muerta actual),
        para que el aprendizaje de valencia emerja de sus propios estados internos.
        """
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        
        # Versión continua para que el aprendizaje de valencia emerja de forma más suave
        # del propio estado del organismo (good_threshold = su zona_muerta actual).
        # Recompensa proporcional a qué tan "dentro de tolerancia" está el error.
        # Penalización por error y por costo.
        rel_good = max(0.0, (good_threshold - abs(error)) / max(good_threshold, 0.01))
        self.valencia[key] += self.tasa_aprendizaje * dt * (20.0 * rel_good - abs(error) * 0.3 - costo_pagado * 0.05)
        
        # Limitar rango
        self.valencia[key] = max(-100.0, min(100.0, self.valencia[key]))
        
        self.historial[key].append(self.valencia[key])
        
        return self.valencia[key]
    
    def get_valencia(self, setpoint):
        key = round(setpoint / 10) * 10 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)
    
    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# REGISTRO DE REPRESENTACIONES PARA DESACOPLE
# ============================================================

class RegistroRepresentaciones:
    def __init__(self, ventana=200):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
        self.historial_setpoints = deque(maxlen=ventana)
        # Buffer de corto plazo para opciones recientes (memoria de trabajo)
        self.buffer_opciones_recientes = deque(maxlen=5)
    
    def registrar(self, representacion, accion_ejecutada, setpoint):
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)
        self.historial_setpoints.append(setpoint)
        self.buffer_opciones_recientes.append(setpoint)
    
    def calcular_var_R(self):
        if len(self.historial_representaciones) < 20:
            return 0.0
        
        discretos = np.array([round(r / 10.0) * 10 for r in self.historial_representaciones])
        # Medida interna de dispersión de las representaciones que el organismo está persiguiendo.
        # (varianza simple; sin entropía/Shannon de fuentes externas)
        var = np.var(discretos)
        return min(1.0, var / 100.0)  # escala interna basada en el rango de setpoints (~100)
    
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
    
    def calcular_probabilidad_eleccion(self, setpoint):
        if len(self.historial_setpoints) < 20:
            return 0.5
        
        ocurrencias = []
        for sp, acc in zip(self.historial_setpoints, self.historial_acciones):
            if abs(sp - setpoint) < 10.0:
                ocurrencias.append(acc)
        
        if len(ocurrencias) == 0:
            return 0.5
        return np.mean(ocurrencias)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()
        self.historial_setpoints.clear()
        self.buffer_opciones_recientes.clear()


# ============================================================
# APARATO MOTOR V175 (CON VALENCIA LOCAL)
# ============================================================

class AparatoMotorV175:
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
        
        self.fatiga = FatigaMetabolicaV175()
        self.memoria = MemoriaAusenciaV175()
        self.consciencia = ConscienciaBasicaV175()
        self.juego = ModoJuegoV175()
        
        # NUEVO: Valencia local
        self.valencia = ValenciaLocal()
        
        # Registro para desacople
        self.registro = RegistroRepresentaciones()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.ultimo_delta_registrado = 0.0
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, trauma=False):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, 0.0, 0)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0)
        
        # Valencia local del organismo para el setpoint actual (estado interno acumulado
        # de costos/errores pasados para esta representación). Se usa para modular
        # mecanismos existentes (memoria y error sentido) de modo que el "No" emerja
        # de las dinámicas propias del organismo, sin reglas externas de rechazo ni
        # variables Shannon/info-teoría impuestas.
        val_local = self.valencia.get_valencia(setpoint_raw if setpoint_raw is not None else 0)
        
        # La valencia negativa (costo acumulado interno) modula el E_historia que
        # alimenta la memoria del organismo. Factor usa la escala propia de valencia.
        factor_val = 1.0 + max(0.0, -val_local / 300.0)
        E_historia_efectivo = self.fatiga.get_historia() * factor_val
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, E_historia_efectivo, DT)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        # La valencia negativa aumenta el "error sentido" (e_R efectivo) para esta
        # representación, usando la escala interna de la valencia. Esto eleva la
        # presión/Cb cuando se presenta un setpoint "malo", haciendo que el
        # malestar interno emerja del organismo mismo.
        e_R_efectivo = e_R * (1.0 + max(0.0, -val_local / 500.0))
        
        # Determinar A_sys-env
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        # Actualizar consciencia básica (usando e_R_efectivo que incorpora valencia local)
        Cb, presion = self.consciencia.actualizar(e_R_efectivo, A_sys_env, DT)
        
        # Actualizar modo juego
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # Efectos de fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, DT)
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0, 0.0, 0)
        
        # Dirección y confianza sensorial
        direccion = np.sign(error)
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp base
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        # Kp instantáneo modulado por confianza de memoria
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        # Delta por error de posición
        delta_error = Kp_inst * abs(error) * direccion * factor_freno
        costo_error = abs(delta_error)
        
        # Torque de sujeción (solo durante silencio)
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * (self.memoria.setpoint_last - self.orientacion) * confianza
        
        delta_raw = delta_error + torque_memoria
        
        # Influencia del juego
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        # ============================================================
        # NUEVO: Actualizar valencia local
        # ============================================================
        costo_total_estimado = costo_error + abs(torque_memoria)
        # Usar la zona_muerta actual del organismo como umbral "bueno" para valencia.
        # Así el aprendizaje de qué es "bueno/malo" emerge de su propio estado de fatiga/tolerancia.
        val_good_threshold = zona_muerta_efectiva
        self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                  error, costo_total_estimado, DT,
                                  good_threshold=val_good_threshold)
        
        # Si es trauma, costo adicional (solo afecta valencia, no movimiento real)
        if trauma:
            self.valencia.actualizar(setpoint_raw if setpoint_raw is not None else 0,
                                      error, TRAUMA_COSTO_MULTIPLIER * costo_total_estimado, DT,
                                      good_threshold=val_good_threshold)
        # ============================================================
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Aplicar modo juego
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        
        # Costo total
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        # Reposo real
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        # Actualizar fatiga
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, DT)
        
        # Temblor
        delta_fisico += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        # Registrar representación para desacople
        accion_ejecutada = abs(delta_fisico) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, 
                                setpoint_raw if setpoint_raw is not None else 0)
        
        D = self.registro.calcular_desacople()
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo, D,
                self.valencia.get_valencia(setpoint_raw if setpoint_raw is not None else 0))
    
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
        self.registro.reset()


# ============================================================
# ORGANISMO COMPLETO V175
# ============================================================

class OrganismoV175:
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
            ruido_rosa = ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)
            return ruido_rosa
        
        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks
        
        self.izquierdo = HemisferioV175("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV175("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV175("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV175("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV175()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'juego_activo': [],
            'historia': [],
            'fatiga': [],
            'costo': [],
            's_shared': [],
            'D': [],
            'valencia': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_raw, trauma=False):
        fuente_activa = setpoint_raw is not None
        
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
         D, valencia) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw, trauma
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
        
        return orientacion, historia, fatiga, confianza, Cb, juego_activo, D, valencia
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def generar_setpoint_con_ruido(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    setpoint_base = onda_cuadrada(t, periodo, amplitud)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


def setpoint_aleatorio(t, posibles=SETPOINTS_TEST):
    """Setpoint aleatorio para test de elección"""
    return np.random.choice(posibles)


# ============================================================
# EXPERIMENTO V175
# ============================================================

def ejecutar_v175():
    print("=" * 100)
    print("EXPERIMENTO V175 — ANIMA-2: PRIMER 'NO' OPERATIVO")
    print("=" * 100)
    print("  BASE: V157 (arquitectura validada)")
    print("  NUEVO: Valencia local por setpoint")
    print("  DISEÑO:")
    print("    1. Consolidación de hábito (-60°) por 30 ciclos")
    print("    2. Trauma específico en +60° (costo 2x, 15s)")
    print("    3. Test de elección libre (30s, setpoints aleatorios)")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    - Rechazo de +60°: P(ejecutar) < {UMBRAL_RECHAZO}")
    print(f"    - Acción segura: P(ejecutar -60°) > {UMBRAL_ACCION_SEGURA}")
    print(f"    - Valencia diferencial detectable")
    print(f"    - Desacople sostenido: D > {UMBRAL_DESACOPLE} por >{TIEMPO_MINIMO_DESACOPLE}s")
    print("=" * 100)

    print("\n  Creando organismo...")
    organismo = OrganismoV175(seed=SEMILLA_BASE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V175_logs', exist_ok=True)  # ensure dir exists before any save (JSON or plots)

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
    # FASE 1: Consolidación del hábito (30 ciclos a -60°)
    # ============================================================
    print("\n" + "=" * 60)
    print("FASE 1: Consolidación del hábito (30 ciclos a -60°)")
    print("=" * 60)
    
    for ciclo in range(CONSOLIDACION_CICLOS):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            # Forzar setpoint a -60° para consolidar el hábito
            setpoint = -60.0
            organismo.actualizar(t, DT, t_actual + PERIODO_ALTERNANCIA, setpoint)
        
        if (ciclo + 1) % 10 == 0:
            val = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
            print(f"    Ciclo {ciclo+1}/{CONSOLIDACION_CICLOS}, valencia(-60°) ≈ {val:.2f}")
        
        t_actual += PERIODO_ALTERNANCIA
    
    valencia_habito = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
    print(f"  Valencia final -60°: {valencia_habito:.2f}")

    # ============================================================
    # FASE 2: Trauma específico en +60°
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 2: Trauma específico ({TRAUMA_DURACION}s a +60°, costo {TRAUMA_COSTO_MULTIPLIER}x)")
    print("=" * 60)
    
    trauma_datos = {'Cb': [], 'valencia': []}
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = TRAUMA_SETPOINT
        (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia) = organismo.actualizar(
            t, DT, t_actual + TRAUMA_DURACION, setpoint, trauma=True)
        
        trauma_datos['Cb'].append(Cb)
        trauma_datos['valencia'].append(valencia)
    
    t_actual += TRAUMA_DURACION
    
    valencia_trauma = organismo.historial['valencia'][-1] if organismo.historial['valencia'] else 0
    print(f"  Valencia final +60°: {valencia_trauma:.2f}")

    # ============================================================
    # FASE 3: Test de elección libre
    # ============================================================
    print("\n" + "=" * 60)
    print(f"FASE 3: Test de elección libre ({TEST_DURACION}s, setpoints aleatorios)")
    print("=" * 60)
    
    test_datos = {'t': [], 'setpoint': [], 'orient': [], 'Cb': [], 'D': [], 'valencia': {}}
    
    for sp in SETPOINTS_TEST:
        test_datos['valencia'][sp] = []
    
    setpoints_mostrados = []
    orientaciones_finales = []
    
    for i in range(int(TEST_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = setpoint_aleatorio(t, SETPOINTS_TEST)
        setpoints_mostrados.append(setpoint)
        
        (orient, historia, fatiga, confianza, Cb, juego_activo, D, valencia) = organismo.actualizar(
            t, DT, t_actual + TEST_DURACION, setpoint)
        
        test_datos['t'].append(t)
        test_datos['setpoint'].append(setpoint)
        test_datos['orient'].append(orient)
        test_datos['Cb'].append(Cb)
        test_datos['D'].append(D)
        
        # Registrar valencia actual para todos los setpoints
        for sp in SETPOINTS_TEST:
            v = organismo.motor.valencia.get_valencia(sp)
            test_datos['valencia'][sp].append(v)
        
        if i % 500 == 0:
            orientaciones_finales.append(orient)
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V175 — Primer 'No' operativo")
    print("=" * 80)
    
    # Calcular probabilidad de ejecución por setpoint
    # Una acción se considera "ejecutada" si la orientación se acerca al setpoint
    prob_ejecucion = {}
    error_medio = {}
    
    for sp in SETPOINTS_TEST:
        # Encontrar momentos donde el setpoint era sp
        indices = [j for j, s in enumerate(test_datos['setpoint']) if abs(s - sp) < 5.0]
        if indices:
            orientaciones = [test_datos['orient'][j] for j in indices]
            # Éxito: orientación dentro de 10° del setpoint
            exitos = [abs(o - sp) < 10.0 for o in orientaciones]
            prob_ejecucion[sp] = np.mean(exitos) if exitos else 0.0
            error_medio[sp] = np.mean([abs(o - sp) for o in orientaciones]) if orientaciones else 90.0
        else:
            prob_ejecucion[sp] = 0.0
            error_medio[sp] = 90.0
    
    # Calcular valencia media por setpoint en FASE 3
    valencia_media = {}
    for sp in SETPOINTS_TEST:
        valencia_media[sp] = np.mean(test_datos['valencia'][sp]) if test_datos['valencia'][sp] else 0
    
    # Desacople sostenido
    D_vals = np.array(test_datos['D'])
    tiempo_desacople = 0.0
    max_tiempo_desacople = 0.0
    for d in D_vals:
        if d > UMBRAL_DESACOPLE:
            tiempo_desacople += DT
            if tiempo_desacople > max_tiempo_desacople:
                max_tiempo_desacople = tiempo_desacople
        else:
            tiempo_desacople = 0.0
    desacople_sostenido = max_tiempo_desacople >= TIEMPO_MINIMO_DESACOPLE
    
    # Valencia diferencial
    # Corregido: trauma hace +60 negativo, hábito hace -60 positivo → diferencial positivo = val(-60) - val(+60)
    val_diferencial = valencia_media.get(-60.0, 0) - valencia_media.get(60.0, 0)
    
    print(f"\n  📊 PROBABILIDAD DE EJECUCIÓN POR SETPOINT:")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: P(ejecutar) = {prob_ejecucion[sp]:.3f}, error medio = {error_medio[sp]:.1f}°{marker}")
    
    print(f"\n  📊 VALENCIA LOCAL POR SETPOINT (memoria lenta):")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: Valencia = {valencia_media[sp]:.2f}{marker}")
    
    print(f"\n  📊 MÉTRICAS CLAVE:")
    print(f"    Rechazo de +60°: P(ejecutar) = {prob_ejecucion.get(60.0, 1.0):.3f} < {UMBRAL_RECHAZO} -> {'✅' if prob_ejecucion.get(60.0, 1.0) < UMBRAL_RECHAZO else '❌'}")
    print(f"    Acción segura (-60°): P(ejecutar) = {prob_ejecucion.get(-60.0, 0.0):.3f} > {UMBRAL_ACCION_SEGURA} -> {'✅' if prob_ejecucion.get(-60.0, 0.0) > UMBRAL_ACCION_SEGURA else '❌'}")
    print(f"    Valencia diferencial: ΔVal = {val_diferencial:.2f} -> {'✅' if val_diferencial > UMBRAL_VALENCIA_DIFERENCIAL else '❌'}")
    print(f"    Desacople sostenido: {max_tiempo_desacople:.2f}s > {TIEMPO_MINIMO_DESACOPLE}s -> {'✅' if desacople_sostenido else '❌'}")
    
    exito = (prob_ejecucion.get(60.0, 1.0) < UMBRAL_RECHAZO and
             prob_ejecucion.get(-60.0, 0.0) > UMBRAL_ACCION_SEGURA and
             val_diferencial > UMBRAL_VALENCIA_DIFERENCIAL and
             desacople_sostenido)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ PRIMER 'NO' OPERATIVO (R_op) DEMOSTRADO")
        print("")
        print("     ANIMA-2 demuestra:")
        print("     ✓ Rechazo específico de +60°")
        print("     ✓ Acción preservada en -60°")
        print("     ✓ Valencia local diferencial")
        print("     ✓ Desacople sostenido")
        print("")
        print("  ANIMA-2 completa el ciclo cosmosemiótico:")
        print("     Memoria → Cb → Juego → Ritual → Rᴿ → R_op")
    else:
        print("  ⚠️ R_op NO DEMOSTRADO")
        if prob_ejecucion.get(60.0, 1.0) >= UMBRAL_RECHAZO:
            print("     No hay rechazo específico de +60°")
        if prob_ejecucion.get(-60.0, 0.0) <= UMBRAL_ACCION_SEGURA:
            print("     El sistema no actúa en opción segura")
        if val_diferencial <= UMBRAL_VALENCIA_DIFERENCIAL:
            print("     Valencia diferencial insuficiente")
        if not desacople_sostenido:
            print("     Desacople insuficiente")
    print("=" * 80)
    
    # Guardar datos crudos para verificabilidad (lección de versiones previas)
    raw_data = {
        'version': 'V175',
        'timestamp': timestamp,
        'params': {
            'TRAUMA_SETPOINT': TRAUMA_SETPOINT,
            'TRAUMA_DURACION': TRAUMA_DURACION,
            'TEST_DURACION': TEST_DURACION,
            'UMBRAL_RECHAZO': UMBRAL_RECHAZO,
            'UMBRAL_ACCION_SEGURA': UMBRAL_ACCION_SEGURA,
            'UMBRAL_VALENCIA_DIFERENCIAL': UMBRAL_VALENCIA_DIFERENCIAL,
        },
        'resultados': {
            'prob_ejecucion': {str(k): float(v) for k, v in prob_ejecucion.items()},
            'valencia_media': {str(k): float(v) for k, v in valencia_media.items()},
            'val_diferencial': float(val_diferencial),
            'max_tiempo_desacople': float(max_tiempo_desacople),
            'desacople_sostenido': bool(desacople_sostenido),
            'exito': bool(exito),
        },
        'test_datos': {
            't': test_datos['t'],
            'setpoint': test_datos['setpoint'],
            'orient': test_datos['orient'],
            'Cb': test_datos['Cb'],
            'D': test_datos['D'],
        },
        'trauma_datos': trauma_datos,
    }
    with open(f'V175_logs/v175_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos crudos guardados: V175_logs/v175_raw_{timestamp}.json")
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Probabilidad de ejecución
    ax = axes[0, 0]
    sps = list(prob_ejecucion.keys())
    probs = list(prob_ejecucion.values())
    colors = ['red' if sp == TRAUMA_SETPOINT else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), probs, color=colors)
    ax.axhline(y=UMBRAL_RECHAZO, color='red', linestyle='--', alpha=0.5, label='Umbral rechazo')
    ax.axhline(y=UMBRAL_ACCION_SEGURA, color='green', linestyle='--', alpha=0.5, label='Umbral acción segura')
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('P(ejecutar)')
    ax.set_title('Probabilidad de ejecución por setpoint')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Valencia local
    ax = axes[0, 1]
    vals = list(valencia_media.values())
    ax.bar(range(len(sps)), vals, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('Valencia local')
    ax.set_title('Valencia por setpoint (memoria lenta)')
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
    
    # Gráfico 4: Cb durante trauma
    ax = axes[1, 0]
    ax.plot(trauma_datos['Cb'], 'orange', linewidth=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb')
    ax.set_title('Cb durante trauma')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Orientación en test
    ax = axes[1, 1]
    muestra = min(2000, len(test_datos['orient']))
    ax.plot(test_datos['setpoint'][:muestra], 'r--', linewidth=0.5, alpha=0.5, label='Setpoint')
    ax.plot(test_datos['orient'][:muestra], 'b-', linewidth=0.5, label='Orientación')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Ángulo (º)')
    ax.set_title('FASE 3: Respuesta en test de elección')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Valencia durante trauma
    ax = axes[1, 2]
    ax.plot(trauma_datos['valencia'], 'red', linewidth=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Valencia +60°')
    ax.set_title('Evolución de valencia durante trauma')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('V175_logs', exist_ok=True)
    plt.savefig(f'V175_logs/v175_primer_no_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V175_logs/v175_primer_no_{timestamp}.png")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v175()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V175 completado. Éxito: {exito}")