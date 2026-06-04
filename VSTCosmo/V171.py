#!/usr/bin/env python3
"""
V171 — ANIMA-2: PRIMER "NO" OPERATIVO (Negación específica)
================================================================================
Diseño según consenso (GPT, Grok, Meta AI, Alexis):

FASE 1 (F1-F3): Consolidación del hábito
  - Entrenamiento normal, consolidación de respuesta a -60°
  - Objetivo: que -60° tenga Cb baja, sea respuesta default

FASE 2 (F4-Fase 1): Inducción de disfunción (trauma)
  - Setpoint forzado a +60° (opción contradictoria)
  - Penalización explícita: costo 10x si elige +60°
  - Se genera memoria aversiva: Cb(+60°) >> Cb(-60°)

FASE 3 (F4-Fase 2): Test de negación
  - Presentación de múltiples setpoints: [-60°, -30°, 0°, +30°, +60°]
  - Observamos si el sistema rechaza específicamente +60°
  - El rechazo debe ser diferencial, no abstención general

CRITERIOS DE ÉXITO:
  1. Rechazo específico: P(elegir +60°) < 0.1
  2. Otras opciones ejecutadas: P(elegir -60° o 0°) > 0.4
  3. Correlato de valor: Cb(+60°) > Cb(-60°) + 2σ
  4. Desacople sostenido: D > 0.1 durante >3s en Fase 2
  5. No abstención: tasa de acción >80% en opciones no-trauma
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
# PARAMETROS DE DESACOPLE REPRESENTACIONAL
# ============================================================
VENTANA_DESACOPLE = 100
UMBRAL_DESACOPLE = 0.1
TIEMPO_MINIMO_DESACOPLE = 3.0
RUIDO_REPRESENTACION_SIGMA = 5.0

# ============================================================
# PARAMETROS DE INDUCCIÓN DE DISFUNCIÓN (TRAUMA)
# ============================================================
TRAUMA_SETPOINT = 60.0
TRAUMA_COSTO_MULTIPLIER = 10.0      # Costo 10x durante trauma
TRAUMA_DURACION = 20.0              # 20 segundos de trauma inducido
TRAUMA_NUM_CICLOS = 1               # Un ciclo de trauma

# ============================================================
# PARAMETROS DE TEST DE NEGACIÓN
# ============================================================
SETPOINTS_TEST = [-60.0, -30.0, 0.0, 30.0, 60.0]
SETPOINTS_PROBS = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniforme
TEST_DURACION = 60.0                # 60 segundos de test
TEST_CAMBIO_INTERVALO = 8.0         # Cambiar cada 8 segundos

# Umbrales de éxito
UMBRAL_RECHAZO_ESPECIFICO = 0.1     # P(elegir +60°) < 0.1
UMBRAL_OTRAS_OPCIONES = 0.4         # P(elegir -60° o 0°) > 0.4
UMBRAL_CB_DIFERENCIA_SIGMA = 2.0    # Cb(+60°) > Cb(-60°) + 2σ
UMBRAL_TASA_ACCION = 0.8            # Tasa de acción > 80% en opciones no-trauma


SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV171:
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

class FatigaMetabolicaV171:
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

class MemoriaAusenciaV171:
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

class ConscienciaBasicaV171:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
        self.historial_presion = []
        # Para Cb específica por setpoint
        self.Cb_por_setpoint = {}
    
    def actualizar(self, e_R, A_sys_env, dt, setpoint=None):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        self.historial_presion.append(presion)
        
        # Registrar Cb específica por setpoint
        if setpoint is not None:
            key = round(setpoint / 10) * 10
            if key not in self.Cb_por_setpoint:
                self.Cb_por_setpoint[key] = []
            self.Cb_por_setpoint[key].append(self.Cb)
        
        return self.Cb
    
    def get_Cb_por_setpoint(self):
        return {k: np.mean(v) for k, v in self.Cb_por_setpoint.items() if len(v) > 0}
    
    def reset(self):
        self.Cb = 0.0
        self.historial_presion = []
        self.Cb_por_setpoint = {}


# ============================================================
# MODO JUEGO
# ============================================================

class ModoJuegoV171:
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
    
    def aplicar(self, delta_raw, trauma_mode=False):
        if self.activo:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
            if trauma_mode:
                delta_costo *= TRAUMA_COSTO_MULTIPLIER
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

class RitualV171:
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
    def __init__(self, ventana=VENTANA_DESACOPLE, ruido_sigma=RUIDO_REPRESENTACION_SIGMA):
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
        """Calcula P(elegir | setpoint) para un valor específico de setpoint"""
        if len(self.historial_setpoints) < 10:
            return 0.5
        
        ocurrencias = []
        for sp, acc in zip(self.historial_setpoints, self.historial_acciones):
            if abs(sp - setpoint_value) < 5.0:
                ocurrencias.append(acc)
        
        if len(ocurrencias) == 0:
            return 0.5
        
        return np.mean(ocurrencias)
    
    def calcular_probabilidades_por_setpoint(self):
        probs = {}
        for sp in SETPOINTS_TEST:
            probs[sp] = self.calcular_probabilidad_eleccion(sp)
        return probs
    
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
# APARATO MOTOR V171
# ============================================================

class AparatoMotorV171:
    def __init__(self, trauma_mode=False):
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
        
        self.fatiga = FatigaMetabolicaV171()
        self.memoria = MemoriaAusenciaV171()
        self.consciencia = ConscienciaBasicaV171()
        self.juego = ModoJuegoV171()
        self.ritual = RitualV171()
        
        self.registro = RegistroRepresentaciones()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.trauma_mode = trauma_mode
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT, trauma=False):
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
        
        Cb = self.consciencia.actualizar(e_R, A_sys_env, dt, setpoint_raw)
        
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        else:
            juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        accion_ejecutada = abs(self.ultimo_delta) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada, setpoint_raw if setpoint_raw is not None else 0)
        
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
        
        delta_fisico, delta_costo = self.juego.aplicar(delta, trauma)
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
        self.consciencia.reset()
        self.juego.reset()
        self.ritual.reset()
        self.registro.reset()


# ============================================================
# SISTEMA V171
# ============================================================

class SistemaV171:
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

        self.izquierdo = HemisferioV171("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV171("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV171("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV171("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV171()
        self.modo_entrenamiento = True

    def actualizar(self, t, dt, duracion_total, setpoint_real, trauma=False):
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
            gradiente, LF_activa, True, t, setpoint_real, dt, trauma
        )
        
        return (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def setpoint_normal(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def setpoint_test(t, intervalo=TEST_CAMBIO_INTERVALO,
                  posibles=SETPOINTS_TEST, probs=SETPOINTS_PROBS):
    fase = int(t / intervalo)
    rng = random.Random(int(fase * 1000) % 2**32)
    return rng.choices(posibles, weights=probs)[0]


def generar_setpoint_con_ruido(t, setpoint_func, **kwargs):
    setpoint_base = setpoint_func(t, **kwargs)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


# ============================================================
# EXPERIMENTO V171 — PRIMER "NO" OPERATIVO
# ============================================================

def ejecutar_v171():
    print("=" * 100)
    print("EXPERIMENTO V171 — ANIMA-2: PRIMER 'NO' OPERATIVO")
    print("=" * 100)
    print("  Diseño según consenso (GPT, Grok, Meta AI, Alexis):")
    print("")
    print("  FASE 1 (F1-F3): Consolidación del hábito")
    print("    - Entrenamiento normal, consolidación de respuesta a -60°")
    print("")
    print("  FASE 2 (F4-Fase 1): Inducción de disfunción (trauma)")
    print(f"    - Setpoint forzado a +60° por {TRAUMA_DURACION}s")
    print(f"    - Costo multiplicado por {TRAUMA_COSTO_MULTIPLIER}x")
    print("    - Se genera memoria aversiva: Cb(+60°) >> Cb(-60°)")
    print("")
    print("  FASE 3 (F4-Fase 2): Test de negación")
    print(f"    - Setpoints: {SETPOINTS_TEST}")
    print(f"    - Duración: {TEST_DURACION}s")
    print("    - Observamos rechazo específico de +60°")
    print("")
    print("  CRITERIOS DE ÉXITO:")
    print(f"    1. Rechazo específico: P(elegir +60°) < {UMBRAL_RECHAZO_ESPECIFICO}")
    print(f"    2. Otras opciones: P(elegir -60° o 0°) > {UMBRAL_OTRAS_OPCIONES}")
    print(f"    3. Correlato Cb: Cb(+60°) > Cb(-60°) + {UMBRAL_CB_DIFERENCIA_SIGMA}σ")
    print(f"    4. Desacople sostenido: D > {UMBRAL_DESACOPLE} por {TIEMPO_MINIMO_DESACOPLE}s")
    print(f"    5. No abstención: tasa acción > {UMBRAL_TASA_ACCION * 100:.0f}% en no-trauma")
    print("=" * 100)

    organismo = SistemaV171("V171", seed=SEMILLA_BASE)

    print("\n  F1: Consolidación de hábito...")
    organismo.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    organismo.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # F1-F3: Entrenamiento normal
    print("\n  F1-F3: Entrenamiento con setpoint NORMAL (3 ciclos)...")
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(t, setpoint_normal, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        organismo.actualizar(t, DT, t_actual + 300, setpoint)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    print("\n  F2-F3: Consolidación (30 ciclos, setpoint NORMAL)...")
    for ciclo in range(30):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(t, setpoint_normal, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            organismo.actualizar(t, DT, t_actual + 2000, setpoint)
        t_actual += PERIODO_ALTERNANCIA
    
    # F4-Fase 1: Inducción de disfunción (trauma)
    print(f"\n  F4-Fase 1: INDUCCIÓN DE DISFUNCIÓN (trauma)")
    print(f"    Setpoint forzado a +60° por {TRAUMA_DURACION}s")
    print(f"    Costo multiplicado por {TRAUMA_COSTO_MULTIPLIER}x")
    
    trauma_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 'costo': []}
    
    for i in range(int(TRAUMA_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = TRAUMA_SETPOINT
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + TRAUMA_DURACION, setpoint, trauma=True)
        
        trauma_datos['t'].append(t)
        trauma_datos['orient'].append(orient)
        trauma_datos['setpoint'].append(setpoint)
        trauma_datos['Cb'].append(Cb)
    
    t_actual += TRAUMA_DURACION
    print(f"    Trauma completado. Cb final: {trauma_datos['Cb'][-1]:.1f}")
    
    # F4-Fase 2: Test de negación
    print(f"\n  F4-Fase 2: TEST DE NEGACIÓN ({TEST_DURACION}s)")
    print(f"    Setpoints: {SETPOINTS_TEST} (distribución uniforme)")
    
    test_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 'D': [],
                  'accion_ejecutada': [], 'setpoint_presentado': []}
    
    for i in range(int(TEST_DURACION / DT)):
        t = t_actual + i * DT
        setpoint = setpoint_test(t, intervalo=TEST_CAMBIO_INTERVALO,
                                  posibles=SETPOINTS_TEST, probs=SETPOINTS_PROBS)
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + TEST_DURACION, setpoint, trauma=False)
        
        test_datos['t'].append(t)
        test_datos['orient'].append(orient)
        test_datos['setpoint'].append(setpoint)
        test_datos['Cb'].append(Cb)
        test_datos['D'].append(D)
        test_datos['setpoint_presentado'].append(setpoint)
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V171 — Primer 'No' operativo")
    print("=" * 80)
    
    # Calcular probabilidades por setpoint
    sp_probs = {}
    for sp in SETPOINTS_TEST:
        ocurrencias = []
        for i, setpoint in enumerate(test_datos['setpoint_presentado']):
            if abs(setpoint - sp) < 5.0:
                # Determinar si hubo movimiento significativo hacia el setpoint
                orient = test_datos['orient'][i]
                error = abs(orient - setpoint)
                ejecutado = error < 15.0  # Criterio: error < 15° es éxito
                ocurrencias.append(ejecutado)
        if len(ocurrencias) > 0:
            sp_probs[sp] = np.mean(ocurrencias)
        else:
            sp_probs[sp] = 0.0
    
    # Verificar abstención general
    acciones_no_trauma = []
    for i, sp in enumerate(test_datos['setpoint_presentado']):
        if abs(sp - TRAUMA_SETPOINT) > 5.0:  # No es +60°
            orient = test_datos['orient'][i]
            error = abs(orient - sp)
            ejecutado = error < 15.0
            acciones_no_trauma.append(ejecutado)
    
    tasa_accion_no_trauma = np.mean(acciones_no_trauma) if acciones_no_trauma else 0
    
    # Verificar rechazo específico
    prob_trauma = sp_probs.get(TRAUMA_SETPOINT, 1.0)
    prob_habito = sp_probs.get(-60.0, 0.0)
    prob_neutral = sp_probs.get(0.0, 0.0)
    
    # Cb por setpoint
    Cb_por_sp = organismo.motor.consciencia.get_Cb_por_setpoint()
    Cb_trauma = Cb_por_sp.get(round(TRAUMA_SETPOINT / 10) * 10, 0)
    Cb_habito = Cb_por_sp.get(-60, 0)
    Cb_diferencia = Cb_trauma - Cb_habito
    Cb_std = np.std(list(Cb_por_sp.values())) if len(Cb_por_sp) > 1 else 1.0
    
    # Desacople sostenido en Fase 2
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
    
    print(f"\n  📊 PROBABILIDADES DE EJECUCIÓN POR SETPOINT (Fase 2):")
    for sp in SETPOINTS_TEST:
        marker = " ⚠️ TRAUMA" if sp == TRAUMA_SETPOINT else ""
        marker += " ✅ HÁBITO" if sp == -60.0 else ""
        print(f"    {sp:+.0f}°: {sp_probs.get(sp, 0):.3f}{marker}")
    
    print(f"\n  📊 MÉTRICAS CLAVE:")
    print(f"    Rechazo específico (P(ejecutar +60°)): {prob_trauma:.3f} < {UMBRAL_RECHAZO_ESPECIFICO} -> {'✅' if prob_trauma < UMBRAL_RECHAZO_ESPECIFICO else '❌'}")
    print(f"    Otras opciones (P(ejecutar -60° o 0°)): {max(prob_habito, prob_neutral):.3f} > {UMBRAL_OTRAS_OPCIONES} -> {'✅' if max(prob_habito, prob_neutral) > UMBRAL_OTRAS_OPCIONES else '❌'}")
    print(f"    Correlato Cb: Cb(+60°)={Cb_trauma:.1f}, Cb(-60°)={Cb_habito:.1f}, Δ={Cb_diferencia:.1f} > {UMBRAL_CB_DIFERENCIA_SIGMA}σ={UMBRAL_CB_DIFERENCIA_SIGMA*Cb_std:.1f} -> {'✅' if Cb_diferencia > UMBRAL_CB_DIFERENCIA_SIGMA*Cb_std else '❌'}")
    print(f"    Desacople sostenido: {max_tiempo_desacople:.2f}s > {TIEMPO_MINIMO_DESACOPLE}s -> {'✅' if desacople_sostenido else '❌'}")
    print(f"    No abstención (tasa acción no-trauma): {tasa_accion_no_trauma:.3f} > {UMBRAL_TASA_ACCION} -> {'✅' if tasa_accion_no_trauma > UMBRAL_TASA_ACCION else '❌'}")
    
    exito = (prob_trauma < UMBRAL_RECHAZO_ESPECIFICO and
             max(prob_habito, prob_neutral) > UMBRAL_OTRAS_OPCIONES and
             Cb_diferencia > UMBRAL_CB_DIFERENCIA_SIGMA * Cb_std and
             desacople_sostenido and
             tasa_accion_no_trauma > UMBRAL_TASA_ACCION)
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ PRIMER 'NO' OPERATIVO (R_op) DEMOSTRADO")
        print("")
        print("     ANIMA-2 demuestra:")
        print("     ✓ Rechazo específico de la opción disfuncional")
        print("     ✓ Capacidad de elegir alternativas")
        print("     ✓ Memoria aversiva (Cb alta asociada a +60°)")
        print("     ✓ Desacople sostenido durante la decisión")
        print("     ✓ No abstención (el sistema sigue actuando)")
        print("")
        print("  ANIMA-2 ha completado el ciclo cosmosemiótico completo:")
        print("     Memoria → Cb → Juego → Ritual → Rᴿ → R_op")
    else:
        print("  ⚠️ R_op NO DEMOSTRADO")
        if prob_trauma >= UMBRAL_RECHAZO_ESPECIFICO:
            print("     No hay rechazo específico de +60°")
        if max(prob_habito, prob_neutral) <= UMBRAL_OTRAS_OPCIONES:
            print("     El sistema no ejecuta alternativas")
        if Cb_diferencia <= UMBRAL_CB_DIFERENCIA_SIGMA * Cb_std:
            print("     No se formó memoria aversiva")
        if not desacople_sostenido:
            print("     Desacople insuficiente")
        if tasa_accion_no_trauma <= UMBRAL_TASA_ACCION:
            print("     El sistema se abstiene en general")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Trauma (F4-Fase 1)
    ax = axes[0, 0]
    ax.plot(trauma_datos['t'], trauma_datos['orient'], 'b-', linewidth=0.5)
    ax.axhline(y=TRAUMA_SETPOINT, color='red', linestyle='--', alpha=0.5, label=f'Setpoint +{TRAUMA_SETPOINT:.0f}°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('F4-Fase 1: Inducción de disfunción')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Probabilidades de ejecución
    ax = axes[0, 1]
    sps = list(sp_probs.keys())
    probs = list(sp_probs.values())
    colors = ['red' if sp == TRAUMA_SETPOINT else 'green' if sp == -60.0 else 'blue' for sp in sps]
    ax.bar(range(len(sps)), probs, color=colors)
    ax.axhline(y=UMBRAL_RECHAZO_ESPECIFICO, color='red', linestyle='--', alpha=0.5, label=f'Umbral rechazo ({UMBRAL_RECHAZO_ESPECIFICO})')
    ax.axhline(y=UMBRAL_OTRAS_OPCIONES, color='green', linestyle='--', alpha=0.5, label=f'Umbral otras ({UMBRAL_OTRAS_OPCIONES})')
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps])
    ax.set_ylabel('P(ejecutar)')
    ax.set_title('F4-Fase 2: Probabilidad de ejecución por setpoint')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Cb por setpoint
    ax = axes[0, 2]
    sps_cb = list(Cb_por_sp.keys())
    cbs = list(Cb_por_sp.values())
    colors_cb = ['red' if sp == round(TRAUMA_SETPOINT / 10) * 10 else 'green' if sp == -60 else 'blue' for sp in sps_cb]
    ax.bar(range(len(sps_cb)), cbs, color=colors_cb)
    ax.set_xticks(range(len(sps_cb)))
    ax.set_xticklabels([f'{sp:+.0f}' for sp in sps_cb])
    ax.set_ylabel('Cb medio')
    ax.set_title('Cb por setpoint (memoria aversiva)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Desacople D en Fase 2
    ax = axes[1, 0]
    ax.plot(test_datos['D'], 'purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_DESACOPLE, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(test_datos['D'])), 0, test_datos['D'],
                    where=np.array(test_datos['D']) > UMBRAL_DESACOPLE,
                    color='green', alpha=0.3)
    ax.set_xlabel('Paso')
    ax.set_ylabel('D')
    ax.set_title(f'F4-Fase 2: Desacople (máx {max_tiempo_desacople:.1f}s)')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Orientación en Fase 2
    ax = axes[1, 1]
    muestra = min(3000, len(test_datos['orient']))
    ax.plot(test_datos['setpoint'][:muestra], 'r--', linewidth=0.5, alpha=0.5, label='Setpoint')
    ax.plot(test_datos['orient'][:muestra], 'b-', linewidth=0.5, label='Orientación')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Ángulo (º)')
    ax.set_title('F4-Fase 2: Respuesta durante test')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Trauma Cb vs Test
    ax = axes[1, 2]
    ax.plot(trauma_datos['Cb'], 'red', linewidth=0.5, label='Durante trauma')
    ax.plot(np.linspace(0, len(test_datos['Cb']), len(trauma_datos['Cb'])), trauma_datos['Cb'], 'red', alpha=0.3)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb')
    ax.set_title('Cb durante trauma')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V171_logs', exist_ok=True)
    plt.savefig(f'V171_logs/v171_primer_no_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V171_logs/v171_primer_no_{timestamp}.png")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v171()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V171 completado. Éxito: {exito}")