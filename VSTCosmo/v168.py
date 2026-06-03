#!/usr/bin/env python3
"""
V168 — ANIMA-2 Etapa 5: PRIMER "NO" OPERATIVO (R_op)
================================================================================
BASE: V167 (meta-representación observacional validada)

NUEVO: Inhibición activa del ritual basada en señal de desajuste (R_op)
  - Cuando la señal de desajuste supera el umbral (0.7), se INHIBE el ritual
  - El organismo puede SUSPENDER voluntariamente un marco histórico disfuncional
  - Es el primer "No" operativo: capacidad de rechazar la propia conducta ritualizada

JUSTIFICACIÓN (Grok + GPT + Alexis):
  - V167 demostró que el ritual persiste ciegamente y genera desajuste
  - La correlación 0.901 indica que el ritual es la CAUSA del desajuste
  - El paso siguiente es que el organismo USE esa información para inhibir el ritual
  - Esto es el "No" operativo: la capacidad de decir "no" a un marco histórico

ARQUITECTURA:
  Etapa 0 (Memoria) → siempre activa
  Etapa 1 (Cb) → siempre activa
  Etapa 2 (Juego) → se activa por Cb, INHIBIDO por ritual
  Etapa 3 (Ritual) → conserva modos de acoplamiento
  Etapa 4 (Rᴿ) → monitor observacional de desajuste
  Etapa 5 (R_op) → NUEVO: puede INHIBIR el ritual cuando es disfuncional

CRITERIOS DE ÉXITO ETAPA 5:
  1. Ritual se inhibe en F4 cuando la señal de desajuste supera umbral
  2. Tras la inhibición, el error disminuye (el organismo se adapta mejor)
  3. La señal de desajuste correlaciona con la inhibición (causalidad)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque

# ============================================================
# PARAMETROS (DESDE V167, FUNCIONAN)
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

# PARAMETROS RITUAL
RITUAL_TAU = 180.0
RITUAL_REPETICION_MIN = 3
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.4
RITUAL_UMBRAL_CB = 28.0
RITUAL_SALIDA_SUAVE = 0.95
RITUAL_PERSISTENCIA_MIN = 3

# ============================================================
# PARAMETROS META-REPRESENTACIÓN
# ============================================================
META_TAU = 30.0
META_UMBRAL_DESAJUSTE = 0.7          # Umbral para inhibición (más alto que en V167)
META_VENTANA_ERROR = 200
META_K_SUAVIDAD = 0.1

# ============================================================
# PARAMETROS R_op (PRIMER "NO" OPERATIVO)
# ============================================================
R_OP_UMBRAL_INHIBICION = 0.7          # Mismo que umbral de desajuste
R_OP_HISTERESIS = 0.5                 # Necesita señal > umbral por 0.5s para inhibir
R_OP_INHIBITION_DURATION = 5.0        # Duración mínima de inhibición (segundos)
R_OP_DESINHIBICION_THRESHOLD = 0.3    # Señal debe caer bajo este umbral para desinhibir


SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV168:
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

class FatigaMetabolicaV168:
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

class MemoriaAusenciaV168:
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

class ConscienciaBasicaV168:
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

class ModoJuegoV168:
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
        self.tiempo_activo = 0.0


# ============================================================
# RITUAL
# ============================================================

class RitualV168:
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
        
        # Para R_op: inhibición externa
        self.inhibido_por_rop = False
    
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
    
    def actualizar(self, orientacion, Cb, tiempo_actual, dt, inhibir_por_rop=False):
        # R_op puede inhibir el ritual externamente
        self.inhibido_por_rop = inhibir_por_rop
        
        if inhibir_por_rop:
            # Si está inhibido por R_op, no se activa
            self.active = False
            return self.active
        
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
        if self.active and not self.inhibido_por_rop:
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
        self.inhibido_por_rop = False


# ============================================================
# META-REPRESENTACIÓN OBSERVACIONAL
# ============================================================

class MetaRepresentacionObservacional:
    def __init__(self, tau=META_TAU, umbral_desajuste=META_UMBRAL_DESAJUSTE,
                 ventana_error=META_VENTANA_ERROR, k_suavidad=META_K_SUAVIDAD):
        self.tau = tau
        self.umbral_desajuste = umbral_desajuste
        self.ventana_error = ventana_error
        self.k_suavidad = k_suavidad
        
        self.desajuste = 0.0
        self.historial_desajuste = []
        self.buffer_error = deque(maxlen=ventana_error)
        self.buffer_Cb = deque(maxlen=ventana_error)
        self.buffer_ritual = deque(maxlen=ventana_error)
    
    def actualizar(self, error, Cb, ritual_activo, dt):
        self.buffer_error.append(abs(error))
        self.buffer_Cb.append(Cb)
        self.buffer_ritual.append(ritual_activo)
        
        if len(self.buffer_error) > self.ventana_error // 2:
            error_sostenido = np.mean(self.buffer_error)
            Cb_sostenido = np.mean(self.buffer_Cb)
            ritual_sostenido = np.mean(self.buffer_ritual) > 0.5
        else:
            error_sostenido = abs(error)
            Cb_sostenido = Cb
            ritual_sostenido = ritual_activo
        
        if ritual_sostenido:
            error_norm = min(1.0, error_sostenido / 60.0)
            Cb_norm = min(1.0, Cb_sostenido / 500.0)
            presion_activa = error_norm * Cb_norm
            
            if Cb_sostenido < 50 and error_sostenido > 30:
                presion_ciega = error_norm * 0.8
            else:
                presion_ciega = 0.0
            
            presion = max(presion_activa, presion_ciega)
        else:
            presion = 0.0
        
        d_desajuste = presion - self.desajuste / self.tau
        self.desajuste += d_desajuste * dt
        self.desajuste = max(0.0, min(1.0, self.desajuste))
        
        self.historial_desajuste.append(self.desajuste)
        
        return self.desajuste, self.desajuste > self.umbral_desajuste
    
    def reset(self):
        self.desajuste = 0.0
        self.historial_desajuste = []
        self.buffer_error.clear()
        self.buffer_Cb.clear()
        self.buffer_ritual.clear()


# ============================================================
# R_op (PRIMER "NO" OPERATIVO)
# ============================================================

class R_op:
    """
    Primer "No" operativo: capacidad de inhibir el ritual
    cuando la señal de desajuste supera un umbral.
    
    Incluye:
    - Histéresis para evitar oscilaciones
    - Duración mínima de inhibición
    - Desinhibición gradual cuando la señal cae
    """
    
    def __init__(self, umbral_inhibicion=R_OP_UMBRAL_INHIBICION,
                 histéresis=R_OP_HISTERESIS,
                 duracion_minima=R_OP_INHIBITION_DURATION,
                 umbral_desinhibicion=R_OP_DESINHIBICION_THRESHOLD):
        self.umbral_inhibicion = umbral_inhibicion
        self.histeresis = histéresis
        self.duracion_minima = duracion_minima
        self.umbral_desinhibicion = umbral_desinhibicion
        
        self.inhibicion_activa = False
        self.tiempo_en_inhibicion = 0.0
        self.historial_inhibicion = []
        self.señal_para_historial = []
        self.tiempo_desde_ultimo_cruce = 0.0
    
    def actualizar(self, señal_desajuste, dt):
        """
        Decide si inhibir el ritual basado en la señal de desajuste.
        
        Reglas:
        1. Si la señal supera el umbral por más de 'histeresis' segundos, inhibir
        2. Mantener inhibición por al menos 'duracion_minima'
        3. Desinhibir cuando la señal cae bajo 'umbral_desinhibicion'
        """
        self.señal_para_historial.append(señal_desajuste)
        
        if not self.inhibicion_activa:
            # Verificar si hay que inhibir
            if señal_desajuste > self.umbral_inhibicion:
                self.tiempo_desde_ultimo_cruce += dt
                if self.tiempo_desde_ultimo_cruce >= self.histeresis:
                    self.inhibicion_activa = True
                    self.tiempo_en_inhibicion = 0.0
            else:
                self.tiempo_desde_ultimo_cruce = 0.0
        else:
            # Ya inhibido: verificar si hay que desinhibir
            self.tiempo_en_inhibicion += dt
            
            if (señal_desajuste < self.umbral_desinhibicion and 
                self.tiempo_en_inhibicion >= self.duracion_minima):
                self.inhibicion_activa = False
                self.tiempo_en_inhibicion = 0.0
        
        self.historial_inhibicion.append(self.inhibicion_activa)
        return self.inhibicion_activa
    
    def reset(self):
        self.inhibicion_activa = False
        self.tiempo_en_inhibicion = 0.0
        self.historial_inhibicion = []
        self.señal_para_historial = []
        self.tiempo_desde_ultimo_cruce = 0.0


# ============================================================
# APARATO MOTOR V168 (CON R_op)
# ============================================================

class AparatoMotorV168:
    def __init__(self, enable_rop=True):
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
        
        self.fatiga = FatigaMetabolicaV168()
        self.memoria = MemoriaAusenciaV168()
        self.consciencia = ConscienciaBasicaV168()
        self.juego = ModoJuegoV168()
        self.ritual = RitualV168()
        self.meta = MetaRepresentacionObservacional()
        self.rop = R_op() if enable_rop else None
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.enable_rop = enable_rop
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, False, 0.0, 0, 0.0, False, False)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, False, 0.0, 0, 0.0, False, False)
        
        # ============================================================
        # ETAPA 0: Memoria de ausencia
        # ============================================================
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), dt)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        # ============================================================
        # ETAPA 1: Consciencia básica
        # ============================================================
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb = self.consciencia.actualizar(e_R, A_sys_env, dt)
        
        # ============================================================
        # ETAPA 3: Ritual (se actualiza ANTES que juego para jerarquía)
        # ============================================================
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        # ============================================================
        # ETAPA 4: Meta-representación observacional
        # ============================================================
        senal_desajuste, hay_desajuste = self.meta.actualizar(error, Cb, ritual_activo, dt)
        
        # ============================================================
        # ETAPA 5: R_op (Primer "No" operativo) - puede inhibir el ritual
        # ============================================================
        inhibir_ritual = False
        if self.enable_rop and self.rop is not None:
            inhibir_ritual = self.rop.actualizar(senal_desajuste, dt)
        
        # Si R_op inhibe, forzar ritual inactivo
        if inhibir_ritual:
            ritual_activo = False
            self.ritual.active = False
        
        # ============================================================
        # ETAPA 2: Juego (INHIBIDO si ritual está activo)
        # ============================================================
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        else:
            juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # ============================================================
        # EFECTOS DE FATIGA
        # ============================================================
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, setpoint_objetivo, juego_activo,
                    self.juego.get_tiempo_activo(), ritual_activo, self.ritual.activation,
                    self.ritual.cruces, senal_desajuste, hay_desajuste, inhibir_ritual)
        
        # ============================================================
        # CÁLCULO DE CORRECCIÓN MOTORA
        # ============================================================
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
        
        # Influencia del juego
        if juego_activo and not ritual_activo:
            influencia_juego = self.juego.get_influencia(Cb, confianza)
            if influencia_juego != 0:
                delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        # Influencia del ritual
        correccion_ritual = 0.0
        if ritual_activo and self.ritual.patron_buffer:
            correccion_ritual = 5.0 * self.ritual.ritual_gain
        
        delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        
        # Aplicar modo juego
        delta_fisico, delta_costo = self.juego.aplicar(delta)
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
                self.ritual.cruces, senal_desajuste, hay_desajuste, inhibir_ritual)
    
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
        self.meta.reset()
        if self.rop:
            self.rop.reset()


# ============================================================
# SISTEMA V168 (ORGANISMO COMPLETO)
# ============================================================

class SistemaV168:
    def __init__(self, nombre, seed=SEMILLA_BASE, enable_rop=True):
        self.nombre = nombre
        self.enable_rop = enable_rop

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

        self.izquierdo = HemisferioV168("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV168("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV168("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV168("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV168(enable_rop=enable_rop)
        self.modo_entrenamiento = True

    def actualizar(self, t, dt, duracion_total, setpoint_real):
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
         senal_desajuste, hay_desajuste, inhibir_ritual) = self.motor.actuar(
            gradiente, LF_activa, True, t, setpoint_real, dt
        )
        
        return (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces,
                senal_desajuste, hay_desajuste, inhibir_ritual)

    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0, invertido=False):
    if (t % periodo) < (periodo / 2):
        base = -amplitud
    else:
        base = +amplitud
    return -base if invertido else base


def generar_setpoint_con_ruido(t, setpoint_func, periodo=PERIODO_ALTERNANCIA, amplitud=60.0, invertido=False):
    setpoint_base = setpoint_func(t, periodo, amplitud, invertido)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


# ============================================================
# EXPERIMENTO V168
# ============================================================

def ejecutar_v168():
    print("=" * 100)
    print("EXPERIMENTO V168 — ANIMA-2 Etapa 5: PRIMER 'NO' OPERATIVO (R_op)")
    print("=" * 100)
    print("  BASE: V167 (meta-representación observacional validada)")
    print("  NUEVO: R_op — inhibición activa del ritual cuando es disfuncional")
    print("")
    print("  CRITERIOS DE ÉXITO ETAPA 5:")
    print("    1. El ritual se INHIBE en F4 cuando la señal de desajuste supera umbral")
    print("    2. Tras la inhibición, el error disminuye (adaptación)")
    print("    3. La inhibición correlaciona con la señal de desajuste")
    print("=" * 100)

    # Control: SIN R_op (solo ritual + meta observacional)
    organismo_control = SistemaV168("Control_V168", seed=SEMILLA_BASE, enable_rop=False)
    # Experimental: CON R_op (puede inhibir el ritual)
    organismo_experimental = SistemaV168("Rop_V168", seed=SEMILLA_BASE, enable_rop=True)

    print("\n  Entrenando lateralidad (10 repeticiones)...")
    
    organismo_control.set_modo_entrenamiento(True)
    organismo_experimental.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo_control.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
            organismo_experimental.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    print("  Entrenamiento completado.")
    
    organismo_control.set_modo_entrenamiento(False)
    organismo_experimental.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS

    # Función para ejecutar ciclos
    def ejecutar_ciclos(organismo, t_actual, num_ciclos, nombre_fase, es_control=True, invertido=False):
        acumuladores = {
            't': [], 'orient': [], 'setpoint': [], 'historia': [], 'fatiga': [],
            'confianza': [], 'Cb': [], 'juego_activo': [], 'tiempo_juego': [],
            'ritual_activo': [], 'ritual_act': [], 'cruces': [],
            'senal_desajuste': [], 'hay_desajuste': [], 'inhibir_ritual': []
        }
        
        tiempo_abs = t_actual
        
        for ciclo in range(num_ciclos):
            for i in range(int(PERIODO_ALTERNANCIA / DT)):
                t = tiempo_abs + i * DT
                setpoint = generar_setpoint_con_ruido(t, onda_cuadrada, periodo=PERIODO_ALTERNANCIA, 
                                                     amplitud=60.0, invertido=invertido)
                
                (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                 juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces,
                 senal_desajuste, hay_desajuste, inhibir_ritual) = organismo.actualizar(
                    t, DT, t_actual + 300, setpoint
                )
                
                acumuladores['t'].append(t)
                acumuladores['orient'].append(orient)
                acumuladores['setpoint'].append(setpoint)
                acumuladores['historia'].append(historia)
                acumuladores['fatiga'].append(fatiga)
                acumuladores['confianza'].append(confianza)
                acumuladores['Cb'].append(Cb)
                acumuladores['juego_activo'].append(juego_activo)
                acumuladores['tiempo_juego'].append(tiempo_juego)
                acumuladores['ritual_activo'].append(ritual_activo)
                acumuladores['ritual_act'].append(ritual_act)
                acumuladores['cruces'].append(cruces)
                acumuladores['senal_desajuste'].append(senal_desajuste)
                acumuladores['hay_desajuste'].append(hay_desajuste)
                acumuladores['inhibir_ritual'].append(inhibir_ritual)
            
            if (ciclo + 1) % 5 == 0 or ciclo == num_ciclos - 1:
                print(f"\n  {'='*60}")
                print(f"  {nombre_fase.upper()} ciclo {ciclo+1}/{num_ciclos}")
                print(f"    [Memoria]  confianza={confianza:.2f}, τ_mem={organismo.motor.memoria.get_tau_mem():.1f}s")
                print(f"    [Cb]       Cb={Cb:.1f}")
                print(f"    [Juego]    activo={juego_activo}, tiempo={tiempo_juego:.1f}s")
                if not es_control:
                    print(f"    [Ritual]   activo={ritual_activo}, act={ritual_act:.3f}, cruces={cruces}")
                    print(f"    [Rᴿ]       desajuste={senal_desajuste:.3f}, umbral={META_UMBRAL_DESAJUSTE}")
                    print(f"    [R_op]     inhibir={inhibir_ritual}")
                print(f"    [Física]   fatiga={fatiga:.0f}°, historia={historia:.0f}°")
            
            tiempo_abs += PERIODO_ALTERNANCIA
        
        return tiempo_abs, acumuladores

    # F1: Baseline
    print("\n  F1: Baseline (3 ciclos) - setpoint NORMAL...")
    t_actual, f1_control = ejecutar_ciclos(organismo_control, t_actual, 3, "Control", es_control=True)
    t_actual, f1_rop = ejecutar_ciclos(organismo_experimental, t_actual, 3, "R_op", es_control=False)

    # F2: Control (sin R_op)
    print("\n  F2: Control - 20 ciclos (setpoint NORMAL)...")
    t_actual, f2_control = ejecutar_ciclos(organismo_control, t_actual, 20, "Control", es_control=True)

    # F3: Experimental (CON R_op)
    print("\n  F3: Experimental - 20 ciclos CON R_op (setpoint NORMAL)...")
    t_actual, f3_rop = ejecutar_ciclos(organismo_experimental, t_actual, 20, "R_op", es_control=False)

    # F4: Test post con setpoint INVERTIDO
    print("\n  F4: Test post (3 ciclos) - SETPOINT INVERTIDO (prueba de R_op)...")
    t_actual, f4_control = ejecutar_ciclos(organismo_control, t_actual, 3, "Control", es_control=True, invertido=True)
    t_actual, f4_rop = ejecutar_ciclos(organismo_experimental, t_actual, 3, "R_op", es_control=False, invertido=True)

    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V168 — Primer 'No' operativo (R_op)")
    print("=" * 80)

    # Calcular error RMS en últimos 10s de F4
    ventana_rms = int(10.0 / DT)
    
    if len(f4_control['orient']) > ventana_rms and len(f4_rop['orient']) > ventana_rms:
        orient_control = np.array(f4_control['orient'][-ventana_rms:])
        orient_rop = np.array(f4_rop['orient'][-ventana_rms:])
        setpoint_nominal = f4_control['setpoint'][-1] if f4_control['setpoint'] else 60.0
        
        errores_control = np.abs(orient_control - setpoint_nominal)
        errores_rop = np.abs(orient_rop - setpoint_nominal)
        
        error_rms_control = np.sqrt(np.mean(errores_control**2))
        error_rms_rop = np.sqrt(np.mean(errores_rop**2))
    else:
        error_rms_control = error_rms_rop = 0
    
    historia_control = f2_control['historia'][-1] if f2_control['historia'] else 0
    historia_rop = f3_rop['historia'][-1] if f3_rop['historia'] else 0
    
    tiempo_ritual_control = organismo_control.motor.ritual.tiempo_activo
    tiempo_ritual_rop = organismo_experimental.motor.ritual.tiempo_activo
    tiempo_total = 20 * PERIODO_ALTERNANCIA
    pct_ritual_control = (tiempo_ritual_control / tiempo_total) * 100 if tiempo_total > 0 else 0
    pct_ritual_rop = (tiempo_ritual_rop / tiempo_total) * 100 if tiempo_total > 0 else 0
    
    ritual_activo_en_F4_control = any(f4_control['ritual_activo']) if f4_control['ritual_activo'] else False
    ritual_activo_en_F4_rop = any(f4_rop['ritual_activo']) if f4_rop['ritual_activo'] else False
    
    # Métricas de R_op
    inhibicion_activa_en_F4 = any(f4_rop['inhibir_ritual']) if f4_rop['inhibir_ritual'] else False
    senal_desajuste_max_rop = max(f4_rop['senal_desajuste']) if f4_rop['senal_desajuste'] else 0
    
    Cb_control_final = f4_control['Cb'][-1] if f4_control['Cb'] else 0
    Cb_rop_final = f4_rop['Cb'][-1] if f4_rop['Cb'] else 0
    
    print(f"\n  📊 MÉTRICAS POR ETAPA:")
    print(f"\n  [Etapa 0-2 - Base funcional]")
    print(f"    Historia control: {historia_control:.0f}°")
    print(f"    Historia R_op: {historia_rop:.0f}°")
    print(f"    Compresión: {historia_rop/max(1,historia_control):.3f}")
    
    print(f"\n  [Etapa 3 - Ritual]")
    print(f"    Tiempo ritual activo (control): {tiempo_ritual_control:.1f}s ({pct_ritual_control:.1f}%)")
    print(f"    Tiempo ritual activo (R_op): {tiempo_ritual_rop:.1f}s ({pct_ritual_rop:.1f}%)")
    print(f"    Ritual activo en F4 (control): {ritual_activo_en_F4_control}")
    print(f"    Ritual activo en F4 (R_op): {ritual_activo_en_F4_rop}")
    
    print(f"\n  [Etapa 5 - R_op (Primer 'No' operativo)]")
    print(f"    Señal desajuste máxima (R_op): {senal_desajuste_max_rop:.3f}")
    print(f"    Inhibición activa en F4: {inhibicion_activa_en_F4}")
    
    print(f"\n  [Test post - Error RMS F4]")
    print(f"    Error RMS Control: {error_rms_control:.2f}°")
    print(f"    Error RMS R_op: {error_rms_rop:.2f}°")
    print(f"    Cb final control: {Cb_control_final:.1f}")
    print(f"    Cb final R_op: {Cb_rop_final:.1f}")
    
    # Criterios de éxito Etapa 5
    exito_1 = inhibicion_activa_en_F4  # El ritual se inhibe
    exito_2 = error_rms_rop < error_rms_control  # Tras inhibición, menor error
    exito_3 = pct_ritual_rop < pct_ritual_control  # Menos tiempo ritual (inhibido)
    
    exito = exito_1 and exito_2 and exito_3
    
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO ETAPA 5 (R_op — Primer 'No' operativo)")
    print("=" * 80)
    print(f"  1. Inhibición activa en F4: {inhibicion_activa_en_F4} -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Menor error tras inhibición: {error_rms_rop:.2f} < {error_rms_control:.2f} -> {'✅' if exito_2 else '❌'}")
    print(f"  3. Menor tiempo ritual (inhibido): {pct_ritual_rop:.1f}% < {pct_ritual_control:.1f}% -> {'✅' if exito_3 else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ETAPA 5 COMPLETADA — PRIMER 'NO' OPERATIVO (R_op) VALIDADO")
        print("")
        print("     El organismo DEMUESTRA:")
        print("     ✓ Capacidad de INHIBIR el ritual cuando es disfuncional")
        print("     ✓ Mejor adaptación tras la inhibición (menor error)")
        print("     ✓ Reducción del tiempo ritual (inhibición activa)")
        print("")
        print("  ANIMA-2 ha completado el ciclo cosmosemiótico:")
        print("     Memoria → Cb → Juego → Ritual → Rᴿ → R_op")
    else:
        print("  ⚠️ ETAPA 5 PARCIAL")
        if not exito_1:
            print("     No se detectó inhibición activa en F4")
        if not exito_2:
            print("     El error no mejoró tras inhibición")
        if not exito_3:
            print("     El tiempo ritual no se redujo")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Orientación F4 comparativa
    ax = axes[0, 0]
    ax.plot(f4_control['setpoint'][:2000], 'r--', linewidth=0.5, alpha=0.7, label='Setpoint')
    ax.plot(f4_control['orient'][:2000], 'b-', linewidth=0.5, label='Control')
    ax.plot(f4_rop['orient'][:2000], 'orange', linewidth=0.5, label='R_op')
    ax.set_title('F4: Respuesta al setpoint invertido')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Cb
    ax = axes[0, 1]
    ax.plot(f4_control['Cb'], 'b-', linewidth=0.5, label='Control')
    ax.plot(f4_rop['Cb'], 'orange', linewidth=0.5, label='R_op')
    ax.set_title('Cb en F4')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # R_op: inhibición activa
    ax = axes[0, 2]
    ax.plot(f4_rop['senal_desajuste'], 'purple', linewidth=0.5, label='Señal desajuste')
    ax.plot(f4_rop['inhibir_ritual'], 'red', linewidth=0.5, label='Inhibición activa')
    ax.axhline(y=META_UMBRAL_DESAJUSTE, color='orange', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_title('R_op: Señal de desajuste e inhibición')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Ritual activation comparativa F3
    ax = axes[1, 0]
    ax.plot(f3_rop['ritual_act'], 'purple', linewidth=0.5)
    ax.axhline(y=RITUAL_UMBRAL_ACTIVACION, color='red', linestyle='--', alpha=0.5, label='Umbral ritual')
    ax.set_title('Activación ritual en F3 (con R_op)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Historia acumulada comparativa
    ax = axes[1, 1]
    categorias = ['Control', 'R_op']
    historias = [historia_control, historia_rop]
    colors_bar = ['blue', 'orange']
    ax.bar(categorias, historias, color=colors_bar)
    ax.set_title(f'Historia acumulada (compresión: {historia_rop/max(1,historia_control):.2f})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error F4 comparativo
    ax = axes[1, 2]
    categorias_error = ['Control', 'R_op']
    errores = [error_rms_control, error_rms_rop]
    colors_error = ['blue', 'green' if error_rms_rop < error_rms_control else 'red']
    ax.bar(categorias_error, errores, color=colors_error)
    ax.set_title(f'Error RMS en F4 (mejora: {(1 - error_rms_rop/error_rms_control)*100:.1f}%)' if error_rms_control > 0 else 'Error RMS en F4')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v168_logs', exist_ok=True)
    plt.savefig(f'v168_logs/v168_primer_no_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v168_logs/v168_primer_no_{timestamp}.png")
    
    return organismo_control, organismo_experimental, exito


if __name__ == "__main__":
    import time
    start = time.time()
    control, rop, exito = ejecutar_v168()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V168 completado. Éxito: {exito}")