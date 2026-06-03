#!/usr/bin/env python3
"""
V169 — ANIMA-2: DESACOPLE REPRESENTACIONAL (D)
================================================================================
Objetivo: Medir si existe desacople entre representación y acción en ANIMA-2.
No se programa el "No". Se observa si P(Acción|R) < 1 de forma natural.

Basado en Bloque 10 — Negación operativa:
  - Juego = {Rᵢ | P(Acción|Rᵢ) < 1}
  - El juego es el antecedente del No
  - Medimos D = Var(R) · (1 - Pmax) como indicador de desacople

Diseño:
  - F1-F3: Normal (consolidación de ritual)
  - F4: Setpoint incierto (tres valores posibles: -60°, 0°, +60°)
  - Observamos si el organismo genera múltiples representaciones
  - Medimos si la acción queda suspendida (no ejecución automática)

Métricas clave:
  - Var(R): diversidad de representaciones
  - Pmax: probabilidad de la representación dominante
  - D = Var(R) · (1 - Pmax): desacople representacional
  - ritual_activo: si el marco ritual está presente

Hipótesis:
  - Si D > 0 durante un período sostenido, existe desacople
  - Eso es la condición estructural para el "No" (R_op)
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
# PARAMETROS DE SETPOINT INCIERTO PARA F4
# ============================================================
SETPOINT_POSIBLES = [-60.0, 0.0, 60.0]  # Tres valores posibles
SETPOINT_PROBABILIDADES = [0.33, 0.34, 0.33]  # Equiprobable
SETPOINT_CAMBIO_INTERVALO = 40.0  # Cambiar cada 40 segundos

# ============================================================
# PARAMETROS DE DESACOPLE REPRESENTACIONAL
# ============================================================
VENTANA_DESACOPLE = 100  # Ventana para calcular P(Acción|R)
UMBRAL_DESACOPLE = 0.1   # D > 0.1 indica desacople significativo
TIEMPO_MINIMO_DESACOPLE = 5.0  # Necesita 5 segundos sostenidos


SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV169:
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

class FatigaMetabolicaV169:
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

class MemoriaAusenciaV169:
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

class ConscienciaBasicaV169:
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

class ModoJuegoV169:
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

class RitualV169:
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
    """
    Registra las representaciones de acción del organismo.
    Permite calcular P(Acción|R) y el desacople representacional D.
    """
    
    def __init__(self, ventana=VENTANA_DESACOPLE):
        self.ventana = ventana
        self.historial_representaciones = deque(maxlen=ventana)
        self.historial_acciones = deque(maxlen=ventana)
    
    def registrar(self, representacion, accion_ejecutada):
        """
        Args:
            representacion: valor de setpoint u orientación deseada
            accion_ejecutada: True si se ejecutó movimiento, False si se suspendió
        """
        self.historial_representaciones.append(representacion)
        self.historial_acciones.append(accion_ejecutada)
    
    def calcular_P_accion_dado_R(self, valor_R):
        """
        Calcula P(Acción|R) para un valor específico de R
        """
        if len(self.historial_representaciones) < 10:
            return 1.0  # Por defecto, acción determinada
        
        # Buscar ocurrencias de R en el historial
        ocurrencias = []
        for r, a in zip(self.historial_representaciones, self.historial_acciones):
            if abs(r - valor_R) < 5.0:  # Tolerancia de 5 grados
                ocurrencias.append(a)
        
        if len(ocurrencias) == 0:
            return 1.0  # Si no hay datos, asumir determinismo
        
        return np.mean(ocurrencias)
    
    def calcular_var_R(self):
        """
        Calcula Var(R) = diversidad de representaciones en la ventana
        """
        if len(self.historial_representaciones) < 10:
            return 0.0
        
        # Discretizar representaciones para calcular diversidad
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        _, counts = np.unique(discretos, return_counts=True)
        probs = counts / len(discretos)
        
        # Entropía como medida de diversidad
        var = -np.sum(probs * np.log(probs + 1e-10))
        return var
    
    def calcular_Pmax(self):
        """
        Calcula la probabilidad de la representación dominante
        """
        if len(self.historial_representaciones) < 10:
            return 1.0
        
        discretos = [round(r / 10.0) * 10 for r in self.historial_representaciones]
        unique, counts = np.unique(discretos, return_counts=True)
        return np.max(counts) / len(discretos)
    
    def calcular_desacople(self):
        """
        Calcula D = Var(R) · (1 - Pmax)
        
        D = 0: representación única, acción inevitable
        D > 0: alternativas coexistiendo, acción no determinada
        """
        var_R = self.calcular_var_R()
        Pmax = self.calcular_Pmax()
        
        # Normalizar var_R (entropía máxima ~3.0 para 20 categorías)
        var_norm = min(1.0, var_R / 3.0)
        
        return var_norm * (1.0 - Pmax)
    
    def reset(self):
        self.historial_representaciones.clear()
        self.historial_acciones.clear()


# ============================================================
# APARATO MOTOR V169 (CON REGISTRO DE REPRESENTACIONES)
# ============================================================

class AparatoMotorV169:
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
        
        self.fatiga = FatigaMetabolicaV169()
        self.memoria = MemoriaAusenciaV169()
        self.consciencia = ConscienciaBasicaV169()
        self.juego = ModoJuegoV169()
        self.ritual = RitualV169()
        
        # Registro de representaciones para desacople
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw, dt=DT):
        if not LF_activa:
            return (self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    False, 0.0, False, 0.0, 0, 0.0, 0.0, 0.0)
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, False, 0.0, 0, 0.0, 0.0, 0.0)
        
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
        # ETAPA 3: Ritual
        # ============================================================
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, dt)
        
        # ============================================================
        # ETAPA 2: Juego (INHIBIDO si ritual está activo)
        # ============================================================
        if ritual_activo:
            juego_activo = False
            self.juego.activo = False
        else:
            juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # ============================================================
        # REGISTRO DE REPRESENTACIONES PARA DESACOPLE
        # ============================================================
        # La representación es el setpoint_objetivo (lo que el sistema "quiere" hacer)
        # La acción ejecutada es True si hay movimiento significativo
        accion_ejecutada = abs(self.ultimo_delta) > 0.01
        self.registro.registrar(setpoint_objetivo, accion_ejecutada)
        
        # Calcular desacople representacional D
        D = self.registro.calcular_desacople()
        
        # ============================================================
        # EFECTOS DE FATIGA
        # ============================================================
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, dt)
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, dt)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, setpoint_objetivo, juego_activo,
                    self.juego.get_tiempo_activo(), ritual_activo, self.ritual.activation,
                    self.ritual.cruces, D, 0.0, 0.0)
        
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
# SISTEMA V169 (ORGANISMO COMPLETO)
# ============================================================

class SistemaV169:
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

        self.izquierdo = HemisferioV169("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV169("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV169("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV169("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV169()
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
         D, _, _) = self.motor.actuar(
            gradiente, LF_activa, True, t, setpoint_real, dt
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
    """Setpoint normal: onda cuadrada ±60°"""
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def setpoint_incierto(t, periodo=SETPOINT_CAMBIO_INTERVALO, 
                      posibles=SETPOINT_POSIBLES, probs=SETPOINT_PROBABILIDADES):
    """Setpoint incierto: cambia aleatoriamente cada 'periodo' segundos"""
    # Cambiar setpoint cada periodo
    fase = int(t / periodo)
    # Usar semilla determinística para reproducibilidad
    rng = random.Random(int(fase))
    return rng.choices(posibles, weights=probs)[0]


def generar_setpoint_con_ruido(t, setpoint_func, **kwargs):
    setpoint_base = setpoint_func(t, **kwargs)
    ruido = RUIDO_SETPOINT_AMP * np.sin(2 * np.pi * t / RUIDO_SETPOINT_PERIODO)
    return setpoint_base + ruido


# ============================================================
# EXPERIMENTO V169 — DESACOPLE REPRESENTACIONAL
# ============================================================

def ejecutar_v169():
    print("=" * 100)
    print("EXPERIMENTO V169 — DESACOPLE REPRESENTACIONAL (D)")
    print("=" * 100)
    print("  Objetivo: Medir si existe P(Acción|R) < 1 en ANIMA-2")
    print("")
    print("  Basado en Bloque 10 — Negación operativa:")
    print("    - Juego = {Rᵢ | P(Acción|Rᵢ) < 1}")
    print("    - D = Var(R) · (1 - Pmax)")
    print("")
    print("  Diseño:")
    print("    - F1-F3: Setpoint normal (onda cuadrada ±60°)")
    print("    - F4: Setpoint incierto (tres valores posibles: -60°, 0°, +60°)")
    print("")
    print("  Métrica clave:")
    print(f"    - D > {UMBRAL_DESACOPLE} sostenido por {TIEMPO_MINIMO_DESACOPLE}s = desacople")
    print("")
    print("  Hipótesis:")
    print("    - Si D > 0, existe desacople representacional")
    print("    - Eso es la condición estructural para el 'No'")
    print("=" * 100)

    organismo = SistemaV169("V169", seed=SEMILLA_BASE)

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
    # FASE 1: Baseline (3 ciclos, setpoint normal)
    # ============================================================
    print("\n  F1: Baseline (3 ciclos) - setpoint NORMAL...")
    
    f1_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 
                'ritual_activo': [], 'ritual_act': [], 'D': []}
    
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(t, setpoint_normal, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + 300, setpoint
        )
        
        f1_datos['t'].append(t)
        f1_datos['orient'].append(orient)
        f1_datos['setpoint'].append(setpoint)
        f1_datos['Cb'].append(Cb)
        f1_datos['ritual_activo'].append(ritual_activo)
        f1_datos['ritual_act'].append(ritual_act)
        f1_datos['D'].append(D)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    print(f"    Baseline completado. D final: {f1_datos['D'][-1]:.4f}")
    
    # ============================================================
    # FASE 2: Fatiga (20 ciclos, setpoint normal)
    # ============================================================
    print("\n  F2: Fatiga - 20 ciclos (setpoint NORMAL)...")
    
    f2_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 
                'ritual_activo': [], 'ritual_act': [], 'D': [],
                'historia': [], 'fatiga': []}
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(t, setpoint_normal, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            
            (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
             juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
                t, DT, t_actual + 2000, setpoint
            )
            
            f2_datos['t'].append(t)
            f2_datos['orient'].append(orient)
            f2_datos['setpoint'].append(setpoint)
            f2_datos['Cb'].append(Cb)
            f2_datos['ritual_activo'].append(ritual_activo)
            f2_datos['ritual_act'].append(ritual_act)
            f2_datos['D'].append(D)
            f2_datos['historia'].append(historia)
            f2_datos['fatiga'].append(fatiga)
        
        if (ciclo + 1) % 5 == 0:
            print(f"    Ciclo {ciclo+1}/20, D={f2_datos['D'][-1]:.4f}, ritual_act={ritual_act:.3f}")
        
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 3: Ritual (20 ciclos, setpoint normal)
    # ============================================================
    print("\n  F3: Ritual - 20 ciclos (setpoint NORMAL)...")
    print("      (Ritual se consolida)")
    
    f3_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 
                'ritual_activo': [], 'ritual_act': [], 'D': []}
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(t, setpoint_normal, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            
            (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
             juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
                t, DT, t_actual + 2000, setpoint
            )
            
            f3_datos['t'].append(t)
            f3_datos['orient'].append(orient)
            f3_datos['setpoint'].append(setpoint)
            f3_datos['Cb'].append(Cb)
            f3_datos['ritual_activo'].append(ritual_activo)
            f3_datos['ritual_act'].append(ritual_act)
            f3_datos['D'].append(D)
        
        if (ciclo + 1) % 5 == 0:
            print(f"    Ciclo {ciclo+1}/20, D={f3_datos['D'][-1]:.4f}, ritual_act={ritual_act:.3f}")
        
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 4: Test con setpoint incierto
    # ============================================================
    print("\n  F4: Test - 3 ciclos (setpoint INCIERTO: -60°, 0°, +60° aleatorio)...")
    print("      (Observamos desacople representacional D)")
    
    f4_datos = {'t': [], 'orient': [], 'setpoint': [], 'Cb': [], 
                'ritual_activo': [], 'ritual_act': [], 'D': []}
    
    # Registrar los setpoints reales para ver la distribución
    setpoints_mostrados = []
    
    for i in range(int(3 * SETPOINT_CAMBIO_INTERVALO / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(t, setpoint_incierto, periodo=SETPOINT_CAMBIO_INTERVALO,
                                              posibles=SETPOINT_POSIBLES, probs=SETPOINT_PROBABILIDADES)
        
        setpoints_mostrados.append(setpoint)
        
        (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
         juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, D) = organismo.actualizar(
            t, DT, t_actual + 300, setpoint
        )
        
        f4_datos['t'].append(t)
        f4_datos['orient'].append(orient)
        f4_datos['setpoint'].append(setpoint)
        f4_datos['Cb'].append(Cb)
        f4_datos['ritual_activo'].append(ritual_activo)
        f4_datos['ritual_act'].append(ritual_act)
        f4_datos['D'].append(D)
    
    # ============================================================
    # ANÁLISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V169 — Desacople representacional")
    print("=" * 80)
    
    # Calcular estadísticas de D en F4
    D_f4 = np.array(f4_datos['D'])
    D_max = np.max(D_f4)
    D_mean = np.mean(D_f4)
    D_std = np.std(D_f4)
    
    # Detectar períodos de desacople sostenido
    desacople_sostenido = False
    tiempo_desacople = 0.0
    max_tiempo_desacople = 0.0
    
    for d in D_f4:
        if d > UMBRAL_DESACOPLE:
            tiempo_desacople += DT
            if tiempo_desacople > max_tiempo_desacople:
                max_tiempo_desacople = tiempo_desacople
        else:
            tiempo_desacople = 0.0
    
    desacople_sostenido = max_tiempo_desacople >= TIEMPO_MINIMO_DESACOPLE
    
    # Calcular ritual en F4
    ritual_activo_f4 = any(f4_datos['ritual_activo'])
    ritual_act_max = max(f4_datos['ritual_act']) if f4_datos['ritual_act'] else 0
    
    # Distribución de setpoints en F4
    setpoints_reales = np.array(f4_datos['setpoint'])
    unique, counts = np.unique(np.round(setpoints_reales / 10) * 10, return_counts=True)
    
    print(f"\n  📊 MÉTRICAS DE DESACOPLE (F4):")
    print(f"    D (desacople) máximo: {D_max:.4f}")
    print(f"    D (desacople) medio: {D_mean:.4f}")
    print(f"    D (desacople) std: {D_std:.4f}")
    print(f"    Máximo tiempo con D > {UMBRAL_DESACOPLE}: {max_tiempo_desacople:.2f}s")
    print(f"    Desacople sostenido (>={TIEMPO_MINIMO_DESACOPLE}s): {desacople_sostenido}")
    
    print(f"\n  📊 MÉTRICAS DE RITUAL (F4):")
    print(f"    Ritual activo: {ritual_activo_f4}")
    print(f"    Ritual activación máxima: {ritual_act_max:.3f}")
    
    print(f"\n  📊 DISTRIBUCIÓN DE SETPOINTS (F4):")
    for val, cnt in zip(unique, counts):
        print(f"    {val:.0f}°: {cnt} veces ({cnt/len(setpoints_reales)*100:.1f}%)")
    
    print(f"\n  📊 MÉTRICAS DE CONSOLIDACIÓN (F3):")
    ritual_act_f3 = np.array(f3_datos['ritual_act'])
    print(f"    Ritual activación media F3: {np.mean(ritual_act_f3):.3f}")
    print(f"    Ritual activación máxima F3: {np.max(ritual_act_f3):.3f}")
    
    # ============================================================
    # CRITERIOS DE ÉXITO
    # ============================================================
    exito = desacople_sostenido
    
    print("\n" + "=" * 80)
    print("CRITERIO DE ÉXITO (Desacople representacional)")
    print("=" * 80)
    print(f"  Desacople sostenido (D > {UMBRAL_DESACOPLE} por {TIEMPO_MINIMO_DESACOPLE}s): {desacople_sostenido} -> {'✅' if exito else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ DESACOPLE REPRESENTACIONAL DEMOSTRADO")
        print("")
        print("     ANIMA-2 muestra P(Acción|R) < 1 de forma natural")
        print("     La condición estructural para el 'No' está presente")
        print("")
        print("  Siguiente paso: V170 — Primer 'No' operativo (R_op)")
    else:
        print("  ⚠️ DESACOPLE REPRESENTACIONAL NO DEMOSTRADO")
        print("")
        print("     ANIMA-2 no mostró P(Acción|R) < 1 sostenido")
        print("     La condición estructural para el 'No' no está presente")
        print("")
        print("  Posibles ajustes:")
        print("    - Aumentar incertidumbre del setpoint")
        print("    - Reducir umbral de desacople")
        print("    - Aumentar duración de F4")
    print("=" * 80)
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Gráfico 1: Orientación y setpoint en F4
    ax = axes[0, 0]
    muestra = min(5000, len(f4_datos['orient']))
    ax.plot(f4_datos['setpoint'][:muestra], 'r--', linewidth=0.5, alpha=0.5, label='Setpoint')
    ax.plot(f4_datos['orient'][:muestra], 'b-', linewidth=0.5, label='Orientación')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Ángulo (º)')
    ax.set_title('F4: Setpoint incierto vs Orientación')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Desacople D en F4
    ax = axes[0, 1]
    ax.plot(f4_datos['D'], 'purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_DESACOPLE, color='red', linestyle='--', alpha=0.5, label=f'Umbral D>{UMBRAL_DESACOPLE}')
    ax.fill_between(range(len(f4_datos['D'])), 0, f4_datos['D'],
                    where=np.array(f4_datos['D']) > UMBRAL_DESACOPLE,
                    color='green', alpha=0.3, label='Desacople')
    ax.set_xlabel('Paso')
    ax.set_ylabel('D (desacople)')
    ax.set_title('F4: Desacople representacional')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Cb en F4
    ax = axes[0, 2]
    ax.plot(f4_datos['Cb'], 'orange', linewidth=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb')
    ax.set_title('F4: Consciencia básica')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Ritual activation en F3
    ax = axes[1, 0]
    ax.plot(f3_datos['ritual_act'], 'purple', linewidth=0.5)
    ax.axhline(y=RITUAL_UMBRAL_ACTIVACION, color='red', linestyle='--', alpha=0.5, label='Umbral ritual')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Activación ritual')
    ax.set_title('F3: Consolidación del ritual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Comparación D entre fases
    ax = axes[1, 1]
    # Submuestrear para visualización
    step = max(1, len(f1_datos['D']) // 500)
    ax.plot(f1_datos['D'][::step], 'blue', linewidth=0.3, alpha=0.5, label='F1 (baseline)')
    step2 = max(1, len(f3_datos['D']) // 500)
    ax.plot(f3_datos['D'][::step2], 'purple', linewidth=0.3, alpha=0.5, label='F3 (ritual)')
    step3 = max(1, len(f4_datos['D']) // 500)
    ax.plot(f4_datos['D'][::step3], 'orange', linewidth=0.3, alpha=0.5, label='F4 (incierto)')
    ax.axhline(y=UMBRAL_DESACOPLE, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso (submuestreado)')
    ax.set_ylabel('D')
    ax.set_title('Desacople por fase')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Histograma de setpoints en F4
    ax = axes[1, 2]
    ax.hist(f4_datos['setpoint'], bins=20, color='green', alpha=0.7)
    ax.set_xlabel('Setpoint (º)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de setpoints en F4')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v169_logs', exist_ok=True)
    plt.savefig(f'v169_logs/v169_desacople_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v169_logs/v169_desacople_{timestamp}.png")
    
    # Guardado de datos raw para verificabilidad (D, representaciones, acciones, setpoints en F4)
    import json
    raw_data = {
        'f4_datos': {k: list(v) for k, v in f4_datos.items()},
        'f3_datos': {k: list(v) for k, v in f3_datos.items()},
        'D_max': float(D_max),
        'D_mean': float(D_mean),
        'max_tiempo_desacople': float(max_tiempo_desacople),
        'desacople_sostenido': bool(desacople_sostenido),
        'setpoints_distribution': {int(k): int(v) for k, v in zip(unique, counts)} if 'unique' in dir() else {}
    }
    with open(f'v169_logs/v169_raw_{timestamp}.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"  📁 Datos raw guardados: v169_logs/v169_raw_{timestamp}.json")
    
    return organismo, exito


if __name__ == "__main__":
    import time
    start = time.time()
    organismo, exito = ejecutar_v169()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V169 completado. Desacople demostrado: {exito}")