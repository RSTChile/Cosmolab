#!/usr/bin/env python3
"""
V167 — ANIMA-2 Etapa 4: META-REPRESENTACIÓN OBSERVACIONAL (Rᴿ)
================================================================================
BASE: V166 (ritual validado)

Rᴿ: monitor de desajuste — SOLO OBSERVACIONAL (no inhibe ritual)

CRITERIOS V167:
  1. Ritual activo en F4 (persistencia, heredado de V166)
  2. Señal > 0.5 cuando ritual persiste con error alto
  3. Correlación positiva ritual_activo ↔ señal en ventanas de error sostenido
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque

# ============================================================
# PARAMETROS (DESDE V157, FUNCIONAN)
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

# PARAMETROS RITUAL (DESDE V165, FUNCIONAN)
RITUAL_TAU = 180.0
RITUAL_REPETICION_MIN = 3
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.4
RITUAL_UMBRAL_CB = 28.0
RITUAL_SALIDA_SUAVE = 0.95
RITUAL_PERSISTENCIA_MIN = 3

SEMILLA_BASE = 44
PERIODO_ALTERNANCIA = 80.0

# Meta-representación observacional (Rᴿ)
META_TAU = 30.0
META_UMBRAL_SENAL = 0.5
META_UMBRAL_ERROR = 15.0
META_VENTANA_ERROR = 200


# ============================================================
# HEMISFERIO (DESDE V157)
# ============================================================

class HemisferioV166:
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
# FATIGA METABOLICA (DESDE V157)
# ============================================================

class FatigaMetabolicaV166:
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
# MEMORIA DE AUSENCIA (DESDE V157)
# ============================================================

class MemoriaAusenciaV166:
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
        return 1.0  # Placeholder
    
    def get_tau_mem(self):
        return self.tau_mem
    
    def reset(self):
        self.setpoint_last = 0.0
        self.t_ausencia = 0.0
        self.tau_mem = self.tau_base


# ============================================================
# CONSCIENCIA BÁSICA (DESDE V157)
# ============================================================

class ConscienciaBasicaV166:
    def __init__(self, tau_cb=TAU_CB, cb_max=CB_MAX):
        self.Cb = 0.0
        self.tau_cb = tau_cb
        self.cb_max = cb_max
    
    def actualizar(self, e_R, A_sys_env, dt):
        presion = e_R * (1.0 - A_sys_env)
        dCb_dt = presion - self.Cb / self.tau_cb
        self.Cb += dCb_dt * dt
        self.Cb = max(0.0, min(self.cb_max, self.Cb))
        return self.Cb
    
    def reset(self):
        self.Cb = 0.0


# ============================================================
# MODO JUEGO (DESDE V157)
# ============================================================

class ModoJuegoV166:
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
# RITUAL (CON DETECTOR DE CRUCES POR CERO, DESDE V162)
# ============================================================

class RitualV166:
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
            # Buscar patrón temporal con cruces previos
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
# META-REPRESENTACIÓN OBSERVACIONAL (Rᴿ) — sin inhibición
# ============================================================

class MetaRepresentacionObservacional:
    """Monitor de desajuste: ritual activo + error sostenido alto → señal integrada."""

    def __init__(self, tau=META_TAU, ventana_error=META_VENTANA_ERROR):
        self.tau = tau
        self.ventana_error = ventana_error
        self.desajuste = 0.0
        self.buffer_error = deque(maxlen=ventana_error)

    def actualizar(self, error, ritual_activo, dt):
        self.buffer_error.append(abs(error))
        if len(self.buffer_error) > self.ventana_error // 2:
            error_sostenido = np.mean(self.buffer_error)
        else:
            error_sostenido = abs(error)

        if ritual_activo and error_sostenido >= META_UMBRAL_ERROR:
            presion = min(1.0, error_sostenido / 60.0)
        else:
            presion = 0.0

        d_desajuste = presion - self.desajuste / self.tau
        self.desajuste += d_desajuste * dt
        self.desajuste = max(0.0, min(1.0, self.desajuste))
        return self.desajuste

    def reset(self):
        self.desajuste = 0.0
        self.buffer_error.clear()


# ============================================================
# APARATO MOTOR V166 (CON TODA LA DINÁMICA DE V157)
# ============================================================

class AparatoMotorV166:
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
        
        self.fatiga = FatigaMetabolicaV166()
        self.memoria = MemoriaAusenciaV166()
        self.consciencia = ConscienciaBasicaV166()
        self.juego = ModoJuegoV166()
        self.ritual = RitualV166()
        self.meta = MetaRepresentacionObservacional()
        
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
            return self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, False, 0.0, 0, 0.0
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, False, 0.0, 0, 0.0)
        
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
        
        # Etapa 4: Rᴿ observacional (no modifica ritual)
        senal_desajuste = self.meta.actualizar(error, ritual_activo, dt)
        
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
                    self.ritual.cruces, senal_desajuste)
        
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
        
        # Influencia del juego (solo si está activo y ritual NO activo)
        if juego_activo and not ritual_activo:
            influencia_juego = self.juego.get_influencia(Cb, confianza)
            if influencia_juego != 0:
                delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        # Influencia del ritual
        correccion_ritual = 0.0
        if ritual_activo and self.ritual.patron_buffer:
            correccion_ritual = 5.0 * self.ritual.ritual_gain  # Modulación simple
        
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
                self.ritual.cruces, senal_desajuste)
    
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


# ============================================================
# SISTEMA V166 (ORGANISMO COMPLETO)
# ============================================================

class SistemaV166:
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

        self.izquierdo = HemisferioV166("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV166("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV166("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV166("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV166()
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
         senal_desajuste) = self.motor.actuar(
            gradiente, LF_activa, True, t, setpoint_real, dt
        )
        
        return (orientacion, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces, senal_desajuste)

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
# EXPERIMENTO V167
# ============================================================

def _correlacion_f3_centra(f3_ritual):
    """Correlación directa ritual↔señal en ventana central de F3 (sin filtrar por ritual>0.5)."""
    if len(f3_ritual['ritual_activo']) > 100 and len(f3_ritual['senal_desajuste']) > 100:
        inicio = len(f3_ritual['ritual_activo']) // 4
        fin = 3 * len(f3_ritual['ritual_activo']) // 4
        ritual_vals = np.array(f3_ritual['ritual_activo'][inicio:fin], dtype=float)
        senal_vals = np.array(f3_ritual['senal_desajuste'][inicio:fin], dtype=float)
        if np.std(ritual_vals) > 1e-6 and np.std(senal_vals) > 1e-6:
            correlacion = np.corrcoef(ritual_vals, senal_vals)[0, 1]
            if np.isnan(correlacion):
                correlacion = 0.0
        else:
            correlacion = 0.0
        n_muestras = fin - inicio
    else:
        correlacion = 0.0
        n_muestras = 0
    return correlacion, n_muestras


def ejecutar_v167():
    print("=" * 100)
    print("EXPERIMENTO V167 — ANIMA-2 Etapa 4: META-REPRESENTACIÓN OBSERVACIONAL (Rᴿ)")
    print("=" * 100)
    print("  BASE: V166 | Rᴿ solo observa (no inhibe ritual)")
    print("  CRITERIOS:")
    print("    1. Ritual activo en F4")
    print("    2. Señal > 0.5 con ritual + error alto")
    print("    3. Correlación positiva ritual_activo ↔ señal (ventana central F3)")
    print("=" * 100)

    organismo_control = SistemaV166("Control_V166", seed=SEMILLA_BASE)
    organismo_ritual = SistemaV166("Ritual_V166", seed=SEMILLA_BASE + 1000)

    print("\n  Entrenando lateralidad (10 repeticiones)...")
    
    organismo_control.set_modo_entrenamiento(True)
    organismo_ritual.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            organismo_control.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
            organismo_ritual.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS, 0.0)
    
    print("  Entrenamiento completado.")
    
    organismo_control.set_modo_entrenamiento(False)
    organismo_ritual.set_modo_entrenamiento(False)
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS

    # Función para ejecutar ciclos
    def ejecutar_ciclos(organismo, t_actual, num_ciclos, nombre_fase, es_control=True, invertido=False):
        acumuladores = {
            't': [], 'orient': [], 'setpoint': [], 'historia': [], 'fatiga': [],
            'confianza': [], 'Cb': [], 'juego_activo': [], 'tiempo_juego': [],
            'ritual_activo': [], 'ritual_act': [], 'cruces': [],
            'senal_desajuste': [], 'error_abs': []
        }
        
        tiempo_abs = t_actual
        
        for ciclo in range(num_ciclos):
            for i in range(int(PERIODO_ALTERNANCIA / DT)):
                t = tiempo_abs + i * DT
                setpoint = generar_setpoint_con_ruido(t, onda_cuadrada, periodo=PERIODO_ALTERNANCIA, amplitud=60.0, invertido=invertido)
                
                (orient, historia, fatiga, confianza, zona_muerta, Cb, setpoint_objetivo,
                 juego_activo, tiempo_juego, ritual_activo, ritual_act, cruces,
                 senal_desajuste) = organismo.actualizar(
                    t, DT, t_actual + 300, setpoint
                )
                error_abs = abs(orient - setpoint)
                
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
                acumuladores['error_abs'].append(error_abs)
            
            if (ciclo + 1) % 5 == 0 or ciclo == num_ciclos - 1:
                print(f"\n  {'='*60}")
                print(f"  {nombre_fase.upper()} ciclo {ciclo+1}/{num_ciclos}")
                print(f"    [Memoria]  confianza={confianza:.2f}, τ_mem={organismo.motor.memoria.get_tau_mem():.1f}s")
                print(f"    [Cb]       Cb={Cb:.1f}")
                print(f"    [Juego]    activo={juego_activo}, tiempo={tiempo_juego:.1f}s")
                if not es_control:
                    print(f"    [Ritual]   activo={ritual_activo}, act={ritual_act:.3f}, cruces={cruces}")
                    print(f"    [Meta-R]   desajuste={senal_desajuste:.3f} (observacional)")
                print(f"    [Física]   fatiga={fatiga:.0f}°, historia={historia:.0f}°")
            
            tiempo_abs += PERIODO_ALTERNANCIA
        
        return tiempo_abs, acumuladores

    # F1: Baseline
    print("\n  F1: Baseline (3 ciclos) - setpoint NORMAL...")
    t_actual, f1_control = ejecutar_ciclos(organismo_control, t_actual, 3, "Control", es_control=True)
    t_actual, f1_ritual = ejecutar_ciclos(organismo_ritual, t_actual, 3, "Ritual", es_control=False)

    # F2: Control
    print("\n  F2: Control - 20 ciclos SIN ritual (setpoint NORMAL)...")
    t_actual, f2_control = ejecutar_ciclos(organismo_control, t_actual, 20, "Control", es_control=True)

    # F3: Experimental
    print("\n  F3: Experimental - 20 ciclos CON ritual (setpoint NORMAL)...")
    t_actual, f3_ritual = ejecutar_ciclos(organismo_ritual, t_actual, 20, "Ritual", es_control=False)

    # F4: Test post con setpoint INVERTIDO
    print("\n  F4: Test post (3 ciclos) - SETPOINT INVERTIDO (prueba Rᴿ)...")
    t_actual, f4_control = ejecutar_ciclos(organismo_control, t_actual, 3, "Control", es_control=True, invertido=True)
    t_actual, f4_ritual = ejecutar_ciclos(organismo_ritual, t_actual, 3, "Ritual", es_control=False, invertido=True)

    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V167 — Meta-representación observacional (Rᴿ)")
    print("=" * 80)

    ritual_activo_en_F4 = any(f4_ritual['ritual_activo']) if f4_ritual['ritual_activo'] else False
    ritual_activation_final = f4_ritual['ritual_act'][-1] if f4_ritual['ritual_act'] else 0
    senal_max_f4 = max(f4_ritual['senal_desajuste']) if f4_ritual['senal_desajuste'] else 0.0

    ritual_arr = np.array(f4_ritual['ritual_activo'], dtype=float)
    senal_arr = np.array(f4_ritual['senal_desajuste'], dtype=float)
    error_arr = np.array(f4_ritual['error_abs'], dtype=float)

    mask_ritual_error = (ritual_arr > 0.5) & (error_arr >= META_UMBRAL_ERROR)
    detectable_en_condicion = (
        np.any(senal_arr[mask_ritual_error] > META_UMBRAL_SENAL) if mask_ritual_error.any() else False
    )
    n_pasos_condicion = int(mask_ritual_error.sum())

    corr, n_ventana = _correlacion_f3_centra(f3_ritual)

    ventana_rms = int(10.0 / DT)
    if len(f4_ritual['orient']) > ventana_rms:
        error_rms_ritual = np.sqrt(np.mean(np.array(f4_ritual['error_abs'][-ventana_rms:])**2))
    else:
        error_rms_ritual = 0.0

    print(f"\n  📊 MÉTRICAS Rᴿ (F4):")
    print(f"    Ritual activo en F4: {ritual_activo_en_F4}")
    print(f"    Activación ritual final: {ritual_activation_final:.3f}")
    print(f"    Señal desajuste máxima: {senal_max_f4:.3f}")
    print(f"    Pasos ritual+error_alto (≥{META_UMBRAL_ERROR}°): {n_pasos_condicion}")
    print(f"    Señal > {META_UMBRAL_SENAL} en esas condiciones: {detectable_en_condicion}")
    print(f"    Correlación ritual↔señal (F3 ventana central): {corr:.3f} (n={n_ventana})")
    print(f"    Error RMS F4 ritual: {error_rms_ritual:.2f}°")

    exito_1 = ritual_activo_en_F4
    exito_2 = detectable_en_condicion
    exito_3 = corr > 0.0 and n_ventana >= 100

    exito = exito_1 and exito_2 and exito_3

    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO ETAPA 4 (V167 observacional)")
    print("=" * 80)
    print(f"  1. Ritual activo en F4: {ritual_activo_en_F4} -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Señal > {META_UMBRAL_SENAL} con ritual+error alto: {detectable_en_condicion} -> {'✅' if exito_2 else '❌'}")
    print(f"  3. Correlación positiva (r={corr:.3f}, n≥50): -> {'✅' if exito_3 else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ETAPA 4 COMPLETADA — Rᴿ OBSERVACIONAL VALIDADO")
        print("     El monitor correlaciona ritual activo con desajuste bajo error alto")
    else:
        print("  ⚠️ ETAPA 4 PARCIAL")
        if not exito_1:
            print("     Sin persistencia ritual en F4")
        if not exito_2:
            print("     Señal no detectable bajo ritual+error alto")
        if not exito_3:
            print("     Correlación insuficiente o pocas muestras")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(f4_ritual['setpoint'][:2000], 'r--', linewidth=0.5, alpha=0.7, label='Setpoint')
    ax.plot(f4_ritual['orient'][:2000], 'orange', linewidth=0.5, label='Orientación')
    ax.set_title('F4: Setpoint invertido')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(f4_ritual['ritual_activo'], 'g-', linewidth=0.4, alpha=0.7, label='Ritual activo')
    ax2 = ax.twinx()
    ax2.plot(f4_ritual['senal_desajuste'], 'purple', linewidth=0.5, label='Señal Rᴿ')
    ax2.axhline(y=META_UMBRAL_SENAL, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Ritual activo vs señal desajuste (F4)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(ritual_arr[::10], senal_arr[::10], s=2, alpha=0.3, c=error_arr[::10], cmap='hot')
    ax.set_xlabel('Ritual activo')
    ax.set_ylabel('Señal desajuste')
    ax.set_title(f'Correlación F3 centra (r={corr:.2f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(f4_ritual['error_abs'], 'b-', linewidth=0.4, label='|error|')
    ax.axhline(y=META_UMBRAL_ERROR, color='red', linestyle='--', alpha=0.5, label='Umbral error')
    ax.set_title('Error absoluto en F4')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v167_logs', exist_ok=True)
    plt.savefig(f'v167_logs/v167_meta_obs_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v167_logs/v167_meta_obs_{timestamp}.png")

    return organismo_control, organismo_ritual, exito


if __name__ == "__main__":
    import time
    start = time.time()
    control, ritual, exito = ejecutar_v167()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V167 completado. Éxito: {exito}")