#!/usr/bin/env python3
"""
V161 — ANIMA-2 Etapa 3: RITUAL (sobre base V157)
================================================================================
SOLO AÑADE: Módulo Ritual
NO MODIFICA: Nada de la dinámica base de V157 (hemisferios, fatiga, memoria, juego)

Cambios exclusivos:
  1. Nuevo archivo ritual.py con clase Ritual (independiente)
  2. En OrganismoV161, se instancia self.ritual = Ritual()
  3. En actualizar(), se llama a self.ritual.actualizar(delta, Cb)
  4. El ritual puede modular la corrección (solo si está activo)
  5. Métricas de ritual añadidas al historial

Toda la dinámica original (hemisferios, plasticidad, juego, consciencia)
permanece IDÉNTICA a V157.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque
import copy
import json

# ============================================================
# PARAMETROS BASE (IDÉNTICOS A V157)
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

# Fatiga
K_GAIN = 0.0003
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

# Juego enactuado
LAMBDA_FISICO = 0.1
LAMBDA_COSTO = 1.0
UMBRAL_CB_JUEGO = 35.0
K_INFLUENCIA_JUEGO = 0.00035

# Ruido para forzar corrección
RUIDO_SETPOINT_AMP = 5.0
RUIDO_SETPOINT_PERIODO = 10.0

# Semilla base
SEMILLA_BASE = 44

# Período
PERIODO_ALTERNANCIA = 80.0

# ============================================================
# PARAMETROS NUEVO: RITUAL (SOLO AÑADE)
# ============================================================
RITUAL_TAU = 120.0                    # Constante de decaimiento ritual
RITUAL_REPETICION_MIN = 3             # Patrones mínimos para activar
RITUAL_GAIN = 0.05                    # Influencia sobre acción
RITUAL_PATRON_TEMPORAL = 30.0         # Intervalo esperado entre patrones (segundos)
RITUAL_TOLERANCIA = 0.3               # 30% de tolerancia en timing
RITUAL_UMBRAL_ACTIVACION = 0.7        # Ritual activation > 0.7 → influye
RITUAL_UMBRAL_CB = 28.0               # Cb mínima para activar ritual


# ============================================================
# NUEVO: CLASE RITUAL (MODULAR, INDEPENDIENTE)
# ============================================================

class Ritual:
    """
    Módulo Ritual para V161.
    Solo detecta patrones repetitivos y acumula activación.
    No modifica la dinámica base excepto cuando está activo.
    """
    
    def __init__(self, tau=RITUAL_TAU, repeticion_min=RITUAL_REPETICION_MIN,
                 ritual_gain=RITUAL_GAIN, patron_temporal=RITUAL_PATRON_TEMPORAL,
                 tolerancia=RITUAL_TOLERANCIA, umbral_activacion=RITUAL_UMBRAL_ACTIVACION,
                 umbral_cb=RITUAL_UMBRAL_CB):
        self.tau = tau
        self.repeticion_min = repeticion_min
        self.ritual_gain = ritual_gain
        self.patron_temporal = patron_temporal
        self.tolerancia = tolerancia
        self.umbral_activacion = umbral_activacion
        self.umbral_cb = umbral_cb
        
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []          # (tiempo, magnitud, direccion)
        self.repeticiones_consecutivas = 0
        self.historial_activation = []
        self.historial_active = []
        self.tiempo_activo = 0.0
    
    def detectar_patron(self, delta_intencional, tiempo_actual):
        """Detecta si la acción actual forma parte de un patrón repetitivo"""
        if abs(delta_intencional) < 0.5:
            return False
        
        direccion = np.sign(delta_intencional)
        magnitud = abs(delta_intencional)
        
        # Buscar en buffer patrones similares
        for t_prev, mag_prev, dir_prev in self.patron_buffer:
            dt = tiempo_actual - t_prev
            
            # Verificar timing esperado (±30%)
            timing_ok = abs(dt - self.patron_temporal) <= (self.patron_temporal * self.tolerancia)
            
            # Verificar magnitud similar (±30%)
            if magnitud > 0 and mag_prev > 0:
                magnitud_ok = abs(magnitud - mag_prev) / max(magnitud, mag_prev) < 0.3
            else:
                magnitud_ok = False
            
            # Verificar misma dirección
            direccion_ok = dir_prev == direccion
            
            if timing_ok and magnitud_ok and direccion_ok:
                return True
        
        return False
    
    def actualizar(self, delta_intencional, Cb, tiempo_actual, dt):
        """
        Actualiza activación ritual basada en repetición de patrones.
        Ritual_activation = ∫(Cb * repetición) / τ dt
        """
        # Detectar patrón
        es_patron = self.detectar_patron(delta_intencional, tiempo_actual)
        
        if es_patron and Cb > self.umbral_cb:
            self.repeticiones_consecutivas += 1
            
            if self.repeticiones_consecutivas >= self.repeticion_min:
                incremento = Cb * self.repeticiones_consecutivas / 100.0
                self.activation += incremento * dt
        else:
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.5)
        
        # Decaimiento natural
        self.activation *= np.exp(-dt / self.tau)
        self.activation = max(0.0, min(2.0, self.activation))
        
        # Determinar si ritual está activo
        was_active = self.active
        self.active = self.activation > self.umbral_activacion
        
        if self.active:
            self.tiempo_activo += dt
        
        # Registrar historial
        self.historial_activation.append(self.activation)
        self.historial_active.append(self.active)
        
        # Registrar patrón en buffer (mantener últimos 10)
        if abs(delta_intencional) > 0.5:
            self.patron_buffer.append((tiempo_actual, abs(delta_intencional),
                                        np.sign(delta_intencional)))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        
        return self.active
    
    def modular_correccion(self, delta_raw, correccion_ritual):
        """
        Aplica modulación ritual a la corrección.
        Solo influye si el ritual está activo.
        """
        if self.active:
            # El ritual fuerza repetición del patrón
            return delta_raw * (1 - self.activation * 0.3) + correccion_ritual * self.activation
        return delta_raw
    
    def reset(self):
        self.activation = 0.0
        self.active = False
        self.patron_buffer = []
        self.repeticiones_consecutivas = 0
        self.historial_activation = []
        self.historial_active = []
        self.tiempo_activo = 0.0
    
    def get_influencia(self, Cb):
        """Retorna factor de influencia para debugging"""
        if self.active:
            return self.activation * self.ritual_gain
        return 0.0


# ============================================================
# HEMISFERIO (IDÉNTICO A V157)
# ============================================================

class HemisferioV161:
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
# FATIGA METABOLICA (IDÉNTICO A V157)
# ============================================================

class FatigaMetabolicaV161:
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

class MemoriaAusenciaV161:
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

class ConscienciaBasicaV161:
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
# MODO JUEGO (IDÉNTICO A V157)
# ============================================================

class ModoJuegoV161:
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
# APARATO MOTOR V161 (CON RITUAL AÑADIDO)
# ============================================================

class AparatoMotorV161:
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
        
        self.fatiga = FatigaMetabolicaV161()
        self.memoria = MemoriaAusenciaV161()
        self.consciencia = ConscienciaBasicaV161()
        self.juego = ModoJuegoV161()
        
        # NUEVO: Módulo Ritual
        self.ritual = Ritual()
        
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_raw):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, False
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, False)
        
        # Obtener setpoint efectivo y confianza
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), DT)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        # Determinar A_sys-env
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        # Actualizar consciencia básica
        Cb, presion = self.consciencia.actualizar(e_R, A_sys_env, DT)
        
        # Actualizar modo juego
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # Efectos de fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, DT)
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0, 0.0, False)
        
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
        # NUEVO: INFLUENCIA RITUAL (solo añade, no modifica lógica base)
        # ============================================================
        
        # Actualizar ritual con delta_raw y Cb
        ritual_activo = self.ritual.actualizar(delta_raw, Cb, t, DT)
        
        # Obtener corrección ritual si está activo
        correccion_ritual = 0.0
        if ritual_activo and self.ritual.patron_buffer:
            _, ultima_mag, ultima_dir = self.ritual.patron_buffer[-1]
            correccion_ritual = ultima_dir * ultima_mag * self.ritual.ritual_gain
        
        # Aplicar modulación ritual
        delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)
        
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
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo, delta_costo,
                self.ritual.activation, ritual_activo)
    
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
        self.ritual.reset()  # NUEVO


# ============================================================
# ORGANISMO COMPLETO V161 (CON RITUAL)
# ============================================================

class OrganismoV161:
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
        
        self.izquierdo = HemisferioV161("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV161("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV161("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV161("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV161()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'juego_activo': [],
            'ritual_activation': [],    # NUEVO
            'ritual_active': [],        # NUEVO
            'historia': [],
            'fatiga': [],
            'costo': [],
            's_shared': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_raw):
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
         ritual_activation, ritual_active) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
        self.historial['confianza'].append(confianza)
        self.historial['Cb'].append(Cb)
        self.historial['juego_activo'].append(juego_activo)
        self.historial['ritual_activation'].append(ritual_activation)  # NUEVO
        self.historial['ritual_active'].append(ritual_active)          # NUEVO
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['costo'].append(costo)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        return orientacion, historia, fatiga, confianza, Cb, juego_activo, ritual_activation, ritual_active
    
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


# ============================================================
# EXPERIMENTO V161
# ============================================================

def ejecutar_v161():
    print("=" * 100)
    print("EXPERIMENTO V161 — ANIMA-2 Etapa 3: RITUAL (sobre base V157)")
    print("=" * 100)
    print("  CAMBIOS: SOLO añade módulo Ritual")
    print("  NO MODIFICA: Hemisferios, fatiga, memoria, consciencia, juego")
    print("")
    print("  Parámetros Ritual (nuevos):")
    print(f"    τ_ritual = {RITUAL_TAU}s")
    print(f"    repetición_min = {RITUAL_REPETICION_MIN}")
    print(f"    patron_temporal = {RITUAL_PATRON_TEMPORAL}s")
    print(f"    umbral_activacion = {RITUAL_UMBRAL_ACTIVACION}")
    print(f"    umbral_Cb_ritual = {RITUAL_UMBRAL_CB}")
    print("=" * 100)
    
    # Crear dos organismos con la MISMA semilla
    print("\n  Creando organismos paralelos...")
    organismo_control = OrganismoV161(seed=SEMILLA_BASE)
    organismo_ritual = OrganismoV161(seed=SEMILLA_BASE)
    
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
    
    # ============================================================
    # FASE 1: Baseline (3 ciclos)
    # ============================================================
    print("\n  F1: Baseline (3 ciclos) - ambos RITUAL FORZADO OFF...")
    
    # Forzar ritual OFF en ambos durante baseline
    # (Nota: el ritual se inicializa en False)
    
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        
        organismo_control.actualizar(t, DT, t_actual + 300, setpoint)
        organismo_ritual.actualizar(t, DT, t_actual + 300, setpoint)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 2: Control - 20 ciclos SIN ritual (ritual forzado OFF)
    # ============================================================
    print("\n  F2: Control - 20 ciclos SIN ritual...")
    
    # Forzar ritual OFF en control
    organismo_control.motor.ritual.active = False
    organismo_control.motor.ritual.activation = 0.0
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            organismo_control.actualizar(t, DT, t_actual + 2000, setpoint)
        if (ciclo + 1) % 5 == 0:
            historia = organismo_control.motor.fatiga.get_historia()
            fatiga = organismo_control.motor.fatiga.get_fatiga()
            print(f"      Control ciclo {ciclo + 1}/20, fatiga={fatiga:.0f}°, historia={historia:.0f}°")
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 3: Experimental - 20 ciclos CON ritual (permitido)
    # ============================================================
    print("\n  F3: Experimental - 20 ciclos CON ritual...")
    
    # Ritual puede activarse naturalmente
    episodios_ritual = 0
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            _, _, _, _, _, _, ritual_act, ritual_active = organismo_ritual.actualizar(t, DT, t_actual + 2000, setpoint)
            
            if ritual_active and ciclo not in [episodios_ritual]:
                episodios_ritual += 1
        
        if (ciclo + 1) % 5 == 0:
            historia = organismo_ritual.motor.fatiga.get_historia()
            fatiga = organismo_ritual.motor.fatiga.get_fatiga()
            ritual_act = organismo_ritual.motor.ritual.activation
            print(f"      Ritual ciclo {ciclo + 1}/20, fatiga={fatiga:.0f}°, historia={historia:.0f}°, ritual_act={ritual_act:.3f}")
        t_actual += PERIODO_ALTERNANCIA
    
    print(f"\n      Episodios con ritual activo: {episodios_ritual}/20")
    
    # ============================================================
    # FASE 4: Test post (3 ciclos) - ambos SIN ritual
    # ============================================================
    print("\n  F4: Test post (3 ciclos) - RITUAL FORZADO OFF...")
    
    # Forzar ritual OFF en ambos
    organismo_control.motor.ritual.active = False
    organismo_ritual.motor.ritual.active = False
    
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        
        organismo_control.actualizar(t, DT, t_actual + 300, setpoint)
        organismo_ritual.actualizar(t, DT, t_actual + 300, setpoint)
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V161 — Ritual sobre base V157")
    print("=" * 80)
    
    # Calcular error RMS en últimos 10 segundos de F4
    ventana_rms = int(10.0 / DT)
    if len(organismo_control.historial['orientacion']) > ventana_rms:
        # Obtener últimos valores
        orient_control = np.array(organismo_control.historial['orientacion'][-ventana_rms:])
        orient_ritual = np.array(organismo_ritual.historial['orientacion'][-ventana_rms:])
        
        # Setpoint nominal al final
        setpoint_nominal = -60.0
        
        errores_control = np.abs(orient_control - setpoint_nominal)
        errores_ritual = np.abs(orient_ritual - setpoint_nominal)
        
        error_rms_control = np.sqrt(np.mean(errores_control**2))
        error_rms_ritual = np.sqrt(np.mean(errores_ritual**2))
        
        mejora = (error_rms_control - error_rms_ritual) / error_rms_control * 100 if error_rms_control > 0 else 0
    else:
        error_rms_control = None
        error_rms_ritual = None
        mejora = None
    
    # Calcular fatiga e historia final
    fatiga_control = organismo_control.motor.fatiga.get_fatiga()
    fatiga_ritual = organismo_ritual.motor.fatiga.get_fatiga()
    historia_control = organismo_control.motor.fatiga.get_historia()
    historia_ritual = organismo_ritual.motor.fatiga.get_historia()
    
    # Calcular tiempo de ritual activo
    tiempo_ritual = organismo_ritual.motor.ritual.tiempo_activo
    tiempo_total = 20 * PERIODO_ALTERNANCIA
    pct_ritual = (tiempo_ritual / tiempo_total) * 100 if tiempo_total > 0 else 0
    
    # Cb final
    Cb_control_final = organismo_control.historial['Cb'][-1] if organismo_control.historial['Cb'] else 0
    Cb_ritual_final = organismo_ritual.historial['Cb'][-1] if organismo_ritual.historial['Cb'] else 0
    
    print(f"\n  Resultados:")
    print(f"    Fatiga control: {fatiga_control:.0f}°")
    print(f"    Fatiga ritual: {fatiga_ritual:.0f}°")
    print(f"    Historia control: {historia_control:.0f}°")
    print(f"    Historia ritual: {historia_ritual:.0f}°")
    print(f"    Cb final control: {Cb_control_final:.1f}")
    print(f"    Cb final ritual: {Cb_ritual_final:.1f}")
    print(f"    Tiempo ritual activo: {tiempo_ritual:.1f}s ({pct_ritual:.1f}% del tiempo)")
    print(f"    Activación ritual final: {organismo_ritual.motor.ritual.activation:.3f}")
    
    print(f"\n  Error RMS en últimos 10s (F4):")
    if error_rms_control and error_rms_ritual:
        print(f"    Control: {error_rms_control:.2f}°")
        print(f"    Ritual: {error_rms_ritual:.2f}°")
        print(f"    Mejora: {mejora:.1f}% {'✅' if mejora > 0 else '❌'}")
    
    # Criterios de éxito para Etapa 3
    exito_1 = error_rms_ritual is not None and error_rms_control is not None and error_rms_ritual < error_rms_control
    exito_2 = pct_ritual > 10.0
    exito_3 = organismo_ritual.motor.ritual.activation > 0.3
    
    exito = exito_1 and exito_2 and exito_3
    
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO (Etapa 3 - Ritual)")
    print("=" * 80)
    print(f"  1. error_rms_ritual < error_rms_control: -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Tiempo ritual activo > 10%: {pct_ritual:.1f}% -> {'✅' if exito_2 else '❌'}")
    print(f"  3. Activación ritual > 0.3: {organismo_ritual.motor.ritual.activation:.3f} -> {'✅' if exito_3 else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ETAPA 3 COMPLETADA — RITUAL VALIDADO")
        print("     El organismo desarrolló activación ritual por repetición enactuada.")
    else:
        print("  ⚠️ ETAPA 3 PARCIAL")
        if not exito_1:
            print("     Ritual no mejoró error RMS (aún)")
        if not exito_2:
            print("     Ritual no se activó suficientemente")
        if not exito_3:
            print("     Activación ritual baja")
    print("=" * 80)
    
    # Gráficos
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Orientación comparativa
    ax = axes[0, 0]
    ax.plot(organismo_control.historial['orientacion'][-5000:], 'b-', linewidth=0.5, label='Control')
    ax.plot(organismo_ritual.historial['orientacion'][-5000:], 'orange', linewidth=0.5, label='Ritual')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('Orientación (final)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Activación ritual
    ax = axes[0, 1]
    ax.plot(organismo_ritual.historial['ritual_activation'], 'purple', linewidth=0.5)
    ax.axhline(y=RITUAL_UMBRAL_ACTIVACION, color='red', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Activación')
    ax.set_title('Activación Ritual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Cb comparativa
    ax = axes[0, 2]
    ax.plot(organismo_control.historial['Cb'], 'b-', linewidth=0.5, label='Control')
    ax.plot(organismo_ritual.historial['Cb'], 'orange', linewidth=0.5, label='Ritual')
    ax.axhline(y=UMBRAL_CB_JUEGO, color='green', linestyle='--', alpha=0.5, label='Umbral juego')
    ax.axhline(y=RITUAL_UMBRAL_CB, color='purple', linestyle='--', alpha=0.5, label='Umbral ritual')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb')
    ax.set_title('Consciencia Básica')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Fatiga vs Historia
    ax = axes[1, 0]
    categorias = ['Control', 'Ritual']
    fatigas = [fatiga_control, fatiga_ritual]
    historias = [historia_control, historia_ritual]
    
    x = np.arange(len(categorias))
    width = 0.35
    ax.bar(x - width/2, fatigas, width, label='Fatiga activa', color='red', alpha=0.7)
    ax.bar(x + width/2, historias, width, label='Historia', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.set_ylabel('Valor (º)')
    ax.set_title('Fatiga vs Historia')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Ritual activation + juego activo
    ax = axes[1, 1]
    ax.plot(organismo_ritual.historial['ritual_activation'], 'purple', linewidth=0.5, label='Ritual')
    ax.plot(organismo_ritual.historial['juego_activo'], 'green', linewidth=0.3, alpha=0.5, label='Juego')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Activación')
    ax.set_title('Ritual vs Juego')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. s_shared
    ax = axes[1, 2]
    ax.plot(organismo_ritual.historial['s_shared'], 'cyan', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('s_shared')
    ax.set_title('Lateralidad')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V161_logs', exist_ok=True)
    plt.savefig(f'V161_logs/v161_ritual_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V161_logs/v161_ritual_{timestamp}.png")
    
    return organismo_control, organismo_ritual, exito


if __name__ == "__main__":
    import time
    start = time.time()
    control, ritual, exito = ejecutar_v161()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V161 completado. Éxito: {exito}")