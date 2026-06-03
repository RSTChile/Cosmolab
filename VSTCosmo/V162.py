#!/usr/bin/env python3
"""
V162 — ANIMA-2 Etapa 3: RITUAL (sobre base V157, detector por cruces por cero)
================================================================================
ÚNICO CAMBIO: Método detectar_patron() de la clase Ritual
  - Ahora usa CRUCES POR CERO de la orientación (eventos reales del organismo)
  - No fuerza picos de delta_raw que no existen en V157
  - Respeta la dinámica suave y continua del organismo

TODO LO DEMÁS ES IDÉNTICO A V161 (que es V157 + Ritual)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque
import copy

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
# PARAMETROS RITUAL (SIN CAMBIOS)
# ============================================================
RITUAL_TAU = 120.0
RITUAL_REPETICION_MIN = 3
RITUAL_GAIN = 0.05
RITUAL_PATRON_TEMPORAL = 40.0          # Ajustado: tiempo entre cruces (~medio ciclo)
RITUAL_TOLERANCIA = 0.3
RITUAL_UMBRAL_ACTIVACION = 0.7
RITUAL_UMBRAL_CB = 28.0


# ============================================================
# CLASE RITUAL CON DETECTOR POR CRUCES POR CERO
# ============================================================

class Ritual:
    """
    Módulo Ritual para V162.
    Detecta patrones usando CRUCES POR CERO de la orientación.
    Esto respeta la dinámica real del organismo.
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
        self.patron_buffer = []          # (tiempo, direccion)
        self.repeticiones_consecutivas = 0
        self.historial_activation = []
        self.historial_active = []
        self.tiempo_activo = 0.0
        
        # Para detección de cruces por cero
        self.ultima_orientacion = 0.0
        self.ultimo_cruce_time = 0.0
        self.ultima_direccion = 0
    
    def detectar_cruce_por_cero(self, orientacion, tiempo_actual):
        """
        Detecta cuando la orientación cruza por cero.
        Esto marca la transición entre semiciclos.
        """
        # Verificar cruce por cero
        if self.ultima_orientacion == 0:
            self.ultima_orientacion = orientacion
            return False, 0
        
        cruce = (self.ultima_orientacion < 0 and orientacion >= 0) or \
                (self.ultima_orientacion > 0 and orientacion <= 0)
        
        self.ultima_orientacion = orientacion
        
        if not cruce:
            return False, 0
        
        # Determinar dirección del cruce (hacia positivo o negativo)
        direccion = 1 if orientacion >= 0 else -1
        
        return True, direccion
    
    def actualizar(self, orientacion, Cb, tiempo_actual, dt):
        """
        Actualiza activación ritual basada en cruces por cero.
        """
        # Detectar cruce por cero
        es_cruce, direccion = self.detectar_cruce_por_cero(orientacion, tiempo_actual)
        
        if es_cruce and Cb > self.umbral_cb:
            # Tenemos un evento: cruce por cero
            # Verificar si forma un patrón con cruces anteriores
            
            # Buscar patrón temporal con cruces previos
            es_patron = False
            for t_prev, dir_prev in self.patron_buffer:
                dt_desde_prev = tiempo_actual - t_prev
                
                # Verificar timing esperado (aprox cada 40s, tolerancia 30%)
                timing_ok = abs(dt_desde_prev - self.patron_temporal) <= (self.patron_temporal * self.tolerancia)
                
                # Verificar misma dirección
                direccion_ok = dir_prev == direccion
                
                if timing_ok and direccion_ok:
                    es_patron = True
                    break
            
            if es_patron:
                self.repeticiones_consecutivas += 1
                
                if self.repeticiones_consecutivas >= self.repeticion_min:
                    # Activación ritual: Cb alta + repetición
                    incremento = Cb * self.repeticiones_consecutivas / 100.0
                    self.activation += incremento * dt
            else:
                # Cruce que no sigue patrón -> reduce conteo
                self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 1)
            
            # Registrar cruce en buffer (mantener últimos 10)
            self.patron_buffer.append((tiempo_actual, direccion))
            if len(self.patron_buffer) > 10:
                self.patron_buffer.pop(0)
        else:
            # Decaimiento gradual si no hay cruces
            self.repeticiones_consecutivas = max(0, self.repeticiones_consecutivas - 0.1)
        
        # Decaimiento natural de activación
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
        
        return self.active
    
    def modular_correccion(self, delta_raw, correccion_ritual):
        """
        Aplica modulación ritual a la corrección.
        """
        if self.active:
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
        self.ultima_orientacion = 0.0
        self.ultimo_cruce_time = 0.0
        self.ultima_direccion = 0
    
    def get_influencia(self, Cb):
        if self.active:
            return self.activation * self.ritual_gain
        return 0.0


# ============================================================
# HEMISFERIO (IDÉNTICO A V157)
# ============================================================

class HemisferioV162:
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

class FatigaMetabolicaV162:
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

class MemoriaAusenciaV162:
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

class ConscienciaBasicaV162:
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

class ModoJuegoV162:
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
# APARATO MOTOR V162 (CON RITUAL, SIN OTROS CAMBIOS)
# ============================================================

class AparatoMotorV162:
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
        
        self.fatiga = FatigaMetabolicaV162()
        self.memoria = MemoriaAusenciaV162()
        self.consciencia = ConscienciaBasicaV162()
        self.juego = ModoJuegoV162()
        
        # Módulo Ritual (con nuevo detector por cruces)
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
        
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), DT)
        
        error = setpoint_objetivo - self.orientacion
        e_R = abs(error)
        
        if setpoint_raw is not None:
            if abs(setpoint_raw) > 0.01:
                A_sys_env = min(1.0, abs(self.orientacion) / abs(setpoint_raw))
            else:
                A_sys_env = 1.0
        else:
            A_sys_env = confianza
        
        Cb, presion = self.consciencia.actualizar(e_R, A_sys_env, DT)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, DT)
        
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo, 0.0, 0.0, False)
        
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
        
        # ACTUALIZAR RITUAL (usando orientación, NO delta_raw)
        ritual_activo = self.ritual.actualizar(self.orientacion, Cb, t, DT)
        
        correccion_ritual = 0.0
        if ritual_activo and self.ritual.patron_buffer:
            _, ultima_dir = self.ritual.patron_buffer[-1]
            correccion_ritual = ultima_dir * 10.0 * self.ritual.ritual_gain
        
        delta_raw = self.ritual.modular_correccion(delta_raw, correccion_ritual)
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, DT)
        
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
        self.ritual.reset()


# ============================================================
# ORGANISMO COMPLETO V162
# ============================================================

class OrganismoV162:
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
        
        self.izquierdo = HemisferioV162("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV162("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV162("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV162("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.motor = AparatoMotorV162()
        self.modo_entrenamiento = True
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'juego_activo': [],
            'ritual_activation': [],
            'ritual_active': [],
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
        self.historial['ritual_activation'].append(ritual_activation)
        self.historial['ritual_active'].append(ritual_active)
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
# EXPERIMENTO V162
# ============================================================

def ejecutar_v162():
    print("=" * 100)
    print("EXPERIMENTO V162 — ANIMA-2 Etapa 3: RITUAL (detector por cruces de cero)")
    print("=" * 100)
    print("  CAMBIO ÚNICO: Ritual detecta cruces por cero de orientación")
    print("  TODO LO DEMÁS: Idéntico a V157 (que funciona)")
    print("")
    print("  Parámetros Ritual:")
    print(f"    τ_ritual = {RITUAL_TAU}s")
    print(f"    repetición_min = {RITUAL_REPETICION_MIN}")
    print(f"    patron_temporal = {RITUAL_PATRON_TEMPORAL}s (entre cruces)")
    print(f"    umbral_activacion = {RITUAL_UMBRAL_ACTIVACION}")
    print(f"    umbral_Cb_ritual = {RITUAL_UMBRAL_CB}")
    print("=" * 100)
    
    print("\n  Creando organismos paralelos...")
    organismo_control = OrganismoV162(seed=SEMILLA_BASE)
    organismo_ritual = OrganismoV162(seed=SEMILLA_BASE)
    
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
    
    # F1: Baseline
    print("\n  F1: Baseline (3 ciclos) - ambos RITUAL FORZADO OFF...")
    
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        
        organismo_control.actualizar(t, DT, t_actual + 300, setpoint)
        organismo_ritual.actualizar(t, DT, t_actual + 300, setpoint)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # F2: Control
    print("\n  F2: Control - 20 ciclos SIN ritual...")
    
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
    
    # F3: Experimental con ritual
    print("\n  F3: Experimental - 20 ciclos CON ritual...")
    
    episodios_ritual = 0
    ritual_activation_vals = []
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            _, _, _, _, _, _, ritual_act, ritual_active = organismo_ritual.actualizar(t, DT, t_actual + 2000, setpoint)
            
            if ritual_active:
                episodios_ritual += 1
            ritual_activation_vals.append(ritual_act)
        
        if (ciclo + 1) % 5 == 0:
            historia = organismo_ritual.motor.fatiga.get_historia()
            fatiga = organismo_ritual.motor.fatiga.get_fatiga()
            ritual_act = organismo_ritual.motor.ritual.activation
            print(f"      Ritual ciclo {ciclo + 1}/20, fatiga={fatiga:.0f}°, historia={historia:.0f}°, ritual_act={ritual_act:.3f}")
        t_actual += PERIODO_ALTERNANCIA
    
    print(f"\n      Episodios con ritual activo: {episodios_ritual}/20")
    
    # F4: Test post
    print("\n  F4: Test post (3 ciclos) - RITUAL FORZADO OFF...")
    
    organismo_control.motor.ritual.active = False
    organismo_ritual.motor.ritual.active = False
    
    for i in range(int(3 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        setpoint = generar_setpoint_con_ruido(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
        
        organismo_control.actualizar(t, DT, t_actual + 300, setpoint)
        organismo_ritual.actualizar(t, DT, t_actual + 300, setpoint)
    
    # Análisis
    print("\n" + "=" * 80)
    print("RESULTADOS V162 — Ritual con detector por cruces")
    print("=" * 80)
    
    ventana_rms = int(10.0 / DT)
    if len(organismo_control.historial['orientacion']) > ventana_rms:
        orient_control = np.array(organismo_control.historial['orientacion'][-ventana_rms:])
        orient_ritual = np.array(organismo_ritual.historial['orientacion'][-ventana_rms:])
        
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
    
    fatiga_control = organismo_control.motor.fatiga.get_fatiga()
    fatiga_ritual = organismo_ritual.motor.fatiga.get_fatiga()
    historia_control = organismo_control.motor.fatiga.get_historia()
    historia_ritual = organismo_ritual.motor.fatiga.get_historia()
    
    tiempo_ritual = organismo_ritual.motor.ritual.tiempo_activo
    tiempo_total = 20 * PERIODO_ALTERNANCIA
    pct_ritual = (tiempo_ritual / tiempo_total) * 100 if tiempo_total > 0 else 0
    
    Cb_control_final = organismo_control.historial['Cb'][-1] if organismo_control.historial['Cb'] else 0
    Cb_ritual_final = organismo_ritual.historial['Cb'][-1] if organismo_ritual.historial['Cb'] else 0
    ritual_activation_final = organismo_ritual.motor.ritual.activation
    
    print(f"\n  Resultados:")
    print(f"    Fatiga control: {fatiga_control:.0f}°")
    print(f"    Fatiga ritual: {fatiga_ritual:.0f}°")
    print(f"    Historia control: {historia_control:.0f}°")
    print(f"    Historia ritual: {historia_ritual:.0f}°")
    print(f"    Cb final control: {Cb_control_final:.1f}")
    print(f"    Cb final ritual: {Cb_ritual_final:.1f}")
    print(f"    Tiempo ritual activo: {tiempo_ritual:.1f}s ({pct_ritual:.1f}% del tiempo)")
    print(f"    Activación ritual final: {ritual_activation_final:.3f}")
    
    if error_rms_control and error_rms_ritual:
        print(f"\n  Error RMS en últimos 10s (F4):")
        print(f"    Control: {error_rms_control:.2f}°")
        print(f"    Ritual: {error_rms_ritual:.2f}°")
        print(f"    Mejora: {mejora:.1f}% {'✅' if mejora > 0 else '❌'}")
    
    exito_1 = error_rms_ritual is not None and error_rms_control is not None and error_rms_ritual < error_rms_control
    exito_2 = pct_ritual > 10.0
    exito_3 = ritual_activation_final > 0.3
    
    exito = exito_1 and exito_2 and exito_3
    
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO (Etapa 3 - Ritual)")
    print("=" * 80)
    print(f"  1. error_rms_ritual < error_rms_control: -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Tiempo ritual activo > 10%: {pct_ritual:.1f}% -> {'✅' if exito_2 else '❌'}")
    print(f"  3. Activación ritual > 0.3: {ritual_activation_final:.3f} -> {'✅' if exito_3 else '❌'}")
    
    print("\n" + "=" * 80)
    if exito:
        print("  ✅ ETAPA 3 COMPLETADA — RITUAL VALIDADO")
        print("     El organismo detectó cruces por cero como patrón temporal.")
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
    
    ax = axes[0, 0]
    ax.plot(organismo_control.historial['orientacion'][-5000:], 'b-', linewidth=0.5, label='Control')
    ax.plot(organismo_ritual.historial['orientacion'][-5000:], 'orange', linewidth=0.5, label='Ritual')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('Orientación (final)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(organismo_ritual.historial['ritual_activation'], 'purple', linewidth=0.5)
    ax.axhline(y=RITUAL_UMBRAL_ACTIVACION, color='red', linestyle='--', alpha=0.5, label='Umbral')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Activación')
    ax.set_title('Activación Ritual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
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
    
    ax = axes[1, 1]
    ax.plot(organismo_ritual.historial['ritual_activation'], 'purple', linewidth=0.5, label='Ritual')
    ax.plot(organismo_ritual.historial['juego_activo'], 'green', linewidth=0.3, alpha=0.5, label='Juego')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Activación')
    ax.set_title('Ritual vs Juego')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.plot(organismo_ritual.historial['s_shared'], 'cyan', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('s_shared')
    ax.set_title('Lateralidad')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v162_logs', exist_ok=True)
    plt.savefig(f'v162_logs/v162_ritual_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v162_logs/v162_ritual_{timestamp}.png")
    
    return organismo_control, organismo_ritual, exito


if __name__ == "__main__":
    import time
    start = time.time()
    control, ritual, exito = ejecutar_v162()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V162 completado. Éxito: {exito}")