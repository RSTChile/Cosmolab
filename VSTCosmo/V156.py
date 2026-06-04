#!/usr/bin/env python3
"""
VSTCosmos V156 — ANIMA-2 Etapa 2: Juego enactuado

Basado en V155 (memoria + consciencia básica validada)

Nuevo: Modo JUEGO (O-N7.2 / O-N8.3)
  - Misma dinámica interna, consecuencias físicas amortiguadas
  - Costo energético REAL (se paga siempre)
  - Cb influye en activación del juego

Hipótesis O-N7.2:
  - El juego permite exploración sin riesgo
  - Tras JUEGO, el desempeño en SERIO mejora

Criterios:
  1. Memoria motora se mantiene (error_60s < 25°)
  2. Correlación Cb > 0.7 se mantiene
  3. T_settle_post_juego < T_settle_control (mejora)
  4. Cb_juego < Cb_serio (el juego reduce presión)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS
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

# Juego enactuado (NUEVO)
LAMBDA_FISICO = 0.1      # Movimiento real reducido al 10%
LAMBDA_COSTO = 1.0       # Costo energético REAL (100%)
UMBRAL_CB_JUEGO = 35.0   # Cb a partir del cual se activa juego
K_INFLUENCIA_JUEGO = 0.00035  # Influencia suave de Cb en acción

# Semilla base
SEMILLA_BASE = 44

# Período
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV156:
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
# FATIGA METABOLICA
# ============================================================

class FatigaMetabolicaV156:
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
# MEMORIA DE AUSENCIA
# ============================================================

class MemoriaAusencia:
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
# CONSCIENCIA BÁSICA (INTEGRAL DE PRESIÓN)
# ============================================================

class ConscienciaBasica:
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
# MODO JUEGO (NUEVO V156)
# ============================================================

class ModoJuego:
    """
    Modo "como si" (O-N7.2 / O-N8.3).
    
    - Misma dinámica interna
    - Consecuencias físicas amortiguadas (lambda_fisico)
    - Costo energético REAL (lambda_costo = 1.0)
    - Se activa cuando Cb supera umbral durante silencio
    """
    
    def __init__(self, lambda_fisico=LAMBDA_FISICO, lambda_costo=LAMBDA_COSTO,
                 umbral_cb=UMBRAL_CB_JUEGO, k_influencia=K_INFLUENCIA_JUEGO):
        self.lambda_fisico = lambda_fisico
        self.lambda_costo = lambda_costo
        self.umbral_cb = umbral_cb
        self.k_influencia = k_influencia
        self.activo = False
        self.historial_activo = []
    
    def actualizar(self, Cb, confianza, setpoint_presente):
        """
        Decide si activar el modo juego basado en Cb.
        Durante silencio, si Cb > umbral, activa juego.
        """
        if setpoint_presente is None and Cb > self.umbral_cb:
            self.activo = True
        else:
            self.activo = False
        
        self.historial_activo.append(self.activo)
        return self.activo
    
    def aplicar(self, delta_raw):
        """
        Aplica los coeficientes de juego.
        Retorna (delta_fisico, delta_costo)
        """
        if self.activo:
            delta_fisico = delta_raw * self.lambda_fisico
            delta_costo = abs(delta_raw) * self.lambda_costo
        else:
            delta_fisico = delta_raw
            delta_costo = abs(delta_raw)
        
        return delta_fisico, delta_costo
    
    def get_influencia(self, Cb, confianza):
        """Influencia suave de Cb en la acción (durante juego activo)"""
        if self.activo and Cb > self.umbral_cb:
            return self.k_influencia * (Cb - self.umbral_cb) * (1 - confianza)
        return 0.0
    
    def reset(self):
        self.activo = False
        self.historial_activo = []


# ============================================================
# APARATO MOTOR V156 (con juego enactuado)
# ============================================================

class AparatoMotorV156:
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
        
        self.fatiga = FatigaMetabolicaV156()
        self.memoria = MemoriaAusencia()
        self.consciencia = ConscienciaBasica()
        self.juego = ModoJuego()  # NUEVO
        
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
            return self.orientacion, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        
        if abs(gradiente) < 0.01:
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    0.0, 0.0, 0.0, 0.0, False)
        
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
        
        # Actualizar modo juego (decide si activar basado en Cb)
        juego_activo = self.juego.actualizar(Cb, confianza, setpoint_raw)
        
        # Efectos de fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(0, 0, False, DT)
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                    confianza, zona_muerta_efectiva, Cb, presion, juego_activo)
        
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
        
        # Influencia del juego en la acción (ajuste suave)
        influencia_juego = self.juego.get_influencia(Cb, confianza)
        if influencia_juego != 0:
            delta_raw += influencia_juego * (self.memoria.setpoint_last - self.orientacion) * 0.1
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # APLICAR MODO JUEGO: separa movimiento físico de costo energético
        delta_fisico, delta_costo = self.juego.aplicar(delta)
        
        # Costo total (incluye torque de sujeción y juego)
        costo_total = costo_error + abs(torque_memoria) + delta_costo
        
        # Reposo real
        en_reposo_real = (setpoint_raw is None and abs(delta) < 0.001 and abs(torque_memoria) < 0.001)
        
        # Actualizar fatiga con el costo REAL
        self.fatiga.actualizar(delta_fisico, costo_total, en_reposo_real, DT)
        
        # Temblor
        delta_fisico += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta_fisico
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return (self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(),
                confianza, zona_muerta_efectiva, Cb, presion, juego_activo)
    
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


# ============================================================
# SISTEMA V156
# ============================================================

class SistemaV156:
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
        
        self.izquierdo = HemisferioV156("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV156("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV156("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV156("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV156()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'confianza': [],
            'Cb': [],
            'juego_activo': [],
            'historia': [],
            'fatiga': [],
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
        (orientacion, historia, fatiga, confianza, _, Cb, _, juego_activo) = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
        self.historial['confianza'].append(confianza)
        self.historial['Cb'].append(Cb)
        self.historial['juego_activo'].append(juego_activo)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        return orientacion, historia, fatiga, confianza, Cb, juego_activo
    
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


def analizar_semiciclo(orientaciones, setpoints, dt=DT, umbral_error=2.0, ventana=50):
    if len(orientaciones) == 0:
        return None, None, None, None
    
    fin = min(len(orientaciones), int(40.0 / dt))
    orient_ciclo = orientaciones[:fin]
    setpoint_ciclo = setpoints[:fin]
    
    errores = np.abs(np.array(orient_ciclo) - np.array(setpoint_ciclo))
    
    t_settle = None
    for i in range(len(errores) - ventana):
        if all(errores[i:i+ventana] < umbral_error):
            t_settle = i * dt
            break
    
    if len(errores) > ventana:
        error_final = np.mean(errores[-ventana:])
    else:
        error_final = errores[-1] if len(errores) > 0 else None
    
    amplitud = max(orient_ciclo) if setpoint_ciclo[-1] > 0 else abs(min(orient_ciclo))
    
    return t_settle, error_final, amplitud, None


def ejecutar_v156():
    print("=" * 100)
    print("EXPERIMENTO V156 — ANIMA-2 Etapa 2: Juego enactuado")
    print("=" * 100)
    print("  Modo JUEGO: misma dinámica, consecuencias amortiguadas")
    print("  Costo energético REAL (se paga siempre)")
    print("  Cb influye en activación del juego")
    print("")
    print(f"  Parámetros:")
    print(f"    - Kp_base = {KP_BASE}")
    print(f"    - K_HOLD = {K_HOLD}")
    print(f"    - tau_cb = {TAU_CB}s")
    print(f"    - lambda_fisico = {LAMBDA_FISICO} (movimiento real reducido)")
    print(f"    - lambda_costo = {LAMBDA_COSTO} (costo real)")
    print(f"    - umbral_cb_juego = {UMBRAL_CB_JUEGO}")
    print("=" * 100)
    
    sistema = SistemaV156("V156", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_raw=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de juego enactuado...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # ============================================================
    # FASE 1: Baseline fresco (para medir T_settle control)
    # ============================================================
    print("\n  F1: Baseline fresco (3 ciclos)...")
    
    orientaciones_f1 = []
    setpoints_f1 = []
    
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, _, _, _, _, _ = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
            orientaciones_f1.append(orient)
            setpoints_f1.append(setpoint)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # Analizar F1 (primer semiciclo)
    fin_ciclo = int(40.0 / DT)
    t_settle_f1, error_f1, amp_f1, _ = analizar_semiciclo(
        orientaciones_f1[:fin_ciclo], setpoints_f1[:fin_ciclo]
    )
    
    print(f"      T_settle control: {t_settle_f1:.1f}s" if t_settle_f1 else "      T_settle control: No alcanzado")
    
    # ============================================================
    # FASE 2: Grupo Control SIN JUEGO
    # ============================================================
    print("\n  F2: Grupo Control (20 ciclos SIN juego)...")
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        if (ciclo + 1) % 10 == 0:
            historia = sistema.motor.fatiga.get_historia()
            fatiga = sistema.motor.fatiga.get_fatiga()
            print(f"      Ciclo {ciclo + 1}/20 control, fatiga={fatiga:.0f}°, historia={historia:.0f}°")
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 3: Grupo Experimental CON JUEGO
    # ============================================================
    print("\n  F3: Grupo Experimental (20 ciclos CON juego)...")
    print("      (Modo juego se activa automáticamente cuando Cb > umbral)")
    
    for ciclo in range(20):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, _, _, _, Cb, juego_activo = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        if (ciclo + 1) % 5 == 0:
            historia = sistema.motor.fatiga.get_historia()
            fatiga = sistema.motor.fatiga.get_fatiga()
            print(f"      Ciclo {ciclo + 1}/20 juego, fatiga={fatiga:.0f}°, historia={historia:.0f}°")
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 4: Test post-juego (medir mejora)
    # ============================================================
    print("\n  F4: Test post-juego (3 ciclos)...")
    
    orientaciones_f4 = []
    setpoints_f4 = []
    
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, _, _, _, _, _ = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
            orientaciones_f4.append(orient)
            setpoints_f4.append(setpoint)
    
    # Analizar F4
    t_settle_f4, error_f4, amp_f4, _ = analizar_semiciclo(
        orientaciones_f4[:fin_ciclo], setpoints_f4[:fin_ciclo]
    )
    
    print(f"      T_settle post-juego: {t_settle_f4:.1f}s" if t_settle_f4 else "      T_settle post-juego: No alcanzado")
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V156 — Juego enactuado")
    print("=" * 80)
    
    mejora = None
    if t_settle_f1 and t_settle_f4:
        mejora = (t_settle_f1 - t_settle_f4) / t_settle_f1 * 100
        print(f"\n  Comparación T_settle:")
        print(f"    Control (sin juego): {t_settle_f1:.1f}s")
        print(f"    Post-juego: {t_settle_f4:.1f}s")
        print(f"    Mejora: {mejora:.1f}% {'✅' if mejora > 0 else '❌'}")
    
    # Verificar memoria durante el experimento (últimos registros)
    historias = sistema.historial['historia']
    ultima_historia = historias[-1] if historias else 0
    print(f"\n  Historia acumulada final: {ultima_historia:.0f}°")
    
    exito = (t_settle_f4 is not None and t_settle_f1 is not None and mejora > 0)
    
    print("\n" + "=" * 80)
    print("CONCLUSION V156")
    print("=" * 80)
    
    if exito:
        print("\n  ✅ ETAPA 2 COMPLETADA")
        print("     Modo juego enactuado funcional")
        print(f"     Mejora T_settle: {mejora:.1f}%")
        print("\n  ANIMA-2 listo para Etapa 3 (Ritual - V157)")
    else:
        print("\n  ⚠️ ETAPA 2 PARCIAL")
        if not t_settle_f1:
            print("     Control sin T_settle medible")
        if not t_settle_f4:
            print("     Post-juego sin T_settle medible")
        if mejora is not None and mejora <= 0:
            print(f"     No hubo mejora: {mejora:.1f}%")
    
    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Orientación F1 vs F4
    ax = axes[0, 0]
    ax.plot(orientaciones_f1[:fin_ciclo], 'b-', linewidth=0.6, label='F1 (control)')
    ax.plot(orientaciones_f4[:fin_ciclo], 'orange', linewidth=0.6, label='F4 (post-juego)')
    ax.plot(setpoints_f1[:fin_ciclo], 'r--', linewidth=0.6, alpha=0.5, label='Setpoint')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title(f'Comparativa: Control vs Post-juego (mejora {mejora:.1f}%)' if mejora else 'Comparativa: Control vs Post-juego')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Evolución de Cb
    ax = axes[0, 1]
    Cb_hist = sistema.historial['Cb']
    ax.plot(Cb_hist, 'purple', linewidth=0.8)
    ax.axhline(y=UMBRAL_CB_JUEGO, color='red', linestyle='--', alpha=0.5, label=f'Umbral juego ({UMBRAL_CB_JUEGO})')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Cb')
    ax.set_title('Consciencia básica durante el experimento')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Activación del juego
    ax = axes[1, 0]
    juego_activo = sistema.historial['juego_activo']
    ax.plot([1 if j else 0 for j in juego_activo], 'green', linewidth=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Juego activo')
    ax.set_title('Activación del modo juego')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: s_shared
    ax = axes[1, 1]
    s_shared = sistema.historial['s_shared']
    ax.plot(s_shared, 'purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('s_shared')
    ax.set_title('Lateralidad')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('V156_logs', exist_ok=True)
    plt.savefig(f'V156_logs/v156_juego_enactuado_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: V156_logs/v156_juego_enactuado_{timestamp}.png")
    
    return sistema, exito


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v156()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V156 completado. Éxito: {exito}")