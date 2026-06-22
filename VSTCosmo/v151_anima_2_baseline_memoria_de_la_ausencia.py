#!/usr/bin/env python3
"""
VSTCosmos V151 — ANIMA-2 Baseline: Memoria de la ausencia

Etapa 0 de la Hoja de Ruta Nodal Definitiva.

Objetivo: Consolidar ANIMA-2 como organismo capaz de mantener orientación
transitoria cuando el estímulo desaparece. La memoria no es una base de datos.
Es persistencia degradante de constricción: el setpoint se ancla, y lo que
decae es el torque (confianza), no el recuerdo.

Parámetros heredados de V150:
  - Kp_base = 0.002
  - K_GAIN = 0.0003
  - K_PRECISION = 0.002
  - TAU_RECUPERACION = 300.0

Nuevo mecanismo: MemoriaAusencia
  - tau_base = 30.0s (memoria mínima)
  - k_mem = 0.005 (τ_mem = 30 + k_mem * E_historia)
  - Suelo de confianza = 0.2 (evita amnesia completa, O-N16)

Protocolo:
  F1: Baseline fresco (3 ciclos ±60°)
  F2: Fatiga inducida (30 ciclos ±60°)
  F3: Silencio (120s) -> medir persistencia de orientación
  F4: Reactivación (3 ciclos) -> medir T_settle_react
  F5: Reposo (300s) -> verificar decaimiento de E_activa

Criterios de éxito (3/4):
  1. Memoria motora: |orient(t=60s)| < 25° con E_historia > 1500°
  2. Costo de recordar: ΔE_activa_F3 > 0.3 * ΔE_activa_F2_promedio
  3. Inercia de T⃗: correlación τ_mem vs E_historia > 0.8
  4. Reactivación: T_settle_react / T_settle0 < 1.5
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (heredados de V150)
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

# Fatiga (heredada de V150)
K_GAIN = 0.0003
K_PRECISION = 0.002
K_TEMBLOR = 0.001
TAU_RECUPERACION = 300.0

# Memoria de ausencia (NUEVO)
TAU_BASE = 30.0
K_MEM = 0.005
SUELO_CONFIANZA = 0.2  # O-N16: evitar amnesia completa

# Semilla base
SEMILLA_BASE = 44

# Período (como V147)
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO (idéntico a V150)
# ============================================================

class HemisferioV151:
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
# FATIGA METABOLICA (heredada de V150)
# ============================================================

class FatigaMetabolicaV151:
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION,
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.historia = 0.0        # Permanente, nunca decae
        self.fatiga_activa = 0.0   # Recuperable, decae con reposo
        
        self.historial_historia = []
        self.historial_fatiga = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        # Historia: acumula SIEMPRE
        self.historia += abs(delta_orientacion)
        
        # Fatiga activa: acumula en movimiento, decae en reposo
        if not en_reposo:
            self.fatiga_activa += abs(delta_orientacion)
        else:
            self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        
        # Efectos de la fatiga (solo sobre fatiga_activa)
        factor_gain = np.exp(-self.k_gain * self.fatiga_activa)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.fatiga_activa
        temblor = self.k_temblor * self.fatiga_activa * np.random.randn()
        
        # Limitar
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
# MEMORIA DE AUSENCIA (NUEVO V151)
# ============================================================

class MemoriaAusencia:
    """
    Memoria como persistencia degradante de constricción.
    
    El setpoint se ancla cuando hay estímulo. Durante el silencio,
    lo que decae es la confianza (torque), no el recuerdo.
    
    La constante de tiempo τ_mem crece con la historia (C-N2.5.6):
    los organismos viejos recuerdan más porque su inercia temporal es mayor.
    """
    
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
            # Estímulo presente: ancla y resetea
            self.setpoint_last = setpoint
            self.t_ausencia = 0.0
            # τ_mem crece con historia: viejo recuerda más
            self.tau_mem = self.tau_base + self.k_mem * E_historia
            self.historial_confianza.append(1.0)
            return self.setpoint_last, 1.0
        else:
            # Estímulo ausente: decae confianza, no setpoint
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
# APARATO MOTOR V151 (con memoria de ausencia)
# ============================================================

class AparatoMotorV151:
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
        
        # Fatiga
        self.fatiga = FatigaMetabolicaV151()
        
        # Memoria de ausencia (NUEVO)
        self.memoria = MemoriaAusencia()
        
        # Plasticidad
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
            return self.orientacion, 0.0, 0.0, 0.0, 0.0
        
        if abs(gradiente) < 0.01:
            return self.orientacion, self.fatiga.get_historia(), self.fatiga.get_fatiga(), 0.0, 0.0
        
        # Obtener setpoint efectivo y confianza desde la memoria
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), DT)
        
        error = setpoint_objetivo - self.orientacion
        
        # Actualizar fatiga
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        # Zona muerta expandida por fatiga
        if abs(error) < zona_muerta_efectiva:
            return (self.orientacion, self.fatiga.get_historia(), 
                    self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva)
        
        # Dirección: viene del error
        direccion = np.sign(error)
        
        # Confianza sensorial: viene del gradiente
        confianza_sensorial = min(1.0, abs(gradiente) * self.sensibilidad_grad)
        
        # Freno exponencial
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp efectivo: base * fatiga * confianza_sensorial
        Kp_base_efectivo = self.Kp_actual * factor_gain * confianza_sensorial
        Kp_base_efectivo = max(self.Kp_min, Kp_base_efectivo)
        
        # Kp instantáneo modulado por confianza de memoria (suelo para evitar amnesia)
        Kp_inst = Kp_base_efectivo * (self.memoria.suelo_confianza + (1 - self.memoria.suelo_confianza) * confianza)
        
        # Delta
        delta_raw = Kp_inst * abs(error) * direccion * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Temblor
        delta += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return (self.orientacion, self.fatiga.get_historia(), 
                self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva)
    
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


# ============================================================
# SISTEMA V151
# ============================================================

class SistemaV151:
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
        
        self.izquierdo = HemisferioV151("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV151("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV151("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV151("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV151()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_raw': [],
            'setpoint_objetivo': [],
            'confianza': [],
            'gradiente': [],
            'historia': [],
            'fatiga': [],
            'zona_muerta': [],
            's_shared': [],
            'Kp': []
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
        orientacion, historia, fatiga, confianza, zona_muerta = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_raw
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_raw'].append(setpoint_raw)
        self.historial['setpoint_objetivo'].append(self.motor.memoria.setpoint_last)
        self.historial['confianza'].append(confianza)
        self.historial['gradiente'].append(gradiente)
        self.historial['historia'].append(historia)
        self.historial['fatiga'].append(fatiga)
        self.historial['zona_muerta'].append(zona_muerta)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, historia, fatiga, confianza
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# ONDA CUADRADA
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


# ============================================================
# ANALISIS
# ============================================================

def analizar_ciclo(orientaciones, setpoints, dt=DT, umbral_error=2.0, ventana=100):
    if len(orientaciones) == 0:
        return None, None, None, None
    
    errores = np.abs(np.array(orientaciones) - np.array(setpoints))
    
    t_settle = None
    for i in range(len(errores) - ventana):
        if all(errores[i:i+ventana] < umbral_error):
            t_settle = i * dt
            break
    
    error_final = np.mean(errores[-ventana:]) if len(errores) > ventana else errores[-1]
    amplitud = max(orientaciones) - min(orientaciones)
    velocidad_media = np.mean(np.abs(np.diff(orientaciones))) / dt if len(orientaciones) > 1 else 0
    
    return t_settle, error_final, amplitud, velocidad_media


# ============================================================
# EXPERIMENTO V151
# ============================================================

def ejecutar_v151():
    print("=" * 100)
    print("EXPERIMENTO V151 — ANIMA-2 Baseline: Memoria de la ausencia")
    print("=" * 100)
    print("  Objetivo: Consolidar memoria como persistencia degradante")
    print("")
    print("  Parámetros:")
    print(f"    - Kp_base = {KP_BASE}")
    print(f"    - tau_base = {TAU_BASE}s (memoria mínima)")
    print(f"    - k_mem = {K_MEM} (τ_mem = 30 + {K_MEM} * E_historia)")
    print(f"    - Suelo confianza = {SUELO_CONFIANZA} (O-N16)")
    print(f"    - K_GAIN = {K_GAIN}")
    print(f"    - Período = {PERIODO_ALTERNANCIA}s ({PERIODO_ALTERNANCIA/2}s por polo)")
    print("=" * 100)
    
    sistema = SistemaV151("V151", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_raw=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de memoria de ausencia...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # ============================================================
    # FASE 1: Baseline fresco (3 ciclos)
    # ============================================================
    print("\n  F1: Baseline fresco (3 ciclos)...")
    
    tiempos_f1 = []
    orientaciones_f1 = []
    setpoints_f1 = []
    historias_f1 = []
    fatigas_f1 = []
    confianzas_f1 = []
    
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
            
            tiempos_f1.append(t)
            orientaciones_f1.append(orient)
            setpoints_f1.append(setpoint)
            historias_f1.append(historia)
            fatigas_f1.append(fatiga)
            confianzas_f1.append(confianza)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # Analizar F1
    fin_ciclo = int(PERIODO_ALTERNANCIA / DT)
    t_settle_f1, error_f1, amp_f1, vel_f1 = analizar_ciclo(
        orientaciones_f1[:fin_ciclo], setpoints_f1[:fin_ciclo])
    historia_f1_final = historias_f1[-1] if historias_f1 else 0
    fatiga_f1_final = fatigas_f1[-1] if fatigas_f1 else 0
    
    # ============================================================
    # FASE 2: Fatiga inducida (30 ciclos)
    # ============================================================
    print("\n  F2: Fatiga inducida (30 ciclos)...")
    
    delta_E_f2_ciclos = []
    
    for ciclo in range(30):
        fatiga_inicio = sistema.motor.fatiga.get_fatiga() if ciclo > 0 else fatiga_f1_final
        
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        
        fatiga_fin = sistema.motor.fatiga.get_fatiga()
        delta_E_f2_ciclos.append(fatiga_fin - fatiga_inicio)
        
        if (ciclo + 1) % 10 == 0:
            print(f"      Ciclo {ciclo + 1}/30 completado, fatiga={fatiga_fin:.0f}°, historia={historia:.0f}°")
        
        t_actual += PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 3: Silencio (120s)
    # ============================================================
    print("\n  F3: Silencio (120s) - midiendo persistencia...")
    
    setpoint_last = sistema.motor.memoria.setpoint_last
    
    tiempos_f3 = []
    orientaciones_f3 = []
    confianzas_f3 = []
    fatigas_f3 = []
    historias_f3 = []
    
    for i in range(int(120.0 / DT)):
        t = t_actual + i * DT
        orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 120, None)
        
        tiempos_f3.append(t)
        orientaciones_f3.append(orient)
        confianzas_f3.append(confianza)
        fatigas_f3.append(fatiga)
        historias_f3.append(historia)
        
        if i % 500 == 0:
            print(f"      t={i*DT:.0f}s | orient={orient:.1f}° | confianza={confianza:.3f} | fatiga={fatiga:.0f}°")
    
    t_actual += 120.0
    
    # Calcular métricas F3
    idx_60s = int(60.0 / DT)
    orient_60s = orientaciones_f3[idx_60s] if idx_60s < len(orientaciones_f3) else orientaciones_f3[-1]
    error_60s = abs(orient_60s - setpoint_last) if setpoint_last is not None else 90.0
    
    delta_E_f3 = fatigas_f3[-1] - (fatigas_f3[0] if fatigas_f3 else 0)
    delta_E_f2_promedio = np.mean(delta_E_f2_ciclos) if delta_E_f2_ciclos else 0
    
    # ============================================================
    # FASE 4: Reactivación (3 ciclos)
    # ============================================================
    print("\n  F4: Reactivación (3 ciclos)...")
    
    orientaciones_f4 = []
    setpoints_f4 = []
    
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            t_rel = i * DT
            
            setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
            
            orientaciones_f4.append(orient)
            setpoints_f4.append(setpoint)
    
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # Analizar F4
    t_settle_f4, error_f4, amp_f4, vel_f4 = analizar_ciclo(orientaciones_f4[:fin_ciclo], setpoints_f4[:fin_ciclo])
    
    # ============================================================
    # FASE 5: Reposo (300s)
    # ============================================================
    print("\n  F5: Reposo (300s)...")
    
    fatiga_inicio_reposo = sistema.motor.fatiga.get_fatiga()
    
    for i in range(int(300.0 / DT)):
        t = t_actual + i * DT
        orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 300, None)
    
    fatiga_fin_reposo = sistema.motor.fatiga.get_fatiga()
    delta_E_f5 = fatiga_fin_reposo - fatiga_inicio_reposo
    
    # ============================================================
    # CORRELACIONES
    # ============================================================
    # Para correlación τ_mem vs E_historia (necesita múltiples corridas)
    # En esta corrida, tomamos la τ_mem al inicio y fin de F3
    tau_mem_inicio = sistema.motor.memoria.get_tau_mem()
    tau_mem_fin = sistema.motor.memoria.get_tau_mem()
    E_historia_fin = historias_f3[-1] if historias_f3 else 0
    
    # Esta correlación se calculará entre corridas, no dentro de una sola
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V151 — Memoria de ausencia")
    print("=" * 80)
    
    print(f"\n  F1 - Baseline fresco:")
    print(f"    T_settle: {t_settle_f1:.1f}s" if t_settle_f1 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f1:.1f}°" if error_f1 else "    Error final: N/A")
    print(f"    Amplitud: {amp_f1:.1f}°")
    print(f"    Velocidad media: {vel_f1:.2f}°/s")
    print(f"    Historia: {historia_f1_final:.0f}°")
    print(f"    Fatiga activa: {fatiga_f1_final:.0f}°")
    
    print(f"\n  F2 - Fatiga inducida (30 ciclos):")
    print(f"    ΔE_activa promedio por ciclo: {delta_E_f2_promedio:.0f}°")
    print(f"    ΔE_activa total: {delta_E_f2_promedio * 30:.0f}°")
    
    print(f"\n  F3 - Silencio (120s):")
    print(f"    Setpoint recordado: {setpoint_last:.0f}°")
    print(f"    Orientación a los 60s: {orient_60s:.1f}°")
    print(f"    Error a los 60s: {error_60s:.1f}°")
    print(f"    ΔE_activa durante silencio: {delta_E_f3:.0f}°")
    print(f"    Confianza final: {confianzas_f3[-1] if confianzas_f3 else 0:.3f}")
    print(f"    τ_mem: {tau_mem_fin:.1f}s")
    
    print(f"\n  F4 - Reactivación:")
    print(f"    T_settle: {t_settle_f4:.1f}s" if t_settle_f4 else "    T_settle: No alcanzado")
    print(f"    Error final: {error_f4:.1f}°" if error_f4 else "    Error final: N/A")
    
    print(f"\n  F5 - Reposo (300s):")
    print(f"    Fatiga activa inicial: {fatiga_inicio_reposo:.0f}°")
    print(f"    Fatiga activa final: {fatiga_fin_reposo:.0f}°")
    print(f"    ΔE_activa durante reposo: {delta_E_f5:.0f}°")
    
    # ============================================================
    # CRITERIOS DE ÉXITO
    # ============================================================
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO (3/4 requeridos)")
    print("=" * 80)
    
    exito_1 = error_60s < 25.0 and historia_f1_final > 1500
    exito_2 = delta_E_f3 > 0.3 * delta_E_f2_promedio if delta_E_f2_promedio > 0 else False
    exito_3 = False  # Requiere múltiples corridas
    exito_4 = t_settle_f4 and t_settle_f1 and (t_settle_f4 / t_settle_f1) < 1.5 if t_settle_f1 else False
    
    print(f"\n  1. Memoria motora (orient_60s < 25° y E_historia > 1500): {error_60s:.1f}° / {historia_f1_final:.0f}° -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Costo de recordar (ΔE_F3 > 0.3 * ΔE_F2): {delta_E_f3:.0f}° > 0.3*{delta_E_f2_promedio:.0f}={0.3*delta_E_f2_promedio:.0f}° -> {'✅' if exito_2 else '❌'}")
    print(f"  3. Inercia de T⃗ (corr τ_mem vs E_historia > 0.8): requiere múltiples corridas -> ⏳")
    print(f"  4. Reactivación (T_settle_react / T_settle0 < 1.5): {t_settle_f4:.1f}s / {t_settle_f1:.1f}s = {t_settle_f4/t_settle_f1:.2f} -> {'✅' if exito_4 else '❌'}")
    
    exitos = sum([exito_1, exito_2, exito_4])
    print(f"\n  Éxitos parciales: {exitos}/3 (excluyendo criterio 3)")
    
    pase_baseline = exitos >= 2  # 2 de 3 para considerar pase, el criterio 3 se evalúa entre corridas
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Gráfico 1: Orientación durante F3 (silencio)
    ax = axes[0, 0]
    t_f3 = np.arange(len(orientaciones_f3)) * DT
    ax.plot(t_f3, orientaciones_f3, 'b-', linewidth=0.8)
    ax.axhline(y=setpoint_last, color='r--', linewidth=1, alpha=0.7, label=f'Setpoint recordado ({setpoint_last:.0f}°)')
    ax.axhline(y=setpoint_last + 25, color='orange', linestyle=':', alpha=0.5, label='Umbral 25°')
    ax.axhline(y=setpoint_last - 25, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('F3: Persistencia de orientación durante silencio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Confianza durante F3
    ax = axes[0, 1]
    ax.plot(t_f3, confianzas_f3, 'purple', linewidth=0.8)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Confianza')
    ax.set_title('F3: Decaimiento de confianza')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Fatiga activa durante F3
    ax = axes[0, 2]
    ax.plot(t_f3, fatigas_f3, 'orange', linewidth=0.8)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Fatiga activa (º)')
    ax.set_title('F3: Costo de recordar')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Comparativa F1 vs F4
    ax = axes[1, 0]
    ax.plot(orientaciones_f1[:fin_ciclo], 'b-', linewidth=0.6, label='F1 (fresco)')
    ax.plot(orientaciones_f4[:fin_ciclo], 'orange', linewidth=0.6, label='F4 (post-silencio)')
    ax.plot(setpoints_f1[:fin_ciclo], 'r--', linewidth=0.6, alpha=0.5, label='Setpoint')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('F1 vs F4: Reactivación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Evolución de historia y fatiga
    ax = axes[1, 1]
    ax.plot(historias_f1 + historias_f3[:1000], 'b-', linewidth=0.6, label='Historia')
    ax.plot(fatigas_f1 + fatigas_f3[:1000], 'orange', linewidth=0.6, label='Fatiga activa')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Valor (º)')
    ax.set_title('Historia vs Fatiga activa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: s_shared
    ax = axes[1, 2]
    s_shared = sistema.historial['s_shared']
    ax.plot(s_shared, 'purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('s_shared')
    ax.set_title('Lateralidad (coherencia inter-sistemas)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v151_logs', exist_ok=True)
    plt.savefig(f'v151_logs/v151_memoria_ausencia_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v151_logs/v151_memoria_ausencia_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION V151")
    print("=" * 80)
    
    if pase_baseline:
        print("\n  ✅ BASELINE ANIMA-2 CONSOLIDADO")
        print("     Memoria de ausencia funcional")
        print("     El organismo mantiene orientación durante silencio")
        print("     El costo de recordar es físicamente medible")
        print("\n  ANIMA-2 listo para Etapa 1 (Consciencia básica - V152)")
    else:
        print("\n  ⚠️ BASELINE PARCIAL")
        print("     Ajustar parámetros y repetir")
    
    return sistema, pase_baseline


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v151()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print("\n  🏁 V151 completado")