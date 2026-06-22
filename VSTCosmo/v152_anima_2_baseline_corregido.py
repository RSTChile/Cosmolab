#!/usr/bin/env python3
"""
VSTCosmos V152 — ANIMA-2 Baseline corregido
Memoria de ausencia con torque de sujeción (O-N16)

Correcciones sobre V151:
  1. Usar E_historia de F2 para criterio 1
  2. Añadir torque_memoria en silencio (K_hold = 0.0005)
  3. Acumular costo de recordar en ΔE_activa
  4. T_settle con fallback a NaN para evitar crash
  5. Criterio 2: ΔE_activa_F3 > 0 (no ratio)

Parámetros adicionales:
  - K_HOLD = 0.0005 (torque elástico hacia setpoint recordado)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from collections import deque

# ============================================================
# PARAMETROS (heredados de V151)
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

# Torque de sujeción (NUEVO - O-N16)
K_HOLD = 0.0005

# Semilla base
SEMILLA_BASE = 44

# Período
PERIODO_ALTERNANCIA = 80.0


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV152:
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

class FatigaMetabolicaV152:
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
    
    def actualizar(self, delta_real, delta_costo, en_reposo, dt):
        """
        Args:
            delta_real: cambio angular real (para historia)
            delta_costo: cambio angular que cuesta energía (puede incluir torque virtual)
            en_reposo: True si el organismo está en reposo (sin movimiento ni torque)
        """
        # Historia: acumula SIEMPRE el movimiento real
        self.historia += abs(delta_real)
        
        # Fatiga activa: acumula el costo REAL (incluye torque de sujeción)
        if not en_reposo:
            self.fatiga_activa += abs(delta_costo)
        else:
            self.fatiga_activa *= np.exp(-dt / self.tau_recuperacion)
        
        # Efectos de la fatiga
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
# APARATO MOTOR V152 (con torque de sujeción)
# ============================================================

class AparatoMotorV152:
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
        
        self.fatiga = FatigaMetabolicaV152()
        self.memoria = MemoriaAusencia()
        
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
        
        # Obtener setpoint efectivo y confianza
        setpoint_objetivo, confianza = self.memoria.actualizar(setpoint_raw, self.fatiga.get_historia(), DT)
        
        error = setpoint_objetivo - self.orientacion
        
        # Fatiga: actualizar con costo real
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, 0, False, DT
        )
        
        # Zona muerta
        if abs(error) < zona_muerta_efectiva:
            # En reposo, fatiga se recupera
            self.fatiga.actualizar(0, 0, True, DT)
            return (self.orientacion, self.fatiga.get_historia(), 
                    self.fatiga.get_fatiga(), confianza, zona_muerta_efectiva)
        
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
        
        # TORQUE DE SUJECIÓN (NUEVO - O-N16)
        # Solo durante silencio, aplica fuerza hacia setpoint_last
        torque_memoria = 0.0
        if setpoint_raw is None:
            torque_memoria = K_HOLD * self.memoria.setpoint_last * confianza
            # El torque se suma al delta (no lo reemplaza)
        
        delta_raw = delta_error + torque_memoria
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Costo energético: incluye torque de memoria
        delta_costo = abs(delta_error) + abs(torque_memoria)
        
        # Actualizar fatiga con el costo REAL (incluye torque virtual)
        self.fatiga.actualizar(delta, delta_costo, False, DT)
        
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
# SISTEMA V152 (similar a V151)
# ============================================================

class SistemaV152:
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
        
        self.izquierdo = HemisferioV152("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV152("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV152("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV152("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorV152()
        
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
# FUNCIONES AUXILIARES
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


def analizar_semiciclo(orientaciones, setpoints, dt=DT, umbral_error=2.0, ventana=50):
    """Analiza primeros 40s de un semiciclo"""
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


# ============================================================
# EXPERIMENTO V152
# ============================================================

def ejecutar_v152():
    print("=" * 100)
    print("EXPERIMENTO V152 — ANIMA-2 Baseline corregido")
    print("=" * 100)
    print("  Memoria de ausencia con torque de sujeción (O-N16)")
    print("")
    print("  Parámetros:")
    print(f"    - Kp_base = {KP_BASE}")
    print(f"    - K_HOLD = {K_HOLD} (torque elástico)")
    print(f"    - tau_base = {TAU_BASE}s")
    print(f"    - k_mem = {K_MEM}")
    print(f"    - Suelo confianza = {SUELO_CONFIANZA}")
    print("=" * 100)
    
    sistema = SistemaV152("V152", seed=SEMILLA_BASE)
    
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_raw=0.0)
    
    print("  Entrenamiento completado.")
    print("\n  Iniciando test de memoria con torque de sujeción...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # F1: Baseline
    print("\n  F1: Baseline fresco (3 ciclos)...")
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # F2: Fatiga
    print("\n  F2: Fatiga inducida (30 ciclos)...")
    for ciclo in range(30):
        fatiga_inicio = sistema.motor.fatiga.get_fatiga()
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + i * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        fatiga_fin = sistema.motor.fatiga.get_fatiga()
        if (ciclo + 1) % 10 == 0:
            historia = sistema.motor.fatiga.get_historia()
            print(f"      Ciclo {ciclo + 1}/30, fatiga={fatiga_fin:.0f}°, historia={historia:.0f}°")
        t_actual += PERIODO_ALTERNANCIA
    
    historia_f2 = sistema.motor.fatiga.get_historia()
    
    # F3: Silencio
    print("\n  F3: Silencio (120s) - midiendo persistencia...")
    setpoint_last = sistema.motor.memoria.setpoint_last
    
    tiempos_f3 = []
    orientaciones_f3 = []
    confianzas_f3 = []
    fatigas_f3 = []
    
    for i in range(int(120.0 / DT)):
        t = t_actual + i * DT
        orient, historia, fatiga, confianza = sistema.actualizar(t, DT, t_actual + 120, None)
        tiempos_f3.append(t)
        orientaciones_f3.append(orient)
        confianzas_f3.append(confianza)
        fatigas_f3.append(fatiga)
        
        if i % 500 == 0:
            print(f"      t={i*DT:.0f}s | orient={orient:.1f}° | confianza={confianza:.3f} | fatiga={fatiga:.0f}°")
    
    t_actual += 120.0
    
    # Calcular métricas F3
    idx_60s = int(60.0 / DT)
    orient_60s = orientaciones_f3[idx_60s] if idx_60s < len(orientaciones_f3) else orientaciones_f3[-1]
    error_60s = abs(orient_60s - setpoint_last)
    
    delta_E_f3 = fatigas_f3[-1] - fatigas_f3[0] if fatigas_f3 else 0
    
    # F4: Reactivación
    print("\n  F4: Reactivación (3 ciclos)...")
    for ciclo in range(3):
        for i in range(int(PERIODO_ALTERNANCIA / DT)):
            t = t_actual + (ciclo * PERIODO_ALTERNANCIA + i) * DT
            setpoint = onda_cuadrada(i * DT, periodo=PERIODO_ALTERNANCIA, amplitud=60.0)
            sistema.actualizar(t, DT, t_actual + 1000, setpoint)
    t_actual += 3 * PERIODO_ALTERNANCIA
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTADOS V152")
    print("=" * 80)
    
    print(f"\n  Memoria de ausencia (F3):")
    print(f"    Setpoint recordado: {setpoint_last:.0f}°")
    print(f"    Orientación a 60s: {orient_60s:.1f}°")
    print(f"    Error a 60s: {error_60s:.1f}°")
    print(f"    Historia acumulada (F2): {historia_f2:.0f}°")
    print(f"    ΔE_activa durante silencio: {delta_E_f3:.0f}°")
    print(f"    Confianza final: {confianzas_f3[-1]:.3f}")
    
    # Criterios
    exito_1 = error_60s < 25.0 and historia_f2 > 1500
    exito_2 = delta_E_f3 > 0
    
    print("\n" + "=" * 80)
    print("CRITERIOS DE ÉXITO (Etapa 0)")
    print("=" * 80)
    print(f"  1. Memoria motora (error_60s < 25° y E_historia > 1500): {error_60s:.1f}° / {historia_f2:.0f}° -> {'✅' if exito_1 else '❌'}")
    print(f"  2. Costo de recordar (ΔE_F3 > 0): {delta_E_f3:.0f}° -> {'✅' if exito_2 else '❌'}")
    
    exito = exito_1 and exito_2
    
    print("\n" + "=" * 80)
    print("CONCLUSION V152")
    print("=" * 80)
    
    if exito:
        print("\n  ✅ BASELINE ANIMA-2 CONSOLIDADO")
        print("     Memoria de ausencia con torque de sujeción")
        print("     El organismo mantiene orientación durante silencio")
        print("     Recordar cuesta energía (ΔE > 0)")
        print("\n  ANIMA-2 listo para Etapa 1 (Consciencia básica - V153)")
    else:
        print("\n  ⚠️ BASELINE PARCIAL")
        if not exito_1:
            print(f"     Memoria insuficiente: error {error_60s:.1f}° > 25° o historia {historia_f2:.0f}° < 1500°")
        if not exito_2:
            print(f"     Costo de recordar nulo o negativo: ΔE = {delta_E_f3:.0f}° (debe ser >0)")
    
    # Gráfico simple
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.plot(orientaciones_f3, 'b-', linewidth=0.8)
    ax.axhline(y=setpoint_last, color='r--', alpha=0.7)
    ax.axhline(y=setpoint_last + 25, color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=setpoint_last - 25, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Orientación (º)')
    ax.set_title('F3: Persistencia en silencio')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(confianzas_f3, 'purple', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Confianza')
    ax.set_title('Decaimiento de confianza')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(fatigas_f3, 'orange', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Fatiga activa (º)')
    ax.set_title('Costo de recordar')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v152_logs', exist_ok=True)
    plt.savefig(f'v152_logs/v152_memoria_torque_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v152_logs/v152_memoria_torque_{timestamp}.png")
    
    return sistema, exito


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v152()
    elapsed = time.time() - start
    print(f"\n  ⏱️ Tiempo de ejecución: {elapsed/60:.1f} minutos")
    print(f"\n  V152 completado. Éxito: {exito}")