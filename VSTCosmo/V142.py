#!/usr/bin/env python3
"""
VSTCosmos V142 — Fatiga real corregida (con alternancia forzada)

Correcciones sobre V141:
  1. Onda cuadrada para forzar alternancia -60° ↔ +60°
  2. Zona muerta reducida a 1.0° durante fatiga (evita "escondite")
  3. Parámetros de fatiga 30x más agresivos (escala biológica realista)
  4. Criterio de T_settle corregido (error < zona_muerta * 1.5)
  5. Logging mejorado para verificar alternancia real

Hipotesis O-N11 (revisada):
  - T_settle_fatigado / T_settle_fresco > 1.5
  - Recuperacion > 30% tras 60s reposo
  - Temblor visible (>1°) en estado fatigado
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

# Zona muerta base (reducida para evitar escondite)
ZONA_MUERTA_BASE = 1.0  # ← ANTES: 2.0°

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Fatiga V142 (parámetros corregidos - escala biológica realista)
K_GAIN = 0.003        # 37x más agresivo (ANTES: 0.00008)
K_PRECISION = 0.02    # 40x más (ANTES: 0.0005)
K_TEMBLOR = 0.01      # 33x más (ANTES: 0.0003)
TAU_RECUPERACION = 300.0  # 5 minutos (ANTES: 120s)

# Semilla base
SEMILLA_BASE = 44

# Periodo de alternancia (onda cuadrada)
PERIODO_ALTERNANCIA = 20.0  # segundos: 10s en -60°, 10s en +60°


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV142:
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
# FATIGA METABOLICA REAL (V142 - PARAMETROS CORREGIDOS)
# ============================================================

class FatigaMetabolicaReal:
    """
    Fatiga con parámetros corregidos para escala biológica realista.
    """
    
    def __init__(self, k_gain=K_GAIN, k_precision=K_PRECISION, 
                 k_temblor=K_TEMBLOR, tau_recuperacion=TAU_RECUPERACION):
        self.k_gain = k_gain
        self.k_precision = k_precision
        self.k_temblor = k_temblor
        self.tau_recuperacion = tau_recuperacion
        
        self.energia_total = 0.0  # Acumulador GLOBAL
        self.historial_energia = []
        self.historial_factor_gain = []
        self.historial_zona_muerta = []
        self.historial_temblor = []
    
    def actualizar(self, delta_orientacion, en_reposo, dt):
        # Actualizar energía acumulada
        if not en_reposo:
            self.energia_total += abs(delta_orientacion)
        else:
            # Recuperación exponencial durante reposo
            self.energia_total *= np.exp(-dt / self.tau_recuperacion)
        
        # Calcular efectos de fatiga (ahora con parámetros agresivos)
        factor_gain = np.exp(-self.k_gain * self.energia_total)
        zona_muerta_efectiva = ZONA_MUERTA_BASE + self.k_precision * self.energia_total
        temblor = self.k_temblor * self.energia_total * np.random.randn()
        
        # Limitar valores extremos
        factor_gain = max(0.1, min(1.0, factor_gain))
        zona_muerta_efectiva = min(20.0, zona_muerta_efectiva)
        temblor = np.clip(temblor, -10.0, 10.0)
        
        # Guardar historial
        self.historial_energia.append(self.energia_total)
        self.historial_factor_gain.append(factor_gain)
        self.historial_zona_muerta.append(zona_muerta_efectiva)
        self.historial_temblor.append(temblor)
        
        return factor_gain, zona_muerta_efectiva, temblor
    
    def reset(self):
        self.energia_total = 0.0
        self.historial_energia = []
        self.historial_factor_gain = []
        self.historial_zona_muerta = []
        self.historial_temblor = []
    
    def get_energia(self):
        return self.energia_total


# ============================================================
# APARATO MOTOR CON FATIGA REAL (V142)
# ============================================================

class AparatoMotorConFatigaReal:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        # Fatiga (parámetros corregidos)
        self.fatiga = FatigaMetabolicaReal()
        
        # Plasticidad
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.historial_fatiga = []
        
        self.ultimo_delta_registrado = 0.0
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        if oscilacion > ZONA_MUERTA_BASE * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * 0.99)
        elif oscilacion < ZONA_MUERTA_BASE * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * 1.01)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.fatiga.get_energia(), 0.0
        
        # Setpoint objetivo
        setpoint_objetivo = setpoint_percepcion if fuente_activa else 0.0
        
        error = setpoint_objetivo - self.orientacion
        
        # Actualizar fatiga y obtener efectos
        factor_gain, zona_muerta_efectiva, temblor = self.fatiga.actualizar(
            self.ultimo_delta_registrado, not fuente_activa, DT
        )
        
        # Zona muerta aumentada por fatiga
        if abs(error) < zona_muerta_efectiva:
            return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
        
        # Control proporcional
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp reducido por fatiga
        Kp_efectivo = self.Kp_actual * factor_gain
        
        delta_raw = Kp_efectivo * error * ganancia_grad * factor_freno
        
        # Inercia del motor
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta_raw
        self.ultimo_delta = delta
        self.ultimo_delta_registrado = delta
        
        # Añadir temblor por fatiga
        delta += temblor * DT
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_fatiga.append(factor_gain)
        self.t += DT
        
        return self.orientacion, self.fatiga.get_energia(), zona_muerta_efectiva
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.ultimo_delta_registrado = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.historial_fatiga = []
        self.fatiga.reset()


# ============================================================
# SISTEMA V142
# ============================================================

class SistemaV142:
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
        
        self.izquierdo = HemisferioV142("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV142("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV142("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV142("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConFatigaReal()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_real': [],
            'energia': [],
            'zona_muerta': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_real):
        fuente_activa = True
        
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        sesgo = setpoint_real / 90.0
        gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        orientacion, energia, zona_muerta = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['energia'].append(energia)
        self.historial['zona_muerta'].append(zona_muerta)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion, energia
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# ONDA CUADRADA PARA ALTERNANCIA FORZADA
# ============================================================

def onda_cuadrada(t, periodo=PERIODO_ALTERNANCIA, amplitud=60.0):
    """
    Alterna entre -60° y +60° cada periodo/2 segundos.
    Fuerza al organismo a viajar todo el arco de 120°.
    """
    if (t % periodo) < (periodo / 2):
        return -amplitud
    else:
        return +amplitud


# ============================================================
# FUNCIONES DE ANALISIS
# ============================================================

def calcular_t_settle(orientaciones, setpoints, zona_muerta_efectiva, dt=DT):
    """
    Calcula T_settle: tiempo hasta que error < zona_muerta * 1.5
    durante 50 pasos consecutivos (0.5 segundos).
    """
    if len(orientaciones) == 0:
        return None
    
    errores = np.abs(np.array(orientaciones) - np.array(setpoints))
    umbral_settle = zona_muerta_efectiva * 1.5
    
    for i in range(len(errores) - 50):
        if all(errores[i:i+50] < umbral_settle):
            return i * dt
    return None


def analizar_fase(orientaciones, setpoints, energias, zonas_muerta, nombre_fase):
    """Analiza una fase completa del experimento"""
    
    if len(orientaciones) == 0:
        return {
            'nombre': nombre_fase,
            't_settle': None,
            'error_final': None,
            'energia': 0,
            'velocidad_media': 0,
            'temblor_visible': False,
            'alternancia_verificada': False
        }
    
    # Error final
    error_final = abs(orientaciones[-1] - setpoints[-1]) if len(orientaciones) > 0 else None
    
    # Energia acumulada
    energia_final = energias[-1] if len(energias) > 0 else 0
    
    # Velocidad media
    if len(orientaciones) > 1:
        diffs = np.abs(np.diff(orientaciones))
        velocidad_media = np.mean(diffs) / DT
    else:
        velocidad_media = 0
    
    # T_settle (usando zona muerta efectiva media)
    zona_muerta_media = np.mean(zonas_muerta) if len(zonas_muerta) > 0 else ZONA_MUERTA_BASE
    t_settle = calcular_t_settle(orientaciones, setpoints, zona_muerta_media, DT)
    
    # Verificar alternancia (¿el organismo cambió realmente de polo?)
    alternancia = (max(orientaciones) - min(orientaciones)) > 90.0 if len(orientaciones) > 0 else False
    
    # Temblor visible (desviación estándar > 1.0°)
    temblor_visible = np.std(orientaciones[-500:]) > 1.0 if len(orientaciones) > 500 else False
    
    return {
        'nombre': nombre_fase,
        't_settle': t_settle,
        'error_final': error_final,
        'energia': energia_final,
        'velocidad_media': velocidad_media,
        'alternancia_verificada': alternancia,
        'temblor_visible': temblor_visible
    }


# ============================================================
# EXPERIMENTO V142
# ============================================================

def ejecutar_v142():
    print("=" * 100)
    print("EXPERIMENTO V142 — Fatiga real corregida (alternancia forzada)")
    print("=" * 100)
    print("  ANIMA-2 - Linea 3: Hipotesis O-N11 (revisada)")
    print("")
    print("  Correcciones:")
    print("    - Onda cuadrada: alternancia forzada -60° ↔ +60°")
    print(f"    - Zona muerta base: {ZONA_MUERTA_BASE}° (reducida)")
    print(f"    - K_GAIN: {K_GAIN} (37x V141)")
    print(f"    - K_PRECISION: {K_PRECISION} (40x V141)")
    print(f"    - K_TEMBLOR: {K_TEMBLOR} (33x V141)")
    print(f"    - TAU_RECUPERACION: {TAU_RECUPERACION}s")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV142("V142", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0)
    
    print("  Entrenamiento completado.")
    print("  Iniciando test de fatiga real con alternancia forzada...")
    
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    
    # ============================================================
    # FASE 1: Baseline fresco (10 ciclos de onda cuadrada)
    # ============================================================
    print("\n  Fase 1: Baseline fresco (10 ciclos de alternancia forzada)...")
    
    tiempos_f1 = []
    orientaciones_f1 = []
    setpoints_f1 = []
    energias_f1 = []
    zonas_f1 = []
    
    for i in range(int(10 * PERIODO_ALTERNANCIA / DT)):  # 10 ciclos completos
        t = t_actual + i * DT
        t_rel = i * DT
        
        setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA)
        orient, energia = sistema.actualizar(t, DT, t_actual + 200, setpoint)
        
        tiempos_f1.append(t_rel)
        orientaciones_f1.append(orient)
        setpoints_f1.append(setpoint)
        energias_f1.append(energia)
        zonas_f1.append(sistema.historial['zona_muerta'][-1])
        
        if i % 2000 == 0:
            print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}° | energia={energia:.0f}°")
    
    t_actual += 10 * PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 2: Fatiga inducida (50 ciclos)
    # ============================================================
    print("\n  Fase 2: Fatiga inducida (50 ciclos de alternancia forzada)...")
    
    for i in range(int(50 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA)
        orient, energia = sistema.actualizar(t, DT, t_actual + 1000, setpoint)
        
        if i % 10000 == 0:
            print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}° | energia={energia:.0f}°")
    
    t_actual += 50 * PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 3: Test fatiga (10 ciclos)
    # ============================================================
    print("\n  Fase 3: Test fatiga (10 ciclos)...")
    
    tiempos_f3 = []
    orientaciones_f3 = []
    setpoints_f3 = []
    energias_f3 = []
    zonas_f3 = []
    
    for i in range(int(10 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA)
        orient, energia = sistema.actualizar(t, DT, t_actual + 200, setpoint)
        
        tiempos_f3.append(t_rel)
        orientaciones_f3.append(orient)
        setpoints_f3.append(setpoint)
        energias_f3.append(energia)
        zonas_f3.append(sistema.historial['zona_muerta'][-1])
        
        if i % 2000 == 0:
            print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}° | energia={energia:.0f}°")
    
    t_actual += 10 * PERIODO_ALTERNANCIA
    
    # ============================================================
    # FASE 4: Recuperacion (60s reposo)
    # ============================================================
    print("\n  Fase 4: Recuperacion (60s reposo)...")
    
    for i in range(int(60.0 / DT)):
        t = t_actual + i * DT
        orient, energia = sistema.actualizar(t, DT, t_actual + 60, 0.0)
    
    t_actual += 60.0
    
    # ============================================================
    # FASE 5: Test post-recuperacion (10 ciclos)
    # ============================================================
    print("\n  Fase 5: Post-recuperacion (10 ciclos)...")
    
    tiempos_f5 = []
    orientaciones_f5 = []
    setpoints_f5 = []
    energias_f5 = []
    zonas_f5 = []
    
    for i in range(int(10 * PERIODO_ALTERNANCIA / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        setpoint = onda_cuadrada(t_rel, periodo=PERIODO_ALTERNANCIA)
        orient, energia = sistema.actualizar(t, DT, t_actual + 200, setpoint)
        
        tiempos_f5.append(t_rel)
        orientaciones_f5.append(orient)
        setpoints_f5.append(setpoint)
        energias_f5.append(energia)
        zonas_f5.append(sistema.historial['zona_muerta'][-1])
        
        if i % 2000 == 0:
            print(f"      t={t_rel:.0f}s | setpoint={setpoint:+.0f}° | orient={orient:.1f}° | energia={energia:.0f}°")
    
    # ============================================================
    # ANALISIS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE FATIGA REAL (V142)")
    print("=" * 80)
    
    # Analizar cada fase
    fresco = analizar_fase(orientaciones_f1, setpoints_f1, energias_f1, zonas_f1, "F1: Baseline fresco")
    fatigado = analizar_fase(orientaciones_f3, setpoints_f3, energias_f3, zonas_f3, "F3: Fatigado")
    recuperado = analizar_fase(orientaciones_f5, setpoints_f5, energias_f5, zonas_f5, "F5: Post-recuperacion")
    
    print(f"\n  {fresco['nombre']}:")
    print(f"    T_settle: {fresco['t_settle']:.1f}s" if fresco['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {fresco['error_final']:.2f}°" if fresco['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {fresco['energia']:.0f}°")
    print(f"    Velocidad media: {fresco['velocidad_media']:.2f}°/s")
    print(f"    Alternancia verificada: {'✅' if fresco['alternancia_verificada'] else '❌'}")
    print(f"    Temblor visible: {'✅' if fresco['temblor_visible'] else '❌'}")
    
    print(f"\n  {fatigado['nombre']}:")
    print(f"    T_settle: {fatigado['t_settle']:.1f}s" if fatigado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {fatigado['error_final']:.2f}°" if fatigado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {fatigado['energia']:.0f}°")
    print(f"    Velocidad media: {fatigado['velocidad_media']:.2f}°/s")
    print(f"    Alternancia verificada: {'✅' if fatigado['alternancia_verificada'] else '❌'}")
    print(f"    Temblor visible: {'✅' if fatigado['temblor_visible'] else '❌'}")
    
    print(f"\n  {recuperado['nombre']}:")
    print(f"    T_settle: {recuperado['t_settle']:.1f}s" if recuperado['t_settle'] else "    T_settle: No alcanzado")
    print(f"    Error final: {recuperado['error_final']:.2f}°" if recuperado['error_final'] else "    Error final: N/A")
    print(f"    Energia acumulada: {recuperado['energia']:.0f}°")
    print(f"    Velocidad media: {recuperado['velocidad_media']:.2f}°/s")
    print(f"    Alternancia verificada: {'✅' if recuperado['alternancia_verificada'] else '❌'}")
    print(f"    Temblor visible: {'✅' if recuperado['temblor_visible'] else '❌'}")
    
    # Calcular degradacion y recuperacion
    if fresco['t_settle'] and fatigado['t_settle']:
        degradacion = fatigado['t_settle'] / fresco['t_settle']
        print(f"\n  Degradacion por fatiga: {degradacion:.2f}x {'✅' if degradacion > 1.5 else '❌'} (objetivo >1.5x)")
    else:
        degradacion = None
        print(f"\n  Degradacion por fatiga: No calculable")
    
    if fresco['t_settle'] and fatigado['t_settle'] and recuperado['t_settle']:
        recuperacion = (fatigado['t_settle'] - recuperado['t_settle']) / (fatigado['t_settle'] - fresco['t_settle']) * 100
        print(f"  Recuperacion post-reposo: {recuperacion:.1f}% {'✅' if recuperacion > 30 else '❌'} (objetivo >30%)")
    else:
        recuperacion = None
    
    # Criterios O-N11
    exito_fatiga = degradacion is not None and degradacion > 1.5
    exito_recuperacion = recuperacion is not None and recuperacion > 30
    exito_total = exito_fatiga and exito_recuperacion
    
    print("\n" + "=" * 80)
    print("CONCLUSION V142 — Fatiga real corregida")
    print("=" * 80)
    
    if exito_total:
        print("\n  ✅ O-N11 VALIDADA")
        print("     La fatiga degrada el rendimiento significativamente")
        print("     El reposo restaura parcialmente la funcion")
        print("\n  ANIMA-2 - Linea 3: CERRADA")
    else:
        print("\n  ⚠️ O-N11 NO VALIDADA")
        if degradacion:
            print(f"     Degradacion: {degradacion:.2f}x (<1.5x)")
        if recuperacion:
            print(f"     Recuperacion: {recuperacion:.1f}% (<30%)")
    
    # Graficos
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: Orientacion vs Setpoint (Fase 1 y 3)
    ax = axes[0, 0]
    ax.plot(tiempos_f1, setpoints_f1, 'r--', linewidth=0.8, alpha=0.5, label='Setpoint F1')
    ax.plot(tiempos_f1, orientaciones_f1, 'b-', linewidth=0.6, label='Orientacion F1 (fresco)')
    ax.plot(tiempos_f3, orientaciones_f3, 'orange', linewidth=0.6, label='Orientacion F3 (fatigado)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('Comparativa: Fresco vs Fatigado')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Energia acumulada
    ax = axes[0, 1]
    todas_energias = energias_f1 + energias_f3 + energias_f5
    ax.plot(todas_energias, 'r-', linewidth=0.8)
    ax.set_xlabel('Paso')
    ax.set_ylabel('Energia acumulada (grados)')
    ax.set_title('Energia metabolica global')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: T_settle comparativo
    ax = axes[0, 2]
    fases = ['Fresco', 'Fatigado', 'Recuperado']
    t_settle_vals = [
        fresco['t_settle'] or 0,
        fatigado['t_settle'] or 0,
        recuperado['t_settle'] or 0
    ]
    colores = ['green', 'red', 'blue']
    bars = ax.bar(fases, t_settle_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, t_settle_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('T_settle (segundos)')
    ax.set_title('Degradacion por fatiga')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 4: Zona muerta efectiva durante fatiga
    ax = axes[1, 0]
    ax.plot(zonas_f1[:5000], 'green', linewidth=0.6, label='Fresco')
    ax.plot(zonas_f3[:5000], 'red', linewidth=0.6, label='Fatigado')
    ax.set_xlabel('Paso')
    ax.set_ylabel('Zona muerta (grados)')
    ax.set_title('Zona muerta efectiva por fatiga')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Velocidad media
    ax = axes[1, 1]
    vel_vals = [
        fresco['velocidad_media'],
        fatigado['velocidad_media'],
        recuperado['velocidad_media']
    ]
    bars = ax.bar(fases, vel_vals, color=colores, alpha=0.7)
    for bar, val in zip(bars, vel_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}°/s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Velocidad media (grados/s)')
    ax.set_title('Velocidad de giro (proxy de fatiga)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafico 6: Temblor visible (desviacion estandar)
    ax = axes[1, 2]
    # Calcular desviacion estandar en ventanas deslizantes
    ventana = 500
    if len(orientaciones_f3) > ventana:
        temblor_fatigado = [np.std(orientaciones_f3[i:i+ventana]) for i in range(0, len(orientaciones_f3)-ventana, ventana//2)]
        ax.plot(temblor_fatigado, 'orange', linewidth=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', label='Umbral temblor visible (1°)')
    ax.set_xlabel('Ventana')
    ax.set_ylabel('Desviacion estandar (grados)')
    ax.set_title('Temblor por fatiga')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v142_logs', exist_ok=True)
    plt.savefig(f'v142_logs/v142_fatiga_corregida_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v142_logs/v142_fatiga_corregida_{timestamp}.png")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v142()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")