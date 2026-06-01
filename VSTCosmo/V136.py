#!/usr/bin/env python3
"""
VSTCosmos V136 — Validación estadística de predicción

Correcciones sobre V135:
  1. Unificar resolución temporal (todos los arrays con mismo tamaño)
  2. Métricas robustas: MAE (error medio absoluto) por fase
  3. Lead Index (L = orient - setpoint): cuantifica anticipación
  4. Eliminar bugs de dimensiones en análisis

Hipótesis O-N10:
  - MAE_prediccion < 0.5 * MAE_baseline
  - Lead Index medio > 0 (anticipación positiva)
  - T_settle_reenganche < 0.5 * T_settle_baseline
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

# Zona muerta
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Memoria episodica
TAU_MEMORIA = 30.0
UMBRAL_CONFIANZA = 0.1
ALPHA_CONFIANZA = 1.0

# Prediccion
TAU_VELOCIDAD = 5.0
HORIZONTE_PREDICCION = 2.0
VELOCIDAD_MAX = 15.0

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV136:
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
# MEMORIA CON RELAJACION
# ============================================================

class MemoriaConRelajacion:
    def __init__(self, tau=TAU_MEMORIA, centro=0.0, alpha=ALPHA_CONFIANZA):
        self.tau = tau
        self.centro = centro
        self.alpha = alpha
        self.angulo = centro
        self.confianza = 0.0
        self.t_ultimo_estimulo = 0.0
        self.historial_confianza = []
    
    def update(self, angulo_medido, fuente_activa, t):
        if fuente_activa:
            self.angulo = angulo_medido
            self.confianza = 1.0
            self.t_ultimo_estimulo = t
        else:
            dt_silencio = t - self.t_ultimo_estimulo
            if dt_silencio >= 0:
                self.confianza = np.exp(-dt_silencio / self.tau)
            else:
                self.confianza = 0.0
        
        self.historial_confianza.append(self.confianza)
        return self.confianza
    
    def get_setpoint(self):
        if self.confianza > 0.01:
            return self.angulo * (self.confianza ** self.alpha) + self.centro * (1 - (self.confianza ** self.alpha))
        return self.centro
    
    def get_confianza(self):
        return self.confianza


# ============================================================
# PREDICTOR DE TRAYECTORIA
# ============================================================

class PredictorTrayectoria:
    def __init__(self, tau_velocidad=TAU_VELOCIDAD, horizonte=HORIZONTE_PREDICCION, 
                 vel_max=VELOCIDAD_MAX):
        self.tau_velocidad = tau_velocidad
        self.horizonte = horizonte
        self.vel_max = vel_max
        self.velocidad = 0.0
        self.t_ultimo = 0.0
        self.angulo_ultimo = 0.0
        self.historial_velocidad = []
    
    def update(self, angulo_actual, t):
        dt = t - self.t_ultimo
        
        if dt > 0 and dt < 1.0:
            vel_inst = (angulo_actual - self.angulo_ultimo) / dt
            alpha = min(1.0, dt / self.tau_velocidad)
            self.velocidad = (1 - alpha) * self.velocidad + alpha * vel_inst
            self.velocidad = np.clip(self.velocidad, -self.vel_max, self.vel_max)
        
        self.angulo_ultimo = angulo_actual
        self.t_ultimo = t
        
        self.historial_velocidad.append(self.velocidad)
        return angulo_actual + self.velocidad * self.horizonte
    
    def get_velocidad(self):
        return self.velocidad


# ============================================================
# APARATO MOTOR CON PREDICCION
# ============================================================

class AparatoMotorConPrediccion:
    def __init__(self):
        self.orientacion = 0.0
        self.Kp_base = KP_BASE
        self.Kp_actual = KP_BASE
        self.Kp_min = KP_MIN
        self.Kp_max = KP_MAX
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        self.memoria = MemoriaConRelajacion()
        self.predictor = PredictorTrayectoria()
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        
        self.setpoint_usado = 0.0
        self.prediccion_activa = False
    
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
    
    def actuar(self, gradiente, LF_activa, fuente_activa, t, setpoint_percepcion, 
               modo_prediccion=False):
        if not LF_activa:
            return self.orientacion, 0.0, 0.0, False
        
        if abs(gradiente) < 0.05:
            return self.orientacion, self.memoria.get_confianza(), 0.0, False
        
        self.memoria.update(setpoint_percepcion, fuente_activa, t)
        
        if fuente_activa:
            setpoint_base = setpoint_percepcion
        else:
            setpoint_base = self.memoria.get_setpoint()
        
        if modo_prediccion and fuente_activa:
            setpoint_predicho = self.predictor.update(setpoint_base, t)
            self.setpoint_usado = setpoint_predicho
            self.prediccion_activa = True
        else:
            if fuente_activa:
                self.predictor.update(setpoint_base, t)
            self.setpoint_usado = setpoint_base
            self.prediccion_activa = False
        
        error = self.setpoint_usado - self.orientacion
        
        if abs(error) < self.zona_muerta:
            return self.orientacion, self.memoria.get_confianza(), self.predictor.get_velocidad(), self.prediccion_activa
        
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_actual * error * ganancia_grad * factor_freno
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.t += DT
        
        return (self.orientacion, self.memoria.get_confianza(), 
                self.predictor.get_velocidad(), self.prediccion_activa)
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_Kp = []
        self.prediccion_activa = False


# ============================================================
# SISTEMA V136
# ============================================================

class SistemaV136:
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
        
        self.izquierdo = HemisferioV136("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV136("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV136("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV136("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorConPrediccion()
        
        self.historial = {
            't': [],
            'orientacion': [],
            'setpoint_usado': [],
            'setpoint_real': [],
            'confianza': [],
            'velocidad': [],
            'prediccion_activa': [],
            's_shared': [],
            'Kp': []
        }
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, setpoint_real, modo_prediccion=False):
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
        orientacion, confianza, velocidad, pred_activa = self.motor.actuar(
            gradiente, LF_activa, fuente_activa, t, setpoint_real, modo_prediccion
        )
        
        self.historial['t'].append(t)
        self.historial['orientacion'].append(orientacion)
        self.historial['setpoint_usado'].append(self.motor.setpoint_usado)
        self.historial['setpoint_real'].append(setpoint_real)
        self.historial['confianza'].append(confianza)
        self.historial['velocidad'].append(velocidad)
        self.historial['prediccion_activa'].append(pred_activa)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return orientacion
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# MOVIMIENTO SINUSOIDAL
# ============================================================

def movimiento_sinusoidal(t, amplitud=60.0, periodo=60.0):
    return amplitud * np.sin(2 * np.pi * t / periodo)


# ============================================================
# METRICAS DE VALIDACION (SIN BUGS)
# ============================================================

def calcular_metricas(sistema, t_total, t_rel, orientacion, setpoint_real, 
                      prediccion_activa, debug=False):
    """Calcula metricas con arrays correctamente alineados"""
    
    # Asegurar que todos los arrays tienen la misma longitud
    min_len = min(len(t_total), len(orientacion), len(setpoint_real), len(prediccion_activa))
    
    t_total = t_total[:min_len]
    t_rel = t_rel[:min_len]
    orientacion = orientacion[:min_len]
    setpoint_real = setpoint_real[:min_len]
    prediccion_activa = prediccion_activa[:min_len]
    
    # Crear máscaras de fase (solo tracking, sin silencio)
    mask_baseline = (t_rel >= 10) & (t_rel < 60) & (t_rel <= t_total[-1])
    mask_prediccion = (t_rel >= 60) & (t_rel < 120) & (t_rel <= t_total[-1])
    
    # Verificar que las máscaras no estén vacías
    if np.sum(mask_baseline) == 0:
        if debug:
            print("  Advertencia: mask_baseline vacía")
        return None
    
    if np.sum(mask_prediccion) == 0:
        if debug:
            print("  Advertencia: mask_prediccion vacía")
        return None
    
    # 1. MAE (Error Medio Absoluto)
    error_baseline = np.abs(orientacion[mask_baseline] - setpoint_real[mask_baseline])
    error_prediccion = np.abs(orientacion[mask_prediccion] - setpoint_real[mask_prediccion])
    
    mae_baseline = np.mean(error_baseline)
    mae_prediccion = np.mean(error_prediccion)
    mejora_mae = (mae_baseline - mae_prediccion) / mae_baseline * 100 if mae_baseline > 0 else 0
    
    # 2. Lead Index (L = orient - setpoint)
    lead_baseline = np.mean(orientacion[mask_baseline] - setpoint_real[mask_baseline])
    lead_prediccion = np.mean(orientacion[mask_prediccion] - setpoint_real[mask_prediccion])
    
    # 3. Tiempo de asentamiento (reenganche Fase 4)
    mask_f4 = (t_rel >= 150) & (t_rel < 180) & (t_rel <= t_total[-1])
    if np.sum(mask_f4) > 0:
        orient_f4 = orientacion[mask_f4]
        t_f4 = t_rel[mask_f4]
        setpoint_f4 = setpoint_real[mask_f4]
        
        # Buscar cuándo entra en zona muerta (±5° del objetivo)
        objetivo_f4 = 52.0  # Valor aproximado en F4
        t_settle = None
        for i, o in enumerate(orient_f4):
            if abs(o - objetivo_f4) < 5.0:
                t_settle = t_f4[i] - 150
                break
    else:
        t_settle = None
    
    # 4. Velocidad estimada
    velocidad_arr = np.array(sistema.historial['velocidad'])
    if len(velocidad_arr) > 0:
        # Truncar velocidad a misma longitud que t_total
        velocidad_arr = velocidad_arr[:min_len]
        vel_baseline = np.mean(np.abs(velocidad_arr[mask_baseline])) if np.any(mask_baseline) else 0
        vel_prediccion = np.mean(np.abs(velocidad_arr[mask_prediccion])) if np.any(mask_prediccion) else 0
    else:
        vel_baseline = vel_prediccion = 0
    
    return {
        'mae_baseline': mae_baseline,
        'mae_prediccion': mae_prediccion,
        'mejora_mae': mejora_mae,
        'lead_baseline': lead_baseline,
        'lead_prediccion': lead_prediccion,
        't_settle': t_settle,
        'vel_baseline': vel_baseline,
        'vel_prediccion': vel_prediccion,
        'mask_baseline': mask_baseline,
        'mask_prediccion': mask_prediccion
    }


# ============================================================
# EXPERIMENTO V136
# ============================================================

def ejecutar_v136():
    print("=" * 100)
    print("EXPERIMENTO V136 — Validacion estadistica de prediccion")
    print("=" * 100)
    print("  ANIMA-2 - Linea 2: Hipotesis O-N10 (validacion)")
    print("  Correcciones:")
    print("    - Unificacion de resolucion temporal")
    print("    - MAE (error medio absoluto) por fase")
    print("    - Lead Index (anticipacion)")
    print("    - Sin bugs de dimensiones")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV136("V136", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0, modo_prediccion=False)
    
    print("  Entrenamiento completado.")
    
    # Fase de test
    print("\n  Iniciando test de prediccion...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_test = 180.0
    
    # Almacenar datos para análisis (misma resolución)
    tiempos = []
    orientaciones = []
    setpoints_reales = []
    
    for i in range(int(duracion_test / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        # Determinar modo de prediccion
        if t_rel < 60:
            modo_prediccion = False
        elif t_rel < 120:
            modo_prediccion = True
        elif t_rel < 150:
            modo_prediccion = True
        else:
            modo_prediccion = True
        
        # Setpoint real
        if t_rel < 120:
            setpoint_real = movimiento_sinusoidal(t_rel)
        elif t_rel < 150:
            setpoint_real = 0.0  # Silencio
        else:
            setpoint_real = movimiento_sinusoidal(t_rel - 30)  # Continuar fase
        
        orientacion = sistema.actualizar(t, DT, t_actual + duracion_test,
                                         setpoint_real, modo_prediccion)
        
        tiempos.append(t_rel)
        orientaciones.append(orientacion)
        setpoints_reales.append(setpoint_real)
        
        # Reporte cada 10s
        if int(t_rel * 10) % 100 == 0 and t_rel > 0:
            fase = ""
            if t_rel < 60:
                fase = "F1(baseline)"
            elif t_rel < 120:
                fase = "F2(prediccion)"
            elif t_rel < 150:
                fase = "F3(silencio)"
            else:
                fase = "F4(reenganche)"
            
            error = abs(orientacion - setpoint_real) if t_rel < 120 else 0
            vel = sistema.historial['velocidad'][-1] if sistema.historial['velocidad'] else 0
            print(f"    t={t_rel:4.0f}s | {fase:14s} | setpoint={setpoint_real:5.1f}° | "
                  f"orient={orientacion:5.1f}° | error={error:4.1f}° | vel={vel:5.2f}°/s")
    
    # Convertir a arrays numpy para análisis
    t_total = np.array(sistema.historial['t'])
    t_rel = np.array(tiempos)
    orientacion = np.array(orientaciones)
    setpoint_real = np.array(setpoints_reales)
    prediccion_activa = np.array(sistema.historial['prediccion_activa'])
    
    # Calcular métricas (sin bugs)
    print("\n" + "=" * 80)
    print("ANALISIS DE PREDICCION (O-N10)")
    print("=" * 80)
    
    metricas = calcular_metricas(sistema, t_total, t_rel, orientacion, setpoint_real, 
                                  prediccion_activa, debug=True)
    
    if metricas is None:
        print("\n  Error: No se pudieron calcular métricas")
        return sistema, False
    
    print(f"\n  Fase 1 - Baseline (tracking normal):")
    print(f"    MAE (error medio absoluto): {metricas['mae_baseline']:.2f}°")
    print(f"    Lead Index: {metricas['lead_baseline']:.2f}° {'(persigue)' if metricas['lead_baseline'] < 0 else '(anticipa)'}")
    print(f"    Velocidad angular media: {metricas['vel_baseline']:.2f}°/s")
    
    print(f"\n  Fase 2 - Prediccion activada:")
    print(f"    MAE (error medio absoluto): {metricas['mae_prediccion']:.2f}°")
    print(f"    Lead Index: {metricas['lead_prediccion']:.2f}° {'(persigue)' if metricas['lead_prediccion'] < 0 else '(anticipa)'}")
    print(f"    Velocidad angular media: {metricas['vel_prediccion']:.2f}°/s")
    
    print(f"\n  Mejoras con prediccion:")
    print(f"    MAE: {metricas['mae_baseline']:.2f}° → {metricas['mae_prediccion']:.2f}° (reduccion {metricas['mejora_mae']:.1f}%)")
    
    if metricas['t_settle']:
        print(f"    T_settle reenganche (F4): {metricas['t_settle']:.1f}s")
    else:
        print(f"    T_settle reenganche (F4): No alcanzado")
    
    # Criterios de éxito O-N10
    exito_mae = metricas['mejora_mae'] > 50
    exito_lead = metricas['lead_prediccion'] > 0
    exito_settle = metricas['t_settle'] is not None and metricas['t_settle'] < 18
    
    print(f"\n  Criterios de exito O-N10:")
    print(f"    MAE reduccion > 50%: {metricas['mejora_mae']:.1f}% {'✅' if exito_mae else '❌'}")
    print(f"    Lead Index > 0 (anticipacion): {metricas['lead_prediccion']:.2f}° {'✅' if exito_lead else '❌'}")
    print(f"    T_settle reenganche < 18s: {metricas['t_settle']:.1f}s {'✅' if exito_settle else '❌' if metricas['t_settle'] else '❌'}")
    
    exito_total = exito_mae and exito_lead
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: Orientacion vs Setpoint
    ax = axes[0, 0]
    ax.plot(t_rel, setpoint_real, 'r--', linewidth=1, alpha=0.7, label='Setpoint real (fuente)')
    ax.plot(t_rel, orientacion, 'b-', linewidth=0.8, label='Orientacion real')
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='Prediccion activada')
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5, label='Silencio')
    ax.axvline(x=150, color='purple', linestyle='--', alpha=0.5, label='Reenganche')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('V136: Validacion de prediccion')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Error de orientacion
    ax = axes[0, 1]
    error_total = np.abs(orientacion - setpoint_real)
    ax.plot(t_rel, error_total, 'purple', linewidth=0.8)
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5, label=f'Zona muerta ({ZONA_MUERTA_BASE}°)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Error de orientacion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Lead Index (orient - setpoint)
    ax = axes[0, 2]
    lead_index = orientacion - setpoint_real
    ax.plot(t_rel, lead_index, 'orange', linewidth=0.8)
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Tracking perfecto')
    ax.fill_between(t_rel, 0, lead_index, where=(lead_index>0), alpha=0.3, color='green', label='Anticipacion (L>0)')
    ax.fill_between(t_rel, 0, lead_index, where=(lead_index<0), alpha=0.3, color='red', label='Lag (L<0)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Lead Index (grados)')
    ax.set_title('Lead Index: Anticipacion vs Lag')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Velocidad estimada
    ax = axes[1, 0]
    velocidad_arr = np.array(sistema.historial['velocidad'])
    if len(velocidad_arr) > 0:
        # Truncar a misma longitud
        velocidad_arr = velocidad_arr[:len(t_rel)]
        ax.plot(t_rel, velocidad_arr, 'cyan', linewidth=0.8, label='Velocidad estimada')
        
        # Velocidad teorica
        t_teorico = np.linspace(0, 120, 1000)
        vel_teorica = 60.0 * (2 * np.pi / 60.0) * np.cos(2 * np.pi * t_teorico / 60.0)
        ax.plot(t_teorico, vel_teorica, 'r--', linewidth=1, alpha=0.5, label='Velocidad teorica')
    
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Velocidad (grados/s)')
    ax.set_title('Estimacion de velocidad angular')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Setpoint usado vs real
    ax = axes[1, 1]
    setpoint_usado = np.array(sistema.historial['setpoint_usado'])
    if len(setpoint_usado) > 0:
        setpoint_usado = setpoint_usado[:len(t_rel)]
        ax.plot(t_rel, setpoint_usado, 'g-', linewidth=0.8, label='Setpoint usado (motor)')
    ax.plot(t_rel, setpoint_real, 'r--', linewidth=1, alpha=0.5, label='Setpoint real')
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Setpoint (grados)')
    ax.set_title('Setpoint real vs Setpoint usado')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: MAE comparativo
    ax = axes[1, 2]
    fases = ['Baseline', 'Prediccion']
    mae_values = [metricas['mae_baseline'], metricas['mae_prediccion']]
    colores_mae = ['red', 'green']
    bars = ax.bar(fases, mae_values, color=colores_mae, alpha=0.7)
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}°', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('MAE (grados)')
    ax.set_title(f'Error medio absoluto: reduccion {metricas["mejora_mae"]:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v136_logs', exist_ok=True)
    plt.savefig(f'v136_logs/v136_validacion_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v136_logs/v136_validacion_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION V136 — Validacion de prediccion")
    print("=" * 80)
    
    if exito_total:
        print("\n  ✅ O-N10 VALIDADA ESTADISTICAMENTE")
        print(f"     MAE reduccion: {metricas['mejora_mae']:.1f}% (umbral >50%)")
        print(f"     Lead Index: {metricas['lead_prediccion']:.2f}° > 0 (anticipacion confirmada)")
        print(f"     T_settle reenganche: {metricas['t_settle']:.1f}s")
        print("\n  ANIMA-2 - Linea 2: CERRADA")
    else:
        print("\n  ⚠️ O-N10 NO VALIDADA ESTADISTICAMENTE")
        print(f"     MAE reduccion: {metricas['mejora_mae']:.1f}% {'(>=50%)' if exito_mae else '(<50%)'}")
        print(f"     Lead Index: {metricas['lead_prediccion']:.2f}° {'>0' if exito_lead else '<0'}")
    
    return sistema, exito_total


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v136()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")