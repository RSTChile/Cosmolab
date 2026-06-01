#!/usr/bin/env python3
"""
VSTCosmos V135 — Predicción de trayectoria

ANIMA-2 - Línea 2: Hipotesis O-N10
  El organismo puede anticipar la posición futura de una fuente en movimiento,
  reduciendo retraso y costo energético.

Mecanismo:
  - Memoria episodica (V133) + relajacion conductual (V134)
  - Predictor de velocidad: estima velocidad angular por diferencia finita
  - Horizonte de prediccion: 2 segundos hacia el futuro
  - Setpoint = posicion_actual + velocidad * horizonte

Protocolo:
  Fase 1 (0-60s): Tracking normal (baseline)
  Fase 2 (60-120s): Tracking con prediccion activada
  Fase 3 (120-150s): Silencio + prediccion (mantiene anticipacion)
  Fase 4 (150-180s): Reenganche con prediccion
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

# Prediccion (NUEVO V135)
TAU_VELOCIDAD = 5.0      # segundos, constante de tiempo para suavizado
HORIZONTE_PREDICCION = 2.0  # segundos, cuánto anticipar
VELOCIDAD_MAX = 15.0     # grados/s, límite para evitar extrapolaciones locas

# Semilla base
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO
# ============================================================

class HemisferioV135:
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
# MEMORIA CON RELAJACION (de V134)
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
        """Setpoint modulado por confianza hacia centro"""
        if self.confianza > 0.01:
            return self.angulo * (self.confianza ** self.alpha) + self.centro * (1 - (self.confianza ** self.alpha))
        return self.centro
    
    def get_confianza(self):
        return self.confianza


# ============================================================
# PREDICTOR DE TRAYECTORIA (NUEVO V135)
# ============================================================

class PredictorTrayectoria:
    """
    Estima velocidad angular y predice posicion futura.
    
    Mecanismo:
      - Velocidad instantanea con suavizado exponencial
      - Horizonte de prediccion fijo
      - Limite de velocidad para estabilidad
    """
    
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
        """Actualiza estimacion de velocidad y devuelve posicion predicha"""
        dt = t - self.t_ultimo
        
        if dt > 0 and dt < 1.0:  # Ignorar saltos grandes
            # Velocidad instantanea
            vel_inst = (angulo_actual - self.angulo_ultimo) / dt
            
            # Suavizado exponencial (filtro paso bajo)
            alpha = min(1.0, dt / self.tau_velocidad)
            self.velocidad = (1 - alpha) * self.velocidad + alpha * vel_inst
            
            # Limitar velocidad
            self.velocidad = np.clip(self.velocidad, -self.vel_max, self.vel_max)
        
        self.angulo_ultimo = angulo_actual
        self.t_ultimo = t
        
        self.historial_velocidad.append(self.velocidad)
        
        # Prediccion
        return angulo_actual + self.velocidad * self.horizonte
    
    def get_velocidad(self):
        return self.velocidad


# ============================================================
# APARATO MOTOR CON MEMORIA Y PREDICCION (V135)
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
        
        # Memoria y prediccion
        self.memoria = MemoriaConRelajacion()
        self.predictor = PredictorTrayectoria()
        
        # Plasticidad
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
        
        # Actualizar memoria
        self.memoria.update(setpoint_percepcion, fuente_activa, t)
        
        # Determinar setpoint base
        if fuente_activa:
            setpoint_base = setpoint_percepcion
        else:
            setpoint_base = self.memoria.get_setpoint()
        
        # Aplicar prediccion si esta activa y hay fuente
        if modo_prediccion and fuente_activa:
            setpoint_predicho = self.predictor.update(setpoint_base, t)
            self.setpoint_usado = setpoint_predicho
            self.prediccion_activa = True
        else:
            # Sin prediccion, actualizar predictor igual para mantener velocidad
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
# SISTEMA V135
# ============================================================

class SistemaV135:
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
        
        self.izquierdo = HemisferioV135("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV135("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV135("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV135("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
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
        # Fuente activa = estamos en fase de test
        fuente_activa = True
        
        # Gradiente inter-sistemas
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Espacializacion
        sesgo = setpoint_real / 90.0
        gradiente += sesgo * 0.5
        
        # Motor con prediccion
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
# FUNCION DE MOVIMIENTO
# ============================================================

def movimiento_sinusoidal(t, amplitud=60.0, periodo=60.0):
    """Movimiento sinusoidal entre -60° y +60° con periodo 60s"""
    return amplitud * np.sin(2 * np.pi * t / periodo)


# ============================================================
# EXPERIMENTO V135
# ============================================================

def ejecutar_v135():
    print("=" * 100)
    print("EXPERIMENTO V135 — Prediccion de trayectoria")
    print("=" * 100)
    print("  ANIMA-2 - Linea 2: Hipotesis O-N10")
    print("  Fuente: movimiento sinusoidal -60° a +60°, periodo 60s")
    print("  Mecanismo: estimacion de velocidad + horizonte 2s")
    print("  Protocolo:")
    print("    Fase 1 (0-60s): Tracking normal (baseline)")
    print("    Fase 2 (60-120s): Tracking con prediccion activada")
    print("    Fase 3 (120-150s): Silencio + prediccion")
    print("    Fase 4 (150-180s): Reenganche con prediccion")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV135("V135", seed=SEMILLA_BASE)
    
    # Entrenamiento lateral
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS,
                              setpoint_real=0.0, modo_prediccion=False)
    
    print("  Entrenamiento completado.")
    
    # Fase de test con movimiento sinusoidal
    print("\n  Iniciando test de prediccion...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    duracion_test = 180.0
    
    resultados = {
        't': [],
        'orientacion': [],
        'setpoint_real': [],
        'prediccion_activa': []
    }
    
    for i in range(int(duracion_test / DT)):
        t = t_actual + i * DT
        t_rel = i * DT
        
        # Determinar modo de prediccion
        if t_rel < 60:  # Fase 1: baseline
            modo_prediccion = False
        elif t_rel < 120:  # Fase 2: con prediccion
            modo_prediccion = True
        elif t_rel < 150:  # Fase 3: silencio + prediccion (setpoint_real no importa)
            modo_prediccion = True
            # En silencio, el setpoint_real no se usa (la fuente no suena)
            # Pero mantenemos la funcion para el predictor
        else:  # Fase 4: reenganche con prediccion
            modo_prediccion = True
        
        # Setpoint real (posicion de la fuente)
        if t_rel < 120:
            setpoint_real = movimiento_sinusoidal(t_rel)
        elif t_rel < 150:
            setpoint_real = 0.0  # Silencio
        else:
            setpoint_real = movimiento_sinusoidal(t_rel - 150 + 120)  # Continuar desde donde quedo
        
        orientacion = sistema.actualizar(t, DT, t_actual + duracion_test,
                                         setpoint_real, modo_prediccion)
        
        resultados['t'].append(t_rel)
        resultados['orientacion'].append(orientacion)
        resultados['setpoint_real'].append(setpoint_real)
        resultados['prediccion_activa'].append(modo_prediccion)
        
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
            
            vel = sistema.historial['velocidad'][-1] if sistema.historial['velocidad'] else 0
            error = abs(orientacion - setpoint_real) if t_rel < 120 else 0
            print(f"    t={t_rel:4.0f}s | {fase:14s} | setpoint={setpoint_real:5.1f}° | "
                  f"orient={orientacion:5.1f}° | error={error:4.1f}° | vel={vel:5.2f}°/s")
    
    # ============================================================
    # ANALISIS DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALISIS DE PREDICCION")
    print("=" * 80)
    
    t_arr = np.array(resultados['t'])
    orient_arr = np.array(resultados['orientacion'])
    setpoint_arr = np.array(resultados['setpoint_real'])
    pred_activa = np.array(resultados['prediccion_activa'])
    
    # Separar fases
    mask_baseline = (t_arr >= 10) & (t_arr < 60)  # Ignorar transicion inicial
    mask_prediccion = (t_arr >= 60) & (t_arr < 120)
    
    # Calcular error por fase
    error_baseline = np.abs(orient_arr[mask_baseline] - setpoint_arr[mask_baseline])
    error_prediccion = np.abs(orient_arr[mask_prediccion] - setpoint_arr[mask_prediccion])
    
    # Calcular retraso por correlacion cruzada
    def calcular_retraso(orient, setpoint, dt=DT):
        # Recortar a misma longitud
        min_len = min(len(orient), len(setpoint))
        orient = orient[:min_len]
        setpoint = setpoint[:min_len]
        
        # Normalizar
        orient_norm = orient - np.mean(orient)
        setpoint_norm = setpoint - np.mean(setpoint)
        
        # Correlacion
        corr = np.correlate(orient_norm, setpoint_norm, mode='full')
        lag = np.argmax(corr) - (len(setpoint) - 1)
        return lag * dt
    
    retraso_baseline = calcular_retraso(orient_arr[mask_baseline], setpoint_arr[mask_baseline])
    retraso_prediccion = calcular_retraso(orient_arr[mask_prediccion], setpoint_arr[mask_prediccion])
    
    # Costo energetico
    diff_baseline = np.diff(orient_arr[mask_baseline])
    diff_prediccion = np.diff(orient_arr[mask_prediccion])
    
    E_baseline = np.sum(np.abs(diff_baseline))
    E_prediccion = np.sum(np.abs(diff_prediccion))
    
    # Velocidad estimada
    velocidad_arr = np.array(sistema.historial['velocidad'])
    vel_baseline = np.mean(np.abs(velocidad_arr[mask_baseline])) if len(velocidad_arr) > 0 else 0
    vel_prediccion = np.mean(np.abs(velocidad_arr[mask_prediccion])) if len(velocidad_arr) > 0 else 0
    
    print(f"\n  Fase 1 - Baseline (tracking normal):")
    print(f"    Error medio: {np.mean(error_baseline):.2f}°")
    print(f"    Error maximo: {np.max(error_baseline):.2f}°")
    print(f"    Retraso de fase: {retraso_baseline:.2f}s")
    print(f"    Costo energetico: {E_baseline:.1f}°")
    print(f"    Velocidad angular media: {vel_baseline:.2f}°/s")
    
    print(f"\n  Fase 2 - Prediccion activada:")
    print(f"    Error medio: {np.mean(error_prediccion):.2f}°")
    print(f"    Error maximo: {np.max(error_prediccion):.2f}°")
    print(f"    Retraso de fase: {retraso_prediccion:.2f}s")
    print(f"    Costo energetico: {E_prediccion:.1f}°")
    print(f"    Velocidad angular media: {vel_prediccion:.2f}°/s")
    
    # Mejoras
    mejora_error = (np.mean(error_baseline) - np.mean(error_prediccion)) / np.mean(error_baseline) * 100
    mejora_retraso = (retraso_baseline - retraso_prediccion) / retraso_baseline * 100 if retraso_baseline > 0 else 0
    mejora_energia = (E_baseline - E_prediccion) / E_baseline * 100 if E_baseline > 0 else 0
    
    print(f"\n  Mejoras con prediccion:")
    print(f"    Error: {mejora_error:.1f}% {'✅' if mejora_error > 30 else '⚠️'}")
    print(f"    Retraso: {mejora_retraso:.1f}% {'✅' if mejora_retraso > 30 else '⚠️'}")
    print(f"    Costo energetico: {mejora_energia:.1f}% {'✅' if mejora_energia > 20 else '⚠️'}")
    
    # Criterio de exito O-N10
    exito = mejora_error > 30 and mejora_retraso > 30
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Orientacion vs Setpoint
    ax = axes[0, 0]
    ax.plot(t_arr, setpoint_arr, 'r--', linewidth=1, alpha=0.7, label='Setpoint real (fuente)')
    ax.plot(t_arr, orient_arr, 'b-', linewidth=0.8, label='Orientacion real')
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7, label='Prediccion activada')
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5, label='Silencio')
    ax.axvline(x=150, color='purple', linestyle='--', alpha=0.5, label='Reenganche')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Angulo (grados)')
    ax.set_title('V135: Prediccion de trayectoria')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Error de orientacion
    ax = axes[0, 1]
    error_total = np.abs(orient_arr - setpoint_arr)
    ax.plot(t_arr, error_total, 'purple', linewidth=0.8)
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.axvline(x=120, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=ZONA_MUERTA_BASE, color='red', linestyle='--', alpha=0.5, label=f'Zona muerta ({ZONA_MUERTA_BASE}°)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Error de orientacion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Velocidad estimada
    ax = axes[1, 0]
    if len(velocidad_arr) > 0:
        # Velocidad teorica
        t_teorico = np.linspace(0, 120, 1000)
        vel_teorica = 60.0 * (2 * np.pi / 60.0) * np.cos(2 * np.pi * t_teorico / 60.0)
        ax.plot(t_teorico, vel_teorica, 'r--', linewidth=1, alpha=0.5, label='Velocidad teorica')
        
        # Velocidad estimada (muestrear)
        t_vel = np.array(sistema.historial['t']) - sistema.historial['t'][0]
        ax.plot(t_vel[:len(velocidad_arr)], velocidad_arr, 'orange', linewidth=0.8, label='Velocidad estimada')
    
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Velocidad (grados/s)')
    ax.set_title('Estimacion de velocidad')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Setpoint usado por el motor
    ax = axes[1, 1]
    setpoint_usado = np.array(sistema.historial['setpoint_usado'])
    t_hist = np.array(sistema.historial['t']) - sistema.historial['t'][0]
    ax.plot(t_hist[:len(setpoint_usado)], setpoint_usado, 'g-', linewidth=0.8, label='Setpoint usado (motor)')
    ax.plot(t_arr, setpoint_arr, 'r--', linewidth=1, alpha=0.5, label='Setpoint real')
    ax.axvline(x=60, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Setpoint (grados)')
    ax.set_title('Setpoint real vs Setpoint usado')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v135_logs', exist_ok=True)
    plt.savefig(f'v135_logs/v135_prediccion_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v135_logs/v135_prediccion_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION V135 — Prediccion de trayectoria")
    print("=" * 80)
    
    if exito:
        print("\n  ✅ O-N10 VALIDADA: Prediccion funcional")
        print(f"     Mejora error: {mejora_error:.1f}%")
        print(f"     Mejora retraso: {mejora_retraso:.1f}%")
        print(f"     Mejora energetica: {mejora_energia:.1f}%")
        print("\n  ANIMA-2 - Linea 2: CERRADA")
    else:
        print("\n  ⚠️ O-N10 NO VALIDADA: Mejoras insuficientes")
        print(f"     Error baseline: {np.mean(error_baseline):.2f}°")
        print(f"     Error prediccion: {np.mean(error_prediccion):.2f}°")
        print(f"     Retraso baseline: {retraso_baseline:.2f}s")
        print(f"     Retraso prediccion: {retraso_prediccion:.2f}s")
    
    return sistema, exito


if __name__ == "__main__":
    import time
    start = time.time()
    sistema, exito = ejecutar_v135()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")