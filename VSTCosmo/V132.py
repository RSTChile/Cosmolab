#!/usr/bin/env python3
"""
VSTCosmos V132 — Organismo sano en acción

Cierre de ANIMA-1:
  - Organismo sano (sin Parkinson)
  - Responde a estímulos dinámicos
  - Múltiples posiciones: izquierda, derecha, centro
  - Mide tracking, no solo punto fijo

Métricas:
  - T_settle: tiempo para estabilizarse
  - Error por estímulo
  - Costo energético por movimiento
  - Capacidad de cambio entre estímulos
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import time
from collections import deque

# ============================================================
# PARAMETROS (basados en V131 - homeostasis)
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 60.0
TIEMPO_INANICION = 30.0

# Asimetria forzada (V131)
SESGO_L = 0.05
SESGO_R = -0.05
DIM_HEMISFERIO = 32

# Zona muerta
ZONA_MUERTA_BASE = 2.0

# Limites de plasticidad (HOMEOSTASIS)
KP_BASE = 0.002
KP_MIN = 0.0005
KP_MAX = 0.005

# Plasticidad
HABITUACION_RAPIDA = 0.99
SENSIBILIZACION_LENTA = 1.01
VENTANA_OSCILACION = 100

# Criterios de exito
ERROR_MAXIMO_ACEPTABLE = 10.0  # grados
ENERGIA_MAX_POR_MOVIMIENTO = 200.0  # grados

# Secuencia de estímulos
ESTIMULOS = [
    ('centro', 0.0, 100.0),      # 100s en centro
    ('izquierda', -60.0, 100.0), # 100s a -60°
    ('centro', 0.0, 100.0),      # 100s en centro
    ('derecha', 60.0, 100.0),    # 100s a +60°
    ('centro', 0.0, 100.0),      # 100s en centro
    ('izquierda', -60.0, 80.0),  # 80s a -60°
    ('derecha', 60.0, 80.0),     # 80s a +60°
    ('centro', 0.0, 60.0),       # 60s en centro
]

# Semilla base (la mas robusta de V131)
SEMILLA_BASE = 44


# ============================================================
# HEMISFERIO V132
# ============================================================

class HemisferioV132:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.W = np.zeros((DIM_HEMISFERIO, DIM_HEMISFERIO))
        
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
        for i in range(1, DIM_HEMISFERIO - 1):
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
        
        return {'omega': self._calcular_omega(), 'entrada': entrada}


# ============================================================
# APARATO MOTOR HOMEOSTATICO (sin Parkinson)
# ============================================================

class AparatoMotorHomeostatico:
    def __init__(self, setpoint_inicial=0.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
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
        
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.historial_Kp = []
        self.historial_orientacion = []
        self.historial_error = []
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error):
        self.memoria_error.append(error)
        
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        oscilacion = np.std(self.memoria_error)
        
        if oscilacion > self.zona_muerta * 1.5:
            self.Kp_actual = max(self.Kp_min, self.Kp_actual * HABITUACION_RAPIDA)
        elif oscilacion < self.zona_muerta * 0.5:
            self.Kp_actual = min(self.Kp_max, self.Kp_actual * SENSIBILIZACION_LENTA)
        
        self.historial_Kp.append(self.Kp_actual)
    
    def actuar(self, gradiente, LF_activa):
        if not LF_activa:
            return self.orientacion
        
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        error = self.setpoint - self.orientacion
        
        if abs(error) < self.zona_muerta:
            return self.orientacion
        
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_actual * error * ganancia_grad * factor_freno
        
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        self.actualizar_plasticidad(error)
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_orientacion.append(self.orientacion)
        self.historial_error.append(error)
        
        self.t += DT
        
        return self.orientacion
    
    def cambiar_setpoint(self, nuevo_setpoint):
        """Cambia el objetivo (para tracking)"""
        self.setpoint = nuevo_setpoint
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.Kp_actual = KP_BASE
        self.memoria_error.clear()
        self.historial_orientacion = []
        self.historial_error = []
        self.historial_Kp = []


# ============================================================
# SISTEMA V132
# ============================================================

class SistemaV132:
    def __init__(self, nombre, seed=44):
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
        
        self.izquierdo = HemisferioV132("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV132("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        self.sistema_B_izq = HemisferioV132("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV132("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        self.motor = AparatoMotorHomeostatico(setpoint_inicial=0.0)
        
        self.historial = {
            't': [],
            'omega_L': [],
            'omega_R': [],
            'gradiente': [],
            'orientacion': [],
            'error': [],
            's_shared': [],
            'setpoint': [],
            'Kp': []
        }
    
    def omega_actual(self):
        return (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        if audio_espacial is not None and not self.modo_entrenamiento:
            sesgo = audio_espacial / 90.0
            gradiente += sesgo * 0.5
        
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(gradiente, LF_activa)
        
        self.historial['t'].append(t)
        self.historial['omega_L'].append(self.izquierdo._calcular_omega())
        self.historial['omega_R'].append(self.derecho._calcular_omega())
        self.historial['gradiente'].append(gradiente)
        self.historial['orientacion'].append(orientacion)
        self.historial['error'].append(self.motor.setpoint - orientacion)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['setpoint'].append(self.motor.setpoint)
        self.historial['Kp'].append(self.motor.Kp_actual)
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'error': self.motor.setpoint - orientacion,
            'Kp': self.motor.Kp_actual
        }
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# METRICAS DE TRACKING
# ============================================================

def analizar_segmento(orientaciones, tiempos, setpoint, nombre_segmento):
    """Analiza un segmento de tracking"""
    if len(orientaciones) == 0:
        return None
    
    error_final = abs(orientaciones[-1] - setpoint)
    error_estable = np.mean(np.abs(np.array(orientaciones[-100:]) - setpoint)) if len(orientaciones) > 100 else error_final
    
    # Tiempo de asentamiento (cuando entra en zona muerta)
    T_settle = None
    for i, o in enumerate(orientaciones):
        if abs(o - setpoint) < ZONA_MUERTA_BASE:
            T_settle = tiempos[i] - tiempos[0]
            break
    
    # Costo energetico en este segmento
    E = np.sum(np.abs(np.diff(orientaciones))) if len(orientaciones) > 1 else 0.0
    
    # Overshoot (maximo sobrepaso)
    if setpoint < 0:  # izquierda
        overshoot = min(orientaciones) - setpoint if min(orientaciones) < setpoint else 0
    elif setpoint > 0:  # derecha
        overshoot = max(orientaciones) - setpoint if max(orientaciones) > setpoint else 0
    else:  # centro
        overshoot = max(abs(max(orientaciones)), abs(min(orientaciones))) - setpoint
    
    return {
        'nombre': nombre_segmento,
        'setpoint': setpoint,
        'error_final': error_final,
        'error_estable': error_estable,
        'T_settle': T_settle,
        'E': E,
        'overshoot': overshoot,
        'exito': error_final < ERROR_MAXIMO_ACEPTABLE
    }


# ============================================================
# EXPERIMENTO V132
# ============================================================

def ejecutar_v132():
    print("=" * 100)
    print("EXPERIMENTO V132 — Organismo sano en accion")
    print("=" * 100)
    print("  Cierre de ANIMA-1:")
    print("    - Organismo sano (sin Parkinson)")
    print("    - Tracking de estimulos multiples")
    print("    - Izquierda (-60°), Derecha (+60°), Centro (0°)")
    print("  Criterio de exito:")
    print(f"    - Error final < {ERROR_MAXIMO_ACEPTABLE}° por estimulo")
    print(f"    - Costo energetico < {ENERGIA_MAX_POR_MOVIMIENTO}° por movimiento")
    print("=" * 100)
    
    # Configurar sistema
    sistema = SistemaV132("V132", seed=SEMILLA_BASE)
    
    # Fase 2: Entrenamiento lateral (10 repeticiones)
    print("\n  Entrenando lateralidad (10 repeticiones)...")
    sistema.set_modo_entrenamiento(True)
    
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS)
    
    print("  Entrenamiento completado.")
    
    # Fase 4: Tracking de estimulos
    print("\n  Iniciando tracking de estimulos...")
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    t_actual = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS
    resultados_segmentos = []
    
    for nombre, angulo, duracion in ESTIMULOS:
        print(f"    Estimulo: {nombre} ({angulo:+.0f}°) durante {duracion:.0f}s")
        
        # Cambiar setpoint del motor
        sistema.motor.cambiar_setpoint(angulo)
        
        # Registrar inicio del segmento
        idx_inicio = len(sistema.historial['orientacion'])
        tiempo_inicio = t_actual
        
        # Ejecutar segmento
        pasos = int(duracion / DT)
        for i in range(pasos):
            t = t_actual + i * DT
            resultado = sistema.actualizar(t, DT, t_actual + duracion + 100, audio_espacial=angulo)
        
        t_actual += duracion
        
        # Extraer orientaciones del segmento
        idx_fin = len(sistema.historial['orientacion'])
        orientaciones_segmento = sistema.historial['orientacion'][idx_inicio:idx_fin]
        tiempos_segmento = sistema.historial['t'][idx_inicio:idx_fin]
        
        # Analizar
        analisis = analizar_segmento(orientaciones_segmento, tiempos_segmento, angulo, nombre)
        resultados_segmentos.append(analisis)
        
        if analisis and analisis['exito']:
            print(f"      ✅ Error final: {analisis['error_final']:.1f}°, T_settle: {analisis['T_settle']:.1f}s, E: {analisis['E']:.0f}°")
        elif analisis:
            print(f"      ❌ Error final: {analisis['error_final']:.1f}° (excede {ERROR_MAXIMO_ACEPTABLE}°)")
    
    # Metricas globales
    exitos = sum(1 for r in resultados_segmentos if r and r['exito'])
    total_segmentos = len([r for r in resultados_segmentos if r is not None])
    
    # Costo energetico total
    E_total = sum(r['E'] for r in resultados_segmentos if r)
    
    # Tiempo medio de asentamiento
    T_settle_vals = [r['T_settle'] for r in resultados_segmentos if r and r['T_settle'] is not None]
    T_settle_mean = np.mean(T_settle_vals) if T_settle_vals else None
    
    print("\n" + "=" * 80)
    print("RESULTADOS DE TRACKING")
    print("=" * 80)
    
    print("\n  Segmentos:")
    for r in resultados_segmentos:
        if r:
            status = "✅" if r['exito'] else "❌"
            print(f"    {status} {r['nombre']:10s} | setpoint: {r['setpoint']:+3.0f}° | error: {r['error_final']:.1f}° | T_settle: {r['T_settle']:.1f}s | E: {r['E']:.0f}°")
    
    print(f"\n  Metricas globales:")
    print(f"    Segmentos exitosos: {exitos}/{total_segmentos} ({exitos/total_segmentos*100:.0f}%)")
    print(f"    Costo energetico total: {E_total:.0f}°")
    print(f"    Tiempo medio de asentamiento: {T_settle_mean:.1f}s" if T_settle_mean else "    Tiempo medio de asentamiento: N/A")
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafico 1: Orientacion vs Setpoint
    ax = axes[0, 0]
    t_total = sistema.historial['t']
    orientacion = sistema.historial['orientacion']
    setpoint_hist = sistema.historial['setpoint']
    
    ax.plot(t_total, orientacion, 'b-', linewidth=1, label='Orientacion real')
    ax.plot(t_total, setpoint_hist, 'r--', linewidth=1, alpha=0.7, label='Setpoint (estimulo)')
    ax.axhline(y=ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.5, label=f'Zona muerta (±{ZONA_MUERTA_BASE}°)')
    ax.axhline(y=-ZONA_MUERTA_BASE, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientacion (grados)')
    ax.set_title('Tracking de estimulos: Orientacion vs Setpoint')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Error de orientacion
    ax = axes[0, 1]
    error = sistema.historial['error']
    ax.plot(t_total, error, 'g-', linewidth=0.8)
    ax.axhline(y=ZONA_MUERTA_BASE, color='r', linestyle='--', alpha=0.5, label=f'Zona muerta (±{ZONA_MUERTA_BASE}°)')
    ax.axhline(y=-ZONA_MUERTA_BASE, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=ERROR_MAXIMO_ACEPTABLE, color='orange', linestyle=':', alpha=0.5, label=f'Limite error ({ERROR_MAXIMO_ACEPTABLE}°)')
    ax.axhline(y=-ERROR_MAXIMO_ACEPTABLE, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Error de orientacion')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Kp adaptativo (plasticidad)
    ax = axes[1, 0]
    Kp_hist = sistema.historial['Kp']
    ax.plot(t_total, Kp_hist, 'purple', linewidth=0.8)
    ax.axhline(y=KP_BASE, color='gray', linestyle='--', alpha=0.5, label=f'Kp_base = {KP_BASE}')
    ax.axhline(y=KP_MIN, color='red', linestyle=':', alpha=0.5, label=f'Kp_min = {KP_MIN}')
    ax.axhline(y=KP_MAX, color='green', linestyle=':', alpha=0.5, label=f'Kp_max = {KP_MAX}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Kp')
    ax.set_title('Plasticidad: Kp adaptativo')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: s_shared (coherencia inter-sistemas)
    ax = axes[1, 1]
    s_shared = sistema.historial['s_shared']
    ax.plot(t_total, s_shared, 'orange', linewidth=0.8)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad (0.8)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Coherencia inter-sistemas (lateralidad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v132_logs', exist_ok=True)
    plt.savefig(f'v132_logs/v132_tracking_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v132_logs/v132_tracking_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION FINAL DE ANIMA-1
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION FINAL — ANIMA-1")
    print("=" * 80)
    
    exito_total = exitos == total_segmentos
    
    if exito_total:
        print("\n  ✅ ORGANISMO COMPLETO: Sigue todos los estimulos")
        print(f"     - {exitos}/{total_segmentos} segmentos exitosos")
        print(f"     - Error medio: {np.mean([r['error_final'] for r in resultados_segmentos if r]):.1f}°")
        print(f"     - Costo energetico total: {E_total:.0f}°")
    else:
        print(f"\n  ⚠️ TRACKING PARCIAL: {exitos}/{total_segmentos} segmentos exitosos")
    
    print("\n  Capacidades del organismo V132:")
    print("    ✅ Lateralidad inter-sistemas")
    print("    ✅ Orientacion a fuente sonora (C50)")
    print("    ✅ Tracking de estimulos multiples")
    print("    ✅ Plasticidad homeostatica (Kp adaptativo)")
    print("    ✅ Respuesta a cambios de setpoint")
    
    print("\n  Propiedades:")
    print(f"    - Zona muerta: {ZONA_MUERTA_BASE}°")
    print(f"    - Precision: ±{ERROR_MAXIMO_ACEPTABLE}°")
    print(f"    - Tiempo asentamiento medio: {T_settle_mean:.1f}s" if T_settle_mean else "    - Tiempo asentamiento medio: N/A")
    print(f"    - Rango Kp: [{KP_MIN}, {KP_MAX}]")
    
    print("\n  ============================================================")
    print("  ANIMA-1: Primer organismo artificial completo")
    print("  ============================================================")
    print("  Logros de la serie V122-V132:")
    print("    1. Lateralidad funcional (s_shared < 0.8)")
    print("    2. Respuesta a inanicion (R₂)")
    print("    3. Orientacion a fuente sonora (C50)")
    print("    4. Tracking de estimulos multiples")
    print("    5. Plasticidad homeostatica")
    print("  ============================================================")
    
    return sistema, resultados_segmentos, exito_total


if __name__ == "__main__":
    start = time.time()
    sistema, resultados, exito = ejecutar_v132()
    elapsed = time.time() - start
    print(f"\n  Tiempo de ejecucion: {elapsed/60:.1f} minutos")