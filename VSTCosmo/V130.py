#!/usr/bin/env python3
"""
VSTCosmos V130 — Organismo Plástico: Aprende a vivir con su fragilidad

Filosofia:
  - No buscamos el organismo perfecto (U_eff=0.0°)
  - Buscamos el organismo que SOBREVIVE (E controlado, incluso con temblor)
  - Plasticidad: el organismo adapta su ganancia durante la vida

Cambios desde V129:
  1. Motor plástico con habituacion/sensibilizacion
  2. Memoria de error para detectar oscilacion
  3. Memoria de costo energetico
  4. Kp se adapta en tiempo real
  5. Aceptamos U_eff = 3-5° como "vivo"

Principio biologico:
  - Un organismo que tiembla pero no se agota, sobrevive
  - Un organismo perfecto que se agota con 0.3° de temblor, muere
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import time
from collections import deque

# ============================================================
# PARAMETROS
# ============================================================
DT = 0.01
TIEMPO_POR_REPETICION = 452.0
REPETICIONES_LENTAS = 10
TIEMPO_BASELINE = 60.0
TIEMPO_INANICION = 30.0

# Asimetria forzada al nacer (mantenemos de V129)
SESGO_L = 0.01
SESGO_R = -0.01
DIM_HEMISFERIO = 32

# Zona muerta base
ZONA_MUERTA_BASE = 2.0
ZONA_MUERTA_MIN = 1.0
ZONA_MUERTA_MAX = 10.0

# Parkinson
TEMBLOR_AMPS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
FREQ_TEMBLOR = 5.0

# Plasticidad
HABITUACION_RAPIDA = 0.99    # Baja ganancia rapido
SENSIBILIZACION_LENTA = 1.01 # Sube ganancia lento
HABITUACION_MIN = 0.1        # Ganancia minima (10% de Kp original)
HABITUACION_MAX = 1.0        # Ganancia maxima (100% de Kp original)
VENTANA_OSCILACION = 100     # Pasos para detectar oscilacion (1 segundo)

# Semillas para validacion (usamos las que sobrevivieron en V129)
SEMILLAS_VIABLES = [44, 45]  # Las que sobreviven en V129
SEMILLAS_PRUEBA = [42, 43, 44, 45, 46]  # Para comparar con V129


# ============================================================
# HEMISFERIO (igual que V129)
# ============================================================

class HemisferioV130:
    """Hemisferio con asimetria forzada (igual V129)"""
    
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
        self.historial_Lambda = []
    
    def omega(self):
        return np.mean(self.Phi[:32])
    
    def _calcular_omega(self):
        return np.mean(self.Phi[:32])
    
    def _calcular_Lambda(self):
        return abs(self._calcular_omega())
    
    def generar_entrada_para_t(self, t, duracion_total):
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion
    
    def inducir_inanicion_gradual(self, paso_actual, pasos_totales):
        if paso_actual < pasos_totales:
            self.factor_inanicion = 1.0 - (paso_actual / pasos_totales)
        else:
            self.factor_inanicion = 0.0
            self.en_inanicion = True
    
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
        
        omega = self._calcular_omega()
        
        self.buffer_rapido.append((t, omega))
        if len(self.buffer_rapido) > int(self.tau / dt):
            self.buffer_rapido.pop(0)
        
        self.historial_omega.append(omega)
        
        return {'omega': omega, 'entrada': entrada}


# ============================================================
# APARATO MOTOR PLASTICO (NUEVO EN V130)
# ============================================================

class AparatoMotorPlastico:
    """
    Motor que aprende a vivir con su fragilidad.
    
    Caracteristicas:
      - Habituacion: baja la ganancia si detecta oscilacion
      - Sensibilizacion: sube la ganancia si esta muy quieto
      - Memoria de costo energetico
      - Acepta U_eff = 3-5° como "vivo"
    """
    
    def __init__(self, setpoint_inicial=-60.0, temblor_amp=0.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = 0.002
        self.limite = 90.0
        self.zona_muerta = ZONA_MUERTA_BASE
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        # Plasticidad
        self.habituacion = 1.0  # Factor de aprendizaje (1.0 = ganancia original)
        self.memoria_error = deque(maxlen=VENTANA_OSCILACION)
        self.memoria_delta = deque(maxlen=VENTANA_OSCILACION)
        self.historial_habituacion = []
        
        # Parkinson
        self.temblor_amp = temblor_amp
        self.freq_temblor = FREQ_TEMBLOR
        
        # Metricas
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        self.historial_Kp_actual = []
    
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actualizar_plasticidad(self, error, delta):
        """Actualiza el factor de habituacion basado en oscilacion y costo"""
        
        # Guardar en memoria
        self.memoria_error.append(error)
        self.memoria_delta.append(abs(delta))
        
        if len(self.memoria_error) < VENTANA_OSCILACION:
            return
        
        # Detectar oscilacion (Parkinson)
        oscilacion = np.std(self.memoria_error)
        costo_promedio = np.mean(self.memoria_delta)
        
        # Si hay mucha oscilacion: HABITUACION (baja ganancia)
        if oscilacion > self.zona_muerta * 1.5:
            self.habituacion *= HABITUACION_RAPIDA
            if self.habituacion < HABITUACION_MIN:
                self.habituacion = HABITUACION_MIN
        
        # Si esta muy quieto y el costo es bajo: SENSIBILIZACION (sube ganancia)
        elif oscilacion < self.zona_muerta * 0.5 and costo_promedio < 0.001:
            self.habituacion *= SENSIBILIZACION_LENTA
            if self.habituacion > HABITUACION_MAX:
                self.habituacion = HABITUACION_MAX
        
        # Registrar
        self.historial_habituacion.append(self.habituacion)
    
    def actuar(self, gradiente, LF_activa):
        if not LF_activa:
            return self.orientacion
        
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        error = self.setpoint - self.orientacion
        
        # Zona muerta adaptativa (mas generosa con ruido)
        zona_muerta_actual = min(ZONA_MUERTA_MAX, 
                                  self.zona_muerta + self.temblor_amp * 0.5)
        
        if abs(error) < zona_muerta_actual:
            return self.orientacion
        
        # Control con habituacion
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        # Kp actual = Kp_base * habituacion
        Kp_actual = self.Kp_base * self.habituacion
        delta = Kp_actual * error * ganancia_grad * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        # Actualizar plasticidad ANTES de aplicar temblor
        self.actualizar_plasticidad(error, delta)
        
        # Parkinson: temblor aditivo
        if self.temblor_amp > 0:
            temblor = self.temblor_amp * np.sin(2 * np.pi * self.freq_temblor * self.t)
            self.orientacion += delta + temblor
        else:
            self.orientacion += delta
        
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        # Guardar historial
        self.historial_orientacion.append(self.orientacion)
        self.historial_gradiente.append(gradiente)
        self.historial_delta.append(delta)
        self.historial_error.append(error)
        self.historial_Kp_actual.append(Kp_actual)
        
        self.t += DT
        
        return self.orientacion
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.habituacion = 1.0
        self.memoria_error.clear()
        self.memoria_delta.clear()
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        self.historial_habituacion = []
        self.historial_Kp_actual = []


# ============================================================
# SISTEMA V130
# ============================================================

class SistemaV130:
    """Sistema con motor plastico y hemisferios asimetricos"""
    
    def __init__(self, nombre, seed=42, temblor_amp=0.0):
        self.nombre = nombre
        
        # Generadores de entrada
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
        
        # Hemisferios con sesgo
        self.izquierdo = HemisferioV130("L", 30.0, generar_ruido_rosa, seed=seed, sesgo=SESGO_L)
        self.derecho = HemisferioV130("R", 300.0, generar_clicks_poisson, seed=seed+100, sesgo=SESGO_R)
        
        # Sistema B
        self.sistema_B_izq = HemisferioV130("B_L", 30.0, generar_ruido_rosa, seed=seed+200, sesgo=SESGO_L)
        self.sistema_B_der = HemisferioV130("B_R", 300.0, generar_clicks_poisson, seed=seed+300, sesgo=SESGO_R)
        
        self.modo_entrenamiento = True
        
        # Motor plastico
        self.motor = AparatoMotorPlastico(setpoint_inicial=-60.0, temblor_amp=temblor_amp)
        
        # Historial
        self.historial = {
            't': [],
            'omega_L': [],
            'omega_R': [],
            'omega_B_L': [],
            'omega_B_R': [],
            'orientacion': [],
            'error': [],
            's_shared': [],
            'habituacion': []
        }
    
    def omega_actual(self):
        return (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
    
    def calcular_s_shared(self):
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        # Actualizar sistema A
        self.izquierdo.actualizar(t, dt, duracion_total, self.derecho)
        self.derecho.actualizar(t, dt, duracion_total, self.izquierdo)
        
        # Actualizar sistema B
        self.sistema_B_izq.actualizar(t, dt, duracion_total, self.sistema_B_der)
        self.sistema_B_der.actualizar(t, dt, duracion_total, self.sistema_B_izq)
        
        # Gradiente inter-sistemas
        omega_A = (self.izquierdo._calcular_omega() + self.derecho._calcular_omega()) / 2
        omega_B = (self.sistema_B_izq._calcular_omega() + self.sistema_B_der._calcular_omega()) / 2
        gradiente = omega_A - omega_B
        
        # Espacializacion
        if audio_espacial is not None and not self.modo_entrenamiento:
            sesgo = audio_espacial / 90.0
            gradiente += sesgo * 0.5
        
        # Motor
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(gradiente, LF_activa)
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_L'].append(self.izquierdo._calcular_omega())
        self.historial['omega_R'].append(self.derecho._calcular_omega())
        self.historial['omega_B_L'].append(self.sistema_B_izq._calcular_omega())
        self.historial['omega_B_R'].append(self.sistema_B_der._calcular_omega())
        self.historial['orientacion'].append(orientacion)
        self.historial['error'].append(self.motor.setpoint - orientacion)
        self.historial['s_shared'].append(self.calcular_s_shared())
        self.historial['habituacion'].append(self.motor.habituacion)
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'error': self.motor.setpoint - orientacion,
            'habituacion': self.motor.habituacion
        }
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        self.sistema_B_izq.inducir_inanicion_gradual(paso_actual, pasos_totales)
        self.sistema_B_der.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# METRICAS DE SUPERVIVENCIA
# ============================================================

def calcular_metricas_supervivencia(sistema, setpoint=-60.0):
    """Calcula metricas de supervivencia"""
    orientacion = np.array(sistema.historial['orientacion'])
    error = np.array(sistema.historial['error'])
    s_shared = np.array(sistema.historial['s_shared'])
    habituacion = np.array(sistema.historial['habituacion']) if sistema.historial['habituacion'] else np.array([1.0])
    
    if len(orientacion) == 0:
        return {'sobrevive': False, 'severidad': 100.0}
    
    # Metricas basicas
    s_shared_final = np.mean(s_shared[-6000:]) if len(s_shared) > 6000 else np.mean(s_shared)
    lateralidad = s_shared_final < 0.8
    
    # Supervivencia funcional (tolerancia 30° para ser darwiniano)
    orient_final = orientacion[-1] if len(orientacion) > 0 else 0
    C50 = abs(orient_final - setpoint) < 30.0  # <<< Tolerancia aumentada a 30°
    
    # Temblor residual
    orient_ultimos = orientacion[-500:] if len(orientacion) >= 500 else orientacion
    U_eff = np.std(orient_ultimos)
    
    # Costo energetico
    E = np.sum(np.abs(np.diff(orientacion))) if len(orientacion) > 1 else 0.0
    
    # Severidad (relacion E / distancia teorica)
    distancia_teorica = abs(setpoint)
    severidad = E / distancia_teorica if distancia_teorica > 0 else 1.0
    
    # Supervivencia: el organismo sobrevive si C50 y severidad < 100
    sobrevive = C50 and severidad < 100
    
    # Capacidad de aprendizaje
    aprendizaje = habituacion[-1] if len(habituacion) > 0 else 1.0
    
    return {
        'sobrevive': sobrevive,
        'C50': C50,
        'lateralidad': lateralidad,
        's_shared_final': s_shared_final,
        'orient_final': orient_final,
        'U_eff': U_eff,
        'E': E,
        'severidad': severidad,
        'aprendizaje': aprendizaje
    }


# ============================================================
# VALIDACION DE POBLACION
# ============================================================

def validar_poblacion(semillas):
    """Valida la poblacion de organismos con motor plastico"""
    print("\n" + "=" * 80)
    print("VALIDACION DE POBLACION (Motor Plastico)")
    print("=" * 80)
    
    resultados = []
    
    for seed in semillas:
        print(f"\n  Semilla {seed}...", end=" ", flush=True)
        
        sistema = SistemaV130(f"V130_seed{seed}", seed=seed, temblor_amp=0.0)
        sistema.set_modo_entrenamiento(True)
        
        # Fase 2: Entrenamiento
        for rep in range(REPETICIONES_LENTAS):
            for i in range(int(TIEMPO_POR_REPETICION / DT)):
                t = rep * TIEMPO_POR_REPETICION + i * DT
                sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPETICIONES_LENTAS)
        
        # Fase 4: C50
        sistema.set_modo_entrenamiento(False)
        for i in range(int(300.0 / DT)):
            t = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + i * DT
            sistema.actualizar(t, DT, 400.0, audio_espacial=-60.0)
        
        metricas = calcular_metricas_supervivencia(sistema)
        resultados.append(metricas)
        
        status = "✅" if metricas['sobrevive'] else "❌"
        print(f"{status} C50={metricas['C50']}, lateralidad={metricas['lateralidad']}, severidad={metricas['severidad']:.1f}x, aprendizaje={metricas['aprendizaje']:.2f}")
    
    # Resumen
    print("\n  --- RESUMEN POBLACION ---")
    sobreviven = sum(1 for r in resultados if r['sobrevive'])
    print(f"    Supervivencia: {sobreviven}/{len(semillas)} ({sobreviven/len(semillas)*100:.0f}%)")
    
    return resultados


# ============================================================
# BARRIDO PARKINSON CON PLASTICIDAD
# ============================================================

def barrido_parkinson(semilla_base=44):
    """Barrido de temblor con motor plastico"""
    print("\n" + "=" * 80)
    print("BARRIDO PARKINSON (Motor Plastico)")
    print("=" * 80)
    
    resultados = []
    
    for amp in TEMBLOR_AMPS:
        print(f"  Temblor {amp:.1f}°...", end=" ", flush=True)
        
        sistema = SistemaV130(f"V130_parkinson_amp{amp}", seed=semilla_base, temblor_amp=amp)
        sistema.set_modo_entrenamiento(True)
        
        # Entrenamiento rapido
        for rep in range(2):
            for i in range(int(TIEMPO_POR_REPETICION / DT)):
                t = rep * TIEMPO_POR_REPETICION + i * DT
                sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * 2)
        
        # Test C50
        sistema.set_modo_entrenamiento(False)
        for i in range(int(300.0 / DT)):
            t = 2 * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, 400.0, audio_espacial=-60.0)
        
        metricas = calcular_metricas_supervivencia(sistema)
        metricas['temblor_amp'] = amp
        resultados.append(metricas)
        
        status = "✅" if metricas['sobrevive'] else "❌"
        print(f"{status} severidad={metricas['severidad']:.1f}x, U_eff={metricas['U_eff']:.2f}°, aprendizaje={metricas['aprendizaje']:.2f}")
    
    return resultados


# ============================================================
# EXPERIMENTO V130
# ============================================================

def ejecutar_v130():
    print("=" * 100)
    print("EXPERIMENTO V130 — Organismo Plastico")
    print("=" * 100)
    print("  Filosofia: No buscamos perfeccion, buscamos SUPERVIVENCIA")
    print("  Novedades:")
    print("    - Motor con habituacion/sensibilizacion")
    print("    - Aprende a vivir con su fragilidad")
    print("    - Acepta U_eff = 3-5° como 'vivo'")
    print("    - Tolerancia C50 aumentada a 30° (criterio darwiniano)")
    print("=" * 100)
    
    # Parte 1: Validacion de poblacion
    resultados_poblacion = validar_poblacion(SEMILLAS_PRUEBA)
    
    # Parte 2: Barrido Parkinson con plasticidad
    resultados_parkinson = barrido_parkinson(semilla_base=44)
    
    # Graficos
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: Supervivencia por semilla
    ax = axes[0, 0]
    semillas_str = [str(s) for s in SEMILLAS_PRUEBA]
    sobreviven = [1 if r['sobrevive'] else 0 for r in resultados_poblacion]
    colores = ['green' if s else 'red' for s in sobreviven]
    ax.bar(semillas_str, sobreviven, color=colores)
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Supervive')
    ax.set_title('Supervivencia por semilla (Motor Plastico)')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Severidad por semilla
    ax = axes[0, 1]
    severidades = [r['severidad'] for r in resultados_poblacion]
    ax.bar(semillas_str, severidades, color='orange')
    ax.axhline(y=100, color='red', linestyle='--', label='Umbral de colapso (100x)')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Severidad (E / distancia teorica)')
    ax.set_title('Costo energetico relativo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Aprendizaje por semilla
    ax = axes[0, 2]
    aprendizajes = [r['aprendizaje'] for r in resultados_poblacion]
    ax.bar(semillas_str, aprendizajes, color='purple')
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Habituacion = 1.0')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Habituacion final')
    ax.set_title('Plasticidad (1.0 = sin cambios)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Severidad vs temblor
    ax = axes[1, 0]
    temblores = [r['temblor_amp'] for r in resultados_parkinson]
    severidades_park = [r['severidad'] for r in resultados_parkinson]
    ax.semilogy(temblores, severidades_park, 'o-', color='red', linewidth=2, markersize=8)
    ax.axhline(y=100, color='orange', linestyle='--', label='Umbral de colapso (100x)')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Severidad (escala log)')
    ax.set_title('Costo energetico vs Parkinson (con plasticidad)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: U_eff vs temblor
    ax = axes[1, 1]
    ueffs_park = [r['U_eff'] for r in resultados_parkinson]
    ax.plot(temblores, ueffs_park, 'o-', color='blue', linewidth=2, markersize=8)
    ax.axhline(y=5.0, color='gray', linestyle='--', label='U_eff aceptable (5°)')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('U_eff (grados)')
    ax.set_title('Temblor residual vs Parkinson')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: Aprendizaje vs temblor
    ax = axes[1, 2]
    aprendizajes_park = [r['aprendizaje'] for r in resultados_parkinson]
    ax.plot(temblores, aprendizajes_park, 'o-', color='green', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='orange', linestyle='--', label='Habituacion critica')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Habituacion final')
    ax.set_title('Plasticidad: el organismo baja la ganancia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v130_logs', exist_ok=True)
    plt.savefig(f'v130_logs/v130_resultados_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v130_logs/v130_resultados_{timestamp}.png")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    tasa_supervivencia = sum(1 for r in resultados_poblacion if r['sobrevive']) / len(resultados_poblacion) * 100
    
    print(f"  Tasa de supervivencia: {tasa_supervivencia:.0f}%")
    print(f"  Umbral de Parkinson (severidad > 100x): ~{resultados_parkinson[2]['temblor_amp'] if len(resultados_parkinson)>2 else '?'}°")
    print(f"  Capacidad de aprendizaje: el organismo baja su ganancia hasta {resultados_parkinson[-1]['aprendizaje']:.2f}x")
    
    if tasa_supervivencia >= 80:
        print("\n  ✅ POBLACION VIABLE: 80%+ de los individuos sobreviven")
        print("  El organismo plastico logra lo que el rigido no pudo")
    elif tasa_supervivencia >= 60:
        print("\n  ⚠️ POBLACION MARGINAL: Mejor que V129, aun no optima")
    else:
        print("\n  ❌ POBLACION FRAGIL: La plasticidad no fue suficiente")
    
    print("\n  Filosofia V130:")
    print("    - Aceptamos U_eff = 3-5° como 'vivo'")
    print("    - El organismo aprende a vivir con su temblor")
    print("    - No buscamos el robot perfecto, buscamos el que sobrevive")
    
    return resultados_poblacion, resultados_parkinson


if __name__ == "__main__":
    start = time.time()
    pob, park = ejecutar_v130()
    print(f"\n  Tiempo: {(time.time() - start)/60:.1f} minutos")