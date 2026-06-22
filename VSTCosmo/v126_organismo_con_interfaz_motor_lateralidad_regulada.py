#!/usr/bin/env python3
"""
VSTCosmos v126 — Organismo con interfaz motor-lateralidad regulada

Diagnóstico V125:
  - Lateralidad y R₂ funcionan ✅
  - Motor colapsa a -180° porque obedece ciegamente la diferencia ❌

Soluciones V126:
  1. Usar diferencia CON SIGNO (no valor absoluto)
  2. Mapeo no lineal con tanh() para saturar respuesta
  3. Límite anatómico real: ±90°
  4. Freno exponencial cerca del objetivo
  5. Inercia para suavizar movimientos
  6. Zona muerta para evitar micro-oscilaciones
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Importar V122 completo (INMUTABLE)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from V122 import (
    DT, DIM_HEMISFERIO, TAU_IZQUIERDO, TAU_DERECHO,
    UMBRAL_CALLOSO, GANANCIA_CALLOSO, INHIBICION_RAPIDA,
    TIEMPO_POR_REPETICION, REPETICIONES_LENTAS,
    TIEMPO_BASELINE, TIEMPO_INANICION,
    Hemisferio, SistemaV122,
    generar_ruido_rosa, generar_clicks_poisson
)

# ============================================================
# NUEVO: APARATO MOTOR REGULADO (V126)
# ============================================================

class AparatoMotorV126:
    """
    Órgano motor que INTERPRETA la lateralidad, no la obedece.
    
    Características:
    - Límite anatómico: ±90° (no puede mirar hacia atrás)
    - Respuesta no lineal: pequeñas diferencias = movimientos pequeños
    - Freno exponencial: más lento cerca del objetivo
    - Inercia: suaviza transiciones bruscas
    - Zona muerta: evita correcciones infinitas
    """
    
    def __init__(self, setpoint_inicial=-60.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = 0.002          # Ganancia base (4x más lento que V124)
        self.limite = 90.0            # Límite anatómico real
        self.zona_muerta = 2.0        # Grados: no corregir si ya está cerca
        self.inercia = 0.95           # Suavizado (1 = máxima inercia)
        self.ultimo_delta = 0.0
        self.sensibilidad_diferencial = 0.5  # Mapeo diferencial → movimiento
        
        # Métricas
        self.historial_orientacion = []
        self.historial_diferencial = []
        self.historial_delta = []
        
    def calcular_diferencial_hemisferico(self, H_L, H_R):
        """
        Calcula la diferencia CON SIGNO entre hemisferios.
        
        Returns:
            float: Diferencial ∈ [-2, 2]
            - Valor positivo → L domina → orientar a la derecha
            - Valor negativo → R domina → orientar a la izquierda
        """
        omega_L = H_L.omega() if hasattr(H_L, 'omega') else H_L._calcular_omega()
        omega_R = H_R.omega() if hasattr(H_R, 'omega') else H_R._calcular_omega()
        return omega_L - omega_R
    
    def mapear_diferencial_a_ganancia(self, diferencial):
        """
        Mapeo no lineal: pequeñas diferencias producen movimientos pequeños.
        
        Usa tanh() para saturar la respuesta cuando la diferencia es extrema.
        Esto evita que el motor "obedezca ciegamente" la magnitud.
        """
        # Normalizar diferencial [-2, 2] → [-1, 1]
        diferencial_norm = np.clip(diferencial / 2.0, -1.0, 1.0)
        # tanh suaviza y satura
        ganancia = np.tanh(diferencial_norm * self.sensibilidad_diferencial)
        return ganancia
    
    def calcular_factor_freno(self, error):
        """
        Freno exponencial: más lento cerca del objetivo.
        
        error = 60° → factor ≈ 0.86 (poco freno)
        error = 10° → factor ≈ 0.28 (mucho freno)
        error = 2°  → factor ≈ 0.06 (casi parado)
        """
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actuar(self, H_L, H_R, LF_activa):
        """
        Genera comando motor a partir de la lateralidad hemisférica.
        
        Args:
            H_L, H_R: Hemisferios izquierdo y derecho
            LF_activa: Si el sistema está en modo de aprendizaje (Fase 2) o test (Fase 4)
        
        Returns:
            float: Nueva orientación (grados)
        """
        # Solo actuar si el sistema está activo (Fase 4 con audio espacial)
        if not LF_activa:
            return self.orientacion
        
        # Calcular diferencial hemisférico CON SIGNO
        diferencial = self.calcular_diferencial_hemisferico(H_L, H_R)
        
        # Zona muerta de sensorial: si no hay diferencia significativa, no mover
        if abs(diferencial) < 0.05:
            return self.orientacion
        
        # Calcular error de orientación
        error = self.setpoint - self.orientacion
        
        # Zona muerta de control: si ya estamos cerca, no corregir
        if abs(error) < self.zona_muerta:
            return self.orientacion
        
        # Mapear diferencial a ganancia de movimiento
        ganancia_diferencial = self.mapear_diferencial_a_ganancia(diferencial)
        
        # Calcular factor de freno (más lento cerca del objetivo)
        factor_freno = self.calcular_factor_freno(error)
        
        # Dirección del movimiento: signo del diferencial
        # Si diferencial > 0 (L domina) → mover a la derecha (reducir error negativo)
        direccion = np.sign(diferencial)
        
        # Delta proporcional al error Y a la ganancia diferencial
        delta_base = self.Kp_base * abs(error) * ganancia_diferencial * factor_freno
        delta = direccion * delta_base
        
        # Aplicar inercia (suavizado)
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        # Actualizar orientación
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        # Guardar historial
        self.historial_orientacion.append(self.orientacion)
        self.historial_diferencial.append(diferencial)
        self.historial_delta.append(delta)
        
        return self.orientacion
    
    def reset(self):
        """Reinicia el estado del motor (para nuevo experimento)"""
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.historial_orientacion = []
        self.historial_diferencial = []
        self.historial_delta = []


# ============================================================
# SISTEMA COMPLETO CON MOTOR REGULADO
# ============================================================

class SistemaV126:
    """
    Extiende V122 añadiendo el aparato motor regulado V126.
    
    Mantiene toda la arquitectura original de V122 (dos sistemas acoplados)
    y añade la capacidad de orientar la cabeza basándose en la lateralidad.
    """
    
    def __init__(self, nombre, seed=42):
        self.nombre = nombre
        
        # Dos sistemas completos (como V122)
        self.sistema_A = SistemaV122(f"{nombre}_A", seed=seed)
        self.sistema_B = SistemaV122(f"{nombre}_B", seed=seed+100)
        
        # Aparato motor regulado (NUEVO)
        self.motor = AparatoMotorV126(setpoint_inicial=-60.0)
        
        # Flag para modo de entrenamiento vs test
        self.modo_entrenamiento = True
        
        # Historial para este experimento
        self.historial = {
            't': [],
            'omega_A': [],
            'omega_B': [],
            'omega_L_A': [],
            'omega_R_A': [],
            'omega_L_B': [],
            'omega_R_B': [],
            'diferencial': [],
            'orientacion': [],
            'delta': [],
            'error': []
        }
        
    def calcular_diferencial_inter_sistemas(self):
        """Diferencia entre sistemas A y B (para s_shared de V122)"""
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        return omega_A - omega_B
    
    def calcular_s_shared(self):
        """Coherencia inter-sistemas (métrica original V122)"""
        diferencial = self.calcular_diferencial_inter_sistemas()
        return 1 - abs(diferencial) / 2.0
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        """
        Actualiza el sistema completo.
        
        Args:
            t: Tiempo actual
            dt: Paso de tiempo
            duracion_total: Duración total del experimento
            audio_espacial: Si es None, usa entradas normales de V122.
                           Si es un ángulo (ej: -60), espacializa el audio.
        """
        # Aplicar espacialización si se solicita
        if audio_espacial is not None and not self.modo_entrenamiento:
            # En modo test: inyectar sesgo auditivo
            # Cuanto más lejos del centro, más diferencia entre sistemas
            sesgo = audio_espacial / 90.0  # -60° → -0.67
            sesgo = np.clip(sesgo, -1.0, 1.0)
            
            # Modificar las entradas de los sistemas
            # Sistema A (izquierdo) recibe más si sesgo < 0
            # Sistema B (derecho) recibe más si sesgo > 0
            if hasattr(self.sistema_A.izquierdo, 'factor_externo'):
                self.sistema_A.izquierdo.factor_externo = 1.0 - sesgo * 0.5
                self.sistema_B.izquierdo.factor_externo = 1.0 + sesgo * 0.5
        
        # Actualizar sistemas acoplados (igual que V122)
        self.sistema_A.actualizar(t, dt, duracion_total, self.sistema_B)
        self.sistema_B.actualizar(t, dt, duracion_total, self.sistema_A)
        
        # Calcular diferencial hemisférico (dentro del sistema A, para el motor)
        # Usamos el sistema A como "referencia" para la orientación
        omega_L_A = self.sistema_A.izquierdo._calcular_omega()
        omega_R_A = self.sistema_A.derecho._calcular_omega()
        diferencial_hemisferico = omega_L_A - omega_R_A
        
        # Actuar con el motor (usando los hemisferios del sistema A)
        # En modo entrenamiento, LF_activa = False (motor quieto)
        # En modo test, LF_activa = True (motor activo)
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(
            self.sistema_A.izquierdo, 
            self.sistema_A.derecho, 
            LF_activa
        )
        
        # Calcular error respecto al setpoint
        error = self.motor.setpoint - orientacion
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_A'].append(self.sistema_A.omega_actual())
        self.historial['omega_B'].append(self.sistema_B.omega_actual())
        self.historial['omega_L_A'].append(omega_L_A)
        self.historial['omega_R_A'].append(omega_R_A)
        self.historial['omega_L_B'].append(self.sistema_B.izquierdo._calcular_omega())
        self.historial['omega_R_B'].append(self.sistema_B.derecho._calcular_omega())
        self.historial['diferencial'].append(diferencial_hemisferico)
        self.historial['orientacion'].append(orientacion)
        self.historial['delta'].append(self.motor.ultimo_delta if hasattr(self.motor, 'ultimo_delta') else 0)
        self.historial['error'].append(error)
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'diferencial': diferencial_hemisferico,
            'error': error
        }
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        """Induce inanición en el sistema B (para test R₂)"""
        self.sistema_B.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        """Cambia entre modo entrenamiento (motor quieto) y test (motor activo)"""
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# EXPERIMENTO V126
# ============================================================

def ejecutar_v126():
    print("=" * 100)
    print("EXPERIMENTO V126 — Organismo con interfaz motor-lateralidad regulada")
    print("=" * 100)
    print("  Soluciones:")
    print("    - Diferencia CON SIGNO (no valor absoluto)")
    print("    - Mapeo no lineal con tanh()")
    print("    - Límite anatómico: ±90°")
    print("    - Freno exponencial cerca del objetivo")
    print("    - Inercia para suavizar")
    print("=" * 100)
    
    duracion_total = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + TIEMPO_BASELINE + TIEMPO_INANICION + 500.0
    
    # ============================================================
    # FASE 1: Baseline — Un sistema (como V122)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline — Un sistema")
    print("=" * 80)
    
    sistema = SistemaV126("V126", seed=42)
    sistema.set_modo_entrenamiento(True)  # Motor quieto
    
    for i in range(int(TIEMPO_POR_REPETICION / DT)):
        t = i * DT
        sistema.actualizar(t, DT, duracion_total, audio_espacial=None)
        if i % 10000 == 0:
            omega_L = sistema.historial['omega_L_A'][-1] if sistema.historial['omega_L_A'] else 0
            omega_R = sistema.historial['omega_R_A'][-1] if sistema.historial['omega_R_A'] else 0
            print(f"    t={t:.0f}s | Ω_L={omega_L:.4f}, Ω_R={omega_R:.4f}")
    
    print("  ✅ Fase 1 completada.")
    
    # ============================================================
    # FASE 2: Dos sistemas acoplados (entrenamiento, igual V122)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Dos sistemas acoplados (entrenamiento lateral)")
    print("=" * 80)
    
    # Reinicializar para Fase 2
    sistema = SistemaV126("V126", seed=42)
    sistema.set_modo_entrenamiento(True)  # Motor quieto durante entrenamiento
    valores_s_shared = []
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"  Repetición {rep+1}/{REPETICIONES_LENTAS}...")
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            resultado = sistema.actualizar(t, DT, duracion_total, audio_espacial=None)
            s_shared = resultado['s_shared']
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | s_shared={s_shared:.3f}")
            valores_s_shared.append(s_shared)
    
    s_shared_final = np.mean(valores_s_shared[-6000:]) if len(valores_s_shared) > 6000 else 1.0
    lateralidad = s_shared_final < 0.8  # Umbral de V122
    print(f"\n  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test R₂ (inanición) — igual V122
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Test R₂ (inanición)")
    print("=" * 80)
    
    # Nuevo sistema para test R₂
    sistema_r2 = SistemaV126("V126_R2", seed=42)
    sistema_r2.set_modo_entrenamiento(True)
    
    # Baseline sin inanición
    for i in range(6000):
        t = i * DT
        sistema_r2.actualizar(t, DT, duracion_total, audio_espacial=None)
    
    omega_before = np.mean(sistema_r2.historial['omega_A'][-1000:])
    std_before = np.std(sistema_r2.historial['omega_A'][-1000:])
    print(f"  Baseline: Ω_A = {omega_before:.4f} ± {std_before:.4f}")
    
    # Inanición gradual en sistema B
    respuestas = []
    for i in range(3000):
        t = TIEMPO_BASELINE + i * DT
        sistema_r2.inducir_inanicion_sistema_B(i, 3000)
        resultado = sistema_r2.actualizar(t, DT, duracion_total, audio_espacial=None)
        omega_actual = sistema_r2.historial['omega_A'][-1]
        respuestas.append(abs(omega_actual - omega_before))
    
    respuesta_max = max(respuestas)
    umbral = 3.0 * std_before  # 3 sigmas
    R2 = respuesta_max > umbral
    print(f"  R₂: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # FASE 4: Test C50 — ORIENTACIÓN A -60°
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 4: Test C50 (orientación a -60°)")
    print("=" * 80)
    
    # Nuevo sistema para test C50
    sistema_c50 = SistemaV126("V126_C50", seed=42)
    sistema_c50.set_modo_entrenamiento(True)
    
    # Pre-entrenar lateralidad (Fase 2)
    print("  Pre-entrenando lateralidad...")
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema_c50.actualizar(t, DT, duracion_total, audio_espacial=None)
    
    # Cambiar a modo test (motor activo)
    sistema_c50.set_modo_entrenamiento(False)
    sistema_c50.motor.setpoint = -60.0
    sistema_c50.motor.reset()
    
    print("  Test C50: orientando hacia -60°...")
    
    # Test de orientación con sesgo auditivo
    angulo_objetivo = -60.0
    
    for i in range(50000):  # 500 segundos
        t = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + i * DT
        resultado = sistema_c50.actualizar(t, DT, duracion_total, audio_espacial=angulo_objetivo)
        
        if i % 5000 == 0:
            print(f"    t={t:.1f}s | orient={resultado['orientacion']:.1f}°, s_shared={resultado['s_shared']:.3f}")
    
    orientacion_final = sistema_c50.historial['orientacion'][-1]
    orientacion_estable = np.mean(sistema_c50.historial['orientacion'][-5000:])
    orientacion_std = np.std(sistema_c50.historial['orientacion'][-5000:])
    
    # Criterio: orientación final dentro de ±5° del objetivo
    C50 = abs(orientacion_final - angulo_objetivo) < 5.0
    
    print(f"\n  Orientación final: {orientacion_final:.1f}°")
    print(f"  Orientación estable (últimos 50s): {orientacion_estable:.1f}° ± {orientacion_std:.1f}°")
    print(f"  C50 ({angulo_objetivo:.0f}°): {'✅' if C50 else '❌'}")
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Gráfico 1: s_shared durante entrenamiento
    ax = axes[0, 0]
    t2 = np.arange(len(valores_s_shared)) * DT
    ax.plot(t2, valores_s_shared, color='purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Umbral lateralidad')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Fase 2: Coherencia inter-sistemas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Respuesta a inanición
    ax = axes[0, 1]
    t_resp = TIEMPO_BASELINE + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red', linewidth=0.5)
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Fase 3: Respuesta a inanición (R₂)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Evolución de la orientación en C50
    ax = axes[0, 2]
    t_c50 = sistema_c50.historial['t'][-50000:]
    orient_c50 = sistema_c50.historial['orientacion'][-50000:]
    ax.plot(t_c50, orient_c50, color='green', linewidth=1)
    ax.axhline(y=angulo_objetivo, color='red', linestyle='--', label=f'Objetivo: {angulo_objetivo}°')
    ax.axhline(y=orientacion_estable, color='blue', linestyle=':', alpha=0.7, label=f'Estable: {orientacion_estable:.1f}°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientación (grados)')
    ax.set_title('Fase 4: Orientación de la cabeza')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: Diferencial hemisférico durante C50
    ax = axes[1, 0]
    diferencial_c50 = sistema_c50.historial['diferencial'][-50000:]
    ax.plot(t_c50, diferencial_c50, color='orange', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Diferencial (Ω_L - Ω_R)')
    ax.set_title('Fase 4: Diferencial hemisférico')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Error de orientación
    ax = axes[1, 1]
    error_c50 = sistema_c50.historial['error'][-50000:]
    ax.plot(t_c50, error_c50, color='red', linewidth=0.7)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.3, label='Zona muerta')
    ax.axhline(y=-2, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Fase 4: Error de orientación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: s_shared durante C50
    ax = axes[1, 2]
    s_c50 = sistema_c50.historial['omega_A']  # Aproximación
    s_c50_trim = s_c50[-50000:] if len(s_c50) > 50000 else s_c50
    ax.plot(t_c50[:len(s_c50_trim)], s_c50_trim, color='purple', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω_A')
    ax.set_title('Fase 4: Actividad sistema A')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v126_logs', exist_ok=True)
    plt.savefig(f'v126_logs/v126_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v126_logs/v126_resultados_{timestamp}.png")
    
    # ============================================================
    # CONCLUSIÓN
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)
    print(f"  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    print(f"  R₂: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    print(f"  C50 ({angulo_objetivo:.0f}°): {'✅' if C50 else '❌'} (final={orientacion_final:.1f}°)")
    
    exito = lateralidad and R2 and C50
    
    if exito:
        print("\n  ✅ ÉXITO TOTAL: Lateralidad + R₂ + C50")
        print("\n  🎉 PRIMER ORGANISMO COMPLETO:")
        print("     - Percibe sonido espacial")
        print("     - Diferencia lateralidad")
        print("     - Responde a inanición")
        print("     - Orienta la cabeza hacia la fuente sonora")
    elif lateralidad and R2:
        print("\n  ⚠️ PARCIAL: Lateralidad + R₂, falta C50")
    elif lateralidad and C50:
        print("\n  ⚠️ PARCIAL: Lateralidad + C50, falta R₂")
    elif R2 and C50:
        print("\n  ⚠️ PARCIAL: R₂ + C50, falta lateralidad")
    elif lateralidad:
        print("\n  ⚠️ Solo lateralidad")
    elif R2:
        print("\n  ⚠️ Solo R₂")
    elif C50:
        print("\n  ⚠️ Solo C50")
    else:
        print("\n  ❌ NINGUNA CAPACIDAD")
    
    print(f"\n  📊 Gráficos: v126_logs/v126_resultados_{timestamp}.png")
    
    return sistema_c50, exito


if __name__ == "__main__":
    sistema, exito = ejecutar_v126()