#!/usr/bin/env python3
"""
VSTCosmos v125 — Dos sistemas acoplados con orientación de cabeza

Principios:
  - Respeta V122 como baseline inmutable
  - Añade módulo motor que usa s_shared (coherencia inter-sistemas)
  - Test C50: ¿puede orientar la cabeza hacia -60°?
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
    UMBRAL_R2_SIGMAS, UMBRAL_LATERALIDAD,
    TIEMPO_POR_REPETICION, REPETICIONES_LENTAS,
    TIEMPO_BASELINE, TIEMPO_INANICION,
    Hemisferio, SistemaV122,
    generar_ruido_rosa, generar_clicks_poisson
)

# ============================================================
# NUEVO: SISTEMA CON ORIENTACIÓN MOTORA
# ============================================================

class SistemaOrientador:
    """Extiende V122 añadiendo motor proporcional para orientación"""
    
    def __init__(self, nombre, seed=42):
        self.nombre = nombre
        # Dos sistemas completos como en V122
        self.sistema_A = SistemaV122(f"{nombre}_A", seed=seed)
        self.sistema_B = SistemaV122(f"{nombre}_B", seed=seed+100)
        
        # Parámetros del motor
        self.kp = 0.008
        self.setpoint = -60.0  # grados objetivo para C50
        self.orientacion = 0.0  # posición actual de la cabeza
        
        # Historial para este experimento
        self.historial = {
            't': [],
            'omega_A': [],
            'omega_B': [],
            'omega_total_A': [],
            'omega_total_B': [],
            's_shared': [],
            'orientacion': [],
            'error': []
        }
        
    def calcular_s_shared(self):
        """Coherencia inter-sistemas (misma fórmula que V122)"""
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def mapear_s_shared_a_angulo(self, s_shared):
        """Mapea s_shared ∈ [-1, 1] → ángulo ∈ [-180°, 180°]"""
        return s_shared * 180
    
    def actualizar_con_audio_espacial(self, t, dt, duracion_total, angulo_sonido=None):
        """
        Actualiza ambos sistemas con audio espacializado.
        
        Si angulo_sonido es None: usa las entradas ortogonales normales de V122
        Si angulo_sonido está definido: espacializa el sonido hacia L o R
        """
        if angulo_sonido is None:
            # Modo normal V122: cada sistema con sus entradas ortogonales
            # Esto ya ocurre dentro de sistema_A.actualizar() y sistema_B.actualizar()
            pass
        else:
            # Espacializar: inyectar señales diferenciales
            # Sistema A (izquierdo) recibe más si ángulo < 0
            # Sistema B (derecho) recibe más si ángulo > 0
            
            ganancia_A = max(0.0, 1.0 - abs(angulo_sonido) / 90.0) if angulo_sonido < 0 else 0.1
            ganancia_B = max(0.0, 1.0 - abs(angulo_sonido) / 90.0) if angulo_sonido > 0 else 0.1
            
            # Inyectar directamente en los hemisferios (acceso interno - PARA TESTEO)
            # NOTA: Esto es un parche para V125. En producción se modificaría la generación de entrada.
            if hasattr(self.sistema_A.izquierdo, 'factor_externo'):
                self.sistema_A.izquierdo.factor_externo = ganancia_A
                self.sistema_B.izquierdo.factor_externo = ganancia_B
        
        # Actualizar sistemas acoplados (igual que V122)
        self.sistema_A.actualizar(t, dt, duracion_total, self.sistema_B)
        self.sistema_B.actualizar(t, dt, duracion_total, self.sistema_A)
        
        # Calcular s_shared
        s_shared = self.calcular_s_shared()
        
        # Calcular error y actualizar motor
        angulo_actual = self.mapear_s_shared_a_angulo(s_shared)
        error = self.setpoint - angulo_actual
        self.orientacion += self.kp * error * dt
        self.orientacion = np.clip(self.orientacion, -180, 180)
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_A'].append(self.sistema_A.omega_actual())
        self.historial['omega_B'].append(self.sistema_B.omega_actual())
        self.historial['omega_total_A'].append(self.sistema_A.historial['omega_total'][-1] if self.sistema_A.historial['omega_total'] else 0)
        self.historial['omega_total_B'].append(self.sistema_B.historial['omega_total'][-1] if self.sistema_B.historial['omega_total'] else 0)
        self.historial['s_shared'].append(s_shared)
        self.historial['orientacion'].append(self.orientacion)
        self.historial['error'].append(error)
        
        return s_shared, self.orientacion, error
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        """Induce inanición en el sistema B (para test R₂)"""
        self.sistema_B.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def obtener_omega_sistema_A(self):
        return self.sistema_A.omega_actual()


# ============================================================
# EXPERIMENTO V125
# ============================================================

def ejecutar_v125():
    print("=" * 100)
    print("EXPERIMENTO V125 — Dos sistemas acoplados con orientación")
    print("=" * 100)
    
    duracion_total = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + TIEMPO_BASELINE + TIEMPO_INANICION + 500.0
    
    # ============================================================
    # FASE 1: Baseline — Un sistema (como V122)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline — Un sistema")
    print("=" * 80)
    
    orientador = SistemaOrientador("V125", seed=42)
    
    # Usar solo sistema_A para baseline (como V122)
    for i in range(int(TIEMPO_POR_REPETICION / DT)):
        t = i * DT
        # Actualizar sin acoplamiento inter-sistema
        orientador.sistema_A.actualizar(t, DT, duracion_total, None)
        if i % 10000 == 0:
            omega_L = orientador.sistema_A.historial['omega_L'][-1] if orientador.sistema_A.historial['omega_L'] else 0
            omega_R = orientador.sistema_A.historial['omega_R'][-1] if orientador.sistema_A.historial['omega_R'] else 0
            print(f"    t={t:.0f}s | Ω_L={omega_L:.4f}, Ω_R={omega_R:.4f}")
    
    print("  ✅ Fase 1 completada.")
    
    # ============================================================
    # FASE 2: Dos sistemas acoplados (ENTRENAMIENTO, igual V122)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Dos sistemas acoplados (entrenamiento lateral)")
    print("=" * 80)
    
    # Reinicializar para Fase 2
    orientador = SistemaOrientador("V125", seed=42)
    valores_s_shared = []
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"  Repetición {rep+1}/{REPETICIONES_LENTAS}...")
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            # Sin audio espacial (usar entradas ortogonales normales)
            s_shared, _, _ = orientador.actualizar_con_audio_espacial(t, DT, duracion_total, angulo_sonido=None)
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | s_shared={s_shared:.3f}")
            valores_s_shared.append(s_shared)
    
    s_shared_final = np.mean(valores_s_shared[-6000:]) if len(valores_s_shared) > 6000 else 1.0
    lateralidad = s_shared_final < UMBRAL_LATERALIDAD
    print(f"\n  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test R₂ (inanición) — igual V122
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Test R₂ (inanición)")
    print("=" * 80)
    
    # Nuevo orientador para test R₂
    orientador_r2 = SistemaOrientador("V125_R2", seed=42)
    
    # Baseline sin inanición
    for i in range(6000):
        t = i * DT
        orientador_r2.actualizar_con_audio_espacial(t, DT, duracion_total, angulo_sonido=None)
    
    omega_before = np.mean(orientador_r2.historial['omega_total_A'][-1000:])
    std_before = np.std(orientador_r2.historial['omega_total_A'][-1000:])
    print(f"  Baseline: Ω_A = {omega_before:.4f} ± {std_before:.4f}")
    
    # Inanición gradual en sistema B
    respuestas = []
    for i in range(3000):
        t = TIEMPO_BASELINE + i * DT
        orientador_r2.inducir_inanicion_sistema_B(i, 3000)
        s_shared, _, _ = orientador_r2.actualizar_con_audio_espacial(t, DT, duracion_total, angulo_sonido=None)
        omega_actual = orientador_r2.historial['omega_total_A'][-1]
        respuestas.append(abs(omega_actual - omega_before))
    
    respuesta_max = max(respuestas)
    umbral = UMBRAL_R2_SIGMAS * std_before
    R2 = respuesta_max > umbral
    print(f"  R₂: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # FASE 4: Test C50 — ORIENTACIÓN A -60°
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 4: Test C50 (orientación a -60°)")
    print("=" * 80)
    
    # Nuevo orientador para test C50
    orientador_c50 = SistemaOrientador("V125_C50", seed=42)
    
    # Entrenar primero (Fase 2) para tener lateralidad base
    print("  Pre-entrenando lateralidad...")
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            orientador_c50.actualizar_con_audio_espacial(t, DT, duracion_total, angulo_sonido=None)
    
    print("  Test C50: orientando hacia -60°...")
    
    # Test de orientación con audio espacial
    angulo_objetivo = -60.0
    orientador_c50.setpoint = angulo_objetivo
    
    for i in range(50000):  # 500 segundos a DT=0.01
        t = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + i * DT
        s_shared, orientacion, error = orientador_c50.actualizar_con_audio_espacial(
            t, DT, duracion_total, angulo_sonido=angulo_objetivo
        )
        
        if i % 5000 == 0:
            print(f"    t={t:.1f}s | orient={orientacion:.1f}°, s_shared={s_shared:.3f}")
    
    orientacion_final = orientador_c50.historial['orientacion'][-1]
    orientacion_estable = np.mean(orientador_c50.historial['orientacion'][-5000:])
    orientacion_std = np.std(orientador_c50.historial['orientacion'][-5000:])
    
    # Criterio: orientación final dentro de ±5° del objetivo
    C50 = abs(orientacion_final - angulo_objetivo) < 5.0
    
    print(f"\n  Orientación final: {orientacion_final:.1f}°")
    print(f"  Orientación estable (últimos 50s): {orientacion_estable:.1f}° ± {orientacion_std:.1f}°")
    print(f"  C50 ({angulo_objetivo:.0f}°): {'✅' if C50 else '❌'}")
    
    # ============================================================
    # GRÁFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Gráfico 1: Omega_L y Omega_R del sistema A en baseline
    ax = axes[0, 0]
    t1 = orientador.sistema_A.historial['t'][:int(TIEMPO_POR_REPETICION/DT)]
    ax.plot(t1, orientador.sistema_A.historial['omega_L'][:len(t1)], label='Ω_L (ruido_rosa)', alpha=0.7)
    ax.plot(t1, orientador.sistema_A.historial['omega_R'][:len(t1)], label='Ω_R (clicks)', alpha=0.7)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Ω')
    ax.set_title('Fase 1: Hemisferios con entradas ortogonales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: s_shared durante entrenamiento
    ax = axes[0, 1]
    t2 = np.arange(len(valores_s_shared)) * DT
    ax.plot(t2, valores_s_shared, color='purple', linewidth=0.5)
    ax.axhline(y=UMBRAL_LATERALIDAD, color='red', linestyle='--', alpha=0.5, label=f'Umbral lateralidad ({UMBRAL_LATERALIDAD})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Fase 2: Coherencia inter-sistemas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Respuesta a inanición
    ax = axes[0, 2]
    t_resp = TIEMPO_BASELINE + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red', linewidth=0.5)
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3σ={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|ΔΩ_A|')
    ax.set_title('Fase 3: Respuesta a inanición (R₂)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 4: s_shared y orientación durante C50
    ax = axes[1, 0]
    t_c50 = orientador_c50.historial['t'][-50000:]
    s_c50 = orientador_c50.historial['s_shared'][-50000:]
    ax.plot(t_c50, s_c50, color='purple', linewidth=0.5, label='s_shared')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=orientador_c50.mapear_s_shared_a_angulo(s_c50[-1])/180, color='blue', linestyle='--', alpha=0.5, label=f'Objetivo: {angulo_objetivo}°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Fase 4: s_shared durante orientación')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 5: Evolución de la orientación
    ax = axes[1, 1]
    orient_c50 = orientador_c50.historial['orientacion'][-50000:]
    ax.plot(t_c50, orient_c50, color='green', linewidth=1)
    ax.axhline(y=angulo_objetivo, color='red', linestyle='--', label=f'Objetivo: {angulo_objetivo}°')
    ax.axhline(y=orientacion_estable, color='blue', linestyle=':', alpha=0.7, label=f'Estable: {orientacion_estable:.1f}°')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientación (grados)')
    ax.set_title('Fase 4: Orientación de la cabeza')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 6: Error de orientación
    ax = axes[1, 2]
    error_c50 = orientador_c50.historial['error'][-50000:]
    ax.plot(t_c50, error_c50, color='orange', linewidth=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Fase 4: Error de orientación')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v125_logs', exist_ok=True)
    plt.savefig(f'v125_logs/v125_resultados_{timestamp}.png', dpi=150)
    print(f"\n  📊 Gráficos guardados: v125_logs/v125_resultados_{timestamp}.png")
    
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
        print("\n  ✅ ÉXITO COMPLETO: Lateralidad + R₂ + C50")
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
    
    print(f"\n  📊 Gráficos: v125_logs/v125_resultados_{timestamp}.png")
    
    return orientador_c50, exito


if __name__ == "__main__":
    orientador, exito = ejecutar_v125()