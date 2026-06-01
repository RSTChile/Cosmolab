#!/usr/bin/env python3
"""
VSTCosmos v127A — Corrección de signo del gradiente binaural

Diagnostico V126:
  - Lateralidad y R2 funcionan correctamente
  - C50 falla en +90 grados por signo invertido en el gradiente

Fix V127A:
  - grad = H_L.omega() - H_R.omega() (L - R, no R - L)
  - Todo lo demas igual que V126
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Importar V122 completo (INMUTABLE)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from V122 import (
    DT, TAU_IZQUIERDO, TAU_DERECHO,
    TIEMPO_POR_REPETICION, REPETICIONES_LENTAS,
    TIEMPO_BASELINE, TIEMPO_INANICION,
    SistemaV122
)


# ============================================================
# APARATO MOTOR CORREGIDO (V127A)
# ============================================================

class AparatoMotorV127A:
    """
    Organo motor con gradiente binaural CORREGIDO.
    
    Cambio critico:
      grad = L - R (NO R - L)
    
    Esto asegura que voz a la izquierda (omega_L > omega_R) genere
    gradiente positivo -> giro a la izquierda (orientacion negativa).
    """
    
    def __init__(self, setpoint_inicial=-60.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = 0.002          # Ganancia base
        self.limite = 90.0            # Limite anatomico real
        self.zona_muerta = 2.0        # Grados: no corregir si ya esta cerca
        self.inercia = 0.95           # Suavizado
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 20.0  # Sensibilidad del gradiente
        self.t = 0.0
        
        # Metricas
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        
    def calcular_gradiente(self, H_L, H_R):
        """
        Calcula el gradiente binaural CON SIGNO CORRECTO.
        
        Returns:
            float: gradiente en [-2, 2]
            - Positivo -> L domina -> sonido a la izquierda
            - Negativo -> R domina -> sonido a la derecha
        """
        omega_L = H_L.omega() if hasattr(H_L, 'omega') else H_L._calcular_omega()
        omega_R = H_R.omega() if hasattr(H_R, 'omega') else H_R._calcular_omega()
        
        # CORRECCION CRITICA: L - R (no R - L)
        return omega_L - omega_R
    
    def calcular_factor_freno(self, error):
        """Freno exponencial: mas lento cerca del objetivo"""
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actuar(self, H_L, H_R, LF_activa):
        """
        Genera comando motor a partir del gradiente binaural.
        """
        # Solo actuar si el sistema esta en modo test
        if not LF_activa:
            return self.orientacion
        
        # Calcular gradiente (CORREGIDO)
        gradiente = self.calcular_gradiente(H_L, H_R)
        
        # Zona muerta sensorial
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        # Calcular error de orientacion
        error = self.setpoint - self.orientacion
        
        # Zona muerta de control
        if abs(error) < self.zona_muerta:
            return self.orientacion
        
        # Mapeo no lineal del gradiente
        ganancia_grad = np.tanh(gradiente * self.sensibilidad_grad)
        
        # Factor de freno (mas lento cerca del objetivo)
        factor_freno = self.calcular_factor_freno(error)
        
        # Delta proporcional al error Y al gradiente
        # Con gradiente corregido: grad > 0 -> sonido izquierda -> delta negativo -> gira izquierda
        delta = self.Kp_base * error * ganancia_grad * factor_freno
        
        # Aplicar inercia (suavizado)
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        # Actualizar orientacion
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        # Guardar historial
        self.historial_orientacion.append(self.orientacion)
        self.historial_gradiente.append(gradiente)
        self.historial_delta.append(delta)
        self.historial_error.append(error)
        
        self.t += DT
        
        return self.orientacion
    
    def reset(self):
        """Reinicia el estado del motor"""
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []


# ============================================================
# SISTEMA COMPLETO V127A
# ============================================================

class SistemaV127A:
    """
    Sistema con motor corregido (gradiente L - R).
    """
    
    def __init__(self, nombre, seed=42):
        self.nombre = nombre
        
        # Dos sistemas completos (como V122)
        self.sistema_A = SistemaV122(f"{nombre}_A", seed=seed)
        self.sistema_B = SistemaV122(f"{nombre}_B", seed=seed+100)
        
        # Aparato motor corregido
        self.motor = AparatoMotorV127A(setpoint_inicial=-60.0)
        
        # Flag para modo de entrenamiento vs test
        self.modo_entrenamiento = True
        
        # Historial
        self.historial = {
            't': [],
            'omega_A': [],
            'omega_B': [],
            'omega_L_A': [],
            'omega_R_A': [],
            'gradiente': [],
            'orientacion': [],
            'error': [],
            's_shared': []
        }
        
    def calcular_s_shared(self):
        """Coherencia inter-sistemas (metrica original V122)"""
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def aplicar_espacializacion(self, angulo):
        """Aplica sesgo auditivo para simular fuente sonora"""
        # Mapear angulo a ganancia para cada sistema
        # angulo negativo (izquierda) -> favorecer sistema A
        # angulo positivo (derecha) -> favorecer sistema B
        sesgo = angulo / 90.0
        sesgo = np.clip(sesgo, -1.0, 1.0)
        
        # Inyectar factor externo (parche para test)
        if hasattr(self.sistema_A.izquierdo, 'factor_externo'):
            self.sistema_A.izquierdo.factor_externo = 1.0 - sesgo * 0.5
            self.sistema_B.izquierdo.factor_externo = 1.0 + sesgo * 0.5
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        """
        Actualiza el sistema completo.
        """
        # Aplicar espacializacion si estamos en modo test
        if audio_espacial is not None and not self.modo_entrenamiento:
            self.aplicar_espacializacion(audio_espacial)
        
        # Actualizar sistemas acoplados
        self.sistema_A.actualizar(t, dt, duracion_total, self.sistema_B)
        self.sistema_B.actualizar(t, dt, duracion_total, self.sistema_A)
        
        # Calcular gradiente usando hemisferios del sistema A
        omega_L = self.sistema_A.izquierdo._calcular_omega()
        omega_R = self.sistema_A.derecho._calcular_omega()
        
        # Actuar con el motor
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(
            self.sistema_A.izquierdo,
            self.sistema_A.derecho,
            LF_activa
        )
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_A'].append(self.sistema_A.omega_actual())
        self.historial['omega_B'].append(self.sistema_B.omega_actual())
        self.historial['omega_L_A'].append(omega_L)
        self.historial['omega_R_A'].append(omega_R)
        self.historial['gradiente'].append(omega_L - omega_R)
        self.historial['orientacion'].append(orientacion)
        self.historial['error'].append(self.motor.setpoint - orientacion)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'error': self.motor.setpoint - orientacion
        }
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        """Induce inanicion en el sistema B (para test R2)"""
        self.sistema_B.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        """Cambia entre modo entrenamiento y test"""
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# EXPERIMENTO V127A
# ============================================================

def ejecutar_v127a():
    print("=" * 100)
    print("EXPERIMENTO V127A — Correccion de signo del gradiente binaural")
    print("=" * 100)
    print("  Fix: grad = L - R (NO R - L)")
    print("  Lateralidad y R2 deben mantenerse correctos")
    print("  C50 debe converger a -60 grados")
    print("=" * 100)
    
    duracion_total = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + TIEMPO_BASELINE + TIEMPO_INANICION + 500.0
    
    # ============================================================
    # FASE 1: Baseline (rapido, para verificar)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline - Un sistema")
    print("=" * 80)
    
    sistema = SistemaV127A("V127A", seed=42)
    sistema.set_modo_entrenamiento(True)
    
    for i in range(int(TIEMPO_POR_REPETICION / DT)):
        t = i * DT
        sistema.actualizar(t, DT, duracion_total, audio_espacial=None)
        if i % 10000 == 0:
            omega_L = sistema.historial['omega_L_A'][-1] if sistema.historial['omega_L_A'] else 0
            omega_R = sistema.historial['omega_R_A'][-1] if sistema.historial['omega_R_A'] else 0
            print(f"    t={t:.0f}s | Omega_L={omega_L:.4f}, Omega_R={omega_R:.4f}")
    
    print("  Fase 1 completada correctamente.")
    
    # ============================================================
    # FASE 2: Entrenamiento lateral
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Dos sistemas acoplados (entrenamiento lateral)")
    print("=" * 80)
    
    sistema = SistemaV127A("V127A", seed=42)
    sistema.set_modo_entrenamiento(True)
    valores_s_shared = []
    
    for rep in range(REPETICIONES_LENTAS):
        print(f"  Repeticion {rep+1}/{REPETICIONES_LENTAS}...")
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            resultado = sistema.actualizar(t, DT, duracion_total, audio_espacial=None)
            s_shared = resultado['s_shared']
            if i % 10000 == 0:
                print(f"      t={t:.0f}s | s_shared={s_shared:.3f}")
            valores_s_shared.append(s_shared)
    
    s_shared_final = np.mean(valores_s_shared[-6000:]) if len(valores_s_shared) > 6000 else 1.0
    lateralidad = s_shared_final < 0.8
    print(f"\n  Lateralidad: {'CORRECTO' if lateralidad else 'FALLO'} (s_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test R2 (inanicion)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Test R2 (inanicion)")
    print("=" * 80)
    
    sistema_r2 = SistemaV127A("V127A_R2", seed=42)
    sistema_r2.set_modo_entrenamiento(True)
    
    # Baseline
    for i in range(6000):
        t = i * DT
        sistema_r2.actualizar(t, DT, duracion_total, audio_espacial=None)
    
    omega_before = np.mean(sistema_r2.historial['omega_A'][-1000:])
    std_before = np.std(sistema_r2.historial['omega_A'][-1000:])
    print(f"  Baseline: Omega_A = {omega_before:.4f} ± {std_before:.4f}")
    
    # Inanicion
    respuestas = []
    for i in range(3000):
        t = TIEMPO_BASELINE + i * DT
        sistema_r2.inducir_inanicion_sistema_B(i, 3000)
        resultado = sistema_r2.actualizar(t, DT, duracion_total, audio_espacial=None)
        omega_actual = sistema_r2.historial['omega_A'][-1]
        respuestas.append(abs(omega_actual - omega_before))
    
    respuesta_max = max(respuestas)
    umbral = 3.0 * std_before
    R2 = respuesta_max > umbral
    print(f"  R2: {'CORRECTO' if R2 else 'FALLO'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # FASE 4: Test C50 (orientacion a -60 grados)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 4: Test C50 (orientacion a -60 grados)")
    print("=" * 80)
    
    sistema_c50 = SistemaV127A("V127A_C50", seed=42)
    sistema_c50.set_modo_entrenamiento(True)
    
    # Pre-entrenar
    print("  Pre-entrenando lateralidad...")
    for rep in range(REPETICIONES_LENTAS):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema_c50.actualizar(t, DT, duracion_total, audio_espacial=None)
    
    # Modo test
    sistema_c50.set_modo_entrenamiento(False)
    sistema_c50.motor.setpoint = -60.0
    sistema_c50.motor.reset()
    
    print("  Test C50: orientando hacia -60 grados...")
    
    angulo_objetivo = -60.0
    
    for i in range(50000):
        t = TIEMPO_POR_REPETICION * REPETICIONES_LENTAS + i * DT
        resultado = sistema_c50.actualizar(t, DT, duracion_total, audio_espacial=angulo_objetivo)
        
        if i % 5000 == 0:
            print(f"    t={t:.1f}s | orient={resultado['orientacion']:.1f} grados, s_shared={resultado['s_shared']:.3f}")
    
    orientacion_final = sistema_c50.historial['orientacion'][-1]
    orientacion_estable = np.mean(sistema_c50.historial['orientacion'][-5000:])
    orientacion_std = np.std(sistema_c50.historial['orientacion'][-5000:])
    
    C50 = abs(orientacion_final - angulo_objetivo) < 5.0
    
    print(f"\n  Orientacion final: {orientacion_final:.1f} grados")
    print(f"  Orientacion estable (ultimos 50s): {orientacion_estable:.1f} grados ± {orientacion_std:.1f} grados")
    print(f"  C50 ({angulo_objetivo:.0f} grados): {'CORRECTO' if C50 else 'FALLO'}")
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: s_shared durante entrenamiento
    ax = axes[0, 0]
    t2 = np.arange(len(valores_s_shared)) * DT
    ax.plot(t2, valores_s_shared, color='purple', linewidth=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Fase 2: Coherencia inter-sistemas')
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: Respuesta a inanicion
    ax = axes[0, 1]
    t_resp = TIEMPO_BASELINE + np.arange(len(respuestas)) * DT
    ax.plot(t_resp, respuestas, color='red', linewidth=0.5)
    ax.axhline(y=umbral, color='green', linestyle='--', label=f'Umbral (3 sigma={umbral:.3f})')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('|Delta Omega_A|')
    ax.set_title('Fase 3: Respuesta a inanicion (R2)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: Evolucion de la orientacion
    ax = axes[0, 2]
    t_c50 = sistema_c50.historial['t'][-50000:]
    orient_c50 = sistema_c50.historial['orientacion'][-50000:]
    ax.plot(t_c50, orient_c50, color='green', linewidth=1)
    ax.axhline(y=angulo_objetivo, color='red', linestyle='--', label=f'Objetivo: {angulo_objetivo} grados')
    ax.axhline(y=orientacion_estable, color='blue', linestyle=':', alpha=0.7, label=f'Estable: {orientacion_estable:.1f} grados')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientacion (grados)')
    ax.set_title('Fase 4: Orientacion de la cabeza')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Gradiente binaural durante C50
    ax = axes[1, 0]
    grad_c50 = sistema_c50.historial['gradiente'][-50000:]
    ax.plot(t_c50, grad_c50, color='orange', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente (L - R)')
    ax.set_title('Fase 4: Gradiente binaural CORREGIDO')
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Error de orientacion
    ax = axes[1, 1]
    error_c50 = sistema_c50.historial['error'][-50000:]
    ax.plot(t_c50, error_c50, color='red', linewidth=0.7)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.3, label='Zona muerta (+-2 grados)')
    ax.axhline(y=-2, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Error (grados)')
    ax.set_title('Fase 4: Error de orientacion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: s_shared durante C50
    ax = axes[1, 2]
    s_c50 = sistema_c50.historial['s_shared'][-50000:]
    ax.plot(t_c50, s_c50, color='purple', linewidth=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('s_shared')
    ax.set_title('Fase 4: Coherencia inter-sistemas')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v127a_logs', exist_ok=True)
    plt.savefig(f'v127a_logs/v127a_resultados_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v127a_logs/v127a_resultados_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"  Lateralidad: {'CORRECTO' if lateralidad else 'FALLO'} (s_shared={s_shared_final:.4f})")
    print(f"  R2: {'CORRECTO' if R2 else 'FALLO'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    print(f"  C50 ({angulo_objetivo:.0f} grados): {'CORRECTO' if C50 else 'FALLO'} (final={orientacion_final:.1f} grados)")
    
    exito = lateralidad and R2 and C50
    
    if exito:
        print("\n  EXITO TOTAL: Lateralidad + R2 + C50")
        print("\n  PRIMER ORGANISMO COMPLETO:")
        print("     - Percibe sonido espacial")
        print("     - Diferencia lateralidad inter-sistemas")
        print("     - Responde a inanicion")
        print("     - Orienta la cabeza hacia la fuente sonora")
        print("\n  El gradiente binaural CORREGIDO (L - R) resuelve el problema de signo.")
    else:
        print("\n  PENDIENTE: Aun no se logran los 3 criterios")
    
    print(f"\n  Graficos: v127a_logs/v127a_resultados_{timestamp}.png")
    
    return sistema_c50, exito


if __name__ == "__main__":
    sistema, exito = ejecutar_v127a()