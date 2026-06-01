#!/usr/bin/env python3
"""
VSTCosmos v127B — Correccion de gradiente: usar diferencia inter-sistemas

Diagnostico V127A:
  - Lateralidad y R2 funcionan correctamente
  - C50 sigue fallando a +90 grados
  - El problema: el motor usa gradiente INTRA-sistema (L-R del sistema A)
  - Deberia usar gradiente INTER-sistemas (sistema_A - sistema_B)

Fix V127B:
  - motor.actuar() recibe gradiente = omega_A - omega_B
  - Fases 1-3 reducidas para debugging rapido
  - Debug prints para ver valores reales
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
# CONFIGURACION RAPIDA PARA DEBUG
# ============================================================
MODO_DEBUG = True
if MODO_DEBUG:
    REPS_ENTRENAMIENTO = 2      # En lugar de 10
    DURACION_BASELINE = 10.0    # En lugar de 60
    DURACION_INANICION = 5.0    # En lugar de 30
else:
    REPS_ENTRENAMIENTO = REPETICIONES_LENTAS
    DURACION_BASELINE = TIEMPO_BASELINE
    DURACION_INANICION = TIEMPO_INANICION


# ============================================================
# APARATO MOTOR V127B (usa gradiente externo)
# ============================================================

class AparatoMotorV127B:
    """
    Organo motor que recibe el gradiente como parametro externo.
    
    El gradiente debe ser calculado por el sistema como:
        gradiente = omega_A - omega_B (diferencia inter-sistemas)
    """
    
    def __init__(self, setpoint_inicial=-60.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = 0.002          # Ganancia base
        self.limite = 90.0            # Limite anatomico real
        self.zona_muerta = 2.0        # Grados: no corregir si ya esta cerca
        self.inercia = 0.95           # Suavizado
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0  # Sensibilidad del gradiente
        self.t = 0.0
        
        # Metricas
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        
    def calcular_factor_freno(self, error):
        """Freno exponencial: mas lento cerca del objetivo"""
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actuar(self, gradiente, LF_activa):
        """
        Genera comando motor a partir del gradiente inter-sistemas.
        
        Args:
            gradiente: omega_A - omega_B (diferencia entre sistemas)
            LF_activa: si el sistema esta en modo test
        """
        # Solo actuar si el sistema esta en modo test
        if not LF_activa:
            return self.orientacion
        
        # Zona muerta sensorial
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        # Calcular error de orientacion
        error = self.setpoint - self.orientacion
        
        # Zona muerta de control
        if abs(error) < self.zona_muerta:
            return self.orientacion
        
        # Mapeo no lineal del gradiente
        # Si gradiente > 0 (A > B) -> sonido izquierda -> girar izquierda (negativo)
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        
        # Factor de freno (mas lento cerca del objetivo)
        factor_freno = self.calcular_factor_freno(error)
        
        # Delta proporcional al error Y al gradiente
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
# SISTEMA COMPLETO V127B
# ============================================================

class SistemaV127B:
    """
    Sistema con motor que usa gradiente INTER-sistemas.
    """
    
    def __init__(self, nombre, seed=42):
        self.nombre = nombre
        
        # Dos sistemas completos (como V122)
        self.sistema_A = SistemaV122(f"{nombre}_A", seed=seed)
        self.sistema_B = SistemaV122(f"{nombre}_B", seed=seed+100)
        
        # Aparato motor corregido
        self.motor = AparatoMotorV127B(setpoint_inicial=-60.0)
        
        # Flag para modo de entrenamiento vs test
        self.modo_entrenamiento = True
        
        # Historial
        self.historial = {
            't': [],
            'omega_A': [],
            'omega_B': [],
            'omega_L_A': [],
            'omega_R_A': [],
            'gradiente_inter': [],
            'orientacion': [],
            'error': [],
            's_shared': []
        }
        
        # Debug: guardar valores para inspeccion
        self.debug_values = []
        
    def calcular_s_shared(self):
        """Coherencia inter-sistemas (metrica original V122)"""
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def aplicar_espacializacion(self, angulo):
        """Aplica sesgo auditivo para simular fuente sonora"""
        sesgo = angulo / 90.0
        sesgo = np.clip(sesgo, -1.0, 1.0)
        
        # ANGULO NEGATIVO (izquierda) -> sesgo NEGATIVO
        # Esto da a sistema_A mas senal, a sistema_B menos
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
        
        # Calcular gradiente INTER-sistemas (CORRECTO para binaural)
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        gradiente_inter = omega_A - omega_B
        
        # Tambien guardamos valores intra para debug
        omega_L_A = self.sistema_A.izquierdo._calcular_omega()
        omega_R_A = self.sistema_A.derecho._calcular_omega()
        gradiente_intra = omega_L_A - omega_R_A
        
        # Actuar con el motor usando gradiente INTER-sistemas
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(gradiente_inter, LF_activa)
        
        # Debug: imprimir cada 5 segundos
        if MODO_DEBUG and int(t * 100) % 500 == 0 and not self.modo_entrenamiento:
            print(f"    DBG | t={t:.1f}s | omega_A={omega_A:.3f} omega_B={omega_B:.3f} "
                  f"grad_inter={gradiente_inter:.3f} grad_intra={gradiente_intra:.3f} "
                  f"orient={orientacion:.1f} error={self.motor.setpoint - orientacion:.1f}")
        
        # Guardar historial
        self.historial['t'].append(t)
        self.historial['omega_A'].append(omega_A)
        self.historial['omega_B'].append(omega_B)
        self.historial['omega_L_A'].append(omega_L_A)
        self.historial['omega_R_A'].append(omega_R_A)
        self.historial['gradiente_inter'].append(gradiente_inter)
        self.historial['orientacion'].append(orientacion)
        self.historial['error'].append(self.motor.setpoint - orientacion)
        self.historial['s_shared'].append(self.calcular_s_shared())
        
        # Guardar debug
        self.debug_values.append({
            't': t,
            'omega_A': omega_A,
            'omega_B': omega_B,
            'grad_inter': gradiente_inter,
            'orient': orientacion
        })
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'error': self.motor.setpoint - orientacion,
            'gradiente_inter': gradiente_inter,
            'gradiente_intra': gradiente_intra
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
# EXPERIMENTO V127B (CON FASES REDUCIDAS PARA DEBUG)
# ============================================================

def ejecutar_v127b():
    print("=" * 100)
    print("EXPERIMENTO V127B — Gradiente INTER-sistemas (corregido)")
    print("=" * 100)
    print("  Fix: motor usa gradiente = omega_A - omega_B")
    print("  Modo DEBUG: fases 1-3 reducidas para velocidad")
    print("  C50 debe converger a -60 grados")
    print("=" * 100)
    
    if MODO_DEBUG:
        print("\n  *** MODO DEBUG ACTIVO ***")
        print(f"  Repeticiones entrenamiento: {REPS_ENTRENAMIENTO} (normal: {REPETICIONES_LENTAS})")
        print(f"  Baseline: {DURACION_BASELINE}s (normal: {TIEMPO_BASELINE}s)")
        print(f"  Inanicion: {DURACION_INANICION}s (normal: {TIEMPO_INANICION}s)")
    
    # ============================================================
    # FASE 1: Baseline rapido
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 1: Baseline rapido - Un sistema")
    print("=" * 80)
    
    sistema = SistemaV127B("V127B", seed=42)
    sistema.set_modo_entrenamiento(True)
    
    pasos_baseline = int(DURACION_BASELINE / DT)
    for i in range(pasos_baseline):
        t = i * DT
        sistema.actualizar(t, DT, DURACION_BASELINE, audio_espacial=None)
    
    print(f"  Baseline completado ({DURACION_BASELINE}s)")
    
    # ============================================================
    # FASE 2: Entrenamiento lateral reducido
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 2: Entrenamiento lateral (reducido)")
    print("=" * 80)
    
    sistema = SistemaV127B("V127B", seed=42)
    sistema.set_modo_entrenamiento(True)
    valores_s_shared = []
    
    for rep in range(REPS_ENTRENAMIENTO):
        print(f"  Repeticion {rep+1}/{REPS_ENTRENAMIENTO}...")
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            resultado = sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO, audio_espacial=None)
            s_shared = resultado['s_shared']
            if i % 20000 == 0:  # Menos prints en debug
                print(f"      t={t:.0f}s | s_shared={s_shared:.3f}")
            valores_s_shared.append(s_shared)
    
    s_shared_final = np.mean(valores_s_shared[-6000:]) if len(valores_s_shared) > 6000 else 1.0
    lateralidad = s_shared_final < 0.8
    print(f"\n  Lateralidad: {'✅ CORRECTO' if lateralidad else '❌ FALLO'} (s_shared={s_shared_final:.4f})")
    
    # ============================================================
    # FASE 3: Test R2 reducido
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 3: Test R2 (inanicion) reducido")
    print("=" * 80)
    
    sistema_r2 = SistemaV127B("V127B_R2", seed=42)
    sistema_r2.set_modo_entrenamiento(True)
    
    # Baseline rapido
    pasos_baseline = int(DURACION_BASELINE / DT)
    for i in range(pasos_baseline):
        t = i * DT
        sistema_r2.actualizar(t, DT, DURACION_BASELINE, audio_espacial=None)
    
    omega_before = np.mean(sistema_r2.historial['omega_A'][-1000:]) if len(sistema_r2.historial['omega_A']) > 1000 else 0
    std_before = np.std(sistema_r2.historial['omega_A'][-1000:]) if len(sistema_r2.historial['omega_A']) > 1000 else 0.1
    print(f"  Baseline: Omega_A = {omega_before:.4f} ± {std_before:.4f}")
    
    # Inanicion rapida
    pasos_inanicion = int(DURACION_INANICION / DT)
    respuestas = []
    for i in range(pasos_inanicion):
        t = DURACION_BASELINE + i * DT
        sistema_r2.inducir_inanicion_sistema_B(i, pasos_inanicion)
        resultado = sistema_r2.actualizar(t, DT, DURACION_BASELINE + DURACION_INANICION, audio_espacial=None)
        omega_actual = sistema_r2.historial['omega_A'][-1]
        respuestas.append(abs(omega_actual - omega_before))
    
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = 3.0 * std_before if std_before > 0 else 0.1
    R2 = respuesta_max > umbral
    print(f"  R2: {'✅ CORRECTO' if R2 else '❌ FALLO'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    
    # ============================================================
    # FASE 4: Test C50 (COMPLETO)
    # ============================================================
    print("\n" + "=" * 80)
    print("FASE 4: Test C50 (orientacion a -60 grados) - COMPLETO")
    print("=" * 80)
    
    sistema_c50 = SistemaV127B("V127B_C50", seed=42)
    sistema_c50.set_modo_entrenamiento(True)
    
    # Pre-entrenar (con repeticiones reducidas)
    print(f"  Pre-entrenando lateralidad ({REPS_ENTRENAMIENTO} repeticiones)...")
    for rep in range(REPS_ENTRENAMIENTO):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema_c50.actualizar(t, DT, TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO, audio_espacial=None)
    
    # Modo test
    sistema_c50.set_modo_entrenamiento(False)
    sistema_c50.motor.setpoint = -60.0
    sistema_c50.motor.reset()
    
    print("  Test C50: orientando hacia -60 grados...")
    print("  (Debug prints cada 5 segundos muestran valores reales)")
    
    angulo_objetivo = -60.0
    
    for i in range(50000):  # 500 segundos completos
        t = TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO + i * DT
        resultado = sistema_c50.actualizar(t, DT, 
                                           TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO + 500.0,
                                           audio_espacial=angulo_objetivo)
        
        if i % 5000 == 0:
            print(f"    t={t:.1f}s | orient={resultado['orientacion']:.1f} grados, "
                  f"s_shared={resultado['s_shared']:.3f}, "
                  f"grad_inter={resultado['gradiente_inter']:.3f}")
    
    orientacion_final = sistema_c50.historial['orientacion'][-1]
    orientacion_estable = np.mean(sistema_c50.historial['orientacion'][-5000:])
    orientacion_std = np.std(sistema_c50.historial['orientacion'][-5000:])
    
    C50 = abs(orientacion_final - angulo_objetivo) < 5.0
    
    print(f"\n  Orientacion final: {orientacion_final:.1f} grados")
    print(f"  Orientacion estable (ultimos 50s): {orientacion_estable:.1f} grados ± {orientacion_std:.1f} grados")
    print(f"  C50 ({angulo_objetivo:.0f} grados): {'✅ CORRECTO' if C50 else '❌ FALLO'}")
    
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
    t_resp = DURACION_BASELINE + np.arange(len(respuestas)) * DT
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
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Orientacion (grados)')
    ax.set_title('Fase 4: Orientacion de la cabeza')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Gradiente inter-sistemas durante C50
    ax = axes[1, 0]
    grad_c50 = sistema_c50.historial['gradiente_inter'][-50000:]
    ax.plot(t_c50, grad_c50, color='orange', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Gradiente (omega_A - omega_B)')
    ax.set_title('Fase 4: Gradiente INTER-sistemas')
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
    
    # Grafico 6: omega_A y omega_B durante C50
    ax = axes[1, 2]
    omega_A_c50 = sistema_c50.historial['omega_A'][-50000:]
    omega_B_c50 = sistema_c50.historial['omega_B'][-50000:]
    ax.plot(t_c50, omega_A_c50, color='blue', linewidth=0.5, label='omega_A')
    ax.plot(t_c50, omega_B_c50, color='red', linewidth=0.5, label='omega_B')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Omega')
    ax.set_title('Fase 4: Actividad de sistemas A y B')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v127b_logs', exist_ok=True)
    plt.savefig(f'v127b_logs/v127b_resultados_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v127b_logs/v127b_resultados_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"  Lateralidad: {'✅' if lateralidad else '❌'} (s_shared={s_shared_final:.4f})")
    print(f"  R2: {'✅' if R2 else '❌'} (resp_max={respuesta_max:.4f} > umbral={umbral:.4f})")
    print(f"  C50 ({angulo_objetivo:.0f} grados): {'✅' if C50 else '❌'} (final={orientacion_final:.1f} grados)")
    
    exito = lateralidad and R2 and C50
    
    if exito:
        print("\n  🎉 EXITO TOTAL: Lateralidad + R2 + C50")
        print("\n  PRIMER ORGANISMO COMPLETO:")
        print("     - Percibe sonido espacial")
        print("     - Diferencia lateralidad inter-sistemas")
        print("     - Responde a inanicion")
        print("     - Orienta la cabeza hacia la fuente sonora")
    else:
        print("\n  ⚠️ PENDIENTE: Aun no se logran los 3 criterios")
        print("\n  Diagnostico por debug prints:")
        print("     Revisar si gradiente_inter tiene el signo correcto")
        print("     omega_A deberia ser > omega_B cuando el sonido viene de izquierda")
    
    print(f"\n  Graficos: v127b_logs/v127b_resultados_{timestamp}.png")
    
    return sistema_c50, exito


if __name__ == "__main__":
    sistema, exito = ejecutar_v127b()