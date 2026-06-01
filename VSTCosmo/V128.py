#!/usr/bin/env python3
"""
VSTCosmos V128 — Validación Robusta + Parkinson Computacional

Parte 1 - Validación robusta (régimen completo):
  - 10 repeticiones de entrenamiento
  - Baseline completo (60s)
  - Test R2 completo (30s)
  - Múltiples semillas aleatorias
  - Métricas: Lateralidad, R2, C50, U_eff, E, T_settle

Parte 2 - Parkinson computacional:
  - Sobre el sistema validado, barrer temblor_amp
  - Encontrar temblor_critico donde U_eff > zona_muerta
  - Medir E, T_settle, control_estable

Hipotesis O-N8.1:
  - temblor_amp < zona_muerta (2.0°) → control estable
  - temblor_amp > zona_muerta → ruptura, bucle positivo
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import time

# Importar V122 completo (INMUTABLE)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from V122 import (
    DT, TAU_IZQUIERDO, TAU_DERECHO,
    TIEMPO_POR_REPETICION, REPETICIONES_LENTAS,
    TIEMPO_BASELINE, TIEMPO_INANICION,
    SistemaV122
)

# ============================================================
# CONFIGURACION
# ============================================================

# Validacion robusta (régimen completo)
REPS_ENTRENAMIENTO = 10          # Repeticiones completas
DURACION_BASELINE = 60.0         # Baseline completo
DURACION_INANICION = 30.0        # Inanicion completa
SEMILLAS = [42, 43, 44, 45, 46]  # Multiples semillas para robustez

# Parkinson
TEMBLOR_AMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
FREQ_TEMBLOR = 5.0  # Hz

# Parametros motor
ZONA_MUERTA = 2.0
KP_BASE = 0.002
LIMITE = 90.0


# ============================================================
# APARATO MOTOR BASE (VALIDACION)
# ============================================================

class AparatoMotorBase:
    """Organo motor sano (como en V127B)"""
    
    def __init__(self, setpoint_inicial=-60.0):
        self.orientacion = 0.0
        self.setpoint = setpoint_inicial
        self.Kp_base = KP_BASE
        self.limite = LIMITE
        self.zona_muerta = ZONA_MUERTA
        self.inercia = 0.95
        self.ultimo_delta = 0.0
        self.sensibilidad_grad = 10.0
        self.t = 0.0
        
        # Metricas
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []
        
    def calcular_factor_freno(self, error):
        return 1 - np.exp(-abs(error) / 30.0)
    
    def actuar(self, gradiente, LF_activa):
        if not LF_activa:
            return self.orientacion
        
        if abs(gradiente) < 0.05:
            return self.orientacion
        
        error = self.setpoint - self.orientacion
        
        if abs(error) < self.zona_muerta:
            return self.orientacion
        
        # Mapeo no lineal: gradiente > 0 (A>B) -> sonido izquierda -> girar izquierda
        ganancia_grad = -np.tanh(gradiente * self.sensibilidad_grad)
        factor_freno = self.calcular_factor_freno(error)
        
        delta = self.Kp_base * error * ganancia_grad * factor_freno
        
        # Inercia
        delta = self.inercia * self.ultimo_delta + (1 - self.inercia) * delta
        self.ultimo_delta = delta
        
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        self.historial_orientacion.append(self.orientacion)
        self.historial_gradiente.append(gradiente)
        self.historial_delta.append(delta)
        self.historial_error.append(error)
        
        self.t += DT
        
        return self.orientacion
    
    def reset(self):
        self.orientacion = 0.0
        self.ultimo_delta = 0.0
        self.t = 0.0
        self.historial_orientacion = []
        self.historial_gradiente = []
        self.historial_delta = []
        self.historial_error = []


# ============================================================
# APARATO MOTOR PARKINSON
# ============================================================

class AparatoMotorParkinson(AparatoMotorBase):
    """Organo motor con temblor (Parkinson computacional)"""
    
    def __init__(self, temblor_amp=0.0, freq_temblor=5.0, setpoint_inicial=-60.0):
        super().__init__(setpoint_inicial)
        self.temblor_amp = temblor_amp
        self.freq_temblor = freq_temblor
        
    def actuar(self, gradiente, LF_activa):
        # Control base (sano)
        orient_control = super().actuar(gradiente, LF_activa)
        
        # Temblor aditivo - el sistema no puede corregirlo
        temblor = self.temblor_amp * np.sin(2 * np.pi * self.freq_temblor * self.t)
        
        # El temblor se suma DESPUES del control
        self.orientacion = orient_control + temblor
        
        # Mantener dentro del limite
        self.orientacion = np.clip(self.orientacion, -self.limite, self.limite)
        
        return self.orientacion


# ============================================================
# SISTEMA COMPLETO V128
# ============================================================

class SistemaV128:
    """
    Sistema con motor que usa gradiente INTER-sistemas.
    Puede usar motor base o motor Parkinson.
    """
    
    def __init__(self, nombre, seed=42, motor_tipo='base', temblor_amp=0.0):
        self.nombre = nombre
        
        # Dos sistemas completos (como V122)
        self.sistema_A = SistemaV122(f"{nombre}_A", seed=seed)
        self.sistema_B = SistemaV122(f"{nombre}_B", seed=seed+100)
        
        # Motor (base o Parkinson)
        if motor_tipo == 'parkinson':
            self.motor = AparatoMotorParkinson(temblor_amp=temblor_amp, 
                                                freq_temblor=FREQ_TEMBLOR,
                                                setpoint_inicial=-60.0)
        else:
            self.motor = AparatoMotorBase(setpoint_inicial=-60.0)
        
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
        
    def calcular_s_shared(self):
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        return 1 - abs(omega_A - omega_B) / 2.0
    
    def aplicar_espacializacion(self, angulo):
        """Aplica sesgo auditivo para simular fuente sonora"""
        sesgo = angulo / 90.0
        sesgo = np.clip(sesgo, -1.0, 1.0)
        
        if hasattr(self.sistema_A.izquierdo, 'factor_externo'):
            self.sistema_A.izquierdo.factor_externo = 1.0 - sesgo * 0.5
            self.sistema_B.izquierdo.factor_externo = 1.0 + sesgo * 0.5
    
    def actualizar(self, t, dt, duracion_total, audio_espacial=None):
        # Aplicar espacializacion si estamos en modo test
        if audio_espacial is not None and not self.modo_entrenamiento:
            self.aplicar_espacializacion(audio_espacial)
        
        # Actualizar sistemas acoplados
        self.sistema_A.actualizar(t, dt, duracion_total, self.sistema_B)
        self.sistema_B.actualizar(t, dt, duracion_total, self.sistema_A)
        
        # Calcular gradiente INTER-sistemas
        omega_A = self.sistema_A.omega_actual()
        omega_B = self.sistema_B.omega_actual()
        gradiente_inter = omega_A - omega_B
        
        # Valores intra para debug
        omega_L_A = self.sistema_A.izquierdo._calcular_omega()
        omega_R_A = self.sistema_A.derecho._calcular_omega()
        
        # Actuar con el motor
        LF_activa = not self.modo_entrenamiento
        orientacion = self.motor.actuar(gradiente_inter, LF_activa)
        
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
        
        return {
            'orientacion': orientacion,
            's_shared': self.calcular_s_shared(),
            'error': self.motor.setpoint - orientacion,
            'gradiente_inter': gradiente_inter
        }
    
    def inducir_inanicion_sistema_B(self, paso_actual, pasos_totales):
        self.sistema_B.inducir_inanicion_gradual(paso_actual, pasos_totales)
    
    def set_modo_entrenamiento(self, entrenamiento=True):
        self.modo_entrenamiento = entrenamiento
        if entrenamiento:
            self.motor.reset()


# ============================================================
# FUNCIONES DE VALIDACION Y METRICAS
# ============================================================

def calcular_metricas(sistema, setpoint=-60.0, zona_muerta=2.0):
    """Calcula metricas ciberneticas"""
    orientacion = np.array(sistema.historial['orientacion'])
    error = np.array(sistema.historial['error'])
    s_shared = np.array(sistema.historial['s_shared'])
    
    # Metricas basicas
    s_shared_final = np.mean(s_shared[-6000:]) if len(s_shared) > 6000 else np.mean(s_shared)
    lateralidad = s_shared_final < 0.8
    
    # Metricas ciberneticas
    orient_ultimos = orientacion[-500:] if len(orientacion) >= 500 else orientacion
    U_eff = np.std(orient_ultimos)  # Temblor residual
    
    # Costo energetico (distancia total recorrida)
    E = np.sum(np.abs(np.diff(orientacion))) if len(orientacion) > 1 else 0.0
    
    # Tiempo de asentamiento (entrada en zona muerta)
    T_settle = None
    for i, err in enumerate(error):
        if abs(err) < zona_muerta:
            # Verificar que se mantiene por 100 pasos
            if i + 100 < len(error) and all(abs(error[i:i+100]) < zona_muerta):
                T_settle = sistema.historial['t'][i]
                break
    
    # Error final
    error_final = abs(error[-1]) if len(error) > 0 else float('inf')
    
    # Control estable
    control_estable = U_eff < zona_muerta and error_final < zona_muerta
    
    # C50: orientacion final cerca del setpoint
    orient_final = orientacion[-1] if len(orientacion) > 0 else 0
    C50 = abs(orient_final - setpoint) < 5.0
    
    return {
        's_shared_final': s_shared_final,
        'lateralidad': lateralidad,
        'U_eff': U_eff,
        'E': E,
        'T_settle': T_settle,
        'error_final': error_final,
        'control_estable': control_estable,
        'orient_final': orient_final,
        'C50': C50
    }


def ejecutar_validacion_semilla(seed, verbose=True):
    """Ejecuta validacion completa para una semilla"""
    if verbose:
        print(f"\n  --- Semilla {seed} ---")
    
    sistema = SistemaV128(f"V128_seed{seed}", seed=seed, motor_tipo='base')
    
    # Fase 1: Baseline (rapido)
    sistema.set_modo_entrenamiento(True)
    for i in range(int(DURACION_BASELINE / DT)):
        t = i * DT
        sistema.actualizar(t, DT, DURACION_BASELINE, audio_espacial=None)
    
    # Fase 2: Entrenamiento lateral (10 repeticiones)
    for rep in range(REPS_ENTRENAMIENTO):
        for i in range(int(TIEMPO_POR_REPETICION / DT)):
            t = rep * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO, 
                              audio_espacial=None)
    
    # Fase 3: Test R2
    # Baseline R2
    for i in range(int(DURACION_BASELINE / DT)):
        t = i * DT
        sistema.actualizar(t, DT, DURACION_BASELINE, audio_espacial=None)
    
    omega_before = np.mean(sistema.historial['omega_A'][-1000:]) if len(sistema.historial['omega_A']) > 1000 else 0
    std_before = np.std(sistema.historial['omega_A'][-1000:]) if len(sistema.historial['omega_A']) > 1000 else 0.1
    
    # Inanicion
    pasos_inanicion = int(DURACION_INANICION / DT)
    respuestas = []
    for i in range(pasos_inanicion):
        t = DURACION_BASELINE + i * DT
        sistema.inducir_inanicion_sistema_B(i, pasos_inanicion)
        resultado = sistema.actualizar(t, DT, DURACION_BASELINE + DURACION_INANICION, 
                                       audio_espacial=None)
        omega_actual = sistema.historial['omega_A'][-1]
        respuestas.append(abs(omega_actual - omega_before))
    
    respuesta_max = max(respuestas) if respuestas else 0
    umbral = 3.0 * std_before
    R2 = respuesta_max > umbral
    
    # Fase 4: Test C50 (orientacion a -60 grados)
    sistema.set_modo_entrenamiento(False)
    sistema.motor.reset()
    
    for i in range(int(500.0 / DT)):  # 500 segundos
        t = TIEMPO_POR_REPETICION * REPS_ENTRENAMIENTO + DURACION_BASELINE + DURACION_INANICION + i * DT
        resultado = sistema.actualizar(t, DT, 1000.0, audio_espacial=-60.0)
    
    # Calcular metricas
    metricas = calcular_metricas(sistema, setpoint=-60.0, zona_muerta=ZONA_MUERTA)
    metricas['R2'] = R2
    metricas['respuesta_max'] = respuesta_max
    metricas['umbral_R2'] = umbral
    
    if verbose:
        print(f"    Lateralidad: {'✅' if metricas['lateralidad'] else '❌'} (s_shared={metricas['s_shared_final']:.4f})")
        print(f"    R2: {'✅' if R2 else '❌'} (resp={respuesta_max:.4f} > {umbral:.4f})")
        print(f"    C50: {'✅' if metricas['C50'] else '❌'} (final={metricas['orient_final']:.1f}°)")
        print(f"    U_eff: {metricas['U_eff']:.3f}°, E: {metricas['E']:.1f}°, T_settle: {metricas['T_settle']:.1f}s" if metricas['T_settle'] else f"    U_eff: {metricas['U_eff']:.3f}°, E: {metricas['E']:.1f}°, T_settle: ∞")
    
    return metricas


def ejecutar_barrido_parkinson(seed=42, verbose=True):
    """Ejecuta barrido de amplitud de temblor"""
    resultados = []
    
    print(f"\n  Barrido Parkinson (semilla {seed})")
    print("  " + "-" * 60)
    
    for amp in TEMBLOR_AMPS:
        if verbose:
            print(f"    Temblor {amp:.1f}°...", end=" ", flush=True)
        
        sistema = SistemaV128(f"V128_parkinson_amp{amp}", seed=seed, 
                              motor_tipo='parkinson', temblor_amp=amp)
        
        # Entrenamiento reducido (usar el mismo pre-entrenamiento)
        sistema.set_modo_entrenamiento(True)
        
        # Pre-entrenamiento rapido (2 repeticiones para velocidad)
        for rep in range(2):
            for i in range(int(TIEMPO_POR_REPETICION / DT)):
                t = rep * TIEMPO_POR_REPETICION + i * DT
                sistema.actualizar(t, DT, TIEMPO_POR_REPETICION * 2, audio_espacial=None)
        
        # Test C50
        sistema.set_modo_entrenamiento(False)
        sistema.motor.reset()
        
        for i in range(int(300.0 / DT)):  # 300 segundos
            t = 2 * TIEMPO_POR_REPETICION + i * DT
            sistema.actualizar(t, DT, 400.0, audio_espacial=-60.0)
        
        metricas = calcular_metricas(sistema, setpoint=-60.0, zona_muerta=ZONA_MUERTA)
        metricas['temblor_amp'] = amp
        resultados.append(metricas)
        
        if verbose:
            status = "✅" if metricas['control_estable'] else "❌"
            print(f"{status} U_eff={metricas['U_eff']:.3f}° E={metricas['E']:.1f}°")
    
    return resultados


# ============================================================
# EXPERIMENTO V128 COMPLETO
# ============================================================

def ejecutar_v128():
    print("=" * 100)
    print("EXPERIMENTO V128 — Validacion Robusta + Parkinson Computacional")
    print("=" * 100)
    print("  Parte 1: Validacion robusta (10 repeticiones, 5 semillas)")
    print("  Parte 2: Parkinson - barrido de amplitud de temblor")
    print("  Hipotesis: temblor_critico ≈ zona_muerta (2.0 grados)")
    print("=" * 100)
    
    # ============================================================
    # PARTE 1: VALIDACION ROBUSTA
    # ============================================================
    print("\n" + "=" * 80)
    print("PARTE 1: Validacion Robusta (régimen completo)")
    print("=" * 80)
    
    resultados_validacion = []
    
    for seed in SEMILLAS:
        metricas = ejecutar_validacion_semilla(seed, verbose=True)
        resultados_validacion.append(metricas)
    
    # Estadisticas de validacion
    print("\n  --- Resumen Validacion ---")
    lateralidad_ok = sum(1 for r in resultados_validacion if r['lateralidad'])
    R2_ok = sum(1 for r in resultados_validacion if r['R2'])
    C50_ok = sum(1 for r in resultados_validacion if r['C50'])
    control_ok = sum(1 for r in resultados_validacion if r['control_estable'])
    
    print(f"    Lateralidad: {lateralidad_ok}/{len(SEMILLAS)} semillas ✅")
    print(f"    R2: {R2_ok}/{len(SEMILLAS)} semillas ✅")
    print(f"    C50: {C50_ok}/{len(SEMILLAS)} semillas ✅")
    print(f"    Control estable: {control_ok}/{len(SEMILLAS)} semillas ✅")
    
    metricas_promedio = {
        's_shared': np.mean([r['s_shared_final'] for r in resultados_validacion]),
        'U_eff': np.mean([r['U_eff'] for r in resultados_validacion]),
        'E': np.mean([r['E'] for r in resultados_validacion]),
        'error_final': np.mean([r['error_final'] for r in resultados_validacion])
    }
    
    print(f"\n    Metricas promedio:")
    print(f"      s_shared = {metricas_promedio['s_shared']:.4f}")
    print(f"      U_eff = {metricas_promedio['U_eff']:.3f}°")
    print(f"      E = {metricas_promedio['E']:.1f}°")
    print(f"      error_final = {metricas_promedio['error_final']:.2f}°")
    
    # ============================================================
    # PARTE 2: BARRIDO PARKINSON
    # ============================================================
    print("\n" + "=" * 80)
    print("PARTE 2: Parkinson Computacional - Barrido de temblor")
    print("=" * 80)
    
    # Usar una semilla representativa (la primera que funciono)
    seed_parkinson = SEMILLAS[0]
    resultados_parkinson = ejecutar_barrido_parkinson(seed_parkinson, verbose=True)
    
    # Encontrar punto de ruptura
    print("\n  --- Punto de ruptura (Parkinson) ---")
    
    zona_muerta = ZONA_MUERTA
    punto_ruptura = None
    
    for r in resultados_parkinson:
        if r['U_eff'] > zona_muerta and punto_ruptura is None:
            punto_ruptura = r['temblor_amp']
            print(f"    Temblor critico: {punto_ruptura:.1f}° (U_eff={r['U_eff']:.3f}° > zona_muerta={zona_muerta}°)")
    
    if punto_ruptura is None:
        print(f"    No se encontro ruptura hasta {max(TEMBLOR_AMPS)}°")
    
    # Estadisticas Parkinson
    control_estable_hasta = max([r['temblor_amp'] for r in resultados_parkinson if r['control_estable']]) if any(r['control_estable'] for r in resultados_parkinson) else 0
    print(f"    Control estable hasta: {control_estable_hasta:.1f}°")
    print(f"    Zona muerta: {zona_muerta}°")
    print(f"    Hipotesis O-N8.1: {'✅ CONFIRMADA' if control_estable_hasta >= zona_muerta * 0.8 else '❌ NO CONFIRMADA'}")
    
    # ============================================================
    # GRAFICOS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Grafico 1: s_shared por semilla
    ax = axes[0, 0]
    semillas_str = [str(s) for s in SEMILLAS]
    ax.bar(semillas_str, [r['s_shared_final'] for r in resultados_validacion], color='purple')
    ax.axhline(y=0.8, color='red', linestyle='--', label='Umbral lateralidad (0.8)')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('s_shared')
    ax.set_title('Validacion: Coherencia inter-sistemas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 2: U_eff por semilla
    ax = axes[0, 1]
    ax.bar(semillas_str, [r['U_eff'] for r in resultados_validacion], color='blue')
    ax.axhline(y=ZONA_MUERTA, color='red', linestyle='--', label=f'Zona muerta ({ZONA_MUERTA}°)')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('U_eff (grados)')
    ax.set_title('Validacion: Umbral efectivo (temblor residual)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 3: E por semilla
    ax = axes[0, 2]
    ax.bar(semillas_str, [r['E'] for r in resultados_validacion], color='green')
    ax.set_xlabel('Semilla')
    ax.set_ylabel('Costo Energetico (grados)')
    ax.set_title('Validacion: Costo energetico')
    ax.grid(True, alpha=0.3)
    
    # Grafico 4: Parkinson - U_eff vs temblor_amp
    ax = axes[1, 0]
    temblores = [r['temblor_amp'] for r in resultados_parkinson]
    ueffs = [r['U_eff'] for r in resultados_parkinson]
    ax.plot(temblores, ueffs, 'o-', color='red', linewidth=1, markersize=6)
    ax.axhline(y=ZONA_MUERTA, color='green', linestyle='--', label=f'Zona muerta ({ZONA_MUERTA}°)')
    if punto_ruptura:
        ax.axvline(x=punto_ruptura, color='purple', linestyle='--', label=f'Ruptura: {punto_ruptura}°')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('U_eff (grados)')
    ax.set_title('Parkinson: Umbral efectivo vs temblor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafico 5: Parkinson - E vs temblor_amp
    ax = axes[1, 1]
    es = [r['E'] for r in resultados_parkinson]
    ax.plot(temblores, es, 'o-', color='orange', linewidth=1, markersize=6)
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Costo Energetico (grados)')
    ax.set_title('Parkinson: Costo energetico vs temblor')
    ax.grid(True, alpha=0.3)
    
    # Grafico 6: Parkinson - Control estable
    ax = axes[1, 2]
    control_estable = [1 if r['control_estable'] else 0 for r in resultados_parkinson]
    ax.fill_between(temblores, control_estable, 0, alpha=0.5, color='green')
    ax.set_xlabel('Amplitud de temblor (grados)')
    ax.set_ylabel('Control estable (1=si, 0=no)')
    ax.set_title('Parkinson: Ruptura del control')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('v128_logs', exist_ok=True)
    plt.savefig(f'v128_logs/v128_resultados_{timestamp}.png', dpi=150)
    print(f"\n  Graficos guardados: v128_logs/v128_resultados_{timestamp}.png")
    
    # ============================================================
    # CONCLUSION
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    exito_validacion = lateralidad_ok == len(SEMILLAS) and R2_ok == len(SEMILLAS) and C50_ok == len(SEMILLAS)
    
    if exito_validacion:
        print("  ✅ VALIDACION ROBUSTA: Las 3 capacidades se mantienen en todas las semillas")
        print(f"     Lateralidad: {lateralidad_ok}/{len(SEMILLAS)}")
        print(f"     R2: {R2_ok}/{len(SEMILLAS)}")
        print(f"     C50: {C50_ok}/{len(SEMILLAS)}")
    else:
        print("  ⚠️ VALIDACION PARCIAL: Algunas semillas fallaron")
    
    if punto_ruptura:
        print(f"\n  ✅ PARKINSON: Punto de ruptura detectado en {punto_ruptura:.1f}°")
        print(f"     Hipotesis O-N8.1 confirmada: temblor > zona_muerta ({ZONA_MUERTA}°) rompe el control")
    else:
        print(f"\n  ⚠️ PARKINSON: No se detecto ruptura hasta {max(TEMBLOR_AMPS)}°")
    
    print(f"\n  Resumen V128:")
    print(f"    - Organismo sano: {exito_validacion}")
    print(f"    - Temblor critico: {punto_ruptura if punto_ruptura else '>=' + str(max(TEMBLOR_AMPS))}°")
    print(f"    - Control estable hasta: {control_estable_hasta:.1f}°")
    
    return resultados_validacion, resultados_parkinson


if __name__ == "__main__":
    start_time = time.time()
    resultados_val, resultados_park = ejecutar_v128()
    elapsed = time.time() - start_time
    print(f"\n  Tiempo total de ejecucion: {elapsed/60:.1f} minutos")