#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V124-ANIMA-1: Controlador P + Zona Muerta + Audio Restaurado en Fase 2
=======================================================================
Basado en V123, corrigiendo el bug que dejaba los hemisferios sin audio
durante el entrenamiento lateral.

Cambios respecto a V123:
- Fase 2: Audio restaurado (voz a -60°, clicks a +60°)
- Parámetros ajustados para balancear lateralidad y orientación
- Kp = 0.008 (ligeramente más alto que V123)
- FACTOR_GRAD = 8.0 (reducido para no dominar la dinámica)

Hipótesis: Con audio en Fase 2, los 3 criterios (R₂, lateralidad, C50) coexisten.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from datetime import datetime

# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

FS = 1000.0              # Frecuencia de muestreo (Hz)
DT = 1.0 / FS            # Paso de tiempo (s)
TAU_IZQ = 30.0           # Constante de tiempo izquierda (R₂, ruido_rosa)
TAU_DER = 300.0          # Constante de tiempo derecha (lateralidad, clicks)

# Parámetros del cuerpo calloso
UMBRAL_CALLOSO = 0.5
GANANCIA_CALLOSO = 0.01

# Parámetros de inhibición recíproca
UMBRAL_INHIBICION = 0.1

# Parámetros de ANIMA-1 (gradiente de orientación)
FACTOR_GRAD = 8.0        # REDUCIDO (era 20 en V123) para no dominar
SETPOINT_C50 = -60.0     # Objetivo de orientación para C50

# Parámetros del motor (ajustados)
KP_MOTOR = 0.008         # Ligeramente más alto que V123 (0.005)
UMBRAL_SENSOR = 0.05     # Gate binaural
ZONA_MUERTA = 2.0        # Grados

# ============================================================================
# GENERADORES DE RUIDO
# ============================================================================

def ruido_blanco(duracion, fs=FS):
    """Genera ruido blanco gaussiano."""
    n_samples = int(duracion * fs)
    return np.random.normal(0, 1, n_samples)

def ruido_rosa(duracion, fs=FS):
    """Genera ruido rosa (1/f) mediante filtro de colores."""
    n_samples = int(duracion * fs)
    ruido_b = np.random.normal(0, 1, n_samples)
    b, a = signal.butter(2, 0.1, btype='low')
    return signal.filtfilt(b, a, ruido_b)

def generar_clicks_poisson(duracion, tasa=2.0, fs=FS):
    """Genera clicks con distribución de Poisson."""
    n_samples = int(duracion * fs)
    clicks = np.zeros(n_samples)
    n_clicks = int(tasa * duracion)
    for _ in range(n_clicks):
        pos = int(np.random.uniform(0.1, 0.9) * n_samples)
        clicks[pos:pos+5] = 1.0
    return clicks

def localizar(mono, angulo_grados, fs=FS):
    """
    Localiza un sonido mono a un ángulo usando diferencia de amplitud.
    Ley del seno simplificada.
    """
    angulo = np.clip(angulo_grados, -90, 90)
    # Ganancia izquierda/derecha
    gan_L = (np.cos(np.radians(angulo)) + 1) / 2
    gan_R = (np.sin(np.radians(angulo)) + 1) / 2
    # Normalizar para preservar energía
    suma = gan_L + gan_R + 1e-9
    gan_L, gan_R = gan_L / suma, gan_R / suma
    return mono * gan_L, mono * gan_R

# ============================================================================
# HEMISFERIO (Órgano Mínimo)
# ============================================================================

class Hemisferio:
    """
    Hemisferio cerebral mínimo.
    Integra entrada sensorial + ruido + señal del calloso.
    """
    def __init__(self, tau, nombre):
        self.tau = tau
        self.nombre = nombre
        self.valor = 0.0          # Estado interno (Ω)
        self.s_entrada = 0.0      # Señal sensorial entrante
        self.s_calloso = 0.0      # Señal proveniente del calloso
        self.historial = []       # Historial de valor
        self.phi = []             # Historial de fase (para métricas)

    def actualizar(self, entrada, s_calloso, dt):
        """Actualiza el estado del hemisferio con ecuación diferencial."""
        self.s_entrada = entrada
        self.s_calloso = s_calloso

        # Ecuación: dΩ/dt = (entrada + s_calloso - Ω) / τ
        d_valor = (entrada + s_calloso - self.valor) / self.tau
        self.valor += d_valor * dt

        # Inhibición autolimitante (evita explosión)
        self.valor = np.clip(self.valor, -1.0, 1.0)

        self.historial.append(self.valor)
        # Para métricas de lateralidad, usamos valor como proxy de fase
        self.phi.append(self.valor)

    def get_valor(self):
        """Retorna el valor actual de Ω."""
        return self.valor

    def get_phi(self):
        """Retorna la fase (para correlación)."""
        return np.array(self.phi)

    def reset(self):
        self.valor = 0.0
        self.s_entrada = 0.0
        self.s_calloso = 0.0
        self.historial = []
        self.phi = []

# ============================================================================
# CUERPO CALLOSO (Acoplamiento entre hemisferios)
# ============================================================================

class CuerpoCalloso:
    """
    Cuerpo calloso mínimo.
    Transfiere señal entre hemisferios si la diferencia supera el umbral.
    """
    def __init__(self, umbral, ganancia):
        self.umbral = umbral
        self.ganancia = ganancia

    def transferir(self, valor_L, valor_R):
        """Calcula señal transferida L→R y R→L."""
        diff = valor_L - valor_R

        if abs(diff) > self.umbral:
            s_LR = self.ganancia * max(0, diff)
            s_RL = self.ganancia * max(0, -diff)
        else:
            s_LR = 0.0
            s_RL = 0.0

        return s_LR, s_RL

# ============================================================================
# INHIBICIÓN RECÍPROCA (Competencia hemisférica)
# ============================================================================

class InhibicionReciproca:
    """
    Inhibición recíproca entre hemisferios.
    El más activo inhibe al otro.
    """
    def __init__(self, umbral):
        self.umbral = umbral

    def inhibir(self, valor_L, valor_R):
        """Calcula factor de inhibición para cada hemisferio."""
        if valor_L > valor_R + self.umbral:
            return 1.0, 0.0  # L inhibe completamente a R
        elif valor_R > valor_L + self.umbral:
            return 0.0, 1.0  # R inhibe completamente a L
        else:
            return 0.5, 0.5  # Sin inhibición dominante

# ============================================================================
# APARATO MOTOR (Control P + Zona Muerta)
# ============================================================================

class AparatoMotor:
    """
    Control Proporcional con zona muerta.
    No usa el sensor para modular velocidad, solo como gate.
    """
    def __init__(self):
        self.orientacion = 0.0
        self.setpoint = SETPOINT_C50
        self.Kp = KP_MOTOR
        self.umbral_sensor = UMBRAL_SENSOR
        self.zona_muerta = ZONA_MUERTA
        self.historial = []

    def actuar(self, valor_L, valor_R, LF_activa):
        """
        Retorna la orientación actualizada (grados, -90 a +90).
        """
        if not LF_activa:
            return self.orientacion

        # Gate: ¿hay estímulo binaural suficiente?
        sensor_binaural = abs(valor_L - valor_R)
        if sensor_binaural < self.umbral_sensor:
            return self.orientacion

        # Error: dónde estoy vs dónde debo estar
        error = self.setpoint - self.orientacion

        # Zona muerta: si estoy cerca del objetivo, no mover
        if abs(error) < self.zona_muerta:
            return self.orientacion

        # Control P puro
        delta = self.Kp * error
        self.orientacion += delta
        self.orientacion = np.clip(self.orientacion, -90.0, 90.0)

        self.historial.append(self.orientacion)
        return self.orientacion

    def reset(self):
        self.orientacion = 0.0
        self.historial = []

# ============================================================================
# SISTEMA BINAURAL COMPLETO
# ============================================================================

class SistemaBinaural:
    """
    Sistema completo: dos hemisferios, calloso, inhibición y motor.
    """
    def __init__(self):
        self.H_L = Hemisferio(TAU_IZQ, "Izquierdo")
        self.H_R = Hemisferio(TAU_DER, "Derecho")
        self.calloso = CuerpoCalloso(UMBRAL_CALLOSO, GANANCIA_CALLOSO)
        self.inhibicion = InhibicionReciproca(UMBRAL_INHIBICION)
        self.motor = AparatoMotor()
        self.LF_activa = False
        self.historial_orientacion = []
        self.historial_s_shared = []

    def reset(self):
        self.H_L.reset()
        self.H_R.reset()
        self.motor.reset()
        self.LF_activa = False
        self.historial_orientacion = []
        self.historial_s_shared = []

    def paso(self, entrada_L, entrada_R, dt, entrenamiento_lateral=False):
        """
        Un paso de tiempo del sistema.
        """
        # Obtener valores actuales
        val_L = self.H_L.get_valor()
        val_R = self.H_R.get_valor()

        # 1. Calcular transferencia callosa
        s_LR, s_RL = self.calloso.transferir(val_L, val_R)

        # 2. Calcular inhibición recíproca
        f_L, f_R = self.inhibicion.inhibir(val_L, val_R)

        # 3. Aplicar inhibición a las señales de entrada
        entrada_L_efectiva = entrada_L * f_L
        entrada_R_efectiva = entrada_R * f_R

        # 4. Actualizar hemisferios
        self.H_L.actualizar(entrada_L_efectiva, s_RL, dt)
        self.H_R.actualizar(entrada_R_efectiva, s_LR, dt)

        # 5. Calcular orientación (con gradiente si aplica)
        if entrenamiento_lateral:
            error_grad = self.motor.orientacion - SETPOINT_C50
            fuerza_grad = -FACTOR_GRAD * error_grad / 90.0

            if fuerza_grad > 0:
                entrada_L_efectiva *= (1 - min(0.5, fuerza_grad))
            elif fuerza_grad < 0:
                entrada_R_efectiva *= (1 - min(0.5, -fuerza_grad))

            self.H_L.actualizar(entrada_L_efectiva, s_RL, dt)
            self.H_R.actualizar(entrada_R_efectiva, s_LR, dt)

        # 6. Actualizar motor
        orient = self.motor.actuar(self.H_L.get_valor(), self.H_R.get_valor(), self.LF_activa)
        self.historial_orientacion.append(orient)

        # 7. Calcular s_shared (lateralidad) por correlación
        phi_L = self.H_L.get_phi()
        phi_R = self.H_R.get_phi()
        if len(phi_L) > 10 and np.std(phi_L) > 1e-9 and np.std(phi_R) > 1e-9:
            s_shared = np.corrcoef(phi_L[-100:], phi_R[-100:])[0, 1]
            if np.isnan(s_shared):
                s_shared = 0.0
        else:
            s_shared = 0.0
        
        self.historial_s_shared.append(s_shared)

        return orient, s_shared

    def activar_LF(self):
        """Activa la libertad funcional (R₂ detectado)."""
        self.LF_activa = True

# ============================================================================
# GENERACIÓN DE ESTÍMULOS
# ============================================================================

class Estimulos:
    """
    Generador de estímulos binaurales.
    """
    @staticmethod
    def ruido_rosa_estéreo(duracion, fs=FS):
        """Ruido rosa idéntico en ambos canales."""
        n_samples = int(duracion * fs)
        ruido = ruido_rosa(duracion, fs)
        return ruido, ruido

    @staticmethod
    def ruido_con_bias(duracion, bias, fs=FS):
        """Ruido rosa con bias hacia un canal."""
        L, R = Estimulos.ruido_rosa_estéreo(duracion, fs)
        if bias > 0:
            R *= (1 + bias)
        else:
            L *= (1 - bias)
        return L, R

    @staticmethod
    def audio_entrenamiento_lateral(duracion, fs=FS):
        """
        Genera audio para entrenamiento lateral.
        Voz (ruido rosa) localizada a -60° (izquierda)
        Clicks localizados a +60° (derecha)
        """
        n_samples = int(duracion * fs)
        
        # Fuentes mono
        voz_mono = ruido_rosa(duracion, fs)
        clicks_mono = generar_clicks_poisson(duracion, tasa=2.0, fs=fs)
        
        # Localizar
        voz_L, voz_R = localizar(voz_mono, -60, fs)
        click_L, click_R = localizar(clicks_mono, 60, fs)
        
        # Mezcla
        audio_L = voz_L + click_L * 0.3
        audio_R = click_R + voz_R * 0.3
        
        # Normalizar
        max_val = max(np.max(np.abs(audio_L)), np.max(np.abs(audio_R)))
        if max_val > 0:
            audio_L /= max_val
            audio_R /= max_val
        
        return audio_L, audio_R

# ============================================================================
# MÉTRICAS
# ============================================================================

def calcular_lateralidad(s_shared_hist):
    """Calcula la lateralidad final como promedio de s_shared."""
    if len(s_shared_hist) == 0:
        return 0.0
    ventana = int(len(s_shared_hist) * 0.2)
    if ventana == 0:
        ventana = len(s_shared_hist)
    return np.mean(s_shared_hist[-ventana:])

def calcular_respuesta_R2(sistema, duracion=10.0):
    """Evalúa la respuesta a inanición (R₂)."""
    print("  Evaluando R₂ (inanición)...")
    estimulos = Estimulos()
    sistema.reset()

    # Baseline: ruido rosa simétrico
    L_base, R_base = estimulos.ruido_rosa_estéreo(duracion)
    baseline = []
    for i in range(len(L_base)):
        orient, s_shared = sistema.paso(L_base[i], R_base[i], DT)
        baseline.append(sistema.H_L.get_valor() + sistema.H_R.get_valor())
    
    ventana = int(0.3 * len(baseline))
    if ventana == 0:
        ventana = len(baseline)
    omega_base = np.mean(np.abs(baseline[-ventana:]))

    # Test: ruido con bias fuerte
    sistema.reset()
    L_test, R_test = estimulos.ruido_con_bias(duracion, bias=0.5)
    respuesta = []
    for i in range(len(L_test)):
        orient, s_shared = sistema.paso(L_test[i], R_test[i], DT)
        respuesta.append(sistema.H_L.get_valor() + sistema.H_R.get_valor())
    
    ventana = int(0.3 * len(respuesta))
    if ventana == 0:
        ventana = len(respuesta)
    omega_resp = np.mean(np.abs(respuesta[-ventana:]))

    if omega_base < 0.001:
        resp = omega_resp
    else:
        resp = omega_resp / (omega_base + 0.001)

    return resp, omega_resp, omega_base

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def generar_graficos(sistema, timestamp, output_dir="v124_anima1_logs"):
    """Genera gráficos de resultados."""
    os.makedirs(output_dir, exist_ok=True)

    t = np.arange(len(sistema.historial_orientacion)) * DT

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Gráfico 1: Orientación del motor
    axes[0].plot(t, sistema.historial_orientacion, 'b-', linewidth=1)
    axes[0].axhline(y=SETPOINT_C50, color='r', linestyle='--', label=f'Setpoint ({SETPOINT_C50}°)')
    axes[0].axhline(y=-90, color='gray', linestyle=':', alpha=0.5)
    axes[0].axhline(y=90, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Orientación (°)')
    axes[0].set_title(f'V124 - Orientación del Motor (objetivo: {SETPOINT_C50}°)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico 2: s_shared (lateralidad)
    axes[1].plot(t, sistema.historial_s_shared, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5, label='Umbral lateralidad')
    axes[1].set_ylabel('s_shared')
    axes[1].set_title('Lateralidad (s_shared)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gráfico 3: Valor de cada hemisferio
    ventana = int(100.0 / DT)
    if len(sistema.H_L.historial) > ventana:
        t_ult = t[-ventana:]
        val_L_ult = sistema.H_L.historial[-ventana:]
        val_R_ult = sistema.H_R.historial[-ventana:]
    else:
        t_ult = t
        val_L_ult = sistema.H_L.historial
        val_R_ult = sistema.H_R.historial

    axes[2].plot(t_ult, val_L_ult, 'r-', linewidth=0.8, label='Ω Izquierdo')
    axes[2].plot(t_ult, val_R_ult, 'b-', linewidth=0.8, label='Ω Derecho')
    axes[2].set_ylabel('Ω')
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_title('Actividad Hemisférica (últimos 100s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{output_dir}/v124_anima1_resultados_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  📊 Gráficos guardados: {filename}")

# ============================================================================
# EXPERIMENTO PRINCIPAL
# ============================================================================

def ejecutar_experimento():
    """Ejecuta el experimento V124 completo."""
    print("\n" + "="*100)
    print("EXPERIMENTO V124-ANIMA-1 (CONTROL P + AUDIO RESTAURADO)")
    print("="*100)
    print(f"\nParámetros:")
    print(f"  TAU_Izq = {TAU_IZQ}s (R₂, ruido_rosa)")
    print(f"  TAU_Der = {TAU_DER}s (lateralidad, clicks)")
    print(f"  Calloso: umbral={UMBRAL_CALLOSO}, ganancia={GANANCIA_CALLOSO}")
    print(f"  Inhibición: umbral={UMBRAL_INHIBICION}")
    print(f"  ANIMA-1: gradiente={FACTOR_GRAD}, setpoint={SETPOINT_C50}°")
    print(f"  Motor: Kp={KP_MOTOR}, umbral_sensor={UMBRAL_SENSOR}, zona_muerta={ZONA_MUERTA}°")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # FASE 1: Baseline — Un sistema sin acoplamiento
    # ========================================================================
    print("="*80)
    print("FASE 1: Baseline — Un sistema")
    print("="*80)

    sistema = SistemaBinaural()
    estimulos = Estimulos()

    duracion_fase1 = 400.0
    n_steps_fase1 = int(duracion_fase1 / DT)
    L_fase1, R_fase1 = estimulos.ruido_rosa_estéreo(duracion_fase1)

    for i in range(n_steps_fase1):
        orient, s_shared = sistema.paso(L_fase1[i], R_fase1[i], DT)
        if i % int(100.0 / DT) == 0 and i > 0:
            print(f"    t={i*DT:.0f}s | Ω_L={sistema.H_L.get_valor():.4f}, Ω_R={sistema.H_R.get_valor():.4f}")

    print("  ✅ Fase 1 completada.\n")

    # ========================================================================
    # FASE 2: Dos sistemas acoplados (entrenamiento lateral CON AUDIO)
    # ========================================================================
    print("="*80)
    print("FASE 2: Dos sistemas acoplados (entrenamiento lateral CON AUDIO)")
    print("="*80)

    N_REPETICIONES = 10
    duracion_rep = 45.0  # segundos por repetición
    pasos_rep = int(duracion_rep / DT)

    s_shared_finales = []

    for rep in range(N_REPETICIONES):
        print(f"  Repetición {rep+1}/{N_REPETICIONES}...")
        sistema_ent = SistemaBinaural()
        sistema_ent.activar_LF()
        
        # Generar audio para esta repetición
        audio_L, audio_R = Estimulos.audio_entrenamiento_lateral(duracion_rep + 1.0, FS)
        
        for i in range(pasos_rep):
            t = i * DT
            idx = int(t * FS)
            
            # Extraer valor de audio
            if idx < len(audio_L):
                L_val = audio_L[idx]
                R_val = audio_R[idx]
            else:
                L_val, R_val = 0.0, 0.0
            
            orient, s_shared = sistema_ent.paso(L_val, R_val, DT, entrenamiento_lateral=True)
            
            if i % int(10.0 / DT) == 0 and i > 0:
                print(f"      t={i*DT:.0f}s | s_shared={s_shared:.3f}, orient={orient:.1f}°")
        
        s_shared_final = calcular_lateralidad(sistema_ent.historial_s_shared)
        s_shared_finales.append(s_shared_final)
        print(f"    Lateralidad final: s_shared={s_shared_final:.4f}")

    lateralidad_prom = np.mean(s_shared_finales)
    print(f"\n  Lateralidad: {'✅' if lateralidad_prom < -0.3 else '❌'} (s_shared={lateralidad_prom:.4f})\n")

    # ========================================================================
    # FASE 3: Test R₂ (inanición)
    # ========================================================================
    print("="*80)
    print("FASE 3: Test R₂ (inanición)")
    print("="*80)

    sistema_test = SistemaBinaural()
    # Entrenamiento previo rápido
    audio_L_short, audio_R_short = Estimulos.audio_entrenamiento_lateral(10.0, FS)
    for i in range(min(len(audio_L_short), int(10.0 / DT))):
        sistema_test.paso(audio_L_short[i], audio_R_short[i], DT, entrenamiento_lateral=True)

    resp, omega_resp, omega_base = calcular_respuesta_R2(sistema_test)
    umbral_R2 = 0.03
    print(f"  Baseline: Ω_base = {omega_base:.4f}")
    print(f"  Respuesta: Ω_resp = {omega_resp:.4f}")
    print(f"  R₂: {'✅' if resp > umbral_R2 else '❌'} (resp={resp:.4f} > umbral={umbral_R2})\n")

    # ========================================================================
    # FASE 4: Test C50 (orientación a -60°)
    # ========================================================================
    print("="*80)
    print("FASE 4: Test C50 (orientación a -60°)")
    print("="*80)

    sistema_c50 = SistemaBinaural()
    sistema_c50.activar_LF()

    duracion_c50 = 500.0
    n_steps_c50 = int(duracion_c50 / DT)
    
    # Audio de prueba: ruido rosa simétrico + bias gradual
    for i in range(n_steps_c50):
        # Bias que aumenta lentamente para empujar la orientación
        bias = min(0.5, i / n_steps_c50)
        L_val = 0.5 * (1 - bias)
        R_val = 0.5 * (1 + bias)
        
        orient, s_shared = sistema_c50.paso(L_val, R_val, DT, entrenamiento_lateral=True)

        if i % int(50.0 / DT) == 0 and i > 0:
            print(f"    t={i*DT:.1f}s | orient={orient:.1f}°, s_shared={s_shared:.3f}")

    orient_final = sistema_c50.motor.orientacion
    ventana_estable = int(50.0 / DT)
    if len(sistema_c50.historial_orientacion) > ventana_estable:
        orient_estable = np.mean(sistema_c50.historial_orientacion[-ventana_estable:])
        orient_std = np.std(sistema_c50.historial_orientacion[-ventana_estable:])
    else:
        orient_estable = orient_final
        orient_std = 0

    exito_c50 = abs(orient_estable - SETPOINT_C50) <= 5.0

    print(f"\n  Orientación final: {orient_final:.1f}°")
    print(f"  Orientación estable (últimos 50s): {orient_estable:.1f}° ± {orient_std:.1f}°")
    print(f"  C50 ({SETPOINT_C50}°): {'✅' if exito_c50 else '❌'}")

    # ========================================================================
    # GENERAR GRÁFICOS
    # ========================================================================
    generar_graficos(sistema_c50, timestamp)

    # ========================================================================
    # CONCLUSIÓN
    # ========================================================================
    print("\n" + "="*100)
    print("CONCLUSIÓN")
    print("="*100)
    print(f"  Lateralidad: {'✅' if lateralidad_prom < -0.3 else '❌'} (s_shared={lateralidad_prom:.4f})")
    print(f"  R₂: {'✅' if resp > umbral_R2 else '❌'} (resp={resp:.4f})")
    print(f"  C50 ({SETPOINT_C50}°): {'✅' if exito_c50 else '❌'} (final={orient_estable:.1f}°)")
    print()

    if exito_c50 and lateralidad_prom < -0.3 and resp > umbral_R2:
        print("  ✅ ÉXITO COMPLETO: R₂, LATERALIDAD Y C50 COEXISTEN")
        print("  🧠 ANIMA-1 validado: los tres subsistemas operan simultáneamente")
    elif exito_c50:
        print("  ✅ ÉXITO PARCIAL: C50 alcanzado, pero lateralidad/R₂ pendiente")
    elif lateralidad_prom < -0.3 and resp > umbral_R2:
        print("  ⚠️ ÉXITO PARCIAL: R₂ y lateralidad OK, C50 pendiente")
    else:
        print("  ❌ NO ALCANZADO: Revisar parámetros")

    print(f"\n  📊 Gráficos: v124_anima1_logs/v124_anima1_resultados_{timestamp}.png")

    return {
        'lateralidad': lateralidad_prom,
        'respuesta_R2': resp,
        'orient_final': orient_estable,
        'exito_c50': exito_c50
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    resultados = ejecutar_experimento()