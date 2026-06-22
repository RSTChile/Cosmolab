#!/usr/bin/env python3
"""
VSTCosmos - v72-B Fase 3: Plasticidad Hebbiana (Diagnóstico)
Propósito: Separar aprendizaje real de amplificación artificial.
GAMMA_PLAST reducido a 0.03 para evitar autoexcitación.
Test C usa RUIDO BLANCO sin entrenamiento (no silencio).
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DEL CAMPO TOTAL
# ============================================================
DIM_INTERNA = 32
DIM_AUDITIVA = 32
DIM_TOTAL = DIM_INTERNA + DIM_AUDITIVA

DIM_TIME = 100
DT = 0.01
DURACION_ENTRENO = 30.0
DURACION_TEST = 30.0
N_PASOS_ENTRENO = int(DURACION_ENTRENO / DT)
N_PASOS_TEST = int(DURACION_TEST / DT)

# Parámetros de dinámica
DIFUSION_BASE = 0.15
GANANCIA_REACCION = 0.05

OMEGA_MIN = 0.05
OMEGA_MAX = 0.50
AMORT_MIN = 0.01
AMORT_MAX = 0.08
PHI_EQUILIBRIO = 0.5

# Parámetros de plasticidad hebbiana (REDUCIDOS)
ETA_HEBB = 0.02        # tasa de aprendizaje hebbiano
TAU_W = 0.005          # decaimiento de pesos
GAMMA_PLAST = 0.03     # REDUCIDO de 0.15 para evitar autoexcitación
W_MAX = 1.0            # límite para estabilidad numérica

# Parámetros de FFT
VENTANA_FFT_MS = 25
HOP_FFT_MS = 10
F_MIN = 80
F_MAX = 8000

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0


# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta, duracion):
    if "Tono puro" in ruta:
        sr = 48000
        t = np.arange(int(sr * duracion)) / sr
        return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    elif "Ruido blanco" in ruta:
        sr = 48000
        return sr, np.random.normal(0, 0.5, int(sr * duracion))
    else:
        sr, data = wav.read(ruta)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        muestras_necesarias = int(sr * duracion)
        if len(data) < muestras_necesarias:
            data = np.pad(data, (0, muestras_necesarias - len(data)))
        else:
            data = data[:muestras_necesarias]
        return sr, data


def inicializar_campo_total(seed=42):
    np.random.seed(seed)
    return np.random.rand(DIM_TOTAL, DIM_TIME) * 0.2 + 0.4


def inicializar_plasticidad():
    """W: matriz de correlaciones aprendidas entre región interna y auditiva."""
    return np.zeros((DIM_INTERNA, DIM_AUDITIVA))


def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0


def preparar_objetivo_audio(audio, sr, idx_ventana, ventana_muestras, hop_muestras,
                            dim_auditiva, DIM_TIME):
    inicio = idx_ventana * hop_muestras
    if inicio + ventana_muestras > len(audio):
        return np.zeros((dim_auditiva, DIM_TIME))
    
    fragmento = audio[inicio:inicio + ventana_muestras]
    ventana_hann = np.hanning(len(fragmento))
    fragmento = fragmento * ventana_hann
    
    fft = np.fft.rfft(fragmento)
    potencia = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(fragmento), 1/sr)
    
    bandas = np.logspace(np.log10(F_MIN), np.log10(F_MAX), dim_auditiva + 1)
    objetivo = np.zeros(dim_auditiva)
    for b in range(dim_auditiva):
        mask = (freqs >= bandas[b]) & (freqs < bandas[b+1])
        if np.any(mask):
            objetivo[b] = np.mean(potencia[mask])
    
    max_val = np.max(objetivo)
    if max_val > 0:
        objetivo = objetivo / max_val
    
    return objetivo.reshape(-1, 1) * np.ones((1, DIM_TIME))


def calcular_frecuencias_naturales(dim_total, dim_interna):
    bandas = np.arange(dim_total)
    t = np.log1p(bandas) / np.log1p(dim_total - 1)
    omega = OMEGA_MIN + (OMEGA_MAX - OMEGA_MIN) * t
    amort = AMORT_MIN + (AMORT_MAX - AMORT_MIN) * t
    return omega.reshape(-1, 1), amort.reshape(-1, 1)


def actualizar_hebb_y_plasticidad(Phi_total, W, dt):
    region_int = Phi_total[:DIM_INTERNA, :]
    region_aud = Phi_total[DIM_INTERNA:, :]

    # Regla de Hebb
    correlacion = (region_int @ region_aud.T) / DIM_TIME
    dW = ETA_HEBB * correlacion - TAU_W * W
    W_nueva = np.clip(W + dW * dt, -W_MAX, W_MAX)

    # Término plástico
    patron_esperado = W_nueva @ region_aud
    M_plasticidad = GAMMA_PLAST * (patron_esperado - region_int)

    return W_nueva, M_plasticidad


def actualizar_campo_total(Phi_total, Phi_vel_total, W,
                           objetivo_audio, alpha,
                           omega_natural, amort_natural):
    # Difusión
    promedio_local = vecinos(Phi_total)
    difusion = DIFUSION_BASE * (promedio_local - Phi_total)
    
    # Reacción
    desviacion = Phi_total - promedio_local
    reaccion = GANANCIA_REACCION * desviacion * (1 - desviacion**2)
    
    # Oscilación
    term_osc = (-omega_natural**2 * (Phi_total - PHI_EQUILIBRIO)
                - amort_natural * Phi_vel_total)
    
    # Plasticidad hebbiana
    W_nueva, M_plasticidad = actualizar_hebb_y_plasticidad(Phi_total, W, DT)
    
    # M_plasticidad solo actúa sobre region_int
    M_campo = np.zeros_like(Phi_total)
    M_campo[:DIM_INTERNA, :] = M_plasticidad
    
    # Actualización del campo
    dPhi_vel = term_osc + reaccion + difusion + M_campo
    Phi_vel_nueva = Phi_vel_total + DT * dPhi_vel
    Phi_nueva = Phi_total + DT * Phi_vel_nueva
    
    # Sesgo operativo
    if alpha > 0:
        region_auditiva_nueva = Phi_nueva[DIM_INTERNA:, :]
        region_auditiva_nueva = (1 - alpha) * region_auditiva_nueva + alpha * objetivo_audio
        Phi_nueva[DIM_INTERNA:, :] = region_auditiva_nueva
    
    return (np.clip(Phi_nueva, LIMITE_MIN, LIMITE_MAX),
            np.clip(Phi_vel_nueva, -5.0, 5.0),
            W_nueva)


def calcular_gradiente(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return np.mean(np.abs(region_int - region_aud))


def calcular_acoplamiento(Phi_total, dim_interna):
    region_int = Phi_total[:dim_interna, :]
    region_aud = Phi_total[dim_interna:, :]
    return float(np.mean(region_int * region_aud))


# ============================================================
# EXPERIMENTOS
# ============================================================
def experimento_persistencia(seed=42):
    """Test A: Entrena voz, testea voz"""
    print("  Test A: Persistencia (voz -> voz)")
    
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr, audio = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO + DURACION_TEST)
    ventana_muestras = int(sr * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr * HOP_FFT_MS / 1000)
    
    gradientes_entreno = []
    w_norma_historia = []
    
    # Entrenamiento (alpha=0.05)
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_entreno.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        w_norma_historia.append(np.mean(np.abs(W)))
    
    w_tras_entreno = np.mean(np.abs(W))
    
    # Test (alpha=0.0)
    gradientes_test = []
    acoplamientos_test = []
    for idx in range(N_PASOS_ENTRENO, N_PASOS_ENTRENO + N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio, sr, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))
    
    gradiente_estable = np.mean(gradientes_test[-int(10.0/DT):])
    persistencia = sum(1 for g in gradientes_test if g > 0.08) * DT
    
    print(f"    W_tras_entreno = {w_tras_entreno:.4f}")
    print(f"    grad_estable = {gradiente_estable:.4f}")
    print(f"    persistencia = {persistencia:.1f}s")
    
    return {
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': gradiente_estable,
        'persistencia': persistencia,
        'min_acop': min(acoplamientos_test)
    }


def experimento_asimetria(seed=43):
    """Test B: Entrena voz, testea ruido blanco"""
    print("  Test B: Asimetría (voz -> ruido)")
    
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_voz, audio_voz = cargar_audio("Voz_Estudio.wav", duracion=DURACION_ENTRENO)
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_TEST)
    ventana_muestras = int(sr_voz * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_voz * HOP_FFT_MS / 1000)
    
    # Entrenamiento (alpha=0.05) con voz
    for idx in range(N_PASOS_ENTRENO):
        objetivo = preparar_objetivo_audio(audio_voz, sr_voz, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.05,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
    
    w_tras_entreno = np.mean(np.abs(W))
    
    # Test (alpha=0.0) con ruido
    gradientes_test = []
    acoplamientos_test = []
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))
    
    gradiente_estable = np.mean(gradientes_test[-int(10.0/DT):])
    
    print(f"    W_tras_entreno = {w_tras_entreno:.4f}")
    print(f"    grad_estable = {gradiente_estable:.4f}")
    
    return {
        'w_tras_entreno': w_tras_entreno,
        'grad_estable': gradiente_estable,
        'min_acop': min(acoplamientos_test)
    }


def experimento_control_ruido_sin_entreno(seed=44):
    """Test C: Ruido blanco sin entrenamiento (control correcto)"""
    print("  Test C: Control - Ruido sin entrenamiento")
    
    np.random.seed(seed)
    Phi_total = inicializar_campo_total()
    Phi_vel_total = np.zeros_like(Phi_total)
    W = inicializar_plasticidad()  # W = 0
    omega_natural, amort_natural = calcular_frecuencias_naturales(DIM_TOTAL, DIM_INTERNA)
    
    sr_ruido, audio_ruido = cargar_audio("Ruido blanco", duracion=DURACION_TEST)
    ventana_muestras = int(sr_ruido * VENTANA_FFT_MS / 1000)
    hop_muestras = int(sr_ruido * HOP_FFT_MS / 1000)
    
    gradientes_test = []
    acoplamientos_test = []
    
    # Sin entrenamiento previo, alpha=0.0 desde inicio
    for idx in range(N_PASOS_TEST):
        objetivo = preparar_objetivo_audio(audio_ruido, sr_ruido, idx, ventana_muestras, hop_muestras,
                                          DIM_AUDITIVA, DIM_TIME)
        Phi_total, Phi_vel_total, W = actualizar_campo_total(
            Phi_total, Phi_vel_total, W, objetivo, alpha=0.0,
            omega_natural=omega_natural, amort_natural=amort_natural
        )
        gradientes_test.append(calcular_gradiente(Phi_total, DIM_INTERNA))
        acoplamientos_test.append(calcular_acoplamiento(Phi_total, DIM_INTERNA))
    
    gradiente_estable = np.mean(gradientes_test[-int(10.0/DT):])
    
    print(f"    grad_estable = {gradiente_estable:.4f}")
    
    return {
        'grad_estable': gradiente_estable,
        'min_acop': min(acoplamientos_test)
    }


def main():
    print("=" * 100)
    print("VSTCosmos - v72-B Fase 3: Diagnóstico con Parámetros Reducidos")
    print(f"ETA_HEBB = {ETA_HEBB}, TAU_W = {TAU_W}, GAMMA_PLAST = {GAMMA_PLAST}")
    print("Test A: Persistencia (entrena voz, testea voz)")
    print("Test B: Asimetría (entrena voz, testea ruido)")
    print("Test C: Control (ruido sin entrenamiento)")
    print("=" * 100)
    
    print("\n[Ejecutando experimentos...]")
    test_A = experimento_persistencia()
    test_B = experimento_asimetria()
    test_C = experimento_control_ruido_sin_entreno()
    
    # ============================================================
    # DIAGNÓSTICO
    # ============================================================
    print("\n" + "=" * 100)
    print("RESULTADOS")
    print("=" * 100)
    
    print(f"\n  Test A (Voz -> Voz):")
    print(f"    W_tras_entreno = {test_A['w_tras_entreno']:.4f}")
    print(f"    grad_estable = {test_A['grad_estable']:.4f}")
    print(f"    persistencia = {test_A['persistencia']:.1f}s")
    print(f"    min_acop = {test_A['min_acop']:.6f}")
    
    print(f"\n  Test B (Voz -> Ruido):")
    print(f"    W_tras_entreno = {test_B['w_tras_entreno']:.4f}")
    print(f"    grad_estable = {test_B['grad_estable']:.4f}")
    print(f"    min_acop = {test_B['min_acop']:.6f}")
    
    print(f"\n  Test C (Ruido sin entrenamiento):")
    print(f"    grad_estable = {test_C['grad_estable']:.4f}")
    print(f"    min_acop = {test_C['min_acop']:.6f}")
    
    # ============================================================
    # CRITERIOS DE DECISIÓN
    # ============================================================
    print("\n" + "=" * 100)
    print("DIAGNÓSTICO")
    print("=" * 100)
    
    # Verificar si el control funciona (gradiente bajo sin entrenamiento)
    control_funciona = test_C['grad_estable'] < 0.03
    
    # Verificar si W aprendió (W > umbral)
    w_aprendio = test_A['w_tras_entreno'] > 0.001
    
    # Verificar si hay persistencia real
    persistencia_real = test_A['grad_estable'] > 0.10 and test_A['persistencia'] > 10.0
    
    # Verificar asimetría real
    asimetria_real = test_B['grad_estable'] > test_C['grad_estable'] * 1.5
    
    print(f"\n  Control (grad_estable < 0.03): {test_C['grad_estable']:.4f} -> {'✅' if control_funciona else '❌'}")
    print(f"  W aprendió (W > 0.001): {test_A['w_tras_entreno']:.4f} -> {'✅' if w_aprendio else '❌'}")
    print(f"  Persistencia real (grad > 0.10 y >10s): grad={test_A['grad_estable']:.4f}, t={test_A['persistencia']:.1f}s -> {'✅' if persistencia_real else '❌'}")
    print(f"  Asimetría real (B > C x 1.5): {test_B['grad_estable']:.4f} vs {test_C['grad_estable']:.4f} -> {'✅' if asimetria_real else '❌'}")
    
    print("\n" + "=" * 100)
    print("DECISIÓN")
    print("=" * 100)
    
    if control_funciona and w_aprendio and persistencia_real and asimetria_real:
        print("\n  ✅ VALIDACIÓN COMPLETA")
        print("     GAMMA_PLAST = 0.03 es suficiente para evitar autoexcitación.")
        print("     El control (ruido sin entrenamiento) produce gradiente bajo.")
        print("     W_tras_entreno > 0.001 confirma aprendizaje.")
        print("     grad_estable en Test A > 0.10 confirma persistencia real.")
        print("     grad_estable en Test B > Test C x 1.5 confirma asimetría real.")
        print("\n     → La plasticidad hebbiana funciona sin amplificación artificial.")
        print("     → Proceder a ajuste fino de parámetros o cierre.")
    elif control_funciona and w_aprendio and persistencia_real:
        print("\n  ✅ VALIDACIÓN PARCIAL")
        print("     Control y aprendizaje funcionan, pero asimetría insuficiente.")
        print("     → W aprendió pero no generaliza de voz a ruido.")
        print("     → Aumentar ETA_HEBB a 0.05 para fortalecer W.")
    elif control_funciona and w_aprendio:
        print("\n  ⚠️ VALIDACIÓN MÍNIMA")
        print("     Control y aprendizaje OK, pero persistencia baja.")
        print("     → GAMMA_PLAST = 0.03 puede ser insuficiente para sostener.")
        print("     → Aumentar GAMMA_PLAST a 0.05-0.08")
    elif control_funciona:
        print("\n  ❌ NO APRENDIZAJE")
        print("     Control funciona (gradiente bajo sin entrada),")
        print("     pero W_tras_entreno < 0.001. No hubo aprendizaje.")
        print("     → Subir ETA_HEBB a 0.05")
    else:
        print("\n  ❌ CONTROL FALLIDO")
        print("     Gradiente alto incluso sin entrenamiento.")
        print("     La dinámica base del campo es demasiado activa.")
        print("     → Reducir GANANCIA_REACCION o aumentar DIFUSION_BASE")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()