#!/usr/bin/env python3
"""
Simulación VSTCosmo - Entrada Estructurada por Amplitud
La perturbación entra en regiones específicas según la amplitud de la muestra.
Sin análisis de frecuencia. Solo mapeo directo.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (SIN CAMBIOS, SOLO ESTRUCTURA DE ENTRADA)
# ============================================================
DIM_FREQ = 16
DIM_TIME = 50
DT = 0.01
DURACION = 10.0
N_PASOS = int(DURACION / DT)

# Dinámica de A (sin cambios)
REFUERZO_A = 0.1
INHIBICION_A = 0.15
DIFUSION_A = 0.05
DISIPACION_A = 0.02
BASAL_A = 0.1

# Acoplamiento (sin cambios)
DIFUSION_BASE = 0.2

# Límites
LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

# ============================================================
# CARGA DE AUDIO
# ============================================================
def cargar_audio(ruta):
    sr, data = wav.read(ruta)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return sr, data

print("Cargando audio: Voz+Viento_1.wav")
sr, audio = cargar_audio('Voz+Viento_1.wav')
print(f"  Frecuencia: {sr} Hz")
print(f"  Duración: {len(audio)/sr:.2f} s")
print(f"  Simulando primeros {DURACION:.1f} segundos ({N_PASOS} pasos)")

# ============================================================
# INICIALIZACIÓN
# ============================================================
def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.1 + 0.05

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

# ============================================================
# PERTURBACIÓN ESTRUCTURADA POR AMPLITUD
# ============================================================
def aplicar_perturbacion_estructurada(Phi, muestra, paso):
    """
    La perturbación entra en regiones específicas según la amplitud.
    NO es un análisis de frecuencia. Es un mapeo directo.
    """
    # Normalizar muestra a [0,1]
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    
    # Mapeo amplitud → banda (sin semántica)
    banda = int(m * (DIM_FREQ - 1))
    t_idx = paso % DIM_TIME
    
    # La amplitud modula la intensidad
    intensidad = 0.05 * abs(muestra)
    
    # Excitar banda central y vecinos
    for db in range(-2, 3):
        b = (banda + db) % DIM_FREQ
        peso = np.exp(-(db**2) / 4)  # gaussiano simple
        Phi[b, t_idx] += intensidad * peso
    
    # Propagación temporal (continuidad)
    for dt in range(-1, 2):
        t = (t_idx + dt) % DIM_TIME
        for db in range(-1, 2):
            b = (banda + db) % DIM_FREQ
            peso_t = np.exp(-(dt**2) / 2)
            Phi[b, t] += intensidad * 0.3 * peso_t
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# DINÁMICA DE A (sin cambios)
# ============================================================
def vecinos_inmediatos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4

def actualizar_atencion(A, Phi):
    # Auto-refuerzo
    auto = REFUERZO_A * A * (1 - A)
    
    # Inhibición lateral
    inhib = -INHIBICION_A * vecinos_inmediatos(A)
    
    # Difusión
    dif = DIFUSION_A * (vecinos_inmediatos(A) - A)
    
    # Disipación
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA = auto + inhib + dif + dis
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ACOPLAMIENTO A → Φ (sin cambios)
# ============================================================
def acoplamiento_atencion_campo(Phi, A):
    # Vecinos
    vecinos = (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
               np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4
    
    # Mezcla modulada por A
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    
    # Flujo
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_BASE * flujo
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
print("\n" + "=" * 60)
print("SIMULACIÓN VSTCosmo - Entrada Estructurada por Amplitud")
print("La perturbación entra en regiones específicas del campo")
print("Sin tocar parámetros de A")
print("=" * 60)

Phi = inicializar_campo()
A = inicializar_atencion()

registro = {'A': [], 'Phi': [], 'tiempo': [], 'energia_A': [], 'rango_A': []}
momentos = [0, N_PASOS // 4, N_PASOS // 2, 3 * N_PASOS // 4, N_PASOS - 1]

muestra_anterior = 0.0

for paso in range(N_PASOS):
    t = paso * DT
    
    # Obtener muestra del audio
    idx = int(t * sr)
    idx = min(idx, len(audio) - 1)
    muestra = audio[idx] if idx >= 0 else 0.0
    
    # 1. Perturbación estructurada
    Phi = aplicar_perturbacion_estructurada(Phi, muestra, paso)
    
    # 2. Actualizar atención
    A = actualizar_atencion(A, Phi)
    
    # 3. Acoplamiento A → Φ
    Phi = acoplamiento_atencion_campo(Phi, A)
    
    # Registrar
    if paso in momentos:
        registro['A'].append(A.copy())
        registro['Phi'].append(Phi.copy())
        registro['tiempo'].append(t)
        registro['energia_A'].append(np.mean(A))
        registro['rango_A'].append(np.max(A) - np.min(A))
        print(f"  Registrado t={t:.1f}s | energía A={np.mean(A):.3f} | rango A={np.max(A)-np.min(A):.3f}")
    
    muestra_anterior = muestra
    
    if paso % (N_PASOS // 10) == 0:
        print(f"  Progreso: {100 * paso // N_PASOS}%")

print("Simulación completada.\n")

# ============================================================
# OBSERVACIÓN
# ============================================================
print("=" * 60)
print("DOCUMENTACIÓN DE FENÓMENOS")
print("=" * 60)

for i, (A, t, energia, rango) in enumerate(zip(registro['A'], registro['tiempo'], 
                                                  registro['energia_A'], registro['rango_A'])):
    print(f"\nt={t:.1f}s:")
    print(f"  Atención: media={np.mean(A):.4f}, var={np.var(A):.6f}, rango={rango:.4f}")
    print(f"  Energía total: {energia:.4f}")
    
    # Múltiples focos (sin umbral fijo)
    if rango > 0.15:
        print(f"    → Posible formación de focos (rango {rango:.3f})")
        # Contar regiones sobre media + 1 desvío
        umbral = np.mean(A) + np.std(A)
        n_focos = np.sum(A > umbral)
        print(f"    → Regiones sobre umbral: {n_focos}")

# Resumen
print("\n" + "=" * 60)
print("RESUMEN DE FENÓMENOS")
print("=" * 60)

rangos_finales = [r for r in registro['rango_A'] if r > 0]
if rangos_finales:
    max_rango = max(rangos_finales)
    print(f"Rango máximo de atención: {max_rango:.4f}")
    if max_rango > 0.15:
        print("✓ Se observaron regiones diferenciadas")
    else:
        print("✗ No se observaron regiones claramente diferenciadas")
else:
    print("✗ No se observaron regiones diferenciadas")

# Gráfico
if len(registro['A']) > 0:
    fig, axes = plt.subplots(2, len(registro['A']), figsize=(4*len(registro['A']), 6))
    if len(registro['A']) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (A, Phi, t) in enumerate(zip(registro['A'], registro['Phi'], registro['tiempo'])):
        axes[0, i].imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Atención t={t:.1f}s')
        axes[0, i].set_xlabel('Memoria')
        axes[0, i].set_ylabel('Banda')
        
        axes[1, i].imshow(Phi, aspect='auto', cmap='viridis')
        axes[1, i].set_title(f'Campo Φ t={t:.1f}s')
        axes[1, i].set_xlabel('Memoria')
        axes[1, i].set_ylabel('Banda')
    
    plt.suptitle('VSTCosmo - Entrada Estructurada por Amplitud', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_entrada_estructurada.png', dpi=150)
    print("\nGráfico guardado: vstcosmo_entrada_estructurada.png")