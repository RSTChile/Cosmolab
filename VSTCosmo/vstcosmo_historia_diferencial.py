#!/usr/bin/env python3
"""
Simulación VSTCosmo - Campo con Inestabilidad Simétrica
La dinámica de Φ puede romper la homogeneidad por sí sola.
Sin métricas. Sin evaluación externa.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (SOLO UNO NUEVO: GANANCIA)
# ============================================================
DIM_FREQ = 16
DIM_TIME = 50
DT = 0.01
DURACION = 10.0
N_PASOS = int(DURACION / DT)

# Parámetros de Φ (nuevos)
DIFUSION_BASE = 0.1    # reducida para que la inestabilidad tenga efecto
GANANCIA = 0.5         # intensidad de la inestabilidad (simétrica)

# Parámetros de A (sin cambios)
REFUERZO_A = 0.1
INHIBICION_A = 0.15
DIFUSION_A = 0.05
DISIPACION_A = 0.02
BASAL_A = 0.1

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
print(f"  Simulando primeros {DURACION:.1f} segundos")

# ============================================================
# INICIALIZACIÓN
# ============================================================
def inicializar_campo():
    # Pequeña fluctuación para que la inestabilidad tenga algo que amplificar
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.1 + 0.45  # centrado en 0.5

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

# ============================================================
# PERTURBACIÓN ESTRUCTURADA (sin cambios)
# ============================================================
def aplicar_perturbacion_estructurada(Phi, muestra, paso):
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    banda = int(m * (DIM_FREQ - 1))
    t_idx = paso % DIM_TIME
    intensidad = 0.05 * abs(muestra)
    
    for db in range(-2, 3):
        b = (banda + db) % DIM_FREQ
        peso = np.exp(-(db**2) / 4)
        Phi[b, t_idx] += intensidad * peso
    
    for dt in range(-1, 2):
        t = (t_idx + dt) % DIM_TIME
        for db in range(-1, 2):
            b = (banda + db) % DIM_FREQ
            peso_t = np.exp(-(dt**2) / 2)
            Phi[b, t] += intensidad * 0.3 * peso_t
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# DINÁMICA DE Φ CON INESTABILIDAD SIMÉTRICA
# ============================================================
def actualizar_campo(Phi, perturbacion, paso):
    # 1. Perturbación
    Phi = aplicar_perturbacion_estructurada(Phi, perturbacion, paso)
    
    # 2. Difusión base
    vecinos = (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
               np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4
    difusion = DIFUSION_BASE * (vecinos - Phi)
    
    # 3. Inestabilidad simétrica (amplifica desviaciones del punto medio)
    #    (Phi - 0.5) es la desviación. (1 - (Phi-0.5)^2) controla el crecimiento.
    #    Esto es una ecuación de Landau: puntos fijos en 0.5, inestable para pequeñas fluctuaciones.
    desviacion = Phi - 0.5
    inestabilidad = GANANCIA * desviacion * (1 - desviacion**2)
    
    # 4. Evolución
    Phi = Phi + DT * (difusion + inestabilidad)
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# DINÁMICA DE A (sin cambios)
# ============================================================
def vecinos_inmediatos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4

def actualizar_atencion(A, Phi):
    # Phi no se usa en esta versión (solo el acoplamiento después)
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_inmediatos(A)
    dif = DIFUSION_A * (vecinos_inmediatos(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    dA = auto + inhib + dif + dis
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ACOPLAMIENTO A → Φ (sin cambios)
# ============================================================
def acoplamiento_atencion_campo(Phi, A):
    vecinos = (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
               np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_BASE * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
print("\n" + "=" * 60)
print("SIMULACIÓN VSTCosmo - Campo con Inestabilidad Simétrica")
print("Φ puede volverse inestable por sí mismo")
print("Sin métricas, sin evaluación externa")
print("=" * 60)

Phi = inicializar_campo()
A = inicializar_atencion()

registro = {'A': [], 'Phi': [], 'tiempo': [], 
            'energia_A': [], 'rango_A': [], 'var_A': [],
            'rango_Phi': [], 'var_Phi': []}
momentos = [0, N_PASOS // 4, N_PASOS // 2, 3 * N_PASOS // 4, N_PASOS - 1]

for paso in range(N_PASOS):
    t = paso * DT
    
    # Muestra de audio
    idx = int(t * sr)
    idx = min(idx, len(audio) - 1)
    muestra = audio[idx] if idx >= 0 else 0.0
    
    # 1. Actualizar campo con inestabilidad
    Phi = actualizar_campo(Phi, muestra, paso)
    
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
        registro['var_A'].append(np.var(A))
        registro['rango_Phi'].append(np.max(Phi) - np.min(Phi))
        registro['var_Phi'].append(np.var(Phi))
        print(f"  t={t:.1f}s | rango A={np.max(A)-np.min(A):.4f} | rango Φ={np.max(Phi)-np.min(Phi):.4f}")
    
    if paso % (N_PASOS // 10) == 0:
        print(f"  Progreso: {100 * paso // N_PASOS}%")

print("Simulación completada.\n")

# ============================================================
# OBSERVACIÓN
# ============================================================
print("=" * 60)
print("DOCUMENTACIÓN DE FENÓMENOS")
print("=" * 60)

for i, t in enumerate(registro['tiempo']):
    rango_A = registro['rango_A'][i]
    var_A = registro['var_A'][i]
    rango_Phi = registro['rango_Phi'][i]
    print(f"\nt={t:.1f}s:")
    print(f"  Atención: rango={rango_A:.4f}, var={var_A:.6f}")
    print(f"  Campo Φ:  rango={rango_Phi:.4f}")

print("\n" + "=" * 60)
print("RESUMEN DE FENÓMENOS")
print("=" * 60)

max_rango_A = max(registro['rango_A']) if registro['rango_A'] else 0
max_rango_Phi = max(registro['rango_Phi']) if registro['rango_Phi'] else 0

print(f"Rango máximo de A: {max_rango_A:.4f}")
print(f"Rango máximo de Φ: {max_rango_Phi:.4f}")

if max_rango_A > 0.05:
    print("✓ A comenzó a diferenciarse")
else:
    print("✗ A se mantuvo prácticamente uniforme")

if max_rango_Phi > 0.2:
    print("✓ Φ desarrolló estructura por inestabilidad")
else:
    print("✗ Φ no desarrolló estructura significativa")

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
    
    plt.suptitle('VSTCosmo - Campo con Inestabilidad Simétrica', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_inestabilidad.png', dpi=150)
    print("\nGráfico guardado: vstcosmo_inestabilidad.png")