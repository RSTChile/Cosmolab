#!/usr/bin/env python3
"""
VSTCosmo - Experimento v7
Campo con inestabilidad simétrica + Atención sensible a estructura local de Φ

Características:
- Φ: inestabilidad tipo Landau (amplifica desviaciones simétricamente)
- A: sensible a varianza local de Φ (sin métricas de éxito)
- Entrada: audio real estructurado por amplitud
- Sin señales externas de "acoplamiento bueno/malo"
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS
# ============================================================
DIM_FREQ = 16
DIM_TIME = 50
DT = 0.01
DURACION = 10.0
N_PASOS = int(DURACION / DT)

# Parámetros de Φ
DIFUSION_PHI = 0.1      # difusión base
GANANCIA_INEST = 0.5    # intensidad de inestabilidad simétrica

# Parámetros de A
REFUERZO_A = 0.1
INHIBICION_A = 0.15
DIFUSION_A = 0.05
DISIPACION_A = 0.02
BASAL_A = 0.1

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

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

print("=" * 60)
print("VSTCosmo - Experimento v7")
print("Campo con inestabilidad simétrica + Atención sensible a Φ")
print("=" * 60)

print("\n[1] Cargando audio...")
sr, audio = cargar_audio('Voz+Viento_1.wav')
print(f"    Frecuencia: {sr} Hz")
print(f"    Duración: {len(audio)/sr:.2f} s")
print(f"    Simulando primeros {DURACION:.1f} segundos ({N_PASOS} pasos)")

# ============================================================
# INICIALIZACIÓN
# ============================================================
def inicializar_campo():
    """Φ centrado en 0.5 con pequeña fluctuación"""
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.1 + 0.45

def inicializar_atencion():
    """A uniforme en valor basal"""
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

# ============================================================
# PERTURBACIÓN ESTRUCTURADA POR AMPLITUD
# ============================================================
def aplicar_perturbacion(Phi, muestra, paso):
    """La entrada entra en regiones del campo según la amplitud"""
    # Normalizar muestra a [0,1]
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    
    # Mapeo amplitud → banda
    banda = int(m * (DIM_FREQ - 1))
    t_idx = paso % DIM_TIME
    intensidad = 0.05 * abs(muestra)
    
    # Excitar banda central y vecinos
    for db in range(-2, 3):
        b = (banda + db) % DIM_FREQ
        peso = np.exp(-(db**2) / 4)
        Phi[b, t_idx] += intensidad * peso
    
    # Propagación temporal
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
def vecinos_phi(Phi):
    """Promedio de vecinos inmediatos para Φ"""
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def actualizar_campo(Phi, muestra, paso):
    # 1. Perturbación
    Phi = aplicar_perturbacion(Phi, muestra, paso)
    
    # 2. Difusión base
    vecinos = vecinos_phi(Phi)
    difusion = DIFUSION_PHI * (vecinos - Phi)
    
    # 3. Inestabilidad simétrica (Landau)
    #    Amplifica desviaciones del punto medio 0.5 sin preferir dirección
    desviacion = Phi - 0.5
    inestabilidad = GANANCIA_INEST * desviacion * (1 - desviacion**2)
    
    # 4. Evolución
    Phi = Phi + DT * (difusion + inestabilidad)
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# DINÁMICA DE A (ahora sensible a estructura local de Φ)
# ============================================================
def vecinos_a(A):
    """Promedio de vecinos inmediatos para A"""
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def varianza_local(Phi, radio=1):
    """Calcula varianza local de Φ como modulador (sin semántica)"""
    var_local = np.zeros_like(Phi)
    for i in range(radio, DIM_FREQ - radio):
        for j in range(radio, DIM_TIME - radio):
            ventana = Phi[i-radio:i+radio+1, j-radio:j+radio+1]
            var_local[i, j] = np.var(ventana)
    # Bordes: copiar de vecinos más cercanos
    for i in range(DIM_FREQ):
        for j in range(DIM_TIME):
            if var_local[i, j] == 0:
                # Buscar vecino más cercano con valor no cero
                for r in range(1, min(DIM_FREQ, DIM_TIME)):
                    encontrado = False
                    for di in [-r, r]:
                        for dj in [-r, r]:
                            ii, jj = i+di, j+dj
                            if 0 <= ii < DIM_FREQ and 0 <= jj < DIM_TIME:
                                if var_local[ii, jj] > 0:
                                    var_local[i, j] = var_local[ii, jj]
                                    encontrado = True
                                    break
                        if encontrado:
                            break
                    if encontrado:
                        break
    return var_local

def actualizar_atencion(A, Phi):
    """
    A evoluciona con reglas locales, modulada por varianza local de Φ.
    Donde Φ tiene más estructura, la dinámica de A es más rápida.
    Esto NO es 'seguir lo importante'. Es solo 'ser sensible'.
    """
    # Términos base
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    # Modulador por estructura local de Φ (varianza)
    var_local = varianza_local(Phi, radio=1)
    # Normalizar varianza para que sea factor entre 1 y 3
    max_var = np.max(var_local)
    if max_var > 0:
        modulador = 1 + 2 * var_local / max_var
    else:
        modulador = np.ones_like(Phi)
    
    # Aplicar modulación
    dA = dA_base * modulador
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ACOPLAMIENTO A → Φ
# ============================================================
def acoplamiento_atencion_campo(Phi, A):
    """A modula la mezcla entre Φ y sus vecinos (sin dirección semántica)"""
    vecinos = vecinos_phi(Phi)
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_ACOPLAMIENTO * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
print("\n[2] Ejecutando simulación...")
print("    Φ: difusión + inestabilidad simétrica (Landau)")
print("    A: auto-refuerzo + inhibición + difusión + disipación")
print("    A ← Φ: modulación por varianza local (sin métricas)")
print("    A → Φ: modulación de mezcla\n")

Phi = inicializar_campo()
A = inicializar_atencion()

# Registro para observación
registro = {
    'A': [], 'Phi': [], 'tiempo': [],
    'rango_A': [], 'var_A': [], 'media_A': [],
    'rango_Phi': [], 'var_Phi': [], 'media_Phi': []
}
momentos = [0, N_PASOS // 4, N_PASOS // 2, 3 * N_PASOS // 4, N_PASOS - 1]

for paso in range(N_PASOS):
    t = paso * DT
    
    # Muestra de audio
    idx = int(t * sr)
    idx = min(idx, len(audio) - 1)
    muestra = audio[idx] if idx >= 0 else 0.0
    
    # 1. Actualizar campo Φ
    Phi = actualizar_campo(Phi, muestra, paso)
    
    # 2. Actualizar atención A (ahora sensible a Φ)
    A = actualizar_atencion(A, Phi)
    
    # 3. Acoplamiento A → Φ
    Phi = acoplamiento_atencion_campo(Phi, A)
    
    # Registrar momentos clave
    if paso in momentos:
        registro['A'].append(A.copy())
        registro['Phi'].append(Phi.copy())
        registro['tiempo'].append(t)
        registro['rango_A'].append(np.max(A) - np.min(A))
        registro['var_A'].append(np.var(A))
        registro['media_A'].append(np.mean(A))
        registro['rango_Phi'].append(np.max(Phi) - np.min(Phi))
        registro['var_Phi'].append(np.var(Phi))
        registro['media_Phi'].append(np.mean(Phi))
    
    # Progreso
    if paso % (N_PASOS // 10) == 0:
        pct = 100 * paso // N_PASOS
        print(f"    Progreso: {pct}% | rango A={np.max(A)-np.min(A):.4f} | rango Φ={np.max(Phi)-np.min(Phi):.4f}")

print("\n[3] Simulación completada.")

# ============================================================
# DOCUMENTACIÓN DE FENÓMENOS
# ============================================================
print("\n" + "=" * 60)
print("DOCUMENTACIÓN DE FENÓMENOS")
print("=" * 60)

print("\nEvolución temporal:")
print("-" * 50)
print(f"{'t(s)':>6} | {'rango A':>10} | {'var A':>12} | {'rango Φ':>10} | {'var Φ':>12}")
print("-" * 50)

for i, t in enumerate(registro['tiempo']):
    print(f"{t:6.1f} | {registro['rango_A'][i]:10.4f} | {registro['var_A'][i]:12.6f} | "
          f"{registro['rango_Phi'][i]:10.4f} | {registro['var_Phi'][i]:12.6f}")

print("\n" + "=" * 60)
print("RESUMEN DE FENÓMENOS")
print("=" * 60)

max_rango_A = max(registro['rango_A']) if registro['rango_A'] else 0
max_rango_Phi = max(registro['rango_Phi']) if registro['rango_Phi'] else 0
max_var_A = max(registro['var_A']) if registro['var_A'] else 0
max_var_Phi = max(registro['var_Phi']) if registro['var_Phi'] else 0

print(f"\nRango máximo de A:     {max_rango_A:.6f}")
print(f"Varianza máxima de A:  {max_var_A:.6f}")
print(f"Rango máximo de Φ:     {max_rango_Phi:.6f}")
print(f"Varianza máxima de Φ:  {max_var_Phi:.6f}")

if max_rango_A > 0.05:
    print("\n✓ La atención desarrolló regiones diferenciadas")
else:
    print("\n✗ La atención se mantuvo prácticamente uniforme")

if max_rango_Phi > 0.2:
    print("✓ El campo Φ desarrolló estructura por inestabilidad")
else:
    print("✗ El campo Φ no desarrolló estructura significativa")

if max_rango_A > 0.05 and max_rango_Phi > 0.2:
    print("\n★ Posible acoplamiento: A sigue la estructura de Φ")
elif max_rango_Phi > 0.2 and max_rango_A <= 0.05:
    print("\n⚠ Φ tiene estructura pero A es ciega a ella")
elif max_rango_A > 0.05 and max_rango_Phi <= 0.2:
    print("\n⚠ A se diferenció pero Φ es homogéneo")

# ============================================================
# VISUALIZACIÓN
# ============================================================
print("\n[4] Generando visualización...")

if len(registro['A']) > 0:
    fig, axes = plt.subplots(2, len(registro['A']), figsize=(4*len(registro['A']), 7))
    if len(registro['A']) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (A, Phi, t) in enumerate(zip(registro['A'], registro['Phi'], registro['tiempo'])):
        # Atención
        im1 = axes[0, i].imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Atención A t={t:.1f}s')
        axes[0, i].set_xlabel('Memoria temporal')
        axes[0, i].set_ylabel('Banda (frecuencia análoga)')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Campo
        im2 = axes[1, i].imshow(Phi, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Campo Φ t={t:.1f}s')
        axes[1, i].set_xlabel('Memoria temporal')
        axes[1, i].set_ylabel('Banda')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle('VSTCosmo v7 - Inestabilidad Simétrica + Atención Sensible a Φ', fontsize=14)
    plt.tight_layout()
    
    # Guardar con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'vstcosmo_v7_resultado_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    print(f"    Gráfico guardado: {filename}")

# ============================================================
# CIERRE
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENTO COMPLETADO")
print("=" * 60)
print("\nEste experimento implementa:")
print("  ✓ Φ con inestabilidad simétrica (Landau)")
print("  ✓ A sensible a varianza local de Φ (sin métricas)")
print("  ✓ Entrada estructurada por amplitud")
print("  ✓ Acoplamiento A → Φ")
print("\nNo se utilizan métricas de éxito externas.")
print("Solo se observa si emerge organización diferencial.")
print("=" * 60)