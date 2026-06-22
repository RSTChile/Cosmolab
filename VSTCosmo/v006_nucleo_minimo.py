#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación VSTCosmo - Núcleo Mínimo
Basada en la síntesis canónica del problema.

No separa voz de ruido.
No presupone relevancia.
No usa métricas de éxito.
Solo observa si emerge organización diferencial.

Entrada: grabación real (Voz+Viento_1.wav)
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS FIJOS (arbitrarios, no optimizados)
# ============================================================
DIM_FREQ = 32          # bandas (sin correspondencia física fija)
DIM_TIME = 100         # pasos de memoria (aprox 1 segundo a 100 Hz)
DT = 0.01              # 10 ms por paso (100 Hz)
DURACION = None        # se define al cargar el audio
HOP = 10               # muestras por paso (simplificado)

# Parámetros de dinámica (solo para que el sistema no colapse)
D_PHI = 0.1            # difusión del campo
REFUERZO_A = 0.3       # sensibilidad al cambio
DIFUSION_A = 0.2       # propagación de atención
DISIPACION_A = 0.05    # decaimiento de atención
BASAL_A = 0.1          # valor basal de atención
ACOPLAMIENTO_S = 0.01  # retroalimentación de salida

# ============================================================
# CARGA DE AUDIO REAL (sin preprocesamiento)
# ============================================================
def cargar_audio(ruta):
    """Carga audio WAV, convierte a mono, normaliza sin modificar la relación señal/ruido."""
    sr, data = wav.read(ruta)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    return sr, data

# Cargar la grabación
print("Cargando audio: Voz+Viento_1.wav")
sr, audio = cargar_audio('Voz+Viento_1.wav')
print(f"  Frecuencia: {sr} Hz")
print(f"  Duración: {len(audio)/sr:.2f} s")
print(f"  Muestras: {len(audio)}")

# Configurar duración de simulación (usar primeros segundos para pruebas)
DURACION = min(10.0, len(audio) / sr)  # máx 10 segundos
N_PASOS = int(DURACION / DT)
print(f"  Simulando primeros {DURACION:.1f} segundos ({N_PASOS} pasos)")
print()

# ============================================================
# CAMPO Φ (medio físico abstracto)
# ============================================================
def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.01

def aplicar_perturbacion(Phi, muestra):
    """
    La perturbación (muestra de audio) entra distribuida, sin privilegio.
    No se asigna a una celda "especial". Se suma proporcionalmente a todo el campo.
    """
    # La perturbación modula la intensidad de entrada, sin dirección preferencial
    factor = 0.1 * muestra
    Phi += factor * (np.random.rand(DIM_FREQ, DIM_TIME) * 0.5 + 0.5)
    return Phi

def actualizar_campo(Phi, perturbacion):
    # 1. Aplicar perturbación distribuida
    Phi = aplicar_perturbacion(Phi, perturbacion)
    
    # 2. Difusión local (acoplamiento espacial)
    Phi_nuevo = 0.9 * Phi + 0.1 * (
        np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
        np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)
    ) / 4
    
    # 3. Conservación (normalización para que RC no desaparezca)
    suma = np.sum(Phi_nuevo)
    if suma > 0:
        Phi_nuevo = Phi_nuevo / suma
    
    return Phi_nuevo

# ============================================================
# ATENCIÓN A (coevoluciona, sin objetivo)
# ============================================================
def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

def actualizar_atencion(A, Phi, Phi_prev):
    # 1. Respuesta al cambio local (no es "coherencia", solo cambio)
    cambio = np.abs(Phi - Phi_prev)
    refuerzo = REFUERZO_A * cambio * (1 - A)
    
    # 2. Difusión (propagación local)
    difusion = DIFUSION_A * (
        np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) - 2 * A
    )
    
    # 3. Disipación hacia valor basal
    disipacion = -DISIPACION_A * (A - BASAL_A)
    
    # 4. Evolución
    dA = refuerzo + difusion + disipacion
    A_nuevo = A + DT * dA
    A_nuevo = np.clip(A_nuevo, 0.0, 1.0)
    
    return A_nuevo

# ============================================================
# SALIDA Y ACOPLAMIENTO CERRADO
# ============================================================
def generar_salida(Phi, A):
    """Proyección simple: suma ponderada de campo por atención."""
    return np.sum(Phi * A)

def aplicar_acoplamiento(Phi, S):
    """La salida retroalimenta al campo (cierre del ciclo)."""
    Phi += ACOPLAMIENTO_S * S * (0.5 - Phi)
    suma = np.sum(Phi)
    if suma > 0:
        Phi = Phi / suma
    return Phi

# ============================================================
# GENERACIÓN DE PERTURBACIÓN DESDE AUDIO REAL
# ============================================================
def generar_perturbacion_desde_audio(audio, sr, paso, dt, hop_muestras):
    """
    Toma una muestra del audio real y la convierte en perturbación.
    No se extraen "features". Se usa directamente el valor de amplitud.
    """
    # Mapear tiempo de simulación a índice de muestra
    t = paso * dt
    idx = int(t * sr)
    if idx < len(audio):
        return audio[idx]
    else:
        return 0.0

# Calcular hop en muestras
HOP_MUESTRAS = int(DT * sr)
print(f"  Hop: {HOP_MUESTRAS} muestras por paso (~{HOP_MUESTRAS/sr*1000:.1f} ms)")

# ============================================================
# SIMULACIÓN
# ============================================================
def simular(audio, sr, n_pasos):
    print("=" * 60)
    print("SIMULACIÓN VSTCosmo - Núcleo Mínimo")
    print("Observando si emerge organización diferencial")
    print("=" * 60)
    print()
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    # Registro para observación (sin evaluación)
    registro = {
        'A': [],
        'Phi': [],
        'tiempo': [],
        'salida': []
    }
    
    # También guardamos momentos específicos para documentar cambios
    momentos = [0, N_PASOS // 4, N_PASOS // 2, 3 * N_PASOS // 4, N_PASOS - 1]
    
    for paso in range(n_pasos):
        t = paso * DT
        
        # 1. Perturbación desde audio real
        perturbacion = generar_perturbacion_desde_audio(audio, sr, paso, DT, HOP_MUESTRAS)
        
        # 2. Actualizar campo
        Phi = actualizar_campo(Phi, perturbacion)
        
        # 3. Actualizar atención
        A = actualizar_atencion(A, Phi, Phi_prev)
        
        # 4. Generar salida
        S = generar_salida(Phi, A)
        
        # 5. Acoplamiento (cerrar ciclo)
        Phi = aplicar_acoplamiento(Phi, S)
        
        # 6. Registrar momentos clave
        if paso in momentos:
            registro['A'].append(A.copy())
            registro['Phi'].append(Phi.copy())
            registro['tiempo'].append(t)
            registro['salida'].append(S)
            print(f"  Momento {paso}/{n_pasos} (t={t:.1f}s) registrado")
        
        Phi_prev = Phi.copy()
        
        # Progreso
        if paso % (n_pasos // 10) == 0:
            pct = 100 * paso / n_pasos
            print(f"  Progreso: {pct:.0f}%")
    
    print()
    print("Simulación completada.")
    return registro

# ============================================================
# OBSERVACIÓN (sin evaluación, solo documentación)
# ============================================================
def observar(registro):
    """Documentar fenómenos observados, sin juzgar éxito o fracaso."""
    
    print("=" * 60)
    print("DOCUMENTACIÓN DE FENÓMENOS OBSERVADOS")
    print("=" * 60)
    print()
    
    n_momentos = len(registro['A'])
    if n_momentos == 0:
        print("No se registraron datos.")
        return
    
    # Documentar evolución de la atención
    print("Atención A (mapa de relevancia interna):")
    print("-" * 40)
    for i, (A, t) in enumerate(zip(registro['A'], registro['tiempo'])):
        media = np.mean(A)
        varianza = np.var(A)
        maximo = np.max(A)
        minimo = np.min(A)
        print(f"  t={t:.1f}s: media={media:.3f}, var={varianza:.5f}, max={maximo:.3f}, min={minimo:.3f}")
        
        # Observar si hay regiones diferenciadas
        if maximo > media * 2:
            print(f"    → Posible región de atención concentrada detectada")
    
    print()
    print("Campo Φ (medio interno):")
    print("-" * 40)
    for i, (Phi, t) in enumerate(zip(registro['Phi'], registro['tiempo'])):
        media = np.mean(Phi)
        varianza = np.var(Phi)
        print(f"  t={t:.1f}s: media={media:.3f}, var={varianza:.5f}")
    
    print()
    print("Salida S (proyección):")
    print("-" * 40)
    for i, (S, t) in enumerate(zip(registro['salida'], registro['tiempo'])):
        print(f"  t={t:.1f}s: S={S:.4f}")
    
    # Observaciones cualitativas
    print()
    print("=" * 60)
    print("OBSERVACIONES CUALITATIVAS")
    print("=" * 60)
    
    # ¿Aparecieron regiones diferenciadas?
    concentraciones = []
    for A in registro['A']:
        concentraciones.append(np.max(A) / (np.mean(A) + 1e-6))
    
    if max(concentraciones) > 2.0:
        print("✓ Se observaron regiones de atención concentrada (max/media > 2)")
    else:
        print("✗ No se observaron regiones claramente concentradas")
    
    # ¿La atención se reorganizó?
    varianzas = [np.var(A) for A in registro['A']]
    if max(varianzas) - min(varianzas) > 0.01:
        print("✓ La atención mostró reorganización (varianza cambió significativamente)")
    else:
        print("✗ La atención no mostró reorganización clara")
    
    # ¿El campo conservó actividad?
    energias = [np.sum(Phi) for Phi in registro['Phi']]
    if np.mean(energias) > 0.1:
        print("✓ El campo mantuvo actividad (energía media > 0.1)")
    else:
        print("✗ El campo perdió actividad")
    
    print()
    print("NOTA: Estas observaciones no son 'éxito' o 'fracaso'.")
    print("Solo documentan lo que apareció en esta simulación particular.")
    print("El sistema no busca 'separar' ni 'reconocer' nada.")
    print("Solo se observa si emerge organización diferencial.")

def visualizar(registro):
    """Generar gráficos para documentar visualmente."""
    if len(registro['A']) == 0:
        return
    
    fig, axes = plt.subplots(2, len(registro['A']), figsize=(4*len(registro['A']), 8))
    if len(registro['A']) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (A, Phi, t) in enumerate(zip(registro['A'], registro['Phi'], registro['tiempo'])):
        # Atención
        ax = axes[0, i]
        im = ax.imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Atención t={t:.1f}s')
        ax.set_xlabel('Memoria (tiempo)')
        ax.set_ylabel('Banda (sin correspondencia)')
        plt.colorbar(im, ax=ax)
        
        # Campo
        ax = axes[1, i]
        im = ax.imshow(Phi, aspect='auto', cmap='viridis')
        ax.set_title(f'Campo Φ t={t:.1f}s')
        ax.set_xlabel('Memoria (tiempo)')
        ax.set_ylabel('Banda')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('VSTCosmo - Documentación de Fenómenos', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_observacion.png', dpi=150)
    print()
    print("Gráfico guardado: vstcosmo_observacion.png")

# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    try:
        registro = simular(audio, sr, N_PASOS)
        observar(registro)
        visualizar(registro)
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        import traceback
        traceback.print_exc()