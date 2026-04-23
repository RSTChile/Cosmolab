#!/usr/bin/env python3
"""
VSTCosmo - v12_caracterizacion_mezcla.py
Barrido de mezcla Voz:Viento en diferentes proporciones.
Misma dinámica que v12, sin tocar A.
Objetivo: encontrar el punto óptimo de apertura (ecotono).
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (idénticos a v12)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION = 20.0
N_PASOS = int(DURACION / DT)

GANANCIA_INEST = 0.15
DIFUSION_PHI = 0.1
SENSIBILIDAD_ENTRADA = 0.15

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

DIFUSION_ACOPLAMIENTO = 0.2

LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

# ============================================================
# FUNCIONES (idénticas a v12)
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

def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def actualizar_campo_permeable(Phi, muestra):
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    target_banda = 0.3 + 0.4 * m
    m_banda = int(m * (DIM_FREQ - 1))
    target = np.ones_like(Phi) * 0.5
    
    for i in range(DIM_FREQ):
        distancia = min(abs(i - m_banda), DIM_FREQ - abs(i - m_banda))
        influencia = np.exp(-distancia**2 / 10)
        target[i] = target_banda * influencia + 0.5 * (1 - influencia)
    
    desviacion = Phi - target
    inestabilidad = GANANCIA_INEST * desviacion * (1 - desviacion**2)
    vecinos = vecinos_phi(Phi)
    difusion = DIFUSION_PHI * (vecinos - Phi)
    entrada_directa = 0.02 * muestra
    
    Phi = Phi + DT * (inestabilidad + difusion) + entrada_directa
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def actualizar_atencion(A, Phi, Phi_prev):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    grad_temporal = Phi - Phi_prev
    prop = 0.02 * np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    prop += 0.01 * np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
    dA = dA_base + prop
    dA += np.random.randn(*A.shape) * 0.001
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def acoplamiento_atencion_campo(Phi, A):
    vecinos = vecinos_phi(Phi)
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_ACOPLAMIENTO * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def simular(audio, sr, nombre, num_pasos=N_PASOS):
    print(f"    {nombre}...", end=" ", flush=True)
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        Phi = actualizar_campo_permeable(Phi, muestra)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        Phi_prev = Phi.copy()
    
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
    return rango_phi, rango_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Caracterización: Barrido de Mezcla")
    print("Misma dinámica que v12. Variando proporción Voz:Viento")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr_v, voz = cargar_audio('Voz_Estudio.wav')
    sr_w, viento = cargar_audio('Viento.wav')
    
    # Recortar a la misma duración
    min_len = min(len(voz), len(viento))
    voz = voz[:min_len]
    viento = viento[:min_len]
    
    print(f"    Duración: {min_len/sr_v:.1f}s")
    
    # Proporciones a probar
    proporciones = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # También añadir algunos puntos más finos cerca del centro
    proporciones_fino = [0.45, 0.55]
    proporciones = sorted(set(proporciones + proporciones_fino))
    
    resultados_phi = []
    resultados_a = []
    
    print("\n[2] Ejecutando barrido...")
    print("-" * 50)
    
    for prop in proporciones:
        nombre = f"{prop*100:.0f}% voz"
        # Mezclar
        mezcla = (1 - prop) * viento + prop * voz
        rango_phi, rango_a = simular(mezcla, sr_v, nombre)
        resultados_phi.append(rango_phi)
        resultados_a.append(rango_a)
    
    print("-" * 50)
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[3] Generando visualización...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de rango Φ
    ax1.plot([p*100 for p in proporciones], resultados_phi, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.85, color='r', linestyle='--', label='Umbral saturación')
    ax1.axhline(y=0.3, color='g', linestyle='--', label='Umbral homogeneidad')
    ax1.set_xlabel('Proporción de voz (%)')
    ax1.set_ylabel('rango Φ (estructura del campo)')
    ax1.set_title('Apertura del campo vs mezcla voz:viento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de rango A
    ax2.plot([p*100 for p in proporciones], resultados_a, 'r-s', linewidth=2, markersize=8)
    ax2.axhline(y=0.01, color='orange', linestyle='--', label='Umbral atención mínima')
    ax2.set_xlabel('Proporción de voz (%)')
    ax2.set_ylabel('rango A (diferenciación de atención)')
    ax2.set_title('Atención vs mezcla')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v12 - Caracterización: Barrido de Mezcla', fontsize=14)
    plt.tight_layout()
    plt.savefig('v12_barrido_mezcla.png', dpi=150)
    print("    Gráfico guardado: v12_barrido_mezcla.png")
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL BARRIDO")
    print("=" * 60)
    
    # Encontrar región de apertura (rango Φ entre 0.3 y 0.85)
    region_abierta = []
    for i, (prop, rphi) in enumerate(zip(proporciones, resultados_phi)):
        if 0.3 < rphi < 0.85:
            region_abierta.append((prop, rphi))
    
    if region_abierta:
        print(f"\nRegión de apertura encontrada:")
        for prop, rphi in region_abierta:
            print(f"  {prop*100:.0f}% voz : rango Φ = {rphi:.3f}")
        
        # Mejor punto (rango Φ más cercano a 0.5-0.6)
        mejor = min(region_abierta, key=lambda x: abs(x[1] - 0.55))
        print(f"\n★ Punto óptimo aproximado: {mejor[0]*100:.0f}% voz (rango Φ={mejor[1]:.3f})")
    else:
        print("\n✗ No se encontró región de apertura clara en este barrido")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()