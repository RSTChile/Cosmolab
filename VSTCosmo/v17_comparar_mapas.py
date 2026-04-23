#!/usr/bin/env python3
"""
v17 - Comparación de mapas entre entradas
Mismos parámetros, diferentes archivos.
Guardamos los mapas finales de Φ y A para cada entrada.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (idénticos a v17)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 30.0
N_PASOS = int(DURACION_SIM / DT)

GANANCIA_ENTRADA = 0.02
DIFUSION_PHI = 0.08
DECAIMIENTO_PHI = 0.04
GANANCIA_TARGET = 0.12
GANANCIA_SOSTENIMIENTO = 0.25
GANANCIA_GENERACION = 0.15

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08

LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35  # ~1120
INHIB_GLOBAL = 0.5

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# ============================================================
# FUNCIONES
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

def inicializar_campo(semilla=42):
    np.random.seed(semilla)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def construir_target(muestra):
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    target_banda = 0.3 + 0.4 * m
    banda = int(m * (DIM_FREQ - 1))
    target = np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.5
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        influencia = np.exp(-(distancia ** 2) / 10.0)
        target[i, :] = target_banda * influencia + 0.5 * (1.0 - influencia)
    return target

def actualizar_campo(Phi, A, muestra):
    target = construir_target(muestra)
    promedio_local = vecinos(Phi)
    
    arrastre_entrada = GANANCIA_TARGET * (target - Phi)
    difusion = DIFUSION_PHI * (promedio_local - Phi)
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local)
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    desviacion = Phi - promedio_local
    generacion = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    entrada_directa = GANANCIA_ENTRADA * muestra
    
    dPhi = arrastre_entrada + difusion + generacion + decaimiento + sostenimiento
    Phi = Phi + DT * dPhi + entrada_directa
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_local = FUERZA_RELIEVE * (relieve_local - A)
    dA = auto + inhib_local + difusion + acoplamiento_local
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def simular(audio, sr, nombre, semilla=42, num_pasos=N_PASOS):
    print(f"    {nombre}...", end=" ", flush=True)
    
    Phi = inicializar_campo(semilla)
    A = inicializar_atencion()
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        A = actualizar_atencion(A, Phi)
        Phi = actualizar_campo(Phi, A, muestra)
    
    rango_phi = float(np.max(Phi) - np.min(Phi))
    rango_a = float(np.max(A) - np.min(A))
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
    return Phi, A

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("VSTCosmo - Comparación de mapas entre entradas")
    print("=" * 70)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    
    entradas = {
        "Silencio": None,
        "Viento_puro": "Viento.wav",
        "Voz_Estudio": "Voz_Estudio.wav",
        "Voz+Viento_real": "Voz+Viento_1.wav",
        "BigBang": "BigBang.wav"
    }
    
    sr_ref = 48000
    resultados = {}
    
    print("\n[2] Ejecutando simulaciones...")
    print("-" * 70)
    
    for nombre, archivo in entradas.items():
        if archivo is None:
            print(f"    {nombre}...", end=" ", flush=True)
            Phi = inicializar_campo(42)
            A = inicializar_atencion()
            audio = np.zeros(int(sr_ref * DURACION_SIM))
            for paso in range(N_PASOS):
                t = paso * DT
                idx = int(t * sr_ref)
                muestra = audio[idx] if idx < len(audio) else 0.0
                A = actualizar_atencion(A, Phi)
                Phi = actualizar_campo(Phi, A, muestra)
            rango_phi = np.max(Phi) - np.min(Phi)
            rango_a = np.max(A) - np.min(A)
            print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
            resultados[nombre] = (Phi, A)
        else:
            sr, audio = cargar_audio(archivo)
            Phi, A = simular(audio, sr, nombre)
            resultados[nombre] = (Phi, A)
    
    # ============================================================
    # VISUALIZACIÓN DE MAPAS
    # ============================================================
    print("\n[3] Generando mapas comparativos...")
    
    n_entradas = len(resultados)
    fig, axes = plt.subplots(2, n_entradas, figsize=(4*n_entradas, 8))
    
    for i, (nombre, (Phi, A)) in enumerate(resultados.items()):
        # Campo Φ
        im1 = axes[0, i].imshow(Phi, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0, i].set_title(f'{nombre}\nΦ (rango={np.max(Phi)-np.min(Phi):.3f})')
        axes[0, i].set_xlabel('Memoria')
        axes[0, i].set_ylabel('Banda')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Atención A
        im2 = axes[1, i].imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'Atención A (rango={np.max(A)-np.min(A):.4f})')
        axes[1, i].set_xlabel('Memoria')
        axes[1, i].set_ylabel('Banda')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle('VSTCosmo v17 - Comparativa de mapas (misma semilla 42)', fontsize=14)
    plt.tight_layout()
    plt.savefig('v17_mapas_comparativos.png', dpi=150)
    print("    Gráfico guardado: v17_mapas_comparativos.png")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()