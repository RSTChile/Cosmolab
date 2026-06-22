#!/usr/bin/env python3
"""
VSTCosmo - Estabilidad del régimen abierto
Ejecutamos Voz+Viento_real 10 veces con diferentes condiciones iniciales.
Observamos si el rango Φ y rango A se mantienen estables o fluctúan.
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
DURACION_SIM = 30.0
N_PASOS = int(DURACION_SIM / DT)

GANANCIA_INEST = 0.15
DIFUSION_PHI = 0.1

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

DIFUSION_ACOPLAMIENTO = 0.2
LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

# ============================================================
# FUNCIONES BASE
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

def inicializar_campo(semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion(semilla=None):
    if semilla is not None:
        np.random.seed(semilla + 1000)  # offset para diferenciar
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

def simular(audio, sr, nombre, semilla=None, num_pasos=N_PASOS):
    Phi = inicializar_campo(semilla)
    A = inicializar_atencion(semilla)
    Phi_prev = Phi.copy()
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        Phi = actualizar_campo_permeable(Phi, muestra)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        Phi_prev = Phi.copy()
    
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    return rango_phi, rango_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Estabilidad del régimen abierto")
    print("Ejecutando Voz+Viento_real 10 veces con diferentes semillas")
    print("=" * 60)
    
    # Cargar audio
    print("\n[1] Cargando Voz+Viento_1.wav...")
    sr, audio = cargar_audio('Voz+Viento_1.wav')
    print(f"    Duración: {len(audio)/sr:.1f}s")
    print(f"    Simulando {DURACION_SIM:.0f} segundos por ejecución")
    
    # Semillas aleatorias
    semillas = [42, 123, 987, 1, 2, 3, 100, 200, 300, 400]
    
    print("\n[2] Ejecutando 10 simulaciones...")
    print("-" * 50)
    
    resultados_phi = []
    resultados_a = []
    
    for i, semilla in enumerate(semillas):
        print(f"    Ejecución {i+1}/10 (semilla={semilla})...", end=" ", flush=True)
        rphi, ra = simular(audio, sr, f"Run_{i}", semilla=semilla)
        resultados_phi.append(rphi)
        resultados_a.append(ra)
        print(f"rango Φ={rphi:.3f}, rango A={ra:.4f}")
    
    print("-" * 50)
    
    # ============================================================
    # ANÁLISIS ESTADÍSTICO
    # ============================================================
    print("\n[3] Análisis estadístico")
    print("=" * 60)
    
    media_phi = np.mean(resultados_phi)
    std_phi = np.std(resultados_phi)
    min_phi = np.min(resultados_phi)
    max_phi = np.max(resultados_phi)
    
    media_a = np.mean(resultados_a)
    std_a = np.std(resultados_a)
    min_a = np.min(resultados_a)
    max_a = np.max(resultados_a)
    
    print(f"\n  rango Φ:")
    print(f"    media  = {media_phi:.4f}")
    print(f"    std    = {std_phi:.4f}")
    print(f"    min    = {min_phi:.4f}")
    print(f"    max    = {max_phi:.4f}")
    print(f"    rango  = {max_phi - min_phi:.4f}")
    
    print(f"\n  rango A:")
    print(f"    media  = {media_a:.4f}")
    print(f"    std    = {std_a:.4f}")
    print(f"    min    = {min_a:.4f}")
    print(f"    max    = {max_a:.4f}")
    print(f"    rango  = {max_a - min_a:.4f}")
    
    # ============================================================
    # INTERPRETACIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN")
    print("=" * 60)
    
    if std_phi < 0.05:
        print("\n  ✓ El rango Φ es MUY estable (std < 0.05)")
        print("    → El sistema tiene un atractor fuerte para esta entrada")
    elif std_phi < 0.1:
        print("\n  ~ El rango Φ es moderadamente estable (std < 0.1)")
        print("    → El atractor existe pero con cierta variabilidad")
    else:
        print("\n  ✗ El rango Φ es inestable (std > 0.1)")
        print("    → El sistema es sensible a condiciones iniciales")
    
    if std_a < 0.002:
        print("  ✓ El rango A es muy estable (std < 0.002)")
    elif std_a < 0.005:
        print("  ~ El rango A es moderadamente estable")
    else:
        print("  ✗ El rango A es inestable")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de rango Φ
    ax1.plot(range(1, 11), resultados_phi, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=media_phi, color='r', linestyle='--', label=f'media = {media_phi:.3f}')
    ax1.fill_between(range(1, 11), media_phi - std_phi, media_phi + std_phi, 
                      alpha=0.2, color='r', label=f'±1σ = {std_phi:.3f}')
    ax1.set_xlabel('Ejecución')
    ax1.set_ylabel('rango Φ')
    ax1.set_title('Estabilidad del campo (Φ)')
    ax1.set_ylim(0.3, 0.6)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de rango A
    ax2.plot(range(1, 11), resultados_a, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=media_a, color='r', linestyle='--', label=f'media = {media_a:.4f}')
    ax2.fill_between(range(1, 11), media_a - std_a, media_a + std_a, 
                      alpha=0.2, color='r', label=f'±1σ = {std_a:.4f}')
    ax2.set_xlabel('Ejecución')
    ax2.set_ylabel('rango A')
    ax2.set_title('Estabilidad de la atención (A)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo - Estabilidad con Voz+Viento_real (10 ejecuciones, diferentes semillas)', fontsize=14)
    plt.tight_layout()
    plt.savefig('v12_estabilidad_voz_viento.png', dpi=150)
    print("    Gráfico guardado: v12_estabilidad_voz_viento.png")
    
    # ============================================================
    # CONCLUSIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    
    if std_phi < 0.05 and std_a < 0.002:
        print("\n  El sistema es notablemente ESTABLE.")
        print("  Voz+Viento_real produce un régimen consistente.")
        print("  La apertura no es un artefacto. Es un ATractor.")
    elif std_phi < 0.1:
        print("\n  El sistema es moderadamente estable.")
        print("  Hay un atractor, pero con cierta sensibilidad.")
    else:
        print("\n  El sistema es inestable.")
        print("  La apertura depende de condiciones iniciales.")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()