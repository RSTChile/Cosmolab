#!/usr/bin/env python3
"""
VSTCosmo - v15: Competencia Global
La atención tiene un límite total. Las regiones compiten por persistencia.
No hay métrica de "qué es mejor". La selección emerge.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS
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

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

# Acoplamiento Φ → A (simétrico, pero ahora con competencia)
FUERZA_ACOPLAMIENTO = 0.1

# Competencia global
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.3  # ~960
INHIB_GLOBAL = 0.5

LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

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

def inicializar_campo(semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * 0.1

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

def actualizar_atencion(A, Phi):
    # Términos locales
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    
    # Acoplamiento a Φ (simétrico)
    acoplamiento = FUERZA_ACOPLAMIENTO * (Phi - A)
    
    dA = auto + inhib + dif + acoplamiento
    
    # NUEVO: Competencia global
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        inhibicion_global = -INHIB_GLOBAL * exceso * A
        dA += inhibicion_global
    
    # Ruido mínimo
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
    print(f"    {nombre}...", end=" ", flush=True)
    
    Phi = inicializar_campo(semilla)
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        Phi = actualizar_campo_permeable(Phi, muestra)
        A = actualizar_atencion(A, Phi)
        Phi = acoplamiento_atencion_campo(Phi, A)
        Phi_prev = Phi.copy()
    
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    media_a = np.mean(A)
    total_a = np.sum(A)
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}, media A={media_a:.4f}, total A={total_a:.0f}")
    return rango_phi, rango_a, media_a, total_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - v15: Competencia Global")
    print("La atención tiene un límite total. Las regiones compiten.")
    print("No hay métrica de 'qué es mejor'. La selección emerge.")
    print("=" * 60)
    
    print("\n[1] Cargando archivos...")
    sr, voz_viento = cargar_audio('Voz+Viento_1.wav')
    print(f"    Voz+Viento_1.wav: {len(voz_viento)/sr:.1f}s")
    
    semillas = [42, 123, 987, 1, 2, 3, 100, 200, 300, 400]
    
    print("\n[2] Ejecutando 10 simulaciones con Voz+Viento_real...")
    print("-" * 70)
    
    resultados_phi = []
    resultados_a = []
    resultados_total_a = []
    
    for i, semilla in enumerate(semillas):
        rphi, ra, media_a, total_a = simular(voz_viento, sr, f"Run_{i}", semilla=semilla)
        resultados_phi.append(rphi)
        resultados_a.append(ra)
        resultados_total_a.append(total_a)
    
    print("-" * 70)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n[3] Análisis estadístico")
    print("=" * 60)
    
    media_phi = np.mean(resultados_phi)
    std_phi = np.std(resultados_phi)
    media_a = np.mean(resultados_a)
    std_a = np.std(resultados_a)
    media_total = np.mean(resultados_total_a)
    std_total = np.std(resultados_total_a)
    
    print(f"\n  rango Φ: media={media_phi:.3f}, std={std_phi:.4f}")
    print(f"  rango A: media={media_a:.4f}, std={std_a:.4f}")
    print(f"  total A: media={media_total:.0f}, std={std_total:.0f}")
    
    # ============================================================
    # INTERPRETACIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN")
    print("=" * 60)
    
    if media_total < LIMITE_ATENCION * 0.9:
        print("\n  ✓ La atención total está por debajo del límite.")
        print("    No hay inhibición global activa.")
    else:
        print("\n  ⚠ La atención total está cerca o supera el límite.")
        print("    La inhibición global está activa.")
    
    if std_phi < 0.05:
        print("\n  ✓ Φ es estable. Posible acoplamiento con A.")
    else:
        print("\n  ✗ Φ sigue siendo caótico.")
    
    if std_a < 0.01:
        print("\n  ✓ A es muy estable en rango y total.")
        print("    Atención consistente.")
    else:
        print("\n  ✗ A es inestable.")
    
    if media_total < LIMITE_ATENCION * 0.9 and std_phi < 0.05:
        print("\n  ★ POSIBLE RÉGIMEN ESTABLE")
        print("    La competencia por recurso limitado puede estar funcionando.")
    else:
        print("\n  → Aún no hay competencia efectiva.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # rango Φ
    axes[0,0].plot(range(1, 11), resultados_phi, 'bo-', markersize=8)
    axes[0,0].axhline(y=media_phi, color='r', linestyle='--')
    axes[0,0].set_title('rango Φ')
    axes[0,0].set_xlabel('Ejecución')
    axes[0,0].grid(True, alpha=0.3)
    
    # rango A
    axes[0,1].plot(range(1, 11), resultados_a, 'go-', markersize=8)
    axes[0,1].axhline(y=media_a, color='r', linestyle='--')
    axes[0,1].set_title('rango A')
    axes[0,1].set_xlabel('Ejecución')
    axes[0,1].grid(True, alpha=0.3)
    
    # total A
    axes[1,0].plot(range(1, 11), resultados_total_a, 'mo-', markersize=8)
    axes[1,0].axhline(y=LIMITE_ATENCION, color='r', linestyle='--', label=f'límite = {LIMITE_ATENCION}')
    axes[1,0].set_title('total A (con límite)')
    axes[1,0].set_xlabel('Ejecución')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # rango Φ vs total A
    axes[1,1].scatter(resultados_phi, resultados_total_a, c='blue', s=100)
    axes[1,1].set_xlabel('rango Φ')
    axes[1,1].set_ylabel('total A')
    axes[1,1].set_title('Relación Φ ↔ total A')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v15 - Competencia Global', fontsize=14)
    plt.tight_layout()
    plt.savefig('v15_competencia_global.png', dpi=150)
    print("    Gráfico guardado: v15_competencia_global.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()