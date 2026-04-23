#!/usr/bin/env python3
"""
VSTCosmo - v14: Acoplamiento Total
A no tiene atractor propio. Su única forma de persistir es acoplarse a Φ.
No hay BASAL_A externo. No hay disipación independiente.
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

# Parámetros de A (sin BASAL_A externo)
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

# Nuevo: fuerza con la que A tiende a Φ (única fuente de estabilidad)
FUERZA_ACOPLAMIENTO = 0.1

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
    # Inicialización uniforme baja (sin BASAL_A)
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
    """
    A evoluciona con reglas locales, pero su única fuente de estabilidad
    es el acoplamiento con Φ. No hay BASAL_A externo.
    """
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    
    # NUEVO: A tiende a Φ localmente (única fuerza estabilizadora)
    # No hay disipación a un basal externo. A solo se estabiliza si se acopla a Φ.
    acoplamiento_a_phi = FUERZA_ACOPLAMIENTO * (Phi - A)
    
    dA = auto + inhib + dif + acoplamiento_a_phi
    
    # Ruido mínimo (natural)
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
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}, media A={media_a:.4f}")
    return rango_phi, rango_a, media_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - v14: Acoplamiento Total")
    print("A no tiene atractor propio. Solo persiste si se acopla a Φ.")
    print("No hay BASAL_A externo. No hay disipación independiente.")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr, voz_viento = cargar_audio('Voz+Viento_1.wav')
    print(f"    Voz+Viento_1.wav: {len(voz_viento)/sr:.1f}s")
    
    # Semillas para test de estabilidad
    semillas = [42, 123, 987, 1, 2, 3, 100, 200, 300, 400]
    
    print("\n[2] Ejecutando 10 simulaciones con Voz+Viento_real...")
    print("    (misma entrada, diferentes condiciones iniciales)")
    print("-" * 60)
    
    resultados_phi = []
    resultados_a = []
    resultados_media_a = []
    
    for i, semilla in enumerate(semillas):
        rphi, ra, media_a = simular(voz_viento, sr, f"Run_{i}", semilla=semilla)
        resultados_phi.append(rphi)
        resultados_a.append(ra)
        resultados_media_a.append(media_a)
    
    print("-" * 60)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n[3] Análisis estadístico")
    print("=" * 60)
    
    media_phi = np.mean(resultados_phi)
    std_phi = np.std(resultados_phi)
    min_phi = np.min(resultados_phi)
    max_phi = np.max(resultados_phi)
    
    media_a = np.mean(resultados_a)
    std_a = np.std(resultados_a)
    media_total_a = np.mean(resultados_media_a)
    
    print(f"\n  rango Φ:")
    print(f"    media  = {media_phi:.3f}")
    print(f"    std    = {std_phi:.4f}")
    print(f"    min    = {min_phi:.3f}")
    print(f"    max    = {max_phi:.3f}")
    
    print(f"\n  rango A:")
    print(f"    media  = {media_a:.4f}")
    print(f"    std    = {std_a:.4f}")
    
    print(f"\n  media A (valor promedio):")
    print(f"    media  = {media_total_a:.4f}")
    
    # ============================================================
    # INTERPRETACIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN")
    print("=" * 60)
    
    if std_phi < 0.05:
        print("\n  ✓ Φ es estable. A logró acoplarse a Φ.")
    elif std_phi < 0.1:
        print("\n  ~ Φ es moderadamente estable.")
    else:
        print("\n  ✗ Φ sigue siendo caótico. A no logró acoplarse consistentemente.")
    
    # Verificar si A siguió a Φ
    if media_total_a > 0.2:
        print("  ✓ A tiene actividad significativa.")
    else:
        print("  ✗ A permanece en valores bajos.")
    
    if std_phi < 0.05 and media_total_a > 0.2:
        print("\n  ★ POSIBLE ACOPLAMIENTO REAL")
        print("    A se estabilizó junto con Φ.")
        print("    A no tiene atractor propio. Solo persiste acoplada.")
    else:
        print("\n  → Aún no hay acoplamiento fuerte.")
        print("    A sigue sin depender realmente de Φ.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # rango Φ
    ax1.plot(range(1, 11), resultados_phi, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=media_phi, color='r', linestyle='--', label=f'media = {media_phi:.3f}')
    ax1.fill_between(range(1, 11), media_phi - std_phi, media_phi + std_phi, 
                      alpha=0.2, color='r', label=f'±1σ')
    ax1.set_xlabel('Ejecución')
    ax1.set_ylabel('rango Φ')
    ax1.set_title('Estabilidad del campo Φ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # rango A
    ax2.plot(range(1, 11), resultados_a, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=media_a, color='r', linestyle='--', label=f'media = {media_a:.4f}')
    ax2.fill_between(range(1, 11), media_a - std_a, media_a + std_a, 
                      alpha=0.2, color='r', label=f'±1σ')
    ax2.set_xlabel('Ejecución')
    ax2.set_ylabel('rango A')
    ax2.set_title('Diferenciación de la atención A')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # media A (actividad)
    ax3.plot(range(1, 11), resultados_media_a, 'mo-', linewidth=2, markersize=8)
    ax3.axhline(y=media_total_a, color='r', linestyle='--', label=f'media = {media_total_a:.4f}')
    ax3.set_xlabel('Ejecución')
    ax3.set_ylabel('media A')
    ax3.set_title('Actividad de la atención (media)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Relación Φ vs A
    ax4.scatter(resultados_phi, resultados_a, c='blue', s=100, alpha=0.7)
    ax4.set_xlabel('rango Φ')
    ax4.set_ylabel('rango A')
    ax4.set_title('Relación Φ ↔ A (¿acoplamiento?)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v14 - Acoplamiento Total\nA solo persiste si se acopla a Φ', fontsize=14)
    plt.tight_layout()
    plt.savefig('v14_acoplamiento_total.png', dpi=150)
    print("    Gráfico guardado: v14_acoplamiento_total.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()