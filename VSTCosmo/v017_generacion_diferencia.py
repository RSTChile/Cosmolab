#!/usr/bin/env python3
"""
VSTCosmo - v17: Campo con generación de diferencia
Φ puede amplificar pequeñas diferencias locales.
No es ruido. Es generación interna.
A compite globalmente.
Φ sin A → decae. Φ con A → puede sostener y amplificar estructura.
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

# Campo Φ
GANANCIA_ENTRADA = 0.02
DIFUSION_PHI = 0.08
DECAIMIENTO_PHI = 0.04          # Reducido para no matar todo
GANANCIA_TARGET = 0.12
GANANCIA_SOSTENIMIENTO = 0.25   # Aumentado: A protege estructura
GANANCIA_GENERACION = 0.15      # NUEVO: amplifica diferencias locales

# Atención A
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08           # A siente relieve local de Φ

# Competencia global en A
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35  # ~1120
INHIB_GLOBAL = 0.5

# Límites
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# ============================================================
# UTILIDADES
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
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

# ============================================================
# CAMPO Φ (ahora con generación de diferencia)
# ============================================================
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
    
    # 1. Tendencia del entorno a deformar el campo
    arrastre_entrada = GANANCIA_TARGET * (target - Phi)
    
    # 2. Difusión local (suaviza)
    difusion = DIFUSION_PHI * (promedio_local - Phi)
    
    # 3. Decaimiento (elimina estructura donde no hay A)
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local)
    
    # 4. Sostenimiento por A: protege la diferencia donde A es alto
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    # 5. NUEVO: Generación local de diferencia
    #    Amplifica pequeñas desviaciones del promedio local
    desviacion = Phi - promedio_local
    generacion = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    
    # 6. Entrada directa (pequeña)
    entrada_directa = GANANCIA_ENTRADA * muestra
    
    dPhi = arrastre_entrada + difusion + generacion + decaimiento + sostenimiento
    Phi = Phi + DT * dPhi + entrada_directa
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ATENCIÓN A (sin cambios estructurales)
# ============================================================
def actualizar_atencion(A, Phi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    # Relieve local de Φ (diferencia respecto al promedio)
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_local = FUERZA_RELIEVE * (relieve_local - A)
    
    dA = auto + inhib_local + difusion + acoplamiento_local
    
    # Competencia global
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    # Ruido mínimo
    dA += np.random.randn(*A.shape) * 0.001
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
def simular(audio, sr, nombre, semilla=None, num_pasos=N_PASOS):
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
        
        A = actualizar_atencion(A, Phi)  # Primero A observa Φ
        Phi = actualizar_campo(Phi, A, muestra)  # Luego Φ evoluciona con A
    
    rango_phi = float(np.max(Phi) - np.min(Phi))
    rango_a = float(np.max(A) - np.min(A))
    media_a = float(np.mean(A))
    total_a = float(np.sum(A))
    
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}, media A={media_a:.4f}, total A={total_a:.0f}")
    return rango_phi, rango_a, media_a, total_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("VSTCosmo - v17: Campo con generación de diferencia")
    print("Φ puede amplificar pequeñas diferencias locales")
    print("=" * 70)
    
    print("\n[1] Cargando archivo...")
    sr, voz_viento = cargar_audio('Voz+Viento_1.wav')
    print(f"    Voz+Viento_1.wav: {len(voz_viento)/sr:.1f}s")
    
    semillas = [42, 123, 987, 1, 2, 3, 100, 200, 300, 400]
    
    print("\n[2] Ejecutando 10 simulaciones...")
    print("-" * 70)
    
    resultados_phi = []
    resultados_a = []
    resultados_media_a = []
    resultados_total_a = []
    
    for i, semilla in enumerate(semillas):
        rphi, ra, media_a, total_a = simular(voz_viento, sr, f"Run_{i}", semilla=semilla)
        resultados_phi.append(rphi)
        resultados_a.append(ra)
        resultados_media_a.append(media_a)
        resultados_total_a.append(total_a)
    
    print("-" * 70)
    
    # Análisis estadístico
    print("\n[3] Análisis estadístico")
    print("=" * 70)
    
    media_phi = np.mean(resultados_phi)
    std_phi = np.std(resultados_phi)
    media_a = np.mean(resultados_a)
    std_a = np.std(resultados_a)
    media_total = np.mean(resultados_total_a)
    std_total = np.std(resultados_total_a)
    
    print(f"  rango Φ: media={media_phi:.3f}, std={std_phi:.4f}")
    print(f"  rango A: media={media_a:.4f}, std={std_a:.4f}")
    print(f"  total A: media={media_total:.0f}, std={std_total:.0f}")
    
    # Interpretación
    print("\n[4] Interpretación")
    print("=" * 70)
    
    if std_phi < 0.02:
        print("  ✓ Φ es consistente entre ejecuciones.")
    else:
        print("  ✗ Φ es sensible a condiciones iniciales.")
    
    if std_a < 0.02:
        print("  ✓ A es consistente entre ejecuciones.")
    else:
        print("  ✗ A es sensible.")
    
    if media_phi > 0.15:
        print("  ✓ Φ tiene relieve significativo.")
    else:
        print("  ✗ Φ tiene relieve bajo.")
    
    if media_a > 0.3:
        print("  ✓ A tiene alta diferenciación.")
    else:
        print("  ✗ A tiene baja diferenciación.")
    
    if std_phi < 0.02 and std_a < 0.02 and media_phi > 0.15 and media_a > 0.3:
        print("\n  ★★ POSIBLE RÉGIMEN CO-SOSTENIDO ★★")
        print("      Φ y A son consistentes y tienen estructura.")
    elif std_phi < 0.05 and std_a < 0.05:
        print("\n  → Régimen consistente pero estructura podría mejorar.")
    else:
        print("\n  → Aún no hay régimen co-sostenido claro.")
    
    # Visualización
    print("\n[5] Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].plot(range(1, 11), resultados_phi, 'bo-', markersize=8)
    axes[0,0].axhline(y=media_phi, color='r', linestyle='--')
    axes[0,0].set_title('rango Φ')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(range(1, 11), resultados_a, 'go-', markersize=8)
    axes[0,1].axhline(y=media_a, color='r', linestyle='--')
    axes[0,1].set_title('rango A')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(range(1, 11), resultados_total_a, 'mo-', markersize=8)
    axes[1,0].axhline(y=LIMITE_ATENCION, color='r', linestyle='--', label='límite')
    axes[1,0].set_title('total A')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(resultados_phi, resultados_a, s=100, alpha=0.7)
    axes[1,1].set_xlabel('rango Φ')
    axes[1,1].set_ylabel('rango A')
    axes[1,1].set_title('Relación Φ ↔ A')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v17 - Generación de diferencia', fontsize=14)
    plt.tight_layout()
    plt.savefig('v17_generacion_diferencia.png', dpi=150)
    print("    Gráfico guardado: v17_generacion_diferencia.png")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 70)

if __name__ == "__main__":
    main()