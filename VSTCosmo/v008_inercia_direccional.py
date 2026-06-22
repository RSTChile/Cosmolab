#!/usr/bin/env python3
"""
VSTCosmo - Experimento v8: Inercia Direccional
A puede seguir trayectorias mediante propagación en dirección del cambio de Φ.
Sin memoria explícita, sin métricas, sin evaluación externa.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
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
DIFUSION_PHI = 0.1
GANANCIA_INEST = 0.2

# Parámetros de A
REFUERZO_A = 0.1
INHIBICION_A = 0.15
DIFUSION_A = 0.05
DISIPACION_A = 0.02
BASAL_A = 0.1

# Nueva: inercia direccional (propagación en dirección del cambio)
INERCIA = 0.1  # fuerza de propagación

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

# Límites
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

def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.1 + 0.45

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

# ============================================================
# PERTURBACIÓN
# ============================================================
def aplicar_perturbacion(Phi, muestra, paso):
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
# DINÁMICA DE Φ
# ============================================================
def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def actualizar_campo(Phi, muestra, paso):
    Phi = aplicar_perturbacion(Phi, muestra, paso)
    vecinos = vecinos_phi(Phi)
    difusion = DIFUSION_PHI * (vecinos - Phi)
    desviacion = Phi - 0.5
    inestabilidad = GANANCIA_INEST * desviacion * (1 - desviacion**2)
    Phi = Phi + DT * (difusion + inestabilidad)
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ATENCIÓN CON INERCIA DIRECCIONAL (NUEVO)
# ============================================================
def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def propagar_atencion(A, grad_temporal):
    """
    Propaga A en la dirección del cambio temporal de Φ.
    Donde Φ está aumentando, A se propaga hacia adelante en el tiempo.
    Donde Φ está disminuyendo, A se propaga hacia atrás.
    Esto NO es una métrica. Es solo inercia direccional.
    """
    # grad_temporal positivo: Φ aumentó → propagar A hacia adelante
    # grad_temporal negativo: Φ disminuyó → propagar A hacia atrás
    
    # Propagación hacia adelante (en tiempo)
    prop_adelante = np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    # Propagación hacia atrás (en tiempo)
    prop_atras = np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
    # El grad_temporal también puede propagar en frecuencia (si es fuerte)
    prop_frecuencia = 0.5 * (
        np.roll(A, 1, axis=0) * np.abs(grad_temporal) +
        np.roll(A, -1, axis=0) * np.abs(grad_temporal)
    )
    
    return INERCIA * (prop_adelante + prop_atras + prop_frecuencia)

def actualizar_atencion(A, Phi, Phi_prev):
    # Términos base
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    # Gradiente temporal de Φ
    grad_temporal = Phi - Phi_prev
    
    # Propagación direccional (inercia)
    propagacion = propagar_atencion(A, grad_temporal)
    
    dA = dA_base + propagacion
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# ACOPLAMIENTO A → Φ
# ============================================================
def acoplamiento_atencion_campo(Phi, A):
    vecinos = vecinos_phi(Phi)
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_ACOPLAMIENTO * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

# ============================================================
# SIMULACIÓN
# ============================================================
def simular(audio, sr, nombre, num_pasos=N_PASOS):
    print(f"\n  Simulando: {nombre}")
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    registro_trayectoria = []  # para seguir el centro de masa de A
    centros = []
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        Phi = actualizar_campo(Phi, muestra, paso)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        
        # Calcular centro de masa de A (para seguir trayectoria)
        y_coords, x_coords = np.indices(A.shape)
        if np.sum(A) > 0:
            centro_y = np.sum(y_coords * A) / np.sum(A)
            centro_x = np.sum(x_coords * A) / np.sum(A)
            centros.append((centro_y, centro_x, t))
        
        Phi_prev = Phi.copy()
        
        if paso % (num_pasos // 10) == 0 and paso > 0:
            pct = 100 * paso // num_pasos
            rango_a = np.max(A) - np.min(A)
            print(f"      Progreso: {pct}% | rango A={rango_a:.4f}")
    
    return {
        'nombre': nombre,
        'A': A.copy(),
        'Phi': Phi.copy(),
        'rango_A': np.max(A) - np.min(A),
        'var_A': np.var(A),
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'centros': centros
    }

# ============================================================
# VISUALIZACIÓN
# ============================================================
def visualizar_resultados(resultados):
    n_casos = len(resultados)
    fig, axes = plt.subplots(2, n_casos, figsize=(4*n_casos, 8))
    
    for i, r in enumerate(resultados):
        # Atención A
        im1 = axes[0, i].imshow(r['A'], aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'{r["nombre"]}\nAtención A (rango={r["rango_A"]:.3f})')
        axes[0, i].set_xlabel('Memoria temporal')
        axes[0, i].set_ylabel('Banda')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Campo Φ
        im2 = axes[1, i].imshow(r['Phi'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Campo Φ (rango={r["rango_Phi"]:.3f})')
        axes[1, i].set_xlabel('Memoria temporal')
        axes[1, i].set_ylabel('Banda')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle('VSTCosmo v8 - Inercia Direccional\nPropagación de A en dirección del cambio de Φ', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v8_inercia.png', dpi=150)
    print("\n  Gráfico guardado: vstcosmo_v8_inercia.png")

def visualizar_trayectorias(resultados):
    """Muestra la evolución temporal del centro de masa de A"""
    fig, axes = plt.subplots(1, len(resultados), figsize=(5*len(resultados), 4))
    if len(resultados) == 1:
        axes = [axes]
    
    for i, r in enumerate(resultados):
        if r['centros']:
            tiempos = [c[2] for c in r['centros']]
            bandas = [c[0] for c in r['centros']]
            axes[i].plot(tiempos, bandas, 'b-', linewidth=0.5, alpha=0.7)
            axes[i].set_xlabel('Tiempo (s)')
            axes[i].set_ylabel('Banda (foco de atención)')
            axes[i].set_title(f'{r["nombre"]}\nTrayectoria del centro de atención')
            axes[i].set_ylim(0, DIM_FREQ-1)
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'Sin trayectoria detectable', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{r["nombre"]}')
    
    plt.suptitle('VSTCosmo v8 - Seguimiento de Trayectorias', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v8_trayectorias.png', dpi=150)
    print("  Gráfico de trayectorias guardado: vstcosmo_v8_trayectorias.png")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v8: Inercia Direccional")
    print("A puede seguir trayectorias mediante propagación")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    
    sr_v, voz_viento = cargar_audio('Voz+Viento_1.wav')
    print(f"    Voz+Viento_1.wav: {len(voz_viento)/sr_v:.2f}s")
    
    sr_w, viento = cargar_audio('Viento.wav')
    print(f"    Viento.wav: {len(viento)/sr_w:.2f}s")
    
    sr_vc, voz_limpia = cargar_audio('Voz_Estudio.wav')
    print(f"    Voz_Estudio.wav: {len(voz_limpia)/sr_vc:.2f}s")
    
    # Ejecutar simulaciones
    print("\n[2] Ejecutando simulaciones con inercia direccional...")
    
    resultados = []
    
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    resultados.append(simular(viento, sr_w, "Viento"))
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # ============================================================
    # COMPARACIÓN
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)
    
    print("\n" + "-" * 70)
    print(f"{'Entrada':<20} | {'rango A':>10} | {'var A':>12} | {'rango Φ':>10} | {'trayectoria':>12}")
    print("-" * 70)
    
    for r in resultados:
        tiene_trayectoria = "Sí" if len(r['centros']) > 10 else "No"
        print(f"{r['nombre']:<20} | {r['rango_A']:10.4f} | {r['var_A']:12.6f} | "
              f"{r['rango_Phi']:10.4f} | {tiene_trayectoria:>12}")
    
    print("-" * 70)
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[3] Generando visualizaciones...")
    visualizar_resultados(resultados)
    visualizar_trayectorias(resultados)
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)
    print("\nPreguntas a responder con los gráficos de trayectoria:")
    print("  1. ¿El centro de atención se mueve en el tiempo?")
    print("  2. ¿El viento muestra una trayectoria más estable que la voz?")
    print("  3. ¿La voz produce una trayectoria o solo fluctuaciones?")
    print("  4. ¿El silencio produce deriva o se mantiene fijo?")
    print("=" * 60)

if __name__ == "__main__":
    main()