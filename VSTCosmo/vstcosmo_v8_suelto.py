#!/usr/bin/env python3
"""
VSTCosmo - Experimento v8: Suelto y Ruidoso
Parámetros más libres, menos control, más naturaleza.
No busca precisión. Busca persistencia.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS SUELTOS (menos rigurosidad)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION = 20.0
N_PASOS = int(DURACION / DT)

# Φ: inestabilidad con holgura
DIFUSION_PHI = 0.1
GANANCIA_INEST = 0.25   # un poco más que antes

# A: más libertad
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

# Inercia: más cruda
INERCIA = 0.05

# Acoplamiento
DIFUSION_ACOPLAMIENTO = 0.2

# Límites amplios
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

def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

def aplicar_perturbacion(Phi, muestra, paso):
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    banda = int(m * (DIM_FREQ - 1))
    t_idx = paso % DIM_TIME
    intensidad = 0.1 * abs(muestra)  # más intensidad
    
    for db in range(-3, 4):  # más rango
        b = (banda + db) % DIM_FREQ
        peso = np.exp(-(db**2) / 6)  # gaussiano más ancho
        Phi[b, t_idx] += intensidad * peso
    
    for dt in range(-2, 3):
        t = (t_idx + dt) % DIM_TIME
        for db in range(-2, 3):
            b = (banda + db) % DIM_FREQ
            peso_t = np.exp(-(dt**2) / 3)
            Phi[b, t] += intensidad * 0.2 * peso_t
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

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

def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def actualizar_atencion(A, Phi, Phi_prev):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    # Gradiente temporal crudo (sin normalizar)
    grad_temporal = Phi - Phi_prev
    
    # Propagación sucia y ruidosa
    prop = INERCIA * np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    prop += INERCIA * 0.5 * np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
    dA = dA_base + prop
    
    # Pequeño ruido (naturaleza)
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
    print(f"\n  Simulando: {nombre}")
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        Phi = actualizar_campo(Phi, muestra, paso)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        
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
        'rango_Phi': np.max(Phi) - np.min(Phi)
    }

def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v8: Suelto y Ruidoso")
    print("Parámetros más libres, menos control")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    
    sr_v, voz_viento = cargar_audio('Voz+Viento_1.wav')
    sr_w, viento = cargar_audio('Viento.wav')
    sr_vc, voz_limpia = cargar_audio('Voz_Estudio.wav')
    
    print(f"    Voz+Viento: {len(voz_viento)/sr_v:.2f}s")
    print(f"    Viento: {len(viento)/sr_w:.2f}s")
    print(f"    Voz_Estudio: {len(voz_limpia)/sr_vc:.2f}s")
    
    # Ejecutar
    print("\n[2] Ejecutando simulaciones sueltas...")
    
    resultados = []
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    resultados.append(simular(viento, sr_w, "Viento"))
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # Resultados
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS (suelta)")
    print("=" * 60)
    
    for r in resultados:
        print(f"{r['nombre']:15} | rango A={r['rango_A']:.4f} | var A={r['var_A']:.6f} | rango Φ={r['rango_Phi']:.4f}")
    
    # Visualización
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, r in enumerate(resultados):
        axes[0, i].imshow(r['A'], aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'{r["nombre"]}\nA (rango={r["rango_A"]:.3f})')
        axes[0, i].set_xlabel('Tiempo')
        axes[0, i].set_ylabel('Banda')
        
        axes[1, i].imshow(r['Phi'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Φ (rango={r["rango_Phi"]:.3f})')
        axes[1, i].set_xlabel('Tiempo')
        axes[1, i].set_ylabel('Banda')
    
    plt.suptitle('VSTCosmo v8 - Suelto y Ruidoso (menos control, más naturaleza)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v8_suelto.png', dpi=150)
    print("\n[3] Gráfico guardado: vstcosmo_v8_suelto.png")
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()