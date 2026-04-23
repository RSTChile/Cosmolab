#!/usr/bin/env python3
"""
VSTCosmo - Experimento v10: Supresión de Inestabilidad
La perturbación suprime la inestabilidad donde es fuerte.
El campo puede ser permeable sin saturar.
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
DURACION = 20.0
N_PASOS = int(DURACION / DT)

# Parámetros de Φ
DIFUSION_PHI = 0.1
GANANCIA_INEST = 0.25

# Parámetros de A
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

# Supresión: qué tanto la entrada inhibe la inestabilidad
SUPRESION = 0.8

# Límites
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

def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def reconfigurar_campo(Phi, muestra):
    """
    La perturbación suprime la inestabilidad donde es fuerte.
    La inestabilidad se debilita en la banda preferida por la entrada,
    permitiendo que el campo sea permeable sin saturar.
    """
    # Normalizar muestra a [0,1]
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    
    # Banda preferida por la perturbación
    banda_preferida = int(m * (DIM_FREQ - 1))
    
    Phi_nuevo = Phi.copy()
    
    for i in range(DIM_FREQ):
        # Distancia a la banda preferida
        distancia = min(abs(i - banda_preferida), DIM_FREQ - abs(i - banda_preferida))
        
        # Preferencia de la entrada (gaussiana más estrecha)
        preferencia = np.exp(-distancia**2 / 4)
        
        # Inestabilidad base (Landau)
        desviacion = Phi[i] - 0.5
        inestabilidad_base = GANANCIA_INEST * desviacion * (1 - desviacion**2)
        
        # La perturbación SUPRIME la inestabilidad donde es fuerte
        # (evita que el campo sature donde hay entrada)
        inestabilidad = inestabilidad_base * (1 - SUPRESION * preferencia)
        
        Phi_nuevo[i] = Phi[i] + DT * inestabilidad
    
    # Difusión suave (para mantener coherencia espacial)
    vecinos = vecinos_phi(Phi_nuevo)
    difusion = DIFUSION_PHI * (vecinos - Phi_nuevo)
    Phi_nuevo = Phi_nuevo + DT * difusion
    
    return np.clip(Phi_nuevo, LIMITE_MIN, LIMITE_MAX)

def actualizar_campo(Phi, muestra, paso):
    """La perturbación suprime la inestabilidad en su banda preferida"""
    return reconfigurar_campo(Phi, muestra)

def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def actualizar_atencion(A, Phi, Phi_prev):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    # Gradiente temporal (crudo)
    grad_temporal = Phi - Phi_prev
    
    # Propagación simple (inercia baja)
    prop = 0.02 * np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    prop += 0.01 * np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
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
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        # 1. Campo se reconfigura (inestabilidad suprimida por entrada)
        Phi = actualizar_campo(Phi, muestra, paso)
        
        # 2. Atención evoluciona
        A = actualizar_atencion(A, Phi, Phi_prev)
        
        # 3. Acoplamiento A → Φ
        Phi = acoplamiento_atencion_campo(Phi, A)
        
        Phi_prev = Phi.copy()
        
        if paso % (num_pasos // 10) == 0 and paso > 0:
            pct = 100 * paso // num_pasos
            rango_a = np.max(A) - np.min(A)
            rango_phi = np.max(Phi) - np.min(Phi)
            print(f"      Progreso: {pct}% | rango A={rango_a:.4f}, rango Φ={rango_phi:.4f}")
    
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
    print("VSTCosmo - Experimento v10: Supresión de Inestabilidad")
    print("La perturbación suprime la inestabilidad donde es fuerte")
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
    print("\n[2] Ejecutando simulaciones con supresión...")
    
    resultados = []
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    resultados.append(simular(viento, sr_w, "Viento"))
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # Resultados
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS (supresión)")
    print("=" * 60)
    
    for r in resultados:
        print(f"{r['nombre']:15} | rango A={r['rango_A']:.4f} | var A={r['var_A']:.6f} | rango Φ={r['rango_Phi']:.4f}")
    
    # Visualización
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, r in enumerate(resultados):
        axes[0, i].imshow(r['A'], aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'{r["nombre"]}\nAtención A (rango={r["rango_A"]:.3f})')
        axes[0, i].set_xlabel('Memoria temporal')
        axes[0, i].set_ylabel('Banda')
        
        axes[1, i].imshow(r['Phi'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Campo Φ (rango={r["rango_Phi"]:.3f})')
        axes[1, i].set_xlabel('Memoria temporal')
        axes[1, i].set_ylabel('Banda')
    
    plt.suptitle('VSTCosmo v10 - Supresión de Inestabilidad por el Entorno', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v10_supresion.png', dpi=150)
    print("\n[3] Gráfico guardado: vstcosmo_v10_supresion.png")
    
    # Análisis
    print("\n" + "=" * 60)
    print("ANÁLISIS")
    print("=" * 60)
    
    ref = resultados[0]
    for r in resultados[1:]:
        diff = abs(r['rango_A'] - ref['rango_A'])
        print(f"Diferencia {ref['nombre']} vs {r['nombre']}: {diff:.4f}")
    
    # Verificar si Φ saturó
    saturados = sum(1 for r in resultados if r['rango_Phi'] >= 0.99)
    print(f"\nCampos saturados (rango Φ ≥ 0.99): {saturados}/{len(resultados)}")
    
    if saturados == len(resultados):
        print("⚠ El campo sigue saturando. La supresión no fue suficiente.")
    elif saturados == 0:
        print("✓ El campo no satura. La permeabilidad mejora.")
    else:
        print(f"→ {saturados} casos saturan, {len(resultados)-saturados} no.")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()