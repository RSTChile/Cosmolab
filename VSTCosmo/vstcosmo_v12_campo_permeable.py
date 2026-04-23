#!/usr/bin/env python3
"""
VSTCosmo - Experimento v12: Campo Permeable por Diseño
El punto de equilibrio de Φ sigue a la entrada.
La permeabilidad es estructural, no regulada desde fuera.
El Pastor solo observa, no ajusta.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS FIJOS (el Pastor NO ajusta)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION = 20.0
N_PASOS = int(DURACION / DT)

# Parámetros de Φ (estructurales, no ajustables)
GANANCIA_INEST = 0.15      # inestabilidad moderada
DIFUSION_PHI = 0.1
SENSIBILIDAD_ENTRADA = 0.15  # cuánto la entrada arrastra el target

# Parámetros de A (sin cambios)
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

# Acoplamiento A → Φ
DIFUSION_ACOPLAMIENTO = 0.2

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

def actualizar_campo_permeable(Phi, muestra):
    """
    El punto de equilibrio del campo sigue a la entrada.
    La permeabilidad es estructural, no regulada.
    """
    # 1. Target sugerido por la entrada (no impuesto)
    #    Normalizar muestra a [0,1] y mapear a [0.3, 0.7]
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    target_banda = 0.3 + 0.4 * m  # rango 0.3-0.7
    
    # 2. Construir target por banda (la entrada afecta más a su banda preferida)
    #    No es "seguimiento", es "deformación del paisaje de potencial"
    m_banda = int(m * (DIM_FREQ - 1))
    target = np.ones_like(Phi) * 0.5  # valor base neutro
    
    for i in range(DIM_FREQ):
        distancia = min(abs(i - m_banda), DIM_FREQ - abs(i - m_banda))
        influencia = np.exp(-distancia**2 / 10)  # gaussiana ancha
        target[i] = target_banda * influencia + 0.5 * (1 - influencia)
    
    # 3. Inestabilidad alrededor del target local
    desviacion = Phi - target
    inestabilidad = GANANCIA_INEST * desviacion * (1 - desviacion**2)
    
    # 4. Difusión suave (para mantener coherencia espacial)
    vecinos = vecinos_phi(Phi)
    difusion = DIFUSION_PHI * (vecinos - Phi)
    
    # 5. Pequeña influencia directa de la entrada (para que el target no sea decorativo)
    entrada_directa = 0.02 * muestra
    
    # 6. Evolución
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
    print(f"\n  Simulando: {nombre}")
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        Phi = actualizar_campo_permeable(Phi, muestra)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        
        Phi_prev = Phi.copy()
        
        if paso % (num_pasos // 10) == 0 and paso > 0:
            pct = 100 * paso // num_pasos
            rango_phi = np.max(Phi) - np.min(Phi)
            rango_a = np.max(A) - np.min(A)
            print(f"      Progreso: {pct}% | rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
    
    return {
        'nombre': nombre,
        'A': A.copy(),
        'Phi': Phi.copy(),
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'rango_A': np.max(A) - np.min(A),
        'var_A': np.var(A)
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v12: Campo Permeable por Diseño")
    print("El punto de equilibrio de Φ sigue a la entrada")
    print("La permeabilidad es estructural, no regulada")
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
    print("\n[2] Ejecutando simulaciones...")
    
    resultados = []
    
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    resultados.append(simular(viento, sr_w, "Viento"))
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    
    print("\n" + "-" * 70)
    print(f"{'Entrada':<20} | {'rango Φ':>10} | {'rango A':>10} | {'var A':>12}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['nombre']:<20} | {r['rango_Phi']:10.3f} | {r['rango_A']:10.4f} | {r['var_A']:12.6f}")
    
    print("-" * 70)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 60)
    print("ANÁLISIS")
    print("=" * 60)
    
    for r in resultados:
        print(f"\n{r['nombre']}:")
        if r['rango_Phi'] < 0.85:
            print(f"  ✓ Φ NO satura (rango={r['rango_Phi']:.3f})")
        else:
            print(f"  ✗ Φ SATURA (rango={r['rango_Phi']:.3f})")
        
        if r['rango_A'] > 0.015:
            print(f"  ✓ Atención diferenciada (rango={r['rango_A']:.4f})")
        elif r['rango_A'] > 0.008:
            print(f"  ~ Atención débil (rango={r['rango_A']:.4f})")
        else:
            print(f"  ✗ Atención uniforme (rango={r['rango_A']:.4f})")
    
    # Verificar si hay diferencias entre entradas
    ref = resultados[0]
    diferencias = []
    for r in resultados[1:]:
        diff = abs(r['rango_A'] - ref['rango_A'])
        diferencias.append(diff)
    
    print("\n" + "-" * 70)
    print("Diferencias entre entradas (rango A):")
    for i, r in enumerate(resultados[1:], 1):
        print(f"  {ref['nombre']} vs {r['nombre']}: {diferencias[i-1]:.5f}")
    
    if max(diferencias) > 0.003:
        print("\n  ✓ El sistema distingue entre entradas")
    else:
        print("\n  ✗ El sistema no distingue claramente entre entradas")
    
    # Visualización
    print("\n[3] Generando visualización...")
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
    
    plt.suptitle('VSTCosmo v12 - Campo Permeable por Diseño\nEl punto de equilibrio de Φ sigue a la entrada', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v12_permeable.png', dpi=150)
    print("  Gráfico guardado: vstcosmo_v12_permeable.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()