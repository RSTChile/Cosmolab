#!/usr/bin/env python3
"""
VSTCosmo - Experimento v9: Reconfiguración del Campo
La perturbación no se suma. Reorienta la inestabilidad del campo.
El entorno deforma Φ, no compite con él.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (manteniendo holgura)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION = 20.0
N_PASOS = int(DURACION / DT)

# Parámetros de Φ (inestabilidad base)
DIFUSION_PHI = 0.1
GANANCIA_INEST = 0.25

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
# CARGA DE AUDIO
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

# ============================================================
# NUEVO: RECONFIGURACIÓN DEL CAMPO POR EL ENTORNO
# ============================================================
def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def reconfigurar_campo(Phi, muestra):
    """
    La perturbación no se suma. Reconfigura la inestabilidad.
    La banda preferida por la muestra tiene inestabilidad más fuerte.
    """
    # Normalizar muestra a [0,1]
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    
    # Banda preferida por la perturbación
    banda_preferida = int(m * (DIM_FREQ - 1))
    
    # Crear nuevo campo (no modificación in-place)
    Phi_nuevo = Phi.copy()
    
    for i in range(DIM_FREQ):
        # Distancia a la banda preferida
        distancia = min(abs(i - banda_preferida), DIM_FREQ - abs(i - banda_preferida))
        # Influencia gaussiana (ancha para permitir transición suave)
        influencia = np.exp(-distancia**2 / 8)
        
        # La perturbación modula la intensidad de la inestabilidad
        # No es "seguir". Es "deformar la forma del campo"
        desviacion = Phi[i] - 0.5
        inestabilidad_base = GANANCIA_INEST * desviacion * (1 - desviacion**2)
        
        # La inestabilidad es más fuerte donde la perturbación lo indica
        inestabilidad = inestabilidad_base * (0.5 + 0.5 * influencia)
        
        Phi_nuevo[i] = Phi[i] + DT * inestabilidad
    
    # Difusión suave (para que el campo no se vuelva demasiado rugoso)
    vecinos = vecinos_phi(Phi_nuevo)
    difusion = DIFUSION_PHI * (vecinos - Phi_nuevo)
    Phi_nuevo = Phi_nuevo + DT * difusion
    
    return np.clip(Phi_nuevo, LIMITE_MIN, LIMITE_MAX)

def actualizar_campo(Phi, muestra, paso):
    """Ahora solo reconfiguración, sin suma directa de perturbación"""
    return reconfigurar_campo(Phi, muestra)

# ============================================================
# ATENCIÓN A (sin cambios mayores)
# ============================================================
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
    
    # Pequeño ruido
    dA += np.random.randn(*A.shape) * 0.001
    
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
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        # 1. Campo se reconfigura por el entorno
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

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v9: Reconfiguración del Campo")
    print("La perturbación deforma Φ (no se suma)")
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
    print("\n[2] Ejecutando simulaciones con reconfiguración...")
    
    resultados = []
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    resultados.append(simular(viento, sr_w, "Viento"))
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # Resultados
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS (reconfiguración)")
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
    
    plt.suptitle('VSTCosmo v9 - Reconfiguración del Campo por el Entorno', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v9_reconfiguracion.png', dpi=150)
    print("\n[3] Gráfico guardado: vstcosmo_v9_reconfiguracion.png")
    
    # Análisis rápido
    print("\n" + "=" * 60)
    print("ANÁLISIS")
    print("=" * 60)
    
    # Diferencia entre casos
    voz_vs_viento = abs(resultados[0]['rango_A'] - resultados[1]['rango_A'])
    voz_vs_silencio = abs(resultados[0]['rango_A'] - resultados[3]['rango_A'])
    
    print(f"\nDiferencia Voz+Viento vs Viento: {voz_vs_viento:.4f}")
    print(f"Diferencia Voz+Viento vs Silencio: {voz_vs_silencio:.4f}")
    
    if voz_vs_viento > 0.002:
        print("✓ El sistema distingue entre voz y viento")
    else:
        print("✗ El sistema no distingue claramente entre voz y viento")
    
    if voz_vs_silencio > 0.002:
        print("✓ La presencia de sonido afecta la atención")
    else:
        print("✗ El silencio produce resultados similares")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()