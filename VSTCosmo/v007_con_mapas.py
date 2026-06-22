#!/usr/bin/env python3
"""
VSTCosmo - Experimento v7 con mapas espaciales
GANANCIA_INEST = 0.2
Visualización de la geometría de la atención para cada entrada.
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
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.1 + 0.45

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

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

def varianza_local(Phi, radio=1):
    var_local = np.zeros_like(Phi)
    for i in range(radio, DIM_FREQ - radio):
        for j in range(radio, DIM_TIME - radio):
            ventana = Phi[i-radio:i+radio+1, j-radio:j+radio+1]
            var_local[i, j] = np.var(ventana)
    # Propagación a bordes
    for i in range(DIM_FREQ):
        for j in range(DIM_TIME):
            if var_local[i, j] == 0:
                for r in range(1, min(DIM_FREQ, DIM_TIME)):
                    encontrado = False
                    for di in [-r, r]:
                        for dj in [-r, r]:
                            ii, jj = i+di, j+dj
                            if 0 <= ii < DIM_FREQ and 0 <= jj < DIM_TIME:
                                if var_local[ii, jj] > 0:
                                    var_local[i, j] = var_local[ii, jj]
                                    encontrado = True
                                    break
                        if encontrado:
                            break
                    if encontrado:
                        break
    return var_local

def actualizar_atencion(A, Phi):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    var_local = varianza_local(Phi, radio=1)
    max_var = np.max(var_local)
    if max_var > 0:
        modulador = 1 + 2 * var_local / max_var
    else:
        modulador = np.ones_like(Phi)
    
    dA = dA_base * modulador
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
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        Phi = actualizar_campo(Phi, muestra, paso)
        A = actualizar_atencion(A, Phi)
        Phi = acoplamiento_atencion_campo(Phi, A)
        
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
        'var_Phi': np.var(Phi)
    }

def visualizar_resultados(resultados):
    """Genera visualización comparativa de mapas de atención y campo"""
    
    n_casos = len(resultados)
    fig, axes = plt.subplots(2, n_casos, figsize=(4*n_casos, 8))
    
    for i, r in enumerate(resultados):
        # Atención A
        im1 = axes[0, i].imshow(r['A'], aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'{r["nombre"]}\nAtención A (rango={r["rango_A"]:.3f})')
        axes[0, i].set_xlabel('Memoria temporal')
        axes[0, i].set_ylabel('Banda (frecuencia análoga)')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Campo Φ
        im2 = axes[1, i].imshow(r['Phi'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Campo Φ (rango={r["rango_Phi"]:.3f})')
        axes[1, i].set_xlabel('Memoria temporal')
        axes[1, i].set_ylabel('Banda')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle('VSTCosmo v7 - Geometría de la Atención y el Campo\nGANANCIA_INEST = 0.2', fontsize=14)
    plt.tight_layout()
    plt.savefig('vstcosmo_v7_mapas_ganancia_0.2.png', dpi=150)
    print("\n  Gráfico guardado: vstcosmo_v7_mapas_ganancia_0.2.png")

def main():
    print("=" * 60)
    print("VSTCosmo - Experimento v7 con Mapas Espaciales")
    print("GANANCIA_INEST = 0.2")
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
    print("\n[2] Ejecutando simulaciones...")
    
    resultados = []
    
    # Voz + Viento
    resultados.append(simular(voz_viento, sr_v, "Voz+Viento"))
    
    # Viento puro
    resultados.append(simular(viento, sr_w, "Viento"))
    
    # Voz estudio
    resultados.append(simular(voz_limpia, sr_vc, "Voz_Estudio"))
    
    # Silencio
    print(f"\n  Simulando: Silencio")
    audio_silencio = np.zeros(int(sr_v * DURACION))
    resultados.append(simular(audio_silencio, sr_v, "Silencio"))
    
    # ============================================================
    # COMPARACIÓN DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)
    
    print("\n" + "-" * 70)
    print(f"{'Entrada':<20} | {'rango A':>10} | {'var A':>12} | {'rango Φ':>10} | {'var Φ':>12}")
    print("-" * 70)
    
    for r in resultados:
        print(f"{r['nombre']:<20} | {r['rango_A']:10.4f} | {r['var_A']:12.6f} | "
              f"{r['rango_Phi']:10.4f} | {r['var_Phi']:12.6f}")
    
    print("-" * 70)
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[3] Generando visualización...")
    visualizar_resultados(resultados)
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)
    print("\nObservaciones a realizar en los mapas:")
    print("  1. ¿Dónde se concentra A en cada caso?")
    print("  2. ¿Los focos de A coinciden con regiones de alta varianza en Φ?")
    print("  3. ¿El viento genera múltiples focos vs la voz uno más estable?")
    print("  4. ¿El silencio produce patrones internos diferentes?")
    print("=" * 60)

if __name__ == "__main__":
    main()