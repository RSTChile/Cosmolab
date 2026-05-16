#!/usr/bin/env python3
"""
VSTCosmo - v31.2: Memoria persistente con múltiples entradas
El sistema vive múltiples experiencias con diferentes entradas.
M acumula memoria de configuraciones que permitieron persistir.
No hay etiquetas. Solo experiencia repetida.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 60.0  # segundos por experiencia
N_PASOS = int(DURACION_SIM / DT)

# Parámetros base
DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION_BASE = 0.05
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
K_COMP_BASE = 0.05

LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

MOD_DECAY = 1.0
MOD_GENERACION = 1.5

TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_ESTABILIDAD = 0.1
BLOQUEO_MAXIMO_BASE = 0.8

TASA_OMEGA = 0.15
DISIPACION_OMEGA = 0.05
FUERZA_COHERENCIA = 0.2

# Rangos viables
A_RANGE_MIN = 0.24
A_RANGE_MAX = 0.61
PHI_RANGE_MIN = 0.08
PHI_RANGE_MAX = 0.37

HOMEOSTASIS_INTERVALO = 500

# Parámetros de memoria M (ajustados para persistencia)
ETA_MEMORIA = 0.05      # aprendizaje rápido
ETA_DECAY = 0.001       # disipación muy lenta
K_MEMORIA = 0.20        # influencia fuerte
UMBRAL_PERSISTENCIA = 0.20  # bajo, para aprender incluso en regímenes modestos

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# Entradas
ENTRADAS = ["Voz_Estudio.wav", "Tono puro", "Ruido blanco", "Silencio"]
# Generamos tono puro y ruido blanco sintéticamente
def generar_tono_puro(sr, duracion, freq=440):
    t = np.arange(int(sr * duracion)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)

def generar_ruido_blanco(sr, duracion):
    n_muestras = int(sr * duracion)
    return np.random.normal(0, 0.3, n_muestras)

# Ciclos de experiencia
EXPERIENCIAS_POR_ENTRADA = 3  # cada entrada se repite 3 veces
TOTAL_EXPERIENCIAS = len(ENTRADAS) * EXPERIENCIAS_POR_ENTRADA


# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta):
    if "Tono puro" in ruta:
        sr = 48000
        return sr, generar_tono_puro(sr, DURACION_SIM, 440)
    elif "Ruido blanco" in ruta:
        sr = 48000
        return sr, generar_ruido_blanco(sr, DURACION_SIM)
    elif "Silencio" in ruta:
        sr = 48000
        return sr, np.zeros(int(sr * DURACION_SIM))
    else:
        sr, data = wav.read(ruta)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        return sr, data[:int(sr * DURACION_SIM)]


def inicializar_campo(semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4


def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1


def inicializar_memoria():
    # Inicialización con ruido pequeño para que no parta de cero
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.05


def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0


def perfil_modulacion(muestra):
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    banda = int(m * (DIM_FREQ - 1))
    perfil = np.zeros(DIM_FREQ)
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        perfil[i] = np.exp(-(distancia ** 2) / 8.0)
    return perfil


def actualizar_memoria_estabilidad(Psi, Phi, A):
    cambio_local = np.abs(Phi - vecinos(Phi))
    cambio_norm = cambio_local / (cambio_local + 0.1)
    dPsi = (TASA_CRECIMIENTO * A * (1 - cambio_norm) * (1 - Psi) -
            TASA_DISIPACION * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)


def actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A):
    cambio_actual = Phi - Phi_prev
    cambio_anterior = Phi_prev - Phi_prev2
    signo_actual = np.tanh(cambio_actual * 10)
    signo_anterior = np.tanh(cambio_anterior * 10)
    coherencia = (1 + signo_actual * signo_anterior) / 2
    dOmega = (TASA_OMEGA * A * coherencia * (1 - Omega) -
              DISIPACION_OMEGA * Omega) * DT
    Omega = Omega + dOmega
    return np.clip(Omega, 0.0, 1.0)


def actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max):
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    perfil_2d = perfil.reshape(-1, 1)
    
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = ganancia_gen * desviacion * (1 - desviacion**2)
    mod_entrada = (1 + MOD_GENERACION * perfil_2d)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)
    generacion = generacion_base * mod_entrada * mod_memoria
    
    mod_entrada_decay = 1 - MOD_DECAY * perfil_2d
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada_decay
    
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi_propuesto = difusion + generacion + decaimiento + sostenimiento
    dPhi_real = dPhi_propuesto * (1 - bloqueo_max * Psi)
    
    Phi = Phi + DT * dPhi_real
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)


def actualizar_atencion(A, Phi, Psi, Omega, M, k_comp):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_estabilidad = FUERZA_ESTABILIDAD * (Psi - A)
    acoplamiento_coherencia = FUERZA_COHERENCIA * (Omega - A)
    
    A_mean = np.mean(A)
    competencia = -k_comp * (A - A_mean)
    sesgo_memoria = K_MEMORIA * (M - A)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          competencia +
          sesgo_memoria)
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


def actualizar_memoria_configuracion(M, A, rango_A):
    if rango_A > UMBRAL_PERSISTENCIA:
        M += ETA_MEMORIA * (A - M)
    else:
        M *= (1 - ETA_DECAY)
    return np.clip(M, 0.0, 1.0)


def homeostasis(A, Phi, G, R, C, a_min, a_max, p_min, p_max):
    rango_A = np.max(A) - np.min(A)
    rango_Phi = np.max(Phi) - np.min(Phi)
    
    if rango_A < a_min:
        G *= 1.02
        C *= 1.02
    elif rango_A > a_max:
        R *= 1.02
        G *= 0.98
    
    if rango_Phi < p_min:
        G *= 1.02
    elif rango_Phi > p_max:
        R *= 1.03
        G *= 0.97
    
    G = np.clip(G, 0.3, 3.0)
    R = np.clip(R, 0.3, 3.0)
    C = np.clip(C, 0.3, 3.0)
    
    return G, R, C


def simular_experiencia(audio, sr, M):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
    G, R, C = 1.0, 1.0, 1.0
    
    for paso in range(N_PASOS):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, M, k_comp)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        rango_A = np.max(A) - np.min(A)
        M = actualizar_memoria_configuracion(M, A, rango_A)
        
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C = homeostasis(A, Phi, G, R, C,
                                  A_RANGE_MIN, A_RANGE_MAX,
                                  PHI_RANGE_MIN, PHI_RANGE_MAX)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
    return M, A, Phi, rango_A, np.mean(A), np.mean(Psi), np.mean(Omega)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v31.2: Memoria persistente con múltiples entradas")
    print(f"{EXPERIENCIAS_POR_ENTRADA} experiencias por cada tipo de entrada")
    print(f"Total de experiencias: {TOTAL_EXPERIENCIAS}")
    print("=" * 100)
    
    M = inicializar_memoria()  # memoria persistente a través de todas las experiencias
    
    historial = []
    
    print("\n[2] Ciclo de experiencias...")
    print("-" * 80)
    
    experiencia_id = 0
    for entrada_nombre in ENTRADAS:
        print(f"\n  Entrada: {entrada_nombre}")
        sr, audio = cargar_audio(entrada_nombre)
        
        for rep in range(EXPERIENCIAS_POR_ENTRADA):
            experiencia_id += 1
            print(f"    Experiencia {experiencia_id}/{TOTAL_EXPERIENCIAS} (rep {rep+1})...", end=" ", flush=True)
            
            M, A, Phi, rango_A, media_A, media_Psi, media_Omega = simular_experiencia(audio, sr, M)
            
            historial.append({
                'id': experiencia_id,
                'entrada': entrada_nombre,
                'rep': rep+1,
                'rango_A': rango_A,
                'media_A': media_A,
                'media_M': np.mean(M),
                'rango_M': np.max(M) - np.min(M),
                'media_Psi': media_Psi,
                'media_Omega': media_Omega
            })
            
            print(f"rango_A={rango_A:.3f}, media_M={np.mean(M):.3f}")
    
    print("-" * 80)
    print("Ciclo de experiencias completado.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[3] Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Evolución de la memoria M
    ids = [h['id'] for h in historial]
    ax = axes[0, 0]
    ax.plot(ids, [h['media_M'] for h in historial], 'g-o', label='media M', markersize=4)
    ax.plot(ids, [h['rango_M'] for h in historial], 'm-s', label='rango M', markersize=4)
    ax.set_xlabel('Experiencia')
    ax.set_ylabel('Valor')
    ax.set_title('Evolución de la memoria M')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evolución del rango_A
    ax = axes[0, 1]
    colores = {'Voz_Estudio.wav': 'blue', 'Tono puro': 'orange', 'Ruido blanco': 'green', 'Silencio': 'red'}
    for entrada in ENTRADAS:
        idxs = [h['id'] for h in historial if h['entrada'] == entrada]
        rangos = [h['rango_A'] for h in historial if h['entrada'] == entrada]
        ax.plot(idxs, rangos, 'o-', color=colores.get(entrada, 'gray'), label=entrada, markersize=4)
    ax.set_xlabel('Experiencia')
    ax.set_ylabel('rango A')
    ax.set_title('Diferenciación de la atención por entrada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mapa final de M
    ax = axes[1, 0]
    im = ax.imshow(M, aspect='auto', cmap='plasma', vmin=0, vmax=1)
    ax.set_title('Mapa final de M (memoria consolidada)')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Banda')
    plt.colorbar(im, ax=ax)
    
    # Mapa final de A (última experiencia)
    ax = axes[1, 1]
    im = ax.imshow(A, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_title('Mapa final de A (atención, última experiencia)')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Banda')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'VSTCosmo v31.2 - Memoria persistente ({TOTAL_EXPERIENCIAS} experiencias)', fontsize=14)
    plt.tight_layout()
    plt.savefig('v31.2_memoria_persistente.png', dpi=150)
    print("  Gráfico guardado: v31.2_memoria_persistente.png")
    
    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS DE LA MEMORIA CONSOLIDADA")
    print("=" * 100)
    
    print(f"\nMemoria M después de {TOTAL_EXPERIENCIAS} experiencias:")
    print(f"  media_M = {np.mean(M):.4f}")
    print(f"  rango_M = {np.max(M)-np.min(M):.4f}")
    
    if np.mean(M) > 0.15:
        print("\n  ✓ M consolidó memoria significativa")
    elif np.mean(M) > 0.10:
        print("\n  ~ M tiene memoria moderada")
    else:
        print("\n  ✗ M no consolidó memoria significativa")
    
    if np.max(M)-np.min(M) > 0.3:
        print("  ✓ M tiene estructura espacial fuerte")
    
    # Evolución de media_M a lo largo de las experiencias
    media_M_inicial = historial[0]['media_M'] if historial else 0
    media_M_final = historial[-1]['media_M'] if historial else 0
    print(f"\nEvolución de media_M: {media_M_inicial:.4f} → {media_M_final:.4f}")
    
    if media_M_final > media_M_inicial * 1.5:
        print("  ✓ La memoria creció significativamente con la experiencia")
    else:
        print("  ~ La memoria no creció significativamente")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()