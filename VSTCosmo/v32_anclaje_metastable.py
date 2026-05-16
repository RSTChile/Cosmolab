#!/usr/bin/env python3
"""
VSTCosmo - v32: Anclaje metastable (L)
L ralentiza el cambio de A en regiones de alta persistencia.
No es memoria descriptiva, es inercia local.
Permite que algunas configuraciones permanezcan más tiempo.
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
DURACION_SIM = 60.0
N_PASOS = int(DURACION_SIM / DT)

# Parámetros base (v30)
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

# Parámetros de memoria M
ETA_MEMORIA = 0.02
ETA_DECAY = 0.005
K_MEMORIA = 0.10
UMBRAL_MEMORIA = 0.35

# Nuevos parámetros de anclaje metastable (L)
ETA_L = 0.01          # tasa de crecimiento de L
DECAY_L = 0.002       # disipación de L
INFLUENCIA_L = 0.5    # cuánto ralentiza L el cambio de A
PERSISTENCIA_UMBRAL = 0.05  # persistencia local mínima para acumular L

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

ENTRADA = "Voz_Estudio.wav"


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
    return sr, data[:int(sr * DURACION_SIM)]


def inicializar_campo():
    np.random.seed(42)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4


def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1


def inicializar_memoria():
    return np.zeros((DIM_FREQ, DIM_TIME), dtype=np.float32)


def inicializar_anclaje():
    return np.zeros((DIM_FREQ, DIM_TIME), dtype=np.float32)


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


def actualizar_atencion(A, Phi, Psi, Omega, M, L, k_comp):
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
    
    # NUEVO: anclaje metastable - L ralentiza el cambio donde es alto
    factor_anclaje = 1.0 - INFLUENCIA_L * L
    dA = dA * factor_anclaje
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


def actualizar_memoria_configuracion(M, A, rango_A):
    if rango_A > UMBRAL_MEMORIA:
        M += ETA_MEMORIA * (A - M)
    else:
        M *= (1 - ETA_DECAY)
    return np.clip(M, 0.0, 1.0)


def actualizar_anclaje(L, A):
    """
    L crece donde A tiene alta persistencia local (cambio lento).
    Se disipa uniformemente.
    No es un objetivo. Es inercia.
    """
    # Persistencia local: qué tan poco cambia A en el tiempo
    # Aproximación: derivada temporal suave
    cambio_local = np.abs(A - np.roll(A, 1, axis=1))  # diferencia en tiempo
    persistencia = 1.0 - cambio_local / (cambio_local + PERSISTENCIA_UMBRAL)
    
    dL = (ETA_L * persistencia * (1 - L) - DECAY_L * L) * DT
    L = L + dL
    return np.clip(L, 0.0, 1.0)


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


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v32: Anclaje metastable (L)")
    print("L ralentiza el cambio de A donde ha habido persistencia local")
    print("Permite que algunas configuraciones permanezcan más tiempo")
    print("=" * 100)
    
    sr, audio = cargar_audio(ENTRADA)
    print(f"\n[1] Cargando: {ENTRADA}")
    print(f"    Duración: {len(audio)/sr:.1f}s")
    
    print("\n[2] Inicializando sistema...")
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    M = inicializar_memoria()
    L = inicializar_anclaje()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
    G, R, C = 1.0, 1.0, 1.0
    
    print("\n[3] Ejecutando simulación con anclaje metastable...")
    print("-" * 80)
    
    registro = {
        'paso': [], 'tiempo': [],
        'rango_A': [], 'rango_Phi': [],
        'media_A': [], 'media_Phi': [],
        'media_M': [], 'rango_M': [],
        'media_L': [], 'rango_L': []
    }
    
    for paso in range(N_PASOS):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, M, L, k_comp)
        M = actualizar_memoria_configuracion(M, A, np.max(A)-np.min(A))
        L = actualizar_anclaje(L, A)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C = homeostasis(A, Phi, G, R, C,
                                  A_RANGE_MIN, A_RANGE_MAX,
                                  PHI_RANGE_MIN, PHI_RANGE_MAX)
        
        if paso % (N_PASOS // 100) == 0:
            registro['paso'].append(paso)
            registro['tiempo'].append(t)
            registro['rango_A'].append(np.max(A)-np.min(A))
            registro['rango_Phi'].append(np.max(Phi)-np.min(Phi))
            registro['media_A'].append(np.mean(A))
            registro['media_Phi'].append(np.mean(Phi))
            registro['media_M'].append(np.mean(M))
            registro['rango_M'].append(np.max(M)-np.min(M))
            registro['media_L'].append(np.mean(L))
            registro['rango_L'].append(np.max(L)-np.min(L))
        
        if paso % (N_PASOS // 10) == 0:
            pct = 100 * paso // N_PASOS
            print(f"  Progreso: {pct}% | rango_A={np.max(A)-np.min(A):.3f} | media_L={np.mean(L):.3f}")
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
    print("-" * 80)
    print("Simulación completada.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # rango_A
    ax = axes[0, 0]
    ax.plot(registro['tiempo'], registro['rango_A'], 'b-', linewidth=1)
    ax.set_ylabel('rango A')
    ax.set_title('Diferenciación de la atención')
    ax.grid(True, alpha=0.3)
    
    # rango_Phi
    ax = axes[0, 1]
    ax.plot(registro['tiempo'], registro['rango_Phi'], 'r-', linewidth=1)
    ax.set_ylabel('rango Φ')
    ax.set_title('Estructura del campo')
    ax.grid(True, alpha=0.3)
    
    median = axes[1, 0]
    median.plot(registro['tiempo'], registro['media_A'], 'b-', linewidth=1)
    median.set_ylabel('media A')
    median.set_title('Actividad media de la atención')
    median.grid(True, alpha=0.3)
    
    # Memoria M
    ax = axes[1, 1]
    ax.plot(registro['tiempo'], registro['media_M'], 'g-', label='media M', linewidth=1)
    ax.plot(registro['tiempo'], registro['rango_M'], 'g--', label='rango M', linewidth=1)
    ax.set_ylabel('M')
    ax.set_title('Memoria de configuraciones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Anclaje L
    ax = axes[2, 0]
    ax.plot(registro['tiempo'], registro['media_L'], 'c-', label='media L', linewidth=1)
    ax.plot(registro['tiempo'], registro['rango_L'], 'c--', label='rango L', linewidth=1)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('L')
    ax.set_title('Anclaje metastable')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mapas finales
    ax = axes[2, 1]
    im = ax.imshow(L, aspect='auto', cmap='plasma', vmin=0, vmax=1)
    ax.set_title('Mapa final de L (anclaje)')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Banda')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('VSTCosmo v32 - Anclaje metastable (L)', fontsize=14)
    plt.tight_layout()
    plt.savefig('v32_anclaje_metastable.png', dpi=150)
    print("  Gráfico guardado: v32_anclaje_metastable.png")
    
    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS DEL ANCLAJE METASTABLE")
    print("=" * 100)
    
    media_L_final = registro['media_L'][-1] if registro['media_L'] else 0
    rango_L_final = registro['rango_L'][-1] if registro['rango_L'] else 0
    rango_A_final = registro['rango_A'][-1] if registro['rango_A'] else 0
    
    print(f"\nAnclaje L final:")
    print(f"  media_L = {media_L_final:.4f}")
    print(f"  rango_L = {rango_L_final:.4f}")
    print(f"  rango_A_final = {rango_A_final:.3f}")
    
    if media_L_final > 0.1:
        print("\n  ✓ L desarrolló anclaje significativo")
        print("    El sistema comenzó a ralentizar el cambio en regiones persistentes")
    else:
        print("\n  ✗ L no desarrolló anclaje significativo")
    
    if rango_L_final > 0.2:
        print("  ✓ L tiene estructura espacial")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()