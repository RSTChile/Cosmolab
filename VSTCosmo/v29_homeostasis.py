#!/usr/bin/env python3
"""
VSTCosmo - v29: Homeostasis del régimen fértil
El sistema regula su propia distancia al colapso.
No optimiza señal. Mantiene rangos viables de A y Φ.
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
DURACION_SIM = 60.0  # más largo para estabilización
N_PASOS = int(DURACION_SIM / DT)

# Parámetros base (de v28)
DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION_BASE = 0.05
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
K_COMP_BASE = 0.05

# Límites globales
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

# Módulos de entrada
MOD_DECAY = 1.0
MOD_GENERACION = 1.5

# Parámetros de Ψ
TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_ESTABILIDAD = 0.1
BLOQUEO_MAXIMO_BASE = 0.8

# Parámetros de Ω
TASA_OMEGA = 0.15
DISIPACION_OMEGA = 0.05
FUERZA_COHERENCIA = 0.2

# Rangos viables para homeostasis (no óptimos, solo para evitar colapso)
A_RANGE_MIN = 0.25
A_RANGE_MAX = 0.55
PHI_RANGE_MIN = 0.08
PHI_RANGE_MAX = 0.35

# Homeostasis: factores multiplicadores (inician en 1.0)
G = 1.0   # generación
R = 1.0   # restricción (BLOQUEO_MAXIMO)
C = 1.0   # competencia (K_COMP)

# Intervalo de ajuste homeostático (cada 500 pasos ~5 segundos)
HOMEOSTASIS_INTERVALO = 500

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

ENTRADA = "Voz_Estudio.wav"  # una sola entrada para observar homeostasis


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
    np.random.seed(42)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4


def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.1


def inicializar_memoria():
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


def actualizar_atencion(A, Phi, Psi, Omega, k_comp):
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
    
    # Competencia por recurso
    A_mean = np.mean(A)
    competencia = -k_comp * (A - A_mean)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          competencia)
    
    # Competencia global suave
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


# ============================================================
# HOMEOSTASIS
# ============================================================
def homeostasis(A, Phi, G, R, C):
    rango_A = np.max(A) - np.min(A)
    rango_Phi = np.max(Phi) - np.min(Phi)
    
    ajustes = []
    
    # Regulación de rango_A
    if rango_A < A_RANGE_MIN:
        G *= 1.02
        C *= 1.02
        ajustes.append(f"bajo rango_A: G={G:.3f}, C={C:.3f}")
    elif rango_A > A_RANGE_MAX:
        R *= 1.02
        G *= 0.98
        ajustes.append(f"alto rango_A: R={R:.3f}, G={G:.3f}")
    
    # Regulación de rango_Phi
    if rango_Phi < PHI_RANGE_MIN:
        G *= 1.02
        ajustes.append(f"bajo rango_Phi: G={G:.3f}")
    elif rango_Phi > PHI_RANGE_MAX:
        R *= 1.03
        G *= 0.97
        ajustes.append(f"alto rango_Phi: R={R:.3f}, G={G:.3f}")
    
    # Mantener en rangos razonables
    G = np.clip(G, 0.3, 3.0)
    R = np.clip(R, 0.3, 3.0)
    C = np.clip(C, 0.3, 3.0)
    
    return G, R, C, ajustes


# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v29: Homeostasis del régimen fértil")
    print("El sistema regula su distancia al colapso (cierre vs saturación)")
    print(f"Duración de simulación: {DURACION_SIM} segundos")
    print("=" * 100)
    
    sr = 48000
    duracion = DURACION_SIM
    
    print("\n[1] Cargando archivo...")
    _, audio = cargar_audio(ENTRADA)
    print(f"    {ENTRADA}: {len(audio)/sr:.1f}s")
    
    n_muestras = int(duracion * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    audio = audio / np.max(np.abs(audio))
    
    print("\n[2] Inicializando sistema...")
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
    # Parámetros regulados por homeostasis
    G = 1.0
    R = 1.0
    C = 1.0
    
    # Registro para análisis
    registro = {
        'paso': [], 'tiempo': [],
        'rango_A': [], 'rango_Phi': [],
        'media_A': [], 'media_Phi': [],
        'G': [], 'R': [], 'C': []
    }
    
    print("\n[3] Ejecutando simulación con homeostasis...")
    print("-" * 80)
    
    for paso in range(N_PASOS):
        t = paso * DT
        
        # Muestra de audio
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        # Parámetros actuales (modulados por homeostasis)
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        # Actualizar dinámica
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        # Aplicar homeostasis periódicamente
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C, ajustes = homeostasis(A, Phi, G, R, C)
            if ajustes:
                t_seg = paso * DT
                print(f"  t={t_seg:.1f}s: " + " | ".join(ajustes))
        
        # Registrar cada cierto tiempo
        if paso % (N_PASOS // 100) == 0:
            registro['paso'].append(paso)
            registro['tiempo'].append(t)
            registro['rango_A'].append(np.max(A) - np.min(A))
            registro['rango_Phi'].append(np.max(Phi) - np.min(Phi))
            registro['media_A'].append(np.mean(A))
            registro['media_Phi'].append(np.mean(Phi))
            registro['G'].append(G)
            registro['R'].append(R)
            registro['C'].append(C)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
        
        # Progreso
        if paso % (N_PASOS // 10) == 0:
            pct = 100 * paso // N_PASOS
            print(f"  Progreso: {pct}% | rango_A={np.max(A)-np.min(A):.3f} | rango_Phi={np.max(Phi)-np.min(Phi):.3f}")
    
    print("-" * 80)
    print("Simulación completada.")
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("\n[4] Generando visualización...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Gráfico 1: rango_A y rango_Phi
    ax = axes[0]
    ax.plot(registro['tiempo'], registro['rango_A'], 'b-', label='rango A', linewidth=1)
    ax.plot(registro['tiempo'], registro['rango_Phi'], 'r-', label='rango Φ', linewidth=1)
    ax.axhline(y=A_RANGE_MIN, color='b', linestyle='--', alpha=0.5, label='A min')
    ax.axhline(y=A_RANGE_MAX, color='b', linestyle='--', alpha=0.5, label='A max')
    ax.axhline(y=PHI_RANGE_MIN, color='r', linestyle='--', alpha=0.5, label='Φ min')
    ax.axhline(y=PHI_RANGE_MAX, color='r', linestyle='--', alpha=0.5, label='Φ max')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Rango')
    ax.set_title('Rangos de A y Φ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Factores homeostáticos
    ax = axes[1]
    ax.plot(registro['tiempo'], registro['G'], 'g-', label='G (generación)', linewidth=1)
    ax.plot(registro['tiempo'], registro['R'], 'm-', label='R (restricción)', linewidth=1)
    ax.plot(registro['tiempo'], registro['C'], 'c-', label='C (competencia)', linewidth=1)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Factor multiplicador')
    ax.set_title('Factores homeostáticos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Media de A y Φ
    ax = axes[2]
    ax.plot(registro['tiempo'], registro['media_A'], 'b-', label='media A', linewidth=1)
    ax.plot(registro['tiempo'], registro['media_Phi'], 'r-', label='media Φ', linewidth=1)
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Media')
    ax.set_title('Medias de A y Φ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('VSTCosmo v29 - Homeostasis del régimen fértil', fontsize=14)
    plt.tight_layout()
    plt.savefig('v29_homeostasis.png', dpi=150)
    print("  Gráfico guardado: v29_homeostasis.png")
    
    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS DEL RÉGIMEN")
    print("=" * 100)
    
    rango_A_final = registro['rango_A'][-1] if registro['rango_A'] else 0
    rango_Phi_final = registro['rango_Phi'][-1] if registro['rango_Phi'] else 0
    
    print(f"\nRango final de A: {rango_A_final:.3f}")
    print(f"Rango final de Φ: {rango_Phi_final:.3f}")
    
    if A_RANGE_MIN < rango_A_final < A_RANGE_MAX:
        print("  ✓ A en rango viable")
    else:
        print("  ✗ A fuera de rango viable")
    
    if PHI_RANGE_MIN < rango_Phi_final < PHI_RANGE_MAX:
        print("  ✓ Φ en rango viable")
    else:
        print("  ✗ Φ fuera de rango viable")
    
    print("\nFactores homeostáticos finales:")
    print(f"  G (generación) = {registro['G'][-1]:.3f}")
    print(f"  R (restricción) = {registro['R'][-1]:.3f}")
    print(f"  C (competencia) = {registro['C'][-1]:.3f}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()