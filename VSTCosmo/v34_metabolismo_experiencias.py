#!/usr/bin/env python3
"""
VSTCosmo - v34: Metabolismo de Experiencias
El sistema evalúa cada experiencia por su efecto metabólico:
- ¿Aumenta diferenciación? (bien)
- ¿Aumenta homeostasis útil? (bien)
- ¿Aumenta rigidez? (mal)

No hay semántica externa. Solo consecuencia histórica.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DEL SISTEMA (heredados de v33)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 60.0
DURACION_REPOSO = 30.0  # Tiempo de metabolización entre experiencias
N_PASOS = int(DURACION_SIM / DT)
N_REPOSO = int(DURACION_REPOSO / DT)

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

A_RANGE_MIN = 0.24
A_RANGE_MAX = 0.61
PHI_RANGE_MIN = 0.08
PHI_RANGE_MAX = 0.37

HOMEOSTASIS_INTERVALO = 500

ETA_MEMORIA = 0.02
ETA_DECAY = 0.005
K_MEMORIA = 0.10
UMBRAL_MEMORIA = 0.35

ETA_L = 0.01
DECAY_L_DINAMICO = 0.002
INFLUENCIA_L = 0.5
PERSISTENCIA_UMBRAL = 0.05

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# ============================================================
# PARÁMETROS METABÓLICOS (NUEVOS)
# ============================================================
# Pesos para el índice metabólico
PESO_DIFERENCIACION = 1.0    # ↑ rango_A es bueno
PESO_HOMEOSTASIS = 0.5       # ↑ estabilidad útil es bueno
PESO_RIGIDEZ = -1.5          # ↑ rigidez (L) es malo

# Umbrales para clasificación
UMBRAL_NUTRITIVO = 0.03
UMBRAL_TOXICO = -0.02

# Retroalimentación metabólica
ETA_METABOLICO = 0.05        # rapidez con que M se ajusta al IM
ALPHA_RIGIDEZ = 0.1          # tasa de penalización de L por experiencias tóxicas

# Decaimientos entre experiencias (igual que v33)
DECAY_M = 0.995
DECAY_L_ENTRE = 0.970
DECAY_PSI = 0.900
PESO_M_PARA_A = 0.3
A_BASAL = 0.1

# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta):
    # Casos sintéticos
    if "Tono puro" in ruta:
        sr = 48000
        t = np.arange(int(sr * DURACION_SIM)) / sr
        return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    elif "Ruido blanco" in ruta:
        sr = 48000
        return sr, np.random.normal(0, 0.3, int(sr * DURACION_SIM))
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
        muestras_necesarias = int(sr * DURACION_SIM)
        if len(data) < muestras_necesarias:
            data = np.pad(data, (0, muestras_necesarias - len(data)))
        else:
            data = data[:muestras_necesarias]
        return sr, data


def inicializar_campo():
    np.random.seed(42)
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4


def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * A_BASAL


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
    cambio_local = np.abs(A - np.roll(A, 1, axis=1))
    persistencia = 1.0 - cambio_local / (cambio_local + PERSISTENCIA_UMBRAL)
    dL = (ETA_L * persistencia * (1 - L) - DECAY_L_DINAMICO * L) * DT
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
# NUEVAS FUNCIONES METABÓLICAS
# ============================================================
def extraer_estado(S):
    """Extrae métricas relevantes del estado actual"""
    Phi, A, Psi, Omega, M, L = S
    return {
        'rango_A': np.max(A) - np.min(A),
        'media_A': np.mean(A),
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_L': np.mean(L),
        'rango_L': np.max(L) - np.min(L),
        'media_M': np.mean(M),
        'variacion_A': np.mean(np.abs(A - np.roll(A, 1, axis=1)))
    }


def calcular_indice_metabolico(estado_pre, estado_post):
    """
    Calcula IM (Índice Metabólico) según el protocolo:
    IM = w1 * Δ_diferenciacion + w2 * Δ_homeostasis - w3 * Δ_rigidez
    """
    delta_dif = estado_post['rango_A'] - estado_pre['rango_A']
    delta_homeo = (estado_post['rango_A'] + estado_post['rango_Phi']) - (estado_pre['rango_A'] + estado_pre['rango_Phi'])
    delta_rigidez = estado_post['media_L'] - estado_pre['media_L']
    
    IM = (PESO_DIFERENCIACION * delta_dif + 
          PESO_HOMEOSTASIS * delta_homeo + 
          PESO_RIGIDEZ * delta_rigidez)
    
    return IM


def clasificar_experiencia(IM):
    """Clasifica la experiencia según su índice metabólico"""
    if IM > UMBRAL_NUTRITIVO:
        return "NUTRITIVA"
    elif IM < UMBRAL_TOXICO:
        return "TOXICA"
    else:
        return "NEUTRA"


def retroalimentacion_metabolica(M, L, A, IM, estado_final):
    """
    Retroalimentación interna basada en el índice metabólico:
    - M se ajusta hacia la configuración de A si la experiencia fue nutritiva
    - L se debilita si la experiencia fue tóxica
    """
    if IM > 0:
        # Experiencia positiva: reforzar memoria hacia la configuración final
        M = M + ETA_METABOLICO * IM * (A - M)
    else:
        # Experiencia negativa: decaimiento adicional de M hacia basal
        M = M * (1 - ETA_METABOLICO * abs(IM))
    
    if IM < 0:
        # Experiencia tóxica: debilitar L (romper rigidez)
        L = L * (1 - ALPHA_RIGIDEZ * abs(IM))
    
    return np.clip(M, 0.0, 1.0), np.clip(L, 0.0, 1.0)


def respirar_entre_experiencias(M, Psi, L, A):
    """Fase de reposo entre experiencias (metabolización)"""
    M = M * DECAY_M
    Psi = Psi * DECAY_PSI
    L = L * DECAY_L_ENTRE
    
    A_nueva = A_BASAL * (1 - PESO_M_PARA_A) + PESO_M_PARA_A * M
    
    return M, Psi, L, A_nueva


def simular_experiencia(S, audio_nombre, sr, audio, G, R, C):
    """Simula una experiencia completa (ingesta)"""
    Phi, A, Psi, Omega, M, L = S
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
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
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
    return (Phi, A, Psi, Omega, M, L), G, R, C


def simular_reposo(S, G, R, C):
    """Simula reposo sin entrada (metabolización)"""
    Phi, A, Psi, Omega, M, L = S
    
    for paso in range(N_REPOSO):
        muestra = 0.0  # silencio
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, M, L, k_comp)
        # Durante reposo, no actualizar M ni L con reglas metabólicas
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi, Phi, A)  # reposo
        
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C = homeostasis(A, Phi, G, R, C,
                                  A_RANGE_MIN, A_RANGE_MAX,
                                  PHI_RANGE_MIN, PHI_RANGE_MAX)
    
    return (Phi, A, Psi, Omega, M, L), G, R, C


# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================
def simular_secuencia_metabolica(orden):
    """Procesa secuencia con metabolización entre experiencias"""
    print("=" * 100)
    print("VSTCosmo - v34: Metabolismo de Experiencias")
    print("Cada experiencia se evalúa por su índice metabólico (IM)")
    print("IM = w1*Δdiferenciación + w2*Δhomeostasis + w3*Δrigidez")
    print("=" * 100)
    
    # Estado inicial
    np.random.seed(42)
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    M = inicializar_memoria()
    L = inicializar_anclaje()
    
    G, R, C = 1.0, 1.0, 1.0
    
    registro = []
    
    for exp_idx, entrada_nombre in enumerate(orden):
        print(f"\n{'='*80}")
        print(f"Experiencia {exp_idx + 1}: {entrada_nombre}")
        print(f"{'='*80}")
        
        # Estado antes de la experiencia
        S_before = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
        estado_before = extraer_estado(S_before)
        print(f"  Estado antes: rango_A={estado_before['rango_A']:.3f}, media_L={estado_before['media_L']:.3f}")
        
        # Cargar audio
        sr, audio = cargar_audio(entrada_nombre)
        
        # FASE 1: Ingesta
        print("  [Ingesta] Procesando experiencia...")
        S_after, G, R, C = simular_experiencia(S_before, entrada_nombre, sr, audio, G, R, C)
        estado_after = extraer_estado(S_after)
        print(f"    Después ingesta: rango_A={estado_after['rango_A']:.3f}, media_L={estado_after['media_L']:.3f}")
        
        # FASE 2: Reposo/Metabolización
        print("  [Metabolización] Reposo sin entrada...")
        S_post, G, R, C = simular_reposo(S_after, G, R, C)
        estado_post = extraer_estado(S_post)
        print(f"    Después reposo: rango_A={estado_post['rango_A']:.3f}, media_L={estado_post['media_L']:.3f}")
        
        # Calcular índice metabólico
        IM = calcular_indice_metabolico(estado_before, estado_post)
        clasificacion = clasificar_experiencia(IM)
        
        print(f"\n  ÍNDICE METABÓLICO: {IM:.4f} → {clasificacion}")
        
        # Retroalimentación metabólica
        Phi, A, Psi, Omega, M, L = S_post
        M, L = retroalimentacion_metabolica(M, L, A, IM, estado_post)
        S_post = (Phi, A, Psi, Omega, M, L)
        
        # Respirar (decaimiento diferencial)
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        
        # Actualizar estado para siguiente iteración
        Phi, A, Psi, Omega, M, L = S_post
        # Aplicar respiración a A
        S_final = (Phi, A, Psi, Omega, M, L)
        
        # Registrar
        registro.append({
            'experiencia': exp_idx + 1,
            'entrada': entrada_nombre,
            'IM': IM,
            'clasificacion': clasificacion,
            'rango_A_before': estado_before['rango_A'],
            'rango_A_after': estado_after['rango_A'],
            'rango_A_post': estado_post['rango_A'],
            'media_L_before': estado_before['media_L'],
            'media_L_after': estado_after['media_L'],
            'media_L_post': estado_post['media_L'],
            'media_M_post': estado_post['media_M']
        })
        
        print(f"\n  Estado final (para próxima experiencia):")
        print(f"    rango_A = {estado_post['rango_A']:.3f}")
        print(f"    media_L = {estado_post['media_L']:.3f}")
        print(f"    media_M = {estado_post['media_M']:.3f}")
    
    return registro


# ============================================================
# MAIN
# ============================================================
def main():
    # Secuencia de experiencias
    secuencia = [
        "Voz_Estudio.wav",
        "Ruido blanco",
        "Voz_Estudio.wav",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz_Estudio.wav",
        "Silencio",
        "Voz_Estudio.wav",
        "BigBang.wav",
        "Voz_Estudio.wav",
        "Viento.wav",
        "Voz_Estudio.wav"
    ]
    
    print("\n" + "=" * 100)
    print("SECUENCIA DE EXPERIENCIAS")
    print("=" * 100)
    for i, exp in enumerate(secuencia, 1):
        print(f"  {i}. {exp}")
    
    resultados = simular_secuencia_metabolica(secuencia)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 100)
    print("RESUMEN DE ÍNDICES METABÓLICOS")
    print("=" * 100)
    
    print(f"\n{'Exp':<4} | {'Entrada':<25} | {'IM':>10} | {'Clasificación':<12} | {'Δ rango_A':>10} | {'Δ L':>10}")
    print("-" * 85)
    
    for res in resultados:
        delta_rango = res['rango_A_post'] - res['rango_A_before']
        delta_L = res['media_L_post'] - res['media_L_before']
        nombre = res['entrada'][:25]
        print(f"{res['experiencia']:<4} | {nombre:<25} | {res['IM']:10.4f} | {res['clasificacion']:<12} | {delta_rango:10.3f} | {delta_L:10.3f}")
    
    # Análisis de evolución de la voz
    print("\n" + "=" * 100)
    print("EVOLUCIÓN DE LA VOZ (EXPOSICIONES REPETIDAS)")
    print("=" * 100)
    
    voz_exps = [r for r in resultados if r['entrada'] == "Voz_Estudio.wav"]
    print(f"\n{'Exposición':<12} | {'IM':>10} | {'Clasificación':<12} | {'rango_A_post':>12} | {'media_L_post':>12}")
    print("-" * 65)
    
    for i, res in enumerate(voz_exps, 1):
        print(f"Voz #{i:<7} | {res['IM']:10.4f} | {res['clasificacion']:<12} | {res['rango_A_post']:12.3f} | {res['media_L_post']:12.3f}")
    
    # Métricas acumuladas
    print("\n" + "=" * 100)
    print("MÉTRICAS ACUMULADAS")
    print("=" * 100)
    
    nutritivas = [r for r in resultados if r['clasificacion'] == "NUTRITIVA"]
    toxicas = [r for r in resultados if r['clasificacion'] == "TOXICA"]
    
    print(f"\nExperiencias nutritivas: {len(nutritivas)}")
    print(f"Experiencias tóxicas: {len(toxicas)}")
    
    # Mostrar cuáles fueron nutritivas/tóxicas
    print("\nClasificación por experiencia:")
    for res in resultados:
        print(f"  {res['experiencia']:2d}. {res['entrada'][:20]:20} → {res['clasificacion']} (IM={res['IM']:.4f})")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()