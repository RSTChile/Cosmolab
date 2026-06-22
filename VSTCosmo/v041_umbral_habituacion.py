#!/usr/bin/env python3
"""
VSTCosmo - v41: Umbral Adaptativo + Habituación
El sistema no normaliza por basal. Usa umbral relativo al basal.
Además, introduce habituación: entradas repetidas pierden valor.

IM = asimilacion_post - asimilacion_basal
Nutritivo si IM > gamma * asimilacion_basal
Tóxico si IM < -gamma * asimilacion_basal
"""

import numpy as np
import scipy.io.wavfile as wav
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS DEL SISTEMA
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 60.0
DURACION_REPOSO = 30.0
DURACION_BASAL = 10.0
N_PASOS = int(DURACION_SIM / DT)
N_REPOSO = int(DURACION_REPOSO / DT)
N_BASAL = int(DURACION_BASAL / DT)

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

DECAY_M = 0.995
DECAY_L_ENTRE = 0.970
DECAY_PSI = 0.900
PESO_M_PARA_A = 0.3
A_BASAL = 0.1

# Parámetros de asimilación
BETA_L = 1.0
BETA_VAR = 0.5

# Parámetros de K
K_INICIAL = 1.0
K_MIN = 0.5
K_MAX = 2.0
ALPHA_UP = 0.05
ALPHA_DOWN = 0.02

# NUEVOS: Umbral adaptativo y habituación
GAMMA = 0.5  # Umbral relativo al basal (0.5 = 50% del basal)
HABITUACION_DECAY = 0.95  # Cada exposición repetida reduce el impacto
HABITUACION_SUBE = 0.05   # Cuánto sube el contador de habituación


# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta, duracion=DURACION_SIM):
    if "Tono puro" in ruta:
        sr = 48000
        t = np.arange(int(sr * duracion)) / sr
        return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    elif "Ruido blanco" in ruta:
        sr = 48000
        return sr, np.random.normal(0, 0.3, int(sr * duracion))
    else:
        sr, data = wav.read(ruta)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        muestras_necesarias = int(sr * duracion)
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


def inicializar_habituacion(comidas):
    """Inicializa contador de habituación para cada tipo de comida"""
    return {comida: 0.0 for comida in comidas}


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
    es_silencio = abs(muestra) < 1e-6
    
    if es_silencio:
        perfil_2d = np.zeros((DIM_FREQ, 1))
        mod_entrada = 1.0
        mod_entrada_decay = 1.0
    else:
        perfil = perfil_modulacion(muestra)
        perfil_2d = perfil.reshape(-1, 1)
        mod_entrada = (1 + MOD_GENERACION * perfil_2d)
        mod_entrada_decay = 1 - MOD_DECAY * perfil_2d
    
    promedio_local = vecinos(Phi)
    
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = ganancia_gen * desviacion * (1 - desviacion**2)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)
    generacion = generacion_base * mod_entrada * mod_memoria
    
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


def extraer_estado(S):
    Phi, A, Psi, Omega, M, L = S
    return {
        'rango_A': np.max(A) - np.min(A),
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_L': np.mean(L),
        'media_M': np.mean(M),
        'var_A': np.var(A)
    }


def calcular_variacion_A(A_prev, A_curr):
    return np.mean(np.abs(A_curr - A_prev))


def calcular_asimilacion(delta_rango_A, delta_rango_Phi, delta_L, variacion_A, K):
    intensidad_base = abs(delta_rango_A) + abs(delta_rango_Phi)
    intensidad = intensidad_base * K
    costo = BETA_L * max(0, delta_L) + BETA_VAR * variacion_A
    asimilacion = intensidad * np.exp(-costo)
    return asimilacion, intensidad_base, intensidad, costo


def actualizar_K(K, IM, basal):
    """Actualiza K con habituación: menos cambio cuando hay habituación"""
    # La habituación reduce el impacto en K
    # Esto se maneja externamente, aquí solo la actualización base
    if IM > 0:
        K = K + ALPHA_UP * IM
    elif IM < 0:
        K = K + ALPHA_DOWN * IM
    return np.clip(K, K_MIN, K_MAX)


def actualizar_habituacion(habituacion, comida, delta_t=1.0):
    """Decaimiento natural + aumento al exponerse"""
    # Decaimiento natural (olvido)
    for c in habituacion:
        habituacion[c] *= HABITUACION_DECAY
    
    # Aumento por esta exposición
    habituacion[comida] += HABITUACION_SUBE
    
    # Limitar a [0, 1]
    habituacion[comida] = min(habituacion[comida], 1.0)
    
    return habituacion


def calcular_umbral_adaptativo(basal, habituacion_comida):
    """
    Umbral = gamma * basal * (1 + habituacion)
    Cuando hay habituación, el umbral aumenta (más difícil ser nutritivo)
    """
    return GAMMA * basal * (1 + habituacion_comida)


def medir_asimilacion_basal(S, G, R, C, K):
    Phi, A, Psi, Omega, M, L = S
    
    A_prev = A.copy()
    rango_A_pre = np.max(A) - np.min(A)
    rango_Phi_pre = np.max(Phi) - np.min(Phi)
    media_L_pre = np.mean(L)
    
    Phi_temp = Phi.copy()
    A_temp = A.copy()
    Psi_temp = Psi.copy()
    Omega_temp = Omega.copy()
    M_temp = M.copy()
    L_temp = L.copy()
    
    Phi_prev = Phi_temp.copy()
    Phi_prev2 = Phi_temp.copy()
    
    for paso in range(N_BASAL):
        muestra = 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A_temp = actualizar_atencion(A_temp, Phi_temp, Psi_temp, Omega_temp, M_temp, L_temp, k_comp)
        Phi_temp = actualizar_campo(Phi_temp, A_temp, muestra, Psi_temp, ganancia_gen, bloqueo_max)
        Psi_temp = actualizar_memoria_estabilidad(Psi_temp, Phi_temp, A_temp)
        Omega_temp = actualizar_memoria_coherencia(Omega_temp, Phi_temp, Phi_prev, Phi_prev2, A_temp)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi_temp.copy()
    
    rango_A_post = np.max(A_temp) - np.min(A_temp)
    rango_Phi_post = np.max(Phi_temp) - np.min(Phi_temp)
    media_L_post = np.mean(L_temp)
    variacion_A = calcular_variacion_A(A_prev, A_temp)
    
    delta_rango_A = rango_A_post - rango_A_pre
    delta_rango_Phi = rango_Phi_post - rango_Phi_pre
    delta_L = media_L_post - media_L_pre
    
    asimilacion_basal, _, _, _ = calcular_asimilacion(
        delta_rango_A, delta_rango_Phi, delta_L, variacion_A, K
    )
    
    S_basal = (Phi_temp, A_temp, Psi_temp, Omega_temp, M_temp, L_temp)
    
    return asimilacion_basal, S_basal


def simular_experiencia(S, entrada_nombre, G, R, C):
    Phi, A, Psi, Omega, M, L = S
    
    A_prev = A.copy()
    rango_A_pre = np.max(A) - np.min(A)
    rango_Phi_pre = np.max(Phi) - np.min(Phi)
    media_L_pre = np.mean(L)
    
    sr, audio = cargar_audio(entrada_nombre)
    
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
    
    rango_A_post = np.max(A) - np.min(A)
    rango_Phi_post = np.max(Phi) - np.min(Phi)
    media_L_post = np.mean(L)
    variacion_A = calcular_variacion_A(A_prev, A)
    
    metricas = {
        'delta_rango_A': rango_A_post - rango_A_pre,
        'delta_rango_Phi': rango_Phi_post - rango_Phi_pre,
        'delta_L': media_L_post - media_L_pre,
        'variacion_A': variacion_A,
        'rango_A_post': rango_A_post,
        'media_L_post': media_L_post
    }
    
    return (Phi, A, Psi, Omega, M, L), G, R, C, metricas


def simular_reposo(S, G, R, C):
    Phi, A, Psi, Omega, M, L = S
    
    for paso in range(N_REPOSO):
        muestra = 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, M, L, k_comp)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi, Phi, A)
        
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C = homeostasis(A, Phi, G, R, C,
                                  A_RANGE_MIN, A_RANGE_MAX,
                                  PHI_RANGE_MIN, PHI_RANGE_MAX)
    
    return (Phi, A, Psi, Omega, M, L), G, R, C


def respirar_entre_experiencias(M, Psi, L, A):
    M = M * DECAY_M
    Psi = Psi * DECAY_PSI
    L = L * DECAY_L_ENTRE
    A_nueva = A_BASAL * (1 - PESO_M_PARA_A) + PESO_M_PARA_A * M
    return M, Psi, L, A_nueva


# ============================================================
# SIMULACIÓN PRINCIPAL CON HABITUACIÓN
# ============================================================
def simular_dieta(comidas, n_ciclos=3):
    print("=" * 100)
    print("VSTCosmo - v41: Umbral Adaptativo + Habituación")
    print("Nutritivo si IM > gamma * basal * (1 + habituacion)")
    print("gamma = 0.5. Habituación reduce el impacto de entradas repetidas.")
    print("=" * 100)
    
    # Estado inicial
    np.random.seed(42)
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    M = inicializar_memoria()
    L = inicializar_anclaje()
    K = K_INICIAL
    
    # Habituación
    habituacion = inicializar_habituacion(comidas)
    
    G, R, C = 1.0, 1.0, 1.0
    
    # Reposo inicial
    print("\n[Reposo inicial] Estabilizando sistema...")
    S_actual, G, R, C = simular_reposo((Phi, A, Psi, Omega, M, L), G, R, C)
    Phi, A, Psi, Omega, M, L = S_actual
    
    registro = []
    experiencia_global = 0
    
    for ciclo in range(n_ciclos):
        print(f"\n{'#'*100}")
        print(f"CICLO {ciclo + 1} (K actual = {K:.4f})")
        print(f"{'#'*100}")
        
        for comida_idx, comida in enumerate(comidas):
            experiencia_global += 1
            
            print(f"\n  [{experiencia_global}] Comida: {comida}")
            
            # MEDIR BASAL REAL
            S_basal_antes = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
            asimilacion_basal, S_despues_basal = medir_asimilacion_basal(S_basal_antes, G, R, C, K)
            
            print(f"    Basal real: asimilacion = {asimilacion_basal:.4f}")
            print(f"    Habituación para {comida}: {habituacion[comida]:.3f}")
            
            Umbral = calcular_umbral_adaptativo(asimilacion_basal, habituacion[comida])
            print(f"    Umbral adaptativo: {Umbral:.4f}")
            
            # COMER
            S_despues_comer, G, R, C, metricas = simular_experiencia(S_despues_basal, comida, G, R, C)
            
            # METABOLIZAR
            S_post, G, R, C = simular_reposo(S_despues_comer, G, R, C)
            
            estado_post = extraer_estado(S_post)
            
            # Calcular asimilación POST
            asimilacion_post, intensidad_base, intensidad_K, costo = calcular_asimilacion(
                metricas['delta_rango_A'],
                metricas['delta_rango_Phi'],
                metricas['delta_L'],
                metricas['variacion_A'],
                K
            )
            
            # ÍNDICE METABÓLICO
            IM = asimilacion_post - asimilacion_basal
            
            # CLASIFICACIÓN CON UMBRAL ADAPTATIVO
            if IM > Umbral:
                clasificacion = "NUTRITIVA ✨"
            elif IM < -Umbral:
                clasificacion = "TÓXICA 💀"
            else:
                clasificacion = "NEUTRA"
            
            # Actualizar K (con habituación)
            K_antes = K
            K = actualizar_K(K, IM, asimilacion_basal)
            
            # Actualizar memoria M
            if estado_post['rango_A'] > UMBRAL_MEMORIA:
                _, A_actual, _, _, M_actual, _ = S_post
                M = M_actual + ETA_MEMORIA * (A_actual - M_actual)
            else:
                M = M * (1 - ETA_DECAY * max(0, -IM))
            M = np.clip(M, 0.0, 1.0)
            
            # Actualizar habituación
            habituacion = actualizar_habituacion(habituacion, comida)
            
            # RESPIRAR
            M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
            
            # Actualizar estado
            Phi, A, Psi, Omega, M, L = S_post
            
            print(f"    Δrango_A = {metricas['delta_rango_A']:+.4f}")
            print(f"    Δrango_Φ = {metricas['delta_rango_Phi']:+.4f}")
            print(f"    ΔL = {metricas['delta_L']:+.4f}")
            print(f"    asimilación_post = {asimilacion_post:.4f}")
            print(f"    IM = {asimilacion_post:.4f} - {asimilacion_basal:.4f} = {IM:+.4f}")
            print(f"    Umbral = {Umbral:.4f} → {clasificacion}")
            print(f"    K: {K_antes:.4f} → {K:.4f}")
            print(f"    Estado: rango_A={estado_post['rango_A']:.3f}, media_L={estado_post['media_L']:.3f}")
            
            registro.append({
                'ciclo': ciclo + 1,
                'experiencia': experiencia_global,
                'comida': comida,
                'IM': IM,
                'basal': asimilacion_basal,
                'umbral': Umbral,
                'clasificacion': clasificacion,
                'habituacion': habituacion[comida],
                'K': K,
                'K_antes': K_antes,
                'rango_A_post': estado_post['rango_A'],
                'media_L_post': estado_post['media_L']
            })
    
    return registro


# ============================================================
# MAIN
# ============================================================
def main():
    comidas = [
        "Ruido blanco",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz+Viento_1.wav",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Ruido blanco"
    ]
    
    print("\n" + "=" * 100)
    print("MENÚ DE COMIDAS")
    print("=" * 100)
    for i, comida in enumerate(comidas, 1):
        print(f"  {i}. {comida}")
    
    resultados = simular_dieta(comidas, n_ciclos=3)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 100)
    print("RESUMEN DE RESULTADOS (con umbral adaptativo)")
    print("=" * 100)
    
    print(f"\n{'Exp':<4} | {'Comida':<25} | {'IM':>10} | {'Basal':>10} | {'Umbral':>10} | {'Clasif':<12} | {'Hab':>6}")
    print("-" * 95)
    
    for res in resultados:
        print(f"{res['experiencia']:<4} | {res['comida'][:25]:<25} | {res['IM']:10.4f} | {res['basal']:10.4f} | {res['umbral']:10.4f} | {res['clasificacion']:<12} | {res['habituacion']:6.3f}")
    
    # Estadísticas por comida
    print("\n" + "=" * 100)
    print("ESTADÍSTICAS POR COMIDA (con umbral adaptativo)")
    print("=" * 100)
    
    comidas_unicas = set(r['comida'] for r in resultados)
    for comida in sorted(comidas_unicas):
        datos = [r for r in resultados if r['comida'] == comida]
        ims = [r['IM'] for r in datos]
        clasifs = [r['clasificacion'] for r in datos]
        
        im_prom = np.mean(ims)
        nutritivas = clasifs.count("NUTRITIVA ✨")
        toxicas = clasifs.count("TÓXICA 💀")
        
        print(f"  {comida:<25} | IM_prom = {im_prom:+.4f} | Nutritivas={nutritivas} | Tóxicas={toxicas} | n={len(datos)}")
    
    # Evolución de K
    print("\n" + "=" * 100)
    print("EVOLUCIÓN DE K")
    print("=" * 100)
    
    for res in resultados:
        print(f"  Exp {res['experiencia']:2d} ({res['comida'][:20]:20}): K = {res['K']:.4f}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()