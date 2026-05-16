#!/usr/bin/env python3
"""
VSTCosmos - v44: Asimetría Ganancia/Pérdida
El sistema NO solo busca lo nutritivo.
APRENDE A EVITAR lo que le hizo mal.

IM negativo → impacto mayor en decisiones que IM positivo.
El miedo pesa más que el placer.
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

# Umbrales de clasificación
UMBRAL_NUTRITIVO = 0.005
UMBRAL_TOXICO = -0.005

# Habituación
HABITUACION_DECAY = 0.95
HABITUACION_SUBE = 0.05
HABITUACION_MAX = 0.8
HABITUACION_MODULACION = 0.8

# Parámetros de selección autónoma
EXPLORACION_INICIAL = 0.5
EXPLORACION_DECAY = 0.95
EXPLORACION_MIN = 0.1

# NUEVO: ASIMETRÍA GANANCIA/PÉRDIDA
PESO_PERDIDA = 2.0          # El dolor pesa el doble que el placer
MEMORIA_TOXICA_DECAY = 0.95  # Cuánto dura el recuerdo de lo tóxico
MEMORIA_TOXICA_SUBE = 0.2    # Qué tanto se aprende de una experiencia tóxica


# ============================================================
# FUNCIONES BASE (sin cambios)
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


def inicializar_memoria_toxica(opciones):
    """Memoria de experiencias tóxicas: cuánto evitarlas"""
    return {comida: 0.0 for comida in opciones}


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


def actualizar_K(K, IM):
    if IM > 0:
        K = K + ALPHA_UP * IM
    elif IM < 0:
        K = K + ALPHA_DOWN * IM
    return np.clip(K, K_MIN, K_MAX)


def actualizar_habituacion(habituacion, comida):
    for c in habituacion:
        habituacion[c] *= HABITUACION_DECAY
    habituacion[comida] += HABITUACION_SUBE
    habituacion[comida] = min(habituacion[comida], HABITUACION_MAX)
    return habituacion


def aplicar_habituacion(muestra, habituacion_comida):
    factor = 1.0 - (habituacion_comida * HABITUACION_MODULACION)
    return muestra * factor


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


def simular_experiencia(S, entrada_nombre, G, R, C, habituacion_comida):
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
        
        muestra = aplicar_habituacion(muestra, habituacion_comida)
        
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
# NUEVAS FUNCIONES CON ASIMETRÍA GANANCIA/PÉRDIDA
# ============================================================
def actualizar_memoria_toxica(memoria_toxica, comida, IM):
    """
    Actualiza la memoria de experiencias tóxicas.
    El dolor (IM negativo) pesa más que el placer.
    """
    # Decaimiento general
    for c in memoria_toxica:
        memoria_toxica[c] *= MEMORIA_TOXICA_DECAY
    
    if IM < 0:
        # Experiencia tóxica: se aprende FUERTE
        memoria_toxica[comida] += MEMORIA_TOXICA_SUBE * abs(IM) * PESO_PERDIDA
    elif IM > 0:
        # Experiencia positiva: alivia el miedo, pero lentamente
        memoria_toxica[comida] -= MEMORIA_TOXICA_SUBE * IM * 0.3
    
    # Limitar
    memoria_toxica[comida] = np.clip(memoria_toxica[comida], 0.0, 1.0)
    
    return memoria_toxica


def calcular_utilidad_con_miedo(comida, historial_im, habituacion, memoria_toxica, exploracion_prob):
    """
    Utilidad = valor_metabolico - habituacion - MIEDO
    El miedo (memoria_toxica) reduce la utilidad.
    """
    # Valor metabólico (historial de IM)
    ims = [h['IM'] for h in historial_im if h['comida'] == comida]
    
    if len(ims) == 0:
        return 0.5 + exploracion_prob
    
    pesos = np.exp(-0.3 * np.arange(len(ims))[::-1])
    valor_metabolico = np.average(ims, weights=pesos)
    
    # Penalizaciones
    hab = habituacion.get(comida, 0.0)
    miedo = memoria_toxica.get(comida, 0.0) * PESO_PERDIDA
    
    # Utilidad con asimetría
    utilidad = valor_metabolico - hab - miedo
    
    # Ruido de exploración
    utilidad += np.random.normal(0, exploracion_prob * 0.1)
    
    return utilidad


def elegir_proxima_comida_con_miedo(opciones, historial_im, habituacion, memoria_toxica, exploracion_prob):
    utilidades = {}
    for comida in opciones:
        utilidad = calcular_utilidad_con_miedo(comida, historial_im, habituacion, memoria_toxica, exploracion_prob)
        utilidades[comida] = utilidad
    
    mejor_comida = max(utilidades, key=utilidades.get)
    return mejor_comida, utilidades


# ============================================================
# SIMULACIÓN PRINCIPAL CON MIEDO
# ============================================================
def simular_vida_con_miedo(opciones, n_experiencias=30):
    print("=" * 100)
    print("VSTCosmos - v44: Asimetría Ganancia/Pérdida")
    print("El sistema APRENDE A EVITAR lo que le hizo mal.")
    print("El miedo pesa más que el placer.")
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
    
    habituacion = {comida: 0.0 for comida in opciones}
    memoria_toxica = inicializar_memoria_toxica(opciones)
    historial_im = []
    
    G, R, C = 1.0, 1.0, 1.0
    
    print("\n[Reposo inicial] Estabilizando sistema...")
    S_actual, G, R, C = simular_reposo((Phi, A, Psi, Omega, M, L), G, R, C)
    Phi, A, Psi, Omega, M, L = S_actual
    
    exploracion_prob = EXPLORACION_INICIAL
    
    for exp in range(n_experiencias):
        print(f"\n{'#'*100}")
        print(f"EXPERIENCIA {exp + 1}")
        print(f"K = {K:.4f} | Exploración = {exploracion_prob:.3f}")
        print(f"{'#'*100}")
        
        # Elegir comida (considerando el MIEDO)
        comida_elegida, utilidades = elegir_proxima_comida_con_miedo(
            opciones, historial_im, habituacion, memoria_toxica, exploracion_prob
        )
        
        print(f"\n  🍽️  Elige: {comida_elegida}")
        print(f"  📊 Utilidades: {', '.join([f'{k[:12]}={v:.3f}' for k, v in sorted(utilidades.items(), key=lambda x: x[1], reverse=True)[:4]])}")
        print(f"  😨 Miedo acumulado para {comida_elegida}: {memoria_toxica[comida_elegida]:.3f}")
        
        # Medir basal
        S_basal_antes = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
        asimilacion_basal, S_despues_basal = medir_asimilacion_basal(S_basal_antes, G, R, C, K)
        
        # Comer
        S_despues_comer, G, R, C, metricas = simular_experiencia(
            S_despues_basal, comida_elegida, G, R, C, habituacion[comida_elegida]
        )
        
        # Metabolizar
        S_post, G, R, C = simular_reposo(S_despues_comer, G, R, C)
        
        # Calcular IM
        estado_post = extraer_estado(S_post)
        asimilacion_post, _, _, _ = calcular_asimilacion(
            metricas['delta_rango_A'],
            metricas['delta_rango_Phi'],
            metricas['delta_L'],
            metricas['variacion_A'],
            K
        )
        
        IM = asimilacion_post - asimilacion_basal
        
        # Clasificación
        if IM > UMBRAL_NUTRITIVO:
            clasificacion = "NUTRITIVA ✨"
        elif IM < UMBRAL_TOXICO:
            clasificacion = "TÓXICA 💀💀💀"
        else:
            clasificacion = "NEUTRA"
        
        # Actualizar K
        K_antes = K
        K = actualizar_K(K, IM)
        
        # Actualizar memoria
        if estado_post['rango_A'] > UMBRAL_MEMORIA:
            _, A_actual, _, _, M_actual, _ = S_post
            M = M_actual + ETA_MEMORIA * (A_actual - M_actual)
        else:
            M = M * (1 - ETA_DECAY * max(0, -IM))
        M = np.clip(M, 0.0, 1.0)
        
        # Actualizar habituación
        habituacion = actualizar_habituacion(habituacion, comida_elegida)
        
        # ACTUALIZAR MEMORIA TÓXICA (el miedo)
        memoria_toxica = actualizar_memoria_toxica(memoria_toxica, comida_elegida, IM)
        
        # Respirar
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        Phi, A, Psi, Omega, M, L = S_post
        
        # Registrar
        historial_im.append({
            'experiencia': exp + 1,
            'comida': comida_elegida,
            'IM': IM,
            'clasificacion': clasificacion,
            'basal': asimilacion_basal,
            'K': K,
            'miedo': memoria_toxica[comida_elegida]
        })
        
        # Reducir exploración
        exploracion_prob = max(exploracion_prob * EXPLORACION_DECAY, EXPLORACION_MIN)
        
        print(f"\n  📈 RESULTADO:")
        print(f"    IM = {IM:+.4f} → {clasificacion}")
        print(f"    K: {K_antes:.4f} → {K:.4f}")
        print(f"    Miedo aumentado: {memoria_toxica[comida_elegida]:.3f}")
        print(f"    Estado: rango_A={estado_post['rango_A']:.3f}, media_L={estado_post['media_L']:.3f}")
    
    return historial_im, K, habituacion, memoria_toxica


# ============================================================
# MAIN
# ============================================================
def main():
    opciones = [
        "Ruido blanco",
        "Tono puro",
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz+Viento_1.wav"
    ]
    
    print("\n" + "=" * 100)
    print("MENÚ DE COMIDAS")
    print("=" * 100)
    for i, opcion in enumerate(opciones, 1):
        print(f"  {i}. {opcion}")
    
    print("\n⚠️  El sistema ahora APRENDE A EVITAR lo que le hace mal.")
    print("   El miedo pesa el doble que el placer.")
    
    historial, K_final, habituacion_final, memoria_toxica_final = simular_vida_con_miedo(opciones, n_experiencias=30)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 100)
    print("HISTORIAL DE DECISIONES CON MIEDO")
    print("=" * 100)
    
    print(f"\n{'Exp':<4} | {'Comida':<22} | {'IM':>10} | {'Clasif':<14} | {'Miedo':>8}")
    print("-" * 75)
    
    for h in historial:
        print(f"{h['experiencia']:<4} | {h['comida'][:22]:<22} | {h['IM']:10.4f} | {h['clasificacion']:<14} | {h['miedo']:8.3f}")
    
    print("\n" + "=" * 100)
    print("ESTADÍSTICAS FINALES")
    print("=" * 100)
    
    print(f"\nK final: {K_final:.4f}")
    
    print(f"\n📊 Habituación final:")
    for comida, hab in sorted(habituacion_final.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comida:<25}: {hab:.3f}")
    
    print(f"\n😨 MIEDO acumulado (memoria tóxica):")
    for comida, miedo in sorted(memoria_toxica_final.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comida:<25}: {miedo:.3f}")
    
    # Estadísticas de elecciones
    print(f"\n🍽️  FRECUENCIA DE ELECCIONES:")
    conteo = {}
    for h in historial:
        conteo[h['comida']] = conteo.get(h['comida'], 0) + 1
    for comida, count in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comida:<25}: {count} veces")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("El sistema APRENDIÓ A EVITAR lo que le hace mal.")
    print("=" * 100)


if __name__ == "__main__":
    main()