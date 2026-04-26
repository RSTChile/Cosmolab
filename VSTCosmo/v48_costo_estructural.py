#!/usr/bin/env python3
"""
VSTCosmos - v48: Costo Estructural Real
El sistema NO tiene "miedo". No hay penalizaciones externas.
Ciertas entradas degradan su capacidad interna (K).
La evitación emerge cuando ya no puede sostener la interacción.
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
DURACION_SIM = 20.0
DURACION_REPOSO = 15.0
DURACION_BASAL = 8.0
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

# Parámetros de K (modificados para costo estructural)
K_INICIAL = 1.0
K_MIN = 0.3
K_MAX = 1.8
ALPHA_UP = 0.05
BETA_COST = 0.25      # Cuánto penaliza el costo estructural a K

# Umbrales de clasificación (solo observación)
UMBRAL_NUTRITIVO = 0.005
UMBRAL_TOXICO = -0.005

# Habituación (se mantiene)
HABITUACION_DECAY = 0.95
HABITUACION_SUBE = 0.05
HABITUACION_MAX = 0.8
HABITUACION_MODULACION = 0.8

# Parámetros de selección autónoma
EXPLORACION_INICIAL = 0.5
EXPLORACION_DECAY = 0.95
EXPLORACION_MIN = 0.1

# Pesos para costo estructural
PESO_PERDIDA_DIFERENCIACION = 1.0
PESO_AUMENTO_RIGIDEZ = 1.5
PESO_PERDIDA_VARIABILIDAD = 1.0


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
        return sr, np.random.normal(0, 0.5, int(sr * duracion))
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


def actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max, K):
    es_silencio = abs(muestra) < 1e-6
    
    if es_silencio:
        perfil_2d = np.zeros((DIM_FREQ, 1))
        mod_entrada = 1.0
        mod_entrada_decay = 1.0
    else:
        perfil = perfil_modulacion(muestra)
        perfil_2d = perfil.reshape(-1, 1)
        # NUEVO: modulación por capacidad metabólica K
        mod_entrada = (1 + MOD_GENERACION * perfil_2d) * K
        mod_entrada_decay = (1 - MOD_DECAY * perfil_2d) * K
    
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


def calcular_costo_estructural(estado_before, estado_after):
    """
    Calcula cuánto daño estructural causó la experiencia.
    Sin interpretación psicológica. Solo pérdida de capacidad.
    """
    perdida_diferenciacion = max(0, estado_before['rango_Phi'] - estado_after['rango_Phi'])
    aumento_rigidez = max(0, estado_after['media_L'] - estado_before['media_L'])
    perdida_variabilidad = max(0, estado_before['var_A'] - estado_after['var_A'])
    
    costo = (PESO_PERDIDA_DIFERENCIACION * perdida_diferenciacion +
             PESO_AUMENTO_RIGIDEZ * aumento_rigidez +
             PESO_PERDIDA_VARIABILIDAD * perdida_variabilidad)
    
    return costo


def calcular_asimilacion(delta_rango_A, delta_rango_Phi, delta_L, variacion_A, K):
    intensidad_base = abs(delta_rango_A) + abs(delta_rango_Phi)
    intensidad = intensidad_base * K
    costo = BETA_L * max(0, delta_L) + BETA_VAR * variacion_A
    asimilacion = intensidad * np.exp(-costo)
    return asimilacion, intensidad_base, intensidad, costo


def actualizar_K(K, IM, costo_estructural):
    """
    K aumenta con experiencias nutritivas (IM > 0).
    K disminuye con costo estructural (pérdida real de capacidad).
    NO hay "miedo". Solo consecuencias.
    """
    if IM > 0:
        K = K + ALPHA_UP * min(IM, 0.1)  # Cap para evitar explosión
    K = K - BETA_COST * costo_estructural
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
        Phi_temp = actualizar_campo(Phi_temp, A_temp, muestra, Psi_temp, ganancia_gen, bloqueo_max, K)
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


def simular_experiencia(S, entrada_nombre, G, R, C, habituacion_comida, K):
    Phi, A, Psi, Omega, M, L = S
    
    A_prev = A.copy()
    rango_A_pre = np.max(A) - np.min(A)
    rango_Phi_pre = np.max(Phi) - np.min(Phi)
    media_L_pre = np.mean(L)
    var_A_pre = np.var(A)
    
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
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max, K)
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
    var_A_post = np.var(A)
    
    metricas = {
        'delta_rango_A': rango_A_post - rango_A_pre,
        'delta_rango_Phi': rango_Phi_post - rango_Phi_pre,
        'delta_L': media_L_post - media_L_pre,
        'variacion_A': variacion_A,
        'rango_A_post': rango_A_post,
        'media_L_post': media_L_post,
        'rango_Phi_before': rango_Phi_pre,
        'rango_Phi_after': rango_Phi_post,
        'var_A_before': var_A_pre,
        'var_A_after': var_A_post,
        'media_L_before': media_L_pre
    }
    
    return (Phi, A, Psi, Omega, M, L), G, R, C, metricas


def simular_reposo(S, G, R, C, K):
    Phi, A, Psi, Omega, M, L = S
    
    for paso in range(N_REPOSO):
        muestra = 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, M, L, k_comp)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max, K)
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


def calcular_utilidad_simple(comida, historial_im, habituacion, exploracion_prob):
    """
    Utilidad SIN miedo. Solo basada en:
    - Valor metabólico pasado
    - Habituación
    - Exploración
    """
    ims = [h['IM'] for h in historial_im if h['comida'] == comida]
    
    if len(ims) == 0:
        return 0.5 + exploracion_prob
    
    pesos = np.exp(-0.3 * np.arange(len(ims))[::-1])
    valor_metabolico = np.average(ims, weights=pesos)
    
    hab = habituacion.get(comida, 0.0)
    
    utilidad = valor_metabolico - hab * 0.3
    
    utilidad += np.random.normal(0, exploracion_prob * 0.05)
    
    return utilidad


def elegir_proxima_comida_simple(opciones, historial_im, habituacion, exploracion_prob):
    utilidades = {}
    for comida in opciones:
        utilidad = calcular_utilidad_simple(comida, historial_im, habituacion, exploracion_prob)
        utilidades[comida] = utilidad
    
    mejor_comida = max(utilidades, key=utilidades.get)
    return mejor_comida, utilidades


# ============================================================
# SIMULACIÓN PRINCIPAL CON COSTO ESTRUCTURAL
# ============================================================
def simular_vida_con_costo_estructural(opciones, n_experiencias=50):
    print("=" * 100)
    print("VSTCosmos - v48: Costo Estructural Real")
    print("NO hay miedo. NO hay penalizaciones externas.")
    print("Ciertas entradas degradan K (capacidad metabólica).")
    print("La evitación emerge cuando el sistema ya no puede sostener la interacción.")
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
    historial_im = []
    historial_costos = []
    
    G, R, C = 1.0, 1.0, 1.0
    
    print("\n[Reposo inicial] Estabilizando sistema...")
    S_actual, G, R, C = simular_reposo((Phi, A, Psi, Omega, M, L), G, R, C, K)
    Phi, A, Psi, Omega, M, L = S_actual
    
    exploracion_prob = EXPLORACION_INICIAL
    
    for exp in range(n_experiencias):
        print(f"\n{'#'*100}")
        print(f"EXPERIENCIA {exp + 1}")
        print(f"K = {K:.4f} | Exploración = {exploracion_prob:.3f}")
        print(f"{'#'*100}")
        
        comida_elegida, utilidades = elegir_proxima_comida_simple(
            opciones, historial_im, habituacion, exploracion_prob
        )
        
        print(f"\n  🍽️  Elige: {comida_elegida}")
        
        # Medir basal
        S_basal_antes = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
        estado_before = extraer_estado(S_basal_antes)
        asimilacion_basal, S_despues_basal = medir_asimilacion_basal(S_basal_antes, G, R, C, K)
        
        # Comer
        S_despues_comer, G, R, C, metricas = simular_experiencia(
            S_despues_basal, comida_elegida, G, R, C, habituacion[comida_elegida], K
        )
        
        # Metabolizar (reposo)
        S_post, G, R, C = simular_reposo(S_despues_comer, G, R, C, K)
        
        estado_after = extraer_estado(S_post)
        
        # Calcular IM
        asimilacion_post, _, _, _ = calcular_asimilacion(
            metricas['delta_rango_A'],
            metricas['delta_rango_Phi'],
            metricas['delta_L'],
            metricas['variacion_A'],
            K
        )
        
        IM = asimilacion_post - asimilacion_basal
        
        # NUEVO: Calcular costo estructural (pérdida real de capacidad)
        costo_struct = calcular_costo_estructural(estado_before, estado_after)
        
        if IM > UMBRAL_NUTRITIVO:
            clasificacion = "NUTRITIVA ✨"
        elif IM < UMBRAL_TOXICO:
            clasificacion = "TÓXICA 💀💀💀"
        else:
            clasificacion = "NEUTRA"
        
        # Actualizar K (con costo estructural real)
        K_antes = K
        K = actualizar_K(K, IM, costo_struct)
        
        # Actualizar memoria
        if estado_after['rango_A'] > UMBRAL_MEMORIA:
            _, A_actual, _, _, M_actual, _ = S_post
            M = M_actual + ETA_MEMORIA * (A_actual - M_actual)
        else:
            M = M * (1 - ETA_DECAY * max(0, -IM))
        M = np.clip(M, 0.0, 1.0)
        
        # Actualizar habituación
        habituacion = actualizar_habituacion(habituacion, comida_elegida)
        
        # Respirar
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        Phi, A, Psi, Omega, M, L = S_post
        
        historial_im.append({
            'experiencia': exp + 1,
            'comida': comida_elegida,
            'IM': IM,
            'clasificacion': clasificacion,
            'K': K,
            'costo_estructural': costo_struct
        })
        historial_costos.append(costo_struct)
        
        exploracion_prob = max(exploracion_prob * EXPLORACION_DECAY, EXPLORACION_MIN)
        
        print(f"\n  📈 RESULTADO:")
        print(f"    IM = {IM:+.4f} → {clasificacion}")
        print(f"    Costo estructural = {costo_struct:.4f}")
        print(f"    K: {K_antes:.4f} → {K:.4f}")
        print(f"    Estado: rango_A={estado_after['rango_A']:.3f}, media_L={estado_after['media_L']:.3f}")
    
    return historial_im, K, habituacion, historial_costos


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
    
    print("\n⚠️  COSTO ESTRUCTURAL REAL (sin miedo)")
    print("   - El sistema NO tiene penalizaciones externas")
    print("   - Algunas entradas degradan K (capacidad metabólica)")
    print("   - La evitación emerge por pérdida de capacidad")
    
    historial, K_final, habituacion_final, costos = simular_vida_con_costo_estructural(opciones, n_experiencias=50)
    
    print("\n" + "=" * 100)
    print("HISTORIAL DE EXPERIENCIAS")
    print("=" * 100)
    
    print(f"\n{'Exp':<4} | {'Comida':<22} | {'IM':>10} | {'Clasif':<14} | {'Costo':>8} | {'K':>8}")
    print("-" * 85)
    
    for h in historial[-25:]:
        print(f"{h['experiencia']:<4} | {h['comida'][:22]:<22} | {h['IM']:10.4f} | {h['clasificacion']:<14} | {h['costo_estructural']:8.4f} | {h['K']:8.4f}")
    
    print("\n" + "=" * 100)
    print("ESTADÍSTICAS FINALES")
    print("=" * 100)
    
    print(f"\n📈 K final: {K_final:.4f}")
    print(f"📉 Costo estructural promedio: {np.mean(costos):.4f}")
    
    print(f"\n🍽️  FRECUENCIA DE ELECCIONES:")
    conteo = {}
    for h in historial:
        conteo[h['comida']] = conteo.get(h['comida'], 0) + 1
    for comida, count in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comida:<25}: {count} veces")
    
    # Análisis de K por comida
    print(f"\n📊 EVOLUCIÓN DE K (últimos valores):")
    for comida in opciones:
        ks = [h['K'] for h in historial if h['comida'] == comida]
        if ks:
            print(f"  {comida:<25}: inicio={ks[0]:.4f} → fin={ks[-1]:.4f} (Δ={ks[-1]-ks[0]:+.4f})")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("El sistema NO tiene miedo. Solo consecuencias estructurales.")
    print("=" * 100)


if __name__ == "__main__":
    main()