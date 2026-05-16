#!/usr/bin/env python3
"""
VSTCosmo - v35: Preferencia Alimentaria Emergente (CORREGIDO)
La prueba de digestión NO daña al sistema. Mide capacidad interna:
capacidad = rango_A * (1 + varianza_A)

El sistema puede aprender qué experiencias aumentan su capacidad futura.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
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
# PARÁMETROS DE PREFERENCIA ALIMENTARIA
# ============================================================
DECAY_M = 0.995
DECAY_L_ENTRE = 0.970
DECAY_PSI = 0.900
PESO_M_PARA_A = 0.3
A_BASAL = 0.1

ALPHA_PREFERENCIA = 0.15
BETA_OLVIDO = 0.05
EPSILON_EXPLORACION = 0.2

UMBRAL_NUTRITIVO = 0.01
UMBRAL_TOXICO = -0.01

# ============================================================
# FUNCIONES BASE
# ============================================================
def cargar_audio(ruta):
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


def extraer_estado(S):
    Phi, A, Psi, Omega, M, L = S
    return {
        'rango_A': np.max(A) - np.min(A),
        'media_A': np.mean(A),
        'var_A': np.var(A),
        'rango_Phi': np.max(Phi) - np.min(Phi),
        'media_L': np.mean(L),
        'media_M': np.mean(M)
    }


# ============================================================
# NUEVA PRUEBA DE DIGESTIÓN NO INVASIVA
# ============================================================
def prueba_digestion(S, G, R, C):
    """
    Mide capacidad interna SIN exponer el sistema a nada externo.
    Capacidad = rango_A * (1 + varianza_A)
    
    Un sistema vivo debe poder:
    - Generar diferencia (rango_A alto)
    - Cambiar espontáneamente (var_A alta)
    """
    Phi, A, Psi, Omega, M, L = S
    
    rango_A = np.max(A) - np.min(A)
    var_A = np.var(A)
    
    # Capacidad de generar diferencia y cambio
    capacidad = rango_A * (1 + var_A)
    
    return capacidad


def simular_experiencia(S, entrada_nombre, G, R, C):
    """Simula una experiencia completa (ingesta)"""
    Phi, A, Psi, Omega, M, L = S
    
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
    
    return (Phi, A, Psi, Omega, M, L), G, R, C


def simular_reposo(S, G, R, C):
    """Simula reposo sin entrada (metabolización)"""
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
    """Decaimiento diferencial entre experiencias"""
    M = M * DECAY_M
    Psi = Psi * DECAY_PSI
    L = L * DECAY_L_ENTRE
    A_nueva = A_BASAL * (1 - PESO_M_PARA_A) + PESO_M_PARA_A * M
    return M, Psi, L, A_nueva


def actualizar_preferencias(preferencias, comida, valor_alimentario):
    """Actualiza preferencia basado en valor alimentario"""
    if comida not in preferencias:
        preferencias[comida] = 0.0
    
    preferencias[comida] = (1 - ALPHA_PREFERENCIA) * preferencias[comida] + ALPHA_PREFERENCIA * valor_alimentario
    
    # Olvido competitivo
    for otra in list(preferencias.keys()):
        if otra != comida:
            preferencias[otra] = preferencias[otra] * (1 - BETA_OLVIDO)
    
    return preferencias


def elegir_comida(opciones, preferencias, exploracion=EPSILON_EXPLORACION):
    """Elige comida según preferencias + exploración"""
    if not preferencias:
        return np.random.choice(opciones)
    
    if np.random.random() < exploracion:
        # Explorar algo no preferido o desconocido
        no_preferidas = [c for c in opciones if preferencias.get(c, 0) < 0.1]
        if no_preferidas:
            return np.random.choice(no_preferidas)
    
    # Elegir según preferencia (probabilidad proporcional)
    probs = []
    for opcion in opciones:
        pref = preferencias.get(opcion, 0.0)
        probs.append(np.exp(pref * 3))  # Temperatura para diferenciar
    probs = np.array(probs) / np.sum(probs)
    
    return np.random.choice(opciones, p=probs)


# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================
def simular_preferencia_alimentaria(opciones, n_ciclos=12):
    """Ciclo completo de preferencia alimentaria"""
    print("=" * 100)
    print("VSTCosmo - v35: Preferencia Alimentaria Emergente")
    print("Prueba de digestión NO invasiva")
    print("Capacidad = rango_A * (1 + varianza_A)")
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
    
    preferencias = {}
    historial = []
    
    # Capacidad basal
    S_basal = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
    capacidad_basal = prueba_digestion(S_basal, G, R, C)
    print(f"\nCapacidad basal del sistema: {capacidad_basal:.4f}")
    print(f"  (rango_A basal: {np.max(A)-np.min(A):.3f}, var_A basal: {np.var(A):.4f})")
    
    for ciclo in range(n_ciclos):
        print(f"\n{'='*80}")
        print(f"CICLO {ciclo + 1}")
        print(f"{'='*80}")
        
        comida = elegir_comida(opciones, preferencias)
        print(f"\n  Elige: {comida}")
        if preferencias:
            print(f"  Preferencias actuales: {', '.join([f'{k[:15]}={v:.3f}' for k, v in sorted(preferencias.items(), key=lambda x: x[1], reverse=True)[:3]])}")
        
        # Capacidad antes
        S_before = (Phi.copy(), A.copy(), Psi.copy(), Omega.copy(), M.copy(), L.copy())
        capacidad_before = prueba_digestion(S_before, G, R, C)
        print(f"  Capacidad antes: {capacidad_before:.4f}")
        
        # Comer
        print(f"  [Ingesta]...")
        S_after, G, R, C = simular_experiencia(S_before, comida, G, R, C)
        
        # Metabolizar
        print(f"  [Metabolización]...")
        S_post, G, R, C = simular_reposo(S_after, G, R, C)
        
        # Capacidad después
        capacidad_after = prueba_digestion(S_post, G, R, C)
        
        # Valor alimentario
        valor_alimentario = capacidad_after - capacidad_before
        print(f"  VALOR ALIMENTARIO: {valor_alimentario:.4f}")
        
        if valor_alimentario > UMBRAL_NUTRITIVO:
            print(f"  → {comida} fue NUTRITIVA ✨")
        elif valor_alimentario < UMBRAL_TOXICO:
            print(f"  → {comida} fue TÓXICA 💀")
        else:
            print(f"  → {comida} fue NEUTRA")
        
        # Aprender
        preferencias = actualizar_preferencias(preferencias, comida, valor_alimentario)
        
        # Respirar
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        Phi, A, Psi, Omega, M, L = S_post
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        
        historial.append({
            'ciclo': ciclo + 1,
            'comida': comida,
            'valor_alimentario': valor_alimentario,
            'capacidad_before': capacidad_before,
            'capacidad_after': capacidad_after,
            'preferencias': preferencias.copy()
        })
        
        estado = extraer_estado((Phi, A, Psi, Omega, M, L))
        print(f"  Estado: rango_A={estado['rango_A']:.3f}, var_A={estado['var_A']:.4f}, media_L={estado['media_L']:.3f}")
    
    return historial, preferencias


# ============================================================
# MAIN
# ============================================================
def main():
    opciones = [
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "Voz+Viento_1.wav",
        "Ruido blanco",
        "Tono puro",
        "Silencio",
        "Viento.wav"
    ]
    
    print("\n" + "=" * 100)
    print("MENÚ DE COMIDAS")
    print("=" * 100)
    for i, op in enumerate(opciones, 1):
        print(f"  {i}. {op}")
    
    historial, preferencias = simular_preferencia_alimentaria(opciones, n_ciclos=12)
    
    print("\n" + "=" * 100)
    print("RESUMEN DE PREFERENCIAS")
    print("=" * 100)
    
    print(f"\n{'Ciclo':<6} | {'Comida':<22} | {'Valor Alimentario':>18}")
    print("-" * 55)
    
    for h in historial:
        print(f"{h['ciclo']:<6} | {h['comida'][:22]:<22} | {h['valor_alimentario']:18.4f}")
    
    print("\n" + "=" * 100)
    print("PREFERENCIAS FINALES")
    print("=" * 100)
    
    for comida, pref in sorted(preferencias.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comida:<30} → {pref:.4f}")
    
    if preferencias:
        mejor = max(preferencias, key=preferencias.get)
        print(f"\n✨ EL SISTEMA PREFIERE: {mejor}")
        print(f"   (valor: {preferencias[mejor]:.4f})")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()