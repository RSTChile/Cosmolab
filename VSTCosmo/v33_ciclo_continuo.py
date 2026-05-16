#!/usr/bin/env python3
"""
VSTCosmo - v33: Ciclo continuo de experiencia
El sistema NO se reinicia entre entradas.
Respira entre experiencias con decaimiento diferencial.
Incluye todos los audios disponibles.
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

# PARÁMETROS DE CONTINUIDAD ENTRE EXPERIENCIAS
DECAY_M = 0.995      # M sobrevive mucho
DECAY_L_ENTRE = 0.970  # L decae si se rigidiza
DECAY_PSI = 0.900    # Ψ decae rápido
PESO_M_PARA_A = 0.3
A_BASAL = 0.1

# ============================================================
# FUNCIONES
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
        # Archivos reales
        sr, data = wav.read(ruta)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        # Asegurar duración
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


def respirar_entre_experiencias(M, Psi, L, A):
    """Fase de reposo entre experiencias."""
    M = M * DECAY_M
    Psi = Psi * DECAY_PSI
    L = L * DECAY_L_ENTRE
    
    A_nueva = A_BASAL * (1 - PESO_M_PARA_A) + PESO_M_PARA_A * M
    
    return M, Psi, L, A_nueva


def simular_secuencia(orden):
    """Procesa una secuencia de entradas sin reiniciar el sistema."""
    print("=" * 100)
    print("VSTCosmo - v33: Ciclo continuo de experiencia")
    print("Incluye: Voz, Brandemburgo, BigBang, Viento, Mezclas, Tono, Ruido, Silencio")
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
    
    registro_historico = []
    
    for exp_idx, entrada_nombre in enumerate(orden):
        print(f"\n{'='*80}")
        print(f"Experiencia {exp_idx + 1}: {entrada_nombre}")
        print(f"{'='*80}")
        
        sr, audio = cargar_audio(entrada_nombre)
        
        Phi_prev = Phi.copy()
        Phi_prev2 = Phi.copy()
        
        # Métricas de progreso
        ultimo_reporto = 0
        
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
            
            # Reporte cada 10%
            progreso = int(100 * paso / N_PASOS)
            if progreso >= ultimo_reporto + 10:
                ultimo_reporto = progreso
                print(f"  Progreso: {progreso}% | rango_A={np.max(A)-np.min(A):.3f} | media_L={np.mean(L):.3f}")
            
            Phi_prev2 = Phi_prev.copy()
            Phi_prev = Phi.copy()
        
        registro_historico.append({
            'experiencia': exp_idx + 1,
            'entrada': entrada_nombre,
            'rango_A_final': np.max(A)-np.min(A),
            'media_A_final': np.mean(A),
            'media_M_final': np.mean(M),
            'media_L_final': np.mean(L),
            'rango_L_final': np.max(L)-np.min(L)
        })
        
        print(f"\n  Final experiencia {exp_idx + 1}:")
        print(f"    rango_A = {registro_historico[-1]['rango_A_final']:.3f}")
        print(f"    media_L = {registro_historico[-1]['media_L_final']:.3f}")
        print(f"    media_M = {registro_historico[-1]['media_M_final']:.3f}")
        
        print(f"\n  [Reposo] Decaimiento diferencial...")
        M, Psi, L, A = respirar_entre_experiencias(M, Psi, L, A)
        print(f"    media_M después reposo = {np.mean(M):.4f}")
        print(f"    media_L después reposo = {np.mean(L):.4f}")
    
    return registro_historico


def main():
    # Secuencia completa con los nombres reales de los archivos
    secuencia = [
        "Voz_Estudio.wav",
        "Brandemburgo.wav",
        "BigBang.wav",
        "Tono puro",
        "Voz_Estudio.wav",
        "Ruido blanco",
        "Brandemburgo.wav",
        "Silencio",
        "BigBang.wav",
        "Viento.wav",
        "Voz+Viento_1.wav",
        "Voz+Viento_2.wav",
        "Voz_Estudio.wav"
    ]
    
    print("\n" + "=" * 100)
    print("SECUENCIA DE EXPERIENCIAS")
    print("=" * 100)
    for i, exp in enumerate(secuencia, 1):
        print(f"  {i}. {exp}")
    print("=" * 100)
    
    resultados = simular_secuencia(secuencia)
    
    print("\n" + "=" * 100)
    print("RESUMEN DE SECUENCIA COMPLETA")
    print("=" * 100)
    
    print(f"\n{'Exp':<4} | {'Entrada':<25} | {'rango_A':>10} | {'media_L':>10} | {'media_M':>10}")
    print("-" * 70)
    
    for res in resultados:
        # Acortar nombres para mejor visualización
        nombre = res['entrada']
        if len(nombre) > 25:
            nombre = nombre[:22] + "..."
        print(f"{res['experiencia']:<4} | {nombre:<25} | {res['rango_A_final']:10.3f} | "
              f"{res['media_L_final']:10.3f} | {res['media_M_final']:10.3f}")
    
    # Análisis por tipo de entrada
    print("\n" + "=" * 100)
    print("ANÁLISIS POR TIPO DE ENTRADA")
    print("=" * 100)
    
    tipos = {}
    for res in resultados:
        tipo = res['entrada']
        if tipo not in tipos:
            tipos[tipo] = {'L': [], 'A_rango': [], 'M': []}
        tipos[tipo]['L'].append(res['media_L_final'])
        tipos[tipo]['A_rango'].append(res['rango_A_final'])
        tipos[tipo]['M'].append(res['media_M_final'])
    
    print(f"\n{'Entrada':<30} | {'media_L (prom)':>12} | {'evolución L':>15} | {'rango_A prom':>12}")
    print("-" * 75)
    
    for tipo, datos in tipos.items():
        media_L = np.mean(datos['L'])
        if len(datos['L']) > 1:
            evol = f"{datos['L'][0]:.3f} → {datos['L'][-1]:.3f}"
        else:
            evol = "única"
        # Acortar nombre
        nombre_corto = tipo[:30]
        print(f"{nombre_corto:<30} | {media_L:12.4f} | {evol:>15} | {np.mean(datos['A_rango']):12.3f}")
    
    # Pregunta clave: ¿la voz aumenta su anclaje con la repetición?
    print("\n" + "=" * 100)
    print("PREGUNTA CLAVE: ¿LA VOZ AUMENTA SU ANCLAJE CON LA REPETICIÓN?")
    print("=" * 100)
    
    voz_exps = [res for res in resultados if res['entrada'] == "Voz_Estudio.wav"]
    if len(voz_exps) >= 2:
        print(f"\n  Primera exposición a voz: media_L = {voz_exps[0]['media_L_final']:.4f}")
        print(f"  Última exposición a voz: media_L = {voz_exps[-1]['media_L_final']:.4f}")
        if voz_exps[-1]['media_L_final'] > voz_exps[0]['media_L_final']:
            print("\n  ★★ LA VOZ AUMENTÓ SU ANCLAJE CON LA EXPERIENCIA ★★")
            print("  El sistema está aprendiendo a privilegiar la voz.")
        else:
            print("\n  ✗ La voz no aumentó su anclaje.")
            print("  El silencio o el ruido siguen dominando.")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()