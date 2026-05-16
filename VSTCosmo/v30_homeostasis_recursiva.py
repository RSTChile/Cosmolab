#!/usr/bin/env python3
"""
VSTCosmo - v30: Homeostasis recursiva (segunda capa)
El sistema ejecuta múltiples ciclos con diferentes entradas.
Ajusta sus rangos viables para favorecer regímenes de alta persistencia.
No sabe qué entrada es cuál. Solo aprende a mantener parámetros que dan estabilidad.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 60.0  # segundos por simulación
N_PASOS = int(DURACION_SIM / DT)

# Parámetros fijos (de v29)
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

# Rangos viables iniciales (primera capa)
A_RANGE_MIN = 0.25
A_RANGE_MAX = 0.55
PHI_RANGE_MIN = 0.08
PHI_RANGE_MAX = 0.35

# Segunda capa: aprendizaje de rangos (se ajustan después de cada simulación)
# Estos son los "objetivos" que el sistema aprende a mantener
APRENDIZAJE_TASA = 0.02  # velocidad de ajuste de rangos

HOMEOSTASIS_INTERVALO = 500
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

ENTRADAS = ["Voz_Estudio.wav", "Tono puro", "Ruido blanco", "Silencio"]
CICLOS = 5  # número de ciclos de simulación (cada ciclo = todas las entradas)


# ============================================================
# FUNCIONES (como en v29)
# ============================================================
def cargar_audio(ruta):
    if "Tono puro" in ruta:
        sr = 48000
        duracion = DURACION_SIM
        t = np.arange(int(sr * duracion)) / sr
        return sr, 0.5 * np.sin(2 * np.pi * 440 * t)
    elif "Ruido blanco" in ruta:
        sr = 48000
        duracion = DURACION_SIM
        n_muestras = int(sr * duracion)
        return sr, np.random.normal(0, 0.3, n_muestras)
    elif "Silencio" in ruta:
        sr = 48000
        duracion = DURACION_SIM
        return sr, np.zeros(int(sr * duracion))
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
    
    A_mean = np.mean(A)
    competencia = -k_comp * (A - A_mean)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          competencia)
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


def homeostasis(A, Phi, G, R, C, a_min, a_max, p_min, p_max):
    rango_A = np.max(A) - np.min(A)
    rango_Phi = np.max(Phi) - np.min(Phi)
    
    ajustes = []
    
    if rango_A < a_min:
        G *= 1.02
        C *= 1.02
        ajustes.append(f"bajo rango_A: G={G:.3f}, C={C:.3f}")
    elif rango_A > a_max:
        R *= 1.02
        G *= 0.98
        ajustes.append(f"alto rango_A: R={R:.3f}, G={G:.3f}")
    
    if rango_Phi < p_min:
        G *= 1.02
        ajustes.append(f"bajo rango_Phi: G={G:.3f}")
    elif rango_Phi > p_max:
        R *= 1.03
        G *= 0.97
        ajustes.append(f"alto rango_Phi: R={R:.3f}, G={G:.3f}")
    
    G = np.clip(G, 0.3, 3.0)
    R = np.clip(R, 0.3, 3.0)
    C = np.clip(C, 0.3, 3.0)
    
    return G, R, C, ajustes


def simular_una_entrada(audio, sr, nombre, ciclo, a_min, a_max, p_min, p_max, verbose=False):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    
    G, R, C = 1.0, 1.0, 1.0
    
    rango_A_vals = []
    rango_Phi_vals = []
    
    n_muestras = int(N_PASOS * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(N_PASOS):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1)
        muestra = audio[idx] if idx >= 0 else 0.0
        
        ganancia_gen = GANANCIA_GENERACION_BASE * G
        bloqueo_max = np.clip(BLOQUEO_MAXIMO_BASE * R, 0.3, 0.95)
        k_comp = K_COMP_BASE * C
        
        A = actualizar_atencion(A, Phi, Psi, Omega, k_comp)
        Phi = actualizar_campo(Phi, A, muestra, Psi, ganancia_gen, bloqueo_max)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        if paso > 0 and paso % HOMEOSTASIS_INTERVALO == 0:
            G, R, C, _ = homeostasis(A, Phi, G, R, C, a_min, a_max, p_min, p_max)
        
        if paso % (N_PASOS // 20) == 0:
            rango_A_vals.append(np.max(A) - np.min(A))
            rango_Phi_vals.append(np.max(Phi) - np.min(Phi))
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
    # Valores en la fase estable (último 20%)
    n_estable = len(rango_A_vals) // 5
    rango_A_estable = np.mean(rango_A_vals[-n_estable:]) if n_estable > 0 else rango_A_vals[-1]
    rango_Phi_estable = np.mean(rango_Phi_vals[-n_estable:]) if n_estable > 0 else rango_Phi_vals[-1]
    media_A_final = np.mean(A)
    
    return {
        'nombre': nombre,
        'ciclo': ciclo,
        'rango_A_estable': rango_A_estable,
        'rango_Phi_estable': rango_Phi_estable,
        'media_A_final': media_A_final,
        'G_final': G,
        'R_final': R,
        'C_final': C
    }


# ============================================================
# SEGUNDA CAPA DE HOMEOSTASIS (aprendizaje de rangos)
# ============================================================
def segunda_capa_homeostasis(rangos_A, rangos_Phi, persistencia_media):
    """
    Ajusta los rangos viables para favorecer regímenes de alta persistencia.
    persistencia_media: promedio de rango_A_estable en el último ciclo
    """
    a_min, a_max = rangos_A
    p_min, p_max = rangos_Phi
    
    # Si el sistema logró alta persistencia (rango_A > 0.4), ampliar ligeramente el rango
    if persistencia_media > 0.4:
        a_max = min(a_max * (1 + APRENDIZAJE_TASA), 0.65)
        a_min = max(a_min * (1 - APRENDIZAJE_TASA * 0.5), 0.15)
    # Si la persistencia es baja, reducir el rango para hacerlo más exigente
    elif persistencia_media < 0.25:
        a_max = max(a_max * (1 - APRENDIZAJE_TASA), 0.35)
        a_min = min(a_min * (1 + APRENDIZAJE_TASA), 0.35)
    
    # Ajuste de rango de Φ
    if persistencia_media > 0.4:
        p_max = min(p_max * (1 + APRENDIZAJE_TASA * 0.5), 0.45)
    elif persistencia_media < 0.25:
        p_max = max(p_max * (1 - APRENDIZAJE_TASA), 0.25)
    
    return [a_min, a_max], [p_min, p_max]


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v30: Homeostasis recursiva (segunda capa)")
    print("El sistema ejecuta múltiples ciclos con diferentes entradas")
    print("Aprende a ajustar sus rangos viables para favorecer regímenes de alta persistencia")
    print(f"Ciclos: {CICLOS}")
    print("=" * 100)
    
    # Inicializar rangos viables (objetivos homeostáticos)
    rangos_A = [A_RANGE_MIN, A_RANGE_MAX]
    rangos_Phi = [PHI_RANGE_MIN, PHI_RANGE_MAX]
    
    # Historial de resultados por ciclo y entrada
    historial = []
    
    for ciclo in range(CICLOS):
        print(f"\n{'=' * 80}")
        print(f"CICLO {ciclo + 1}/{CICLOS}")
        print(f"Rangos actuales: A=[{rangos_A[0]:.2f}, {rangos_A[1]:.2f}] | Φ=[{rangos_Phi[0]:.2f}, {rangos_Phi[1]:.2f}]")
        print("-" * 80)
        
        resultados_ciclo = []
        persistencia_acumulada = []
        
        for entrada_nombre in ENTRADAS:
            print(f"  Simulando: {entrada_nombre}...", end=" ", flush=True)
            sr, audio = cargar_audio(entrada_nombre)
            
            res = simular_una_entrada(
                audio, sr, entrada_nombre, ciclo,
                rangos_A[0], rangos_A[1], rangos_Phi[0], rangos_Phi[1],
                verbose=False
            )
            resultados_ciclo.append(res)
            persistencia_acumulada.append(res['rango_A_estable'])
            print(f"rango_A_estable={res['rango_A_estable']:.3f}, media_A={res['media_A_final']:.3f}")
        
        # Promedio de persistencia en este ciclo
        persistencia_promedio = np.mean(persistencia_acumulada)
        print(f"\n  Persistencia promedio del ciclo: {persistencia_promedio:.3f}")
        
        # Segunda capa: ajustar rangos para el próximo ciclo
        nuevos_rangos_A, nuevos_rangos_Phi = segunda_capa_homeostasis(
            rangos_A, rangos_Phi, persistencia_promedio
        )
        
        print(f"  Nuevos rangos: A=[{nuevos_rangos_A[0]:.2f}, {nuevos_rangos_A[1]:.2f}] | "
              f"Φ=[{nuevos_rangos_Phi[0]:.2f}, {nuevos_rangos_Phi[1]:.2f}]")
        
        rangos_A = nuevos_rangos_A
        rangos_Phi = nuevos_rangos_Phi
        
        historial.append(resultados_ciclo)
    
    # ============================================================
    # ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 100)
    print("RESUMEN POR ENTRADA (último ciclo)")
    print("=" * 100)
    
    ultimo_ciclo = historial[-1] if historial else []
    for res in ultimo_ciclo:
        print(f"\n{res['nombre']}:")
        print(f"  rango_A_estable: {res['rango_A_estable']:.3f}")
        print(f"  media_A_final: {res['media_A_final']:.3f}")
        print(f"  rango_Φ_estable: {res['rango_Phi_estable']:.3f}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    main()