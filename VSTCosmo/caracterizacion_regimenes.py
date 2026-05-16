#!/usr/bin/env python3
"""
VSTCosmo - Caracterización de regímenes
Múltiples runs para observar cuándo emerge el régimen fértil.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (fijos, los que dieron régimen fértil en una ocasión)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 30.0
N_PASOS = int(DURACION_SIM / DT)

DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION = 0.05
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

MOD_DECAY = 1.0
MOD_GENERACION = 1.5

# Parámetros de Ψ (estabilidad)
TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_ESTABILIDAD = 0.1
BLOQUEO_MAXIMO = 0.8

# Parámetros de Ω (coherencia)
TASA_OMEGA = 0.15
DISIPACION_OMEGA = 0.05
FUERZA_COHERENCIA = 0.2

# Parámetros de Σ (estructura en la variación)
TASA_SIGMA = 0.15
DISIPACION_SIGMA = 0.05
FUERZA_ESTRUCTURA = 0.3

ENTRADAS = ["Tono puro", "Ruido blanco", "Voz_Estudio", "Silencio"]
SEMILLAS = list(range(10))  # 10 ejecuciones con diferentes semillas

# ============================================================
# FUNCIONES (idénticas a la versión exitosa)
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

def generar_tono_puro(sr, duracion, freq=440):
    t = np.arange(int(sr * duracion)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)

def generar_ruido_blanco(sr, duracion):
    n_muestras = int(sr * duracion)
    return np.random.normal(0, 0.3, n_muestras)

def inicializar_campo(semilla):
    np.random.seed(semilla)
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

def actualizar_memoria_estructura(Sigma, Phi, Phi_prev, Phi_prev2, A, Sigma_prev):
    dd = Phi - 2*Phi_prev + Phi_prev2
    signo_dd = np.tanh(dd * 5.0)
    signo_dd_prev = np.tanh(Sigma_prev * 5.0)
    consistencia = (1.0 + signo_dd * signo_dd_prev) / 2.0
    estructura = consistencia
    
    dSigma = (TASA_SIGMA * A * estructura * (1 - Sigma) -
              DISIPACION_SIGMA * Sigma) * DT
    Sigma = Sigma + dSigma
    return np.clip(Sigma, 0.0, 1.0)

def actualizar_campo(Phi, A, muestra, Psi):
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    perfil_2d = perfil.reshape(-1, 1)
    
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    desviacion = Phi - promedio_local
    generacion_base = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    mod_entrada = (1 + MOD_GENERACION * perfil_2d)
    mod_memoria = (1 - GANANCIA_HISTORIA * Psi)
    generacion = generacion_base * mod_entrada * mod_memoria
    
    mod_entrada_decay = 1 - MOD_DECAY * perfil_2d
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada_decay
    
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi_propuesto = difusion + generacion + decaimiento + sostenimiento
    dPhi_real = dPhi_propuesto * (1 - BLOQUEO_MAXIMO * Psi)
    
    Phi = Phi + DT * dPhi_real
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi, Psi, Omega, Sigma):
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
    acoplamiento_estructura = FUERZA_ESTRUCTURA * (Sigma - A)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          acoplamiento_estructura)
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def simular(audio, sr, nombre, semilla, num_pasos=N_PASOS):
    Phi = inicializar_campo(semilla)
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    Sigma = inicializar_memoria()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    Sigma_prev = Sigma.copy()
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        A = actualizar_atencion(A, Phi, Psi, Omega, Sigma)
        Phi = actualizar_campo(Phi, A, muestra, Psi)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        Sigma = actualizar_memoria_estructura(Sigma, Phi, Phi_prev, Phi_prev2, A, Sigma_prev)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
        Sigma_prev = Sigma.copy()
    
    # Caracterización del régimen
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    var_phi = np.var(Phi)
    media_psi = np.mean(Psi)
    media_sigma = np.mean(Sigma)
    
    return rango_phi, rango_a, var_phi, media_psi, media_sigma

def caracterizar_regimen(rango_phi, rango_a, var_phi, media_psi, media_sigma):
    """Clasificación cualitativa del régimen basada en métricas observables."""
    if rango_phi < 0.05:
        return "PLANO"
    elif rango_phi > 0.9:
        return "SATURADO"
    elif media_psi > 0.1 and rango_a > 0.2 and media_sigma > 0.05:
        return "FÉRTIL"
    elif media_psi < 0.06 and rango_a < 0.25:
        return "COLAPSADO"
    else:
        return "PERIÓDICO"

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - Caracterización de regímenes")
    print("Múltiples runs para observar cuándo emerge el régimen fértil")
    print("=" * 100)
    
    sr = 48000
    duracion = DURACION_SIM
    
    # Generar señales
    print("\n[1] Generando señales...")
    tono = generar_tono_puro(sr, duracion, 440)
    ruido = generar_ruido_blanco(sr, duracion)
    _, voz = cargar_audio('Voz_Estudio.wav')
    silencio = np.zeros(int(sr * duracion))
    
    n_muestras = int(duracion * sr)
    tono = tono[:n_muestras]
    ruido = ruido[:n_muestras]
    voz = voz[:n_muestras] if len(voz) > n_muestras else voz
    voz = voz / np.max(np.abs(voz))
    
    entradas = [
        ("Voz_Estudio", voz),
        ("Tono puro", tono),
        ("Ruido blanco", ruido),
        ("Silencio", silencio)
    ]
    
    print("\n[2] Ejecutando múltiples runs...")
    print(f"Semillas: {SEMILLAS}")
    print("-" * 120)
    
    resultados_por_semilla = defaultdict(lambda: defaultdict(dict))
    
    for semilla in SEMILLAS:
        print(f"\n--- Semilla {semilla} ---")
        for nombre, audio in entradas:
            rphi, ra, var_phi, mpsi, msigma = simular(audio, sr, nombre, semilla)
            regimen = caracterizar_regimen(rphi, ra, var_phi, mpsi, msigma)
            resultados_por_semilla[semilla][nombre] = {
                'rango_phi': rphi,
                'rango_a': ra,
                'var_phi': var_phi,
                'media_psi': mpsi,
                'media_sigma': msigma,
                'regimen': regimen
            }
            vozna = "Voz_Estudio" if nombre == "Voz_Estudio" else nombre
            print(f"  {vozna:<15}: rΦ={rphi:.4f}, rA={ra:.4f}, Ψ={mpsi:.4f}, Σ={msigma:.4f} → {regimen}")
    
    # ============================================================
    # ANÁLISIS POR REGIMEN
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS: REGÍMENES POR SEMILLA")
    print("=" * 100)
    
    for semilla in SEMILLAS:
        print(f"\nSemilla {semilla}:")
        for nombre in ["Voz_Estudio", "Tono puro", "Ruido blanco", "Silencio"]:
            r = resultados_por_semilla[semilla][nombre]
            print(f"  {nombre:15}: {r['regimen']} (Σ={r['media_sigma']:.4f})")
    
    # ============================================================
    # REGISTRO DE REGÍMENES FÉRTILES
    # ============================================================
    print("\n" + "=" * 100)
    print("REGISTRO DE REGÍMENES FÉRTILES")
    print("(donde la voz tiene Σ diferenciada)")
    print("=" * 100)
    
    fertil_count = 0
    for semilla in SEMILLAS:
        voz_res = resultados_por_semilla[semilla]["Voz_Estudio"]
        tono_res = resultados_por_semilla[semilla]["Tono puro"]
        ruido_res = resultados_por_semilla[semilla]["Ruido blanco"]
        
        if (voz_res['regimen'] == "FÉRTIL" and 
            voz_res['media_sigma'] > tono_res['media_sigma'] and 
            voz_res['media_sigma'] > ruido_res['media_sigma']):
            fertil_count += 1
            print(f"\nSemilla {semilla} → RÉGIMEN FÉRTIL")
            print(f"  Voz:  Σ={voz_res['media_sigma']:.4f}, rΦ={voz_res['rango_phi']:.4f}")
            print(f"  Tono: Σ={tono_res['media_sigma']:.4f}, rΦ={tono_res['rango_phi']:.4f}")
            print(f"  Ruido:Σ={ruido_res['media_sigma']:.4f}, rΦ={ruido_res['rango_phi']:.4f}")
    
    print(f"\nTotal de regímenes fértiles: {fertil_count}/{len(SEMILLAS)}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    main()