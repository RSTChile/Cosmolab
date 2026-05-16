#!/usr/bin/env python3
"""
VSTCosmo - v27.2: Refuerzo local por curvatura
La curvatura de la trayectoria de A modula su tasa de crecimiento local.
No hay reasignación global. La atención compite suavemente por límite.
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

# Competencia global suave
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5

# Curvatura
PESO_CURVATURA = 5.0   # refuerzo local por curvatura

MOD_DECAY = 1.0
MOD_GENERACION = 1.5

# Parámetros de Ψ
TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_ESTABILIDAD = 0.1
BLOQUEO_MAXIMO = 0.8

# Parámetros de Ω
TASA_OMEGA = 0.15
DISIPACION_OMEGA = 0.05
FUERZA_COHERENCIA = 0.2

LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

ENTRADAS = ["Tono puro", "Ruido blanco", "Voz_Estudio", "Silencio"]


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


def generar_tono_puro(sr, duracion, freq=440):
    t = np.arange(int(sr * duracion)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


def generar_ruido_blanco(sr, duracion):
    n_muestras = int(sr * duracion)
    return np.random.normal(0, 0.3, n_muestras)


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


# ============================================================
# MEMORIAS
# ============================================================
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


def curvatura_atencion(A, A_prev, A_prev2):
    """Estimación local de la curvatura (cambio de dirección) de la trayectoria de A."""
    flujo = A - A_prev
    flujo_prev = A_prev - A_prev2
    
    magnitud_flujo = np.sqrt(np.sum(flujo**2, axis=(0,1), keepdims=True)) + 1e-6
    magnitud_flujo_prev = np.sqrt(np.sum(flujo_prev**2, axis=(0,1), keepdims=True)) + 1e-6
    
    producto_punto = np.sum(flujo * flujo_prev, axis=(0,1), keepdims=True)
    cos_theta = producto_punto / (magnitud_flujo * magnitud_flujo_prev)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    curvatura = 1.0 - cos_theta
    curvatura = np.clip(curvatura, 0.0, 1.0)
    return curvatura


# ============================================================
# DINÁMICA DE Φ Y A
# ============================================================
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


def actualizar_atencion(A, Phi, Psi, Omega, A_prev, A_prev2):
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
    
    # Refuerzo local por curvatura
    curvatura = curvatura_atencion(A, A_prev, A_prev2)
    refuerzo_curvatura = PESO_CURVATURA * curvatura * A * (1 - A)
    
    dA = (auto + inhib_local + difusion +
          acoplamiento_relieve +
          acoplamiento_estabilidad +
          acoplamiento_coherencia +
          refuerzo_curvatura)
    
    # Competencia global suave (sin normalización forzada)
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    # Ruido mínimo
    dA += np.random.randn(*A.shape) * 0.001
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)


def simular(audio, sr, nombre, num_pasos=N_PASOS):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    Omega = inicializar_memoria()
    
    Phi_prev = Phi.copy()
    Phi_prev2 = Phi.copy()
    A_prev = A.copy()
    A_prev2 = A.copy()
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        A = actualizar_atencion(A, Phi, Psi, Omega, A_prev, A_prev2)
        Phi = actualizar_campo(Phi, A, muestra, Psi)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
        A_prev2 = A_prev.copy()
        A_prev = A.copy()
    
    return Phi, A, Psi, Omega


def metrics(Phi, A, Psi, Omega):
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    media_a = np.mean(A)
    media_psi = np.mean(Psi)
    media_omega = np.mean(Omega)
    return rango_phi, rango_a, media_a, media_psi, media_omega


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - v27.2: Refuerzo local por curvatura")
    print("La curvatura de la trayectoria de A refuerza su crecimiento local.")
    print("No hay reasignación global. Competencia suave por límite.")
    print("=" * 100)
    
    sr = 48000
    duracion = DURACION_SIM
    
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
        ("Tono puro", tono),
        ("Ruido blanco", ruido),
        ("Voz_Estudio", voz),
        ("Silencio", silencio)
    ]
    
    print("\n[2] Ejecutando simulaciones...")
    print("-" * 120)
    print(f"{'Entrada':<15} | {'rango Φ':>10} | {'rango A':>10} | {'media A':>10} | "
          f"{'media Ψ':>10} | {'media Ω':>10}")
    print("-" * 120)
    
    resultados = []
    for nombre, audio in entradas:
        Phi, A, Psi, Omega = simular(audio, sr, nombre)
        rphi, ra, ma, mpsi, momega = metrics(Phi, A, Psi, Omega)
        resultados.append((nombre, rphi, ra, ma, mpsi, momega))
        print(f"{nombre:<15} | {rphi:10.4f} | {ra:10.4f} | {ma:10.4f} | "
              f"{mpsi:10.4f} | {momega:10.4f}")
    
    print("-" * 120)
    
    print("\n" + "=" * 100)
    print("ANÁLISIS")
    print("=" * 100)
    
    voz_res = [r for r in resultados if r[0] == "Voz_Estudio"][0]
    ruido_res = [r for r in resultados if r[0] == "Ruido blanco"][0]
    tono_res = [r for r in resultados if r[0] == "Tono puro"][0]
    
    print(f"\nMedia de atención (A):")
    print(f"  Tono puro:    media A = {tono_res[3]:.4f}")
    print(f"  Ruido blanco: media A = {ruido_res[3]:.4f}")
    print(f"  Voz:          media A = {voz_res[3]:.4f}")
    print(f"  Silencio:     media A = {resultados[3][3]:.4f}")
    
    if voz_res[3] > ruido_res[3] and voz_res[3] > tono_res[3]:
        print("\n  ★★ EXITO: La voz tiene la mayor actividad de atención ★★")
        print("  La curvatura de su trayectoria refuerza la atención localmente.")
        print("  El sistema privilegia la complejidad de la trayectoria sobre la simplicidad.")
    else:
        print("\n  ✗ La voz no tiene la mayor actividad de atención")
        print("  Ajustar PESO_CURVATURA o INHIB_GLOBAL.")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)


if __name__ == "__main__":
    main()