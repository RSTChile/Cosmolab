#!/usr/bin/env python3
"""
VSTCosmo - v22: Doble memoria (Ψ y Ω)
Ψ: memoria de estabilidad (poco cambio)
Ω: memoria de coherencia en el cambio (persistencia de trayectoria)
A se acopla a ambas.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (base de v21.2)
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

# Parámetros de Ψ
TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_HISTORIA = 0.1
BLOQUEO_MAXIMO = 0.8

# Nuevos parámetros para Ω
TASA_OMEGA = 0.15       # velocidad de acumulación de coherencia
DISIPACION_OMEGA = 0.05 # olvido
FUERZA_OMEGA = 0.2      # cuánto pesa Ω en A

ENTRADAS = ["Tono puro", "Ruido blanco", "Voz_Estudio", "Silencio"]

# ============================================================
# FUNCIONES
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

def actualizar_memoria_estabilidad(Psi, Phi, A):
    cambio_local = np.abs(Phi - vecinos(Phi))
    cambio_norm = cambio_local / (cambio_local + 0.1)
    dPsi = (TASA_CRECIMIENTO * A * (1 - cambio_norm) * (1 - Psi) -
            TASA_DISIPACION * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)

def actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A):
    """
    Ω acumula donde el cambio es coherente (misma dirección).
    """
    cambio_actual = Phi - Phi_prev
    cambio_anterior = Phi_prev - Phi_prev2
    
    # Coherencia: 1 si el signo del cambio se mantiene, 0 si cambia
    # Usamos signo suave para gradiente
    signo_actual = np.tanh(cambio_actual * 10)  # suave pero nítido
    signo_anterior = np.tanh(cambio_anterior * 10)
    coherencia = (1 + signo_actual * signo_anterior) / 2  # 0 a 1
    
    # Ω acumula donde hay coherencia Y hay atención
    dOmega = (TASA_OMEGA * A * coherencia * (1 - Omega) -
              DISIPACION_OMEGA * Omega) * DT
    Omega = Omega + dOmega
    return np.clip(Omega, 0.0, 1.0)

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

def actualizar_atencion(A, Phi, Psi, Omega):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_estabilidad = FUERZA_HISTORIA * (Psi - A)
    acoplamiento_coherencia = FUERZA_OMEGA * (Omega - A)
    
    dA = auto + inhib_local + difusion + acoplamiento_relieve + acoplamiento_estabilidad + acoplamiento_coherencia
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
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
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        A = actualizar_atencion(A, Phi, Psi, Omega)
        Phi = actualizar_campo(Phi, A, muestra, Psi)
        Psi = actualizar_memoria_estabilidad(Psi, Phi, A)
        Omega = actualizar_memoria_coherencia(Omega, Phi, Phi_prev, Phi_prev2, A)
        
        Phi_prev2 = Phi_prev.copy()
        Phi_prev = Phi.copy()
    
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
    print("VSTCosmo - v22: Doble memoria (Ψ y Ω)")
    print("Ψ: estabilidad | Ω: coherencia en el cambio")
    print("=" * 100)
    
    sr = 48000
    duracion = DURACION_SIM
    
    # Generar señales
    print("\n[1] Generando señales...")
    tono = generar_tono_puro(sr, duracion, 440)
    ruido = generar_ruido_blanco(sr, duracion)
    _, voz = cargar_audio('Voz_Estudio.wav')
    silencio = np.zeros(int(sr * duracion))
    
    # Recortar
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
    print("-" * 100)
    print(f"{'Entrada':<20} | {'rango Φ':>10} | {'rango A':>10} | {'media A':>10} | {'media Ψ':>10} | {'media Ω':>10}")
    print("-" * 100)
    
    resultados = []
    for nombre, audio in entradas:
        Phi, A, Psi, Omega = simular(audio, sr, nombre)
        rphi, ra, ma, mpsi, momega = metrics(Phi, A, Psi, Omega)
        resultados.append((nombre, rphi, ra, ma, mpsi, momega))
        print(f"{nombre:<20} | {rphi:10.4f} | {ra:10.4f} | {ma:10.4f} | {mpsi:10.4f} | {momega:10.4f}")
    
    print("-" * 100)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS")
    print("=" * 100)
    
    # Orden de memoria de coherencia (Ω)
    orden_omega = sorted(resultados, key=lambda x: x[5], reverse=True)
    print("\nOrden de memoria de coherencia Ω (mayor = cambio más coherente):")
    for i, (nombre, _, _, _, _, momega) in enumerate(orden_omega):
        print(f"  {i+1}. {nombre}: {momega:.4f}")
    
    # Voz vs Ruido vs Tono
    voz_res = [r for r in resultados if r[0] == "Voz_Estudio"][0]
    ruido_res = [r for r in resultados if r[0] == "Ruido blanco"][0]
    tono_res = [r for r in resultados if r[0] == "Tono puro"][0]
    
    print(f"\nComparación:")
    print(f"  Voz:        Ω={voz_res[5]:.4f}, Ψ={voz_res[4]:.4f}, rΦ={voz_res[1]:.4f}")
    print(f"  Ruido:      Ω={ruido_res[5]:.4f}, Ψ={ruido_res[4]:.4f}, rΦ={ruido_res[1]:.4f}")
    print(f"  Tono puro:  Ω={tono_res[5]:.4f}, Ψ={tono_res[4]:.4f}, rΦ={tono_res[1]:.4f}")
    
    if voz_res[5] > ruido_res[5] and voz_res[5] > tono_res[5]:
        print("\n  ✓ La voz tiene la mayor coherencia Ω")
        print("    (lo que era esperable: cambio regular pero no periódico perfecto)")
    else:
        print("\n  ✗ La voz no tiene la mayor coherencia")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    main()