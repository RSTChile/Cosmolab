#!/usr/bin/env python3
"""
VSTCosmo - Test de interpretación
Entradas con estructura estable pero sin semántica.
¿El sistema distingue voz de tono puro?
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (idénticos a v21.2)
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

TASA_CRECIMIENTO = 0.10
TASA_DISIPACION = 0.03
GANANCIA_HISTORIA = 0.5
FUERZA_HISTORIA = 0.1
BLOQUEO_MAXIMO = 0.8

ENTRADAS = ["Silencio", "Tono_puro", "Tono_modulado", "Ruido_blanco", "Ruido_rosa", "Voz_Estudio", "BigBang"]

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

def generar_tono_modulado(sr, duracion, freq=440, mod_freq=2):
    t = np.arange(int(sr * duracion)) / sr
    portadora = np.sin(2 * np.pi * freq * t)
    moduladora = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    return 0.5 * portadora * moduladora

def generar_ruido_blanco(sr, duracion):
    n_muestras = int(sr * duracion)
    return np.random.normal(0, 0.3, n_muestras)

def generar_ruido_rosa(sr, duracion):
    n_muestras = int(sr * duracion)
    blanco = np.random.normal(0, 1, n_muestras)
    from scipy.signal import lfilter
    b = [0.5, 0.5]
    a = [1, -0.5]
    rosa = lfilter(b, a, blanco)
    rosa = rosa / np.max(np.abs(rosa)) * 0.3
    return rosa

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

def actualizar_memoria(Psi, Phi, A):
    cambio_local = np.abs(Phi - vecinos(Phi))
    cambio_norm = cambio_local / (cambio_local + 0.1)
    dPsi = (TASA_CRECIMIENTO * A * (1 - cambio_norm) * (1 - Psi) -
            TASA_DISIPACION * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)

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

def actualizar_atencion(A, Phi, Psi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_historia = FUERZA_HISTORIA * (Psi - A)
    
    dA = auto + inhib_local + difusion + acoplamiento_relieve + acoplamiento_historia
    
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
    
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        A = actualizar_atencion(A, Phi, Psi)
        Phi = actualizar_campo(Phi, A, muestra, Psi)
        Psi = actualizar_memoria(Psi, Phi, A)
    
    return Phi, A, Psi

def metrics(Phi, A, Psi):
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    media_a = np.mean(A)
    media_psi = np.mean(Psi)
    return rango_phi, rango_a, media_a, media_psi

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 100)
    print("VSTCosmo - Test de interpretación")
    print("¿El sistema distingue voz de tono puro?")
    print("=" * 100)
    
    sr = 48000
    duracion = DURACION_SIM
    
    # Generar señales sintéticas
    print("\n[1] Generando señales...")
    tono_puro = generar_tono_puro(sr, duracion, 440)
    tono_mod = generar_tono_modulado(sr, duracion, 440, 2)
    ruido_b = generar_ruido_blanco(sr, duracion)
    ruido_r = generar_ruido_rosa(sr, duracion)
    
    # Cargar audio real
    _, voz = cargar_audio('Voz_Estudio.wav')
    _, bigbang = cargar_audio('BigBang.wav')
    
    # Recortar a la duración de simulación
    n_muestras = int(duracion * sr)
    tono_puro = tono_puro[:n_muestras]
    tono_mod = tono_mod[:n_muestras]
    ruido_b = ruido_b[:n_muestras]
    ruido_r = ruido_r[:n_muestras]
    voz = voz[:n_muestras] if len(voz) > n_muestras else voz
    bigbang = bigbang[:n_muestras] if len(bigbang) > n_muestras else bigbang
    silencio = np.zeros(n_muestras)
    
    entradas = [
        ("Silencio", silencio),
        ("Tono puro (440Hz)", tono_puro),
        ("Tono modulado AM", tono_mod),
        ("Ruido blanco", ruido_b),
        ("Ruido rosa", ruido_r),
        ("Voz_Estudio", voz),
        ("BigBang", bigbang)
    ]
    
    print("\n[2] Ejecutando simulaciones...")
    print("-" * 100)
    print(f"{'Entrada':<25} | {'rango Φ':>10} | {'rango A':>10} | {'media A':>10} | {'media Ψ':>10}")
    print("-" * 100)
    
    resultados = []
    for nombre, audio in entradas:
        Phi, A, Psi = simular(audio, sr, nombre)
        rphi, ra, ma, mpsi = metrics(Phi, A, Psi)
        resultados.append((nombre, rphi, ra, ma, mpsi))
        print(f"{nombre:<25} | {rphi:10.4f} | {ra:10.4f} | {ma:10.4f} | {mpsi:10.4f}")
    
    print("-" * 100)
    
    # ============================================================
    # ANÁLISIS
    # ============================================================
    print("\n" + "=" * 100)
    print("ANÁLISIS")
    print("=" * 100)
    
    # Buscar voz y tono puro
    voz_res = [r for r in resultados if r[0] == "Voz_Estudio"][0]
    tono_res = [r for r in resultados if r[0] == "Tono puro (440Hz)"][0]
    
    print(f"\nComparación Voz vs Tono puro:")
    print(f"  rango Φ: voz={voz_res[1]:.4f} | tono={tono_res[1]:.4f} | diferencia={abs(voz_res[1]-tono_res[1]):.4f}")
    print(f"  media Ψ: voz={voz_res[4]:.4f} | tono={tono_res[4]:.4f} | diferencia={abs(voz_res[4]-tono_res[4]):.4f}")
    
    if abs(voz_res[1] - tono_res[1]) > 0.02:
        print("\n  ✓ Diferencia significativa: el sistema distingue voz de tono puro")
    else:
        print("\n  ✗ Poca diferencia: el sistema no distingue claramente voz de tono puro")
    
    # Orden de rango Φ (menor = más estable)
    orden_phi = sorted(resultados, key=lambda x: x[1])
    print("\nOrden de estabilidad (menor rango Φ = más estable):")
    for i, (nombre, rphi, _, _, _) in enumerate(orden_phi):
        print(f"  {i+1}. {nombre}: {rphi:.4f}")
    
    # Orden de memoria Ψ (mayor = más huella)
    orden_psi = sorted(resultados, key=lambda x: x[4], reverse=True)
    print("\nOrden de memoria Ψ (mayor = más huella):")
    for i, (nombre, _, _, _, mpsi) in enumerate(orden_psi):
        print(f"  {i+1}. {nombre}: {mpsi:.4f}")
    
    print("\n" + "=" * 100)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    main()