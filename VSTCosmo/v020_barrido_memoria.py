#!/usr/bin/env python3
"""
VSTCosmo - v20: Memoria estructural Ψ
Ψ acumula dónde el campo ha sido sostenible.
Modula la dinámica de Φ y la atención A.
Sin métricas. Sin selectividad externa.
"""

import numpy as np
import scipy.io.wavfile as wav
import itertools
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS FIJOS (basados en v19.2 mejor combinación)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 15.0
N_PASOS = int(DURACION_SIM / DT)

DECAIMIENTO_PHI = 0.01
GANANCIA_GENERACION = 0.20
GANANCIA_SOSTENIMIENTO = 0.25
DIFUSION_BASE = 0.20  # de la mejor combinación

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0

# Modulación de entrada (fija, de la mejor combinación)
MOD_DECAY = 1.0
MOD_GENERACION = 0.0

# Parámetros de Ψ (a barrer)
TASA_CRECIMIENTO_VALS = [0.01, 0.03, 0.05, 0.07, 0.10]
TASA_DISIPACION_VALS = [0.01, 0.02, 0.03, 0.05]
GANANCIA_HISTORIA_VALS = [0.2, 0.5, 1.0, 1.5]
FUERZA_HISTORIA_VALS = [0.05, 0.1, 0.2, 0.3]

ENTRADAS = ["Silencio", "Viento_puro", "Voz_Estudio", "Voz+Viento_real"]

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

def actualizar_campo(Phi, A, Psi, muestra, params):
    perfil = perfil_modulacion(muestra)
    promedio_local = vecinos(Phi)
    perfil_2d = perfil.reshape(-1, 1)
    
    # Difusión (sin modulación)
    difusion = DIFUSION_BASE * (promedio_local - Phi)
    
    # Generación base
    desviacion = Phi - promedio_local
    generacion_base = GANANCIA_GENERACION * desviacion * (1 - desviacion**2)
    # Modulada por entrada y por historia (Ψ)
    generacion = generacion_base * (1 + MOD_GENERACION * perfil_2d) * (1 + params['ganancia_historia'] * Psi)
    
    # Decaimiento modulado por entrada y por historia (Ψ reduce decaimiento)
    mod_entrada = 1 - MOD_DECAY * perfil_2d
    mod_historia = 1 - 0.5 * Psi
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local) * mod_entrada * mod_historia
    
    # Sostenimiento por A
    sostenimiento = GANANCIA_SOSTENIMIENTO * A * (Phi - promedio_local)
    
    dPhi = difusion + generacion + decaimiento + sostenimiento
    Phi = Phi + DT * dPhi
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi, Psi, params):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_relieve = FUERZA_RELIEVE * (relieve_local - A)
    acoplamiento_historia = params['fuerza_historia'] * (Psi - A)
    
    dA = auto + inhib_local + difusion + acoplamiento_relieve + acoplamiento_historia
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def actualizar_memoria(Psi, Phi, A, params):
    # Ψ crece donde Φ y A co-sostienen estructura
    # Usamos la diferencia Φ - vecinos(Φ) como medida local de estructura
    estructura_local = np.abs(Phi - vecinos(Phi))
    dPsi = (params['tasa_crecimiento'] * A * estructura_local * (1 - Psi) - 
            params['tasa_disipacion'] * Psi) * DT
    Psi = Psi + dPsi
    return np.clip(Psi, 0.0, 1.0)

def simular(entrada, sr, params):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Psi = inicializar_memoria()
    
    audio = entrada['audio'] if entrada['tipo'] != 'Silencio' else np.zeros(int(sr * DURACION_SIM))
    
    for paso in range(N_PASOS):
        t = paso * DT
        if entrada['tipo'] != 'Silencio':
            idx = int(t * sr)
            idx = min(idx, len(audio) - 1)
            muestra = audio[idx] if idx >= 0 else 0.0
        else:
            muestra = 0.0
        
        A = actualizar_atencion(A, Phi, Psi, params)
        Phi = actualizar_campo(Phi, A, Psi, muestra, params)
        Psi = actualizar_memoria(Psi, Phi, A, params)
    
    return np.max(Phi) - np.min(Phi)

def cargar_entradas():
    entradas = []
    sr_ref = 48000
    for nombre in ENTRADAS:
        if nombre == "Silencio":
            entradas.append({"tipo": nombre, "audio": None, "sr": sr_ref})
        elif nombre == "Viento_puro":
            sr, audio = cargar_audio('Viento.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
        elif nombre == "Voz_Estudio":
            sr, audio = cargar_audio('Voz_Estudio.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
        elif nombre == "Voz+Viento_real":
            sr, audio = cargar_audio('Voz+Viento_1.wav')
            entradas.append({"tipo": nombre, "audio": audio, "sr": sr})
    return entradas

# ============================================================
# BARRIDO DE Ψ
# ============================================================
def barrido_memoria():
    print("=" * 100)
    print("VSTCosmo - v20: Memoria estructural Ψ (barrido)")
    print("Ψ acumula dónde el campo ha sido sostenible")
    print("Modula Φ y A")
    print("=" * 100)
    
    entradas = cargar_entradas()
    
    print("\nParámetros a barrer:")
    print(f"  Tasa crecimiento: {TASA_CRECIMIENTO_VALS}")
    print(f"  Tasa disipación: {TASA_DISIPACION_VALS}")
    print(f"  Ganancia historia: {GANANCIA_HISTORIA_VALS}")
    print(f"  Fuerza historia: {FUERZA_HISTORIA_VALS}")
    print(f"Total combinaciones: {len(TASA_CRECIMIENTO_VALS) * len(TASA_DISIPACION_VALS) * len(GANANCIA_HISTORIA_VALS) * len(FUERZA_HISTORIA_VALS)}")
    print("-" * 100)
    
    resultados = []
    
    for tc, td, gh, fh in itertools.product(TASA_CRECIMIENTO_VALS, TASA_DISIPACION_VALS,
                                              GANANCIA_HISTORIA_VALS, FUERZA_HISTORIA_VALS):
        params = {
            'tasa_crecimiento': tc,
            'tasa_disipacion': td,
            'ganancia_historia': gh,
            'fuerza_historia': fh
        }
        
        valores = []
        for e in entradas:
            rphi = simular(e, e['sr'], params)
            valores.append(rphi)
        
        silencio, viento, voz, voz_viento = valores
        rango = max(valores) - min(valores)
        
        if rango > 0.005:
            resultados.append({
                'tc': tc, 'td': td, 'gh': gh, 'fh': fh,
                'silencio': silencio, 'viento': viento, 'voz': voz, 'voz_viento': voz_viento,
                'rango': rango
            })
            print(f"✓ tc={tc:.3f}, td={td:.3f}, gh={gh:.2f}, fh={fh:.2f} | "
                  f"S={silencio:.4f}, Vi={viento:.4f}, Vo={voz:.4f}, Vx={voz_viento:.4f} | "
                  f"rango={rango:.4f}")
    
    print("\n" + "=" * 100)
    if resultados:
        print(f"Se encontraron {len(resultados)} combinaciones con diferenciación (rango > 0.005):")
        for r in resultados[:10]:
            print(f"  tc={r['tc']:.3f}, td={r['td']:.3f}, gh={r['gh']:.2f}, fh={r['fh']:.2f} | "
                  f"S={r['silencio']:.4f}, Vi={r['viento']:.4f}, Vo={r['voz']:.4f}, Vx={r['voz_viento']:.4f}")
        
        mejor = max(resultados, key=lambda x: x['rango'])
        print("\n" + "=" * 100)
        print("MEJOR COMBINACIÓN (mayor rango):")
        print(f"  TASA_CRECIMIENTO = {mejor['tc']:.3f}")
        print(f"  TASA_DISIPACION = {mejor['td']:.3f}")
        print(f"  GANANCIA_HISTORIA = {mejor['gh']:.2f}")
        print(f"  FUERZA_HISTORIA = {mejor['fh']:.2f}")
        print(f"\n  Rangos Φ:")
        print(f"    Silencio:     {mejor['silencio']:.4f}")
        print(f"    Viento:       {mejor['viento']:.4f}")
        print(f"    Voz_Estudio:  {mejor['voz']:.4f}")
        print(f"    Voz+Viento:   {mejor['voz_viento']:.4f}")
        print(f"    Rango total:  {mejor['rango']:.4f}")
    else:
        print("  No se encontraron combinaciones con diferenciación > 0.005.")
        print("  Ajustar criterios o rangos.")
    
    print("\n" + "=" * 100)
    print("BARRIDO COMPLETADO")
    print("=" * 100)

if __name__ == "__main__":
    barrido_memoria()