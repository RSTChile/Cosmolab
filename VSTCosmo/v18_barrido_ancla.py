#!/usr/bin/env python3
"""
VSTCosmo - v18: Entrada como ancla
Barrido fino de TASA_MEZCLA y parámetros internos.
"""

import numpy as np
import scipy.io.wavfile as wav
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS BASE (se sobreescriben en barridos)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 15.0  # Reducido para barrido rápido
N_PASOS = int(DURACION_SIM / DT)

DIFUSION_PHI = 0.08
DECAIMIENTO_PHI = 0.04
REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
FUERZA_RELIEVE = 0.08
LIMITE_ATENCION = DIM_FREQ * DIM_TIME * 0.35
INHIB_GLOBAL = 0.5
LIMITE_MIN = 0.0
LIMITE_MAX = 1.0
GANANCIA_ENTRADA = 0.02  # mínimo residual

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

def vecinos(X):
    return (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
            np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1)) / 4.0

def construir_target(muestra):
    m = (muestra + 1.0) / 2.0
    m = np.clip(m, 0.0, 1.0)
    target_banda = 0.3 + 0.4 * m
    banda = int(m * (DIM_FREQ - 1))
    target = np.ones((DIM_FREQ, DIM_TIME), dtype=np.float32) * 0.5
    for i in range(DIM_FREQ):
        distancia = min(abs(i - banda), DIM_FREQ - abs(i - banda))
        influencia = np.exp(-(distancia ** 2) / 10.0)
        target[i, :] = target_banda * influencia + 0.5 * (1.0 - influencia)
    return target

def aplicar_dinamica_interna(Phi, A, params):
    """Difusión, decaimiento, generación, sostenimiento"""
    promedio_local = vecinos(Phi)
    
    difusion = DIFUSION_PHI * (promedio_local - Phi)
    decaimiento = -DECAIMIENTO_PHI * (Phi - promedio_local)
    sostenimiento = params['sostenimiento'] * A * (Phi - promedio_local)
    desviacion = Phi - promedio_local
    generacion = params['generacion'] * desviacion * (1 - desviacion**2)
    
    dPhi = difusion + decaimiento + sostenimiento + generacion
    return Phi + DT * dPhi

def actualizar_campo(Phi, A, muestra, params):
    # 1. La entrada define el estado base (constitutivo)
    estado_base = construir_target(muestra)
    
    # 2. Mezcla: la entrada es ancla, no corrección
    tasa_mezcla = params['tasa_mezcla']
    Phi = (1 - tasa_mezcla) * Phi + tasa_mezcla * estado_base
    
    # 3. Dinámica interna sobre un Φ que ya trae la huella de la entrada
    Phi = aplicar_dinamica_interna(Phi, A, params)
    
    # 4. Pequeña entrada directa (residual)
    entrada_directa = GANANCIA_ENTRADA * muestra
    Phi = Phi + entrada_directa
    
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def actualizar_atencion(A, Phi):
    vA = vecinos(A)
    auto = REFUERZO_A * A * (1.0 - A)
    inhib_local = -INHIBICION_A * vA
    difusion = DIFUSION_A * (vA - A)
    
    relieve_local = np.abs(Phi - vecinos(Phi))
    max_relieve = np.max(relieve_local)
    if max_relieve > 0:
        relieve_local = relieve_local / max_relieve
    
    acoplamiento_local = FUERZA_RELIEVE * (relieve_local - A)
    dA = auto + inhib_local + difusion + acoplamiento_local
    
    atencion_total = np.sum(A)
    if atencion_total > LIMITE_ATENCION:
        exceso = (atencion_total - LIMITE_ATENCION) / LIMITE_ATENCION
        dA += -INHIB_GLOBAL * exceso * A
    
    dA += np.random.randn(*A.shape) * 0.001
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def simular(entrada, sr, params):
    Phi = inicializar_campo()
    A = inicializar_atencion()
    
    audio = entrada['audio'] if entrada['tipo'] != 'Silencio' else np.zeros(int(sr * DURACION_SIM))
    
    for paso in range(N_PASOS):
        t = paso * DT
        if entrada['tipo'] != 'Silencio':
            idx = int(t * sr)
            idx = min(idx, len(audio) - 1)
            muestra = audio[idx] if idx >= 0 else 0.0
        else:
            muestra = 0.0
        
        A = actualizar_atencion(A, Phi)
        Phi = actualizar_campo(Phi, A, muestra, params)
    
    return np.max(Phi) - np.min(Phi)

# ============================================================
# Cargar entradas
# ============================================================
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
# BARRIDO 1: TASA_MEZCLA (entrada como ancla)
# ============================================================
def barrido_tasa_mezcla():
    print("\n" + "=" * 90)
    print("BARRIDO 1: TASA_MEZCLA (cuánto domina la entrada sobre Φ)")
    print("rango Φ para diferentes entradas")
    print("=" * 90)
    
    entradas = cargar_entradas()
    valores = np.arange(0.05, 1.01, 0.05).round(2)
    
    print(f"\n{'tasa_mezcla':>12}", end="")
    for e in entradas:
        print(f" | {e['tipo']:>20}", end="")
    print()
    print("-" * (12 + 23 * len(entradas)))
    
    for tm in valores:
        params = {
            'tasa_mezcla': tm,
            'generacion': 0.15,
            'sostenimiento': 0.25
        }
        print(f"{tm:12.2f}", end="")
        for e in entradas:
            rphi = simular(e, e['sr'], params)
            print(f" | {rphi:20.4f}", end="")
        print()

# ============================================================
# BARRIDO 2: GENERACION (dinámica interna)
# ============================================================
def barrido_generacion():
    print("\n" + "=" * 90)
    print("BARRIDO 2: GANANCIA_GENERACION (con TASA_MEZCLA óptima)")
    print("rango Φ para diferentes entradas")
    print("=" * 90)
    
    entradas = cargar_entradas()
    valores = np.arange(0.15, 0.01, -0.02).round(2)
    
    # Usar tasa_mezcla óptima del barrido anterior (donde empieza a diferenciar)
    TASA_MEZCLA_OPT = 0.50
    
    print(f"\n{'generacion':>12} (mezcla={TASA_MEZCLA_OPT:.2f})", end="")
    for e in entradas:
        print(f" | {e['tipo']:>20}", end="")
    print()
    print("-" * (12 + 23 * len(entradas)))
    
    for gen in valores:
        params = {
            'tasa_mezcla': TASA_MEZCLA_OPT,
            'generacion': gen,
            'sostenimiento': 0.25
        }
        print(f"{gen:12.2f}", end="")
        for e in entradas:
            rphi = simular(e, e['sr'], params)
            print(f" | {rphi:20.4f}", end="")
        print()

# ============================================================
# BARRIDO 3: SOSTENIMIENTO
# ============================================================
def barrido_sostenimiento():
    print("\n" + "=" * 90)
    print("BARRIDO 3: GANANCIA_SOSTENIMIENTO (con TASA_MEZCLA y GENERACION óptimos)")
    print("rango Φ para diferentes entradas")
    print("=" * 90)
    
    entradas = cargar_entradas()
    valores = np.arange(0.25, 0.04, -0.02).round(2)
    
    TASA_MEZCLA_OPT = 0.50
    GENERACION_OPT = 0.05
    
    print(f"\n{'sostenimiento':>14} (mezcla={TASA_MEZCLA_OPT:.2f}, gen={GENERACION_OPT:.2f})", end="")
    for e in entradas:
        print(f" | {e['tipo']:>20}", end="")
    print()
    print("-" * (14 + 23 * len(entradas)))
    
    for sost in valores:
        params = {
            'tasa_mezcla': TASA_MEZCLA_OPT,
            'generacion': GENERACION_OPT,
            'sostenimiento': sost
        }
        print(f"{sost:14.2f}", end="")
        for e in entradas:
            rphi = simular(e, e['sr'], params)
            print(f" | {rphi:20.4f}", end="")
        print()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 90)
    print("VSTCosmo - v18: Entrada como ancla (barrido fino)")
    print("Buscando el punto donde las entradas comienzan a diferenciarse")
    print("=" * 90)
    
    barrido_tasa_mezcla()
    barrido_generacion()
    barrido_sostenimiento()
    
    print("\n" + "=" * 90)
    print("BARRIDO COMPLETADO")
    print("=" * 90)