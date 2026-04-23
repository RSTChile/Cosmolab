#!/usr/bin/env python3
"""
VSTCosmo - Caracterización: Entradas Culturales
Observando el régimen del sistema con música y poema musicalizado.
No recortamos duraciones.
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS (idénticos a v12)
# ============================================================
DIM_FREQ = 32
DIM_TIME = 100
DT = 0.01
DURACION_SIM = 30.0  # simulamos 30 segundos de cada archivo
N_PASOS = int(DURACION_SIM / DT)

GANANCIA_INEST = 0.15
DIFUSION_PHI = 0.1

REFUERZO_A = 0.15
INHIBICION_A = 0.2
DIFUSION_A = 0.08
DISIPACION_A = 0.01
BASAL_A = 0.05

DIFUSION_ACOPLAMIENTO = 0.2

LIMITE_MAX = 1.0
LIMITE_MIN = 0.0

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

def inicializar_campo():
    return np.random.rand(DIM_FREQ, DIM_TIME) * 0.2 + 0.4

def inicializar_atencion():
    return np.ones((DIM_FREQ, DIM_TIME)) * BASAL_A

def vecinos_phi(Phi):
    return (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
            np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1)) / 4

def actualizar_campo_permeable(Phi, muestra):
    m = (muestra + 1) / 2
    m = np.clip(m, 0, 1)
    target_banda = 0.3 + 0.4 * m
    m_banda = int(m * (DIM_FREQ - 1))
    target = np.ones_like(Phi) * 0.5
    
    for i in range(DIM_FREQ):
        distancia = min(abs(i - m_banda), DIM_FREQ - abs(i - m_banda))
        influencia = np.exp(-distancia**2 / 10)
        target[i] = target_banda * influencia + 0.5 * (1 - influencia)
    
    desviacion = Phi - target
    inestabilidad = GANANCIA_INEST * desviacion * (1 - desviacion**2)
    vecinos = vecinos_phi(Phi)
    difusion = DIFUSION_PHI * (vecinos - Phi)
    entrada_directa = 0.02 * muestra
    
    Phi = Phi + DT * (inestabilidad + difusion) + entrada_directa
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def vecinos_a(A):
    return (np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)) / 4

def actualizar_atencion(A, Phi, Phi_prev):
    auto = REFUERZO_A * A * (1 - A)
    inhib = -INHIBICION_A * vecinos_a(A)
    dif = DIFUSION_A * (vecinos_a(A) - A)
    dis = -DISIPACION_A * (A - BASAL_A)
    
    dA_base = auto + inhib + dif + dis
    
    grad_temporal = Phi - Phi_prev
    prop = 0.02 * np.roll(A, 1, axis=1) * np.maximum(grad_temporal, 0)
    prop += 0.01 * np.roll(A, -1, axis=1) * np.maximum(-grad_temporal, 0)
    
    dA = dA_base + prop
    dA += np.random.randn(*A.shape) * 0.001
    
    A = A + DT * dA
    return np.clip(A, LIMITE_MIN, LIMITE_MAX)

def acoplamiento_atencion_campo(Phi, A):
    vecinos = vecinos_phi(Phi)
    mezcla = (1 - 0.5 * A) * Phi + 0.5 * A * vecinos
    flujo = mezcla - Phi
    Phi = Phi + DT * DIFUSION_ACOPLAMIENTO * flujo
    return np.clip(Phi, LIMITE_MIN, LIMITE_MAX)

def simular(audio, sr, nombre, num_pasos=N_PASOS):
    print(f"    {nombre}...", end=" ", flush=True)
    
    # Usar solo los primeros num_pasos * DT segundos
    n_muestras = int(num_pasos * DT * sr)
    audio = audio[:n_muestras] if len(audio) > n_muestras else audio
    
    Phi = inicializar_campo()
    A = inicializar_atencion()
    Phi_prev = Phi.copy()
    
    for paso in range(num_pasos):
        t = paso * DT
        idx = int(t * sr)
        idx = min(idx, len(audio) - 1) if len(audio) > 0 else 0
        muestra = audio[idx] if idx >= 0 and len(audio) > 0 else 0.0
        
        Phi = actualizar_campo_permeable(Phi, muestra)
        A = actualizar_atencion(A, Phi, Phi_prev)
        Phi = acoplamiento_atencion_campo(Phi, A)
        Phi_prev = Phi.copy()
    
    rango_phi = np.max(Phi) - np.min(Phi)
    rango_a = np.max(A) - np.min(A)
    print(f"rango Φ={rango_phi:.3f}, rango A={rango_a:.4f}")
    return rango_phi, rango_a

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Caracterización: Entradas Culturales")
    print("Observando régimen con música (Brandemburgo) y poema musicalizado (BigBang)")
    print("=" * 60)
    
    # Cargar todas las entradas
    print("\n[1] Cargando archivos...")
    
    sr1, voz_viento = cargar_audio('Voz+Viento_1.wav')
    print(f"    Voz+Viento_1: {len(voz_viento)/sr1:.1f}s")
    
    sr2, viento = cargar_audio('Viento.wav')
    print(f"    Viento: {len(viento)/sr2:.1f}s")
    
    sr3, voz_estudio = cargar_audio('Voz_Estudio.wav')
    print(f"    Voz_Estudio: {len(voz_estudio)/sr3:.1f}s")
    
    sr4, brandemburgo = cargar_audio('Brandemburgo.wav')
    print(f"    Brandemburgo: {len(brandemburgo)/sr4:.1f}s (Concierto N°2, F mayor)")
    
    sr5, bigbang = cargar_audio('BigBang.wav')
    print(f"    BigBang: {len(bigbang)/sr5:.1f}s (Poema musicalizado)")
    
    print("\n[2] Observando regímenes...")
    print("-" * 60)
    
    resultados = []
    
    # Referencias
    print("\n  === REFERENCIAS ===")
    r1 = simular(voz_viento, sr1, "Voz+Viento_real")
    r2 = simular(viento, sr2, "Viento_puro")
    r3 = simular(voz_estudio, sr3, "Voz_Estudio")
    r4 = simular(np.zeros(int(30*sr1)), sr1, "Silencio")
    
    resultados.append(("Voz+Viento_real", r1[0], r1[1]))
    resultados.append(("Viento_puro", r2[0], r2[1]))
    resultados.append(("Voz_Estudio", r3[0], r3[1]))
    resultados.append(("Silencio", r4[0], r4[1]))
    
    # Entradas culturales
    print("\n  === ENTRADAS CULTURALES ===")
    r5 = simular(brandemburgo, sr4, "Brandemburgo (Bach)")
    r6 = simular(bigbang, sr5, "BigBang (poema musicalizado)")
    
    resultados.append(("Brandemburgo (Bach)", r5[0], r5[1]))
    resultados.append(("BigBang (poema)", r6[0], r6[1]))
    
    # ============================================================
    # TABLA DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 60)
    print("TABLA DE OBSERVACIONES")
    print("=" * 60)
    
    print("\n" + "-" * 65)
    print(f"{'Entrada':<35} | {'rango Φ':>10} | {'rango A':>10} | {'Régimen':>12}")
    print("-" * 65)
    
    for nombre, rphi, ra in resultados:
        if rphi < 0.85:
            regimen = "ABIERTO"
        elif rphi < 0.95:
            regimen = "BORDE"
        else:
            regimen = "CERRADO"
        print(f"{nombre:<35} | {rphi:10.3f} | {ra:10.4f} | {regimen:>12}")
    
    print("-" * 65)
    
    # ============================================================
    # RESUMEN DE OBSERVACIONES
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE OBSERVACIONES")
    print("=" * 60)
    
    abiertos = [(nombre, rphi) for nombre, rphi, _ in resultados if rphi < 0.85]
    bordes = [(nombre, rphi) for nombre, rphi, _ in resultados if 0.85 <= rphi < 0.95]
    
    if abiertos:
        print(f"\n★ Entradas que mantienen el campo ABIERTO (rango Φ < 0.85):")
        for nombre, rphi in abiertos:
            print(f"    - {nombre}: rango Φ = {rphi:.3f}")
    
    if bordes:
        print(f"\n○ Entradas en el BORDE (0.85 ≤ rango Φ < 0.95):")
        for nombre, rphi in bordes:
            print(f"    - {nombre}: rango Φ = {rphi:.3f}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()