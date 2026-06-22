#!/usr/bin/env python3
"""
VSTCosmo - Caracterización: Desintegración de entradas
Observando qué tipos de ruptura de integración reabren el sistema.
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
DURACION_SIM = 30.0
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
# GENERACIÓN DE SEÑALES DESINTEGRADAS
# ============================================================
def desintegrar_bigbang(bigbang, sr, retardo_ms=200):
    """Desplazar temporalmente la señal para romper integración"""
    retardo_muestras = int(retardo_ms * sr / 1000)
    bigbang_desinc = np.roll(bigbang, retardo_muestras)
    # Atenuar el inicio para evitar transiente
    bigbang_desinc[:retardo_muestras] = 0
    return bigbang_desinc

def ruido_banda(sr, duracion, f_min=0, f_max=200):
    """Ruido limitado en banda (como viento)"""
    from scipy.signal import butter, lfilter
    n_muestras = int(sr * duracion)
    ruido = np.random.normal(0, 1, n_muestras)
    # Filtro pasa bajos
    b, a = butter(4, f_max / (sr/2), btype='low')
    ruido_filt = lfilter(b, a, ruido)
    if f_min > 0:
        b, a = butter(4, f_min / (sr/2), btype='high')
        ruido_filt = lfilter(b, a, ruido_filt)
    ruido_filt = ruido_filt / np.max(np.abs(ruido_filt)) * 0.3
    return ruido_filt

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Caracterización: Desintegración de entradas")
    print("Observando qué tipo de ruptura de integración reabre el sistema")
    print("=" * 60)
    
    # Cargar archivos
    print("\n[1] Cargando archivos...")
    sr_ref, voz_viento = cargar_audio('Voz+Viento_1.wav')
    sr_b, bigbang = cargar_audio('BigBang.wav')
    sr_v, voz = cargar_audio('Voz_Estudio.wav')
    sr_w, viento = cargar_audio('Viento.wav')
    sr_m, bach = cargar_audio('Brandemburgo.wav')
    
    print(f"    Voz+Viento_real: {len(voz_viento)/sr_ref:.1f}s")
    print(f"    BigBang: {len(bigbang)/sr_b:.1f}s")
    print(f"    Voz_Estudio: {len(voz)/sr_v:.1f}s")
    print(f"    Viento: {len(viento)/sr_w:.1f}s")
    print(f"    Brandemburgo: {len(bach)/sr_m:.1f}s")
    
    # Generar señales desintegradas
    print("\n[2] Generando variantes desintegradas...")
    bigbang_desinc = desintegrar_bigbang(bigbang, sr_b, retardo_ms=300)
    ruido_viento = ruido_banda(sr_w, DURACION_SIM, f_min=0, f_max=200)
    voz_ruido = (voz[:len(ruido_viento)] + ruido_viento) / 2
    voz_ruido = voz_ruido / np.max(np.abs(voz_ruido))
    
    bach_viento = (bach[:len(viento)] + viento) / 2
    bach_viento = bach_viento / np.max(np.abs(bach_viento))
    
    print("    BigBang con retardo 300ms")
    print("    Ruido de banda (viento sintético)")
    print("    Voz + ruido de banda")
    print("    Bach + viento real")
    
    # ============================================================
    # OBSERVACIONES
    # ============================================================
    print("\n[3] Observando regímenes...")
    print("-" * 60)
    
    resultados = []
    
    # Referencia
    print("\n  === REFERENCIA ===")
    r = simular(voz_viento, sr_ref, "Voz+Viento_real")
    resultados.append(("Voz+Viento_real", r[0], r[1]))
    
    # BigBang original (cerrado)
    print("\n  === BIGBANG ===")
    r = simular(bigbang, sr_b, "BigBang_original")
    resultados.append(("BigBang_original", r[0], r[1]))
    
    # BigBang desintegrado
    r = simular(bigbang_desinc, sr_b, "BigBang_desincronizado")
    resultados.append(("BigBang_desincronizado", r[0], r[1]))
    
    # Voz sintética + ruido
    print("\n  === VOZ + RUIDO ===")
    r = simular(voz_ruido, sr_v, "Voz + ruido banda")
    resultados.append(("Voz+ruido_banda", r[0], r[1]))
    
    # Bach + viento
    print("\n  === BACH + PERTURBACIÓN ===")
    r = simular(bach_viento, sr_m, "Bach + viento")
    resultados.append(("Bach+viento", r[0], r[1]))
    
    # Ruido de banda solo
    r = simular(ruido_viento, sr_w, "Ruido de banda")
    resultados.append(("Ruido_banda", r[0], r[1]))
    
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
    # RESUMEN
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE OBSERVACIONES")
    print("=" * 60)
    
    abiertos = [(nombre, rphi) for nombre, rphi, _ in resultados if rphi < 0.85]
    
    if abiertos:
        print(f"\n★ Entradas que mantienen el campo ABIERTO:")
        for nombre, rphi in abiertos:
            print(f"    - {nombre}: rango Φ = {rphi:.3f}")
        
        if len(abiertos) > 1:
            print("\n  → Hay ALGO que estas entradas tienen en común")
            print("    (no sabemos qué todavía)")
    else:
        print("\n✗ Ninguna de las variantes desintegradas produjo apertura")
        print("  → La integración no se rompe fácilmente")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()