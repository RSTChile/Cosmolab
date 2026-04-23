#!/usr/bin/env python3
"""
VSTCosmo - Caracterización: Diferentes tipos de entrada
Observando qué propiedades de la entrada producen apertura del campo.
No determinamos causa. Solo observamos regímenes.
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
DURACION = 20.0
N_PASOS = int(DURACION / DT)

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
# FUNCIONES BASE (idénticas a v12)
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
# GENERACIÓN DE SEÑALES DE PRUEBA
# ============================================================
def ruido_blanco(sr, duracion):
    """Ruido blanco gaussiano (sin estructura)"""
    n_muestras = int(sr * duracion)
    return np.random.normal(0, 0.3, n_muestras)

def ruido_rosa(sr, duracion):
    """Ruido rosa (estructura de baja frecuencia)"""
    n_muestras = int(sr * duracion)
    # Generar ruido blanco y filtrar
    blanco = np.random.normal(0, 1, n_muestras)
    # Filtro simple para ruido rosa (pendiente 1/f)
    from scipy.signal import lfilter
    b = [0.5, 0.5]
    a = [1, -0.5]
    rosa = lfilter(b, a, blanco)
    rosa = rosa / np.max(np.abs(rosa)) * 0.3
    return rosa

def seno_puro(sr, duracion, freq=440):
    """Seno puro (máximo orden)"""
    t = np.arange(int(sr * duracion)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)

def seno_modulado(sr, duracion, freq=440, mod_freq=2):
    """Seno con modulación de amplitud (estructura temporal)"""
    t = np.arange(int(sr * duracion)) / sr
    portadora = np.sin(2 * np.pi * freq * t)
    moduladora = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    return 0.5 * portadora * moduladora

def caos_logistico(sr, duracion, r=3.9):
    """Mapa logístico (caos determinista con estructura)"""
    n_muestras = int(sr * duracion)
    x = np.zeros(n_muestras)
    x[0] = 0.5
    for i in range(1, n_muestras):
        x[i] = r * x[i-1] * (1 - x[i-1])
    # Normalizar a rango [-0.5, 0.5]
    x = (x - 0.5) * 0.5
    return x

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("VSTCosmo - Caracterización: Tipos de Entrada")
    print("Observando qué propiedades producen apertura del campo")
    print("=" * 60)
    
    sr = 48000
    duracion = DURACION
    
    # Cargar referencias
    print("\n[1] Cargando referencias...")
    _, voz_viento = cargar_audio('Voz+Viento_1.wav')
    _, viento = cargar_audio('Viento.wav')
    _, voz_limpia = cargar_audio('Voz_Estudio.wav')
    
    # Generar señales de prueba
    print("\n[2] Generando señales de prueba...")
    
    seno = seno_puro(sr, duracion)
    seno_mod = seno_modulado(sr, duracion)
    caos = caos_logistico(sr, duracion)
    ruido_b = ruido_blanco(sr, duracion)
    ruido_r = ruido_rosa(sr, duracion)
    
    print("    Señales generadas: seno puro, seno modulado, caos, ruido blanco, ruido rosa")
    
    # Crear mezclas con correlación
    # Mezcla con retardo: voz y viento desplazados temporalmente
    retardo = int(0.1 * sr)  # 100 ms
    voz_retardada = np.roll(voz_limpia[:len(viento)], retardo)
    mezcla_retardo = (viento[:len(voz_retardada)] + voz_retardada) / 2
    mezcla_retardo = mezcla_retardo / np.max(np.abs(mezcla_retardo))
    
    print("\n[3] Observando regímenes...")
    print("-" * 60)
    
    resultados = []
    
    # Referencias
    print("\n  === REFERENCIAS ===")
    r1 = simular(voz_viento[:int(sr*duracion)], sr, "Voz+Viento_real")
    r2 = simular(viento[:int(sr*duracion)], sr, "Viento_puro")
    r3 = simular(voz_limpia[:int(sr*duracion)], sr, "Voz_Estudio")
    r4 = simular(np.zeros(int(sr*duracion)), sr, "Silencio")
    
    resultados.append(("Voz+Viento_real", r1[0], r1[1]))
    resultados.append(("Viento_puro", r2[0], r2[1]))
    resultados.append(("Voz_Estudio", r3[0], r3[1]))
    resultados.append(("Silencio", r4[0], r4[1]))
    
    # Señales artificiales
    print("\n  === SEÑALES ARTIFICIALES ===")
    r5 = simular(seno, sr, "Seno puro (440Hz)")
    r6 = simular(seno_mod, sr, "Seno modulado AM")
    r7 = simular(caos, sr, "Caos logístico (r=3.9)")
    r8 = simular(ruido_b, sr, "Ruido blanco")
    r9 = simular(ruido_r, sr, "Ruido rosa")
    
    resultados.append(("Seno puro", r5[0], r5[1]))
    resultados.append(("Seno modulado", r6[0], r6[1]))
    resultados.append(("Caos logístico", r7[0], r7[1]))
    resultados.append(("Ruido blanco", r8[0], r8[1]))
    resultados.append(("Ruido rosa", r9[0], r9[1]))
    
    # Mezclas con correlación artificial
    print("\n  === MEZCLAS CON CORRELACIÓN ARTIFICIAL ===")
    r10 = simular(mezcla_retardo, sr, "Voz + Viento (100ms retardo)")
    resultados.append(("Voz+Viento_retardo", r10[0], r10[1]))
    
    # ============================================================
    # TABLA DE RESULTADOS
    # ============================================================
    print("\n" + "=" * 60)
    print("TABLA DE OBSERVACIONES")
    print("=" * 60)
    
    print("\n" + "-" * 65)
    print(f"{'Entrada':<30} | {'rango Φ':>10} | {'rango A':>10} | {'Régimen':>12}")
    print("-" * 65)
    
    for nombre, rphi, ra in resultados:
        if rphi < 0.85:
            regimen = "ABIERTO"
        else:
            regimen = "CERRADO"
        print(f"{nombre:<30} | {rphi:10.3f} | {ra:10.4f} | {regimen:>12}")
    
    print("-" * 65)
    
    # ============================================================
    # RESUMEN DE OBSERVACIONES
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE OBSERVACIONES")
    print("=" * 60)
    
    # Encontrar qué entradas producen apertura
    abiertos = [(nombre, rphi) for nombre, rphi, _ in resultados if rphi < 0.85]
    
    if abiertos:
        print(f"\n★ Entradas que mantienen el campo ABIERTO (rango Φ < 0.85):")
        for nombre, rphi in abiertos:
            print(f"    - {nombre}: rango Φ = {rphi:.3f}")
    else:
        print("\n✗ Ninguna entrada adicional produjo apertura.")
        print("  Solo Voz+Viento_real ha mostrado apertura hasta ahora.")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()